//! HTTP API for opas.
//!
//! | Endpoint | Purpose |
//! |---|---|
//! | `POST /v1/chat/completions` | OpenAI Chat Completions, routed and proxied |
//! | `POST /v1/messages` | Anthropic Messages, routed and proxied |
//! | `POST /v1/responses` | Open Responses (OpenAI Responses API), routed and proxied |
//! | `POST /v1beta/interactions`, `POST /v1/interactions` | Gemini Interactions API, routed and proxied |
//! | `POST /v1/route` | Return the routing decision only; nothing is sent upstream |
//! | `GET /v1/models` | The router model and the configured models |
//! | `GET /v1/pricing` | Resolved price per configured model and its source (config, models.dev, tier default) |
//! | `GET /healthz` | Liveness |

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use futures_util::StreamExt;
use opas::gateway::{Gateway, GatewayError, RequestOptions};
use std::collections::HashMap;

use opas::{ApiFormat, Candidate, Preference, Pricing, RouteError, Weights};
use serde::Deserialize;
use serde_json::{json, Value};

/// Response header: the configured model id that served the request.
pub const HEADER_MODEL: &str = "x-opas-model";
/// Response header: decision id, present only on routed requests.
pub const HEADER_DECISION_ID: &str = "x-opas-decision-id";
/// Response header: router name, present only on routed requests.
pub const HEADER_ROUTER: &str = "x-opas-router";
/// Response header: the chosen model's prediction (`p_success`, `cost_usd`,
/// `response_time_s`), present when the router predicts.
pub const HEADER_ESTIMATE: &str = "x-opas-estimate";
/// Response header: model that served the previous turn of this conversation, present
/// when cache-aware routing recognized the request as a continuation.
pub const HEADER_PREVIOUS_MODEL: &str = "x-opas-previous-model";

/// Request bodies can carry long contexts and images.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// How much of a stateful API's response is scanned for its id.
const ID_SCAN_BYTES: usize = 64 * 1024;

type AppState = Arc<Gateway>;

pub fn app(gateway: Arc<Gateway>) -> axum::Router {
    axum::Router::new()
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(open_responses))
        .route("/v1/interactions", post(gemini_interactions))
        .route("/v1beta/interactions", post(gemini_interactions))
        .route("/v1/route", post(route_only))
        .route("/v1/models", get(models))
        .route("/v1/pricing", get(pricing))
        .route("/healthz", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(gateway)
}

async fn openai_chat(State(gw): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    proxy(gw, ApiFormat::OpenaiChat, &headers, &body).await
}

async fn anthropic_messages(
    State(gw): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy(gw, ApiFormat::AnthropicMessages, &headers, &body).await
}

async fn open_responses(State(gw): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    proxy(gw, ApiFormat::OpenResponses, &headers, &body).await
}

async fn gemini_interactions(
    State(gw): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy(gw, ApiFormat::GeminiInteractions, &headers, &body).await
}

async fn proxy(gw: Arc<Gateway>, format: ApiFormat, headers: &HeaderMap, body: &[u8]) -> Response {
    let json: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                format,
                ErrorKind::BadRequest,
                &format!("request body is not valid JSON: {e}"),
            )
        }
    };

    let forwarded = match gw.forward(format, json, headers).await {
        Ok(f) => f,
        Err(e) => return gateway_error(format, &e),
    };

    let status = forwarded.response.status();
    match &forwarded.decision {
        Some(d) => tracing::info!(
            %format, router = %d.router, model = %d.model, decision_id = %d.decision_id,
            status = status.as_u16(), "routed"
        ),
        None => {
            tracing::info!(%format, model = %forwarded.model, status = status.as_u16(), "direct")
        }
    }

    let mut response = Response::builder().status(status);
    let out = response.headers_mut().expect("fresh builder has headers");
    for (name, value) in forwarded.response.headers() {
        if !is_hop_by_hop(name) {
            out.append(name.clone(), value.clone());
        }
    }
    insert_header(out, HEADER_MODEL, &forwarded.model);
    if let Some(d) = &forwarded.decision {
        insert_header(out, HEADER_DECISION_ID, &d.decision_id);
        insert_header(out, HEADER_ROUTER, &d.router);
        if let Some(e) = d.estimates.iter().find(|e| e.model == d.model) {
            insert_header(
                out,
                HEADER_ESTIMATE,
                &format!(
                    "p_success={:.3}; cost_usd={:.6}; response_time_s={:.2}",
                    e.p_success, e.expected_cost_usd, e.expected_response_time_s
                ),
            );
        }
    }
    if let Some(c) = &forwarded.conversation {
        insert_header(out, HEADER_PREVIOUS_MODEL, &c.previous_model);
    }

    // Stream the upstream body through unchanged, so SSE responses stay streamed.
    let mut stream = forwarded.response.bytes_stream().boxed();
    if format.continuation_field().is_some() && status.is_success() {
        // Stateful APIs: learn the response id as it streams past, so a follow-up that
        // references it is pinned to the model that holds the state.
        let mut sniffer = IdSniffer::default();
        let model = forwarded.model.clone();
        stream = stream
            .inspect(move |chunk| {
                if let Ok(bytes) = chunk {
                    if let Some(id) = sniffer.feed(bytes) {
                        tracing::debug!(%id, %model, "recorded stored response id");
                        gw.record_stored_response(&id, &model);
                    }
                }
            })
            .boxed();
    }
    response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| error_response(format, ErrorKind::BadGateway, &e.to_string()))
}

/// Finds the first `"id": "..."` value in a response body (JSON or SSE). The response
/// object's own id comes first in both Open Responses (`response.created` event / the
/// response object) and Gemini Interactions (`interaction.created` / the interaction).
#[derive(Default)]
struct IdSniffer {
    buf: Vec<u8>,
    done: bool,
}

impl IdSniffer {
    fn feed(&mut self, chunk: &[u8]) -> Option<String> {
        if self.done {
            return None;
        }
        let room = ID_SCAN_BYTES.saturating_sub(self.buf.len());
        self.buf.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if let Some(id) = find_id(&self.buf) {
            self.done = true;
            return Some(id);
        }
        if self.buf.len() >= ID_SCAN_BYTES {
            self.done = true;
        }
        None
    }
}

fn find_id(buf: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(buf)
        .unwrap_or_else(|e| std::str::from_utf8(&buf[..e.valid_up_to()]).expect("valid prefix"));
    let mut rest = text;
    while let Some(pos) = rest.find("\"id\"") {
        let after = rest[pos + 4..].trim_start();
        if let Some(after) = after.strip_prefix(':') {
            let after = after.trim_start();
            if let Some(value) = after.strip_prefix('"') {
                // An unterminated value means the chunk ended mid-id; wait for more.
                return value
                    .find('"')
                    .map(|end| value[..end].to_owned())
                    .filter(|v| !v.is_empty());
            }
        }
        rest = &rest[pos + 4..];
    }
    None
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteBody {
    /// Format of `request`.
    api: ApiFormat,
    /// A request body in that format.
    request: Value,
    /// Models to choose between, e.g. `[{"id": "my-small", "tier": "small"}, ...]`.
    /// Defaults to the configured models that serve `api`.
    #[serde(default)]
    candidates: Option<Vec<Candidate>>,
    /// Quality / speed / cost weights, e.g. `{"quality": 60, "speed": 20, "cost": 20}`.
    #[serde(default)]
    weights: Option<Weights>,
    /// A named preset instead of `weights`: quality, balanced, cost or speed.
    #[serde(default)]
    preference: Option<Preference>,
    /// Prices by model id for this request (USD per million tokens), overriding the
    /// operator's and models.dev prices.
    #[serde(default)]
    pricing: HashMap<String, Pricing>,
}

/// Decide which model a request should go to, without forwarding it.
async fn route_only(State(gw): State<AppState>, body: Bytes) -> Response {
    let format = ApiFormat::OpenaiChat; // error shape for this endpoint
    let parsed: RouteBody = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let apis = ApiFormat::ALL.map(ApiFormat::as_str).join(" | ");
            return error_response(
                format,
                ErrorKind::BadRequest,
                &format!("expected {{\"api\": {apis}, \"request\": {{...}}, \"candidates\"?: [...]}}: {e}"),
            );
        }
    };

    if parsed.candidates.as_ref().is_some_and(Vec::is_empty) {
        return error_response(format, ErrorKind::BadRequest, "`candidates` is empty");
    }
    let options = RequestOptions {
        weights: parsed
            .weights
            .or(parsed.preference.map(Preference::weights)),
        pricing: parsed.pricing,
    };
    let decided = gw
        .decide_with(parsed.api, &parsed.request, parsed.candidates, options)
        .await;
    match decided {
        Ok(d) => {
            let upstream_model = gw
                .config()
                .model(&d.model)
                .map(|m| m.upstream_model().to_owned());
            tracing::info!(api = %parsed.api, router = %d.router, model = %d.model, decision_id = %d.decision_id, "decided");
            Json(json!({
                "decision_id": d.decision_id,
                "router": d.router,
                "model": d.model,
                "upstream_model": upstream_model,
                "reason": d.reason,
                "api": parsed.api,
                "conversation": d.conversation,
                "weights": d.weights,
                "estimates": d.estimates,
            }))
            .into_response()
        }
        Err(e) => gateway_error(format, &e),
    }
}

/// Resolved price per configured model and where it came from.
async fn pricing(State(gw): State<AppState>) -> Response {
    let catalog = gw.catalog();
    Json(json!({
        "models_dev": {"enabled": gw.config().pricing.models_dev, "priced_models": catalog.len()},
        "models": gw.price_table(),
    }))
    .into_response()
}

/// List the router model and configured models. Anthropic-shaped if the client sent
/// `anthropic-version`, OpenAI-shaped otherwise.
async fn models(State(gw): State<AppState>, headers: HeaderMap) -> Response {
    let config = gw.config();
    let ids: Vec<&str> = std::iter::once(config.router_model.as_str())
        .chain(config.models.iter().map(|m| m.id.as_str()))
        .collect();

    if headers.contains_key("anthropic-version") {
        let data: Vec<Value> = ids
            .iter()
            .map(|id| json!({"type": "model", "id": id, "display_name": id, "created_at": "1970-01-01T00:00:00Z"}))
            .collect();
        Json(json!({
            "data": data,
            "has_more": false,
            "first_id": ids.first(),
            "last_id": ids.last(),
        }))
        .into_response()
    } else {
        let data: Vec<Value> = ids
            .iter()
            .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "opas"}))
            .collect();
        Json(json!({"object": "list", "data": data})).into_response()
    }
}

/// Error categories, rendered in each API's own vocabulary.
#[derive(Debug, Clone, Copy)]
enum ErrorKind {
    BadRequest,
    NotFound,
    Internal,
    BadGateway,
}

impl ErrorKind {
    fn status(self) -> StatusCode {
        match self {
            ErrorKind::BadRequest => StatusCode::BAD_REQUEST,
            ErrorKind::NotFound => StatusCode::NOT_FOUND,
            ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            ErrorKind::BadGateway => StatusCode::BAD_GATEWAY,
        }
    }
}

fn gateway_error(format: ApiFormat, e: &GatewayError) -> Response {
    let kind = match e {
        GatewayError::BadRequest(_) | GatewayError::UnsupportedFormat { .. } => {
            ErrorKind::BadRequest
        }
        GatewayError::UnknownModel(_) => ErrorKind::NotFound,
        GatewayError::Route(RouteError::NoCandidates) => ErrorKind::BadRequest,
        GatewayError::Route(_) | GatewayError::ApiKey(_) => ErrorKind::Internal,
        GatewayError::Upstream(_) => ErrorKind::BadGateway,
    };
    if kind.status().is_server_error() {
        tracing::error!(%format, error = %e, "request failed");
    }
    error_response(format, kind, &e.to_string())
}

/// An error body in the shape the client's SDK expects.
fn error_response(format: ApiFormat, kind: ErrorKind, message: &str) -> Response {
    let status = kind.status();
    let body = match format {
        ApiFormat::OpenaiChat | ApiFormat::AnthropicMessages => {
            let kind = match kind {
                ErrorKind::BadRequest => "invalid_request_error",
                ErrorKind::NotFound => "not_found_error",
                ErrorKind::Internal | ErrorKind::BadGateway => "api_error",
            };
            if format == ApiFormat::OpenaiChat {
                json!({"error": {"message": message, "type": kind, "param": null, "code": null}})
            } else {
                json!({"type": "error", "error": {"type": kind, "message": message}})
            }
        }
        // Open Responses error types: invalid_request, not_found, server_error, ...
        ApiFormat::OpenResponses => {
            let kind = match kind {
                ErrorKind::BadRequest => "invalid_request",
                ErrorKind::NotFound => "not_found",
                ErrorKind::Internal | ErrorKind::BadGateway => "server_error",
            };
            json!({"error": {"message": message, "type": kind, "param": null, "code": null}})
        }
        // Google API error format (google.rpc.Status).
        ApiFormat::GeminiInteractions => {
            let code = match kind {
                ErrorKind::BadRequest => "INVALID_ARGUMENT",
                ErrorKind::NotFound => "NOT_FOUND",
                ErrorKind::Internal => "INTERNAL",
                ErrorKind::BadGateway => "UNAVAILABLE",
            };
            json!({"error": {"code": status.as_u16(), "message": message, "status": code}})
        }
    };
    (status, Json(body)).into_response()
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(name), v);
    }
}

/// Headers that describe the upstream connection, not the response, and must not be copied.
fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) || name == header::CONTENT_LENGTH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_first_id_in_json_and_sse() {
        assert_eq!(
            find_id(br#"{"created": 1, "id": "int_42", "steps": [{"id": "s1"}]}"#).as_deref(),
            Some("int_42")
        );
        let sse = b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_7\",\"object\":\"response\"}}\n\n";
        assert_eq!(find_id(sse).as_deref(), Some("resp_7"));
        assert_eq!(find_id(br#"{"identity": 1, "x_id": "no"}"#), None);
        assert_eq!(
            find_id(br#"{"id": "resp_unfinish"#),
            None,
            "value split across chunks"
        );
    }

    #[test]
    fn sniffer_handles_ids_split_across_chunks_and_stops() {
        let mut s = IdSniffer::default();
        assert_eq!(s.feed(br#"{"id": "resp_ab"#), None);
        assert_eq!(s.feed(br#"c", "output": []}"#).as_deref(), Some("resp_abc"));
        assert_eq!(s.feed(br#"{"id": "other"}"#), None, "only the first id");
    }
}
