//! HTTP API for opastin.
//!
//! | Endpoint | Purpose |
//! |---|---|
//! | `POST /v1/chat/completions` | OpenAI Chat Completions, routed and proxied |
//! | `POST /v1/messages` | Anthropic Messages, routed and proxied |
//! | `POST /v1/route` | Return the routing decision only; nothing is sent upstream |
//! | `GET /v1/models` | The router model and the configured models |
//! | `GET /healthz` | Liveness |

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use opastin::gateway::{Gateway, GatewayError};
use opastin::{ApiFormat, RouteError};
use serde::Deserialize;
use serde_json::{json, Value};

/// Response header: the configured model id that served the request.
pub const HEADER_MODEL: &str = "x-opastin-model";
/// Response header: decision id, present only on routed requests.
pub const HEADER_DECISION_ID: &str = "x-opastin-decision-id";
/// Response header: router name, present only on routed requests.
pub const HEADER_ROUTER: &str = "x-opastin-router";

/// Request bodies can carry long contexts and images.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

type AppState = Arc<Gateway>;

pub fn app(gateway: Arc<Gateway>) -> axum::Router {
    axum::Router::new()
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/route", post(route_only))
        .route("/v1/models", get(models))
        .route("/healthz", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(gateway)
}

async fn openai_chat(State(gw): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    proxy(&gw, ApiFormat::OpenaiChat, &headers, &body).await
}

async fn anthropic_messages(
    State(gw): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy(&gw, ApiFormat::AnthropicMessages, &headers, &body).await
}

async fn proxy(gw: &Gateway, format: ApiFormat, headers: &HeaderMap, body: &[u8]) -> Response {
    let json: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                format,
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("request body is not valid JSON: {e}"),
            )
        }
    };

    let forwarded = match gw.forward(format, json, headers).await {
        Ok(f) => f,
        Err(e) => return gateway_error(format, &e),
    };

    match &forwarded.decision {
        Some(d) => tracing::info!(
            %format, router = %d.router, model = %d.model, decision_id = %d.decision_id,
            status = forwarded.response.status().as_u16(), "routed"
        ),
        None => tracing::info!(
            %format, model = %forwarded.model,
            status = forwarded.response.status().as_u16(), "direct"
        ),
    }

    let mut response = Response::builder().status(forwarded.response.status());
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
    }

    // Stream the upstream body through unchanged, so SSE responses stay streamed.
    response
        .body(Body::from_stream(forwarded.response.bytes_stream()))
        .unwrap_or_else(|e| {
            error_response(format, StatusCode::BAD_GATEWAY, "api_error", &e.to_string())
        })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteBody {
    /// Format of `request`.
    api: ApiFormat,
    /// A request body in that format.
    request: Value,
}

/// Decide which model a request should go to, without forwarding it.
async fn route_only(State(gw): State<AppState>, body: Bytes) -> Response {
    let format = ApiFormat::OpenaiChat; // error shape for this endpoint
    let parsed: RouteBody = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                format,
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("expected {{\"api\": \"openai_chat\" | \"anthropic_messages\", \"request\": {{...}}}}: {e}"),
            )
        }
    };

    match gw.decide(parsed.api, &parsed.request).await {
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
            }))
            .into_response()
        }
        Err(e) => gateway_error(format, &e),
    }
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
            .map(|id| json!({"id": id, "object": "model", "created": 0, "owned_by": "opastin"}))
            .collect();
        Json(json!({"object": "list", "data": data})).into_response()
    }
}

fn gateway_error(format: ApiFormat, e: &GatewayError) -> Response {
    let (status, kind) = match e {
        GatewayError::BadRequest(_) | GatewayError::UnsupportedFormat { .. } => {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        }
        GatewayError::UnknownModel(_) => (StatusCode::NOT_FOUND, "not_found_error"),
        GatewayError::Route(RouteError::NoCandidates) => {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        }
        GatewayError::Route(_) | GatewayError::ApiKey(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "api_error")
        }
        GatewayError::Upstream(_) => (StatusCode::BAD_GATEWAY, "api_error"),
    };
    if status.is_server_error() {
        tracing::error!(%format, error = %e, "request failed");
    }
    error_response(format, status, kind, &e.to_string())
}

/// An error body in the shape the client's SDK expects.
fn error_response(format: ApiFormat, status: StatusCode, kind: &str, message: &str) -> Response {
    let body = match format {
        ApiFormat::OpenaiChat => json!({
            "error": {"message": message, "type": kind, "param": null, "code": null}
        }),
        ApiFormat::AnthropicMessages => json!({
            "type": "error",
            "error": {"type": kind, "message": message}
        }),
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
