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

pub mod openapi;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use futures_util::StreamExt;
use opas::gateway::{Gateway, GatewayError, RequestOptions};

use opas::{ApiFormat, Preference, RouteError};
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
/// Response headers with the chosen model's prediction, one value each.
pub const HEADER_P_SUCCESS: &str = "x-opas-p-success";
pub const HEADER_COST_USD: &str = "x-opas-cost-usd";
pub const HEADER_RESPONSE_TIME_S: &str = "x-opas-response-time-s";
pub const HEADER_OUTPUT_TOKENS: &str = "x-opas-output-tokens";
/// Response header: every candidate's prediction, best first:
/// `model; p_success=…; cost_usd=…; response_time_s=…, model; …`.
pub const HEADER_CANDIDATES: &str = "x-opas-candidates";
/// Response header: how long the routing decision took, in milliseconds.
pub const HEADER_ROUTING_MS: &str = "x-opas-routing-ms";

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
        .route("/v1/decisions/{id}", get(decision))
        .route("/", get(status))
        .route("/healthz", get(|| async { "ok" }))
        .route("/openapi.json", get(openapi_json))
        .route("/openapi.yaml", get(openapi_yaml))
        .route(
            "/docs",
            get(|| async { axum::response::Html(openapi::DOCS_HTML) }),
        )
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
    if !gw.config().proxy.enabled {
        return error_response(
            format,
            ErrorKind::NotFound,
            "this opas server only makes routing decisions ([proxy] enabled = false): \
             POST the request to /v1/route, then call the chosen model yourself",
        );
    }
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
            routing_ms = d.timing.map_or(-1.0, |t| (t.total_ms * 10.0).round() / 10.0),
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
            insert_header(out, HEADER_ESTIMATE, &estimate_fields(e));
            insert_header(out, HEADER_P_SUCCESS, &format!("{:.3}", e.p_success));
            insert_header(out, HEADER_COST_USD, &format!("{:.6}", e.expected_cost_usd));
            insert_header(
                out,
                HEADER_RESPONSE_TIME_S,
                &format!("{:.2}", e.expected_response_time_s),
            );
            insert_header(
                out,
                HEADER_OUTPUT_TOKENS,
                &format!("{:.0}", e.expected_output_tokens),
            );
        }
        if !d.estimates.is_empty() {
            let all: Vec<String> = d
                .estimates
                .iter()
                .map(|e| format!("{}; {}", e.model, estimate_fields(e)))
                .collect();
            insert_header(out, HEADER_CANDIDATES, &all.join(", "));
        }
        if let Some(t) = d.timing {
            insert_header(out, HEADER_ROUTING_MS, &format!("{:.1}", t.total_ms));
        }
        let upstream_model = gw
            .config()
            .model(&d.model)
            .map(|m| m.upstream_model().to_owned());
        gw.decisions()
            .insert(opas::DecisionRecord::new(d, format, upstream_model));
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
        let gw = Arc::clone(&gw);
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
    // Routed requests: measure what actually happened (tokens, cost, time) as the body
    // streams past, and attach it to the decision record when it ends.
    if let Some(d) = &forwarded.decision {
        let content_type = out
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let meter = Meter {
            gw: Arc::clone(&gw),
            decision_id: d.decision_id.clone(),
            model: forwarded.model.clone(),
            predicted: d.estimates.iter().find(|e| e.model == d.model).cloned(),
            pricing: gw.price_for(&forwarded.model, headers),
            status: status.as_u16(),
            sent_at: forwarded.sent_at,
            first_byte: None,
            scanner: Some(opas::UsageScanner::new(format, content_type.as_deref())),
        };
        stream = MeteredStream {
            inner: stream,
            meter: Some(meter),
        }
        .boxed();
    }
    response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| error_response(format, ErrorKind::BadGateway, &e.to_string()))
}

fn estimate_fields(e: &opas::Estimate) -> String {
    format!(
        "p_success={:.3}; cost_usd={:.6}; response_time_s={:.2}",
        e.p_success, e.expected_cost_usd, e.expected_response_time_s
    )
}

/// Passes a response body through while measuring it; records the outcome when the
/// stream ends or is dropped (client disconnect).
struct MeteredStream<S> {
    inner: S,
    meter: Option<Meter>,
}

struct Meter {
    gw: Arc<Gateway>,
    decision_id: String,
    model: String,
    predicted: Option<opas::Estimate>,
    pricing: Option<opas::Pricing>,
    status: u16,
    sent_at: std::time::Instant,
    first_byte: Option<std::time::Instant>,
    scanner: Option<opas::UsageScanner>,
}

impl Meter {
    fn feed(&mut self, bytes: &[u8]) {
        if self.first_byte.is_none() && !bytes.is_empty() {
            self.first_byte = Some(std::time::Instant::now());
        }
        if let Some(s) = &mut self.scanner {
            s.feed(bytes);
        }
    }

    fn finish(mut self, complete: bool) {
        let end = std::time::Instant::now();
        let usage = self.scanner.take().and_then(opas::UsageScanner::finish);
        let first = self.first_byte.unwrap_or(end);
        let ttfb = first.duration_since(self.sent_at).as_secs_f64();
        let total = end.duration_since(self.sent_at).as_secs_f64();
        let cost_usd = usage.zip(self.pricing).map(|(u, p)| u.cost_usd(&p));
        let streaming_s = end.duration_since(first).as_secs_f64();
        let output_tokens_per_sec = usage
            .filter(|u| u.output_tokens > 0 && streaming_s > 0.05)
            .map(|u| u.output_tokens as f64 / streaming_s);
        let round = |v: f64, d: i32| (v * 10f64.powi(d)).round() / 10f64.powi(d);
        tracing::info!(
            decision_id = %self.decision_id,
            model = %self.model,
            status = self.status,
            complete,
            output_tokens = usage.map_or(-1, |u| u.output_tokens as i64),
            cost_usd = cost_usd.map_or(-1.0, |c| round(c, 6)),
            predicted_cost_usd = self.predicted.as_ref().map_or(-1.0, |e| round(e.expected_cost_usd, 6)),
            total_time_s = round(total, 2),
            predicted_time_s = self.predicted.as_ref().map_or(-1.0, |e| round(e.expected_response_time_s, 2)),
            time_to_first_byte_s = round(ttfb, 2),
            "completed"
        );
        self.gw.decisions().complete(
            &self.decision_id,
            opas::Actual {
                status: self.status,
                complete,
                usage,
                cost_usd,
                time_to_first_byte_s: ttfb,
                total_time_s: total,
                output_tokens_per_sec,
            },
        );
    }
}

impl<S, E> futures_util::Stream for MeteredStream<S>
where
    S: futures_util::Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let polled = std::pin::Pin::new(&mut self.inner).poll_next(cx);
        match &polled {
            std::task::Poll::Ready(Some(Ok(bytes))) => {
                if let Some(m) = &mut self.meter {
                    m.feed(bytes);
                }
            }
            std::task::Poll::Ready(Some(Err(_))) => {
                if let Some(m) = self.meter.take() {
                    m.finish(false);
                }
            }
            std::task::Poll::Ready(None) => {
                if let Some(m) = self.meter.take() {
                    m.finish(true);
                }
            }
            std::task::Poll::Pending => {}
        }
        polled
    }
}

impl<S> Drop for MeteredStream<S> {
    fn drop(&mut self) {
        if let Some(m) = self.meter.take() {
            m.finish(false);
        }
    }
}

/// A recent routing decision with its predicted and measured outcome.
async fn decision(
    State(gw): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match gw.decisions().get(&id) {
        Some(r) => Json(r).into_response(),
        None => error_response(
            ApiFormat::OpenaiChat,
            ErrorKind::NotFound,
            &format!(
                "no decision `{id}` (unknown, or older than the most recent {})",
                opas::DecisionLog::DEFAULT_CAPACITY
            ),
        ),
    }
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

/// What this server is and how it is set up: an HTML page for browsers, JSON otherwise.
async fn status(State(gw): State<AppState>, headers: HeaderMap) -> Response {
    let config = gw.config();
    let prices = gw.price_table();
    let models: Vec<Value> = config
        .models
        .iter()
        .map(|m| {
            let price = prices.iter().find(|p| p.model == m.id);
            json!({
                "id": m.id,
                "tier": m.tier,
                "upstream_model": m.upstream_model(),
                "apis": ApiFormat::ALL
                    .iter()
                    .filter(|f| m.upstream(**f).is_some())
                    .map(|f| f.as_str())
                    .collect::<Vec<_>>(),
                "pricing": price.and_then(|p| p.pricing),
                "price_source": price.map(|p| p.source),
            })
        })
        .collect();
    let info = json!({
        "name": "opas",
        "version": env!("CARGO_PKG_VERSION"),
        "router": gw.router().map(|r| r.name()),
        "router_model": gw.router().map(|_| config.router_model.as_str()),
        "proxy": config.proxy.enabled,
        "cache_aware": config.cache.enabled,
        "decisions_recorded": gw.decisions().len(),
        "models": models,
        "links": {
            "docs": "/docs", "openapi": "/openapi.json", "health": "/healthz",
            "models": "/v1/models", "pricing": "/v1/pricing", "route": "/v1/route",
        },
    });
    let wants_html = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/html"));
    if !wants_html {
        return Json(info).into_response();
    }

    let esc = |s: &str| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    };
    let mut rows = String::new();
    for m in &config.models {
        let price = prices.iter().find(|p| p.model == m.id);
        let pricing = price.and_then(|p| p.pricing).map_or_else(
            || "—".to_owned(),
            |p| format!("${} / ${} / ${}", p.input, p.cached_input, p.output),
        );
        let apis: Vec<&str> = ApiFormat::ALL
            .iter()
            .filter(|f| m.upstream(**f).is_some())
            .map(|f| f.as_str())
            .collect();
        rows.push_str(&format!(
            "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>",
            esc(&m.id),
            m.tier.map_or("—", |t| t.as_str()),
            esc(m.upstream_model()),
            esc(&apis.join(", ")),
            esc(&pricing),
        ));
    }
    let router = match gw.router() {
        Some(r) => format!(
            "<b>{}</b> — send <code>\"model\": \"{}\"</code> to be routed",
            esc(r.name()),
            esc(&config.router_model)
        ),
        None => "none — a plain proxy: requests name a model".to_owned(),
    };
    let proxy = if config.proxy.enabled {
        "on: <code>/v1/chat/completions</code>, <code>/v1/messages</code>, <code>/v1/responses</code>, <code>/v1beta/interactions</code>"
    } else {
        "off: decisions only (<code>POST /v1/route</code>)"
    };
    let page = format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>opas</title>
<style>
:root {{ color-scheme: light dark; --fg: #1d1d1f; --muted: #6e6e73; --bg: #fff; --line: #e5e5ea; --accent: #5b5bd6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg: #f2f2f7; --muted: #a1a1aa; --bg: #111114; --line: #2a2a31; --accent: #8b8bff; }} }}
body {{ font: 15px/1.5 system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg); max-width: 900px; margin: 0 auto; padding: 32px 16px; }}
h1 {{ margin: 0; font-size: 28px; }} .sub {{ color: var(--muted); margin: 4px 0 24px; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0 0 24px; }} dt {{ color: var(--muted); }} dd {{ margin: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }} th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
th {{ color: var(--muted); font-weight: 600; }} code {{ font-size: 13px; }} a {{ color: var(--accent); }}
.links a {{ margin-right: 16px; }} .wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>opas <span style="font-weight:400;color:var(--muted)">{version}</span></h1>
<p class="sub">A model router for LLM requests. This server is running.</p>
<dl>
<dt>Router</dt><dd>{router}</dd>
<dt>Proxy</dt><dd>{proxy}</dd>
<dt>Cache-aware</dt><dd>{cache}</dd>
<dt>Decisions</dt><dd>{decisions} recorded (look one up at <code>/v1/decisions/&lt;id&gt;</code>)</dd>
</dl>
<div class="wrap"><table>
<tr><th>Model</th><th>Tier</th><th>Upstream name</th><th>APIs</th><th>Price per M tokens (in / cached / out)</th></tr>
{rows}
</table></div>
<p class="links"><a href="/docs">API reference</a><a href="/v1/pricing">Prices</a><a href="/v1/models">Models</a><a href="/openapi.json">OpenAPI</a><a href="/healthz">Health</a></p>
</body></html>
"#,
        version = env!("CARGO_PKG_VERSION"),
        cache = if config.cache.enabled { "yes" } else { "no" },
        decisions = gw.decisions().len(),
    );
    axum::response::Html(page).into_response()
}

/// The OpenAPI document, built once.
fn spec() -> &'static utoipa::openapi::OpenApi {
    static SPEC: std::sync::OnceLock<utoipa::openapi::OpenApi> = std::sync::OnceLock::new();
    SPEC.get_or_init(openapi::spec)
}

async fn openapi_json() -> Response {
    Json(spec()).into_response()
}

async fn openapi_yaml() -> Response {
    match spec().to_yaml() {
        Ok(y) => ([(header::CONTENT_TYPE, "application/yaml")], y).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Decide which model a request should go to, without forwarding it.
async fn route_only(State(gw): State<AppState>, body: Bytes) -> Response {
    let format = ApiFormat::OpenaiChat; // error shape for this endpoint
    let parsed: openapi::RouteBody = match serde_json::from_slice(&body) {
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
        conversation_id: parsed.conversation_id,
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
            tracing::info!(api = %parsed.api, router = %d.router, model = %d.model, decision_id = %d.decision_id,
                routing_ms = d.timing.map_or(-1.0, |t| (t.total_ms * 10.0).round() / 10.0), "decided");
            // Recorded for GET /v1/decisions/{id}; nothing is forwarded, so no outcome.
            gw.decisions().insert(opas::DecisionRecord::new(
                &d,
                parsed.api,
                upstream_model.clone(),
            ));
            Json(openapi::RouteResponse {
                decision_id: d.decision_id,
                router: d.router,
                model: d.model,
                upstream_model,
                reason: d.reason,
                api: parsed.api,
                conversation: d.conversation,
                weights: d.weights,
                estimates: d.estimates,
                timing: d.timing,
            })
            .into_response()
        }
        Err(e) => gateway_error(format, &e),
    }
}

/// Resolved price per configured model and where it came from.
async fn pricing(State(gw): State<AppState>) -> Response {
    let catalog = gw.catalog();
    Json(openapi::PricingResponse {
        models_dev: openapi::ModelsDevStatus {
            enabled: gw.config().pricing.models_dev,
            priced_models: catalog.len(),
        },
        models: gw.price_table(),
    })
    .into_response()
}

/// List the router model (when a router is configured) and the configured models.
/// Anthropic-shaped if the client sent `anthropic-version`, OpenAI-shaped otherwise.
async fn models(State(gw): State<AppState>, headers: HeaderMap) -> Response {
    let config = gw.config();
    let ids: Vec<&str> = gw
        .router()
        .map(|_| config.router_model.as_str())
        .into_iter()
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
