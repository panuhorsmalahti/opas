//! OpenAPI 3.1 description of the opas HTTP API, served at `/openapi.json`,
//! `/openapi.yaml` and as a browsable reference at `/docs`.
//!
//! opas's own endpoints (`/v1/route`, `/v1/pricing`, `/v1/models`, `/healthz`) are
//! described with their full request and response schemas, generated from the Rust
//! types. The four proxy endpoints accept and return the provider APIs' own bodies
//! unchanged (except for the model name), so their bodies are described as free-form
//! objects with a link to the provider's specification; the parts opas adds (the
//! `x-opas-*` headers) are described in full.

use std::collections::HashMap;

use opas::catalog::PriceSource;
use opas::gateway::ResolvedPrice;
use opas::{
    ApiFormat, Candidate, Conversation, Estimate, LongContextPricing, Preference, Pricing, Speed,
    Tier, Weights,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::openapi::{self, Server};
use utoipa::{Modify, OpenApi, ToSchema};

/// Body of `POST /v1/route`.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RouteBody {
    /// Format of `request`.
    pub api: ApiFormat,
    /// A request body in that format, exactly as it would be sent to the proxy endpoint.
    #[schema(value_type = HashMap<String, Value>)]
    pub request: Value,
    /// Models to choose between. Defaults to the configured models that serve `api`.
    #[serde(default)]
    pub candidates: Option<Vec<Candidate>>,
    /// Quality / speed / cost weights. Defaults to the router's (60 / 20 / 20).
    #[serde(default)]
    pub weights: Option<Weights>,
    /// A named preset instead of `weights`.
    #[serde(default)]
    pub preference: Option<Preference>,
    /// Prices by model id for this request (USD per million tokens), overriding the
    /// operator's and models.dev prices.
    #[serde(default)]
    pub pricing: HashMap<String, Pricing>,
}

/// A routing decision, with every candidate's estimates.
#[derive(Debug, Serialize, ToSchema)]
pub struct RouteResponse {
    /// Unique id of the decision, also logged by opas.
    pub decision_id: String,
    /// The router that decided: `probe`, or `affinity` when the request continues
    /// provider-stored state that lives on one model.
    pub router: String,
    /// The chosen model (a configured model id, or one of `candidates`).
    pub model: String,
    /// The provider's name for the chosen model, when it is a configured model.
    pub upstream_model: Option<String>,
    /// Human-readable explanation.
    pub reason: String,
    pub api: ApiFormat,
    /// The conversation this request continued, if opas recognized one.
    pub conversation: Option<Conversation>,
    /// The weights that were applied.
    pub weights: Option<Weights>,
    /// Every candidate's predictions and scores, best first.
    pub estimates: Vec<Estimate>,
}

/// Body of `GET /v1/pricing`.
#[derive(Debug, Serialize, ToSchema)]
pub struct PricingResponse {
    pub models_dev: ModelsDevStatus,
    /// The price in effect for each configured model.
    pub models: Vec<ResolvedPrice>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelsDevStatus {
    /// Whether prices are fetched from models.dev.
    pub enabled: bool,
    /// Number of models in the loaded catalog (0 until the first fetch succeeds).
    pub priced_models: usize,
}

/// `GET /v1/models` in the OpenAI shape (default).
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAiModelList {
    /// Always `list`.
    pub object: String,
    pub data: Vec<OpenAiModel>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAiModel {
    pub id: String,
    /// Always `model`.
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// `GET /v1/models` in the Anthropic shape (when the request has `anthropic-version`).
#[derive(Debug, Serialize, ToSchema)]
pub struct AnthropicModelList {
    pub data: Vec<AnthropicModel>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AnthropicModel {
    /// Always `model`.
    #[serde(rename = "type")]
    pub kind: String,
    pub id: String,
    pub display_name: String,
    pub created_at: String,
}

/// Errors from opas itself use the error shape of the API the request was made in.
/// This is the OpenAI shape, used by `/v1/route` and the OpenAI-format endpoints;
/// Anthropic-format endpoints wrap it as `{"type": "error", "error": {...}}`, and
/// Gemini Interactions use `{"error": {"code", "message", "status"}}`.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorDetail {
    pub message: String,
    /// e.g. `invalid_request_error`, `not_found_error`, `api_error`.
    #[serde(rename = "type")]
    pub kind: String,
}

// --- path descriptions -----------------------------------------------------------------
// The handlers live in lib.rs; these stubs carry the OpenAPI annotations.

#[utoipa::path(
    post, path = "/v1/route", tag = "routing",
    summary = "Decide which model a request should go to, without sending it",
    description = "Runs the router on a request body in any supported API format and returns the chosen model with every candidate's predicted success probability, cost and response time. Nothing is sent upstream.",
    request_body = RouteBody,
    responses(
        (status = 200, description = "The decision", body = RouteResponse),
        (status = 400, description = "Invalid body, empty `candidates`, invalid weights, or no router configured", body = ErrorResponse),
        (status = 404, description = "A named model is not configured", body = ErrorResponse),
    )
)]
#[allow(dead_code)]
fn route() {}

#[utoipa::path(
    get, path = "/v1/pricing", tag = "routing",
    summary = "Prices in effect for the configured models",
    description = "Each configured model's price (USD per million tokens) and where it came from: the config, models.dev, or the tier's default. Per-request prices (`x-opas-pricing`) are not included.",
    responses((status = 200, description = "Resolved prices", body = PricingResponse))
)]
#[allow(dead_code)]
fn pricing() {}

#[utoipa::path(
    get, path = "/v1/models", tag = "models",
    summary = "List the router model and the configured models",
    description = "The router model (`auto` by default, only when a router is configured) followed by the configured models. Anthropic-shaped when the request carries `anthropic-version`, OpenAI-shaped otherwise.",
    params(("anthropic-version" = Option<String>, Header, description = "Any value selects the Anthropic response shape")),
    responses((status = 200, description = "Model list (OpenAI shape; `AnthropicModelList` with `anthropic-version`)", body = OpenAiModelList))
)]
#[allow(dead_code)]
fn models() {}

#[utoipa::path(
    get, path = "/healthz", tag = "server",
    summary = "Liveness check",
    responses((status = 200, description = "The server is up", body = String, content_type = "text/plain", example = "ok"))
)]
#[allow(dead_code)]
fn healthz() {}

/// Shared description of the proxy endpoints' headers and responses.
macro_rules! proxy_path {
    ($fn:ident, $path:literal, $summary:literal, $spec:literal) => {
        #[utoipa::path(
            post, path = $path, tag = "proxy",
            summary = $summary,
            description = concat!(
                "Send a request body exactly as the provider's API expects it (", $spec, "). ",
                "With `model` set to the router model (`auto`), opas picks a model; with a configured model id, the request goes straight to that model. ",
                "opas rewrites `model` to the provider's name, forwards the request, and streams the response back unchanged, adding the `x-opas-*` headers. ",
                "Without a server-side key for the model, the client's own credentials are forwarded."
            ),
            request_body(content = HashMap<String, Value>, description = "A request body in the provider's format", content_type = "application/json"),
            params(
                ("x-opas-weights" = Option<String>, Header, description = "Quality / speed / cost weights for this request, e.g. `quality=60,speed=20,cost=20`"),
                ("x-opas-preference" = Option<String>, Header, description = "A preset instead of weights: `quality`, `balanced`, `cost` or `speed`"),
                ("x-opas-pricing" = Option<String>, Header, description = "Prices for this request, USD per million tokens: `model=input/cached/output; other=input/output`"),
            ),
            responses(
                (status = 200, description = "The provider's response, unchanged (JSON, or `text/event-stream` when streaming)",
                    content((HashMap<String, Value> = "application/json"), (String = "text/event-stream")),
                    headers(
                        ("x-opas-model" = String, description = "The configured model that served the request"),
                        ("x-opas-router" = String, description = "Routed requests: the router that decided (`probe` or `affinity`)"),
                        ("x-opas-decision-id" = String, description = "Routed requests: the decision's id"),
                        ("x-opas-estimate" = String, description = "Routed requests: the chosen model's `p_success`, `cost_usd` and `response_time_s`"),
                        ("x-opas-previous-model" = String, description = "Continued conversations: the model that served the previous turn"),
                    )),
                (status = 400, description = "Invalid request, or the router model was requested without a router"),
                (status = 404, description = "Unknown model"),
                (status = 502, description = "The provider could not be reached"),
            )
        )]
        #[allow(dead_code)]
        fn $fn() {}
    };
}

proxy_path!(
    chat_completions,
    "/v1/chat/completions",
    "OpenAI Chat Completions, routed",
    "https://platform.openai.com/docs/api-reference/chat"
);
proxy_path!(
    messages,
    "/v1/messages",
    "Anthropic Messages, routed",
    "https://docs.anthropic.com/en/api/messages"
);
proxy_path!(
    responses,
    "/v1/responses",
    "Open Responses, routed",
    "https://www.openresponses.org"
);
proxy_path!(
    interactions,
    "/v1beta/interactions",
    "Gemini Interactions, routed (also served at /v1/interactions)",
    "https://ai.google.dev/api/interactions"
);

#[derive(OpenApi)]
#[openapi(
    info(
        title = "opas",
        description = "A model router for LLM requests. opas predicts each candidate model's success probability, cost and response time from an open-weight model's activations, and picks one by the quality / speed / cost weights the client sets. It serves four provider APIs as a routing proxy, plus a decision-only endpoint.",
        license(name = "MIT", identifier = "MIT"),
    ),
    paths(route, pricing, models, healthz, chat_completions, messages, responses, interactions),
    components(schemas(
        RouteBody, RouteResponse, PricingResponse, ModelsDevStatus, OpenAiModelList, OpenAiModel,
        AnthropicModelList, AnthropicModel, ErrorResponse, ErrorDetail, ApiFormat, Candidate,
        Tier, Pricing, LongContextPricing, Speed, Weights, Preference, Estimate, Conversation,
        ResolvedPrice, PriceSource,
    )),
    modifiers(&LocalServer),
    tags(
        (name = "routing", description = "Routing decisions and prices"),
        (name = "proxy", description = "Provider APIs, routed by opas"),
        (name = "models", description = "Model listing"),
        (name = "server", description = "Operations"),
    )
)]
pub struct ApiDoc;

struct LocalServer;

impl Modify for LocalServer {
    fn modify(&self, api: &mut openapi::OpenApi) {
        api.servers = Some(vec![Server::new("http://127.0.0.1:8080")]);
    }
}

/// The OpenAPI document.
pub fn spec() -> openapi::OpenApi {
    ApiDoc::openapi()
}

/// A browsable reference for the spec, rendered by Scalar (loaded from a CDN).
pub const DOCS_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>opas API</title>
</head>
<body>
  <script id="api-reference" data-url="openapi.json"></script>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.78.1"></script>
</body>
</html>
"#;
