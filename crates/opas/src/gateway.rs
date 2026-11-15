//! Forwarding a request to the chosen model's upstream (feature `gateway`).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use http::header::AUTHORIZATION;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::api::ApiFormat;
use crate::catalog::{parse_price_overrides, PriceCatalog, PriceSource};
use crate::config::{Config, ConfigError, Upstream};
use crate::conversation::Pricing;
use crate::conversation::{ConversationTracker, PrefixKeys};
use crate::request::{RequestError, RouteRequest};
use crate::router::{Candidate, Decision, RouteError, Router};
use crate::scoring::{Preference, Weights};

/// `anthropic-version` sent upstream when the client did not send one.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const ANTHROPIC_VERSION: HeaderName = HeaderName::from_static("anthropic-version");
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");
const OPENAI_ORGANIZATION: HeaderName = HeaderName::from_static("openai-organization");
const OPENAI_PROJECT: HeaderName = HeaderName::from_static("openai-project");
const X_GOOG_API_KEY: HeaderName = HeaderName::from_static("x-goog-api-key");
/// Request header: quality / speed / cost weights, e.g. `quality=60,speed=20,cost=20`.
pub const HEADER_WEIGHTS: &str = "x-opas-weights";
/// Request header: a named preset (`quality`, `balanced`, `cost`, `speed`).
pub const HEADER_PREFERENCE: &str = "x-opas-preference";

/// Request header: per-request model prices, `model=input/cached/output; …`
/// (USD per million tokens).
pub const HEADER_PRICING: &str = "x-opas-pricing";

/// What a consumer can set per request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestOptions {
    /// Quality / speed / cost weights (default: the router's).
    pub weights: Option<Weights>,
    /// Prices by model id, overriding the operator's and models.dev prices.
    pub pricing: HashMap<String, Pricing>,
    /// A client-supplied conversation id (`x-opas-conversation-id`, or Codex's
    /// `thread-id`): turns with the same id are recognized as one conversation even
    /// when the history fingerprints differ.
    pub conversation_id: Option<String>,
}

/// Request header: a client-chosen id shared by all turns of one conversation.
pub const HEADER_CONVERSATION_ID: &str = "x-opas-conversation-id";

/// The conversation id from `x-opas-conversation-id`, else Codex's `thread-id`.
pub fn conversation_id_from_headers(headers: &HeaderMap) -> Option<String> {
    [HEADER_CONVERSATION_ID, "thread-id"]
        .iter()
        .find_map(|h| headers.get(*h)?.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty() && v.len() <= 256)
        .map(str::to_owned)
}

impl RequestOptions {
    /// From `x-opas-weights` / `x-opas-preference` and `x-opas-pricing`.
    pub fn from_headers(headers: &HeaderMap) -> Result<Self, GatewayError> {
        let pricing = match headers.get(HEADER_PRICING) {
            Some(v) => parse_price_overrides(v.to_str().unwrap_or_default())
                .map_err(|e| GatewayError::BadRequest(format!("{HEADER_PRICING}: {e}")))?,
            None => HashMap::new(),
        };
        Ok(Self {
            weights: weights_from_headers(headers)?,
            conversation_id: conversation_id_from_headers(headers),
            pricing,
        })
    }
}

/// A model's resolved price and where it came from.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResolvedPrice {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Pricing>,
    pub source: PriceSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models_dev: Option<String>,
}

/// Per-request weights from `x-opas-weights` or `x-opas-preference` (explicit weights
/// win). `None` means the router's defaults.
pub fn weights_from_headers(headers: &HeaderMap) -> Result<Option<Weights>, GatewayError> {
    let get = |name: &str| {
        headers
            .get(name)
            .map(|v| v.to_str().unwrap_or_default().trim().to_owned())
    };
    if let Some(v) = get(HEADER_WEIGHTS) {
        return v
            .parse::<Weights>()
            .map(Some)
            .map_err(|e| GatewayError::BadRequest(format!("{HEADER_WEIGHTS}: {e}")));
    }
    if let Some(v) = get(HEADER_PREFERENCE) {
        return v
            .parse::<Preference>()
            .map(|p| Some(p.weights()))
            .map_err(|e| GatewayError::BadRequest(format!("{HEADER_PREFERENCE}: {e}")));
    }
    Ok(None)
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("unknown model `{0}`")]
    UnknownModel(String),
    #[error("model `{model}` has no {format} upstream configured")]
    UnsupportedFormat { model: String, format: ApiFormat },
    #[error(transparent)]
    Route(#[from] RouteError),
    #[error("upstream API key: environment variable `{0}` is not set or not a valid header value")]
    ApiKey(String),
    #[error("upstream request failed: {0}")]
    Upstream(#[from] reqwest::Error),
}

impl From<RequestError> for GatewayError {
    fn from(e: RequestError) -> Self {
        GatewayError::BadRequest(e.to_string())
    }
}

/// An upstream response plus which model served it.
pub struct Forwarded {
    /// The raw upstream response. Its body may be a stream (SSE) if the client asked for one.
    pub response: reqwest::Response,
    /// The configured model id the request went to.
    pub model: String,
    /// The routing decision, if the client asked for the router model.
    pub decision: Option<Decision>,
    /// The conversation this request continued, if cache-aware routing recognized it.
    pub conversation: Option<crate::Conversation>,
    /// When the request was sent upstream (after routing), for timing the response.
    pub sent_at: std::time::Instant,
}

/// Routes requests and forwards them to the chosen model's upstream.
pub struct Gateway {
    config: Config,
    router: Option<Arc<dyn Router>>,
    tracker: Option<ConversationTracker>,
    catalog: RwLock<Arc<PriceCatalog>>,
    http: reqwest::Client,
    decisions: crate::decisions::DecisionLog,
}

impl Gateway {
    /// A plain proxy without a router: requests must name a configured model. Fails if
    /// the config has a `[router]`, which must be built by the caller (it needs
    /// `opas-encoder`) and passed to [`Gateway::with_router`].
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        if config.router.is_some() {
            return Err(ConfigError::Invalid(
                "the configured router must be built by the caller (Gateway::with_router)".into(),
            ));
        }
        Ok(Self::build(config, None))
    }

    /// Build with a router.
    pub fn with_router(config: Config, router: Arc<dyn Router>) -> Self {
        Self::build(config, Some(router))
    }

    fn build(config: Config, router: Option<Arc<dyn Router>>) -> Self {
        Self {
            tracker: config.cache.tracker(),
            config,
            router,
            catalog: RwLock::new(Arc::new(PriceCatalog::default())),
            http: reqwest::Client::new(),
            decisions: crate::decisions::DecisionLog::default(),
        }
    }

    /// Recent decisions and their measured outcomes.
    pub fn decisions(&self) -> &crate::decisions::DecisionLog {
        &self.decisions
    }

    /// The price that applies to `model_id` for a request with these client headers: a
    /// per-request `x-opas-pricing` override, else the config, models.dev or the tier's
    /// default.
    pub fn price_for(&self, model_id: &str, client_headers: &HeaderMap) -> Option<Pricing> {
        RequestOptions::from_headers(client_headers)
            .ok()
            .and_then(|o| o.pricing.get(model_id).copied())
            .or_else(|| self.model_price(model_id).0)
    }

    /// Replace the models.dev price catalog.
    pub fn set_catalog(&self, catalog: PriceCatalog) {
        *self.catalog.write().unwrap_or_else(|p| p.into_inner()) = Arc::new(catalog);
    }

    pub fn catalog(&self) -> Arc<PriceCatalog> {
        self.catalog
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Load the catalog from `[pricing] cache_file`, if it exists. Returns the number of
    /// priced models.
    pub fn load_price_cache(&self) -> Option<usize> {
        let path = self.config.pricing.cache_file.as_ref()?;
        let doc: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
        let catalog = PriceCatalog::from_models_dev(&doc);
        let n = catalog.len();
        (n > 0).then(|| self.set_catalog(catalog)).map(|_| n)
    }

    /// Fetch the models.dev catalog and use it; also save it to `cache_file`. Returns the
    /// number of priced models.
    pub async fn refresh_prices(&self) -> Result<usize, GatewayError> {
        let body = self
            .http
            .get(&self.config.pricing.models_dev_url)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        let doc: Value = serde_json::from_slice(&body)
            .map_err(|e| GatewayError::BadRequest(format!("models.dev: invalid JSON: {e}")))?;
        let catalog = PriceCatalog::from_models_dev(&doc);
        let n = catalog.len();
        if n == 0 {
            return Err(GatewayError::BadRequest(
                "models.dev: no priced models".into(),
            ));
        }
        self.set_catalog(catalog);
        if let Some(path) = &self.config.pricing.cache_file {
            if let Err(e) = std::fs::write(path, &body) {
                tracing::warn!(path = %path.display(), error = %e, "could not write the price cache");
            }
        }
        Ok(n)
    }

    /// The operator-level price of a configured model: config `pricing`, else models.dev
    /// (explicit key, or inferred from the upstream for any API format), else none.
    fn model_price(&self, model_id: &str) -> (Option<Pricing>, PriceSource, Option<String>) {
        let Some(m) = self.config.model(model_id) else {
            return (None, PriceSource::Unknown, None);
        };
        if let Some(p) = m.pricing {
            return (Some(p), PriceSource::Config, m.models_dev.clone());
        }
        if self.config.pricing.models_dev {
            let catalog = self.catalog();
            if let Some(key) = &m.models_dev {
                if let Some(p) = catalog.get(key) {
                    return (Some(p), PriceSource::ModelsDev, Some(key.clone()));
                }
            }
            for format in ApiFormat::ALL {
                if let Some(up) = m.upstream(format) {
                    if let Some((key, p)) = catalog.infer(&up.base_url, m.upstream_model()) {
                        return (Some(p), PriceSource::ModelsDev, Some(key));
                    }
                }
            }
        }
        let source = if m.tier.is_some() {
            PriceSource::TierDefault
        } else {
            PriceSource::Unknown
        };
        (
            m.tier.map(crate::Tier::default_pricing),
            source,
            m.models_dev.clone(),
        )
    }

    /// Resolved prices of all configured models, for `/v1/pricing`.
    pub fn price_table(&self) -> Vec<ResolvedPrice> {
        self.config
            .models
            .iter()
            .map(|m| {
                let (pricing, source, models_dev) = self.model_price(&m.id);
                ResolvedPrice {
                    model: m.id.clone(),
                    pricing,
                    source,
                    models_dev,
                }
            })
            .collect()
    }

    /// Fill in candidate prices: request prices, then the candidate's own `pricing`, then
    /// its `models_dev` key, then (configured models) the operator price. Candidates left
    /// without a price fall back to their tier's default when scored.
    fn price_candidates(&self, candidates: &mut [Candidate], overrides: &HashMap<String, Pricing>) {
        let catalog = self.catalog();
        for c in candidates.iter_mut() {
            // A caller-supplied candidate that names a configured model inherits its tier
            // and speed unless it sets its own.
            if let Some(m) = self.config.model(&c.id) {
                c.tier = c.tier.or(m.tier);
                c.speed = c.speed.or(m.speed);
            }
            if let Some(p) = overrides.get(&c.id) {
                c.pricing = Some(*p);
                continue;
            }
            if c.pricing.is_some() {
                continue;
            }
            if let Some(key) = &c.models_dev {
                if let Some(p) = catalog.get(key) {
                    c.pricing = Some(p);
                    continue;
                }
            }
            if let (Some(p), PriceSource::Config | PriceSource::ModelsDev, _) =
                self.model_price(&c.id)
            {
                c.pricing = Some(p);
            }
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The router, if any (a plain proxy has none).
    pub fn router(&self) -> Option<&Arc<dyn Router>> {
        self.router.as_ref()
    }

    /// Choose a model for a request body without sending it anywhere, among the
    /// configured models that serve this API format.
    ///
    /// With cache-aware routing on, the request is matched against earlier turns; the
    /// router sees the previous model in `RouteRequest::conversation`, and the decision
    /// is recorded so the next turn can find it. A request that continues a
    /// provider-stored conversation (`previous_response_id` / `previous_interaction_id`)
    /// is pinned to the model that created it, without asking the router.
    pub async fn decide(&self, format: ApiFormat, body: &Value) -> Result<Decision, GatewayError> {
        self.decide_with(format, body, None, RequestOptions::default())
            .await
    }

    /// Like [`Gateway::decide`], with a caller-supplied candidate list (for example
    /// models described only by tier and pricing in a `/v1/route` request).
    pub async fn decide_among(
        &self,
        format: ApiFormat,
        body: &Value,
        candidates: Vec<Candidate>,
    ) -> Result<Decision, GatewayError> {
        self.decide_with(format, body, Some(candidates), RequestOptions::default())
            .await
    }

    /// The general form: optional caller-supplied candidates (default: configured models
    /// for this format) and per-request options (weights, prices).
    pub async fn decide_with(
        &self,
        format: ApiFormat,
        body: &Value,
        candidates: Option<Vec<Candidate>>,
        options: RequestOptions,
    ) -> Result<Decision, GatewayError> {
        let mut candidates = candidates.unwrap_or_else(|| self.config.candidates(format));
        for p in options.pricing.values() {
            p.validate().map_err(GatewayError::BadRequest)?;
        }
        self.price_candidates(&mut candidates, &options.pricing);
        let mut request = RouteRequest::from_json(format, body)?;
        request.weights = options
            .weights
            .map(Weights::validate)
            .transpose()
            .map_err(GatewayError::BadRequest)?;
        let keys = self.tracker.as_ref().map(|_| PrefixKeys::new(format, body));
        if let (Some(tracker), Some(keys)) = (&self.tracker, &keys) {
            request.conversation = tracker.lookup(keys).or_else(|| {
                options
                    .conversation_id
                    .as_deref()
                    .and_then(|id| tracker.lookup_thread(id, keys))
            });
        }
        if let Some(id) = &request.continuation_id {
            match self.tracker.as_ref().and_then(|t| t.lookup_stored(id)) {
                Some(model) if candidates.iter().any(|c| c.id == model) => {
                    return Ok(Decision {
                        decision_id: uuid::Uuid::new_v4().to_string(),
                        router: "affinity".into(),
                        reason: format!(
                            "`{}` refers to state stored at `{model}`; only that model can continue it",
                            format.continuation_field().unwrap_or("continuation id")
                        ),
                        model,
                        conversation: request.conversation,
                        estimates: vec![],
                        weights: None,
                        timing: None,
                    });
                }
                Some(model) => {
                    return Err(GatewayError::BadRequest(format!(
                        "this conversation is stored at model `{model}`, which is not a candidate"
                    )))
                }
                None => tracing::warn!(
                    %id,
                    "continuation id not seen by this router (expired or created elsewhere); routing normally"
                ),
            }
        }
        let Some(router) = &self.router else {
            return Err(GatewayError::BadRequest(format!(
                "no router is configured: request a configured model by id instead of `{}`",
                self.config.router_model
            )));
        };
        let started = std::time::Instant::now();
        let mut decision = router.route(&request, &candidates).await?;
        let total_ms = started.elapsed().as_secs_f64() * 1e3;
        decision.timing = Some(crate::router::RouterTiming {
            total_ms,
            ..decision.timing.unwrap_or_default()
        });
        if let (Some(tracker), Some(keys)) = (&self.tracker, &keys) {
            tracker.record(keys, &decision.model);
            if let Some(id) = &options.conversation_id {
                tracker.record_thread(id, &decision.model);
            }
        }
        decision.conversation = request.conversation;
        Ok(decision)
    }

    /// Remember that `model` produced the provider-stored response `id`, so follow-ups
    /// that reference it are pinned to `model`. Called with the id of each upstream
    /// response for APIs that store state (Open Responses, Gemini Interactions).
    pub fn record_stored_response(&self, id: &str, model: &str) {
        if let Some(t) = &self.tracker {
            t.record_stored(id, model);
        }
    }

    /// Send a request upstream. If its `model` is the router model, route it first;
    /// if it names a configured model, send it there directly.
    pub async fn forward(
        &self,
        format: ApiFormat,
        mut body: Value,
        client_headers: &HeaderMap,
    ) -> Result<Forwarded, GatewayError> {
        let requested = body
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| GatewayError::BadRequest("missing `model` field".into()))?
            .to_owned();

        let (model_id, decision, conversation) = if requested == self.config.router_model {
            let options = RequestOptions::from_headers(client_headers)?;
            let decision = self.decide_with(format, &body, None, options).await?;
            let conversation = decision.conversation.clone();
            (decision.model.clone(), Some(decision), conversation)
        } else {
            // Direct requests also update the tracker, so a later routed turn of the same
            // conversation knows which model's cache is warm.
            let thread = conversation_id_from_headers(client_headers);
            let conversation = self.tracker.as_ref().and_then(|t| {
                let keys = PrefixKeys::new(format, &body);
                let conversation = t
                    .lookup(&keys)
                    .or_else(|| thread.as_deref().and_then(|id| t.lookup_thread(id, &keys)));
                if self.config.model(&requested).is_some() {
                    t.record(&keys, &requested);
                    if let Some(id) = &thread {
                        t.record_thread(id, &requested);
                    }
                }
                conversation
            });
            (requested, None, conversation)
        };

        let model = self
            .config
            .model(&model_id)
            .ok_or_else(|| GatewayError::UnknownModel(model_id.clone()))?;
        let upstream = model
            .upstream(format)
            .ok_or_else(|| GatewayError::UnsupportedFormat {
                model: model_id.clone(),
                format,
            })?;

        // Encrypted reasoning items can only be read by the model that produced them.
        // Drop them unless that is known to be the model this request goes to: for
        // routed requests whenever the previous model is unknown or different, for
        // direct requests only when it is known to be different.
        if format == ApiFormat::OpenResponses {
            let same = conversation.as_ref().map(|c| c.previous_model == model_id);
            let strip = if decision.is_some() {
                same != Some(true)
            } else {
                same == Some(false)
            };
            if strip {
                let n = strip_reasoning_items(&mut body);
                if n > 0 {
                    tracing::info!(
                        model = %model_id,
                        previous_model = conversation.as_ref().map_or("unknown", |c| c.previous_model.as_str()),
                        items = n,
                        "dropped reasoning items from another model"
                    );
                }
            }
        }

        body["model"] = Value::String(model.upstream_model().to_owned());
        let url = format!(
            "{}{}",
            upstream.base_url.trim_end_matches('/'),
            format.endpoint_path()
        );
        let headers = upstream_headers(format, upstream, client_headers)?;

        tracing::debug!(%format, model = %model_id, %url, "forwarding");
        let sent_at = std::time::Instant::now();
        let response = self
            .http
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;
        Ok(Forwarded {
            response,
            model: model_id,
            decision,
            conversation,
            sent_at,
        })
    }
}

/// Remove `reasoning` items from an Open Responses `input` array. Returns how many.
pub fn strip_reasoning_items(body: &mut Value) -> usize {
    let Some(Value::Array(items)) = body.get_mut("input") else {
        return 0;
    };
    let before = items.len();
    items.retain(|i| i.get("type").and_then(Value::as_str) != Some("reasoning"));
    before - items.len()
}

/// Auth and protocol headers for the upstream request. A configured key replaces the
/// client's credentials; otherwise the client's credentials are passed through.
fn upstream_headers(
    format: ApiFormat,
    upstream: &Upstream,
    client: &HeaderMap,
) -> Result<HeaderMap, GatewayError> {
    let key = match &upstream.api_key_env {
        Some(var) => {
            let value = std::env::var(var).map_err(|_| GatewayError::ApiKey(var.clone()))?;
            Some((var, value))
        }
        None => None,
    };
    let header_value = |var: &String, v: String| {
        HeaderValue::from_str(&v).map_err(|_| GatewayError::ApiKey(var.clone()))
    };

    let mut h = HeaderMap::new();
    let copy = |h: &mut HeaderMap, name: &HeaderName| {
        if let Some(v) = client.get(name) {
            h.insert(name.clone(), v.clone());
        }
    };

    match format {
        ApiFormat::OpenaiChat | ApiFormat::OpenResponses => {
            match key {
                Some((var, k)) => {
                    h.insert(AUTHORIZATION, header_value(var, format!("Bearer {k}"))?);
                }
                None => copy(&mut h, &AUTHORIZATION),
            }
            copy(&mut h, &OPENAI_ORGANIZATION);
            copy(&mut h, &OPENAI_PROJECT);
        }
        ApiFormat::GeminiInteractions => match key {
            Some((var, k)) => {
                h.insert(X_GOOG_API_KEY, header_value(var, k)?);
            }
            None => {
                copy(&mut h, &X_GOOG_API_KEY);
                copy(&mut h, &AUTHORIZATION);
            }
        },
        ApiFormat::AnthropicMessages => {
            match key {
                Some((var, k)) => {
                    h.insert(X_API_KEY, header_value(var, k)?);
                }
                None => {
                    copy(&mut h, &X_API_KEY);
                    copy(&mut h, &AUTHORIZATION);
                }
            }
            let version = client
                .get(&ANTHROPIC_VERSION)
                .cloned()
                .unwrap_or(HeaderValue::from_static(DEFAULT_ANTHROPIC_VERSION));
            h.insert(ANTHROPIC_VERSION, version);
            copy(&mut h, &ANTHROPIC_BETA);
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_reasoning_items_and_reads_conversation_ids() {
        let mut body = serde_json::json!({"input": [
            {"type": "message", "role": "user", "content": "a"},
            {"type": "reasoning", "encrypted_content": "x"},
            {"type": "function_call", "name": "f"},
            {"type": "reasoning", "encrypted_content": "y"}
        ]});
        assert_eq!(strip_reasoning_items(&mut body), 2);
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        assert_eq!(
            strip_reasoning_items(&mut serde_json::json!({"input": "text"})),
            0
        );

        let mut h = HeaderMap::new();
        assert_eq!(conversation_id_from_headers(&h), None);
        h.insert("thread-id", HeaderValue::from_static("t1"));
        assert_eq!(conversation_id_from_headers(&h).as_deref(), Some("t1"));
        h.insert(HEADER_CONVERSATION_ID, HeaderValue::from_static("c1"));
        assert_eq!(
            conversation_id_from_headers(&h).as_deref(),
            Some("c1"),
            "explicit header wins"
        );
    }

    fn upstream(env: Option<&str>) -> Upstream {
        Upstream {
            base_url: "http://x/v1".into(),
            api_key_env: env.map(str::to_owned),
        }
    }

    #[test]
    fn passes_client_credentials_when_no_key_configured() {
        let mut client = HeaderMap::new();
        client.insert(AUTHORIZATION, HeaderValue::from_static("Bearer client"));
        client.insert(X_API_KEY, HeaderValue::from_static("client-key"));
        client.insert("cookie", HeaderValue::from_static("secret"));

        let h = upstream_headers(ApiFormat::OpenaiChat, &upstream(None), &client).unwrap();
        assert_eq!(h.get(AUTHORIZATION).unwrap(), "Bearer client");
        assert!(h.get("cookie").is_none());

        let h = upstream_headers(ApiFormat::AnthropicMessages, &upstream(None), &client).unwrap();
        assert_eq!(h.get(X_API_KEY).unwrap(), "client-key");
        assert_eq!(h.get(ANTHROPIC_VERSION).unwrap(), DEFAULT_ANTHROPIC_VERSION);
    }

    #[test]
    fn configured_key_replaces_client_credentials() {
        // Unique variable name so parallel tests don't interfere.
        let var = "OPAS_TEST_KEY_REPLACES";
        std::env::set_var(var, "server-key");
        let mut client = HeaderMap::new();
        client.insert(X_API_KEY, HeaderValue::from_static("client-key"));
        client.insert(ANTHROPIC_VERSION, HeaderValue::from_static("2026-10-21"));

        let h =
            upstream_headers(ApiFormat::AnthropicMessages, &upstream(Some(var)), &client).unwrap();
        assert_eq!(h.get(X_API_KEY).unwrap(), "server-key");
        assert_eq!(h.get(ANTHROPIC_VERSION).unwrap(), "2026-10-21");

        let h = upstream_headers(ApiFormat::OpenaiChat, &upstream(Some(var)), &client).unwrap();
        assert_eq!(h.get(AUTHORIZATION).unwrap(), "Bearer server-key");
    }

    #[test]
    fn missing_key_env_is_an_error() {
        let err = upstream_headers(
            ApiFormat::OpenaiChat,
            &upstream(Some("OPAS_TEST_KEY_DOES_NOT_EXIST")),
            &HeaderMap::new(),
        )
        .unwrap_err();
        assert!(matches!(err, GatewayError::ApiKey(_)));
    }
}
