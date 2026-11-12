//! Forwarding a request to the chosen model's upstream (feature `gateway`).

use std::sync::Arc;

use http::header::AUTHORIZATION;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::api::ApiFormat;
use crate::config::{Config, ConfigError, Upstream};
use crate::conversation::{ConversationTracker, PrefixKeys};
use crate::request::{RequestError, RouteRequest};
use crate::router::{Candidate, Decision, RouteError, Router};

/// `anthropic-version` sent upstream when the client did not send one.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const ANTHROPIC_VERSION: HeaderName = HeaderName::from_static("anthropic-version");
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");
const OPENAI_ORGANIZATION: HeaderName = HeaderName::from_static("openai-organization");
const OPENAI_PROJECT: HeaderName = HeaderName::from_static("openai-project");
const X_GOOG_API_KEY: HeaderName = HeaderName::from_static("x-goog-api-key");

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
}

/// Routes requests and forwards them to the chosen model's upstream.
pub struct Gateway {
    config: Config,
    router: Arc<dyn Router>,
    tracker: Option<ConversationTracker>,
    http: reqwest::Client,
}

impl Gateway {
    /// Build with the router named in `config.router`. Fails for routers this crate
    /// cannot build (the probe router); use [`Gateway::with_router`] for those.
    pub fn new(config: Config) -> Result<Self, ConfigError> {
        let router = config.router.build().ok_or_else(|| {
            ConfigError::Invalid(
                "this router kind must be built by the caller (Gateway::with_router)".into(),
            )
        })?;
        Ok(Self::with_router(config, router))
    }

    /// Build with a custom router.
    pub fn with_router(config: Config, router: Arc<dyn Router>) -> Self {
        Self {
            tracker: config.cache.tracker(),
            config,
            router,
            http: reqwest::Client::new(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn router(&self) -> &Arc<dyn Router> {
        &self.router
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
        self.decide_among(format, body, self.config.candidates(format))
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
        let mut request = RouteRequest::from_json(format, body)?;
        let keys = self.tracker.as_ref().map(|_| PrefixKeys::new(format, body));
        if let (Some(tracker), Some(keys)) = (&self.tracker, &keys) {
            request.conversation = tracker.lookup(keys);
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
        let mut decision = self.router.route(&request, &candidates).await?;
        if let (Some(tracker), Some(keys)) = (&self.tracker, &keys) {
            tracker.record(keys, &decision.model);
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
            let decision = self.decide(format, &body).await?;
            let conversation = decision.conversation.clone();
            (decision.model.clone(), Some(decision), conversation)
        } else {
            // Direct requests also update the tracker, so a later routed turn of the same
            // conversation knows which model's cache is warm.
            let conversation = self.tracker.as_ref().and_then(|t| {
                let keys = PrefixKeys::new(format, &body);
                let conversation = t.lookup(&keys);
                if self.config.model(&requested).is_some() {
                    t.record(&keys, &requested);
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

        body["model"] = Value::String(model.upstream_model().to_owned());
        let url = format!(
            "{}{}",
            upstream.base_url.trim_end_matches('/'),
            format.endpoint_path()
        );
        let headers = upstream_headers(format, upstream, client_headers)?;

        tracing::debug!(%format, model = %model_id, %url, "forwarding");
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
        })
    }
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
