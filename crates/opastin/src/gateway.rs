//! Forwarding a request to the chosen model's upstream (feature `gateway`).

use std::sync::Arc;

use http::header::AUTHORIZATION;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::api::ApiFormat;
use crate::config::{Config, Upstream};
use crate::request::{RequestError, RouteRequest};
use crate::router::{Decision, RouteError, Router};

/// `anthropic-version` sent upstream when the client did not send one.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const ANTHROPIC_VERSION: HeaderName = HeaderName::from_static("anthropic-version");
const ANTHROPIC_BETA: HeaderName = HeaderName::from_static("anthropic-beta");
const OPENAI_ORGANIZATION: HeaderName = HeaderName::from_static("openai-organization");
const OPENAI_PROJECT: HeaderName = HeaderName::from_static("openai-project");

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
}

/// Routes requests and forwards them to the chosen model's upstream.
pub struct Gateway {
    config: Config,
    router: Arc<dyn Router>,
    http: reqwest::Client,
}

impl Gateway {
    /// Build with the router named in `config.router`.
    pub fn new(config: Config) -> Self {
        let router = config.router.build();
        Self::with_router(config, router)
    }

    /// Build with a custom router.
    pub fn with_router(config: Config, router: Arc<dyn Router>) -> Self {
        Self {
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

    /// Choose a model for a request body without sending it anywhere.
    pub async fn decide(&self, format: ApiFormat, body: &Value) -> Result<Decision, GatewayError> {
        let request = RouteRequest::from_json(format, body)?;
        let candidates = self.config.candidates(format);
        Ok(self.router.route(&request, &candidates).await?)
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

        let (model_id, decision) = if requested == self.config.router_model {
            let decision = self.decide(format, &body).await?;
            (decision.model.clone(), Some(decision))
        } else {
            (requested, None)
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
        ApiFormat::OpenaiChat => {
            match key {
                Some((var, k)) => {
                    h.insert(AUTHORIZATION, header_value(var, format!("Bearer {k}"))?);
                }
                None => copy(&mut h, &AUTHORIZATION),
            }
            copy(&mut h, &OPENAI_ORGANIZATION);
            copy(&mut h, &OPENAI_PROJECT);
        }
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
        let var = "OPASTIN_TEST_KEY_REPLACES";
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
            &upstream(Some("OPASTIN_TEST_KEY_DOES_NOT_EXIST")),
            &HeaderMap::new(),
        )
        .unwrap_err();
        assert!(matches!(err, GatewayError::ApiKey(_)));
    }
}
