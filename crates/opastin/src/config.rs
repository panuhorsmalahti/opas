//! Configuration: which models exist, where each is served, and which router to use.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::api::ApiFormat;
use crate::router::{Candidate, RandomRouter, Router};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Address the server listens on.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// The virtual model name clients send to get a routed request.
    #[serde(default = "default_router_model")]
    pub router_model: String,
    #[serde(default)]
    pub router: RouterConfig,
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RouterConfig {
    /// Uniform random choice. `seed` makes the sequence reproducible.
    Random {
        #[serde(default)]
        seed: Option<u64>,
    },
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig::Random { seed: None }
    }
}

impl RouterConfig {
    pub fn build(&self) -> Arc<dyn Router> {
        match self {
            RouterConfig::Random { seed: Some(s) } => Arc::new(RandomRouter::with_seed(*s)),
            RouterConfig::Random { seed: None } => Arc::new(RandomRouter::new()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Id clients and routers use for this model.
    pub id: String,
    /// Model name sent upstream, if different from `id`.
    #[serde(default)]
    pub upstream_model: Option<String>,
    /// Where to send OpenAI Chat Completions requests for this model.
    #[serde(default)]
    pub openai: Option<Upstream>,
    /// Where to send Anthropic Messages requests for this model.
    #[serde(default)]
    pub anthropic: Option<Upstream>,
}

impl ModelConfig {
    pub fn upstream(&self, format: ApiFormat) -> Option<&Upstream> {
        match format {
            ApiFormat::OpenaiChat => self.openai.as_ref(),
            ApiFormat::AnthropicMessages => self.anthropic.as_ref(),
        }
    }

    pub fn upstream_model(&self) -> &str {
        self.upstream_model.as_deref().unwrap_or(&self.id)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Upstream {
    /// Base URL ending in the API version, e.g. `https://api.anthropic.com/v1`.
    pub base_url: String,
    /// Environment variable holding the upstream API key. If unset, the client's own
    /// credentials (`Authorization` / `x-api-key`) are forwarded.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
}

fn default_listen() -> String {
    "127.0.0.1:8080".to_owned()
}

fn default_router_model() -> String {
    "auto".to_owned()
}

impl Config {
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        let config: Config = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml_str(&text)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.models.is_empty() {
            return Err(ConfigError::Invalid(
                "at least one [[models]] entry is required".into(),
            ));
        }
        let mut seen = HashSet::new();
        for m in &self.models {
            if !seen.insert(m.id.as_str()) {
                return Err(ConfigError::Invalid(format!(
                    "duplicate model id `{}`",
                    m.id
                )));
            }
            if m.id == self.router_model {
                return Err(ConfigError::Invalid(format!(
                    "model id `{}` clashes with router_model",
                    m.id
                )));
            }
            if m.openai.is_none() && m.anthropic.is_none() {
                return Err(ConfigError::Invalid(format!(
                    "model `{}` needs an `openai` or `anthropic` upstream",
                    m.id
                )));
            }
        }
        Ok(())
    }

    pub fn model(&self, id: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Models that can serve requests in `format`, in config order.
    pub fn candidates(&self, format: ApiFormat) -> Vec<Candidate> {
        self.models
            .iter()
            .filter(|m| m.upstream(format).is_some())
            .map(|m| Candidate::new(&m.id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
        router_model = "auto"
        [router]
        kind = "random"
        seed = 3

        [[models]]
        id = "big"
        upstream_model = "big-2026"
        anthropic = { base_url = "https://a.example/v1", api_key_env = "A_KEY" }
        openai = { base_url = "https://a.example/v1" }

        [[models]]
        id = "small"
        openai = { base_url = "https://o.example/v1" }
    "#;

    #[test]
    fn parses_and_filters_by_format() {
        let c = Config::from_toml_str(EXAMPLE).unwrap();
        assert_eq!(c.listen, "127.0.0.1:8080");
        assert_eq!(c.router, RouterConfig::Random { seed: Some(3) });
        assert_eq!(c.model("big").unwrap().upstream_model(), "big-2026");
        assert_eq!(c.model("small").unwrap().upstream_model(), "small");
        let ids = |f| {
            c.candidates(f)
                .into_iter()
                .map(|c| c.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(ApiFormat::OpenaiChat), ["big", "small"]);
        assert_eq!(ids(ApiFormat::AnthropicMessages), ["big"]);
    }

    #[test]
    fn rejects_invalid_configs() {
        let dup = r#"
            [[models]]
            id = "x"
            openai = { base_url = "u" }
            [[models]]
            id = "x"
            openai = { base_url = "u" }
        "#;
        assert!(Config::from_toml_str(dup).is_err());
        let clash = "[[models]]\nid = \"auto\"\nopenai = { base_url = \"u\" }\n";
        assert!(Config::from_toml_str(clash).is_err());
        let no_upstream = "[[models]]\nid = \"x\"\n";
        assert!(Config::from_toml_str(no_upstream).is_err());
        assert!(Config::from_toml_str("models = []").is_err());
    }
}
