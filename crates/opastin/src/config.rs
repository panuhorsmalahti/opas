//! Configuration: which models exist, where each is served, and which router to use.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::api::ApiFormat;
use crate::conversation::{ConversationTracker, Pricing};
use crate::router::{Candidate, RandomRouter, Router};
use crate::tier::Tier;

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
    #[serde(default)]
    pub cache: CacheConfig,
    pub models: Vec<ModelConfig>,
}

/// Cache-aware routing: track which model served each conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// How long a provider's prompt cache is assumed to stay warm.
    #[serde(default = "default_cache_ttl")]
    pub ttl_secs: u64,
    /// Upper bound on tracked conversation turns kept in memory.
    #[serde(default = "default_max_conversations")]
    pub max_conversations: usize,
    /// How long provider-stored response ids (`previous_response_id`,
    /// `previous_interaction_id`) are remembered, to pin follow-ups to the same model.
    #[serde(default = "default_state_ttl")]
    pub state_ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_secs: default_cache_ttl(),
            max_conversations: default_max_conversations(),
            state_ttl_secs: default_state_ttl(),
        }
    }
}

impl CacheConfig {
    pub fn tracker(&self) -> Option<ConversationTracker> {
        self.enabled.then(|| {
            ConversationTracker::new(Duration::from_secs(self.ttl_secs), self.max_conversations)
                .with_state_ttl(Duration::from_secs(self.state_ttl_secs))
        })
    }
}

fn default_true() -> bool {
    true
}

fn default_cache_ttl() -> u64 {
    300
}

fn default_max_conversations() -> usize {
    100_000
}

fn default_state_ttl() -> u64 {
    7 * 24 * 3600
}

fn default_lambda() -> f64 {
    10.0
}

fn default_gpu_layers() -> i32 {
    -1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RouterConfig {
    /// Uniform random choice. `seed` makes the sequence reproducible.
    Random {
        #[serde(default)]
        seed: Option<u64>,
    },
    /// Activation probe: runs a local encoder model and a trained linear probe
    /// (opastin-encoder). Built by opastin-server, not by this crate.
    Probe {
        /// GGUF encoder model, e.g. `models/gguf/Qwen3.5-9B-Q8_0.gguf`.
        encoder_model: PathBuf,
        /// Probe weights exported by `probe/train_probe.py`.
        probe: PathBuf,
        /// Success probability traded per USD of expected cost: route to
        /// `argmax p − lambda × (cost + switch_cost)`.
        #[serde(default = "default_lambda")]
        lambda: f64,
        /// Encoder layers offloaded to the GPU (negative = all).
        #[serde(default = "default_gpu_layers")]
        n_gpu_layers: i32,
    },
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig::Random { seed: None }
    }
}

impl RouterConfig {
    /// Build a router implemented in this crate. Returns `None` for routers that need
    /// another crate (the probe router); build those yourself and use
    /// `Gateway::with_router`.
    pub fn build(&self) -> Option<Arc<dyn Router>> {
        match self {
            RouterConfig::Random { seed: Some(s) } => {
                Some(Arc::new(RandomRouter::with_seed(*s)))
            }
            RouterConfig::Random { seed: None } => Some(Arc::new(RandomRouter::new())),
            RouterConfig::Probe { .. } => None,
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
    /// Where to send Open Responses (`/responses`) requests for this model.
    #[serde(default)]
    pub responses: Option<Upstream>,
    /// Where to send Gemini Interactions (`/interactions`) requests for this model,
    /// e.g. `https://generativelanguage.googleapis.com/v1beta`.
    #[serde(default)]
    pub interactions: Option<Upstream>,
    /// Capability tier. The probe router predicts outcomes per tier and needs one.
    #[serde(default)]
    pub tier: Option<Tier>,
    /// Prices in USD per million tokens, for cost estimates and cache-aware routing.
    /// Defaults to the tier's typical prices.
    #[serde(default)]
    pub pricing: Option<Pricing>,
}

impl ModelConfig {
    pub fn upstream(&self, format: ApiFormat) -> Option<&Upstream> {
        match format {
            ApiFormat::OpenaiChat => self.openai.as_ref(),
            ApiFormat::AnthropicMessages => self.anthropic.as_ref(),
            ApiFormat::OpenResponses => self.responses.as_ref(),
            ApiFormat::GeminiInteractions => self.interactions.as_ref(),
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
            if ApiFormat::ALL.iter().all(|f| m.upstream(*f).is_none()) {
                return Err(ConfigError::Invalid(format!(
                    "model `{}` needs at least one upstream (`openai`, `anthropic`, `responses` or `interactions`)",
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
            .map(|m| {
                Candidate::new(&m.id)
                    .with_tier(m.tier)
                    .with_pricing(m.pricing)
            })
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
    fn parses_cache_pricing_and_probe_router() {
        let c = Config::from_toml_str(
            r#"
            [router]
            kind = "probe"
            encoder_model = "models/gguf/enc.gguf"
            probe = "models/probes/p.json"
            lambda = 20.0

            [cache]
            ttl_secs = 3600

            [[models]]
            id = "gpt-5"
            tier = "large"
            pricing = { input = 1.25, cached_input = 0.125, output = 10.0 }
            openai = { base_url = "u" }
            "#,
        )
        .unwrap();
        assert!(
            matches!(c.router, RouterConfig::Probe { lambda, n_gpu_layers: -1, .. } if lambda == 20.0)
        );
        assert!(
            c.router.build().is_none(),
            "probe router is built outside the core crate"
        );
        assert_eq!(c.cache.ttl_secs, 3600);
        assert!(c.cache.enabled);
        let cands = c.candidates(ApiFormat::OpenaiChat);
        assert_eq!(cands[0].pricing.unwrap().cached_input, 0.125);
        assert_eq!(cands[0].tier, Some(Tier::Large));

        // Defaults: cache on, 5-minute TTL, random router.
        let d = Config::from_toml_str("[[models]]\nid = \"x\"\nopenai = { base_url = \"u\" }\n")
            .unwrap();
        assert_eq!(d.cache, CacheConfig::default());
        assert!(d.router.build().is_some());
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
