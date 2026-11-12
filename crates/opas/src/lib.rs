//! opas: model routing for LLM requests.
//!
//! This crate is the SDK: request parsing for four LLM APIs, the [`Router`] trait,
//! capability tiers, quality / speed / cost scoring, cache-aware conversation tracking,
//! prices (config, models.dev, per request) and configuration.
//!
//! The router that predicts with a model's activations is `opas_encoder::ProbeRouter`;
//! any type implementing [`Router`] works the same way:
//!
//! ```
//! use opas::{ApiFormat, Candidate, Decision, RouteError, RouteRequest, Router, Tier};
//! use serde_json::json;
//!
//! /// Long prompts to the large tier, everything else to the small one.
//! struct ByLength;
//!
//! #[async_trait::async_trait]
//! impl Router for ByLength {
//!     fn name(&self) -> &str { "by-length" }
//!     async fn route(&self, req: &RouteRequest, cands: &[Candidate]) -> Result<Decision, RouteError> {
//!         let want = if req.estimated_tokens() > 2_000 { Tier::Large } else { Tier::Small };
//!         let pick = cands.iter().find(|c| c.tier == Some(want)).ok_or(RouteError::NoCandidates)?;
//!         Ok(Decision {
//!             decision_id: "example".into(),
//!             router: self.name().into(),
//!             model: pick.id.clone(),
//!             reason: format!("≈{} prompt tokens", req.estimated_tokens()),
//!             conversation: None,
//!             estimates: vec![],
//!             weights: None,
//!         })
//!     }
//! }
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let body = json!({"model": "auto", "messages": [{"role": "user", "content": "hi"}]});
//! let request = RouteRequest::from_json(ApiFormat::OpenaiChat, &body).unwrap();
//! let candidates = [
//!     Candidate::new("claude-haiku-5-5").with_tier(Some(Tier::Small)),
//!     Candidate::new("claude-opus-5-5").with_tier(Some(Tier::Large)),
//! ];
//! let decision = ByLength.route(&request, &candidates).await.unwrap();
//! assert_eq!(decision.model, "claude-haiku-5-5");
//! # });
//! ```
//!
//! **Route and forward** (feature `gateway`): a [`Gateway`](gateway::Gateway) holds a
//! [`Config`] of models and upstreams, picks a model with its router, and sends the
//! request on. This is what `opas-server` runs behind its HTTP endpoints. Without a
//! router it is a plain proxy: requests must name a configured model.

pub mod api;
pub mod catalog;
pub mod config;
pub mod conversation;
#[cfg(feature = "gateway")]
pub mod gateway;
pub mod request;
pub mod router;
pub mod scoring;
pub mod tier;

pub use api::ApiFormat;
pub use catalog::{PriceCatalog, PriceSource};
pub use config::{CacheConfig, Config, ConfigError, ModelConfig, RouterConfig, Upstream};
pub use conversation::{Conversation, ConversationTracker, LongContextPricing, Pricing, Speed};
pub use request::{Message, RequestError, RouteRequest};
pub use router::{Candidate, Decision, RouteError, Router};
pub use scoring::{Estimate, Preference, Weights};
pub use tier::Tier;
