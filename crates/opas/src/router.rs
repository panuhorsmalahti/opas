//! The [`Router`] trait and the types routers work with. The activation-probe router
//! lives in `opas-encoder`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::conversation::{Conversation, Pricing, Speed};
use crate::request::RouteRequest;
use crate::scoring::{Estimate, Weights};
use crate::tier::Tier;

/// A model the router may choose.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// Model id as exposed by the router (matches a configured model id).
    pub id: String,
    /// Capability tier, if configured; used by routers that predict outcomes per tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    /// Prices, if configured; used for cost estimates and cache-aware routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Pricing>,
    /// Serving speed, if configured; used for response-time estimates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<Speed>,
    /// models.dev catalog key (`provider/model`) to price this candidate from, when
    /// `pricing` is not given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models_dev: Option<String>,
}

impl Candidate {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tier: None,
            pricing: None,
            speed: None,
            models_dev: None,
        }
    }

    pub fn with_tier(mut self, tier: Option<Tier>) -> Self {
        self.tier = tier;
        self
    }

    pub fn with_speed(mut self, speed: Option<Speed>) -> Self {
        self.speed = speed;
        self
    }

    /// Configured speed, or the tier's default speed.
    pub fn effective_speed(&self) -> Option<Speed> {
        self.speed.or_else(|| self.tier.map(Tier::default_speed))
    }

    /// Configured pricing, or the tier's default pricing.
    pub fn effective_pricing(&self) -> Option<Pricing> {
        self.pricing
            .or_else(|| self.tier.map(Tier::default_pricing))
    }

    pub fn with_pricing(mut self, pricing: Option<Pricing>) -> Self {
        self.pricing = pricing;
        self
    }
}

impl From<&str> for Candidate {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

/// The outcome of one routing decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// Unique id for this decision, for logging and later feedback.
    pub decision_id: String,
    /// Name of the router that decided.
    pub router: String,
    /// The chosen candidate id.
    pub model: String,
    /// Human-readable explanation.
    pub reason: String,
    /// The conversation this request continued, if any (set by the gateway).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,
    /// Predicted outcome and scores per candidate, best first, from routers that predict
    /// (the probe router). Callers can apply their own trade-off to these.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub estimates: Vec<Estimate>,
    /// The quality / speed / cost weights the router used, if it uses them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Weights>,
    /// How long the decision took (set by routers that measure it, and by the gateway).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<RouterTiming>,
}

/// Where the time of a routing decision went, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RouterTiming {
    /// The whole decision, as the gateway measured it.
    pub total_ms: f64,
    /// Waiting for the encoder, which runs one request at a time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    /// The encoder's pass over the prompt: computing the activations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encode_ms: Option<f64>,
    /// The predictor heads and the scoring of every candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predict_ms: Option<f64>,
    /// Prompt tokens the encoder read (after truncation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_tokens: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("no candidate models to route between")]
    NoCandidates,
    #[error("router failed: {0}")]
    Failed(String),
}

/// Picks one candidate model for a request.
///
/// Async so that future routers can call out to classifiers or other models.
#[async_trait]
pub trait Router: Send + Sync {
    /// Short name, reported in [`Decision::router`].
    fn name(&self) -> &str;

    /// Choose one of `candidates` for `request`.
    async fn route(
        &self,
        request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Decision, RouteError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_pricing_and_speed_fall_back_to_tier_defaults() {
        let c = Candidate::new("m").with_tier(Some(Tier::Medium));
        assert_eq!(c.effective_pricing(), Some(Tier::Medium.default_pricing()));
        assert_eq!(c.effective_speed(), Some(Tier::Medium.default_speed()));
        assert_eq!(Candidate::new("x").effective_pricing(), None);
    }
}
