//! `ProbeRouter`: routes on the encoder model's activations.
//!
//! For each request:
//! 1. Run the encoder's prefill (no generation) and take the pooled hidden state at the
//!    probe's layer.
//! 2. For each candidate, look up its tier's head: success probability `p̂` and expected
//!    output tokens.
//! 3. Price the request with the candidate's pricing (or its tier's default prices):
//!    input tokens at the input price, output tokens at the output price.
//!    - **Cache-aware:** if the request continues a conversation and the candidate served
//!      the previous turn, the cached part of the prompt is billed at the cached-input
//!      price. Every other candidate pays full price for it, so switching models mid-
//!      conversation must buy enough expected quality to cover the lost cache.
//! 4. Pick `argmax p̂ − λ·cost`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use opastin::{Candidate, Decision, RouteError, RouteRequest, Router, Tier};

use crate::probe::{Pooling, Prediction, Probe, ProbeError};
use crate::{Encoder, EncoderConfig, EncoderError, ThinkingMode};

#[derive(Debug, thiserror::Error)]
pub enum ProbeRouterError {
    #[error(transparent)]
    Encoder(#[from] EncoderError),
    #[error(transparent)]
    Probe(#[from] ProbeError),
    #[error("probe does not match encoder: {0}")]
    Mismatch(String),
}

/// One candidate's prediction, estimated cost and routing score.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub candidate: String,
    pub tier: Tier,
    pub prediction: Prediction,
    /// Expected USD cost of the request on this candidate.
    pub cost: f64,
    /// Prompt tokens billed at the cached price (non-zero only for the model that
    /// served the previous turn of the conversation).
    pub cached_tokens: u64,
    /// `p_success − λ·cost`.
    pub score: f64,
}

/// Score candidates and return them best first.
pub fn rank(mut scored: Vec<Scored>, lambda: f64) -> Vec<Scored> {
    for s in &mut scored {
        s.score = s.prediction.p_success - lambda * s.cost;
    }
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

/// Expected USD cost of a request on a candidate, and how many prompt tokens are billed
/// at the cached price.
pub fn estimate_cost(
    candidate: &Candidate,
    prompt_tokens: u64,
    out_tokens: f64,
    request: &RouteRequest,
) -> (f64, u64) {
    let Some(price) = candidate.effective_pricing() else {
        return (0.0, 0);
    };
    let cached = match &request.conversation {
        Some(c) if c.previous_model == candidate.id => c.cached_tokens.min(prompt_tokens),
        _ => 0,
    };
    let input = cached as f64 * price.cached_input + (prompt_tokens - cached) as f64 * price.input;
    ((input + out_tokens * price.output) / 1e6, cached)
}

/// Cheap to clone: all state is shared.
#[derive(Clone)]
pub struct ProbeRouter {
    inner: Arc<Inner>,
}

struct Inner {
    encoder: Mutex<Encoder>,
    probe: Probe,
    lambda: f64,
}

impl ProbeRouter {
    /// Load the encoder model configured to reproduce the probe's feature extraction.
    pub fn load(
        encoder_model: impl AsRef<Path>,
        probe_path: impl AsRef<Path>,
        lambda: f64,
        n_gpu_layers: i32,
    ) -> Result<Self, ProbeRouterError> {
        let probe = Probe::load(probe_path)?;
        let spec = probe.spec().clone();
        let config = EncoderConfig {
            n_gpu_layers,
            max_tokens: spec.max_tokens,
            layers: Some(vec![spec.layer]),
            thinking: if spec.thinking {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::NonThinking
            },
            verbose: false,
        };
        let encoder = Encoder::load(encoder_model, config)?;
        Self::new(encoder, probe, lambda)
    }

    pub fn new(encoder: Encoder, probe: Probe, lambda: f64) -> Result<Self, ProbeRouterError> {
        let spec = probe.spec();
        if encoder.n_embd() != spec.n_embd {
            return Err(ProbeRouterError::Mismatch(format!(
                "encoder hidden size {} but probe expects {}",
                encoder.n_embd(),
                spec.n_embd
            )));
        }
        if spec.layer > encoder.n_layer() {
            return Err(ProbeRouterError::Mismatch(format!(
                "probe layer {} but encoder has {} layers",
                spec.layer,
                encoder.n_layer()
            )));
        }
        let thinking = encoder.config().thinking == ThinkingMode::Thinking;
        if thinking != spec.thinking {
            return Err(ProbeRouterError::Mismatch(
                "thinking mode differs from training".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                encoder: Mutex::new(encoder),
                probe,
                lambda,
            }),
        })
    }

    pub fn probe(&self) -> &Probe {
        &self.inner.probe
    }

    pub fn lambda(&self) -> f64 {
        self.inner.lambda
    }

    /// Encode the request and return the probe's feature vector.
    fn features(&self, request: &RouteRequest) -> Result<Vec<f32>, EncoderError> {
        let spec = self.inner.probe.spec();
        let mut encoder = self.inner.encoder.lock().unwrap_or_else(|p| p.into_inner());
        let acts = encoder.encode_request(request)?;
        let k = acts
            .layers
            .iter()
            .position(|&l| l == spec.layer)
            .expect("encoder captures the probe layer");
        Ok(match spec.pooling {
            Pooling::Last => acts.last[k].clone(),
            Pooling::Mean => acts.mean[k].clone(),
        })
    }

    /// Predict and rank candidates for a request (blocking; runs the encoder).
    /// Candidates without a tier the probe knows are skipped.
    pub fn rank_blocking(
        &self,
        request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Vec<Scored>, RouteError> {
        let probe = &self.inner.probe;
        let usable: Vec<(&Candidate, Tier)> = candidates
            .iter()
            .filter_map(|c| c.tier.filter(|t| probe.has_tier(*t)).map(|t| (c, t)))
            .collect();
        if usable.is_empty() {
            return Err(if candidates.is_empty() {
                RouteError::NoCandidates
            } else {
                RouteError::Failed(format!(
                    "no candidate has a tier the probe predicts ({})",
                    probe
                        .tiers()
                        .map(Tier::as_str)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            });
        }
        let features = self
            .features(request)
            .map_err(|e| RouteError::Failed(e.to_string()))?;
        let prompt_tokens = request.estimated_tokens();
        let scored = usable
            .into_iter()
            .map(|(candidate, tier)| {
                let prediction = probe.predict(tier, &features).expect("tier checked");
                let (cost, cached_tokens) =
                    estimate_cost(candidate, prompt_tokens, prediction.out_tokens, request);
                Scored {
                    candidate: candidate.id.clone(),
                    tier,
                    prediction,
                    cost,
                    cached_tokens,
                    score: 0.0,
                }
            })
            .collect();
        Ok(rank(scored, self.inner.lambda))
    }
}

#[async_trait]
impl Router for ProbeRouter {
    fn name(&self) -> &str {
        "probe"
    }

    async fn route(
        &self,
        request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Decision, RouteError> {
        let this = self.clone();
        let request = request.clone();
        let candidates = candidates.to_vec();
        // The prefill takes 100+ ms of GPU work; keep it off the async executor.
        let ranked = tokio::task::spawn_blocking(move || this.rank_blocking(&request, &candidates))
            .await
            .map_err(|e| RouteError::Failed(format!("encoder task failed: {e}")))??;

        let reason = ranked
            .iter()
            .map(|s| {
                let cached = if s.cached_tokens > 0 {
                    format!(" ({} cached)", s.cached_tokens)
                } else {
                    String::new()
                };
                format!(
                    "{} ({}): p={:.2} out≈{:.0} tok cost=${:.4}{cached} score={:.3}",
                    s.candidate,
                    s.tier,
                    s.prediction.p_success,
                    s.prediction.out_tokens,
                    s.cost,
                    s.score
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        Ok(Decision {
            decision_id: uuid::Uuid::new_v4().to_string(),
            router: "probe".into(),
            model: ranked[0].candidate.clone(),
            reason: format!("lambda={}; {reason}", self.inner.lambda),
            conversation: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opastin::{ApiFormat, Conversation, Pricing};

    fn scored(name: &str, p: f64, cost: f64) -> Scored {
        Scored {
            candidate: name.into(),
            tier: Tier::Medium,
            prediction: Prediction {
                p_success: p,
                out_tokens: 0.0,
            },
            cost,
            cached_tokens: 0,
            score: 0.0,
        }
    }

    #[test]
    fn lambda_trades_success_against_cost() {
        let options = || vec![scored("strong", 0.90, 0.050), scored("cheap", 0.70, 0.001)];
        assert_eq!(rank(options(), 0.0)[0].candidate, "strong");
        // λ = 10/$: strong 0.90 − 0.5 = 0.40, cheap 0.70 − 0.01 = 0.69.
        let r = rank(options(), 10.0);
        assert_eq!(r[0].candidate, "cheap");
        assert!((r[0].score - 0.69).abs() < 1e-9);
        // Break-even near λ ≈ 4.08.
        assert_eq!(rank(options(), 4.0)[0].candidate, "strong");
        assert_eq!(rank(options(), 4.2)[0].candidate, "cheap");
    }

    fn request(conversation: Option<Conversation>) -> RouteRequest {
        let mut r = RouteRequest::from_json(
            ApiFormat::OpenaiChat,
            &serde_json::json!({"messages": [{"role": "user", "content": "x"}]}),
        )
        .unwrap();
        r.conversation = conversation;
        r
    }

    #[test]
    fn cost_uses_pricing_tier_defaults_and_cache() {
        let priced = Candidate::new("mine")
            .with_tier(Some(Tier::Large))
            .with_pricing(Some(Pricing {
                input: 1.0,
                cached_input: 0.1,
                output: 10.0,
            }));
        // 10k prompt tokens × $1/M + 1k output × $10/M = $0.02.
        let (c, cached) = estimate_cost(&priced, 10_000, 1_000.0, &request(None));
        assert!((c - 0.02).abs() < 1e-12);
        assert_eq!(cached, 0);

        // Tier defaults when unpriced: Small = $0.10 in / $0.50 out per M.
        let small = Candidate::new("s").with_tier(Some(Tier::Small));
        let (c, _) = estimate_cost(&small, 10_000, 1_000.0, &request(None));
        assert!((c - (10_000.0 * 0.10 + 1_000.0 * 0.50) / 1e6).abs() < 1e-12);

        // Continuing a conversation: 8k of the 10k prompt tokens are cached at "mine".
        let conv = Conversation {
            previous_model: "mine".into(),
            cached_tokens: 8_000,
        };
        let (c, cached) = estimate_cost(&priced, 10_000, 1_000.0, &request(Some(conv.clone())));
        assert_eq!(cached, 8_000);
        assert!((c - (8_000.0 * 0.1 + 2_000.0 * 1.0 + 1_000.0 * 10.0) / 1e6).abs() < 1e-12);
        // Another candidate pays full price for the whole prompt.
        let other = Candidate::new("other")
            .with_tier(Some(Tier::Large))
            .with_pricing(priced.pricing);
        let (c2, cached2) = estimate_cost(&other, 10_000, 1_000.0, &request(Some(conv)));
        assert_eq!(cached2, 0);
        assert!(c2 > c);
    }

    #[test]
    fn cache_keeps_a_conversation_on_its_model_unless_quality_justifies_a_switch() {
        // Two medium-tier models, equal predicted quality. 200k cached tokens at "a".
        let price = Some(Pricing {
            input: 2.0,
            cached_input: 0.1,
            output: 10.0,
        });
        let a = Candidate::new("a")
            .with_tier(Some(Tier::Medium))
            .with_pricing(price);
        let b = Candidate::new("b")
            .with_tier(Some(Tier::Medium))
            .with_pricing(price);
        let req = request(Some(Conversation {
            previous_model: "a".into(),
            cached_tokens: 200_000,
        }));
        let score = |c: &Candidate, p: f64| {
            let (cost, cached) = estimate_cost(c, 210_000, 500.0, &req);
            Scored {
                candidate: c.id.clone(),
                tier: Tier::Medium,
                prediction: Prediction {
                    p_success: p,
                    out_tokens: 500.0,
                },
                cost,
                cached_tokens: cached,
                score: 0.0,
            }
        };
        // Equal quality: stay on the cached model.
        assert_eq!(
            rank(vec![score(&a, 0.6), score(&b, 0.6)], 1.0)[0].candidate,
            "a"
        );
        // Switching costs 200k × $1.9/M = $0.38 more; at λ = 1 that needs > 0.38 more p.
        assert_eq!(
            rank(vec![score(&a, 0.5), score(&b, 0.85)], 1.0)[0].candidate,
            "a"
        );
        assert_eq!(
            rank(vec![score(&a, 0.5), score(&b, 0.95)], 1.0)[0].candidate,
            "b"
        );
    }
}
