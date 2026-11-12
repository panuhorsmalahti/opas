//! `ProbeRouter`: routes on the encoder model's activations.
//!
//! For each request:
//! 1. Run the encoder's prefill (no generation) and take the pooled hidden state at the
//!    probe's layer.
//! 2. For each candidate, look up its tier's head: success probability and expected
//!    output tokens.
//! 3. Turn tokens into expected cost and response time with the candidate's pricing and
//!    speed (or its tier's defaults).
//!    - **Cache-aware:** if the request continues a conversation and the candidate served
//!      the previous turn, the cached part of the prompt is billed at the cached-input
//!      price. Every other candidate pays full price for it, so switching models mid-
//!      conversation must be worth the lost cache.
//! 4. Score quality, speed and cost with the request's weights (default 60/20/20; see
//!    `opas::scoring`) and pick the best. All estimates are returned in the decision.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use opas::{Candidate, Decision, Estimate, RouteError, RouteRequest, Router, Tier, Weights};

use crate::probe::{Pooling, Probe, ProbeError};
use crate::{Encoder, EncoderConfig, EncoderError, ThinkingMode};

#[derive(Debug, thiserror::Error)]
pub enum ProbeRouterError {
    #[error(transparent)]
    Encoder(#[from] EncoderError),
    #[error(transparent)]
    Probe(#[from] ProbeError),
    #[error("probe does not match encoder: {0}")]
    Mismatch(String),
    #[error("invalid weights: {0}")]
    Weights(String),
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

/// Build an unscored estimate for a candidate from a tier prediction.
pub fn estimate(
    candidate: &Candidate,
    tier: Tier,
    p_success: f64,
    out_tokens: f64,
    prompt_tokens: u64,
    request: &RouteRequest,
) -> Estimate {
    let (cost, cached) = estimate_cost(candidate, prompt_tokens, out_tokens, request);
    let time = candidate
        .effective_speed()
        .map_or(0.0, |s| s.response_time_s(out_tokens));
    Estimate {
        model: candidate.id.clone(),
        tier: Some(tier),
        p_success,
        expected_cost_usd: cost,
        expected_response_time_s: time,
        expected_output_tokens: out_tokens,
        cached_prompt_tokens: cached,
        quality_score: 0.0,
        speed_score: 0.0,
        cost_score: 0.0,
        score: 0.0,
    }
}

/// Cheap to clone: all state is shared.
#[derive(Clone)]
pub struct ProbeRouter {
    inner: Arc<Inner>,
}

struct Inner {
    encoder: Mutex<Encoder>,
    probe: Probe,
    weights: Weights,
}

impl ProbeRouter {
    /// Load the encoder model configured to reproduce the probe's feature extraction.
    /// `weights` are the defaults for requests that don't set their own.
    pub fn load(
        encoder_model: impl AsRef<Path>,
        probe_path: impl AsRef<Path>,
        weights: Weights,
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
        Self::new(encoder, probe, weights)
    }

    pub fn new(encoder: Encoder, probe: Probe, weights: Weights) -> Result<Self, ProbeRouterError> {
        let weights = weights.validate().map_err(ProbeRouterError::Weights)?;
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
                weights,
            }),
        })
    }

    pub fn probe(&self) -> &Probe {
        &self.inner.probe
    }

    /// The default weights.
    pub fn weights(&self) -> Weights {
        self.inner.weights
    }

    /// Weights for a request: its own (`RouteRequest::weights`) or the defaults.
    pub fn weights_for(&self, request: &RouteRequest) -> Weights {
        request.weights.unwrap_or(self.inner.weights)
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

    /// Predict, score and rank candidates for a request, best first (blocking; runs the
    /// encoder). Candidates without a tier the probe knows are skipped.
    pub fn estimate_blocking(
        &self,
        request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Vec<Estimate>, RouteError> {
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
        let mut estimates: Vec<Estimate> = usable
            .into_iter()
            .map(|(candidate, tier)| {
                let p = probe.predict(tier, &features).expect("tier checked");
                estimate(
                    candidate,
                    tier,
                    p.p_success,
                    p.out_tokens,
                    prompt_tokens,
                    request,
                )
            })
            .collect();
        opas::scoring::score(&mut estimates, self.weights_for(request));
        Ok(estimates)
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
        let weights = self.weights_for(request);
        let this = self.clone();
        let request = request.clone();
        let candidates = candidates.to_vec();
        // The prefill takes 100+ ms of GPU work; keep it off the async executor.
        let estimates =
            tokio::task::spawn_blocking(move || this.estimate_blocking(&request, &candidates))
                .await
                .map_err(|e| RouteError::Failed(format!("encoder task failed: {e}")))??;

        let best = &estimates[0];
        let reason = format!(
            "best weighted score {:.3} ({weights}): p={:.2}, cost ${:.4}, {:.1}s",
            best.score, best.p_success, best.expected_cost_usd, best.expected_response_time_s
        );
        Ok(Decision {
            decision_id: uuid::Uuid::new_v4().to_string(),
            router: "probe".into(),
            model: best.model.clone(),
            reason,
            conversation: None,
            estimates,
            weights: Some(weights),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opas::{ApiFormat, Conversation, Pricing, Speed};

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
    fn response_time_uses_speed_or_tier_default() {
        let fast = Candidate::new("f")
            .with_tier(Some(Tier::Medium))
            .with_speed(Some(Speed {
                output_tokens_per_sec: 100.0,
                ttft_s: 0.5,
            }));
        let e = estimate(&fast, Tier::Medium, 0.7, 1_000.0, 100, &request(None));
        assert!(
            (e.expected_response_time_s - 10.5).abs() < 1e-9,
            "0.5 s + 1000 / 100 tok/s"
        );
        let default = Candidate::new("d").with_tier(Some(Tier::Large));
        let e = estimate(&default, Tier::Large, 0.7, 500.0, 100, &request(None));
        let s = Tier::Large.default_speed();
        assert!(
            (e.expected_response_time_s - (s.ttft_s + 500.0 / s.output_tokens_per_sec)).abs()
                < 1e-9
        );
    }

    #[test]
    fn warm_cache_keeps_a_conversation_on_its_model_at_equal_quality() {
        // Two medium-tier models with the same price, speed and predicted quality; a 200k
        // token prompt prefix is cached at "a".
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
        let mut e = vec![
            estimate(&a, Tier::Medium, 0.6, 500.0, 210_000, &req),
            estimate(&b, Tier::Medium, 0.6, 500.0, 210_000, &req),
        ];
        opas::scoring::score(&mut e, Weights::default());
        assert_eq!(e[0].model, "a");
        assert!(e[0].expected_cost_usd < e[1].expected_cost_usd);
    }
}
