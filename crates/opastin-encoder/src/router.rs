//! `ProbeRouter`: routes on the encoder model's activations.
//!
//! For each request: run the encoder's prefill (no generation), take the pooled hidden
//! state at the probe's layer, predict each candidate's success probability `p̂` and
//! expected cost `ĉ` (USD), and pick `argmax p̂ − λ·(ĉ + switch)`.
//!
//! Cache-aware: when the request continues a conversation, `switch` is the extra input
//! cost a candidate pays because the provider's prompt cache is warm only at the model
//! that served the previous turn (see `opastin::conversation::switch_cost`). It is zero
//! for that model and for candidates without configured pricing.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use opastin::{Candidate, Decision, RouteError, RouteRequest, Router};

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

/// One candidate's prediction and routing score.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub candidate: String,
    pub head: String,
    pub prediction: Prediction,
    /// Extra USD from losing the prompt cache by switching to this candidate.
    pub switch_cost: f64,
    /// `p_success − λ·(cost + switch_cost)`.
    pub score: f64,
}

/// Score candidates and return them best first.
pub fn rank(mut scored: Vec<Scored>, lambda: f64) -> Vec<Scored> {
    for s in &mut scored {
        s.score = s.prediction.p_success - lambda * (s.prediction.cost + s.switch_cost);
    }
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

/// Cheap to clone: all state is shared.
#[derive(Clone)]
pub struct ProbeRouter {
    inner: Arc<Inner>,
}

struct Inner {
    encoder: Mutex<Encoder>,
    probe: Probe,
    /// Candidate id → probe head name. Candidates not listed use their own id as head.
    heads: HashMap<String, String>,
    lambda: f64,
}

impl ProbeRouter {
    /// Load the encoder model configured to reproduce the probe's feature extraction.
    pub fn load(
        encoder_model: impl AsRef<Path>,
        probe_path: impl AsRef<Path>,
        heads: HashMap<String, String>,
        lambda: f64,
        n_gpu_layers: i32,
    ) -> Result<Self, ProbeRouterError> {
        let probe = Probe::load(probe_path)?;
        let spec = probe.spec().clone();
        let config = EncoderConfig {
            n_gpu_layers,
            max_tokens: spec.max_tokens,
            layers: Some(vec![spec.layer]),
            thinking: if spec.thinking { ThinkingMode::Thinking } else { ThinkingMode::NonThinking },
            verbose: false,
        };
        let encoder = Encoder::load(encoder_model, config)?;
        Self::new(encoder, probe, heads, lambda)
    }

    pub fn new(
        encoder: Encoder,
        probe: Probe,
        heads: HashMap<String, String>,
        lambda: f64,
    ) -> Result<Self, ProbeRouterError> {
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
            return Err(ProbeRouterError::Mismatch("thinking mode differs from training".into()));
        }
        for head in heads.values() {
            if !probe.has_head(head) {
                return Err(ProbeRouterError::Mismatch(format!("probe has no head `{head}`")));
            }
        }
        Ok(Self {
            inner: Arc::new(Inner { encoder: Mutex::new(encoder), probe, heads, lambda }),
        })
    }

    pub fn probe(&self) -> &Probe {
        &self.inner.probe
    }

    pub fn lambda(&self) -> f64 {
        self.inner.lambda
    }

    fn head_for(&self, candidate: &str) -> Option<String> {
        match self.inner.heads.get(candidate) {
            Some(h) => Some(h.clone()),
            None => self.inner.probe.has_head(candidate).then(|| candidate.to_owned()),
        }
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
    pub fn rank_blocking(
        &self,
        request: &RouteRequest,
        candidates: &[Candidate],
    ) -> Result<Vec<Scored>, RouteError> {
        let usable: Vec<(&Candidate, String)> = candidates
            .iter()
            .filter_map(|c| self.head_for(&c.id).map(|h| (c, h)))
            .collect();
        if usable.is_empty() {
            return Err(if candidates.is_empty() {
                RouteError::NoCandidates
            } else {
                RouteError::Failed("none of the candidates has a probe head".into())
            });
        }
        let features = self.features(request).map_err(|e| RouteError::Failed(e.to_string()))?;
        let scored = usable
            .into_iter()
            .map(|(candidate, head)| {
                let prediction = self.inner.probe.predict(&head, &features).expect("validated head");
                Scored {
                    candidate: candidate.id.clone(),
                    head,
                    prediction,
                    switch_cost: opastin::conversation::switch_cost(candidate, request.conversation.as_ref()),
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

        let best = &ranked[0];
        let reason = ranked
            .iter()
            .map(|s| {
                format!(
                    "{}: p={:.2} cost=${:.4}{} score={:.3}",
                    s.candidate,
                    s.prediction.p_success,
                    s.prediction.cost,
                    if s.switch_cost > 0.0 { format!(" +switch ${:.4}", s.switch_cost) } else { String::new() },
                    s.score
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        Ok(Decision {
            decision_id: uuid::Uuid::new_v4().to_string(),
            router: "probe".into(),
            model: best.candidate.clone(),
            reason: format!("lambda={}; {reason}", self.inner.lambda),
            conversation: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(name: &str, p: f64, cost: f64) -> Scored {
        Scored {
            candidate: name.into(),
            head: name.into(),
            prediction: Prediction { p_success: p, cost },
            switch_cost: 0.0,
            score: 0.0,
        }
    }

    #[test]
    fn switch_cost_keeps_a_conversation_on_its_cached_model() {
        // Mid-conversation: cheap would win on its own (see below), but 200k cached
        // tokens at the strong model make switching cost 200k × $2.7/M = $0.54 extra.
        let mut cheap = scored("cheap", 0.70, 0.001);
        cheap.switch_cost = 0.54;
        let r = rank(vec![scored("strong", 0.90, 0.050), cheap], 10.0);
        assert_eq!(r[0].candidate, "strong");
        // Without the cache penalty the cheap model wins at the same λ.
        assert_eq!(rank(vec![scored("strong", 0.90, 0.050), scored("cheap", 0.70, 0.001)], 10.0)[0].candidate, "cheap");
    }

    #[test]
    fn lambda_trades_success_against_cost() {
        let options = || vec![scored("strong", 0.90, 0.050), scored("cheap", 0.70, 0.001)];
        // λ = 0: quality only.
        assert_eq!(rank(options(), 0.0)[0].candidate, "strong");
        // λ = 10/$: strong scores 0.90 − 0.5 = 0.40, cheap 0.70 − 0.01 = 0.69.
        let r = rank(options(), 10.0);
        assert_eq!(r[0].candidate, "cheap");
        assert!((r[0].score - 0.69).abs() < 1e-9);
        // Break-even: 0.2 success difference for $0.049 → λ ≈ 4.08.
        assert_eq!(rank(options(), 4.0)[0].candidate, "strong");
        assert_eq!(rank(options(), 4.2)[0].candidate, "cheap");
    }
}
