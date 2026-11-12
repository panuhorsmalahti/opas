//! Linear probes over encoder activations: predict, per capability tier, the probability
//! that a model of that tier answers a request successfully and how many output tokens
//! it will produce. Tiers keep the probe independent of specific model names; the
//! consumer maps its models to tiers and supplies prices.
//!
//! Training (training/train_probe.py) fits standardize → PCA → logistic / ridge regression.
//! Every step is linear, so each head is exported collapsed into one weight vector over
//! the raw hidden state plus a bias:
//!
//!   p_success  = sigmoid(w · h + b)       (logistic regression)
//!   out_tokens = exp(w · h + b) − 1       (ridge regression on log(1 + tokens))
//!
//! where `h` is the encoder's hidden state at `encoder.layer` with `encoder.pooling`.

use std::collections::BTreeMap;
use std::path::Path;

use opastin::Tier;
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "opastin-probe/2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    Last,
    Mean,
}

/// How the features were extracted. The router must reproduce this exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderSpec {
    /// Hidden-state index (0 = embeddings, i = after decoder layer i).
    pub layer: usize,
    pub pooling: Pooling,
    /// Whether the prompt ends with an open `<think>` block.
    pub thinking: bool,
    /// Prompt truncation used when extracting training features.
    pub max_tokens: usize,
    /// Hidden size; every weight vector has this length.
    pub n_embd: usize,
    /// File name of the encoder model the probe was trained on (informational).
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub w: Vec<f32>,
    pub b: f32,
}

impl Linear {
    fn apply(&self, x: &[f32]) -> f64 {
        let dot: f64 = self
            .w
            .iter()
            .zip(x)
            .map(|(w, x)| f64::from(*w) * f64::from(*x))
            .sum();
        dot + f64::from(self.b)
    }
}

/// Predictors for one tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Head {
    pub success: Linear,
    pub log_out_tokens: Linear,
    /// True when the head was not fit on data but extrapolated from other tiers.
    #[serde(default)]
    pub extrapolated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeFile {
    pub format: String,
    pub encoder: EncoderSpec,
    pub heads: BTreeMap<Tier, Head>,
    /// Training provenance and evaluation results (free-form).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A prediction for one tier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Prediction {
    /// Probability that a model of the tier answers successfully.
    pub p_success: f64,
    /// Expected output tokens.
    pub out_tokens: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("reading probe {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parsing probe: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("invalid probe: {0}")]
    Invalid(String),
}

pub struct Probe {
    file: ProbeFile,
}

impl Probe {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ProbeError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ProbeError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_file(serde_json::from_str(&text)?)
    }

    pub fn from_file(file: ProbeFile) -> Result<Self, ProbeError> {
        if file.format != FORMAT {
            return Err(ProbeError::Invalid(format!(
                "format `{}`, expected `{FORMAT}`",
                file.format
            )));
        }
        if file.heads.is_empty() {
            return Err(ProbeError::Invalid("no heads".into()));
        }
        let n = file.encoder.n_embd;
        for (tier, head) in &file.heads {
            for (kind, lin) in [
                ("success", &head.success),
                ("log_out_tokens", &head.log_out_tokens),
            ] {
                if lin.w.len() != n {
                    return Err(ProbeError::Invalid(format!(
                        "{tier} {kind}: {} weights, expected n_embd = {n}",
                        lin.w.len()
                    )));
                }
                if !lin.b.is_finite() || lin.w.iter().any(|w| !w.is_finite()) {
                    return Err(ProbeError::Invalid(format!(
                        "{tier} {kind}: non-finite weights"
                    )));
                }
            }
        }
        Ok(Self { file })
    }

    pub fn spec(&self) -> &EncoderSpec {
        &self.file.encoder
    }

    pub fn file(&self) -> &ProbeFile {
        &self.file
    }

    pub fn tiers(&self) -> impl Iterator<Item = Tier> + '_ {
        self.file.heads.keys().copied()
    }

    pub fn has_tier(&self, tier: Tier) -> bool {
        self.file.heads.contains_key(&tier)
    }

    /// Predict for one tier from the pooled hidden state at `spec().layer`.
    pub fn predict(&self, tier: Tier, features: &[f32]) -> Option<Prediction> {
        let h = self.file.heads.get(&tier)?;
        if features.len() != self.file.encoder.n_embd {
            return None;
        }
        let logit = h.success.apply(features);
        Some(Prediction {
            p_success: 1.0 / (1.0 + (-logit).exp()),
            out_tokens: (h.log_out_tokens.apply(features).clamp(-30.0, 15.0).exp() - 1.0).max(0.0),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A 4-dimensional probe with small and large heads, for tests.
    pub(crate) fn tiny_probe() -> ProbeFile {
        let lin = |w: [f32; 4], b: f32| Linear { w: w.to_vec(), b };
        ProbeFile {
            format: FORMAT.into(),
            encoder: EncoderSpec {
                layer: 2,
                pooling: Pooling::Last,
                thinking: true,
                max_tokens: 512,
                n_embd: 4,
                model: None,
            },
            heads: BTreeMap::from([
                // small: success depends on feature 0, ~200 output tokens
                (
                    Tier::Small,
                    Head {
                        success: lin([2.0, 0.0, 0.0, 0.0], 0.0),
                        log_out_tokens: lin([0.0; 4], 201f32.ln()),
                        extrapolated: false,
                    },
                ),
                // large: succeeds often, ~1000 output tokens
                (
                    Tier::Large,
                    Head {
                        success: lin([1.0, 0.0, 0.0, 0.0], 2.0),
                        log_out_tokens: lin([0.0; 4], 1001f32.ln()),
                        extrapolated: false,
                    },
                ),
            ]),
            metadata: serde_json::Value::Null,
        }
    }

    #[test]
    fn predicts_success_and_tokens() {
        let probe = Probe::from_file(tiny_probe()).unwrap();
        let p = probe.predict(Tier::Small, &[0.0, 5.0, 5.0, 5.0]).unwrap();
        assert!((p.p_success - 0.5).abs() < 1e-9);
        assert!((p.out_tokens - 200.0).abs() < 1e-3);
        let p = probe.predict(Tier::Large, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((p.p_success - 1.0 / (1.0 + (-3.0f64).exp())).abs() < 1e-9);
        assert!(probe.predict(Tier::Medium, &[0.0; 4]).is_none());
        assert!(
            probe.predict(Tier::Small, &[0.0; 3]).is_none(),
            "wrong dimension"
        );
    }

    #[test]
    fn rejects_malformed_probes() {
        let mut f = tiny_probe();
        f.format = "opastin-probe/1".into();
        assert!(Probe::from_file(f).is_err());
        let mut f = tiny_probe();
        f.heads.get_mut(&Tier::Small).unwrap().success.w.pop();
        assert!(Probe::from_file(f).is_err());
        let mut f = tiny_probe();
        f.heads.get_mut(&Tier::Small).unwrap().log_out_tokens.b = f32::NAN;
        assert!(Probe::from_file(f).is_err());
    }

    #[test]
    fn round_trips_json_with_tier_keys() {
        let json = serde_json::to_string(&tiny_probe()).unwrap();
        assert!(json.contains("\"small\"") && json.contains("\"large\""));
        let probe = Probe::from_file(serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(probe.spec().layer, 2);
        assert_eq!(
            probe.tiers().collect::<Vec<_>>(),
            [Tier::Small, Tier::Large]
        );
    }
}
