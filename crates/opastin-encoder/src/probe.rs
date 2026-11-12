//! Linear probes over encoder activations: predict, per candidate model, the probability
//! of success and the expected cost of a request.
//!
//! Training (probe/train_probe.py) fits standardize → PCA → logistic / ridge regression.
//! Every step is linear, so each head is exported collapsed into one weight vector over
//! the raw hidden state plus a bias:
//!
//!   p_success = sigmoid(w · h + b)        (logistic regression)
//!   cost      = exp(w · h + b)            (ridge regression on log cost, USD)
//!
//! where `h` is the encoder's hidden state at `encoder.layer` with `encoder.pooling`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "opastin-probe/1";

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
        let dot: f64 = self.w.iter().zip(x).map(|(w, x)| f64::from(*w) * f64::from(*x)).sum();
        dot + f64::from(self.b)
    }
}

/// Predictors for one candidate model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Head {
    pub success: Linear,
    pub log_cost: Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeFile {
    pub format: String,
    pub encoder: EncoderSpec,
    /// One head per model the probe was trained to predict, keyed by model name.
    pub heads: BTreeMap<String, Head>,
    /// Training provenance and evaluation results (free-form).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A prediction for one candidate model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Prediction {
    /// Probability the model answers the request successfully.
    pub p_success: f64,
    /// Expected cost of the model's answer in USD.
    pub cost: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("reading probe {path}: {source}")]
    Io { path: String, source: std::io::Error },
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
        for (name, head) in &file.heads {
            for (kind, lin) in [("success", &head.success), ("log_cost", &head.log_cost)] {
                if lin.w.len() != n {
                    return Err(ProbeError::Invalid(format!(
                        "head `{name}` {kind}: {} weights, expected n_embd = {n}",
                        lin.w.len()
                    )));
                }
                if !lin.b.is_finite() || lin.w.iter().any(|w| !w.is_finite()) {
                    return Err(ProbeError::Invalid(format!("head `{name}` {kind}: non-finite weights")));
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

    pub fn has_head(&self, name: &str) -> bool {
        self.file.heads.contains_key(name)
    }

    pub fn head_names(&self) -> impl Iterator<Item = &str> {
        self.file.heads.keys().map(String::as_str)
    }

    /// Predict for one head from the pooled hidden state at `spec().layer`.
    pub fn predict(&self, head: &str, features: &[f32]) -> Option<Prediction> {
        let h = self.file.heads.get(head)?;
        if features.len() != self.file.encoder.n_embd {
            return None;
        }
        let logit = h.success.apply(features);
        Some(Prediction {
            p_success: 1.0 / (1.0 + (-logit).exp()),
            cost: h.log_cost.apply(features).clamp(-30.0, 10.0).exp(),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A 4-dimensional probe with two heads, for tests.
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
                // "strong": succeeds often, costs ~$0.05
                ("strong".to_string(), Head { success: lin([1.0, 0.0, 0.0, 0.0], 2.0), log_cost: lin([0.0; 4], (0.05f32).ln()) }),
                // "cheap": success depends on feature 0, costs ~$0.001
                ("cheap".to_string(), Head { success: lin([2.0, 0.0, 0.0, 0.0], 0.0), log_cost: lin([0.0; 4], (0.001f32).ln()) }),
            ]),
            metadata: serde_json::Value::Null,
        }
    }

    #[test]
    fn predicts_sigmoid_and_exp() {
        let probe = Probe::from_file(tiny_probe()).unwrap();
        let p = probe.predict("cheap", &[0.0, 5.0, 5.0, 5.0]).unwrap();
        assert!((p.p_success - 0.5).abs() < 1e-9);
        assert!((p.cost - 0.001).abs() < 1e-9);
        let p = probe.predict("strong", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((p.p_success - 1.0 / (1.0 + (-3.0f64).exp())).abs() < 1e-9);
        assert!(probe.predict("missing", &[0.0; 4]).is_none());
        assert!(probe.predict("cheap", &[0.0; 3]).is_none(), "wrong dimension");
    }

    #[test]
    fn rejects_malformed_probes() {
        let mut f = tiny_probe();
        f.format = "other".into();
        assert!(Probe::from_file(f).is_err());
        let mut f = tiny_probe();
        f.heads.get_mut("cheap").unwrap().success.w.pop();
        assert!(Probe::from_file(f).is_err());
        let mut f = tiny_probe();
        f.heads.get_mut("cheap").unwrap().log_cost.b = f32::NAN;
        assert!(Probe::from_file(f).is_err());
    }

    #[test]
    fn round_trips_json() {
        let json = serde_json::to_string(&tiny_probe()).unwrap();
        let probe = Probe::from_file(serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(probe.spec().layer, 2);
        assert_eq!(probe.head_names().collect::<Vec<_>>(), ["cheap", "strong"]);
    }
}
