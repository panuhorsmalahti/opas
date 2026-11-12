//! Weighted quality / speed / cost scoring of candidate estimates.
//!
//! Each candidate gets three scores in 0..1, then a weighted average:
//!
//! - **quality** = the predicted probability of success relative to the most likely
//!   candidate (p / max p): 1 for the best, proportionally less for the others. On hard
//!   prompts the tiers differ a lot and this term dominates; on easy prompts everyone is
//!   near 1 and cost and speed decide. (Absolute p made `balanced` send 85% of
//!   LLMRouterBench prompts to the smallest tier; see docs/probe-results.md.)
//! - **cost** = 1 for the cheapest candidate, falling on a log scale to 0 at
//!   [`COST_RANGE`]× the cheapest;
//! - **speed** = 1 for the fastest predicted response time, falling on a log scale to 0
//!   at [`TIME_RANGE`]× the fastest.
//!
//! The ranges are fixed (not stretched to the spread among candidates), so small
//! differences in price or time cannot swing a decision on their own. The default
//! weights, quality 60 / speed 20 / cost 20, follow OpenRouter's Router Index.

use serde::{Deserialize, Serialize};

use crate::tier::Tier;

/// A candidate this many times more expensive than the cheapest scores 0 on cost.
pub const COST_RANGE: f64 = 100.0;
/// A candidate this many times slower than the fastest scores 0 on speed.
pub const TIME_RANGE: f64 = 10.0;

/// Relative importance of quality, speed and cost. Only the ratios matter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Weights {
    #[serde(default)]
    pub quality: f64,
    #[serde(default)]
    pub speed: f64,
    #[serde(default)]
    pub cost: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Preference::Balanced.weights()
    }
}

impl Weights {
    pub fn new(quality: f64, speed: f64, cost: f64) -> Self {
        Self {
            quality,
            speed,
            cost,
        }
    }

    /// Weights must be finite, non-negative, and not all zero.
    pub fn validate(self) -> Result<Self, String> {
        let all = [self.quality, self.speed, self.cost];
        if all.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(format!("weights must be finite and >= 0, got {self}"));
        }
        if all.iter().sum::<f64>() <= 0.0 {
            return Err("at least one weight must be > 0".into());
        }
        Ok(self)
    }

    /// The weights scaled to sum to 1.
    pub fn normalized(self) -> Self {
        let sum = (self.quality + self.speed + self.cost).max(f64::MIN_POSITIVE);
        Self::new(self.quality / sum, self.speed / sum, self.cost / sum)
    }
}

impl std::fmt::Display for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quality={},speed={},cost={}",
            self.quality, self.speed, self.cost
        )
    }
}

impl std::str::FromStr for Weights {
    type Err = String;

    /// Parses `quality=60,speed=20,cost=20`; omitted weights are 0.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut w = Weights::new(0.0, 0.0, 0.0);
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("expected key=value, got `{part}`"))?;
            let value: f64 = value
                .trim()
                .parse()
                .map_err(|_| format!("`{part}`: not a number"))?;
            match key.trim() {
                "quality" => w.quality = value,
                "speed" => w.speed = value,
                "cost" => w.cost = value,
                other => {
                    return Err(format!(
                        "unknown weight `{other}` (expected quality, speed, cost)"
                    ))
                }
            }
        }
        w.validate()
    }
}

/// Named weight presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preference {
    /// quality 100: the most likely to succeed, whatever it costs.
    Quality,
    /// quality 60 / speed 20 / cost 20 (the default).
    Balanced,
    /// quality 20 / speed 10 / cost 70.
    Cost,
    /// quality 20 / speed 70 / cost 10.
    Speed,
}

impl Preference {
    pub fn weights(self) -> Weights {
        match self {
            Preference::Quality => Weights::new(100.0, 0.0, 0.0),
            Preference::Balanced => Weights::new(60.0, 20.0, 20.0),
            Preference::Cost => Weights::new(20.0, 10.0, 70.0),
            Preference::Speed => Weights::new(20.0, 70.0, 10.0),
        }
    }
}

impl std::str::FromStr for Preference {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "quality" => Ok(Preference::Quality),
            "balanced" => Ok(Preference::Balanced),
            "cost" => Ok(Preference::Cost),
            "speed" => Ok(Preference::Speed),
            other => Err(format!(
                "unknown preference `{other}` (expected quality, balanced, cost or speed)"
            )),
        }
    }
}

/// A router's prediction for one candidate model, with its scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    /// Candidate model id.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    /// Predicted probability that the model answers successfully (0..1).
    pub p_success: f64,
    /// Expected cost of the request on this model, in USD.
    pub expected_cost_usd: f64,
    /// Expected time until the complete response, in seconds: time to first token +
    /// output tokens / output speed. Per request (one turn), not per multi-call task.
    pub expected_response_time_s: f64,
    /// Expected output tokens.
    pub expected_output_tokens: f64,
    /// Prompt tokens billed at the cached price (warm prompt cache of the model that
    /// served the previous turn).
    #[serde(default)]
    pub cached_prompt_tokens: u64,
    /// Quality score: p_success relative to the best candidate (1 for the best).
    #[serde(default)]
    pub quality_score: f64,
    /// Speed score: 1 for the fastest candidate, 0 at TIME_RANGE× slower.
    #[serde(default)]
    pub speed_score: f64,
    /// Cost score: 1 for the cheapest candidate, 0 at COST_RANGE× pricier.
    #[serde(default)]
    pub cost_score: f64,
    /// Weighted average of the three scores; the router picks the highest.
    #[serde(default)]
    pub score: f64,
}

/// 1 at `best`, falling on a log scale to 0 at `range` × `best`.
fn log_score(value: f64, best: f64, range: f64) -> f64 {
    if value <= best || best <= 0.0 {
        return 1.0;
    }
    (1.0 - (value / best).ln() / range.ln()).clamp(0.0, 1.0)
}

/// Fill in each estimate's scores and sort best first.
pub fn score(estimates: &mut [Estimate], weights: Weights) {
    let w = weights.normalized();
    let positive_min = |f: fn(&Estimate) -> f64| {
        estimates
            .iter()
            .map(f)
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min)
    };
    let min_cost = positive_min(|e| e.expected_cost_usd);
    let min_time = positive_min(|e| e.expected_response_time_s);
    let max_p = estimates
        .iter()
        .map(|e| e.p_success.clamp(0.0, 1.0))
        .fold(0.0, f64::max);
    for e in estimates.iter_mut() {
        e.quality_score = if max_p > 0.0 {
            e.p_success.clamp(0.0, 1.0) / max_p
        } else {
            1.0
        };
        e.cost_score = log_score(e.expected_cost_usd, min_cost, COST_RANGE);
        e.speed_score = log_score(e.expected_response_time_s, min_time, TIME_RANGE);
        e.score = w.quality * e.quality_score + w.speed * e.speed_score + w.cost * e.cost_score;
    }
    estimates.sort_by(|a, b| b.score.total_cmp(&a.score));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn est(model: &str, p: f64, cost: f64, time: f64) -> Estimate {
        Estimate {
            model: model.into(),
            tier: None,
            p_success: p,
            expected_cost_usd: cost,
            expected_response_time_s: time,
            expected_output_tokens: 0.0,
            cached_prompt_tokens: 0,
            quality_score: 0.0,
            speed_score: 0.0,
            cost_score: 0.0,
            score: 0.0,
        }
    }

    fn pick(weights: Weights) -> String {
        let mut e = vec![
            est("small", 0.45, 0.0005, 2.0),
            est("medium", 0.80, 0.005, 6.0),
            est("large", 0.88, 0.02, 12.0),
        ];
        score(&mut e, weights);
        e[0].model.clone()
    }

    #[test]
    fn presets_pick_what_they_promise() {
        assert_eq!(pick(Preference::Quality.weights()), "large");
        assert_eq!(pick(Preference::Cost.weights()), "small");
        assert_eq!(pick(Preference::Speed.weights()), "small");
        // Balanced (60/20/20), quality relative to the best (0.88): small 0.6·0.51 + 0.4 =
        // 0.71; medium 0.6·0.91 + 0.10 + 0.10 = 0.75; large 0.6 + 0.04 + 0.04 = 0.68.
        assert_eq!(pick(Preference::Balanced.weights()), "medium");
    }

    #[test]
    fn scores_are_on_fixed_log_scales() {
        let mut e = vec![
            est("a", 0.5, 0.01, 1.0),
            est("b", 0.5, 0.10, 3.162_277_66),
            est("c", 0.5, 2.0, 50.0),
        ];
        score(&mut e, Weights::new(1.0, 1.0, 1.0));
        let by = |m: &str| e.iter().find(|x| x.model == m).unwrap().clone();
        assert!((by("a").cost_score - 1.0).abs() < 1e-9);
        assert!(
            (by("b").cost_score - 0.5).abs() < 1e-9,
            "10x of a 100x range = half"
        );
        assert_eq!(by("c").cost_score, 0.0, "beyond the range clamps to 0");
        assert!(
            (by("b").speed_score - 0.5).abs() < 1e-6,
            "sqrt(10)x of a 10x range = half"
        );
        assert!(
            (by("a").quality_score - 1.0).abs() < 1e-12,
            "equal p: all are the best"
        );
        // Small differences do not swing the decision: a 5% cheaper model scores ~0.01 higher.
        let mut close = vec![est("x", 0.5, 0.0100, 1.0), est("y", 0.5, 0.0105, 1.0)];
        score(&mut close, Weights::new(0.0, 0.0, 1.0));
        assert!(close[0].score - close[1].score < 0.011);
    }

    #[test]
    fn quality_is_relative_to_the_best_candidate() {
        // Hard prompt: small 25%, xlarge 74%. Balanced weights must not pick small just
        // because it is cheap and fast.
        let mut hard = vec![
            est("small", 0.25, 0.0005, 2.0),
            est("xlarge", 0.74, 0.04, 20.0),
        ];
        score(&mut hard, Preference::Balanced.weights());
        assert!(
            (hard
                .iter()
                .find(|e| e.model == "small")
                .unwrap()
                .quality_score
                - 0.25 / 0.74)
                .abs()
                < 1e-12
        );
        // Easy prompt: 91% vs 99%: cheap and fast wins.
        let mut easy = vec![
            est("small", 0.91, 0.0005, 2.0),
            est("xlarge", 0.99, 0.04, 20.0),
        ];
        score(&mut easy, Preference::Balanced.weights());
        assert_eq!(easy[0].model, "small");
    }

    #[test]
    fn parses_and_validates_weights() {
        assert_eq!(
            "quality=60, speed=20,cost=20".parse::<Weights>().unwrap(),
            Weights::new(60.0, 20.0, 20.0)
        );
        assert_eq!(
            "cost=1".parse::<Weights>().unwrap(),
            Weights::new(0.0, 0.0, 1.0)
        );
        assert!("quality=0".parse::<Weights>().is_err(), "all zero");
        assert!("quality=-1,cost=1".parse::<Weights>().is_err());
        assert!("taste=1".parse::<Weights>().is_err());
        assert_eq!(Weights::default(), Weights::new(60.0, 20.0, 20.0));
        assert_eq!("speed".parse::<Preference>().unwrap(), Preference::Speed);
        let n = Weights::new(60.0, 20.0, 20.0).normalized();
        assert!((n.quality + n.speed + n.cost - 1.0).abs() < 1e-12);
    }
}
