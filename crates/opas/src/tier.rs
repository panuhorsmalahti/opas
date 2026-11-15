//! Model capability tiers. Routers that predict outcomes (the probe router) reason about
//! tiers, not specific model names, so a deployment maps its own models onto tiers.

use serde::{Deserialize, Serialize};

use crate::conversation::{Pricing, Speed};

/// Capability band of a model.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Fast, cheap models (Haiku-class).
    Small,
    /// Everyday models (Sonnet-class).
    Medium,
    /// Frontier models (Opus-class).
    Large,
    /// The most capable, most expensive models (Fable-class).
    Xlarge,
}

impl Tier {
    pub const ALL: [Tier; 4] = [Tier::Small, Tier::Medium, Tier::Large, Tier::Xlarge];

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Small => "small",
            Tier::Medium => "medium",
            Tier::Large => "large",
            Tier::Xlarge => "xlarge",
        }
    }

    /// Typical list prices for the tier (USD per million tokens), used when a model has
    /// no configured pricing. Based on Anthropic's Haiku / Sonnet / Opus / Fable prices
    /// in November 2026; set `pricing` per model for anything else.
    pub fn default_pricing(self) -> Pricing {
        let (input, cached_input, output) = match self {
            Tier::Small => (0.10, 0.01, 0.50),
            Tier::Medium => (2.00, 0.10, 10.00),
            Tier::Large => (4.00, 0.20, 20.00),
            Tier::Xlarge => (8.00, 0.20, 40.00),
        };
        Pricing {
            input,
            cached_input,
            output,
            long_context: None,
        }
    }
}

impl Tier {
    /// Typical serving speed for the tier, used when a model has no configured `speed`.
    ///
    /// Output speed counts thinking tokens, as the predicted output tokens do; time to
    /// first token is to the first token of any kind. Set from Anthropic's models measured
    /// in November 2026 (Haiku 5.5 about 210–260 tokens/s, Sonnet 5.5 115–175, Opus 5.5
    /// 100–125, Fable 5.2 100–125; first token after 0.6–2 s, nearly independent of
    /// prompt length up to 30k tokens), rounded down a little because the tiers also
    /// cover other providers. Real numbers vary with provider, load and effort level;
    /// configure `speed` per model when they matter.
    pub fn default_speed(self) -> Speed {
        let (output_tokens_per_sec, ttft_s) = match self {
            Tier::Small => (180.0, 0.8),
            Tier::Medium => (120.0, 1.2),
            Tier::Large => (90.0, 1.4),
            Tier::Xlarge => (80.0, 1.5),
        };
        Speed {
            output_tokens_per_sec,
            ttft_s,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Tier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Tier::ALL
            .into_iter()
            .find(|t| t.as_str() == s)
            .ok_or_else(|| format!("unknown tier `{s}` (expected small, medium, large or xlarge)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_orders_and_prices() {
        assert_eq!("xlarge".parse::<Tier>().unwrap(), Tier::Xlarge);
        assert!("huge".parse::<Tier>().is_err());
        assert!(Tier::Small < Tier::Medium && Tier::Large < Tier::Xlarge);
        assert_eq!(serde_json::to_string(&Tier::Medium).unwrap(), "\"medium\"");
        let prices: Vec<f64> = Tier::ALL
            .iter()
            .map(|t| t.default_pricing().output)
            .collect();
        assert!(
            prices.windows(2).all(|w| w[0] < w[1]),
            "pricier with each tier"
        );
    }
}
