//! Model prices from [models.dev](https://models.dev), an open catalog of model specs and
//! prices (`https://models.dev/api.json`).
//!
//! Prices resolve, most specific first:
//!
//! 1. per request: the consumer's prices (`x-opas-pricing` header, `/v1/route` `pricing`
//!    or `candidates[].pricing`);
//! 2. the operator's `pricing` for the model in the config;
//! 3. models.dev, by the model's `models_dev` key (`provider/model`), or inferred from
//!    the upstream URL's host and the upstream model name;
//! 4. the model tier's default prices.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::Value;

use crate::conversation::{LongContextPricing, Pricing};

/// Where a resolved price came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceSource {
    Request,
    Config,
    ModelsDev,
    TierDefault,
    Unknown,
}

/// Prices keyed by `provider/model`, as on models.dev.
#[derive(Debug, Clone, Default)]
pub struct PriceCatalog {
    prices: HashMap<String, Pricing>,
    /// API host → provider id, from the catalog's `api` fields plus well-known hosts.
    hosts: HashMap<String, String>,
}

/// First-party API hosts that the catalog lists without an `api` URL.
const KNOWN_HOSTS: &[(&str, &str)] = &[
    ("api.anthropic.com", "anthropic"),
    ("api.openai.com", "openai"),
    ("generativelanguage.googleapis.com", "google"),
    ("api.mistral.ai", "mistral"),
    ("api.groq.com", "groq"),
    ("api.x.ai", "xai"),
    ("api.together.xyz", "togetherai"),
    ("api.cerebras.ai", "cerebras"),
];

/// The host part of a URL (`https://api.anthropic.com/v1` → `api.anthropic.com`).
pub fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let host = rest.split(['/', '?', '#']).next()?.split(':').next()?;
    (!host.is_empty()).then(|| host.to_ascii_lowercase())
}

impl PriceCatalog {
    /// Parse the models.dev `api.json` document. Models without a `cost` are skipped;
    /// a missing `cache_read` price means no cache discount (cached = input price).
    pub fn from_models_dev(doc: &Value) -> Self {
        let mut catalog = PriceCatalog::default();
        for (host, provider) in KNOWN_HOSTS {
            catalog.hosts.insert((*host).into(), (*provider).into());
        }
        let Some(providers) = doc.as_object() else {
            return catalog;
        };
        for (provider_id, provider) in providers {
            if let Some(host) = provider
                .get("api")
                .and_then(Value::as_str)
                .and_then(host_of)
            {
                catalog
                    .hosts
                    .entry(host)
                    .or_insert_with(|| provider_id.clone());
            }
            let Some(models) = provider.get("models").and_then(Value::as_object) else {
                continue;
            };
            for (model_id, model) in models {
                if let Some(p) = model.get("cost").and_then(parse_cost) {
                    catalog
                        .prices
                        .insert(format!("{provider_id}/{model_id}"), p);
                }
            }
        }
        catalog
    }

    pub fn len(&self) -> usize {
        self.prices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Price for a `provider/model` key.
    pub fn get(&self, key: &str) -> Option<Pricing> {
        self.prices.get(key).copied()
    }

    /// Price for a model served at `base_url` under `model` (the upstream name), with the
    /// provider inferred from the URL's host. Returns the catalog key used.
    pub fn infer(&self, base_url: &str, model: &str) -> Option<(String, Pricing)> {
        let provider = self.hosts.get(&host_of(base_url)?)?;
        let key = format!("{provider}/{model}");
        self.get(&key).map(|p| (key, p))
    }
}

fn parse_cost(cost: &Value) -> Option<Pricing> {
    let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64);
    let input = num(cost, "input")?;
    let output = num(cost, "output")?;
    let cached_input = num(cost, "cache_read").unwrap_or(input);
    // Context-size price tiers (e.g. higher prices above 100k or 200k prompt tokens):
    // keep the first one.
    let long_context = cost
        .get("tiers")
        .and_then(Value::as_array)
        .and_then(|tiers| tiers.first())
        .and_then(|t| {
            let above = t.get("tier")?.get("size")?.as_u64()?;
            let tin = num(t, "input")?;
            Some(LongContextPricing {
                above_tokens: above,
                input: tin,
                cached_input: num(t, "cache_read").unwrap_or(tin),
                output: num(t, "output")?,
            })
        });
    Some(Pricing {
        input,
        cached_input,
        output,
        long_context,
    })
}

/// Parse per-request prices: `model=input/cached/output; other=…` (USD per million
/// tokens; `cached` may be omitted, meaning no cache discount).
pub fn parse_price_overrides(s: &str) -> Result<HashMap<String, Pricing>, String> {
    let mut out = HashMap::new();
    for entry in s.split(';').map(str::trim).filter(|e| !e.is_empty()) {
        let (model, prices) = entry
            .split_once('=')
            .ok_or_else(|| format!("expected model=input/cached/output, got `{entry}`"))?;
        let nums: Vec<f64> = prices
            .split('/')
            .map(|v| {
                v.trim()
                    .parse::<f64>()
                    .map_err(|_| format!("`{entry}`: not a number"))
            })
            .collect::<Result<_, _>>()?;
        let (input, cached_input, output) = match nums[..] {
            [i, c, o] => (i, c, o),
            [i, o] => (i, i, o),
            _ => {
                return Err(format!(
                    "`{entry}`: expected input/cached/output or input/output"
                ))
            }
        };
        let p = Pricing {
            input,
            cached_input,
            output,
            long_context: None,
        };
        p.validate().map_err(|e| format!("`{entry}`: {e}"))?;
        out.insert(model.trim().to_owned(), p);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc() -> Value {
        json!({
            "anthropic": {"id": "anthropic", "models": {
                "claude-opus-5-5": {"cost": {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5}},
                "claude-haiku-5-5": {"cost": {"input": 0.1, "output": 0.5, "cache_read": 0.01,
                    "tiers": [{"input": 0.5, "output": 2.5, "cache_read": 0.05, "tier": {"type": "context", "size": 100000}}]}},
                "no-price": {"name": "x"}
            }},
            "openrouter": {"id": "openrouter", "api": "https://openrouter.ai/api/v1", "models": {
                "z-ai/glm-5.3": {"cost": {"input": 0.6, "output": 2.2}}
            }}
        })
    }

    #[test]
    fn parses_prices_cache_and_context_tiers() {
        let c = PriceCatalog::from_models_dev(&doc());
        assert_eq!(c.len(), 3, "the model without a cost is skipped");
        let opus = c.get("anthropic/claude-opus-5-5").unwrap();
        assert_eq!(
            (opus.input, opus.cached_input, opus.output),
            (4.0, 0.2, 20.0)
        );
        let haiku = c.get("anthropic/claude-haiku-5-5").unwrap();
        let lc = haiku.long_context.unwrap();
        assert_eq!((lc.above_tokens, lc.input, lc.output), (100_000, 0.5, 2.5));
        let glm = c.get("openrouter/z-ai/glm-5.3").unwrap();
        assert_eq!(glm.cached_input, 0.6, "no cache_read → no discount");
    }

    #[test]
    fn infers_provider_from_upstream_url() {
        let c = PriceCatalog::from_models_dev(&doc());
        let (key, p) = c
            .infer("https://api.anthropic.com/v1", "claude-opus-5-5")
            .unwrap();
        assert_eq!(key, "anthropic/claude-opus-5-5");
        assert_eq!(p.output, 20.0);
        // Hosts come from the catalog's `api` field too.
        assert!(c
            .infer("https://openrouter.ai/api/v1/", "z-ai/glm-5.3")
            .is_some());
        assert!(c
            .infer("http://localhost:8000/v1", "claude-opus-5-5")
            .is_none());
        assert_eq!(
            host_of("https://API.Anthropic.com:443/v1").as_deref(),
            Some("api.anthropic.com")
        );
    }

    #[test]
    fn parses_request_price_overrides() {
        let o = parse_price_overrides("fast=0.1/0.01/0.5; best = 8/40").unwrap();
        assert_eq!(o["fast"].cached_input, 0.01);
        assert_eq!(
            (o["best"].input, o["best"].cached_input, o["best"].output),
            (8.0, 8.0, 40.0)
        );
        assert!(parse_price_overrides("x=1").is_err());
        assert!(parse_price_overrides("x=a/b/c").is_err());
        assert!(parse_price_overrides("x=-1/0/1").is_err());
        assert!(parse_price_overrides("").unwrap().is_empty());
    }
}
