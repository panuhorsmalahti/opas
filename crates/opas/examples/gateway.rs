//! Route and forward a request in-process, without running the server.
//!
//!     ANTHROPIC_API_KEY=... cargo run -p opas --features gateway --example gateway -- opas.example.toml
//!
//! The probe router lives in `opas-encoder` (it runs the encoder model); this example
//! plugs in a small rule-based router instead, through `Gateway::with_router`.

use std::sync::Arc;

use http::HeaderMap;
use opas::gateway::Gateway;
use opas::{ApiFormat, Candidate, Config, Decision, RouteError, RouteRequest, Router, Tier};
use serde_json::json;

/// Long prompts to the largest tier, short ones to the smallest.
struct ByLength;

#[async_trait::async_trait]
impl Router for ByLength {
    fn name(&self) -> &str {
        "by-length"
    }

    async fn route(&self, req: &RouteRequest, cands: &[Candidate]) -> Result<Decision, RouteError> {
        let tokens = req.estimated_tokens();
        let pick = if tokens > 2_000 {
            cands.iter().max_by_key(|c| c.tier.unwrap_or(Tier::Small))
        } else {
            cands.iter().min_by_key(|c| c.tier.unwrap_or(Tier::Xlarge))
        }
        .ok_or(RouteError::NoCandidates)?;
        Ok(Decision {
            decision_id: format!("{tokens}"),
            router: self.name().into(),
            model: pick.id.clone(),
            reason: format!("prompt ≈ {tokens} tokens"),
            conversation: None,
            estimates: Vec::new(),
            weights: None,
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opas.example.toml".into());
    let mut config = Config::load(&path)?;
    config.router = None; // the config's [router] is replaced by ByLength
    let router_model = config.router_model.clone();
    let gateway = Gateway::with_router(config, Arc::new(ByLength));

    let body = json!({
        "model": router_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say hello in five words."}]
    });
    let forwarded = gateway
        .forward(ApiFormat::AnthropicMessages, body, &HeaderMap::new())
        .await?;

    println!(
        "routed to {} ({:?})",
        forwarded.model,
        forwarded.decision.map(|d| d.reason)
    );
    println!("status {}", forwarded.response.status());
    println!("{}", forwarded.response.text().await?);
    Ok(())
}
