//! Use the router in-process to route and forward a request, without running the server.
//!
//!     ANTHROPIC_API_KEY=... cargo run -p opastin --features gateway --example gateway -- opastin.example.toml

use http::HeaderMap;
use opastin::gateway::Gateway;
use opastin::{ApiFormat, Config};
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opastin.example.toml".into());
    let config = Config::load(&path)?;
    let router_model = config.router_model.clone();
    let gateway = Gateway::new(config)?;

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
