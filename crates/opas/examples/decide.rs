//! Use the router as a library: decide which model to call, then call it yourself.
//!
//!     cargo run -p opas --example decide

use opas::{ApiFormat, Candidate, RandomRouter, RouteRequest, Router};
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let body = json!({
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Refactor this function to be iterative."}]
    });
    let request = RouteRequest::from_json(ApiFormat::AnthropicMessages, &body).unwrap();
    let candidates = [
        Candidate::from("claude-opus-5-5"),
        Candidate::from("claude-haiku-5-5"),
    ];
    let router = RandomRouter::new();

    for _ in 0..5 {
        let decision = router.route(&request, &candidates).await.unwrap();
        println!(
            "{} -> {} ({})",
            decision.decision_id, decision.model, decision.reason
        );
    }
}
