# Using opastin as a Rust SDK

The crates can be used as libraries, without running the server. The crates are not
published to crates.io yet, so depend on them by path or git:

```toml
[dependencies]
opastin = { git = "https://github.com/OWNER/opastin", features = ["gateway"] }
opastin-encoder = { git = "https://github.com/OWNER/opastin" }  # probe router; builds llama.cpp (cmake)
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde_json = "1"
```

| Crate | Use it for |
|---|---|
| `opastin` | Parsing requests in four APIs, the `Router` trait, tiers, cache-aware conversation tracking, configuration |
| `opastin` with `gateway` | Forwarding requests to providers (`Gateway`) |
| `opastin-encoder` | Running the encoder model in-process (`Encoder`) and the activation-probe router (`ProbeRouter`) |

## 1. Decide which model to use

Parse the request, describe your models by tier, and ask a router:

```rust
use opastin::{ApiFormat, Candidate, RandomRouter, Pricing, RouteRequest, Router, Tier};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let body = json!({"model": "auto", "messages": [{"role": "user", "content": "Explain monads"}]});
    let request = RouteRequest::from_json(ApiFormat::OpenaiChat, &body)?;

    let candidates = [
        Candidate::new("my-haiku").with_tier(Some(Tier::Small)),
        Candidate::new("my-opus")
            .with_tier(Some(Tier::Large))
            .with_pricing(Some(Pricing { input: 4.0, cached_input: 0.2, output: 20.0 })),
    ];

    let decision = RandomRouter::new().route(&request, &candidates).await?;
    println!("{} ({})", decision.model, decision.reason);
    Ok(())
}
```

`RouteRequest::from_json` accepts `ApiFormat::OpenaiChat`, `AnthropicMessages`,
`OpenResponses` and `GeminiInteractions` bodies.

## 2. Route with activations: `ProbeRouter`

The probe router needs the encoder model and a trained probe file (see "Training the
probe" in the main README):

```rust
use opastin_encoder::ProbeRouter;

let router = ProbeRouter::load(
    "models/gguf/Qwen3.5-9B-Q8_0.gguf",              // encoder model
    "models/probes/llmrouterbench-qwen35-9b.json",   // probe weights
    10.0,                                            // λ: success probability per USD
    -1,                                              // encoder layers on the GPU (-1 = all)
)?;
let decision = router.route(&request, &candidates).await?;
```

**Loading and threading:** loading takes a few seconds. `ProbeRouter` is cheap to clone,
and it runs the encoder on a blocking thread, so it is safe to call from async code.

**Candidates** need a `tier`; those without one are skipped. The reason string lists each
candidate's predicted success, output tokens, cost and score.

**Without the async trait:** `rank_blocking(&request, &candidates)` returns the scored
candidates, best first.

## 3. Cache-aware routing in your own loop

Without the `Gateway`, keep a `ConversationTracker` yourself:

```rust
use std::time::Duration;
use opastin::conversation::{ConversationTracker, PrefixKeys};

let tracker = ConversationTracker::new(Duration::from_secs(300), 100_000);

let keys = PrefixKeys::new(ApiFormat::AnthropicMessages, &body);
let mut request = RouteRequest::from_json(ApiFormat::AnthropicMessages, &body)?;
request.conversation = tracker.lookup(&keys);         // previous model + cached tokens, if any
let decision = router.route(&request, &candidates).await?;
tracker.record(&keys, &decision.model);               // so the next turn finds it
```

Routers see `request.conversation`:

- The random router stays on the previous model.
- The probe router bills the cached prefix at the cached price for that model only.

## 4. Route and forward: `Gateway`

`Gateway` is what the server runs. It loads the config, routes, tracks conversations and
provider-stored ids, and forwards to the upstream over HTTP:

```rust
use std::sync::Arc;
use opastin::gateway::Gateway;
use opastin::{ApiFormat, Config};

let config = Config::load("opastin.toml")?;
let gateway = Gateway::new(config)?;                        // random router from the config
// or, with the probe router:
// let gateway = Gateway::with_router(config, Arc::new(router));

let forwarded = gateway
    .forward(ApiFormat::OpenResponses, body, &http::HeaderMap::new())
    .await?;
println!("served by {}", forwarded.model);
let text = forwarded.response.text().await?;                // reqwest::Response (may be SSE)
```

`Gateway::decide` and `Gateway::decide_among` return a decision without forwarding;
`decide_among` takes your own candidate list.

**Stored-state affinity:** when you forward Open Responses or Gemini Interactions
requests yourself, call `gateway.record_stored_response(&response_id, &model)` with each
response's id. Follow-ups with `previous_response_id` / `previous_interaction_id` are
then pinned to that model. The server does this automatically.

## 5. Your own routing logic

Implement `Router` and plug it into the `Gateway`:

```rust
use async_trait::async_trait;
use opastin::{Candidate, Decision, RouteError, RouteRequest, Router, Tier};

struct LengthRouter;

#[async_trait]
impl Router for LengthRouter {
    fn name(&self) -> &str { "length" }

    async fn route(&self, req: &RouteRequest, cands: &[Candidate]) -> Result<Decision, RouteError> {
        let want = if req.estimated_tokens() > 2_000 { Tier::Large } else { Tier::Small };
        let pick = cands.iter().find(|c| c.tier == Some(want)).or(cands.first())
            .ok_or(RouteError::NoCandidates)?;
        Ok(Decision {
            decision_id: uuid::Uuid::new_v4().to_string(),
            router: self.name().into(),
            model: pick.id.clone(),
            reason: format!("prompt ≈ {} tokens", req.estimated_tokens()),
            conversation: None,
        })
    }
}
```

## 6. Raw activations: `Encoder`

To use the encoder's hidden states for your own analysis or training:

```rust
use opastin_encoder::{Encoder, EncoderConfig};

let mut enc = Encoder::load("models/gguf/Qwen3.5-9B-Q8_0.gguf", EncoderConfig {
    layers: Some(vec![16, 24]),   // hidden-state indices to capture (None = all)
    max_tokens: 2048,             // longer prompts keep their first and last 1024 tokens
    ..EncoderConfig::default()
})?;
let acts = enc.encode_request(&request)?;
// acts.layers == [16, 24]; acts.last[k] / acts.mean[k] are 4096-dim vectors
```

`Encoder` is `Send`, not `Sync`: wrap it in a `Mutex` to share it. The
`opastin-encode` CLI writes the same activations to `.npy` files for batch jobs.

## Examples in the repository

- `cargo run -p opastin --example decide`
- `cargo run -p opastin --features gateway --example gateway -- opastin.toml`
- `cargo run --release -p opastin-encoder --example overhead -- models/gguf/Qwen3.5-0.8B-Q8_0.gguf`
