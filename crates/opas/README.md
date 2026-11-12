# opas (library / SDK)

The core of opas: everything needed to make a routing decision, with no dependency on
an inference runtime. Use it inside your own Rust service, or through the
`opas-server` binary.

## What it contains

| Module | Purpose |
|---|---|
| `api` | `ApiFormat`: OpenAI Chat Completions, Anthropic Messages, Open Responses, Gemini Interactions |
| `request` | `RouteRequest`: parses a request body in any of those formats into one view (system prompt, messages, tools, stream flag, continuation id) |
| `router` | The `Router` trait, `Candidate`, `Decision`, and `RandomRouter` (random baseline) |
| `tier` | `Tier`: small / medium / large / xlarge capability bands, with default prices |
| `conversation` | Cache-aware routing: `ConversationTracker` fingerprints message prefixes to find the model that served the previous turn; `switch_cost` prices a lost prompt cache |
| `config` | TOML configuration: models, upstreams per API format, tiers, pricing, router, cache settings |
| `gateway` (feature `gateway`) | `Gateway`: routes and forwards requests to upstream providers over HTTP (reqwest). Adds affinity for provider-stored state (`previous_response_id`, `previous_interaction_id`) |

## Use as a library

Decide only:

```rust
use opas::{ApiFormat, Candidate, RandomRouter, RouteRequest, Router, Tier};

let request = RouteRequest::from_json(ApiFormat::OpenaiChat, &body)?;
let candidates = [
    Candidate::new("my-small").with_tier(Some(Tier::Small)),
    Candidate::new("my-large").with_tier(Some(Tier::Large)),
];
let decision = RandomRouter::new().route(&request, &candidates).await?;
```

Route and forward (feature `gateway`):

```rust
let gateway = opas::gateway::Gateway::new(opas::Config::load("opas.toml")?)?;
let forwarded = gateway.forward(ApiFormat::AnthropicMessages, body, &headers).await?;
```

To plug in custom routing logic, implement `Router` and pass it to
`Gateway::with_router`. `opas-encoder`'s `ProbeRouter` works this way.

Examples: `cargo run -p opas --example decide`, and
`cargo run -p opas --features gateway --example gateway`.
