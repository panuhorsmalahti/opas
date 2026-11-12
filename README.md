# opastin

An LLM router written in Rust. It picks a model for each request and works in three ways:

1. **As a gateway.** Exposes OpenAI-compatible and Anthropic-compatible endpoints,
   routes each request, and proxies it to the chosen model, including streamed (SSE)
   responses.
2. **As a decision API.** Returns which model it would choose and sends nothing
   upstream. Your own code then makes the call.
3. **As a library (SDK).** Add the `opastin` crate and call the router in-process.

The only router so far is **random**: a uniform random choice between the
configured models. A smarter router should beat it at the same cost.

Research notes on existing routers, their APIs, and routing benchmarks are in
[llm-routers.md](llm-routers.md).

## Layout

| Path | What |
|---|---|
| `crates/opastin` | Library / SDK: request parsing, the `Router` trait, `RandomRouter`, config, and the `gateway` feature for forwarding |
| `crates/opastin-server` | The `opastin` binary: HTTP endpoints on top of the library |
| `crates/opastin-encoder` | In-process llama.cpp encoder: runs a local model's prefill pass and captures per-layer hidden states, the input for complexity-based routing. Includes the `opastin-encode` CLI. |
| `probe/` | Python (MLX) reference extractor and cross-check scripts |
| `docs/complexity-from-activations.md` | Research and design notes for routing on model activations |
| `opastin.example.toml` | Example config with two Claude models |

## Run the server

```sh
cp opastin.example.toml opastin.toml
export ANTHROPIC_API_KEY=...
cargo run -p opastin-server -- --config opastin.toml
```

Set `RUST_LOG=debug` for more detailed logs.

### 1. Routing endpoints

| Endpoint | Format |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages` | Anthropic Messages |

- **Routed:** send `"model": "auto"` (the `router_model` setting) to have the router
  choose.
- **Direct:** send a configured model id to skip routing and go straight to that model.
- **Point an SDK at it:** existing SDKs and tools only need a new base URL, e.g.
  `http://127.0.0.1:8080/v1` for OpenAI or `http://127.0.0.1:8080` for Anthropic.

```sh
curl localhost:8080/v1/messages -H 'content-type: application/json' -d '{
  "model": "auto", "max_tokens": 256,
  "messages": [{"role": "user", "content": "Say hello"}]
}'
```

Response headers on these endpoints:

- `x-opastin-model`: the model that served the request.
- `x-opastin-router`, `x-opastin-decision-id`: present on routed requests only.

Errors use the error format of the API that was called.

**Credentials:**

- If a model's upstream sets `api_key_env`, opastin sends that key. Otherwise it
  forwards the client's `Authorization` or `x-api-key` header.
- `anthropic-version` and `anthropic-beta` are forwarded as sent. `anthropic-version`
  defaults to `2023-06-01` if the client doesn't send one.

**Limitation:** a request is only routed to models that have an upstream for the same
API format. Translating between OpenAI and Anthropic formats is not implemented.

### 2. Decision-only endpoint

```sh
curl localhost:8080/v1/route -H 'content-type: application/json' -d '{
  "api": "anthropic_messages",
  "request": {"model": "auto", "max_tokens": 256, "messages": [{"role": "user", "content": "hi"}]}
}'
# {"decision_id":"…","router":"random","model":"claude-haiku-5-5",
#  "upstream_model":"claude-haiku-5-5","reason":"uniform random pick from 2 candidates",
#  "api":"anthropic_messages"}
```

`api` is `openai_chat` or `anthropic_messages`. `GET /v1/models` lists the router model
and the configured models.

### 3. Library / SDK

```toml
[dependencies]
opastin = { path = "crates/opastin" }                         # decision only
# opastin = { path = "crates/opastin", features = ["gateway"] }  # plus forwarding
```

```rust
use opastin::{ApiFormat, Candidate, RandomRouter, RouteRequest, Router};

let request = RouteRequest::from_json(ApiFormat::OpenaiChat, &body)?;
let candidates = [Candidate::from("claude-opus-5-5"), Candidate::from("claude-haiku-5-5")];
let decision = RandomRouter::new().route(&request, &candidates).await?;
// decision.model is the model to call
```

Two runnable examples:

- `cargo run -p opastin --example decide`: decision only.
- `cargo run -p opastin --features gateway --example gateway`: in-process routing and
  forwarding with `opastin::gateway::Gateway`.

To add your own routing logic, implement the `Router` trait. It is async, so a router
can call a classifier or another model. Then pass it to `Gateway::with_router`.

## Activation encoder (experimental)

The encoder runs a GGUF model (default: Qwen3.5-9B Q8_0) inside the Rust process with
llama.cpp. It captures the hidden state after every layer for one prefill pass, and
generates no tokens. Model files live in `models/`, which git ignores.

The first build compiles llama.cpp, so it needs cmake (`brew install cmake`).

```sh
cargo build --release -p opastin-encoder
./target/release/opastin-encode --demo
./target/release/opastin-encode --input requests.jsonl --out probe/out/acts   # .npy outputs
OPASTIN_TEST_MODEL=models/gguf/Qwen3.5-0.8B-Q8_0.gguf cargo test --release -p opastin-encoder
```

See [docs/complexity-from-activations.md](docs/complexity-from-activations.md) for how
these activations are meant to drive routing.

## Development

```sh
cargo test     # unit tests, plus end-to-end tests against a mock upstream
cargo clippy --all-targets --all-features
```
