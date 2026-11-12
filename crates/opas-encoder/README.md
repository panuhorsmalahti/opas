# opas-encoder

Routing on model activations. This crate runs a local LLM (default: Qwen3.5-9B, Q8_0 GGUF)
inside the Rust process with llama.cpp, and turns its hidden states into routing
decisions. Nothing is generated: one prefill pass over the request is enough.

## What it contains

| Module | Purpose |
|---|---|
| `Encoder` (lib root) | Loads a GGUF model and runs one `llama_decode` per request. llama.cpp's graph callback (`cb_eval`) copies the residual stream after every decoder layer (`l_out-N`), pooled by last token and by mean. Prompts are formatted with the Qwen chat template; long prompts keep their head and tail. |
| `chat` | Qwen ChatML formatting from an opas `RouteRequest`, byte-identical to the HF template (golden tests); the thinking mode is pinned explicitly |
| `probe` | The probe file format (`opas-probe/2`): per-tier linear heads for success probability and output tokens, over one layer's hidden state |
| `router` | `ProbeRouter`, an opas `Router`. It predicts each candidate tier's success and output tokens, prices them with the candidate's pricing (cache-aware), and picks `argmax p − λ·cost` |
| `npy` | Minimal `.npy` writer for exporting activations to training |
| `bin/encode.rs` | `opas-encode` CLI: extract activations for JSONL requests into `.npy` files (training input) |

## Requirements

- cmake: the first build compiles llama.cpp through `llama-cpp-sys-2`, pinned at 0.1.160.
- A GGUF model in `models/`, e.g. `models/gguf/Qwen3.5-9B-Q8_0.gguf`.

## Usage

```sh
cargo build --release -p opas-encoder
./target/release/opas-encode --demo
./target/release/opas-encode --input requests.jsonl --out training/out/acts
```

```rust
let mut enc = opas_encoder::Encoder::load("models/gguf/Qwen3.5-9B-Q8_0.gguf", Default::default())?;
let acts = enc.encode_prompt("Fix the race condition in this worker pool.")?;
```

## Tests

Unit tests run anywhere. Model-backed tests run when a model is configured:

```sh
OPAS_TEST_MODEL=models/gguf/Qwen3.5-0.8B-Q8_0.gguf cargo test --release -p opas-encoder
```

The design, measurements and the cross-check against a Python reference are in
`docs/complexity-from-activations.md`. Training is in `training/`.
