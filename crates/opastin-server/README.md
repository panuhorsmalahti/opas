# opastin-server

The `opastin` binary: an HTTP router that speaks four LLM APIs. Clients point their
existing SDKs at it, it picks a model for each request, and it proxies the request to that
model's provider, streaming responses through unchanged.

## Endpoints

| Endpoint | API |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/responses` | Open Responses (OpenAI Responses API) |
| `POST /v1beta/interactions`, `POST /v1/interactions` | Gemini Interactions API |
| `POST /v1/route` | Decision only, nothing forwarded; accepts its own `candidates` list (id, tier, pricing) |
| `GET /v1/models` | Router model and configured models |
| `GET /healthz` | Liveness |

**Routed vs direct:** send `"model": "auto"` (the `router_model` setting) to be routed;
send a configured model id to go direct.

**Errors** come back in the calling API's own format.

**Response headers:**

| Header | Present | Meaning |
|---|---|---|
| `x-opastin-model` | always | the model that served the request |
| `x-opastin-router` | routed requests | the router that decided |
| `x-opastin-decision-id` | routed requests | the decision's id |
| `x-opastin-previous-model` | continued conversations | the model that served the previous turn |

## Routers

The router is set by `[router] kind` in the config:

- `random`: random, and cache-aware (a continued conversation stays on its model).
- `probe`: the activation-probe router from `opastin-encoder`. It needs the `probe`
  feature, which is on by default, links llama.cpp, and needs cmake to build.

Build without the probe router and without llama.cpp:
`cargo build -p opastin-server --no-default-features`.

## Run

```sh
cargo run --release -p opastin-server -- --config opastin.toml   # or OPASTIN_CONFIG / OPASTIN_LISTEN
```

The repository README covers configuration, the probe router, and deployment (Docker,
Kubernetes, Helm).
