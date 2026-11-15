# opas-server

The `opas` binary: an HTTP router that speaks four LLM APIs. Clients point their
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
| `GET /v1/pricing` | Resolved price per model and its source |
| `GET /healthz` | Liveness |
| `GET /openapi.json`, `/openapi.yaml`, `/docs` | OpenAPI 3.1 document (also `opas --openapi`) and a browsable reference |

**Routed vs direct:** send `"model": "auto"` (the `router_model` setting) to be routed;
send a configured model id to go direct.

**Errors** come back in the calling API's own format.

**Response headers:**

| Header | Present | Meaning |
|---|---|---|
| `x-opas-model` | always | the model that served the request |
| `x-opas-router` | routed requests | the router that decided |
| `x-opas-decision-id` | routed requests | the decision's id |
| `x-opas-previous-model` | continued conversations | the model that served the previous turn |

## Routers

`[router] kind = "probe"` runs the activation-probe router from `opas-encoder`. It needs
the `probe` feature, which is on by default, links llama.cpp, and needs cmake to build.

Without a `[router]` section the server is a plain proxy: requests must name a
configured model, and `auto` is rejected. Build that without llama.cpp:
`cargo build -p opas-server --no-default-features`.

## Run

```sh
cargo run --release -p opas-server -- --config opas.toml   # or OPAS_CONFIG / OPAS_LISTEN
```

The repository README covers configuration, the probe router, and deployment (Docker,
Kubernetes, Helm).
