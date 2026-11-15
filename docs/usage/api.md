# Using opas over HTTP

opas speaks four LLM APIs. Clients keep their existing SDK and change only the base
URL. Every example below assumes the server runs at `http://localhost:8080`; see
[Docker](docker.md), [Kubernetes](kubernetes.md) or [Helm](helm.md) for running it.

## Routed or direct

The `model` field in a request decides what happens:

| `model` value | What happens |
|---|---|
| `auto` (the `router_model` setting) | The router picks a configured model that serves this API |
| a configured model id, e.g. `claude-haiku-5-5` | Sent straight to that model, with no routing decision |
| anything else | `404`, in the API's error format |

opas rewrites `model` to the upstream model name (`upstream_model` in the config, or
the id) before forwarding. Responses, including SSE streams, are passed through unchanged.

## Credentials

There are two ways to authenticate, chosen per upstream in the config:

- **Server-side keys:** the upstream has `api_key_env = "ANTHROPIC_API_KEY"`, and the
  server's environment holds the key. Clients send any placeholder key; their credentials
  are not forwarded.
- **Pass-through:** without `api_key_env`, the client's own `Authorization`, `x-api-key`
  or `x-goog-api-key` header is forwarded.

The server has no client authentication of its own:

- **Running directly:** it listens on `127.0.0.1:8080` by default, so only the local
  machine can use it.
- **In Docker:** publish the port on loopback (`-p 127.0.0.1:8080:8080`).
- **In Kubernetes:** keep the ClusterIP Service.

To expose it further, put an authenticating proxy or ingress in front. opas logs a
warning at startup when it listens on a non-loopback address.

## OpenAI Chat Completions: `POST /v1/chat/completions`

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused-with-server-keys")
reply = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Summarize this stack trace: ..."}],
)
print(reply.choices[0].message.content)
```

## Open Responses / OpenAI Responses API: `POST /v1/responses`

```python
first = client.responses.create(model="auto", input="Plan a database migration")
# The provider stores this response. A follow-up that references it is pinned to the
# same model, because only that model's provider can resolve the id.
follow = client.responses.create(
    model="auto",
    previous_response_id=first.id,
    input="Now write the rollback script",
)
```

## Anthropic Messages: `POST /v1/messages`

The Anthropic SDK appends `/v1/messages` itself, so the base URL has no `/v1`:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="unused-with-server-keys")
message = client.messages.create(
    model="auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a unit test for parse_date()"}],
)
```

Coding agents that accept a custom Anthropic base URL can be pointed at opas the same
way, with the model set to `auto`.

## Gemini Interactions API: `POST /v1beta/interactions` (or `/v1/interactions`)

```sh
curl localhost:8080/v1beta/interactions \
  -H 'content-type: application/json' -H "x-goog-api-key: $GEMINI_API_KEY" -d '{
    "model": "auto",
    "system_instruction": "Answer briefly.",
    "input": "What changed in HTTP/3?"
  }'
```

A follow-up with `previous_interaction_id` is pinned to the model that created the
interaction, just like `previous_response_id`.

## Choosing the balance: quality, speed and cost

For every candidate, the probe router predicts:

- the probability of success;
- the expected cost, from your prices and the predicted output tokens;
- the expected response time: total time to the complete response, from the model's
  speed and the predicted output tokens.

It scores each 0–1 and picks the best weighted average. The default weights are quality
60 / speed 20 / cost 20. Set your own per request:

| Where | How |
|---|---|
| Proxy endpoints | header `x-opas-weights: quality=80,speed=10,cost=10` or `x-opas-preference: cost` |
| `/v1/route` | body field `"weights": {"quality": 80, "speed": 10, "cost": 10}` or `"preference": "cost"` |
| Server default | `[router] weights = { quality = 60, speed = 20, cost = 20 }` |

| Preset | Quality / speed / cost |
|---|---|
| `quality` | 100 / 0 / 0 |
| `balanced` (default) | 60 / 20 / 20 |
| `cost` | 20 / 10 / 70 |
| `speed` | 20 / 70 / 10 |

**How the scores are computed:**

- **quality** is the predicted success probability.
- **cost** is 1 for the cheapest candidate, falling on a log scale to 0 at 100× the
  cheapest.
- **speed** is 1 for the fastest, falling to 0 at 10× slower.

The ranges are fixed, so a few percent difference in price cannot swing a decision.
Proxied responses carry the chosen model's prediction in `x-opas-estimate`.

## Prices: operator defaults, consumer overrides

Cost estimates use per-model prices in USD per million tokens. The most specific source
wins:

| Level | Who | How |
|---|---|---|
| This request | consumer | header `x-opas-pricing: my-model=4/0.2/20; other=0.1/0.01/0.5` (input/cached/output; `input/output` means no cache discount), or `"pricing": {"my-model": {"input": 4, "cached_input": 0.2, "output": 20}}` in `/v1/route`, or `candidates[].pricing` / `candidates[].models_dev` |
| Configured model | operator | `pricing = {…}` on the model in `opas.toml` |
| models.dev | operator (on by default) | the open catalog at [models.dev](https://models.dev): by `models_dev = "anthropic/claude-opus-5-5"`, or inferred from the upstream URL and model name. Fetched at startup and every `refresh_hours`. Includes higher long-context prices where providers charge them |
| Fallback | — | the tier's default prices |

`GET /v1/pricing` shows each configured model's resolved price and source (`config`,
`models_dev`, `tier_default` or `unknown`).

**Consumer-supplied candidates can be priced by catalog key:**

```json
"candidates": [{"id": "gpt", "tier": "large", "models_dev": "openai/gpt-5.6"}]
```

## Response headers

| Header | When | Meaning |
|---|---|---|
| `x-opas-model` | always | Configured model id that served the request |
| `x-opas-router` | routed requests | `probe`, or `affinity` (pinned by stored provider state) |
| `x-opas-decision-id` | routed requests | Id of the decision: look up its outcome with `GET /v1/decisions/{id}` |
| `x-opas-p-success` | routed requests (probe router) | The chosen model's predicted probability of success, 0–1 |
| `x-opas-cost-usd` | routed requests (probe router) | The chosen model's expected cost of this request, USD |
| `x-opas-response-time-s` | routed requests (probe router) | The chosen model's expected time to the complete response, seconds |
| `x-opas-output-tokens` | routed requests (probe router) | The chosen model's expected output tokens, including reasoning |
| `x-opas-candidates` | routed requests (probe router) | Every candidate, best first: `model; p_success=…; cost_usd=…; response_time_s=…, model; …` |
| `x-opas-routing-ms` | routed requests | How long the decision took (encoder + predictor), milliseconds |
| `x-opas-estimate` | routed requests (probe router) | The chosen model's three predictions in one value (kept for compatibility) |
| `x-opas-previous-model` | continued conversations | Model that served the previous turn (cache-aware routing recognized the conversation) |

Requests that name a model directly are not routed and carry only `x-opas-model`.

**Request header `x-opas-conversation-id`:** an id shared by all turns of one
conversation. opas normally recognizes a conversation by its message history; with this
header (or Codex's `thread-id`) it also recognizes turns whose history was edited or
compacted. For Open Responses requests that move to a different model, opas drops the
encrypted `reasoning` items of the previous model, which the new model cannot read.

## Predicted and actual: `GET /v1/decisions/{id}`

Headers are sent before the response body, so they can only carry predictions. Once a
routed response has finished, opas has measured what actually happened, and
`GET /v1/decisions/{id}` returns both:

```sh
curl localhost:8080/v1/decisions/308a366b-0fe1-41b8-82c1-1b5b823f9202
```

```json
{
  "model": "claude-haiku-5-5",
  "predicted": {"p_success": 0.83, "expected_cost_usd": 0.00125, "expected_response_time_s": 14.7, "expected_output_tokens": 2502, "...": "..."},
  "candidates": [{"model": "claude-haiku-5-5", "...": "..."}, {"model": "claude-opus-5-5", "...": "..."}],
  "timing": {"total_ms": 172.4, "queue_ms": 0.0, "encode_ms": 172.3, "predict_ms": 0.02, "encoded_tokens": 26},
  "actual": {
    "status": 200, "complete": true,
    "usage": {"prompt_tokens": 2040, "cached_prompt_tokens": 2000, "cache_write_tokens": 0, "output_tokens": 350},
    "cost_usd": 0.000199,
    "time_to_first_byte_s": 0.31, "total_time_s": 2.4, "output_tokens_per_sec": 155.2
  }
}
```

- `actual` is `null` while the response is still streaming, and `complete: false` if
  the client disconnected first.
- **Cost** is the provider-reported token usage at the model's prices (the same prices
  as the estimate; cache writes at the input price). `usage` is `null` when the
  provider reported none, e.g. OpenAI Chat streams without
  `stream_options.include_usage`.
- **Time** is measured from sending the request upstream, so it excludes routing
  (`timing`). `output_tokens_per_sec` is set for streamed responses.
- The most recent 10,000 decisions are kept in memory; the log line `completed` has the
  same numbers next to the prediction.

## Multi-turn conversations and caching

Nothing special is needed: send each turn with the full history, as these APIs expect.
opas fingerprints the history and recognizes the conversation. The router then knows
which model's prompt cache is warm. The probe router charges every other candidate the
full input price for the cached prefix, so it switches only when the expected gain is
worth it.

Recognition lasts for `[cache] ttl_secs` (default 300 s, matching typical provider cache
lifetimes). It is kept in memory in each server process.

## Decision only: `POST /v1/route`

Operators who want only this endpoint can turn the proxy off with `[proxy] enabled =
false`: the proxy endpoints then answer 404 and models need no upstream URLs or keys.

Asks the router which model to use, without forwarding anything. Use it when your
application calls the providers itself.

```sh
curl localhost:8080/v1/route -H 'content-type: application/json' -d '{
  "api": "anthropic_messages",
  "request": {"model": "auto", "max_tokens": 512,
              "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational"}]},
  "candidates": [
    {"id": "fast",    "tier": "small"},
    {"id": "default", "tier": "medium"},
    {"id": "best",    "tier": "xlarge", "pricing": {"input": 8, "cached_input": 0.2, "output": 40}}
  ]
}'
```

```json
{
  "decision_id": "6f1c…",
  "router": "probe",
  "model": "default",
  "reason": "best weighted score 0.712 (quality=60,speed=20,cost=20): p=0.71, cost $0.0068, 8.6s",
  "api": "anthropic_messages",
  "weights": {"quality": 60.0, "speed": 20.0, "cost": 20.0},
  "estimates": [
    {"model": "default", "tier": "medium", "p_success": 0.71, "expected_cost_usd": 0.0068,
     "expected_response_time_s": 8.6, "expected_output_tokens": 640, "cached_prompt_tokens": 0,
     "quality_score": 0.71, "speed_score": 0.53, "cost_score": 0.48, "score": 0.629},
    "…one entry per candidate, best first…"
  ]
}
```

**Request fields:**

- `api`: one of `openai_chat`, `anthropic_messages`, `open_responses`,
  `gemini_interactions`.
- `request`: the request body in that API.
- `weights` (optional): `{"quality": 60, "speed": 20, "cost": 20}`; only the ratios
  matter. Or `preference`: `quality`, `balanced`, `cost` or `speed`. The default is
  the server's configured weights (60 / 20 / 20 unless changed).
- `pricing` (optional): prices by model id for this request; see "Prices" below.
- `candidates` (optional): the models to choose from. Each has an `id`, a `tier`
  (`small`, `medium`, `large`, `xlarge`) and optional `pricing` in USD per million
  tokens; without pricing, tier defaults apply. Without `candidates`, the configured
  models that serve `api` are used.

**Response fields:**

- `upstream_model`: set when the chosen model is a configured model.
- `conversation`: reports a recognized earlier turn (`previous_model`, `cached_tokens`).

Decisions are recorded like proxied requests, so cache-aware routing works across turns
of the decision API too.

## Other endpoints

- `GET /v1/pricing` lists the resolved price of every configured model and its source.
- `GET /v1/models` lists `auto` and the configured models. The response is
  Anthropic-shaped when the request carries `anthropic-version`, OpenAI-shaped otherwise.
- `GET /healthz` returns `ok`.
- `GET /openapi.json` and `GET /openapi.yaml` return the OpenAPI 3.1 document for all of
  the above, and `GET /docs` renders it as a browsable reference (the page loads its
  viewer from a CDN). The same document is in the repository at
  [docs/openapi.json](../openapi.json), and `opas --openapi` prints it.

## Errors

Errors use the calling API's own format, so SDKs raise their usual exception types:

| API | Shape |
|---|---|
| Chat Completions | `{"error": {"message", "type": "invalid_request_error" \| "not_found_error" \| "api_error", "param", "code"}}` |
| Anthropic Messages | `{"type": "error", "error": {"type", "message"}}` |
| Open Responses | `{"error": {"message", "type": "invalid_request" \| "not_found" \| "server_error", "param", "code"}}` |
| Gemini Interactions | `{"error": {"code": <http status>, "message", "status": "INVALID_ARGUMENT" \| "NOT_FOUND" \| …}}` |

Upstream provider errors are passed through as the provider sent them.

## Limits

- A request is routed only to models that have an upstream for the same API. Requests
  are not translated between APIs.
- Gemini *agent* interactions (`agent` instead of `model`) are not routed. Send them
  directly to Google.
- Conversation and affinity state is in memory in each process. With several replicas,
  keep a client on one replica; see [Kubernetes](kubernetes.md).
