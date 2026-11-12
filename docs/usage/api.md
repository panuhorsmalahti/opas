# Using opastin over HTTP

opastin speaks four LLM APIs. Clients keep their existing SDK and change only the base
URL. Every example below assumes the server runs at `http://localhost:8080`; see
[Docker](docker.md), [Kubernetes](kubernetes.md) or [Helm](helm.md) for running it.

## Routed or direct

The `model` field in a request decides what happens:

| `model` value | What happens |
|---|---|
| `auto` (the `router_model` setting) | The router picks a configured model that serves this API |
| a configured model id, e.g. `claude-haiku-5-5` | Sent straight to that model, with no routing decision |
| anything else | `404`, in the API's error format |

opastin rewrites `model` to the upstream model name (`upstream_model` in the config, or
the id) before forwarding. Responses, including SSE streams, are passed through unchanged.

## Credentials

There are two ways to authenticate, chosen per upstream in the config:

- **Server-side keys:** the upstream has `api_key_env = "ANTHROPIC_API_KEY"`, and the
  server's environment holds the key. Clients send any placeholder key; their credentials
  are not forwarded.
- **Pass-through:** without `api_key_env`, the client's own `Authorization`, `x-api-key`
  or `x-goog-api-key` header is forwarded.

The server itself has no authentication of its own. Expose it only on trusted networks,
or put it behind an authenticating proxy or ingress.

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

Coding agents that accept a custom Anthropic base URL can be pointed at opastin the same
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

## Response headers

| Header | When | Meaning |
|---|---|---|
| `x-opastin-model` | always | Configured model id that served the request |
| `x-opastin-router` | routed requests | `random`, `probe`, or `affinity` (pinned by stored provider state) |
| `x-opastin-decision-id` | routed requests | Unique id of the decision, for logs and outcome feedback |
| `x-opastin-previous-model` | continued conversations | Model that served the previous turn (cache-aware routing recognized the conversation) |

## Multi-turn conversations and caching

Nothing special is needed: send each turn with the full history, as these APIs expect.
opastin fingerprints the history and recognizes the conversation. The router then knows
which model's prompt cache is warm:

- The random router stays on that model.
- The probe router charges every other candidate the full input price for the cached
  prefix, so it switches only when the expected quality gain is worth it.

Recognition lasts for `[cache] ttl_secs` (default 300 s, matching typical provider cache
lifetimes). It is kept in memory in each server process.

## Decision only: `POST /v1/route`

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
  "upstream_model": null,
  "reason": "lambda=10; default (medium): p=0.71 out≈640 tok cost=$0.0068 score=0.642; …",
  "api": "anthropic_messages",
  "conversation": null
}
```

**Request fields:**

- `api`: one of `openai_chat`, `anthropic_messages`, `open_responses`,
  `gemini_interactions`.
- `request`: the request body in that API.
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

- `GET /v1/models` lists `auto` and the configured models. The response is
  Anthropic-shaped when the request carries `anthropic-version`, OpenAI-shaped otherwise.
- `GET /healthz` returns `ok`.

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
