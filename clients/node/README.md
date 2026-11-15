# opas-client

Node.js client for [opas](https://github.com/panuhorsmalahti/opas), a model router for
LLM requests. It talks to a running opas server, so routing (and the encoder model)
stays in one place while any number of Node services use it.

- **Route through opas with the SDKs you already use:** `opas.openai()` and
  `opas.anthropic()` return options for the official OpenAI and Anthropic SDKs.
- **Decide without sending:** `opas.route()` returns the chosen model and every
  candidate's predicted success probability, cost and response time.
- **See what was chosen:** `readDecision()` reads the `x-opas-*` headers of a proxied
  response.

Typed from opas's OpenAPI document; no runtime dependencies; Node 18 or newer.

## Install

The package is attached to each [GitHub release](https://github.com/panuhorsmalahti/opas/releases):

```sh
npm install https://github.com/panuhorsmalahti/opas/releases/download/v0.1.3/opas-client-0.1.3.tgz
```

Or from a checkout: `cd clients/node && npm install && npm run build`, then
`npm install /path/to/opas/clients/node` in your project.

## Use it with the Anthropic or OpenAI SDK

```ts
import Anthropic from "@anthropic-ai/sdk";
import { Opas, readDecision } from "opas-client";

const opas = new Opas(); // http://127.0.0.1:8080, or $OPAS_URL
const anthropic = new Anthropic(opas.anthropic({ preference: "balanced" }));

const { data, response } = await anthropic.messages
  .create({ model: "auto", max_tokens: 1024, messages: [{ role: "user", content: "Explain monads" }] })
  .withResponse();

console.log(data.content);
const decision = readDecision(response);
// { model: "claude-haiku-5-5", router: "probe", decisionId: "…", routingMs: 172.4,
//   estimate: { pSuccess: 0.83, costUsd: 0.00125, responseTimeS: 14.7, outputTokens: 2502 },
//   candidates: [{ model: "claude-haiku-5-5", pSuccess: 0.83, … }, { model: "claude-opus-5-5", … }] }

// After the response: what it actually cost and how long it took.
const record = await opas.decision(decision!.decisionId!);
record.actual; // { usage: {...}, cost_usd: 0.000199, total_time_s: 2.4, output_tokens_per_sec: 155.2, … }
```

```ts
import OpenAI from "openai";

const openai = new OpenAI(opas.openai({ weights: { quality: 80, speed: 10, cost: 10 } }));
const chat = await openai.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "Write a haiku about routers" }],
});
```

`model: "auto"` lets opas choose; a configured model id goes straight to that model.
Routing options (`weights`, `preference`, `pricing`) become `x-opas-*` headers on every
request the SDK client sends; create another client for another balance.

**API keys.** When the opas config holds the provider key (`api_key_env`), the SDK's key
is ignored, and the helpers default it to a placeholder. When it does not, opas passes
the SDK's key through: set `apiKey`, or the usual `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.

## Decide without sending

```ts
const decision = await opas.route({
  api: "anthropic_messages",
  request: { model: "auto", max_tokens: 1024, messages: [{ role: "user", content: "…" }] },
  // optional: your own candidates, weights or preference, and prices
  candidates: [
    { id: "my-small", tier: "small", pricing: { input: 0.1, cached_input: 0.01, output: 0.5 } },
    { id: "my-large", tier: "large" },
  ],
  preference: "cost",
});

decision.model;      // "my-small"
decision.estimates;  // every candidate: p_success, expected_cost_usd, expected_response_time_s, scores
```

Then call the chosen model however you like.

## Reference

| | |
|---|---|
| `new Opas({ baseUrl?, fetch?, headers?, timeoutMs? })` | A client for one server |
| `opas.route(body)` | `POST /v1/route`: a decision and estimates |
| `opas.decision(id)` | `GET /v1/decisions/{id}`: a decision's prediction, routing time, and the actual tokens, cost and time once the response has ended |
| `opas.pricing()` | `GET /v1/pricing`: prices in effect and their source |
| `opas.models()` | `GET /v1/models` |
| `opas.health()` | `true` when the server answers `/healthz` |
| `opas.openapi()` | The server's OpenAPI document |
| `opas.openai(options?)`, `opas.anthropic(options?)` | SDK constructor options (`baseURL`, `apiKey`, `defaultHeaders`) |
| `routingHeaders(options)` | The `x-opas-*` headers for `weights`, `preference`, `pricing` |
| `readDecision(responseOrHeaders)` | The decision from a proxied response, or `undefined` |
| `OpasError` | Thrown for error responses (`status`, `body`) and when opas is unreachable (`status` 0) |

All request and response types (`RouteRequest`, `RouteResponse`, `Estimate`, `Weights`,
…) are exported. The full HTTP API is described at `http://127.0.0.1:8080/docs`.

## Development

```sh
npm install
npm run generate     # types from ../../docs/openapi.json
npm run typecheck && npm run build
OPAS_BIN=../../target/debug/opas npm test   # also runs against a real opas binary
```
