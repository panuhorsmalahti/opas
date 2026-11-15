// Tests run directly on the TypeScript sources (Node 22.18+ strips types).
//   npm test                      unit tests against a stand-in server
//   OPAS_BIN=../../target/debug/opas npm test
//                                 also starts a real opas (plain proxy) in front of a
//                                 mock provider
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, test } from "node:test";

import { Opas, OpasError, readDecision, routingHeaders } from "../src/index.ts";

async function listen(handler: (req: IncomingMessage, body: string, res: ServerResponse) => void): Promise<Server> {
  const server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    handler(req, body, res);
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return server;
}
const urlOf = (s: Server) => `http://127.0.0.1:${(s.address() as AddressInfo).port}`;

describe("client against a stand-in server", () => {
  let server: Server;
  const seen: { method?: string; path?: string; body?: unknown }[] = [];

  before(async () => {
    server = await listen((req, body, res) => {
      seen.push({ method: req.method, path: req.url, body: body ? JSON.parse(body) : undefined });
      res.setHeader("content-type", "application/json");
      if (req.url === "/v1/route") {
        const parsed = JSON.parse(body);
        if (parsed.api === "grpc") {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: { message: "unknown variant `grpc`", type: "invalid_request_error" } }));
          return;
        }
        res.end(
          JSON.stringify({
            decision_id: "d1",
            router: "probe",
            model: "small",
            upstream_model: "small-upstream",
            reason: "best score",
            api: parsed.api,
            conversation: null,
            weights: { quality: 60, speed: 20, cost: 20 },
            estimates: [],
          }),
        );
      } else if (req.url === "/healthz") {
        res.setHeader("content-type", "text/plain");
        res.end("ok");
      } else {
        res.statusCode = 404;
        res.end(JSON.stringify({ error: { message: "nope", type: "not_found_error" } }));
      }
    });
  });
  after(() => server.close());

  test("route() posts the body and returns the decision", async () => {
    const opas = new Opas({ baseUrl: urlOf(server) + "/" });
    const d = await opas.route({
      api: "anthropic_messages",
      request: { model: "auto", max_tokens: 10, messages: [{ role: "user", content: "hi" }] },
      preference: "cost",
    });
    assert.equal(d.model, "small");
    assert.equal(d.upstream_model, "small-upstream");
    const last = seen.at(-1)!;
    assert.equal(last.method, "POST");
    assert.equal(last.path, "/v1/route");
    assert.deepEqual((last.body as { preference: string }).preference, "cost");
  });

  test("errors become OpasError with the status and message", async () => {
    const opas = new Opas({ baseUrl: urlOf(server) });
    await assert.rejects(
      // @ts-expect-error: deliberately invalid api
      opas.route({ api: "grpc", request: {} }),
      (e: unknown) => e instanceof OpasError && e.status === 400 && /unknown variant/.test(e.message),
    );
  });

  test("unreachable server is OpasError with status 0; health() is false", async () => {
    const opas = new Opas({ baseUrl: "http://127.0.0.1:9", timeoutMs: 2000 });
    await assert.rejects(opas.pricing(), (e: unknown) => e instanceof OpasError && e.status === 0);
    assert.equal(await opas.health(), false);
    assert.equal(await new Opas({ baseUrl: urlOf(server) }).health(), true);
  });
});

describe("helpers", () => {
  test("routingHeaders()", () => {
    assert.deepEqual(routingHeaders({ weights: { quality: 80, cost: 20 } }), { "x-opas-weights": "quality=80,cost=20" });
    assert.deepEqual(routingHeaders({ preference: "speed" }), { "x-opas-preference": "speed" });
    assert.deepEqual(
      routingHeaders({ pricing: { a: { input: 1, output: 5 }, b: { input: 2, cached_input: 0.2, output: 10 } } }),
      { "x-opas-pricing": "a=1/5; b=2/0.2/10" },
    );
    assert.deepEqual(routingHeaders(), {});
  });

  test("readDecision() from Headers, records and responses", () => {
    const h = new Headers({
      "x-opas-model": "big",
      "x-opas-router": "probe",
      "x-opas-decision-id": "abc",
      "x-opas-estimate": "p_success=0.880; cost_usd=0.012000; response_time_s=4.20",
      "x-opas-previous-model": "small",
    });
    const expected = {
      model: "big",
      router: "probe",
      decisionId: "abc",
      previousModel: "small",
      estimate: { pSuccess: 0.88, costUsd: 0.012, responseTimeS: 4.2 },
    };
    assert.deepEqual(readDecision(h), expected);
    assert.deepEqual(readDecision(new Response("", { headers: h })), expected);
    assert.deepEqual(readDecision({ "x-opas-model": "small" }), { model: "small" });
    const full = readDecision({
      "x-opas-model": "big",
      "x-opas-decision-id": "d9",
      "x-opas-p-success": "0.880",
      "x-opas-cost-usd": "0.014000",
      "x-opas-response-time-s": "9.80",
      "x-opas-output-tokens": "800",
      "x-opas-routing-ms": "142.5",
      "x-opas-candidates":
        "big; p_success=0.880; cost_usd=0.014000; response_time_s=9.80, small; p_success=0.450; cost_usd=0.000250; response_time_s=2.47",
    });
    assert.deepEqual(full?.estimate, { pSuccess: 0.88, costUsd: 0.014, responseTimeS: 9.8, outputTokens: 800 });
    assert.equal(full?.routingMs, 142.5);
    assert.deepEqual(full?.candidates, [
      { model: "big", pSuccess: 0.88, costUsd: 0.014, responseTimeS: 9.8 },
      { model: "small", pSuccess: 0.45, costUsd: 0.00025, responseTimeS: 2.47 },
    ]);
    assert.equal(readDecision({ "content-type": "application/json" }), undefined);
  });

  test("SDK options point at opas", () => {
    const opas = new Opas({ baseUrl: "http://opas.internal:8080/" });
    const o = opas.openai({ preference: "quality", apiKey: "k" });
    assert.equal(o.baseURL, "http://opas.internal:8080/v1");
    assert.equal(o.apiKey, "k");
    assert.deepEqual(o.defaultHeaders, { "x-opas-preference": "quality" });
    assert.equal(opas.anthropic().baseURL, "http://opas.internal:8080");
  });
});

const OPAS_BIN = process.env["OPAS_BIN"];

describe("against a real opas server", { skip: !OPAS_BIN && "set OPAS_BIN to the opas binary" }, () => {
  let upstream: Server;
  let opasProcess: ChildProcess;
  let opas: Opas;

  before(async () => {
    // Mock provider: echoes the model it received.
    upstream = await listen((_req, body, res) => {
      const b = JSON.parse(body);
      res.setHeader("content-type", "application/json");
      if (_req.url?.endsWith("/chat/completions")) {
        res.end(
          JSON.stringify({
            id: "chatcmpl_1",
            object: "chat.completion",
            created: 0,
            model: b.model,
            choices: [{ index: 0, message: { role: "assistant", content: "hello" }, finish_reason: "stop" }],
          }),
        );
        return;
      }
      res.end(
        JSON.stringify({
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: b.model,
          content: [{ type: "text", text: "hello" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
      );
    });
    const dir = mkdtempSync(join(tmpdir(), "opas-client-test-"));
    const config = join(dir, "opas.toml");
    writeFileSync(
      config,
      `[pricing]\nmodels_dev = false\n\n[[models]]\nid = "small"\ntier = "small"\nupstream_model = "small-upstream"\n` +
        `pricing = { input = 1.0, cached_input = 0.1, output = 5.0 }\n` +
        `anthropic = { base_url = "${urlOf(upstream)}/v1" }\n` +
        `openai = { base_url = "${urlOf(upstream)}/v1" }\n`,
    );
    // Find a free port for opas.
    const probe = await listen(() => {});
    const port = (probe.address() as AddressInfo).port;
    probe.close();
    opasProcess = spawn(OPAS_BIN!, ["--config", config, "--listen", `127.0.0.1:${port}`], { stdio: "ignore" });
    opas = new Opas({ baseUrl: `http://127.0.0.1:${port}` });
    for (let i = 0; i < 100 && !(await opas.health()); i++) await new Promise((r) => setTimeout(r, 100));
    assert.ok(await opas.health(), "opas did not start");
  });
  after(() => {
    opasProcess?.kill();
    upstream?.close();
  });

  test("pricing() and models()", async () => {
    const p = await opas.pricing();
    assert.equal(p.models[0]?.model, "small");
    assert.equal(p.models[0]?.source, "config");
    assert.deepEqual(
      (await opas.models()).data.map((m) => m.id),
      ["small"],
    );
  });

  test("a proxied request carries the decision headers", async () => {
    const { anthropic } = { anthropic: opas.anthropic({ apiKey: "client-key" }) };
    const res = await fetch(`${anthropic.baseURL}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": anthropic.apiKey },
      body: JSON.stringify({ model: "small", max_tokens: 5, messages: [{ role: "user", content: "hi" }] }),
    });
    assert.equal(res.status, 200);
    assert.equal(((await res.json()) as { model: string }).model, "small-upstream");
    assert.deepEqual(readDecision(res), { model: "small" });
  });

  test("route() without a router is a 400 OpasError", async () => {
    await assert.rejects(
      opas.route({ api: "openai_chat", request: { model: "auto", messages: [] } }),
      (e: unknown) => e instanceof OpasError && e.status === 400 && /no router is configured/.test(e.message),
    );
  });

  test("the server's OpenAPI document matches the one the types were generated from", async () => {
    const { readFileSync } = await import("node:fs");
    const committed = JSON.parse(readFileSync(new URL("../../../docs/openapi.json", import.meta.url), "utf8"));
    const served = await opas.openapi();
    delete (committed.info as Record<string, unknown>).version;
    delete (served["info"] as Record<string, unknown>).version;
    assert.deepEqual(served, committed);
  });

  test("the Anthropic and OpenAI SDKs work through opas", async () => {
    const { default: Anthropic } = await import("@anthropic-ai/sdk");
    const anthropic = new Anthropic(opas.anthropic({ apiKey: "client-key", preference: "cost" }));
    const { data, response } = await anthropic.messages
      .create({ model: "small", max_tokens: 5, messages: [{ role: "user", content: "hi" }] })
      .withResponse();
    assert.equal(data.model, "small-upstream");
    assert.equal(readDecision(response)?.model, "small");

    const { default: OpenAI } = await import("openai");
    const openai = new OpenAI(opas.openai({ apiKey: "client-key" }));
    const chat = await openai.chat.completions
      .create({ model: "small", messages: [{ role: "user", content: "hi" }] })
      .withResponse();
    assert.equal(chat.data.model, "small-upstream");
    assert.equal(readDecision(chat.response)?.model, "small");
  });
});
