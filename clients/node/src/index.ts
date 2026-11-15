/**
 * Node.js client for opas, a model router for LLM requests.
 *
 * - `Opas#route()` asks opas which model a request should go to, with every candidate's
 *   predicted success probability, cost and response time, without sending it.
 * - `Opas#openai()` / `Opas#anthropic()` return options for the official OpenAI and
 *   Anthropic SDKs, so their requests go through opas and get routed.
 * - `readDecision()` reads the routing decision from a proxied response's headers.
 *
 * The types come from opas's OpenAPI document (`src/schema.ts`, generated).
 */
import type { components } from "./schema.js";

type Schemas = components["schemas"];
export type ApiFormat = Schemas["ApiFormat"];
export type Candidate = Schemas["Candidate"];
export type Conversation = Schemas["Conversation"];
export type Estimate = Schemas["Estimate"];
export type Preference = Schemas["Preference"];
export type Pricing = Schemas["Pricing"];
export type PricingResponse = Schemas["PricingResponse"];
export type ResolvedPrice = Schemas["ResolvedPrice"];
export type RouteRequest = Schemas["RouteBody"];
export type RouteResponse = Schemas["RouteResponse"];
export type Speed = Schemas["Speed"];
export type Tier = Schemas["Tier"];
export type Weights = Schemas["Weights"];
export type ModelList = Schemas["OpenAiModelList"];

/** The default address of a local opas server. */
export const DEFAULT_BASE_URL = "http://127.0.0.1:8080";

export interface OpasOptions {
  /** Where opas listens. Default: `$OPAS_URL`, else `http://127.0.0.1:8080`. */
  baseUrl?: string;
  /** A `fetch` implementation; defaults to the global one (Node 18+). */
  fetch?: typeof fetch;
  /** Headers sent with every request to opas's own endpoints. */
  headers?: Record<string, string>;
  /** Abort requests to opas's own endpoints after this many milliseconds (default 30 000). */
  timeoutMs?: number;
}

/** Per-request routing options, sent as `x-opas-*` headers on proxied requests. */
export interface RoutingOptions {
  /** Quality / speed / cost weights; only the ratios matter. */
  weights?: Partial<Weights>;
  /** A preset instead of `weights`. */
  preference?: Preference;
  /** Prices by model id, USD per million tokens. */
  pricing?: Record<string, Pick<Pricing, "input" | "output"> & Partial<Pick<Pricing, "cached_input">>>;
}

/** An error response from opas (or a network failure reaching it). */
export class OpasError extends Error {
  /** HTTP status, or 0 when opas could not be reached. */
  readonly status: number;
  /** The parsed error body, when there was one. */
  readonly body: unknown;

  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.name = "OpasError";
    this.status = status;
    this.body = body;
  }
}

/** The routing decision opas made for a proxied request, from its response headers. */
export interface Decision {
  /** The configured model that served the request. */
  model: string;
  /** `probe` or `affinity`; absent when the request named a model directly. */
  router?: string;
  decisionId?: string;
  /** Predictions for the chosen model (routed requests with the probe router). */
  estimate?: { pSuccess?: number; costUsd?: number; responseTimeS?: number };
  /** The model that served the previous turn, when opas recognized the conversation. */
  previousModel?: string;
}

type HeaderSource =
  | Headers
  | Record<string, string | string[] | undefined>
  | { headers: Headers | Record<string, string | string[] | undefined> };

/**
 * Read opas's decision from a proxied response: a `Response`, a `Headers` object, or a
 * plain header record (as the OpenAI and Anthropic SDKs expose via `.withResponse()`).
 * Returns `undefined` when the response did not come through opas.
 */
export function readDecision(source: HeaderSource): Decision | undefined {
  const raw = "headers" in source && !(source instanceof Headers) ? source.headers : source;
  const get = (name: string): string | undefined => {
    if (raw instanceof Headers) return raw.get(name) ?? undefined;
    const record = raw as Record<string, string | string[] | undefined>;
    const v = record[name] ?? record[name.toLowerCase()];
    return Array.isArray(v) ? v[0] : v;
  };
  const model = get("x-opas-model");
  if (!model) return undefined;
  const decision: Decision = { model };
  const router = get("x-opas-router");
  if (router) decision.router = router;
  const decisionId = get("x-opas-decision-id");
  if (decisionId) decision.decisionId = decisionId;
  const previous = get("x-opas-previous-model");
  if (previous) decision.previousModel = previous;
  const estimate = get("x-opas-estimate");
  if (estimate) {
    const fields = Object.fromEntries(
      estimate.split(/[;,]\s*/).map((kv) => {
        const [k, v] = kv.split("=");
        return [k?.trim(), Number(v)];
      }),
    ) as Record<string, number>;
    decision.estimate = {
      pSuccess: fields["p_success"],
      costUsd: fields["cost_usd"],
      responseTimeS: fields["response_time_s"],
    };
  }
  return decision;
}

/** `x-opas-*` headers for routing options. */
export function routingHeaders(options: RoutingOptions = {}): Record<string, string> {
  const h: Record<string, string> = {};
  if (options.weights) {
    h["x-opas-weights"] = Object.entries(options.weights)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${k}=${v}`)
      .join(",");
  }
  if (options.preference) h["x-opas-preference"] = options.preference;
  if (options.pricing) {
    h["x-opas-pricing"] = Object.entries(options.pricing)
      .map(([model, p]) =>
        p.cached_input === undefined
          ? `${model}=${p.input}/${p.output}`
          : `${model}=${p.input}/${p.cached_input}/${p.output}`,
      )
      .join("; ");
  }
  return h;
}

/** A client for one opas server. */
export class Opas {
  readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;

  constructor(options: OpasOptions = {}) {
    const base = options.baseUrl ?? process.env["OPAS_URL"] ?? DEFAULT_BASE_URL;
    this.baseUrl = base.replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    if (!this.fetchImpl) throw new Error("opas-client needs a global fetch (Node 18+) or options.fetch");
    this.headers = options.headers ?? {};
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  /**
   * Decide which model a request should go to, without sending it. `request` is a body in
   * the `api` format, exactly as you would send it to the provider.
   */
  route(body: RouteRequest): Promise<RouteResponse> {
    return this.call<RouteResponse>("POST", "/v1/route", body);
  }

  /** Prices in effect for the configured models, and where each came from. */
  pricing(): Promise<PricingResponse> {
    return this.call<PricingResponse>("GET", "/v1/pricing");
  }

  /** The router model (`auto`) and the configured models. */
  models(): Promise<ModelList> {
    return this.call<ModelList>("GET", "/v1/models");
  }

  /** Whether the server is up. */
  async health(): Promise<boolean> {
    try {
      const res = await this.fetchImpl(`${this.baseUrl}/healthz`, {
        signal: AbortSignal.timeout(Math.min(this.timeoutMs, 5_000)),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  /** The server's OpenAPI document. */
  openapi(): Promise<Record<string, unknown>> {
    return this.call<Record<string, unknown>>("GET", "/openapi.json");
  }

  /**
   * Options for the OpenAI SDK (`new OpenAI(opas.openai())`): requests go to opas's
   * OpenAI-compatible endpoints. Use `model: "auto"` to have opas choose.
   */
  openai(options: RoutingOptions & { apiKey?: string } = {}): {
    baseURL: string;
    apiKey: string;
    defaultHeaders: Record<string, string>;
  } {
    return {
      baseURL: `${this.baseUrl}/v1`,
      // opas replaces the client key when it holds a provider key itself.
      apiKey: options.apiKey ?? process.env["OPENAI_API_KEY"] ?? "opas-local",
      defaultHeaders: { ...routingHeaders(options) },
    };
  }

  /**
   * Options for the Anthropic SDK (`new Anthropic(opas.anthropic())`): requests go to
   * opas's Messages endpoint. Use `model: "auto"` to have opas choose.
   */
  anthropic(options: RoutingOptions & { apiKey?: string } = {}): {
    baseURL: string;
    apiKey: string;
    defaultHeaders: Record<string, string>;
  } {
    return {
      baseURL: this.baseUrl,
      apiKey: options.apiKey ?? process.env["ANTHROPIC_API_KEY"] ?? "opas-local",
      defaultHeaders: { ...routingHeaders(options) },
    };
  }

  private async call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
    let res: Response;
    try {
      res = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers: {
          accept: "application/json",
          ...(body === undefined ? {} : { "content-type": "application/json" }),
          ...this.headers,
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (e) {
      throw new OpasError(`could not reach opas at ${this.baseUrl}: ${(e as Error).message}`, 0);
    }
    const text = await res.text();
    let parsed: unknown = text;
    try {
      parsed = text ? JSON.parse(text) : undefined;
    } catch {
      // not JSON; keep the text
    }
    if (!res.ok) {
      const message =
        (parsed as { error?: { message?: string } } | undefined)?.error?.message ??
        (typeof parsed === "string" && parsed ? parsed : res.statusText);
      throw new OpasError(`opas ${method} ${path}: ${res.status} ${message}`, res.status, parsed);
    }
    return parsed as T;
  }
}

export default Opas;
