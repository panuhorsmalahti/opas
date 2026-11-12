# LLM routers: products, APIs, and benchmarks

Research notes, compiled 2026-11-11. Details come from vendor docs and web search
results, and some are secondhand. Items marked *unverified* could not be confirmed from a primary source.

## 1. Routers

| Name | Open source | How it decides | LLM as router? |
|---|---|---|---|
| FireRouter (Fireworks AI) | No | Undisclosed classifier that scores each turn and accounts for prompt-cache loss when switching models | Unknown |
| Together Link (Together AI) | No (the CLI is MIT) | "Auto router" looks at the first task in a session and picks one model for the whole session; mechanism undisclosed | Unknown |
| Shift-1 (Applied Compute) | No | Custom 4B-parameter decision model, trained on graded coding turns, decides on every request | Yes: a fine-tuned small model |
| OpenRouter Auto Router | No | Classifies the prompt into a task type, then ranks models by community usage for that type. The original `openrouter/auto` used Not Diamond. | No (classifier type undisclosed) |
| Cursor Router (Cursor Auto mode) | No | "Compass" complexity score (0–1) with a threshold, then a task taxonomy to pick the frontier model; trained on user behaviour | Unknown (model type undisclosed) |
| Not Diamond | No | Trained meta-model that predicts quality per model | No: trained predictor |
| Martian Model Router | No | Undisclosed | Unknown |
| Azure AI Foundry Model Router | No | A "trained language model" that classifies each prompt (Microsoft says it is not itself an LLM) | Yes: a small trained classifier model |
| RouteLLM (LMSYS) | Yes | Choice of matrix factorization, BERT classifier, similarity-weighted ranking, or a fine-tuned Llama-3-8B classifier | Optional (the Llama-3-8B classifier) |
| LiteLLM | Yes | Default: local rule-based complexity scoring with no API calls. Optional: an LLM classifier that makes one small call per request. | Optional: the only prompted-LLM router in this list |
| vLLM Semantic Router | Yes | ModernBERT / embedding classifiers | No |
| Portkey AI Gateway | Yes | Rules on request metadata | No |

Notes:

- **Prompted-LLM routing is rare among these products.** LiteLLM's optional classifier
  is the only router here that prompts a general-purpose LLM to choose. The commercial
  routers instead use small models trained on outcome data, such as Shift-1's 4B
  decision model and Azure's trained classifier.
  - Their docs cite routing latency and the value of training on real outcomes, such
    as passed tests or user corrections.
  - Prompting a cheap LLM such as Haiku is mostly a do-it-yourself pattern.
- **Cache-aware routing:** FireRouter, Shift-1 and Cursor count the prompt-cache misses
  that switching models causes. Together Link avoids them by routing only once per
  session.

- **LiteLLM and Portkey are open core.** The core is MIT or Apache 2.0, and some
  enterprise features are paid. Both are mainly gateways that route by rules,
  fallbacks and load balancing, not by a trained model.
- **RouteLLM may be unmaintained.** One source puts its last commit in August 2024.
- **Martian's router may be gone.** One report from September 2026 says its router
  pages return 404 and the company now presents itself as an interpretability lab.

## 2. APIs

| Router | API format | How the router is selected and configured |
|---|---|---|
| FireRouter | OpenAI compatible, on the standard Fireworks inference endpoint | `model: "firerouter"`. Closed-model keys go in a per-request header, e.g. `x-anthropic-api-key`. |
| Together Link | Together serves OpenAI-compatible and Anthropic Messages (`/v1/messages`) endpoints. No setup details found for Link itself. | Aimed at running Claude Code and other coding agents on Together models |
| Shift-1 | No public API docs found | — |
| OpenRouter Auto | OpenAI Chat Completions. OpenRouter also serves Anthropic Messages and Open Responses endpoints. | `model: "openrouter/auto"` (deprecated, powered by Not Diamond) or `"openrouter/auto-beta"`. Wildcard filters such as `anthropic/*` limit the candidate models. |
| Cursor Router | Cursor's TypeScript and Python agent SDKs, not a raw inference API | `model: "auto-smart"`, with `optimize_for` set to `cost`, `balanced` or `intelligence` |
| Not Diamond | Its own API: `POST https://api.notdiamond.ai/v2/modelRouter/modelSelect` | Returns which model to call. The caller then makes that call itself. |
| Martian | OpenAI compatible. Its gateway is also Anthropic compatible. | Router settings live in its SDK (`RoutersClient`) |
| Azure Foundry Model Router | Deployed as a model in Azure's chat API (*unverified*) | Balanced, Cost or Quality mode |
| RouteLLM | Self-hosted OpenAI Chat Completions server (`python -m routellm.openai_server`) | The model name encodes the router and threshold, e.g. `router-mf-0.11593` |
| LiteLLM | OpenAI-compatible proxy. Anthropic `/v1/messages` and `/responses` endpoints are from memory (*unverified*). | Proxy config file |
| vLLM Semantic Router | Inspects OpenAI-format requests inside a gateway (Envoy ExtProc, agentgateway) | An embedding classifier picks from a pool of models |
| Portkey | OpenAI-compatible gateway | Conditional routing rules in the gateway config |

### Is there a common standard?

There is no standard for routing itself. For the request and response format there is
a de facto standard and one formal spec:

1. **OpenAI Chat Completions is the de facto standard.** Almost every router accepts
   it. One survey found that 105 of 120 "OpenAI-compatible" providers expose
   `/chat/completions`.
2. **Anthropic Messages (`/v1/messages`) comes second.** Coding-focused routers (Together,
   OpenRouter, Martian) add it so tools like Claude Code can switch by
   changing only the base URL.
3. **Open Responses is the only formal spec.** It is based on OpenAI's Responses API.
   - It launched in January 2026 with OpenRouter, Hugging Face, Vercel, Ollama, LM
     Studio and vLLM. It reportedly moved to the Linux Foundation's Agentic AI
     Foundation later; the move date is *unverified*.
   - Anthropic and Google are not partners. Only about 26 of those 120 providers
     expose `/responses`.

How the routing itself is controlled differs between products. Two patterns are common:

- **Router as a model name:** a special model ID is put in a normal request, and the
  router proxies the call. Most products work this way.
- **Recommendation only:** the router returns or injects a model choice, and something
  else makes the call. Not Diamond and vLLM Semantic Router work this way.

Router settings are passed in the model name, in a request parameter or in a header.
Several products offer similar cost / balanced / quality presets (Cursor, Azure), but no spec defines them.

## 3. Routing benchmarks

### Academic benchmarks

| Benchmark | Year | What it contains | Notes |
|---|---|---|---|
| [RouterBench](https://arxiv.org/abs/2403.12031) | 2024 | 405k+ precomputed outputs from 11 LLMs on 7–8 tasks (MMLU, GSM8K, MBPP, MT-Bench and others), with cost | The earliest widely used one. Criticized for old models and easy datasets. |
| RouteLLM eval suite | 2024 | MT-Bench, MMLU and GSM8K, one strong and one weak model | Measures quality recovered against the share of calls sent to the strong model (APGR / CPT). Details from memory, *unverified*. |
| EmbedLLM | 2024 | Model-embedding method that also ships evaluation data | More a routing method or baseline than a benchmark |
| RouterEval | 2025 | Results of about 8,500 LLMs on 12 benchmarks | Open models only, no inference cost |
| [RouterArena](https://arxiv.org/abs/2510.00202) (ICLR 2026) | 2025–26 | Dataset spanning many domains and difficulty levels. Metrics: accuracy, cost efficiency, routing optimality, robustness, latency. | Has a [live leaderboard](https://routeworks.github.io/) and includes commercial routers (Azure, GPT-5 auto, vLLM-SR, RouteLLM). Scores changed between paper versions. |
| [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) (Findings of ACL 2026) | 2026 | 400k+ instances, 21 datasets, 33 models, with cost. Adapters for 10 routing baselines. | Its authors report that many routers don't reliably beat simple baselines. |
| [CodeRouterBench](https://arxiv.org/pdf/2606.22902) (Agent-as-a-Router paper) | 2026 | About 10k coding tasks in 9 dimensions, plus a 176-task out-of-distribution set (SWE-bench Verified, LongCLI-Bench, FeatureBench, SWE-CI) run with mini-swe-agent | Coding-agent routing, scored by performance per dollar |
| SWE-Router (arXiv 2607.00053) | 2026 | Routing in multi-turn agentic software engineering | ICML 2026 workshop paper |

### Industry and vendor benchmarks

- **[OpenRouter Router Benchmarks](https://openrouter.ai/benchmarks/routers)**
  ([announcement](https://openrouter.ai/blog/announcements/model-router-benchmarks/),
  reportedly 2026-10-02)
  - Runs commercial routers on benchmarks from varied domains and combines quality,
    speed and cost into a 0–10 "Router Index".
  - Covers OpenRouter Auto and Auto Beta, Fugu, Jev Router and others. Unbiased Pareto
    and NVIDIA Switchyard are included per a secondary source.
- **Vendor-run coding comparisons:** Fireworks, Together and Applied Compute
  each publish their own head-to-head numbers.
  - Common task sets: SWE-bench Verified, DeepSWE, SWE Atlas, Terminal-Bench.
  - Common baselines: the frontier model alone and random routing at the same cost.
  - These are self-reported and not independently reproduced.

### Common metrics

- **Quality vs. cost Pareto curve:** accuracy or pass rate plotted against $ per task.
  Summary numbers include area under the curve and the AIQ score from RouterBench.
- **Quality recovered at a given strong-model share:** for example, "95% of GPT-4
  quality with 26% of calls sent to GPT-4" (RouteLLM's APGR / CPT).
- **Gap to the oracle:** the oracle router always picks the cheapest model that
  succeeds. Some benchmarks report how close a router gets to it (routing optimality).
- **Comparison with baselines:** the best single model, the cheapest model, and random
  routing at the same cost. Several 2026 papers find many routers barely beat random
  routing.
- **Routing overhead:** added latency, and cache misses caused by switching models.

### Takeaways

- No single benchmark is accepted as the standard. RouterBench is the most cited
  academic benchmark.
  - RouterArena and OpenRouter's index are the only ones that rank commercial routers.
  - LLMRouterBench has the largest and most up-to-date model pool.
- Coding-agent routing, which is where the 2026 commercial products compete, has no
  shared benchmark yet.
  - Vendors test on SWE-bench-family and Terminal-Bench tasks of their own choosing.
  - CodeRouterBench and SWE-Router are early attempts at a shared one.

## Sources

### Routers and APIs

- [Fireworks FireRouter docs](https://docs.fireworks.ai/ecosystem/firerouter/overview)
- [Fireworks: Introducing FireRouter with Opus](https://fireworks.ai/blog/introducing-firerouter-with-opus)
- [Together Messages API](https://docs.together.ai/docs/inference/messages-api)
- [Applied Compute: agentic model router](https://www.appliedcompute.com/research/agentic-model-router)
- [OpenRouter Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router)
- [OpenRouter Anthropic Messages](https://openrouter.ai/docs/api/api-reference/anthropic-messages/create-a-message.md)
- [Cursor Router docs](https://cursor.com/docs/cursor-router)
- [How Cursor Router works](https://cursor.com/blog/how-cursor-router-works)
- [Not Diamond modelSelect API](https://docs.notdiamond.ai/reference/api-modelselect)
- [RouteLLM on GitHub](https://github.com/lm-sys/RouteLLM)
- [Martian router docs](https://docs.withmartian.com/martian-model-router/getting-started/benchmarking-the-router)
- [vLLM Semantic Router via agentgateway](https://agentgateway.dev/docs/standalone/latest/integrations/llm/routing/vllm-semantic-router/)
- [LLM Reference router index](https://www.llmreference.com/routers)
- [ParityLayer: best LLM routers 2026](https://paritylayer.com/blog/best-llm-routers-2026)
- [ClawRouters: free LLM routers re-verified](https://www.clawrouters.com/blog/best-ai-model-router-free-2026)

### Routing mechanisms

- [Fireworks: FireRouter with Opus](https://fireworks.ai/blog/introducing-firerouter-with-opus)
- [Together Link](https://www.together.ai/link)
- [MarkTechPost on Together Link](https://www.marktechpost.com/2026/10/05/meet-together-link-a-free-cli-that-runs-open-models-like-kimi-k3-and-glm-5-3-inside-claude-code-codex-and-opencode/)
- [Applied Compute Shift-1](https://www.appliedcompute.com/platform/shift-1)
- [Azure model router: how it works](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-router-how-it-works)
- [OpenRouter: new Auto Router](https://openrouter.ai/blog/announcements/happy-new-year-introducing-a-new-auto-router/)
- [LiteLLM auto routing](https://docs.litellm.ai/docs/proxy/auto_routing)
- [Morph: Not Diamond alternative (secondary source)](https://www.morphllm.com/notdiamond-alternative)

### API standards

- [Open Responses](https://www.openresponses.org/)
- [Simon Willison on Open Responses](https://simonwillison.net/2026/Jan/15/open-responses/)
- [Digital Applied: OpenAI-compatible API standard](https://www.digitalapplied.com/blog/openai-compatible-api-standard-provider-agnostic-2026)

### Benchmarks

- [RouterArena paper](https://arxiv.org/html/2510.00202v3)
- [RouterArena leaderboard](https://routeworks.github.io/)
- [LLMRouterBench paper](https://arxiv.org/html/2601.07206v1)
- [Survey: dynamic model routing and cascading](https://arxiv.org/pdf/2603.04445)
- [Agent-as-a-Router (CodeRouterBench)](https://arxiv.org/pdf/2606.22902)
- [Scrouting](https://arxiv.org/pdf/2608.04804)
- [OpenRouter: Model Router Benchmarks](https://openrouter.ai/blog/announcements/model-router-benchmarks/)
