# Estimating request complexity from model activations

Research notes for opas, compiled 2026-11-11.

**Question:** can a small open-weights model read a request and, without generating any
tokens, tell us how hard it is? "Hard" here means how likely each candidate model is to
succeed, and how much the request will cost. If so, we can route on that signal.

**Short answer: yes.** The signal comes from the model's hidden states after the
prefill (prompt-processing) pass.

- Several 2026 papers show that a simple probe on those hidden states predicts success
  and difficulty, usually better than surface features or sentence embeddings.
- Gains are not universal. Prompt length or TF-IDF sometimes win, so baselines must
  always be measured alongside the probe.

## 1. Model choice: Qwen3.5-9B

The table compares the two strongest current models in the 7–9B class. All numbers are
vendor-reported, each on its own evaluation setup, so gaps under about 3 points are noise.

| Benchmark | Qwen3.5-9B | Granite 4.2 8B |
|---|---|---|
| MMLU-Pro | **82.5** | 74.0 |
| GPQA Diamond | **81.7** | 64.1 |
| HMMT Feb 25 | **83.2** | 78.3 |
| BFCL-V4 (tool calling) | **66.1** | 52.4 |
| LiveCodeBench v6 | 65.6 | **73.2** |
| IFBench | 64.5 | **79.3** |

Sources: the [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and the
[Granite 4.2 8B model card](https://huggingface.co/ibm-granite/granite-4.2-8b). Other
candidates checked:

- **Ministral 3 8B and Gemma 4 E4B:** no comparable numbers.
  [BenchLM](https://benchlm.ai/best/small-language-models) puts Gemma 4 E4B and
  Granite 4.2 8B level, and doesn't list Qwen3.5-9B.
- **Llama 3.1 8B and Qwen3-8B:** older and weaker.
- **Larger Qwen releases:** Qwen 3.6 and 3.8 (2026) have no dense model in the 7–9B
  size, so Qwen3.5-9B (Feb 2026) is the newest Qwen in this class.

**Why Qwen3.5-9B for this job:**

- **Strongest on knowledge and reasoning.** It leads clearly on MMLU-Pro and GPQA, and
  understanding what a request needs is mostly knowledge and reasoning. Granite is
  better at writing code, but we never ask the probe model to generate anything.
- **The research points the same way.** In the NVIDIA prefill-activation paper below,
  Qwen3.5 encoders gave the best routing signal (Qwen3.5-122B was best on every
  target).
- **Interpretability tooling exists.** Qwen publishes sparse autoencoders for
  Qwen3.5-9B-Base (`Qwen/SAE-Res-Qwen3.5-9B-Base-*`), which could later give
  interpretable features instead of raw directions.
- **Apache-2.0 licence.**

**Caveats:**

- **It is 9B, not 7B.** Nothing current at exactly 7B beats it.
- **It is a hybrid architecture.** It has 32 layers in a repeating pattern: three
  Gated DeltaNet (linear attention) layers, then one full-attention layer. The residual
  stream that we probe is the same as in any transformer, but fewer runtimes support
  the architecture (see section 5).
- **It is multimodal.** The vision tower is dropped when loading text-only.
- **Fallback:** if runtime support becomes the blocker, Granite 4.2 8B is a plain
  dense transformer (40 layers, hidden size 4096) and the next-best choice.

**Installed locally** (in `models/`, which git ignores):

- **Main model:** `lmstudio-community/Qwen3.5-9B-GGUF`, file `Qwen3.5-9B-Q8_0.gguf`
  (9.5 GB, text-only), at `models/gguf/`. The Rust encoder runs it in-process through
  llama.cpp (section 4).
- **Why Q8_0:** bf16 is 18 GB, and the disk is nearly full. 8-bit quantization barely
  changes hidden states.
- **Test models:** `Qwen3.5-0.8B-Q8_0.gguf` for tests, and
  `models/Qwen3.5-0.8B-MLX-4bit` as the Python reference for cross-checking.
- **Not kept:** an MLX 8-bit build of the 9B was started for the Python path, then
  dropped when model execution moved to Rust.

## 2. Evidence from the literature

### Prefill activations predict the model's own success

[LLMs Encode Their Failures (arXiv 2602.09924)](https://arxiv.org/abs/2602.09924), with
code at [KabakaWilliam/llms_know_difficulty](https://github.com/KabakaWilliam/llms_know_difficulty)
(MIT licence).

**Setup:**

- **Features:** residual-stream hidden states at the end of the chat template, i.e.
  the token(s) just before the model would start answering (e.g.
  `<|im_start|>assistant\n`). Layer and position are swept.
- **Probes:** L2 logistic regression for success/failure, and ridge regression for
  continuous difficulty, tuned with 5-fold cross-validation.
- **Baselines:** prompt length, TF-IDF, and a small MLP probe.

**Results (AUROC unless noted):**

- **Math:** Qwen2.5-Math-1.5B scores 0.84 for the linear probe, against 0.64 for
  TF-IDF and 0.61 for length. Qwen2.5-Math-7B scores 0.79 against 0.68 and 0.67.
- **Code:** on LiveCodeBench, Qwen2.5-Coder-7B scores 0.90 against 0.84 (TF-IDF) and
  0.61 (length).
- **Human-rated difficulty** (IRT scores on AMC problems): Spearman 0.83–0.87 for the
  probe, against 0.72–0.74 for TF-IDF.
- **Routing a math pool by probe:** up to 70% lower cost on MATH at equal or better
  accuracy.
- **Where the probe loses:**
  - TF-IDF wins on AIME (0.84 against 0.77).
  - Length beats the linear probe for Qwen2.5-1.5B.
  - GPT-OSS-20B on high reasoning falls below chance on AIME.

**Takeaway:** the signal is real, but always compare against cheap baselines.

### Another model's activations can predict a target model's success

[LLM Router: Rethinking Routing with Prefill Activations (NVIDIA, arXiv 2603.20895)](https://arxiv.org/abs/2603.20895).

**Method:**

- **Encoder-target decoupling.** One open-weights model (the "encoder") supplies the
  activations, and a probe predicts whether *other* models, including closed ones,
  will answer correctly. In several cases another model's activations beat the target
  model's own.
- **Features:** last prefill token. Layers searched over the upper half of the network.
- **Layer choice:** layers are ranked by a Fisher separability score on the
  correct/incorrect classes. This was the best practical selector in 16 of 18
  model-pool and encoder combinations, within 0.0003 AUC of an oracle choice.
  Concatenating several layers did not help.
- **Preprocessing:** a scaler, then PCA to 100 dimensions.
- **Router ("SharedTrunkNet"):** a multi-output MLP that predicts P(correct) for every
  candidate model at once. Trained with BCE loss, 85/15 split, early stopping, averaged
  over an ensemble of 5 seeds.
- **Decision:** routing score `λ·p̂ − (1−λ)·cost`. Cost uses each model's median output
  length.

**Results:**

- **AUROC versus the best embedding-based router:** 0.856 against 0.804 on a frontier
  pool, 0.826 against 0.760 on a pool of 7–9B models, and 0.882 against 0.807 on a
  mixed pool.
- **Routing:** on the frontier pool, it closes 45.6% of the gap between the best single
  model and an oracle router, at 74.3% lower cost than the most expensive model.
- **Overhead:** running the encoder added 0.76% to routed cost and 0.79% to latency.

**Caveats:**

- The encoders were 20B–122B, so the paper does not test a 7–9B encoder.
- No code is released.

### Probes read capability requirements that the model can't verbalize

[Structured but Silent (arXiv 2610.08018)](https://arxiv.org/html/2610.08018v1):

- **Setup:** the hidden state at the final prompt token, fed into L2 logistic
  regression (C = 1). Tested on Llama 3.1 8B, Qwen3-8B, Gemma 2 9B and Mistral 7B.
- **What is predicted:** three query properties, giving 8 classes:
  - whether answering needs external information;
  - whether it transforms content or just retrieves it;
  - whether it changes something in the world.
- **Accuracy:** probes reach 92.5–94.8% joint accuracy, against about 84% for E5
  embeddings, n-grams and bag-of-words.
- **Adversarial queries:** on queries written to mislead, probes beat the baselines by
  20–24 points.
- **The model can't say it:** asked directly with 4-shot prompting, the same models
  managed only 54–69%. The information is in the activations, not in what the model
  says.

**Takeaway:** 7–9B models clearly encode properties of a query that matter for
routing, such as "needs tools" or "needs fresh information".

### Related work

- [DiffAdapt (arXiv 2510.19669)](https://arxiv.org/pdf/2510.19669): a small MLP on the
  last-layer hidden state after prefill classifies a question as easy, normal or hard.
  It notes the step doesn't disrupt batching or KV caching.
- **Probes on the first few reasoning tokens**
  ([Temporal Predictors of Outcome in Reasoning Language Models, arXiv 2511.14773](https://arxiv.org/pdf/2511.14773)):
  they also predict correctness, but that requires starting generation.
- **Benchmarks:** [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) has
  per-prompt, per-model outcomes, and is the training data the NVIDIA paper used.

## 3. Recipe for opas

### 3.1 What to predict

For each candidate model m:

- **Success, p̂_m(x).** Did the model solve the task? Check this with a verifier: tests
  passing, an exact-match answer, or a judge.
- **Cost, ĉ_m(x).** Output tokens, dollars or agent steps.
  Predict `log(cost)`, because cost is heavy-tailed.

Then route on `argmax_m p̂_m − λ·ĉ_m`, with λ exposed as presets (cost / balanced /
quality). Random routing is the baseline every probe router must beat at equal cost.

### 3.2 Training data

You need labelled examples: (request, model, success, cost) for every candidate model.
Options, cheapest first:

1. **Public routing datasets.** LLMRouterBench and RouterBench already have per-prompt
   outcomes for many models. They are fine for validating the pipeline, but their model
   pools are older than our candidates.
2. **Generate our own.** Run each candidate model on verifiable tasks: SWE-style bug
   fixes, LiveCodeBench, Terminal-Bench-style tasks, plus Q&A with known answers.
   Record pass/fail and cost.
   - **How much:** a few thousand tasks per model is what the papers used (1.6k to 14k).
     Sample each task several times for a smoother success rate. The probe paper used
     K = 5–50 samples per prompt.
   - **Cost:** the expensive part is running the candidate models, not extracting
     activations.
3. **Production traffic.** Log the decision id with outcome signals, such as whether the
   user retried or corrected, or whether tests passed. This is how Cursor trains its
   router. The decision ids in opas's headers are there for this.

### 3.3 Features

- **Activations.** `opas-encode` runs Qwen3.5-9B over the chat-templated request
  and saves, for every layer, the hidden state at the last prompt token and the mean
  over all tokens. Last token is the default in every paper. Mean pooling is kept for
  comparison.
- **Cheap features, always included as baselines:** token count, turn count, whether
  tools are present, system-prompt length, and TF-IDF of the last user message.
- **Optional:** structured attributes extracted by a small LLM (scope, domain,
  ambiguity).

### 3.4 Training the probe

1. **Normalize, then PCA.** Standardize per layer, then PCA to about 100 dimensions,
   fitted on training data only. *Outcome:* with strong L2 regularization, skipping PCA
   and using all 4096 dimensions ranked difficulty better (see
   [predictor-capacity.md](predictor-capacity.md)), so the shipped probe has no PCA.
2. **Pick a layer.**
   - Sweep layers in the upper half (16–32) with grouped cross-validation, and pick by
     validation AUROC. Use Fisher separability as a cheap pre-filter.
   - Expect a mid-to-late peak with a drop at the very top.
3. **Probes.**
   - Logistic regression per model for success.
   - Ridge regression on `log(cost)` for cost.
   - Then try one multi-output MLP across all candidates (the SharedTrunkNet idea).
4. **Calibrate.** Apply Platt or isotonic scaling, so that p̂ can be traded against cost.
5. **Split by benchmark, not at random.** Use leave-one-benchmark-out cross-validation.
   A random split rewards memorizing which dataset a task came from (here, dataset
   identity alone reaches 0.79 AUROC).

### 3.5 Evaluation

- **Probe quality:**
  - AUROC and Brier score per model for success.
  - Spearman correlation for cost.
- **Routing quality:**
  - The quality-vs-cost curve against random routing at matched cost.
  - The share of the gap to an oracle router that is closed.
  - Behaviour on a held-out benchmark.

### 3.6 Agentic and multi-turn requests

- **What to feed in:** the system prompt, the tool definitions, and the conversation.
  Qwen's chat template accepts a `tools` argument; `extract.py` doesn't pass tools yet.
- **Long contexts:** truncate from the front, so the most recent turns survive.
- **Repeated prefixes:** in agent loops the system prompt and tools repeat on every
  turn, so cache the prefix in the probe model the same way providers cache prompts.
- **Per-turn routing:** if the router may switch models between turns, include the cost
  of losing the upstream prompt cache, as FireRouter, Shift-1 and Cursor do.

## 4. What extraction looks like (implemented)

### In Rust: `crates/opas-encoder` (main path)

The model runs in-process through llama.cpp, using the raw `llama-cpp-sys-2` bindings
pinned at 0.1.160. The safe `llama-cpp-2` wrapper does not expose what we need.

1. **Format.** Render the request as Qwen ChatML from opas's `RouteRequest`, so the
   router and encoder see the same parsed request. The output is byte-identical to the
   Hugging Face chat template for text conversations (golden tests in `chat.rs`). Tool
   definitions are not rendered yet.
2. **Tokenize** with llama.cpp. No BOS token is added, matching the HF template.
3. **Prefill.** Run a single `llama_decode` with no sampling, so nothing is generated.
   The context's `cb_eval` graph callback asks to observe the tensors
   `model.input_embed` (index 0) and `l_out-<i>` (the residual stream after layer *i*,
   index *i*+1).
4. **Capture.** As each requested tensor is computed, the callback copies it off the
   GPU (`ggml_backend_tensor_get`) and accumulates two pools: the final prompt token,
   and the sum for the mean.
   - Long prompts are split into 512-token micro-batches. They arrive in order, so the
     accumulation spans them.
   - Every captured layer is checked to have seen exactly `n_tokens` rows.
5. **Choose layers.** `EncoderConfig::layers` captures only selected layers. That is the
   production setting once the probe layer is chosen, and fewer callbacks mean fewer
   GPU sync points.

```sh
cargo build --release -p opas-encoder
./target/release/opas-encode --demo                       # 9B by default
./target/release/opas-encode --input requests.jsonl --out training/out/acts
./target/release/opas-encode --model models/gguf/Qwen3.5-0.8B-Q8_0.gguf --demo --print-prompt
```

The input format is one JSON object per line. Each line is either `{"prompt": ...}` or
`{"api": "openai_chat" | "anthropic_messages", "request": {...}}`, so requests captured
by the opas server can be replayed.

The output is a directory of `.npy` files that probe training reads with `np.load`:

- `last.npy` and `mean.npy`, each `[n, layers, hidden]`;
- `n_tokens.npy` and `seconds.npy`;
- `meta.json`.

As a library:

```rust
let mut enc = Encoder::load("models/gguf/Qwen3.5-9B-Q8_0.gguf", EncoderConfig::default())?;
let acts = enc.encode_request(&route_request)?;   // acts.last[k], acts.mean[k] for acts.layers[k]
```

**Tests:** `crates/opas-encoder/tests/model.rs` runs against a real model when
`OPAS_TEST_MODEL` is set. It checks:

- every layer is captured, with finite values;
- repeated runs are deterministic;
- capturing a subset of layers gives the same vectors as a full capture;
- prompts longer than one micro-batch (1,500 tokens) accumulate correctly and are
  truncated to `max_tokens`;
- thinking and non-thinking prompts end at different positions.

### Qwen3.5-9B run (Q8_0, M3 Pro, Metal)

| | Result |
|---|---|
| Load time | 5.7 s; 32 layers, hidden size 4096, 33 hidden states per prompt |
| Model tests (`tests/model.rs`) | all 4 pass |
| Demo prompts (17–53 tokens) | 184–477 ms each |
| 20-token prompt, plain prefill | 164 ms (1 layer captured: 163 ms; all layers: 184 ms) |
| 531-token prompt, plain prefill | 2.25 s (1 layer: 2.24 s; all layers: 2.34 s), about 235 tokens/s |

**Similarity check:** on the demo prompts, at the middle layer, the two trivial prompts
are most alike (0.93), as are the two engineering prompts (0.94). The trivial-vs-
engineering pairs score 0.78–0.83, a wider spread than on the 0.8B (0.85–0.97).

**What this means for routing:**

- **Capture cost is negligible;** prefill speed is the constraint. At about 235 tokens/s
  on a laptop GPU, a short chat turn adds about 0.2 s, but a 30k-token agent context
  would take minutes. Ways to get it down:
  1. **Truncate:** probe only the most recent part of the conversation (`max_tokens`,
     e.g. 1–2k tokens). The routing decision mostly depends on the current task.
  2. **Reuse prefixes across turns:** an agent conversation grows by appending, so keep
     the model state and prefill only the new tokens. Mean pooling then needs running
     sums carried across turns. Qwen3.5's recurrent (DeltaNet) state can't be partially
     rolled back, so the cache must be strictly append-only.
  3. **Early exit:** stop the forward pass at the probe layer (see above).
  4. **A smaller encoder:** Qwen3.5-4B or -2B. Measure routing quality with each.
  5. **Server GPUs:** not yet measured; see [hardware](usage/hardware.md).

### Python reference: `training/reference/` (MLX)

The same extraction in Python, kept as an independent cross-check of the Rust encoder (Apple Silicon only;
see `training/reference/README.md`). It runs the mlx-lm model layer by layer, and `selfcheck.py` confirms the loop reproduces the model's
own forward pass (max logit difference 0.0).

### Cross-check: Rust (llama.cpp, Q8_0) against Python (MLX, 4-bit), Qwen3.5-0.8B

`training/reference/compare.py` produced these numbers on the 5 demo prompts:

| Check | Result |
|---|---|
| Token counts | identical (18, 17, 35, 50, 53) |
| Cosine similarity, mean pooling, per layer | 0.97–0.998 |
| Cosine similarity, last token, per layer | 0.996 at the embeddings, falling to 0.82 at the final layer |
| Agreement of prompt-to-prompt similarity structure | correlation 0.93–0.998 |

The fall-off with depth on last-token vectors fits quantization error building up
through the layers (4-bit MLX against 8-bit GGUF). Mean pooling averages that noise out.
The two runtimes agree on what the activations encode.

Speed on the 0.8B model (M3 Pro, Metal) was 30–50 ms per demo prompt, with occasional
one-off spikes while GPU pipelines compiled for new shapes. 9B timings: see "Qwen3.5-9B run" below.

**Capture overhead** (`cargo run --release -p opas-encoder --example overhead`,
0.8B Q8_0 on an M3 Pro, median of 10 runs):

| Captured | 20-token prompt | 531-token prompt |
|---|---|---|
| nothing (plain prefill) | 24 ms | 277 ms |
| 1 layer | 24 ms | 262 ms |
| all 25 hidden states | 32 ms | 266 ms |

- **Capture is nearly free.** Even capturing all 25 hidden states costs at most about
  8 ms; differences under about 15 ms are run-to-run noise.
- **The prefill itself is the cost.** MLX (4-bit) runs the same 530-token prompt in
  170 ms, so llama.cpp is about 1.5× slower here, with 8-bit weights. Both are dominated
  by the Gated DeltaNet layers.

### Thinking mode: pin it

The default when `enable_thinking` is unset **differs by model size**:

- **Qwen3.5-0.8B** defaults to non-thinking, ending the prompt with a closed block:
  `<think>\n\n</think>\n\n`.
- **Qwen3.5-9B** defaults to thinking, ending with `<think>\n`.
- Otherwise the two official templates are identical; the diff is only this branch.

mlx-lm's tokenizer wrapper also turns thinking on by itself.

Both implementations now set the mode explicitly. The default is **thinking**: the prompt
ends at `<|im_start|>assistant\n<think>\n`, the point where the model would start
reasoning. That is the R1-style end-of-input position probed in arXiv 2602.09924.
`--non-thinking` switches it.

The final tokens differ between the modes, and so do the features. Training and serving
must use the same mode.

### Early exit

In production only layers up to the chosen probe layer are needed. If the best layer is
20 of 32, the rest of the forward pass is wasted work, roughly 35% of the compute.

llama.cpp's `llama_model_params.prune_layers` can drop layers at load time. Whether that
works cleanly with Qwen3.5's mixed layer types still needs testing.

## 5. Integrating with opas

The probe itself is cheap: a PCA projection and a dot product, which is trivial in Rust.
The work is in running the 9B model. Options considered:

| Option | Status |
|---|---|
| **llama.cpp in-process** (chosen) | Implemented in `opas-encoder`. GGUF, Metal, CPU and CUDA backends, all in one binary. The `cb_eval` capture works for Qwen3.5. |
| Python sidecar (MLX / vLLM) | Not needed for serving. The Python MLX implementation is kept in `training/reference/` to cross-check the Rust encoder. |
| candle (pure Rust, no C++) | Qwen3.5's Gated DeltaNet layers are not in candle yet ([PR open](https://github.com/huggingface/candle/pull/3396)). Worth revisiting if removing the C++ dependency matters. |

**Next steps:**

1. **Build a labelled dataset.** Run candidate models on verifiable tasks and record
   success and cost per (task, model). Encode each task with `opas-encode --input`.
2. **Train and select the probe offline.** Pick the layer, PCA, and logistic/ridge
   models with leave-one-benchmark-out cross-validation, against random routing and the
   cheap-feature baselines. This can be Python (scikit-learn) on the `.npy` files, or
   Rust.
3. **Ship it.** If it wins, add a `ProbeRouter` to opas. It holds an `Encoder` (the
   `encode_request` → probe weights path) behind a mutex, loads the exported probe
   weights, and routes on `p̂ − λ·ĉ`.

## 6. Open questions

- **Does a 9B encoder work?** The NVIDIA results use 20B+ encoders. We have to measure
  whether a 9B one predicts success for Opus-class *and* Haiku-class targets on coding
  tasks.
- **Activations alone or combined?** Is the signal enough alone, or does it need
  structured task attributes (scope, domain, ambiguity) from a small LLM?
- **Does 8-bit quantization hurt?** Repeat the probe on the bf16 weights, if disk space
  allows, to check.
- **Hybrid layers:** do Gated DeltaNet layers carry as much signal as the
  full-attention layers (every 4th)? The layer sweep will show it.
- **Robustness:** the probe must hold up on held-out task types, not only held-out
  examples.
