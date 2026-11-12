# Trained predictors

Predictor (probe) files for the probe router, small enough to keep in the repository.

- **Format:** `opas-probe/2`, loaded by `opas-encoder`.
- **Contents:** each file holds per-tier linear heads that read one layer of the encoder
  model's activations and predict success probability and output tokens.

| File | Encoder | Trained on | Results |
|---|---|---|---|
| `llmrouterbench-qwen35-9b.json` | Qwen3.5-9B Q8_0 (GGUF) | LLMRouterBench, performance-cost setting | [docs/probe-results.md](../docs/probe-results.md) |

A probe only works with the encoder it was trained on, with the same layer, pooling,
thinking mode and truncation; these are recorded in the file's `encoder` section. To
retrain, see "Training the probe" in the main README.
