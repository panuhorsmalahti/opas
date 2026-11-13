# Training the probe

Training is offline and optional: the repository ships a trained predictor
(`probes/llmrouterbench-qwen35-9b.json`). Run these steps from the repository root. The router only loads the exported probe file. Model weights and
datasets go in `models/` and `data/`, which git ignores.

```sh
# 1. Encoder model (9.5 GB) and LLMRouterBench (1.3 GB)
mkdir -p models/gguf
curl -L -o models/gguf/Qwen3.5-9B-Q8_0.gguf \
  https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf
./scripts/download-llmrouterbench.sh

# 2. Labels (per prompt and model: score, cost, output tokens) and a stratified sample
cd training && uv run prepare_llmrouterbench.py && cd ..

# 3. Activations for the sample (Rust encoder; every layer, last-token and mean pooling)
cargo build --release -p opas-encoder
./target/release/opas-encode --input data/llmrouterbench/sample.jsonl \
  --out data/llmrouterbench/acts-9b --max-tokens 1024

# 4. Train, evaluate, export probes/llmrouterbench-qwen35-9b.json and docs/probe-results.md
cd training && uv run train_probe.py

# Optional: compare predictor sizes and types (predictor-capacity.md, ~30 min)
uv run predictor_capacity.py
```

**What training does:**

- **Tiers:** the 13 LLMRouterBench models are ranked by overall success and split into
  small, medium and large thirds. The xlarge (Fable-class) head is extrapolated one step
  beyond large, because no benchmark model is in that class.
- **Model selection:** the layer, pooling and regularization are chosen by 5-fold
  cross-validation, scored by *within-benchmark* AUROC, so the probe is rewarded for
  ranking difficulty rather than for recognizing which benchmark a prompt came from.
- **Evaluation:** a held-out split compares probe routing against random routing at the
  same cost, single tiers, a benchmark-identity router and the oracle.

## Results

On the held-out split (832 prompts; full report in
[probe-results.md](probe-results.md)):

| Router | Success | Mean cost / request |
|---|---|---|
| always small / medium / large | 0.475 / 0.517 / 0.574 | $0.014 / $0.018 / $0.042 |
| opas, `quality` preset (100/0/0) | 0.577 | $0.034 |
| opas, default `balanced` (60/20/20) | 0.548 | $0.024 |
| random routing at the same $0.024 | 0.509 | $0.024 |
| oracle (best tier per prompt) | 0.640 | $0.021 |

- **What works:** with the default weights, opas beats random routing at the same cost
  by 3.9 points, and comes within 2.6 points of always using the large tier at 57% of
  its cost. The `quality` preset matches always-large at 83% of its cost.
- **What doesn't (yet):** on this benchmark, a TF-IDF bag-of-words router does about as
  well as the activation probe (within a point at every cost level). Most of the
  signal is which kind of task a prompt is. The probe is better at ranking difficulty
  *within* a benchmark (cross-validated AUROC 0.681 vs 0.576 for TF-IDF), but that gain is small in
  routing terms here.
- **Predictor size** ([predictor-capacity.md](predictor-capacity.md)): on the
  same data, predictors from 51 to 4.5 million learned parameters (linear models, MLPs,
  boosted trees) differ by at most 0.006 in routed success, and no difference is
  distinguishable from noise. In ranking difficulty, a linear model on all 4096
  activation dimensions beats one on 100 PCA dimensions (+0.018 within-benchmark
  AUROC), and a 4.5M-parameter MLP adds nothing beyond that. The shipped predictor is
  therefore linear on all 4096 dimensions: about 4,100 learned numbers per head.
- **Limits:** the tiers are open models from 2025 rather than the Claude models in the
  example config; the xlarge head is extrapolated; speed scores use tier-default speeds.

`training/reference/` holds a Python (MLX) reimplementation of the encoder, used to
cross-check the Rust encoder; see [its README](../training/reference/README.md).
