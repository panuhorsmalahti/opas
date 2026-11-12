"""Compare activations from the Rust encoder (opastin-encode --out DIR) with the Python
MLX reference (extract.py --out FILE.npz), per hidden-state index.

Different runtimes and quantizations will not give identical numbers. What should hold:
- the same prompts tokenize to the same length,
- per-layer cosine similarity between the two runtimes stays high,
- the prompt-by-prompt similarity structure (which prompts look alike) agrees.

Usage (from training/):
  ../target/release/opastin-encode --model ../models/gguf/Qwen3.5-0.8B-Q8_0.gguf --demo --out out/rust-0.8b
  uv run --group reference reference/extract.py --demo --out out/py-0.8b.npz
  uv run --group reference reference/compare.py out/rust-0.8b out/py-0.8b.npz
"""

import json
import sys
from pathlib import Path

import numpy as np


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


def sim_matrix(v: np.ndarray) -> np.ndarray:
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return v @ v.T


rust_dir, py_file = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.loads((rust_dir / "meta.json").read_text())
py = np.load(py_file)

assert list(meta["ids"]) == [str(x) for x in py["ids"]], "prompt ids differ"
rust_tokens, py_tokens = np.load(rust_dir / "n_tokens.npy"), py["n_tokens"]
print("token counts  rust:", rust_tokens.tolist(), " python:", py_tokens.tolist())

layers = meta["layers"]
for pool in ("last", "mean"):
    r = np.load(rust_dir / f"{pool}.npy")
    p = py[pool][:, layers, :].astype(np.float32)
    per_layer = cos_rows(r, p).mean(axis=0)  # mean over prompts
    print(f"\n{pool}-token pooling: cosine(rust, python) by hidden-state index")
    print("  " + " ".join(f"{i}:{c:.3f}" for i, c in zip(layers, per_layer)))
    # Agreement of the prompt-similarity structure at each layer (off-diagonal entries).
    n = r.shape[0]
    mask = ~np.eye(n, dtype=bool)
    corr = [
        np.corrcoef(sim_matrix(r[:, k])[mask], sim_matrix(p[:, k])[mask])[0, 1]
        for k in range(len(layers))
    ]
    print("  similarity-structure correlation: min %.3f, median %.3f" % (np.nanmin(corr), np.nanmedian(corr)))
