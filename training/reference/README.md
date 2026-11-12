# Reference extractor (Python, MLX)

An independent Python implementation of what `opas-encoder` does in Rust: one
prefill pass of a Qwen3.5 model, capturing the hidden state after every layer. It is
used only to cross-check the Rust encoder. Nothing at runtime uses it.

Run the cross-check after changing the encoder or upgrading `llama-cpp-sys-2`. Tensor
names and kernels in llama.cpp can change between releases.

It needs Apple Silicon (MLX) and the 0.8B models in `models/`:

- `Qwen3.5-0.8B-MLX-4bit`, from `lmstudio-community/Qwen3.5-0.8B-MLX-4bit`;
- `gguf/Qwen3.5-0.8B-Q8_0.gguf`.

```sh
cd training
uv run --group reference reference/selfcheck.py      # manual layer loop == model forward
../target/release/opas-encode --model ../models/gguf/Qwen3.5-0.8B-Q8_0.gguf --demo --out out/rust-0.8b
uv run --group reference reference/extract.py --demo --out out/py-0.8b.npz
uv run --group reference reference/compare.py out/rust-0.8b out/py-0.8b.npz
```

What to expect (from the last run): identical token counts, mean-pooling cosine similarity
0.97–0.998 per layer, and last-token cosine similarity falling from 0.996 to about 0.82
with depth. The gap comes from the 4-bit MLX against 8-bit GGUF quantization. See
`docs/complexity-from-activations.md`.
