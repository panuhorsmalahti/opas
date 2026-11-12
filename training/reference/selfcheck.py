"""Check that extract.py's per-layer prefill loop reproduces the model's own forward pass.

Usage (from training/): uv run --group reference reference/selfcheck.py [--model PATH]
"""

import argparse

import mlx.core as mx
from mlx_lm import load

from extract import DEFAULT_MODEL, prefill_hidden_states

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(DEFAULT_MODEL))
args = ap.parse_args()

model, tok = load(args.model)
ids = tok.apply_chat_template(
    [{"role": "user", "content": "Fix the race condition in this Go worker pool."}],
    add_generation_prompt=True,
    tokenize=True,
)
print("prompt ends with:", repr(tok.decode(ids[-6:])))

states = prefill_hidden_states(model, ids)
lm = model.language_model
h = lm.model.norm(states[-1][None])
mine = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
ref = model(mx.array(ids)[None])

diff = mx.abs(mine - ref).max().item()
print(f"layers captured: {len(states)} (embeddings + {len(states) - 1} decoder layers)")
print(f"max |logit diff| manual vs model(): {diff}")
assert diff < 1e-3, "manual layer loop diverges from model forward"
assert (mine[0, -1].argmax() == ref[0, -1].argmax()).item()
print("ok")
