"""Reference implementation (Python, MLX, Apple Silicon only) of the Rust encoder's
prefill activation extraction. Used to cross-check `opastin-encoder`; not used at runtime.

Extract prefill activations from a local LLM, without generating any tokens.

For each request, the prompt goes through the model's chat template (ending at the
assistant-turn start, where generation would begin) and one forward pass (prefill).
The hidden state after every decoder layer is captured, pooled two ways:

- last: the final prompt token, the "end of input" position used in the probing papers
- mean: the average over all prompt tokens

Output is an .npz file with arrays shaped [n_requests, n_layers + 1, hidden]
(index 0 is the embedding output, index i is after decoder layer i), plus token counts
and timings. Those vectors are the input for a complexity probe (see
docs/complexity-from-activations.md).

Input is JSONL. Each line is one of:
  {"id": "...", "prompt": "plain user text"}
  {"id": "...", "api": "openai_chat" | "anthropic_messages", "request": {...request body...}}

Usage:
  uv run extract.py --input requests.jsonl --out out/acts.npz
  uv run extract.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

# Default: the small model used to cross-check the Rust encoder (same architecture as the 9B).
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-0.8B-MLX-4bit"

DEMO = [
    {"id": "trivial", "prompt": "What is 2 + 2?"},
    {"id": "lookup", "prompt": "What is the capital of Australia?"},
    {"id": "rename", "prompt": "Rename the variable `x` to `count` in: `let x = 0; x += 1;`"},
    {
        "id": "bugfix",
        "prompt": "Our Rust async HTTP proxy sometimes truncates SSE streams under load when "
        "the client disconnects mid-stream. Find the likely cause in the hyper/axum body "
        "forwarding path and propose a fix with tests.",
    },
    {
        "id": "design",
        "prompt": "Design a distributed rate limiter for a multi-region API gateway with "
        "per-tenant quotas, burst handling, and graceful degradation during partitions. "
        "Compare token bucket vs. sliding window, and give the data model and failure modes.",
    },
]


# --- request -> chat messages -------------------------------------------------------


def _text(content) -> str:
    """Text of an OpenAI/Anthropic content field (string or list of parts/blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") == "tool_result":
                parts.append(_text(p.get("content")))
            elif p.get("type") == "tool_use":
                parts.append(f"[tool call {p.get('name')}: {json.dumps(p.get('input'))}]")
        return "\n".join(s for s in parts if s)
    return ""


def to_messages(item: dict) -> list[dict]:
    """Convert one input line to chat-template messages (system/user/assistant)."""
    if "prompt" in item:
        return [{"role": "user", "content": item["prompt"]}]

    req = item["request"]
    msgs: list[dict] = []
    if item.get("api") == "anthropic_messages" and req.get("system"):
        msgs.append({"role": "system", "content": _text(req["system"])})
    for m in req.get("messages", []):
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        if role == "tool":  # OpenAI tool result
            role = "user"
        if role not in ("system", "user", "assistant"):
            role = "user"
        text = _text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):  # OpenAI tool calls
            calls = [f"[tool call {c['function']['name']}: {c['function'].get('arguments')}]" for c in m["tool_calls"]]
            text = "\n".join([text, *calls]).strip()
        msgs.append({"role": role, "content": text})
    return msgs


# --- activation capture -------------------------------------------------------------


def prefill_hidden_states(model, token_ids: list[int]) -> list[mx.array]:
    """Run one prefill pass and return the residual stream after each layer.

    Returns n_layers + 1 arrays of shape [seq, hidden]: the embedding output, then
    the output of each decoder layer. No KV cache and no sampling: nothing is generated.
    """
    text_model = model.language_model.model
    h = text_model.embed_tokens(mx.array(token_ids)[None])
    attn_mask = create_attention_mask(h, None)  # causal mask for the full-attention layers
    states = [h[0]]
    for layer in text_model.layers:
        # Linear-attention (Gated DeltaNet) layers take no mask when there is no padding.
        h = layer(h, mask=None if layer.is_linear else attn_mask, cache=None)
        states.append(h[0])
    return states


def extract(model, tokenizer, items: list[dict], max_tokens: int, verbose: bool, thinking: bool = True):
    n = len(items)
    last_all, mean_all, n_tokens, seconds = [], [], [], []
    for i, item in enumerate(items):
        messages = to_messages(item)
        # Explicit: the template's own default is non-thinking, while mlx-lm's wrapper
        # turns thinking on unless told otherwise. Must match the Rust encoder's setting.
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=thinking
        )
        if len(ids) > max_tokens:  # drop the middle, like the Rust encoder
            head = max_tokens // 2
            ids = ids[:head] + ids[len(ids) - (max_tokens - head):]

        t0 = time.perf_counter()
        states = prefill_hidden_states(model, ids)
        last = mx.stack([s[-1] for s in states]).astype(mx.float32)
        mean = mx.stack([s.astype(mx.float32).mean(axis=0) for s in states])
        mx.eval(last, mean)
        dt = time.perf_counter() - t0

        last_all.append(np.array(last, dtype=np.float16))
        mean_all.append(np.array(mean, dtype=np.float16))
        n_tokens.append(len(ids))
        seconds.append(dt)
        if verbose:
            print(
                f"[{i + 1}/{n}] {item.get('id', i)}: {len(ids)} tokens, prefill+capture {dt * 1000:.0f} ms",
                file=sys.stderr,
            )
    return (
        np.stack(last_all),
        np.stack(mean_all),
        np.array(n_tokens),
        np.array(seconds),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="local MLX model directory")
    ap.add_argument("--input", type=Path, help="JSONL file of requests")
    ap.add_argument("--out", type=Path, help="output .npz path")
    ap.add_argument("--max-tokens", type=int, default=16384, help="truncate prompts to N tokens (first and last N/2 kept)")
    ap.add_argument("--demo", action="store_true", help="run built-in example prompts and print a summary")
    ap.add_argument("--non-thinking", action="store_true", help="end the prompt with a closed think block")
    args = ap.parse_args()

    if args.demo:
        items = DEMO
    elif args.input:
        items = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        ap.error("pass --input or --demo")

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"loaded {args.model} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    # Warm-up pass so the first timing excludes kernel compilation.
    prefill_hidden_states(model, tokenizer.encode("warm up"))[-1].tolist()

    last, mean, n_tokens, seconds = extract(
        model, tokenizer, items, args.max_tokens, verbose=True, thinking=not args.non_thinking
    )
    print(f"activations: last {last.shape}, mean {mean.shape} (requests, layers+1, hidden)", file=sys.stderr)

    if args.demo:
        # Cosine similarity of last-token states at a middle layer: a quick check that the
        # vectors differ by prompt, not a complexity measure.
        mid = last.shape[1] // 2
        v = last[:, mid, :].astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sim = v @ v.T
        ids = [it["id"] for it in items]
        print(f"\ncosine similarity of last-token states at layer {mid}:")
        print(" " * 9 + " ".join(f"{s[:8]:>8}" for s in ids))
        for name, row in zip(ids, sim):
            print(f"{name[:8]:>8} " + " ".join(f"{x:8.3f}" for x in row))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            ids=np.array([str(it.get("id", i)) for i, it in enumerate(items)]),
            last=last,
            mean=mean,
            n_tokens=n_tokens,
            seconds=seconds,
            model=np.array(args.model),
        )
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
