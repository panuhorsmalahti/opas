"""Turn the LLMRouterBench archive into compact routing labels and an encoder sample.

Reads data/llmrouterbench/bench-release.tar.gz (fetch it with
scripts/download-llmrouterbench.sh) as a stream, without extracting the raw model
outputs. Keeps the "performance-cost" setting of the paper: 10 datasets, 13 flagship
API models, each answer with a score and its USD cost.

Writes, under data/llmrouterbench/:
  labels.jsonl   one line per prompt: id, dataset, prompt, and per model
                 {"score": 0|0.5|1, "cost": USD, "out_tokens": int}
  sample.jsonl   a stratified sample of prompts in opas-encode's input format

Usage: uv run prepare_llmrouterbench.py [--per-dataset 600] [--seed 0]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "llmrouterbench"

# Performance-cost setting (config/baseline_config_performance_cost.yaml upstream).
DATASETS = {
    "aime": "hybrid",
    "livemathbench": "test",
    "gpqa": "test",
    "hle": "test",
    "livecodebench": "test",
    "mmlupro": "test_3000",
    "swe-bench": "verified",
    "simpleqa": "test",
    "tau2": "test",
    "arenahard": None,  # stored without a split directory
}
MODELS = [
    "claude-sonnet-4", "deepseek-v3-0324", "deepseek-v3.1-terminus", "deepseek-r1-0528",
    "gemini-2.5-flash", "gemini-2.5-pro", "gpt-5-chat", "gpt-5",
    "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507", "glm-4.6", "kimi-k2-0905",
    "intern-s1",
]


def select(name: str) -> tuple[str, str, str] | None:
    """(dataset, split, model) if this archive member is a result file we want."""
    parts = name.split("/")
    if len(parts) < 4 or not parts[-1].endswith(".json"):
        return None
    dataset, model = parts[1], parts[-2]
    if dataset not in DATASETS or model not in MODELS:
        return None
    split = DATASETS[dataset]
    if split is None:
        if len(parts) != 4:  # bench-release/arenahard/<model>/<file>
            return None
        split = "test"
    elif len(parts) != 5 or parts[2] != split:
        return None
    return dataset, split, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DATA / "bench-release.tar.gz")
    ap.add_argument("--per-dataset", type=int, default=600, help="max prompts per dataset in the sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items: dict[str, dict] = {}
    newest: dict[tuple[str, str, str], str] = {}  # (dataset, split, model) -> file name used
    with tarfile.open(args.archive, mode="r|gz") as tar:
        for member in tar:
            sel = select(member.name)
            if sel is None or not member.isfile():
                continue
            dataset, split, model = sel
            fname = member.name.rsplit("/", 1)[1]
            # Several runs may exist; file names end in a timestamp, keep the newest.
            if fname <= newest.get(sel, ""):
                continue
            newest[sel] = fname
            result = json.load(tar.extractfile(member))
            if result.get("demo"):
                continue
            for r in result["records"]:
                key = f"{dataset}/{split}/{r['index']}"
                item = items.setdefault(
                    key,
                    {"id": key, "dataset": dataset, "split": split, "index": r["index"],
                     "prompt": r["prompt"], "models": {}},
                )
                if r.get("score") is None:
                    continue
                item["models"][model] = {
                    "score": float(r["score"]),
                    "cost": float(r.get("cost") or 0.0),
                    "out_tokens": int(r.get("completion_tokens") or 0),
                }
            print(f"read {member.name} ({len(result['records'])} records)", flush=True)

    rows = [it for it in items.values() if isinstance(it["prompt"], str) and it["prompt"].strip()]
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "labels.jsonl", "w") as f:
        for it in sorted(rows, key=lambda x: x["id"]):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Stratified sample: up to --per-dataset prompts from each dataset.
    rng = random.Random(args.seed)
    by_ds = collections.defaultdict(list)
    for it in rows:
        by_ds[it["dataset"]].append(it)
    sample = []
    for ds in sorted(by_ds):
        group = sorted(by_ds[ds], key=lambda x: x["id"])
        rng.shuffle(group)
        sample += group[: args.per_dataset]
    with open(DATA / "sample.jsonl", "w") as f:
        for it in sample:
            f.write(json.dumps({"id": it["id"], "prompt": it["prompt"]}, ensure_ascii=False) + "\n")

    print(f"\n{len(rows)} prompts -> {DATA / 'labels.jsonl'}; sample of {len(sample)} -> {DATA / 'sample.jsonl'}")
    print(f"{'dataset':<15} {'prompts':>7} {'sampled':>7}  mean success per model (cheapest ... priciest)")
    for ds in sorted(by_ds):
        n_sample = min(len(by_ds[ds]), args.per_dataset)
        print(f"{ds:<15} {len(by_ds[ds]):>7} {n_sample:>7}")
    print(f"\n{'model':<32} {'answers':>7} {'success':>8} {'mean cost $':>12}")
    for m in MODELS:
        vals = [it["models"][m] for it in rows if m in it["models"]]
        if vals:
            ok = sum(v["score"] >= 0.5 for v in vals) / len(vals)
            cost = sum(v["cost"] for v in vals) / len(vals)
            print(f"{m:<32} {len(vals):>7} {ok:>8.3f} {cost:>12.5f}")


if __name__ == "__main__":
    main()
