"""Train opas's tier probe on LLMRouterBench activations and evaluate it as a router.

Inputs (see README "Training the probe"):
  data/llmrouterbench/labels.jsonl   from prepare_llmrouterbench.py
  data/llmrouterbench/acts-9b/       from `opas-encode --input sample.jsonl --out ...`

The probe predicts per capability tier (small / medium / large / xlarge), not per named
model, so deployments map their own models onto tiers:
  p_success  = sigmoid(w·h + b)     chance a model of the tier answers successfully
  out_tokens = exp(w·h + b) − 1     expected output tokens
from the encoder's hidden state h at one layer. The router turns tokens into dollars
with each model's own prices.

Tiers from the benchmark: the 13 models are ranked by overall success on all
LLMRouterBench prompts and split into thirds (small / medium / large). A tier's label on
a prompt is the fraction of its models that answered it (fit by weighted logistic
regression); its token target is their mean output tokens. No benchmark model is in the
"xlarge" (Fable) class, so that head is extrapolated one step beyond large:
xlarge = large + (large − medium) in log-odds, with large's token head.

Each head is standardize -> PCA(100) -> logistic / ridge, collapsed into one weight vector
over h (every step is linear), which is what the Rust ProbeRouter loads.

Steps:
  1. Split prompts 80/20, stratified by dataset (seeded).
  2. On the 80%: 5-fold CV over (layer, pooling) and regularization, against baselines
     (prompt length, TF-IDF, dataset identity).
  3. On the 20%: route between tiers with argmax p - lambda*cost; compare with random
     routing at equal cost, single tiers, a dataset-identity router and the oracle.
  4. Refit on all prompts, export the probe JSON and a markdown report.

Usage: uv run train_probe.py [--out ../probes/llmrouterbench-qwen35-9b.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "llmrouterbench"
PCA_DIM = 100
SEED = 0
TRAINED_TIERS = ["small", "medium", "large"]


# --- data -----------------------------------------------------------------------------


def assign_tiers(all_labels: list[dict]) -> dict[str, list[str]]:
    """Rank models by overall success on every prompt and split them into thirds."""
    models = sorted({m for r in all_labels for m in r["models"]})
    rate = {}
    for m in models:
        scores = [r["models"][m]["score"] >= 0.5 for r in all_labels if m in r["models"]]
        rate[m] = float(np.mean(scores))
    ranked = sorted(models, key=lambda m: rate[m])
    bands = np.array_split(np.array(ranked), len(TRAINED_TIERS))
    return {t: list(b) for t, b in zip(TRAINED_TIERS, bands)}, rate


def load(acts_dir: Path):
    meta = json.loads((acts_dir / "meta.json").read_text())
    all_labels = [json.loads(l) for l in open(DATA / "labels.jsonl")]
    tiers, rate = assign_tiers(all_labels)
    by_id = {r["id"]: r for r in all_labels}
    rows = [by_id[i] for i in meta["ids"]]

    n, T = len(rows), len(TRAINED_TIERS)
    frac = np.zeros((n, T))       # fraction of the tier's models that succeeded
    tokens = np.zeros((n, T))     # mean output tokens of the tier's models
    cost = np.zeros((n, T))       # mean recorded USD cost of the tier's models
    member_success = []           # per tier: (n, members) binary matrix, NaN if missing
    for t, tier in enumerate(TRAINED_TIERS):
        ms = tiers[tier]
        succ = np.full((n, len(ms)), np.nan)
        for i, r in enumerate(rows):
            vals = [r["models"][m] for m in ms if m in r["models"]]
            for k, m in enumerate(ms):
                if m in r["models"]:
                    succ[i, k] = float(r["models"][m]["score"] >= 0.5)
            frac[i, t] = np.mean([v["score"] >= 0.5 for v in vals])
            tokens[i, t] = np.mean([v["out_tokens"] for v in vals])
            cost[i, t] = np.mean([v["cost"] for v in vals])
        member_success.append(succ)
    datasets = np.array([r["dataset"] for r in rows])
    prompts = [r["prompt"] for r in rows]
    last = np.load(acts_dir / "last.npy", mmap_mode="r")
    mean = np.load(acts_dir / "mean.npy", mmap_mode="r")
    n_tokens = np.load(acts_dir / "n_tokens.npy")
    return meta, tiers, rate, frac, tokens, cost, member_success, datasets, prompts, last, mean, n_tokens


# --- models ---------------------------------------------------------------------------


def fit_weighted_logistic(F, frac, C):
    """Logistic regression on fractional labels: each row appears as a success with
    weight `frac` and as a failure with weight `1 - frac`."""
    n = F.shape[0]
    X = sparse.vstack([F, F]).tocsr() if sparse.issparse(F) else np.vstack([F, F])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    w = np.concatenate([frac, 1 - frac])
    keep = w > 0
    return LogisticRegression(C=C, max_iter=3000).fit(X[keep], y[keep], sample_weight=w[keep])


class TierProbe:
    """standardize -> PCA -> per-tier logistic (success) and ridge (log tokens)."""

    def __init__(self, pca_dim=PCA_DIM, C=0.1, alpha=10.0, sparse=False):
        self.pca_dim, self.C, self.alpha, self.sparse = pca_dim, C, alpha, sparse

    def features(self, X):
        if self.sparse:
            return X
        return self.pca.transform(self.scaler.transform(X))

    def fit(self, X, frac, tokens):
        if not self.sparse:
            self.scaler = StandardScaler().fit(X)
            Z = self.scaler.transform(X)
            k = min(self.pca_dim, Z.shape[0] - 1, Z.shape[1])
            self.pca = PCA(n_components=k, random_state=SEED).fit(Z)
        F = self.features(X)
        self.succ = [fit_weighted_logistic(F, frac[:, t], self.C) for t in range(frac.shape[1])]
        self.tok = [Ridge(alpha=self.alpha).fit(F, np.log1p(tokens[:, t])) for t in range(frac.shape[1])]
        return self

    def predict(self, X):
        F = self.features(X)
        p = np.column_stack([h.predict_proba(F)[:, 1] for h in self.succ])
        k = np.column_stack([np.expm1(h.predict(F)) for h in self.tok])
        return p, np.maximum(k, 0)

    def collapse(self, model):
        """A fitted linear model on PCA features as (w, b) over the raw features."""
        mu, sd = self.scaler.mean_, self.scaler.scale_
        P, pm = self.pca.components_, self.pca.mean_
        coef = np.ravel(model.coef_)
        intercept = float(np.ravel([model.intercept_])[0])
        v = P.T @ coef
        return v / sd, intercept - float((mu / sd + pm) @ v)


# --- metrics --------------------------------------------------------------------------


def member_auroc(p, member_success, datasets=None):
    """AUROC of the tier prediction against each member model's success, averaged over
    tiers. With `datasets`, computed inside each benchmark and averaged by size (how well
    difficulty is ranked beyond knowing which benchmark a prompt is from)."""
    groups = [np.ones(len(p), bool)] if datasets is None else [datasets == d for d in np.unique(datasets)]
    total, weight = 0.0, 0
    for g in groups:
        vals = []
        for t, succ in enumerate(member_success):
            y = succ[g].ravel()
            s = np.repeat(p[g, t], succ.shape[1])
            ok = ~np.isnan(y)
            if len(np.unique(y[ok])) == 2:
                vals.append(roc_auc_score(y[ok], s[ok]))
        if vals:
            total += np.mean(vals) * g.sum()
            weight += g.sum()
    return float(total / weight)


def token_spearman(true, pred):
    return float(np.mean([spearmanr(true[:, t], pred[:, t]).statistic for t in range(true.shape[1])]))


def cv_score(fx, idx, frac, tokens, member_success, datasets, C=0.1, alpha=10.0, sparse=False, pca_dim=PCA_DIM):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    P = np.zeros((len(idx), frac.shape[1]))
    K = np.zeros_like(P)
    for tr, va in skf.split(idx, datasets[idx]):
        probe = TierProbe(pca_dim=pca_dim, C=C, alpha=alpha, sparse=sparse)
        probe.fit(fx(idx[tr]), frac[idx[tr]], tokens[idx[tr]])
        P[va], K[va] = probe.predict(fx(idx[va]))
    ms = [s[idx] for s in member_success]
    return (member_auroc(P, ms), member_auroc(P, ms, datasets[idx]), token_spearman(tokens[idx], K))


# --- routing evaluation ---------------------------------------------------------------


def routing_eval(p, k_pred, frac, cost, price_per_token, lambdas):
    """Route each prompt to the tier with the best p - lambda * predicted cost. Outcomes
    are the chosen tier's mean success and mean recorded cost on that prompt."""
    rows = np.arange(len(p))
    out = []
    for lam in lambdas:
        score = p - lam * (k_pred * price_per_token[None, :])
        pick = np.argmax(score, axis=1)
        out.append((lam, float(frac[rows, pick].mean()), float(cost[rows, pick].mean()),
                    np.bincount(pick, minlength=p.shape[1]) / len(pick)))
    return out


def random_at_cost(frac, cost, target):
    """Random routing between the cheapest and priciest tier mixed to match `target` cost."""
    c, s = cost.mean(0), frac.mean(0)
    lo, hi = int(np.argmin(c)), int(np.argmax(c))
    q = np.clip((target - c[lo]) / (c[hi] - c[lo]), 0, 1)
    return float(q * s[hi] + (1 - q) * s[lo])


# --- main -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, default=DATA / "acts-9b")
    ap.add_argument("--out", type=Path, default=ROOT / "probes" / "llmrouterbench-qwen35-9b.json")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "probe-results.md")
    ap.add_argument("--layer-step", type=int, default=2, help="sweep every Nth captured layer")
    args = ap.parse_args()

    (meta, tiers, rate, frac, tokens, cost, member_success, datasets, prompts,
     last, mean, n_tokens) = load(args.acts)
    layers = meta["layers"]
    n = len(datasets)
    print(f"{n} prompts, {len(layers)} captured layers, hidden {last.shape[2]}")
    for t in TRAINED_TIERS:
        print(f"tier {t:<6}: " + ", ".join(f"{m} ({rate[m]:.3f})" for m in tiers[t]))

    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=SEED, stratify=datasets)
    idx_train.sort()
    idx_test.sort()

    # Effective USD per output token per tier (recorded cost / output tokens, training
    # split): turns predicted tokens into the cost estimate used for routing evaluation.
    price_per_token = cost[idx_train].sum(0) / np.maximum(tokens[idx_train].sum(0), 1)

    # Baselines.
    ohe = OneHotEncoder().fit(datasets.reshape(-1, 1))
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True).fit(
        [prompts[i] for i in idx_train])
    baselines = {
        "length (log tokens)": (lambda ix: np.log(n_tokens[ix]).reshape(-1, 1).astype(float), False),
        "dataset identity": (lambda ix: ohe.transform(datasets[ix].reshape(-1, 1)).toarray(), False),
        "TF-IDF": (lambda ix: tfidf.transform([prompts[i] for i in ix]), True),
    }
    results = {}
    for name, (fx, sparse) in baselines.items():
        results[name] = cv_score(fx, idx_train, frac, tokens, member_success, datasets,
                                 C=1.0, alpha=1.0, sparse=sparse, pca_dim=10**6)
        print(f"baseline {name:<20} AUROC {results[name][0]:.3f}  within-dataset {results[name][1]:.3f}  tokens rho {results[name][2]:.3f}", flush=True)

    # Activation sweep.
    sweep = []
    for k in range(0, len(layers), args.layer_step):
        for pool, arr in (("last", last), ("mean", mean)):
            fx = lambda ix, arr=arr, k=k: np.asarray(arr[ix, k, :], dtype=np.float32)
            r = cv_score(fx, idx_train, frac, tokens, member_success, datasets)
            sweep.append((layers[k], pool, *r))
            print(f"layer {layers[k]:>2} {pool:<4} AUROC {r[0]:.3f}  within-dataset {r[1]:.3f}  tokens rho {r[2]:.3f}", flush=True)
    # Choose by within-dataset AUROC: difficulty, not benchmark identity.
    best_layer, best_pool = max(sweep, key=lambda t: t[3])[:2]
    k_best = layers.index(best_layer)
    arr_best = last if best_pool == "last" else mean
    fx_best = lambda ix: np.asarray(arr_best[ix, k_best, :], dtype=np.float32)
    grid = []
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        r = cv_score(fx_best, idx_train, frac, tokens, member_success, datasets, C=C)
        grid.append((C, *r))
        print(f"C={C:<5} AUROC {r[0]:.3f}  within-dataset {r[1]:.3f}", flush=True)
    best_C = max(grid, key=lambda t: t[2])[0]
    alpha_grid = [(a, cv_score(fx_best, idx_train, frac, tokens, member_success, datasets, C=best_C, alpha=a)[2])
                  for a in (1.0, 10.0, 100.0, 1000.0)]
    best_alpha = max(alpha_grid, key=lambda t: t[1])[0]
    print(f"chosen: layer {best_layer} {best_pool}, C={best_C}, alpha={best_alpha}")

    # Held-out test.
    probe = TierProbe(C=best_C, alpha=best_alpha).fit(fx_best(idx_train), frac[idx_train], tokens[idx_train])
    p_te, k_te = probe.predict(fx_best(idx_test))
    ms_te = [s[idx_test] for s in member_success]
    test = {
        "auroc": member_auroc(p_te, ms_te),
        "within_dataset_auroc": member_auroc(p_te, ms_te, datasets[idx_test]),
        "tokens_spearman": token_spearman(tokens[idx_test], k_te),
    }
    print("test:", {k: round(v, 3) for k, v in test.items()})

    ds_probe = TierProbe(pca_dim=10**6, C=1.0, alpha=1.0)
    Xd = ohe.transform(datasets.reshape(-1, 1)).toarray()
    ds_probe.fit(Xd[idx_train], frac[idx_train], tokens[idx_train])
    p_ds, k_ds = ds_probe.predict(Xd[idx_test])

    lambdas = [0, 1, 3, 10, 30, 100, 300, 1000]
    f_te, c_te = frac[idx_test], cost[idx_test]
    probe_curve = routing_eval(p_te, k_te, f_te, c_te, price_per_token, lambdas)
    ds_curve = routing_eval(p_ds, k_ds, f_te, c_te, price_per_token, lambdas)
    singles = {t: (float(f_te[:, i].mean()), float(c_te[:, i].mean())) for i, t in enumerate(TRAINED_TIERS)}
    # Oracle: per prompt, the tier with the highest success, cheapest among ties.
    oracle_pick = np.array(
        [min(range(f_te.shape[1]), key=lambda t: (-f_te[i, t], c_te[i, t])) for i in range(len(f_te))])
    rows = np.arange(len(f_te))
    oracle = (float(f_te[rows, oracle_pick].mean()), float(c_te[rows, oracle_pick].mean()))
    routing = []
    print("\nrouting between tiers on the held-out split:")
    for t, (s, c) in singles.items():
        print(f"  always {t:<6} success {s:.3f}  cost ${c:.5f}")
    for (lam, s, c, share), (_, s_ds, c_ds, _) in zip(probe_curve, ds_curve):
        affordable = [v[0] for v in singles.values() if v[1] <= c * 1.0001]
        row = {"lambda": lam, "success": s, "cost": c, "share": dict(zip(TRAINED_TIERS, map(float, share))),
               "random_same_cost": random_at_cost(f_te, c_te, c),
               "best_single_within_cost": max(affordable) if affordable else None,
               "dataset_router": (s_ds, c_ds)}
        routing.append(row)
        print(f"  lambda {lam:>5}: success {s:.3f} cost ${c:.5f} | random at same cost {row['random_same_cost']:.3f}"
              f" | dataset router {s_ds:.3f} @ ${c_ds:.5f} | share {np.round(share, 2)}")
    print(f"  oracle: success {oracle[0]:.3f} cost ${oracle[1]:.5f}")

    # Final probe on all prompts, collapsed to weights over h.
    final = TierProbe(C=best_C, alpha=best_alpha).fit(fx_best(np.arange(n)), frac, tokens)
    heads = {}
    check_X = fx_best(idx_test[:50]).astype(np.float64)
    p_chk, k_chk = final.predict(check_X.astype(np.float32))
    for t, tier in enumerate(TRAINED_TIERS):
        sw, sb = final.collapse(final.succ[t])
        tw, tb = final.collapse(final.tok[t])
        assert np.allclose(1 / (1 + np.exp(-(check_X @ sw + sb))), p_chk[:, t], atol=1e-4), tier
        assert np.allclose(np.maximum(np.expm1(check_X @ tw + tb), 0), k_chk[:, t], rtol=1e-3, atol=1e-3), tier
        heads[tier] = {"success": (sw, sb), "log_out_tokens": (tw, tb), "extrapolated": False}
    # xlarge: one step beyond large in log-odds; large's token head.
    (lw, lb), (mw, mb) = heads["large"]["success"], heads["medium"]["success"]
    heads["xlarge"] = {"success": (2 * lw - mw, 2 * lb - mb),
                       "log_out_tokens": heads["large"]["log_out_tokens"], "extrapolated": True}

    def lin(wb):
        w, b = wb
        return {"w": [float(x) for x in np.asarray(w, dtype=np.float32)], "b": float(b)}

    probe_file = {
        "format": "opas-probe/2",
        "encoder": {
            "layer": int(best_layer), "pooling": best_pool, "thinking": bool(meta["thinking"]),
            "max_tokens": int(meta["max_tokens"]), "n_embd": int(last.shape[2]),
            "model": Path(meta["model"]).name,
        },
        "heads": {t: {"success": lin(h["success"]), "log_out_tokens": lin(h["log_out_tokens"]),
                      "extrapolated": h["extrapolated"]} for t, h in heads.items()},
        "metadata": {
            "trained": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "dataset": "LLMRouterBench performance-cost setting (https://huggingface.co/datasets/NPULH/LLMRouterBench)",
            "n_prompts": n,
            "tiers": tiers,
            "label": "tier success = fraction of the tier's models with score >= 0.5; tokens = their mean output tokens",
            "xlarge": "extrapolated: large + (large - medium) in log-odds; large's token head",
            "pipeline": f"standardize -> PCA({PCA_DIM}) -> weighted logistic(C={best_C}) / ridge(alpha={best_alpha}) on log1p(tokens), collapsed",
            "heldout_test": test,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe_file))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    write_report(args.report, meta, tiers, rate, n, idx_train, idx_test, results, sweep, grid,
                 best_layer, best_pool, best_C, best_alpha, test, singles, routing, oracle,
                 price_per_token, args.out)


def write_report(path, meta, tiers, rate, n, idx_train, idx_test, baselines, sweep, grid,
                 best_layer, best_pool, best_C, best_alpha, test, singles, routing, oracle, ppt, out):
    L = ["# Probe results: LLMRouterBench, Qwen3.5-9B activations\n"]
    L.append(f"Generated by `training/train_probe.py` on {dt.date.today().isoformat()}: {n} prompts "
             f"({len(idx_train)} train / {len(idx_test)} held-out test, stratified by dataset), encoder "
             f"`{Path(meta['model']).name}`, prompts truncated to {meta['max_tokens']} tokens, thinking={meta['thinking']}.\n")
    L.append("## Tiers\n")
    L.append("The 13 benchmark models are ranked by overall success on all 12,446 LLMRouterBench prompts and split into thirds. "
             "A tier's label on a prompt is the fraction of its models that answered it (score ≥ 0.5); its token target is their mean output tokens.\n")
    L.append("| Tier | Models (overall success) |\n|---|---|")
    for t in TRAINED_TIERS:
        L.append(f"| {t} | " + ", ".join(f"{m} ({rate[m]:.2f})" for m in tiers[t]) + " |")
    L.append("| xlarge | none; head extrapolated as large + (large − medium) in log-odds |\n")
    L.append("**Metrics.** *AUROC*: tier prediction against each member model's success, averaged over tiers. "
             "*Within-dataset AUROC*: the same inside each benchmark, averaged; how well difficulty is ranked beyond knowing the benchmark. "
             "*Tokens ρ*: Spearman correlation of predicted and actual mean output tokens.\n")
    L.append("## Cross-validation on the training split (5 folds)\n")
    L.append("| Features | AUROC | Within-dataset AUROC | Tokens ρ |\n|---|---|---|---|")
    for name, (a, w, r) in baselines.items():
        L.append(f"| baseline: {name} | {a:.3f} | {w:.3f} | {r:.3f} |")
    for layer, pool, a, w, r in sweep:
        mark = " **(chosen)**" if (layer, pool) == (best_layer, best_pool) else ""
        L.append(f"| layer {layer}, {'last token' if pool == 'last' else 'mean of tokens'}{mark} | {a:.3f} | {w:.3f} | {r:.3f} |")
    L.append(f"\nRegularization: C ∈ {[g[0] for g in grid]} → C = {best_C}; ridge α = {best_alpha}.\n")
    L.append("## Held-out test split\n")
    L.append(f"AUROC {test['auroc']:.3f}, within-dataset AUROC {test['within_dataset_auroc']:.3f}, tokens ρ {test['tokens_spearman']:.3f}.\n")
    L.append("### Routing between small / medium / large\n")
    L.append("Each prompt goes to the tier with the best `p − λ·cost`, where cost = predicted output tokens × the tier's effective "
             f"USD per output token on the training split ({', '.join(f'{t}: ${v * 1e6:.2f}/M' for t, v in zip(TRAINED_TIERS, ppt))}). "
             "The outcome is the chosen tier's mean success and mean recorded cost on that prompt.\n")
    L.append("| Router | Success | Mean cost / request |\n|---|---|---|")
    for t, (s, c) in singles.items():
        L.append(f"| always {t} | {s:.3f} | ${c:.5f} |")
    L.append(f"| oracle (best tier per prompt, cheapest among ties) | {oracle[0]:.3f} | ${oracle[1]:.5f} |\n")
    L.append("| λ | Probe: success @ cost | Tier share (S/M/L) | Random routing at the same cost | Best single tier at ≤ that cost | Dataset-identity router |\n|---|---|---|---|---|---|")
    for r in routing:
        bs = r["best_single_within_cost"]
        share = "/".join(f"{r['share'][t]:.2f}" for t in TRAINED_TIERS)
        L.append(f"| {r['lambda']:g} | {r['success']:.3f} @ ${r['cost']:.5f} | {share} | {r['random_same_cost']:.3f} | "
                 f"{'—' if bs is None else f'{bs:.3f}'} | {r['dataset_router'][0]:.3f} @ ${r['dataset_router'][1]:.5f} |")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    L.append(f"\nThe exported probe (`{shown}`, committed to the repo) is refit on all {n} prompts. The xlarge head is not evaluated: "
             "the benchmark has no model of that class.\n")
    path.write_text("\n".join(L))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
