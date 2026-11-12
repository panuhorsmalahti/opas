"""Train opastin's activation probe on LLMRouterBench and evaluate it as a router.

Inputs (see README "Training the probe"):
  data/llmrouterbench/labels.jsonl   from prepare_llmrouterbench.py
  data/llmrouterbench/acts-9b/       from `opastin-encode --input sample.jsonl --out ...`

Per candidate model ("head") the probe predicts
  p_success = sigmoid(w·h + b)     logistic regression, label = score >= 0.5
  cost      = exp(w·h + b)         ridge regression on log(cost USD)
from the encoder's hidden state h at one layer. Each head is fit as
standardize -> PCA(100) -> linear model, then collapsed into a single weight vector over h
(all steps are linear), which is what the Rust ProbeRouter loads.

Steps:
  1. Split prompts 80/20, stratified by dataset (seeded).
  2. On the 80%: 5-fold CV over (layer, pooling) and regularization; compare against
     baselines (prompt length, TF-IDF, dataset identity).
  3. On the 20%: route with argmax p - lambda*cost and compare with random routing at
     equal cost, single models, and the oracle.
  4. Refit the chosen configuration on all prompts and export the probe JSON.

Usage: uv run train_probe.py [--out ../models/probes/llmrouterbench-qwen35-9b.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "llmrouterbench"
PCA_DIM = 100
COST_FLOOR = 1e-6  # some recorded costs are 0
SEED = 0


# --- data -----------------------------------------------------------------------------


def load(acts_dir: Path):
    meta = json.loads((acts_dir / "meta.json").read_text())
    ids = meta["ids"]
    labels = {}
    with open(DATA / "labels.jsonl") as f:
        for line in f:
            it = json.loads(line)
            labels[it["id"]] = it
    rows = [labels[i] for i in ids]
    models = sorted({m for r in rows for m in r["models"]})
    n, k = len(rows), len(models)
    success = np.full((n, k), np.nan)
    cost = np.full((n, k), np.nan)
    for i, r in enumerate(rows):
        for j, m in enumerate(models):
            if m in r["models"]:
                success[i, j] = float(r["models"][m]["score"] >= 0.5)
                cost[i, j] = max(r["models"][m]["cost"], 0.0)
    datasets = np.array([r["dataset"] for r in rows])
    prompts = [r["prompt"] for r in rows]
    last = np.load(acts_dir / "last.npy", mmap_mode="r")
    mean = np.load(acts_dir / "mean.npy", mmap_mode="r")
    n_tokens = np.load(acts_dir / "n_tokens.npy")
    return meta, models, success, cost, datasets, prompts, last, mean, n_tokens


# --- models ---------------------------------------------------------------------------


class LinearProbe:
    """standardize -> PCA -> logistic / ridge, for one feature matrix and many heads."""

    def __init__(self, pca_dim=PCA_DIM, C=0.1, alpha=10.0):
        self.pca_dim, self.C, self.alpha = pca_dim, C, alpha

    def fit(self, X, success, cost):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = PCA(n_components=min(self.pca_dim, Z.shape[0] - 1, Z.shape[1]), random_state=SEED).fit(Z)
        F = self.pca.transform(Z)
        self.success_heads, self.cost_heads = [], []
        for j in range(success.shape[1]):
            ok = ~np.isnan(success[:, j])
            clf = LogisticRegression(C=self.C, max_iter=2000).fit(F[ok], success[ok, j])
            reg = Ridge(alpha=self.alpha).fit(F[ok], np.log(cost[ok, j] + COST_FLOOR))
            self.success_heads.append(clf)
            self.cost_heads.append(reg)
        return self

    def features(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def predict(self, X):
        F = self.features(X)
        p = np.column_stack([h.predict_proba(F)[:, 1] for h in self.success_heads])
        c = np.column_stack([np.exp(h.predict(F)) for h in self.cost_heads])
        return p, c

    def collapse(self, j):
        """Head j as weight vectors over the raw features: logit = h·w + b."""
        mu, sd = self.scaler.mean_, self.scaler.scale_
        P, pm = self.pca.components_, self.pca.mean_

        def lin(coef, intercept):
            v = P.T @ coef
            return v / sd, float(intercept - (mu / sd + pm) @ v)

        sw, sb = lin(self.success_heads[j].coef_[0], self.success_heads[j].intercept_[0])
        cw, cb = lin(self.cost_heads[j].coef_, self.cost_heads[j].intercept_)
        return (sw, sb), (cw, cb)


def auroc_columns(y, p):
    """Mean AUROC over heads (columns), skipping heads with one class."""
    out = []
    for j in range(y.shape[1]):
        ok = ~np.isnan(y[:, j])
        if len(np.unique(y[ok, j])) == 2:
            out.append(roc_auc_score(y[ok, j], p[ok, j]))
    return float(np.mean(out))


def within_dataset_auroc(y, p, datasets):
    """AUROC computed inside each dataset, averaged (weighted by size): how well the
    probe ranks difficulty beyond knowing which benchmark a prompt is from."""
    total, weight = 0.0, 0
    for ds in np.unique(datasets):
        m = datasets == ds
        vals = []
        for j in range(y.shape[1]):
            ok = m & ~np.isnan(y[:, j])
            if len(np.unique(y[ok, j])) == 2:
                vals.append(roc_auc_score(y[ok, j], p[ok, j]))
        if vals:
            total += np.mean(vals) * m.sum()
            weight += m.sum()
    return total / weight


def spearman_cost(c_true, c_pred):
    from scipy.stats import spearmanr

    vals = []
    for j in range(c_true.shape[1]):
        ok = ~np.isnan(c_true[:, j])
        vals.append(spearmanr(c_true[ok, j], c_pred[ok, j]).statistic)
    return float(np.mean(vals))


# --- evaluation -----------------------------------------------------------------------


def cv_score(make_X, idx, success, cost, datasets, C, alpha, folds=5):
    """5-fold CV on rows `idx`; returns mean AUROC, within-dataset AUROC, cost Spearman."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    P = np.zeros((len(idx), success.shape[1]))
    Cp = np.zeros_like(P)
    for tr, va in skf.split(idx, datasets[idx]):
        Xtr, Xva = make_X(idx[tr]), make_X(idx[va])
        probe = LinearProbe(C=C, alpha=alpha).fit(Xtr, success[idx[tr]], cost[idx[tr]])
        P[va], Cp[va] = probe.predict(Xva)
    y, c = success[idx], cost[idx]
    return auroc_columns(y, P), within_dataset_auroc(y, P, datasets[idx]), spearman_cost(c, Cp)


def route_eval(p, c, success, cost, cand, lambdas):
    """Route among candidate columns `cand`; return list of (lambda, success, cost, share)."""
    out = []
    for lam in lambdas:
        score = p[:, cand] - lam * c[:, cand]
        pick = np.array(cand)[np.argmax(score, axis=1)]
        rows = np.arange(len(pick))
        s, k = success[rows, pick], cost[rows, pick]
        ok = ~np.isnan(s)
        share = {m: float(np.mean(pick == m)) for m in cand}
        out.append((lam, float(np.mean(s[ok])), float(np.mean(k[ok])), share))
    return out


def random_at_cost(success, cost, cand, target_cost):
    """Expected success of random routing between the cheapest and priciest candidate
    with the mix that matches target_cost (the random baseline at equal cost)."""
    mean_c = {j: np.nanmean(cost[:, j]) for j in cand}
    mean_s = {j: np.nanmean(success[:, j]) for j in cand}
    lo, hi = min(cand, key=lambda j: mean_c[j]), max(cand, key=lambda j: mean_c[j])
    q = np.clip((target_cost - mean_c[lo]) / (mean_c[hi] - mean_c[lo]), 0, 1)
    return float(q * mean_s[hi] + (1 - q) * mean_s[lo])


# --- main -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, default=DATA / "acts-9b")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "probes" / "llmrouterbench-qwen35-9b.json")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "probe-results.md")
    ap.add_argument("--layer-step", type=int, default=2, help="sweep every Nth captured layer")
    args = ap.parse_args()

    meta, models, success, cost, datasets, prompts, last, mean, n_tokens = load(args.acts)
    layers = meta["layers"]
    n = len(datasets)
    print(f"{n} prompts, {len(models)} models, {len(layers)} captured layers, hidden {last.shape[2]}")

    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=SEED, stratify=datasets)
    idx_train.sort()
    idx_test.sort()

    results = {}

    # Baselines.
    ohe = OneHotEncoder().fit(datasets.reshape(-1, 1))
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True).fit([prompts[i] for i in idx_train])
    baselines = {
        "length (log tokens)": lambda ix: np.log(n_tokens[ix]).reshape(-1, 1).astype(float),
        "dataset identity": lambda ix: ohe.transform(datasets[ix].reshape(-1, 1)).toarray(),
        "TF-IDF": lambda ix: tfidf.transform([prompts[i] for i in ix]),
    }
    for name, fx in baselines.items():
        results[name] = cv_score_baseline(fx, idx_train, success, cost, datasets, sparse=(name == "TF-IDF"))
        print(f"baseline {name:<22} AUROC {results[name][0]:.3f}  within-dataset {results[name][1]:.3f}  cost rho {results[name][2]:.3f}")

    # Activation sweep: every Nth layer, both poolings, then refine C around the best.
    sweep = []
    for k in range(0, len(layers), args.layer_step):
        for pool, arr in (("last", last), ("mean", mean)):
            fx = lambda ix, arr=arr, k=k: np.asarray(arr[ix, k, :], dtype=np.float32)
            r = cv_score(fx, idx_train, success, cost, datasets, C=0.1, alpha=10.0)
            sweep.append((layers[k], pool, *r))
            print(f"layer {layers[k]:>2} {pool:<4}  AUROC {r[0]:.3f}  within-dataset {r[1]:.3f}  cost rho {r[2]:.3f}", flush=True)
    best = max(sweep, key=lambda t: t[3])  # by within-dataset AUROC: difficulty, not benchmark identity
    best_layer, best_pool = best[0], best[1]
    k_best = layers.index(best_layer)
    arr_best = last if best_pool == "last" else mean
    fx_best = lambda ix: np.asarray(arr_best[ix, k_best, :], dtype=np.float32)
    grid = []
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        r = cv_score(fx_best, idx_train, success, cost, datasets, C=C, alpha=10.0)
        grid.append((C, *r))
        print(f"C={C:<5} AUROC {r[0]:.3f}  within-dataset {r[1]:.3f}", flush=True)
    best_C = max(grid, key=lambda t: t[2])[0]
    alpha_grid = []
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        r = cv_score(fx_best, idx_train, success, cost, datasets, C=best_C, alpha=alpha)
        alpha_grid.append((alpha, r[2]))
    best_alpha = max(alpha_grid, key=lambda t: t[1])[0]
    print(f"chosen: layer {best_layer} {best_pool}, C={best_C}, alpha={best_alpha}")

    # Held-out test: fit on train, evaluate on test.
    probe = LinearProbe(C=best_C, alpha=best_alpha).fit(fx_best(idx_train), success[idx_train], cost[idx_train])
    p_te, c_te = probe.predict(fx_best(idx_test))
    y_te, k_te, d_te = success[idx_test], cost[idx_test], datasets[idx_test]
    test_metrics = {
        "auroc": auroc_columns(y_te, p_te),
        "within_dataset_auroc": float(within_dataset_auroc(y_te, p_te, d_te)),
        "cost_spearman": spearman_cost(k_te, c_te),
    }
    # Dataset-identity router on the same split, for comparison.
    ds_probe = LinearProbe(pca_dim=10**6, C=1.0, alpha=1.0)
    Xd = ohe.transform(datasets.reshape(-1, 1)).toarray()
    ds_probe.fit(Xd[idx_train], success[idx_train], cost[idx_train])
    p_ds, c_ds = ds_probe.predict(Xd[idx_test])
    print("test:", {k: round(float(v), 3) for k, v in test_metrics.items()})

    routing = routing_report(models, p_te, c_te, p_ds, c_ds, y_te, k_te)

    # Final probe on all prompts.
    final = LinearProbe(C=best_C, alpha=best_alpha).fit(fx_best(np.arange(n)), success, cost)
    heads = {}
    Xcheck = fx_best(idx_test[:50])
    p_chk, c_chk = final.predict(Xcheck)
    for j, m in enumerate(models):
        (sw, sb), (cw, cb) = final.collapse(j)
        # The collapsed linear map must reproduce the pipeline exactly.
        p_lin = 1 / (1 + np.exp(-(Xcheck.astype(np.float64) @ sw + sb)))
        c_lin = np.exp(Xcheck.astype(np.float64) @ cw + cb)
        assert np.allclose(p_lin, p_chk[:, j], atol=1e-4), m
        assert np.allclose(c_lin, c_chk[:, j], rtol=1e-3), m
        heads[m] = {
            "success": {"w": [float(x) for x in sw.astype(np.float32)], "b": sb},
            "log_cost": {"w": [float(x) for x in cw.astype(np.float32)], "b": cb},
        }

    probe_file = {
        "format": "opastin-probe/1",
        "encoder": {
            "layer": int(best_layer),
            "pooling": best_pool,
            "thinking": bool(meta["thinking"]),
            "max_tokens": int(meta["max_tokens"]),
            "n_embd": int(last.shape[2]),
            "model": Path(meta["model"]).name,
        },
        "heads": heads,
        "metadata": {
            "trained": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "dataset": "LLMRouterBench performance-cost setting (https://huggingface.co/datasets/NPULH/LLMRouterBench)",
            "n_prompts": n,
            "label": "success = score >= 0.5; cost = recorded USD per answer",
            "pipeline": f"standardize -> PCA({PCA_DIM}) -> logistic(C={best_C}) / ridge(alpha={best_alpha}) on log cost, collapsed",
            "heldout_test": test_metrics,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe_file))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    write_report(args.report, meta, models, n, idx_train, idx_test, results, sweep, grid,
                 best_layer, best_pool, best_C, best_alpha, test_metrics, routing, args.out)


class SparseLinear:
    """Logistic / ridge directly on sparse features (TF-IDF baseline)."""

    def fit(self, X, success, cost):
        self.s, self.c = [], []
        for j in range(success.shape[1]):
            ok = ~np.isnan(success[:, j])
            self.s.append(LogisticRegression(C=1.0, max_iter=2000).fit(X[ok], success[ok, j]))
            self.c.append(Ridge(alpha=1.0).fit(X[ok], np.log(cost[ok, j] + COST_FLOOR)))
        return self

    def predict(self, X):
        p = np.column_stack([h.predict_proba(X)[:, 1] for h in self.s])
        c = np.column_stack([np.exp(h.predict(X)) for h in self.c])
        return p, c


def cv_score_baseline(fx, idx, success, cost, datasets, sparse=False):
    """CV for baseline feature sets (small dense features without PCA, or sparse)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    P = np.zeros((len(idx), success.shape[1]))
    Cp = np.zeros_like(P)
    for tr, va in skf.split(idx, datasets[idx]):
        probe = SparseLinear() if sparse else LinearProbe(pca_dim=10**6, C=1.0, alpha=1.0)
        probe.fit(fx(idx[tr]), success[idx[tr]], cost[idx[tr]])
        P[va], Cp[va] = probe.predict(fx(idx[va]))
    y, c = success[idx], cost[idx]
    return auroc_columns(y, P), within_dataset_auroc(y, P, datasets[idx]), spearman_cost(c, Cp)


def routing_report(models, p, c, p_ds, c_ds, y, k):
    j = {m: i for i, m in enumerate(models)}
    pairs = {
        "gpt-5 vs qwen3-235b-a22b-2507": [j["gpt-5"], j["qwen3-235b-a22b-2507"]],
        "gemini-2.5-pro vs deepseek-v3-0324": [j["gemini-2.5-pro"], j["deepseek-v3-0324"]],
        "all 13 models": list(range(len(models))),
    }
    lambdas = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    report = {}
    for name, cand in pairs.items():
        probe_curve = route_eval(p, c, y, k, cand, lambdas)
        ds_curve = route_eval(p_ds, c_ds, y, k, cand, lambdas)
        singles = {models[i]: (float(np.nanmean(y[:, i])), float(np.nanmean(k[:, i]))) for i in cand}
        # Oracle: the cheapest candidate that succeeds (or the cheapest overall if none does).
        oracle_s, oracle_c = [], []
        for r in range(len(y)):
            ok = [i for i in cand if y[r, i] == 1]
            pool = ok or [i for i in cand if not np.isnan(k[r, i])]
            best = min(pool, key=lambda i: k[r, i])
            oracle_s.append(1.0 if ok else 0.0)
            oracle_c.append(k[r, best])
        rows = []
        for (lam, s, cst, share), (_, s_ds, c_ds_, _) in zip(probe_curve, ds_curve):
            affordable = [v[0] for v in singles.values() if v[1] <= cst * 1.0001]
            rows.append({
                "lambda": lam, "success": s, "cost": cst,
                "random_same_cost": random_at_cost(y, k, cand, cst),
                "best_single_within_cost": max(affordable) if affordable else None,
                "dataset_router_success": s_ds, "dataset_router_cost": c_ds_,
                "share": {models[i]: v for i, v in share.items()},
            })
        report[name] = {
            "rows": rows, "singles": singles,
            "oracle": (float(np.mean(oracle_s)), float(np.mean(oracle_c))),
        }
        print(f"\nrouting: {name}")
        for m, (s, cst) in singles.items():
            print(f"  always {m:<30} success {s:.3f}  cost ${cst:.5f}")
        for row in rows:
            print(f"  lambda {row['lambda']:>5}: success {row['success']:.3f}  cost ${row['cost']:.5f}"
                  f"  | random at same cost {row['random_same_cost']:.3f}"
                  f"  | best single <= cost {row['best_single_within_cost'] if row['best_single_within_cost'] is None else round(row['best_single_within_cost'], 3)}"
                  f"  | dataset-router {row['dataset_router_success']:.3f} @ ${row['dataset_router_cost']:.5f}")
        print(f"  oracle: success {report[name]['oracle'][0]:.3f}  cost ${report[name]['oracle'][1]:.5f}")
    return report


def write_report(path, meta, models, n, idx_train, idx_test, baselines, sweep, grid,
                 best_layer, best_pool, best_C, best_alpha, test, routing, out):
    L = []
    L.append("# Probe results: LLMRouterBench, Qwen3.5-9B activations\n")
    L.append(f"Generated by `training/train_probe.py` on {dt.date.today().isoformat()}. "
             f"{n} prompts ({len(idx_train)} train / {len(idx_test)} held-out test, stratified by dataset), "
             f"{len(models)} target models, encoder `{Path(meta['model']).name}`, prompts truncated to "
             f"{meta['max_tokens']} tokens, thinking={meta['thinking']}.\n")
    L.append("**Label:** success = benchmark score ≥ 0.5 (ArenaHard ties count as success); cost = recorded USD per answer.\n")
    L.append("**Metrics:** *AUROC* averages each model's success AUROC. *Within-dataset AUROC* computes it inside each "
             "benchmark and averages: how well difficulty is ranked beyond knowing the benchmark. *Cost ρ* is the "
             "Spearman correlation of predicted and actual cost.\n")
    L.append("## Cross-validation on the training split (5 folds)\n")
    L.append("| Features | AUROC | Within-dataset AUROC | Cost ρ |\n|---|---|---|---|")
    for name, (a, w, r) in baselines.items():
        L.append(f"| baseline: {name} | {a:.3f} | {w:.3f} | {r:.3f} |")
    for layer, pool, a, w, r in sweep:
        mark = " **(chosen)**" if (layer, pool) == (best_layer, best_pool) else ""
        L.append(f"| layer {layer}, {pool} token{'s' if pool == 'mean' else ''}{mark} | {a:.3f} | {w:.3f} | {r:.3f} |")
    L.append(f"\nRegularization: C ∈ {[g[0] for g in grid]} → C = {best_C}; ridge α = {best_alpha}.\n")
    L.append("## Held-out test split\n")
    L.append(f"AUROC {test['auroc']:.3f}, within-dataset AUROC {test['within_dataset_auroc']:.3f}, "
             f"cost ρ {test['cost_spearman']:.3f}.\n")
    for name, rep in routing.items():
        L.append(f"### Routing: {name}\n")
        L.append("| Router | Success | Mean cost / request |\n|---|---|---|")
        for m, (s, c) in rep["singles"].items():
            L.append(f"| always {m} | {s:.3f} | ${c:.5f} |")
        for row in rep["rows"]:
            L.append(f"| probe, λ={row['lambda']:g} | {row['success']:.3f} | ${row['cost']:.5f} |")
        L.append(f"| oracle (cheapest model that succeeds) | {rep['oracle'][0]:.3f} | ${rep['oracle'][1]:.5f} |")
        L.append("\n| λ | Probe success @ cost | Random routing at the same cost | Best single model at ≤ that cost | Dataset-identity router (success @ cost) |\n|---|---|---|---|---|")
        for row in rep["rows"]:
            bs = row["best_single_within_cost"]
            L.append(f"| {row['lambda']:g} | {row['success']:.3f} @ ${row['cost']:.5f} | {row['random_same_cost']:.3f} | "
                     f"{'—' if bs is None else f'{bs:.3f}'} | "
                     f"{row['dataset_router_success']:.3f} @ ${row['dataset_router_cost']:.5f} |")
        L.append("")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    L.append(f"The exported probe (`{shown}`, git-ignored) is refit on all {n} prompts.\n")
    path.write_text("\n".join(L))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
