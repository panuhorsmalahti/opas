"""Does a bigger predictor beat the linear probe? Same activations, same split, same
selection rule; only the success predictor changes.

    cd training && uv run predictor_capacity.py

Reads the activations and labels used by train_probe.py (layer 18, last token: the
shipped probe's choice) and writes docs/predictor-capacity.md. Not used at runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from train_probe import (DATA, PRESETS, ROOT, SEED, TRAINED_TIERS, TierProbe, fit_weighted_logistic,
                         load, member_auroc, random_at_cost, weights_eval)

LAYER = 18
BOOTSTRAP = 300
POOL = "last"


# --- predictors: fit(X_train, frac_train) -> predict(X) gives (n, tiers) scores --------


class Prep:
    """Standardize, then optionally PCA to k dimensions (k=None keeps all 4096)."""

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pca = PCA(n_components=self.k, random_state=SEED).fit(Z) if self.k else None
        return self

    def __call__(self, X):
        Z = self.sc.transform(X)
        return self.pca.transform(Z) if self.pca is not None else Z


class Linear:
    """The shipped design: per-tier weighted logistic regression."""

    def __init__(self, k, C):
        self.k, self.C = k, C

    def name(self):
        return f"linear, PCA {self.k or 4096}" if self.k else "linear, all 4096 dims"

    def params(self, d):
        return len(TRAINED_TIERS) * ((self.k or d) + 1)

    def fit(self, X, frac):
        self.prep = Prep(self.k).fit(X)
        F = self.prep(X)
        self.heads = [fit_weighted_logistic(F, frac[:, t], self.C) for t in range(frac.shape[1])]
        return self

    def predict(self, X):
        F = self.prep(X)
        return np.column_stack([h.predict_proba(F)[:, 1] for h in self.heads])


class MLP:
    """Shared-trunk network (as in the NVIDIA routing paper): hidden layers shared across
    tiers, one sigmoid-free output per tier, trained on success fractions (MSE)."""

    def __init__(self, k, hidden, alpha):
        self.k, self.hidden, self.alpha = k, hidden, alpha

    def name(self):
        return f"MLP {'x'.join(map(str, self.hidden))}, {'PCA ' + str(self.k) if self.k else 'all 4096 dims'}"

    def params(self, d):
        sizes = [self.k or d, *self.hidden, len(TRAINED_TIERS)]
        return sum(a * b + b for a, b in zip(sizes, sizes[1:]))

    def fit(self, X, frac):
        self.prep = Prep(self.k).fit(X)
        self.net = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=20, max_iter=500,
                                learning_rate_init=1e-3, random_state=SEED).fit(self.prep(X), frac)
        return self

    def predict(self, X):
        return np.clip(self.net.predict(self.prep(X)), 1e-4, 1 - 1e-4)


class Boosted:
    """Gradient-boosted trees per tier on PCA features."""

    def __init__(self, k, depth, l2):
        self.k, self.depth, self.l2 = k, depth, l2

    def name(self):
        return f"boosted trees (depth {self.depth}), PCA {self.k}"

    def params(self, d):
        # Leaves are the learned values: up to 2^depth per tree, 300 trees, per tier.
        return len(TRAINED_TIERS) * 300 * 2 ** self.depth

    def fit(self, X, frac):
        self.prep = Prep(self.k).fit(X)
        F = self.prep(X)
        self.heads = [HistGradientBoostingRegressor(max_iter=300, max_depth=self.depth, learning_rate=0.05,
                                                    l2_regularization=self.l2, early_stopping=True,
                                                    random_state=SEED).fit(F, frac[:, t])
                      for t in range(frac.shape[1])]
        return self

    def predict(self, X):
        F = self.prep(X)
        return np.clip(np.column_stack([h.predict(F) for h in self.heads]), 1e-4, 1 - 1e-4)


def families():
    """Each family is a list of configurations; CV on the training split picks one."""
    return {
        "linear, PCA 16": [Linear(16, C) for C in (0.01, 0.1, 1.0)],
        "linear, PCA 32": [Linear(32, C) for C in (0.01, 0.1, 1.0)],
        "linear, PCA 100 (previous probe)": [Linear(100, C) for C in (0.003, 0.01, 0.03, 0.1)],
        "linear, PCA 256": [Linear(256, C) for C in (0.001, 0.003, 0.01, 0.03)],
        "linear, PCA 1024": [Linear(1024, C) for C in (0.0003, 0.001, 0.003, 0.01, 0.03)],
        "linear, all 4096 dims (shipped)": [Linear(None, C) for C in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)],
        "MLP 64, PCA 100": [MLP(100, (64,), a) for a in (0.1, 1.0, 10.0, 30.0, 100.0)],
        "MLP 256, PCA 100": [MLP(100, (256,), a) for a in (0.1, 1.0, 10.0, 30.0, 100.0)],
        "MLP 256x256, PCA 100": [MLP(100, (256, 256), a) for a in (0.1, 1.0, 10.0, 30.0, 100.0)],
        "MLP 512, all 4096 dims": [MLP(None, (512,), a) for a in (1.0, 10.0, 100.0)],
        "MLP 1024x256, all 4096 dims": [MLP(None, (1024, 256), a) for a in (1.0, 10.0, 100.0)],
        "boosted trees, PCA 100": [Boosted(100, d, l2) for d in (2, 4) for l2 in (0.0, 10.0)],
    }


# --- evaluation -----------------------------------------------------------------------


def cv_within(model_fn, X, frac, ms, ds, folds):
    P = np.zeros_like(frac)
    for tr, va in folds:
        P[va] = model_fn().fit(X[tr], frac[tr]).predict(X[va])
    return member_auroc(P, ms, ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, default=DATA / "acts-9b")
    ap.add_argument("--report", type=Path, default=ROOT / "docs" / "predictor-capacity.md")
    ap.add_argument("--only", nargs="*", help="run only these families (by name)")
    args = ap.parse_args()

    (meta, tiers, rate, frac, tokens, cost, member_success, datasets, prompts,
     last, mean, n_tokens) = load(args.acts)
    k = meta["layers"].index(LAYER)
    X = np.asarray((last if POOL == "last" else mean)[:, k, :], dtype=np.float32)
    n, d = X.shape
    idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2, random_state=SEED, stratify=datasets)
    idx_train.sort()
    idx_test.sort()
    tr_ms = [s[idx_train] for s in member_success]
    te_ms = [s[idx_test] for s in member_success]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx_train, datasets[idx_train]))
    price = cost[idx_train].sum(0) / np.maximum(tokens[idx_train].sum(0), 1)

    # Token predictions for routing are the previous probe's ridge heads (PCA 100) for every row, so
    # only the success predictor differs between rows.
    tok = TierProbe(pca_dim=100, C=0.01, alpha=1000.0).fit(X[idx_train], frac[idx_train], tokens[idx_train])
    _, k_te = tok.predict(X[idx_test])

    rows = []
    fams = families()
    for fam, configs in fams.items():
        if args.only and fam not in args.only:
            continue
        t0 = time.time()
        scored = []
        for cfg in configs:
            s = cv_within(lambda cfg=cfg: type(cfg)(**vars(cfg)), X[idx_train], frac[idx_train], tr_ms,
                          datasets[idx_train], folds)
            scored.append((s, cfg))
        cv, best = max(scored, key=lambda t: t[0])
        model = type(best)(**vars(best)).fit(X[idx_train], frac[idx_train])
        p_tr, p_te = model.predict(X[idx_train]), model.predict(X[idx_test])
        succ, c, _ = weights_eval(p_te, k_te, frac[idx_test], cost[idx_test], price, PRESETS["balanced"])
        qsucc, qc, _ = weights_eval(p_te, k_te, frac[idx_test], cost[idx_test], price, PRESETS["quality"])
        row = dict(
            family=fam, config={k: v for k, v in vars(best).items() if k != "k"}, params=best.params(d),
            cv_within=cv,
            train_auroc=member_auroc(p_tr, tr_ms), test_auroc=member_auroc(p_te, te_ms),
            train_within=member_auroc(p_tr, tr_ms, datasets[idx_train]),
            test_within=member_auroc(p_te, te_ms, datasets[idx_test]),
            balanced=(succ, c, random_at_cost(frac[idx_test], cost[idx_test], c)),
            quality=(qsucc, qc, random_at_cost(frac[idx_test], cost[idx_test], qc)),
            seconds=time.time() - t0,
            p_te=p_te,
        )
        rows.append(row)
        print(f"{fam:<30} params {row['params']:>10,}  CV within {cv:.3f}  test AUROC {row['test_auroc']:.3f} "
              f"within {row['test_within']:.3f}  train within {row['train_within']:.3f}  "
              f"balanced {succ:.3f}@${c:.4f}  ({row['seconds']:.0f}s) {row['config']}", flush=True)

    # Paired bootstrap over test prompts: each predictor minus the previous probe (PCA 100).
    ship = next((r for r in rows if r["family"].endswith("(previous probe)")), None)
    if ship is not None:
        rng = np.random.default_rng(SEED)
        te_ds = datasets[idx_test]
        boots = [rng.integers(0, len(idx_test), len(idx_test)) for _ in range(BOOTSTRAP)]

        def stats(p, b):
            ms_b = [s[b] for s in te_ms]
            succ, _, _ = weights_eval(p[b], k_te[b], frac[idx_test][b], cost[idx_test][b], price, PRESETS["balanced"])
            return member_auroc(p[b], ms_b, te_ds[b]), succ

        base = [stats(ship["p_te"], b) for b in boots]
        for r in rows:
            if r is ship:
                continue
            diff = np.array([np.subtract(stats(r["p_te"], b), s0) for b, s0 in zip(boots, base)])
            r["ci_within"] = tuple(np.percentile(diff[:, 0], [2.5, 97.5]))
            r["ci_balanced"] = tuple(np.percentile(diff[:, 1], [2.5, 97.5]))
            print(f"bootstrap {r['family']:<30} within {r['test_within'] - ship['test_within']:+.3f} "
                  f"[{r['ci_within'][0]:+.3f}, {r['ci_within'][1]:+.3f}]  balanced success "
                  f"{r['balanced'][0] - ship['balanced'][0]:+.3f} "
                  f"[{r['ci_balanced'][0]:+.3f}, {r['ci_balanced'][1]:+.3f}]", flush=True)

    # Learning curve: does the bigger model catch up with more data? Same chosen configs,
    # trained on a stratified subsample of the training split, evaluated on the test split.
    curve = []
    by_name = {r["family"]: r for r in rows}
    curve_fams = [f for f in ("linear, PCA 100 (previous probe)", "linear, all 4096 dims (shipped)", "MLP 256, PCA 100",
                              "MLP 1024x256, all 4096 dims") if f in by_name]
    for f in curve_fams:
        cfg_cls = type(fams[f][0])
        cfg = dict(by_name[f]["config"])
        cfg["k"] = fams[f][0].k
        for share in (0.125, 0.25, 0.5, 1.0):
            vals = []
            for rep in range(3 if share < 1 else 1):
                if share < 1:
                    sub, _ = train_test_split(idx_train, train_size=share, random_state=SEED + rep,
                                              stratify=datasets[idx_train])
                else:
                    sub = idx_train
                p = cfg_cls(**cfg).fit(X[sub], frac[sub]).predict(X[idx_test])
                vals.append(member_auroc(p, te_ms, datasets[idx_test]))
            curve.append((f, share, int(round(share * len(idx_train))), float(np.mean(vals))))
            print(f"curve {f:<30} n={curve[-1][2]:>5}  test within {curve[-1][3]:.3f}", flush=True)

    write_report(args.report, n, d, idx_train, idx_test, frac, cost, rows, curve)
    print(f"wrote {args.report}")


def write_report(path, n, d, idx_train, idx_test, frac, cost, rows, curve):
    always = {t: (frac[idx_test, i].mean(), cost[idx_test, i].mean()) for i, t in enumerate(TRAINED_TIERS)}
    L = [
        "# Predictor capacity: is a bigger predictor better?",
        "",
        f"Generated by `training/predictor_capacity.py` on {dt.date.today()}. Same data as "
        f"[probe-results.md](probe-results.md): {n} LLMRouterBench prompts, Qwen3.5-9B layer {LAYER} "
        f"{POOL}-token activations ({d} dims), the same {len(idx_train)} / {len(idx_test)} train / test split.",
        "",
        "Only the success predictor changes between rows. Each family's settings "
        "(regularization, depth) are chosen by 5-fold cross-validation on the training split, scored by "
        "within-benchmark AUROC, as for the shipped probe. Then the chosen model is refit on the "
        "whole training split and scored once on the test split.",
        "",
        "- **Learned parameters:** numbers fitted to the success labels (PCA is fitted without labels "
        "and not counted).",
        "- **AUROC:** tier prediction vs. each member model's success. **Within:** the same inside "
        "each benchmark, averaged; ranking difficulty beyond recognizing the benchmark.",
        "- **Train − test gap:** within-benchmark AUROC on the training data minus on the test data; "
        "large gaps mean memorization.",
        "- **Routing:** opas's weighted scoring on the test split (token predictions are the previous probe's "
        "ridge heads for every row). *Random* is random routing at the same mean cost.",
        "",
        "| Predictor | Learned parameters | CV within | Test AUROC | Test within | Train − test gap | "
        "Balanced: success @ cost (random) | Quality: success @ cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b, q = r["balanced"], r["quality"]
        L.append(f"| {r['family']} | {r['params']:,} | {r['cv_within']:.3f} | {r['test_auroc']:.3f} | "
                 f"{r['test_within']:.3f} | {r['train_within'] - r['test_within']:+.3f} | "
                 f"{b[0]:.3f} @ ${b[1]:.4f} ({b[2]:.3f}) | {q[0]:.3f} @ ${q[1]:.4f} |")
    ship = next((r for r in rows if r["family"].endswith("(previous probe)")), None)
    if ship is not None and any("ci_within" in r for r in rows):
        L += ["", "## Compared with the previous probe (PCA 100)", "",
              f"Difference from the previous probe (PCA 100) on the test split, with 95% intervals from a paired bootstrap "
              f"over the {len(idx_test)} test prompts ({BOOTSTRAP} resamples). An interval that contains 0 "
              "means the test split cannot tell the two apart.", "",
              "| Predictor | Learned parameters | Test within − previous | Balanced success − previous |",
              "|---|---|---|---|"]
        for r in rows:
            if "ci_within" not in r:
                continue
            L.append(f"| {r['family']} | {r['params']:,} | {r['test_within'] - ship['test_within']:+.3f} "
                     f"[{r['ci_within'][0]:+.3f}, {r['ci_within'][1]:+.3f}] | "
                     f"{r['balanced'][0] - ship['balanced'][0]:+.3f} "
                     f"[{r['ci_balanced'][0]:+.3f}, {r['ci_balanced'][1]:+.3f}] |")
    L += ["", "Single tiers on the test split: " + ", ".join(
        f"always {t} {s:.3f} @ ${c:.4f}" for t, (s, c) in always.items()) + ".", ""]
    if curve:
        shares = sorted({c[1] for c in curve})
        L += ["## Learning curve", "",
              "Test within-benchmark AUROC when training on a stratified subsample of the training split "
              "(mean of 3 subsamples below 100%), same chosen settings:", "",
              "| Predictor | " + " | ".join(f"{next(c[2] for c in curve if c[1] == s)} prompts" for s in shares) + " |",
              "|---|" + "---|" * len(shares)]
        for f in dict.fromkeys(c[0] for c in curve):
            vals = {c[1]: c[3] for c in curve if c[0] == f}
            L.append(f"| {f} | " + " | ".join(f"{vals[s]:.3f}" for s in shares) + " |")
        L.append("")
    L.append("Chosen settings: " + "; ".join(f"{r['family']}: {r['config']}" for r in rows) + ".")
    L.append("")
    L.append("Search ranges: " + "; ".join(
        f"{fam}: " + ", ".join(str({k: v for k, v in vars(c).items() if k != 'k'}) for c in cfgs)
        for fam, cfgs in families().items()) + ".")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
