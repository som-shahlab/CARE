#!/usr/bin/env python3
"""Compute α-sweep with 95% CIs via random resplits.

For each dataset and α ∈ sweep, runs N_RESPLITS random 70/30 cal/test splits,
computes calibrated thresholds and test-set violation / workload / recall for
three methods (2D Joint, 1D-Importance, Union-Bound) and factuality.

Because α is the only varying parameter per split, we precompute the full loss
and workload grids once per split and query them for each α — avoiding a 8×
blowup relative to compute_resplit_cis.py.

Outputs alpha_sweep_cis.json with per-α, per-dataset mean ± 95% CI.

Usage:
  python3 compute_alpha_sweep_cis.py
  python3 compute_alpha_sweep_cis.py --datasets ACI_Bench
  python3 compute_alpha_sweep_cis.py --alphas 0.05,0.10,0.15,0.20
"""
import argparse
import json
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from care.config import (
    CANONICAL_DATA_ROOT as DATA_ROOT,
    DATASETS,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)


SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHAS_DEFAULT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
GRID_FACT = 0.01
GRID_OMIT = 0.05


def _summarize(vals):
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
        "median": float(np.median(vals)),
    }


# ---------------------------------------------------------------------------
# Precomputed grids (per split): identical numerics to compute_resplit_cis.py
# ---------------------------------------------------------------------------
def precompute_factuality_grid(docs, thresholds):
    """Return (loss_mat, flagged_mat, tp_mat, halluc_count).

    loss_mat[d,t]   = 1 if any hallucination unflagged in doc d at threshold t
    flagged_mat     = # flagged sentences in doc d at threshold t
    tp_mat          = # true-positive flags in doc d at threshold t
    halluc_count[d] = # hallucinations in doc d
    """
    n_t, n_d = len(thresholds), len(docs)
    loss_mat = np.zeros((n_d, n_t))
    flagged_mat = np.zeros((n_d, n_t), dtype=int)
    tp_mat = np.zeros((n_d, n_t), dtype=int)
    halluc_count = np.zeros(n_d, dtype=int)
    for i, doc in enumerate(docs):
        labels = np.array(doc["factuality_labels"])
        probs = np.array(doc["factuality_probs"])
        if len(labels) == 0:
            continue
        halluc = labels == 0
        halluc_count[i] = int(halluc.sum())
        flagged = probs[:, None] <= thresholds[None, :]  # [n_sent, n_t]
        flagged_mat[i] = flagged.sum(axis=0)
        any_missed = (halluc[:, None] & ~flagged).any(axis=0)
        loss_mat[i] = any_missed.astype(float)
        tp_mat[i] = (flagged & halluc[:, None]).sum(axis=0)
    return loss_mat, flagged_mat, tp_mat, halluc_count


def precompute_omission_grid(docs, tau_vals, gamma_vals):
    """Return (loss_mat, wl_mat, tp_mat, n_true) on full 2D (tau, gamma) grid.

    loss_mat[d, ti, gi]  = fractional omission loss for doc d at (tau,gamma)
    wl_mat[d, ti, gi]    = # surfaced sentences for doc d at (tau,gamma)
    tp_mat[d, ti, gi]    = # surfaced true omissions (for pooled recall)
    n_true[d]            = # true omissions in doc d
    """
    n_tau, n_gamma, n_d = len(tau_vals), len(gamma_vals), len(docs)
    loss_mat = np.zeros((n_d, n_tau, n_gamma))
    wl_mat = np.zeros((n_d, n_tau, n_gamma), dtype=int)
    tp_mat = np.zeros((n_d, n_tau, n_gamma), dtype=int)
    n_true = np.zeros(n_d, dtype=int)
    for i, doc in enumerate(docs):
        imp_probs = np.array(doc["importance_probs"])
        cov_probs = np.array(doc["coverage_probs"])
        imp_l = np.array(doc["importance_labels"])
        cov_l = np.array(doc["coverage_labels"])
        if len(imp_probs) == 0:
            continue
        true_omit = (imp_l == 1) & (cov_l == 0)
        nt = int(true_omit.sum())
        n_true[i] = nt
        # surfaced: [n_sent, n_tau, n_gamma]
        surfaced = (
            (imp_probs[:, None, None] >= tau_vals[None, :, None])
            & ((1 - cov_probs)[:, None, None] >= gamma_vals[None, None, :])
        )
        wl_mat[i] = surfaced.sum(axis=0)
        tp_mat[i] = (surfaced & true_omit[:, None, None]).sum(axis=0)
        if nt > 0:
            n_missed = (true_omit[:, None, None] & ~surfaced).sum(axis=0)
            loss_mat[i] = n_missed / nt
    return loss_mat, wl_mat, tp_mat, n_true


def precompute_cov_only_grid(docs, gamma_vals):
    """Loss grid for coverage-only surfacing (used by Union-Bound's γ calibration).
    surfaced = (1 - cov) >= gamma;  returns loss_mat [n_docs, n_gamma]."""
    n_g, n_d = len(gamma_vals), len(docs)
    loss_mat = np.zeros((n_d, n_g))
    for i, doc in enumerate(docs):
        cov_probs = np.array(doc["coverage_probs"])
        imp_l = np.array(doc["importance_labels"])
        cov_l = np.array(doc["coverage_labels"])
        if len(cov_probs) == 0:
            continue
        true_omit = (imp_l == 1) & (cov_l == 0)
        nt = int(true_omit.sum())
        if nt == 0:
            continue
        surfaced = (1 - cov_probs)[:, None] >= gamma_vals[None, :]
        n_missed = (true_omit[:, None] & ~surfaced).sum(axis=0)
        loss_mat[i] = n_missed / nt
    return loss_mat


# ---------------------------------------------------------------------------
# Calibration queries on precomputed grids (loop over α, not cal docs)
# ---------------------------------------------------------------------------
def calibrate_fact_from_grid(loss_mat_cal, alpha):
    """Return smallest-λ index satisfying adj_risk ≤ α."""
    n = loss_mat_cal.shape[0]
    adj = (n / (n + 1)) * loss_mat_cal.mean(axis=0) + 1 / (n + 1)
    ok = adj <= alpha
    if not ok.any():
        return loss_mat_cal.shape[1] - 1  # λ = 1.0 (flag everything)
    return int(np.argmax(ok))  # first True


def calibrate_omit_2d_from_grid(loss_mat_cal, wl_mat_cal, alpha):
    """Return (tau_idx, gamma_idx): min-workload feasible pair, tie-break max τ, max γ."""
    n = loss_mat_cal.shape[0]
    adj = (n / (n + 1)) * loss_mat_cal.mean(axis=0) + 1 / (n + 1)  # [n_tau, n_gamma]
    feasible = adj <= alpha
    if not feasible.any():
        return 0, 0
    mean_wl = wl_mat_cal.mean(axis=0)
    wl_masked = np.where(feasible, mean_wl, np.inf)
    min_wl = wl_masked.min()
    cands = np.argwhere(wl_masked == min_wl).tolist()
    cands.sort(key=lambda p: (-p[0], -p[1]))  # max tau, then max gamma
    return int(cands[0][0]), int(cands[0][1])


def calibrate_omit_1d_from_grid(loss_mat_cal_gamma0, alpha):
    """1D importance (γ=0 slice): largest τ satisfying risk ≤ α (min workload)."""
    n = loss_mat_cal_gamma0.shape[0]
    adj = (n / (n + 1)) * loss_mat_cal_gamma0.mean(axis=0) + 1 / (n + 1)
    feasible = adj <= alpha
    if not feasible.any():
        return 0
    return int(np.where(feasible)[0].max())


def calibrate_union_from_grid(loss_mat_cal, cov_loss_cal, alpha):
    """Union-Bound: τ at α/2 via importance-only; γ at α/2 via coverage-only."""
    ti = calibrate_omit_1d_from_grid(loss_mat_cal[:, :, 0], alpha / 2)
    n = cov_loss_cal.shape[0]
    adj = (n / (n + 1)) * cov_loss_cal.mean(axis=0) + 1 / (n + 1)
    feasible = adj <= alpha / 2
    gi = int(np.where(feasible)[0].max()) if feasible.any() else 0
    return ti, gi


# ---------------------------------------------------------------------------
# Per-split test-set evaluation at a given (lam_idx) / (ti, gi)
# ---------------------------------------------------------------------------
def _fact_stats(f_loss_t, f_fl_t, f_tp_t, f_halluc_t, lam_idx, avg_sum_sents):
    halluc_total = int(f_halluc_t.sum())
    tp_total = int(f_tp_t[:, lam_idx].sum())
    flagged_mean = float(f_fl_t[:, lam_idx].mean())
    return {
        "violation": float(f_loss_t[:, lam_idx].mean()),
        "workload": flagged_mean,
        "workload_pct": flagged_mean / avg_sum_sents * 100 if avg_sum_sents > 0 else 0.0,
        "recall": tp_total / halluc_total if halluc_total > 0 else 1.0,
    }


def _omit_stats(o_loss_t, o_wl_t, o_tp_t, o_ntrue_t, ti, gi, avg_src_sents):
    ntrue_total = int(o_ntrue_t.sum())
    tp_total = int(o_tp_t[:, ti, gi].sum())
    wl_mean = float(o_wl_t[:, ti, gi].mean())
    return {
        "violation": float(o_loss_t[:, ti, gi].mean()),
        "workload": wl_mean,
        "workload_pct": wl_mean / avg_src_sents * 100 if avg_src_sents > 0 else 0.0,
        "recall": tp_total / ntrue_total if ntrue_total > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_dataset(dataset_key, alphas):
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / "phase_2" / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"SKIP {dataset_key}: {path} not found")
        return None

    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)
    n_test = n - n_cal

    thresholds_fact = np.arange(0.0, 1.0 + GRID_FACT, GRID_FACT)
    tau_vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)
    gamma_vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)

    print(f"\n{'='*70}")
    print(f"  {meta['label']}  (N={n}, cal={n_cal}, test={n_test})")
    print(f"  {N_RESPLITS} resplits × {len(alphas)} α values  ({alphas})")
    print(f"{'='*70}")

    # per_resplit: flat list of records (resplit × α)
    per_resplit = []

    # per-α metric accumulators
    metric_keys = {
        "factuality":   ["violation", "workload", "workload_pct", "recall", "lambda"],
        "2d_joint":     ["violation", "workload", "workload_pct", "recall", "tau", "gamma"],
        "1d_imp":       ["violation", "workload", "workload_pct", "recall", "tau"],
        "union_bound":  ["violation", "workload", "workload_pct", "recall", "tau", "gamma"],
    }
    acc = {a: {mth: {k: [] for k in ks} for mth, ks in metric_keys.items()} for a in alphas}

    rng = np.random.default_rng(SEED)
    for r in tqdm(range(N_RESPLITS), desc=meta["label"]):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]

        avg_sum_sents = float(np.mean([len(d.get("factuality_probs", [])) for d in test_docs]))
        avg_src_sents = float(np.mean([len(d.get("importance_probs", [])) for d in test_docs]))

        # Precompute once per split
        fL_c, fFl_c, _, _ = precompute_factuality_grid(cal_docs, thresholds_fact)
        fL_t, fFl_t, fTp_t, fHall_t = precompute_factuality_grid(test_docs, thresholds_fact)
        oL_c, oWl_c, _, _ = precompute_omission_grid(cal_docs, tau_vals, gamma_vals)
        oL_t, oWl_t, oTp_t, oN_t = precompute_omission_grid(test_docs, tau_vals, gamma_vals)
        cov_c = precompute_cov_only_grid(cal_docs, gamma_vals)

        for alpha in alphas:
            lam_idx = calibrate_fact_from_grid(fL_c, alpha)
            lam = float(thresholds_fact[lam_idx])
            fact = _fact_stats(fL_t, fFl_t, fTp_t, fHall_t, lam_idx, avg_sum_sents)
            fact["lambda"] = lam

            ti2, gi2 = calibrate_omit_2d_from_grid(oL_c, oWl_c, alpha)
            m2d = _omit_stats(oL_t, oWl_t, oTp_t, oN_t, ti2, gi2, avg_src_sents)
            m2d["tau"] = float(tau_vals[ti2]); m2d["gamma"] = float(gamma_vals[gi2])

            ti1 = calibrate_omit_1d_from_grid(oL_c[:, :, 0], alpha)
            m1d = _omit_stats(oL_t, oWl_t, oTp_t, oN_t, ti1, 0, avg_src_sents)
            m1d["tau"] = float(tau_vals[ti1])

            tiu, giu = calibrate_union_from_grid(oL_c, cov_c, alpha)
            mub = _omit_stats(oL_t, oWl_t, oTp_t, oN_t, tiu, giu, avg_src_sents)
            mub["tau"] = float(tau_vals[tiu]); mub["gamma"] = float(gamma_vals[giu])

            row = {"resplit": r, "alpha": float(alpha),
                   "factuality": fact, "2d_joint": m2d,
                   "1d_imp": m1d, "union_bound": mub}
            per_resplit.append(row)

            for mth_name, metrics in [("factuality", fact), ("2d_joint", m2d),
                                       ("1d_imp", m1d), ("union_bound", mub)]:
                for k in metric_keys[mth_name]:
                    acc[alpha][mth_name][k].append(metrics[k])

    by_alpha = {}
    for alpha in alphas:
        by_alpha[f"{alpha:.2f}"] = {
            mth: {k: _summarize(acc[alpha][mth][k]) for k in metric_keys[mth]}
            for mth in metric_keys
        }

    # Print summary
    print(f"\n  {'α':>5s} | {'fact V% (95% CI)':>22s} | {'2D omit V% (95% CI)':>24s} | "
          f"{'1D V% (95% CI)':>22s} | {'UB V% (95% CI)':>22s}")
    print(f"  {'-'*120}")
    for alpha in alphas:
        agg = by_alpha[f"{alpha:.2f}"]
        def fmt(m):
            v = agg[m]["violation"]
            flag = "!" if v["mean"] > alpha else " "
            return f"{flag}{v['mean']:5.1%} [{v['ci_lower']:5.1%},{v['ci_upper']:5.1%}]"
        print(f"  {alpha:5.2f} | {fmt('factuality'):>22s} | {fmt('2d_joint'):>24s} | "
              f"{fmt('1d_imp'):>22s} | {fmt('union_bound'):>22s}")

    return {
        "dataset": meta["label"],
        "n_total": n,
        "n_cal": n_cal,
        "n_test": n_test,
        "n_resplits": N_RESPLITS,
        "alphas": [float(a) for a in alphas],
        "by_alpha": by_alpha,
        "per_resplit": per_resplit,
    }


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=str, default=None,
                   help=f"Comma-separated dataset keys. Default: all. Valid: {list(DATASETS.keys())}")
    p.add_argument("--alphas", type=str, default=None,
                   help=f"Comma-separated α values. Default: {ALPHAS_DEFAULT}")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path. Default: paper/outputs/alpha_sweep_cis.json")
    return p.parse_args()


def main():
    args = _parse_args()

    dataset_filter = None
    if args.datasets is not None:
        dataset_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}
        bad = dataset_filter - set(DATASETS.keys())
        if bad:
            raise SystemExit(f"ERROR: unknown dataset(s) {sorted(bad)}")

    if args.alphas is not None:
        alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    else:
        alphas = ALPHAS_DEFAULT

    out_path = Path(args.out) if args.out else (OUTPUT_DIR / "alpha_sweep_cis.json")

    all_results = OrderedDict()
    for dataset_key in DATASETS:
        if dataset_filter is not None and dataset_key not in dataset_filter:
            continue
        result = run_dataset(dataset_key, alphas)
        if result is not None:
            all_results[dataset_key] = result

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
