#!/usr/bin/env python3
"""FST α-sweep with 95% CIs via 100 random resplits.

Mirrors compute_alpha_sweep_cis.py but selects (τ, γ) via LTT-FST
(τ+γ descending ordering, first-feasible stop) instead of 2D Joint
argmin-workload. Saves to fst_alpha_sweep_cis.json.

Uses identical seeds as compute_alpha_sweep_cis.py so per-resplit values
are paired against the 2D Joint heuristic.
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


def precompute_omission_grid(docs, tau_vals, gamma_vals):
    """[n_d, n_tau, n_gamma] grids: loss, workload, tp; plus n_true per doc."""
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


def fst_order_indices(tau_vals, gamma_vals):
    """Return list of (ti, gi) walking cells in τ+γ descending order
    (ties: τ desc, γ desc). Data-independent."""
    cells = [(ti, gi, float(tau_vals[ti]), float(gamma_vals[gi]))
             for ti in range(len(tau_vals)) for gi in range(len(gamma_vals))]
    cells.sort(key=lambda c: (-(c[2] + c[3]), -c[2], -c[3]))
    return [(c[0], c[1]) for c in cells]


def calibrate_omit_fst_from_grid(loss_mat_cal, alpha, order_idx):
    """Walk cells in pre-specified order; stop at first whose adj risk ≤ α."""
    n = loss_mat_cal.shape[0]
    mean_loss = loss_mat_cal.mean(axis=0)  # [n_tau, n_gamma]
    for ti, gi in order_idx:
        adj = (n / (n + 1)) * mean_loss[ti, gi] + 1.0 / (n + 1)
        if adj <= alpha:
            return int(ti), int(gi), True
    return 0, 0, False  # fall back to (0,0)


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

    tau_vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)
    gamma_vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)
    order_idx = fst_order_indices(tau_vals, gamma_vals)

    print(f"\n{'='*70}")
    print(f"  {meta['label']}  (N={n}, cal={n_cal}, test={n_test})")
    print(f"  {N_RESPLITS} resplits × {len(alphas)} α values  ({alphas})")
    print(f"  FST ordering: tau_plus_gamma  ({len(order_idx)} cells)")
    print(f"{'='*70}")

    metric_keys = ["violation", "workload", "workload_pct", "recall", "tau", "gamma",
                   "is_feasible", "steps_walked"]
    acc = {a: {k: [] for k in metric_keys} for a in alphas}
    per_resplit = []

    rng = np.random.default_rng(SEED)
    for r in tqdm(range(N_RESPLITS), desc=meta["label"]):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]

        avg_src_sents = float(np.mean([len(d.get("importance_probs", [])) for d in test_docs]))

        oL_c, _, _, _ = precompute_omission_grid(cal_docs, tau_vals, gamma_vals)
        oL_t, oWl_t, oTp_t, oN_t = precompute_omission_grid(test_docs, tau_vals, gamma_vals)

        for alpha in alphas:
            # Walk the ordering with mean_loss already cached
            n_c = oL_c.shape[0]
            mean_loss_c = oL_c.mean(axis=0)
            ti, gi, is_feas = 0, 0, False
            steps = 0
            for (ti_, gi_) in order_idx:
                steps += 1
                adj = (n_c / (n_c + 1)) * mean_loss_c[ti_, gi_] + 1.0 / (n_c + 1)
                if adj <= alpha:
                    ti, gi, is_feas = int(ti_), int(gi_), True
                    break

            m = _omit_stats(oL_t, oWl_t, oTp_t, oN_t, ti, gi, avg_src_sents)
            m["tau"] = float(tau_vals[ti])
            m["gamma"] = float(gamma_vals[gi])
            m["is_feasible"] = bool(is_feas)
            m["steps_walked"] = int(steps)

            for k in metric_keys:
                acc[alpha][k].append(m[k] if not isinstance(m[k], bool) else int(m[k]))
            per_resplit.append({"resplit": r, "alpha": float(alpha), **m})

    by_alpha = {}
    for alpha in alphas:
        by_alpha[f"{alpha:.2f}"] = {
            k: _summarize(acc[alpha][k]) for k in metric_keys if k != "is_feasible"
        }
        by_alpha[f"{alpha:.2f}"]["n_feasible"] = int(sum(acc[alpha]["is_feasible"]))

    # Print summary
    print(f"\n  {'α':>5s} | {'FST V% (95% CI)':>22s} | {'WL':>8s} {'WL%':>6s} {'Rec%':>6s} {'Feas':>5s}")
    print(f"  {'-'*80}")
    for alpha in alphas:
        agg = by_alpha[f"{alpha:.2f}"]
        v = agg["violation"]
        flag = "!" if v["mean"] > alpha else " "
        ci = f"{flag}{v['mean']:5.1%} [{v['ci_lower']:5.1%},{v['ci_upper']:5.1%}]"
        print(f"  {alpha:5.2f} | {ci:>22s} | "
              f"{agg['workload']['mean']:>8.1f} {agg['workload_pct']['mean']:>5.1f}% "
              f"{agg['recall']['mean']:>5.1%} {agg['n_feasible']:>5d}")

    return {
        "dataset": meta["label"],
        "n_total": n,
        "n_cal": n_cal,
        "n_test": n_test,
        "n_resplits": N_RESPLITS,
        "alphas": [float(a) for a in alphas],
        "ordering": "tau_plus_gamma",
        "by_alpha": by_alpha,
        "per_resplit": per_resplit,
    }


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--alphas", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    dataset_filter = None
    if args.datasets is not None:
        dataset_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}
    alphas = ([float(a) for a in args.alphas.split(",")]
              if args.alphas else ALPHAS_DEFAULT)
    out_path = Path(args.out) if args.out else (OUTPUT_DIR / "fst_alpha_sweep_cis.json")

    all_results = OrderedDict()
    for dataset_key in DATASETS:
        if dataset_filter is not None and dataset_key not in dataset_filter:
            continue
        result = run_dataset(dataset_key, alphas)
        if result is not None:
            all_results[dataset_key] = result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final validity summary
    print(f"\n{'='*80}")
    print("  FST α-validity summary (mean V% across 100 resplits)")
    print(f"{'='*80}")
    header = f"  {'Dataset':<14}" + "".join(f"{a:>8.2f}" for a in alphas)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dk, r in all_results.items():
        line = f"  {DATASETS[dk]['label']:<14}"
        for a in alphas:
            v = r["by_alpha"][f"{a:.2f}"]["violation"]["mean"]
            mark = "!" if v > a else " "
            line += f"{mark}{v*100:6.2f} "
        print(line)
    print("  ! = exceeds α")


if __name__ == "__main__":
    main()
