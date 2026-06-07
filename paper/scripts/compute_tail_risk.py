#!/usr/bin/env python3
"""Compute tail-risk diagnostics (P50/P90/P95/P99 of per-document loss).

For each dataset, runs the same 100 cal/test resplits as compute_resplit_cis.py,
captures the per-document loss vector on each test set under 2D Joint, pools
across resplits, and reports the empirical percentiles of per-document loss.

Saves to:
    paper/outputs/tail_risk_cis.json

Usage:
    python3 -m paper.scripts.compute_tail_risk
    python3 -m paper.scripts.compute_tail_risk --datasets ACI_Bench,MIMIC_IV_BHC
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from care.config import (
    CANONICAL_DATA_ROOT as DATA_ROOT,
    DATASETS,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)


# Must match compute_resplit_cis.py for paired interpretation.
SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_FACT = 0.15
ALPHA_OMIT = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05

PERCENTILES = [50, 75, 90, 95, 99]


# ---------------------------------------------------------------------------
# Loss helpers (lifted from compute_resplit_cis.py for self-containment)
# ---------------------------------------------------------------------------
def fact_loss_doc(doc, lam):
    labels = np.asarray(doc["factuality_labels"])
    probs = np.asarray(doc["factuality_probs"])
    if len(labels) == 0:
        return 0
    flagged = probs <= lam
    halluc = labels == 0
    return int((halluc & ~flagged).any())


def omit_frac_loss_doc(doc, tau, gamma):
    imp_labels = np.asarray(doc["importance_labels"])
    cov_labels = np.asarray(doc["coverage_labels"])
    imp_probs = np.asarray(doc["importance_probs"])
    cov_probs = np.asarray(doc["coverage_probs"])
    if len(imp_labels) == 0:
        return 0.0
    true_omit = (imp_labels == 1) & (cov_labels == 0)
    n_true = int(true_omit.sum())
    if n_true == 0:
        return 0.0
    surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
    n_missed = int((true_omit & ~surfaced).sum())
    return float(n_missed / n_true)


# ---------------------------------------------------------------------------
# Calibration (matches compute_resplit_cis.py exactly so percentiles align)
# ---------------------------------------------------------------------------
def calibrate_factuality(cal_docs, alpha=ALPHA_FACT, grid=GRID_FACT):
    n = len(cal_docs)
    for lam in np.arange(0.0, 1.0 + grid, grid):
        losses = [fact_loss_doc(d, lam) for d in cal_docs]
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha:
            return float(lam)
    return 1.0


def calibrate_omission_2d(cal_docs, alpha=ALPHA_OMIT, grid=GRID_OMIT):
    n = len(cal_docs)
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)
    feasible = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            losses = [omit_frac_loss_doc(d, tau, gamma) for d in cal_docs]
            adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
            if adj_risk <= alpha:
                # Workload (sentences surfaced per doc on cal).
                wl = np.mean([
                    int(((np.asarray(d["importance_probs"]) >= tau)
                        & ((1 - np.asarray(d["coverage_probs"])) >= gamma)).sum())
                    for d in cal_docs
                ])
                feasible.append((float(tau), float(gamma), wl))
    if not feasible:
        return 0.0, 0.0
    feasible.sort(key=lambda x: (x[2], -x[0], -x[1]))
    return feasible[0][0], feasible[0][1]


def calibrate_omission_fst(cal_docs, alpha=ALPHA_OMIT, grid=GRID_OMIT):
    """LTT-FST omission calibration (tau+gamma descending ordering).

    Walks a pre-specified, data-independent ordering of grid cells and stops
    at the first cell whose finite-sample CRC bound holds (Angelopoulos et
    al. 2021, Fixed-Sequence Testing). This is the CARE omission controller.
    """
    n = len(cal_docs)
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)
    cells = [(float(t), float(g)) for t in tau_vals for g in gamma_vals]
    cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    for tau, gamma in cells:
        losses = [omit_frac_loss_doc(d, tau, gamma) for d in cal_docs]
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha:
            return tau, gamma
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------
def run_dataset(dataset_key):
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / "phase_2" / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"SKIP {dataset_key}: {path} not found")
        return None

    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)
    n_test = n - n_cal

    print(f"\n{'='*72}")
    print(f"  {meta['label']} (N={n}, cal={n_cal}, test={n_test})")
    print(f"  {N_RESPLITS} resplits, α_fact=α_omit={ALPHA_FACT}")
    print(f"{'='*72}")

    rng = np.random.default_rng(SEED)

    # Pooled per-document losses across all resplits.
    pooled_fact_losses = []
    pooled_omit_frac_losses = []
    # Track per-resplit summary to allow median/CI of percentile estimates.
    per_resplit_records = []

    for i in tqdm(range(N_RESPLITS), desc=meta["label"]):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]

        lam = calibrate_factuality(cal_docs)
        tau, gamma = calibrate_omission_fst(cal_docs)

        fact_per_doc = [fact_loss_doc(d, lam) for d in test_docs]
        omit_per_doc = [omit_frac_loss_doc(d, tau, gamma) for d in test_docs]

        pooled_fact_losses.extend(fact_per_doc)
        pooled_omit_frac_losses.extend(omit_per_doc)

        per_resplit_records.append({
            "resplit_idx": i,
            "tau": tau, "gamma": gamma, "lambda": lam,
            "fact_mean": float(np.mean(fact_per_doc)),
            "omit_frac_mean": float(np.mean(omit_per_doc)),
            "omit_frac_p90": float(np.percentile(omit_per_doc, 90)),
            "omit_frac_p95": float(np.percentile(omit_per_doc, 95)),
            "omit_frac_p99": float(np.percentile(omit_per_doc, 99)),
        })

    # Pooled percentiles across all (resplit, test_doc) pairs.
    pooled_fact = np.asarray(pooled_fact_losses, dtype=float)
    pooled_omit = np.asarray(pooled_omit_frac_losses, dtype=float)

    pooled_summary = {
        "fact_binary": {
            "mean": float(pooled_fact.mean()),
            **{f"P{p}": float(np.percentile(pooled_fact, p)) for p in PERCENTILES},
            "max": float(pooled_fact.max()),
            "n_docs": int(len(pooled_fact)),
        },
        "omit_fractional": {
            "mean": float(pooled_omit.mean()),
            **{f"P{p}": float(np.percentile(pooled_omit, p)) for p in PERCENTILES},
            "max": float(pooled_omit.max()),
            "n_docs": int(len(pooled_omit)),
        },
    }

    # Per-resplit percentile distribution: 100 P95 estimates (one per resplit)
    # gives a CI on "what's the typical 95th-percentile per-doc loss?"
    resplit_perc_summary = {}
    for p in [90, 95, 99]:
        key = f"omit_frac_p{p}"
        vals = np.asarray([r[key] for r in per_resplit_records])
        resplit_perc_summary[f"P{p}_per_resplit"] = {
            "mean": float(vals.mean()),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
            "median": float(np.median(vals)),
        }

    # Print summary.
    pf = pooled_summary["fact_binary"]
    po = pooled_summary["omit_fractional"]
    print(f"\n  Pooled per-document loss across {N_RESPLITS} resplits "
          f"({len(pooled_omit)} test-doc observations):")
    print(f"    Hallucination (binary):   "
          f"mean={pf['mean']:.3f}  P90={pf['P90']:.2f}  P95={pf['P95']:.2f}  "
          f"P99={pf['P99']:.2f}  max={pf['max']:.2f}")
    print(f"    Omission (fractional):    "
          f"mean={po['mean']:.3f}  P50={po['P50']:.3f}  P90={po['P90']:.3f}  "
          f"P95={po['P95']:.3f}  P99={po['P99']:.3f}  max={po['max']:.3f}")
    print(f"  Per-resplit P95 of omission frac loss: "
          f"mean={resplit_perc_summary['P95_per_resplit']['mean']:.3f}  "
          f"CI=[{resplit_perc_summary['P95_per_resplit']['ci_lower']:.3f},"
          f"{resplit_perc_summary['P95_per_resplit']['ci_upper']:.3f}]")

    return {
        "dataset": meta["label"],
        "n_total": n,
        "n_cal": n_cal,
        "n_test": n_test,
        "n_resplits": N_RESPLITS,
        "alpha_fact": ALPHA_FACT,
        "alpha_omit": ALPHA_OMIT,
        "pooled_summary": pooled_summary,
        "resplit_percentile_summary": resplit_perc_summary,
        "per_resplit": per_resplit_records,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=str, default=None,
                   help=f"Comma-separated dataset keys. Default: all. Valid: {list(DATASETS.keys())}")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path. Default: paper/outputs/tail_risk_cis.json")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else (OUTPUT_DIR / "tail_risk_cis.json")

    dataset_filter = None
    if args.datasets is not None:
        dataset_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}
        bad = dataset_filter - set(DATASETS.keys())
        if bad:
            print(f"ERROR: unknown dataset(s) {sorted(bad)}. Valid: {list(DATASETS.keys())}")
            sys.exit(1)

    all_results = OrderedDict()
    for dataset_key in DATASETS:
        if dataset_filter is not None and dataset_key not in dataset_filter:
            continue
        rec = run_dataset(dataset_key)
        if rec is not None:
            all_results[dataset_key] = rec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final compact table.
    print(f"\n{'='*100}")
    print(f"  Tail-risk summary (α=0.15, pooled across {N_RESPLITS} resplits)")
    print(f"{'='*100}")
    print(f"  {'Dataset':12s} | {'mean':>6s} {'P50':>6s} {'P75':>6s} {'P90':>6s} "
          f"{'P95':>6s} {'P99':>6s} {'max':>6s} | "
          f"{'Halluc P95':>11s} {'Halluc P99':>11s}")
    print("  " + "-" * 96)
    for dk, r in all_results.items():
        po = r["pooled_summary"]["omit_fractional"]
        pf = r["pooled_summary"]["fact_binary"]
        print(f"  {DATASETS[dk]['label']:12s} | "
              f"{po['mean']:6.3f} {po['P50']:6.3f} {po['P75']:6.3f} "
              f"{po['P90']:6.3f} {po['P95']:6.3f} {po['P99']:6.3f} {po['max']:6.3f} | "
              f"{pf['P95']:11.3f} {pf['P99']:11.3f}")


if __name__ == "__main__":
    main()
