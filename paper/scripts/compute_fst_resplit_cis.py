#!/usr/bin/env python3
"""Compute 100-resplit CIs for the LTT-FST omission variant.

Mirrors compute_resplit_cis.py for the existing 2D Joint heuristic, but
adds three FST orderings (tau_plus_gamma, lex_tau, diagonal). Saves to a
NEW file (fst_resplit_cis.json) so the existing resplit_cis.json is
untouched.

Uses identical seeds and split logic as compute_resplit_cis.py, so per-
resplit values can be paired against the heuristic for direct comparison.

Usage:
    python3 -m paper.scripts.compute_fst_resplit_cis
    python3 -m paper.scripts.compute_fst_resplit_cis --datasets ACI_Bench,MIMIC_IV_BHC
    python3 -m paper.scripts.compute_fst_resplit_cis --orderings tau_plus_gamma
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


# ---------------------------------------------------------------------------
# Config (must match compute_resplit_cis.py for paired comparison)
# ---------------------------------------------------------------------------
SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_OMIT = 0.15
GRID_OMIT = 0.05

ORDERINGS = ["tau_plus_gamma", "lex_tau", "diagonal"]


# ---------------------------------------------------------------------------
# Helpers (subset of compute_resplit_cis.py kept here to avoid coupling)
# ---------------------------------------------------------------------------
def compute_omission_2d(doc, tau, gamma):
    imp_labels = np.asarray(doc["importance_labels"])
    cov_labels = np.asarray(doc["coverage_labels"])
    imp_probs = np.asarray(doc["importance_probs"])
    cov_probs = np.asarray(doc["coverage_probs"])
    if len(imp_labels) == 0:
        return 0.0, 0, 0, 0, 0
    surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
    true_omit = (imp_labels == 1) & (cov_labels == 0)
    n_true = int(true_omit.sum())
    n_surfaced = int(surfaced.sum())
    if n_true == 0:
        return 0.0, n_surfaced, 0, 0, 0
    n_missed = int((true_omit & ~surfaced).sum())
    tp = int((surfaced & true_omit).sum())
    return float(n_missed / n_true), n_surfaced, tp, n_true, (1 if n_missed > 0 else 0)


def grid_cells(grid_resolution):
    vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)
    return [(float(t), float(g)) for t in vals for g in vals]


def order_cells(cells, ordering):
    cs = list(cells)
    if ordering == "tau_plus_gamma":
        cs.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    elif ordering == "lex_tau":
        cs.sort(key=lambda c: (-c[0], -c[1]))
    elif ordering == "diagonal":
        # Restrict to τ = γ chain only.
        diag = sorted({c[0] for c in cs}, reverse=True)
        cs = [(v, v) for v in diag]
    else:
        raise ValueError(f"unknown ordering: {ordering}")
    return cs


# ---------------------------------------------------------------------------
# FST calibration
# ---------------------------------------------------------------------------
def calibrate_omission_fst(cal_docs, alpha, ordering, grid_resolution=GRID_OMIT):
    """Walk the pre-specified ordering; stop at first cell whose CRC bound
    (n/(n+1)) R̂ + 1/(n+1) <= alpha is satisfied.

    Returns (tau, gamma, steps_walked, total_cells, is_feasible).
    """
    n = len(cal_docs)
    cells = order_cells(grid_cells(grid_resolution), ordering)
    for step, (tau, gamma) in enumerate(cells, start=1):
        losses = [compute_omission_2d(d, tau, gamma)[0] for d in cal_docs]
        adj_risk = (n / (n + 1)) * float(np.mean(losses)) + 1.0 / (n + 1)
        if adj_risk <= alpha:
            return float(tau), float(gamma), step, len(cells), True
    # No feasible cell along the sequence -> fall back to (0, 0).
    return 0.0, 0.0, len(cells), len(cells), False


def evaluate_method_on_test(test_docs, tau, gamma, avg_src_sents):
    frac_losses, bin_losses, surfaced, tp_total, true_total = [], [], [], 0, 0
    for d in test_docs:
        fl, ns, tp, nt, bl = compute_omission_2d(d, tau, gamma)
        frac_losses.append(fl)
        bin_losses.append(bl)
        surfaced.append(ns)
        tp_total += tp
        true_total += nt
    rec = tp_total / true_total if true_total > 0 else 1.0
    return {
        "tau": float(tau),
        "gamma": float(gamma),
        "omit_viol": float(np.mean(frac_losses)),
        "omit_binary_viol": float(np.mean(bin_losses)),
        "omit_wl": float(np.mean(surfaced)),
        "omit_wl_pct": float(np.mean(surfaced) / avg_src_sents * 100) if avg_src_sents > 0 else 0,
        "omit_recall": float(rec),
    }


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
        "median": float(np.median(vals)),
    }


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------
def run_resplits_fst(dataset_key, orderings):
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / "phase_2" / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"SKIP {dataset_key}: {path} not found")
        return None

    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)

    print(f"\n{'='*72}")
    print(f"  {meta['label']} (N={n}, cal={n_cal}, test={n - n_cal})")
    print(f"  {N_RESPLITS} resplits, α={ALPHA_OMIT}, orderings={orderings}")
    print(f"{'='*72}")

    rng = np.random.default_rng(SEED)
    per_resplit = []

    for i in tqdm(range(N_RESPLITS), desc=meta["label"]):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]
        avg_src = float(np.mean([len(d.get("importance_probs", [])) for d in test_docs]))

        rec = {"resplit_idx": i, "n_cal": n_cal, "n_test": n - n_cal,
               "avg_src_sents_test": avg_src, "fst": {}}

        for ord_name in orderings:
            tau, gamma, steps, total_cells, is_feasible = calibrate_omission_fst(
                cal_docs, ALPHA_OMIT, ord_name, GRID_OMIT
            )
            test_metrics = evaluate_method_on_test(test_docs, tau, gamma, avg_src)
            test_metrics.update({
                "fst_steps_walked": int(steps),
                "fst_total_cells": int(total_cells),
                "fst_is_feasible": bool(is_feasible),
                "ordering": ord_name,
            })
            rec["fst"][ord_name] = test_metrics

        per_resplit.append(rec)

    # Aggregate.
    fst_summary = {}
    for ord_name in orderings:
        ord_summary = {}
        for metric in ["omit_viol", "omit_binary_viol", "omit_wl", "omit_wl_pct",
                       "omit_recall", "tau", "gamma", "fst_steps_walked"]:
            vals = [r["fst"][ord_name][metric] for r in per_resplit]
            ord_summary[metric] = summarize(vals)
        ord_summary["n_feasible"] = int(sum(r["fst"][ord_name]["fst_is_feasible"] for r in per_resplit))
        fst_summary[ord_name] = ord_summary

    # Print.
    print(f"\n  FST summary (mean omit_viol / WL / WL% / Rec):")
    for ord_name in orderings:
        s = fst_summary[ord_name]
        feas = s["n_feasible"]
        print(f"    {ord_name:18s}: "
              f"V={s['omit_viol']['mean']:.1%} [{s['omit_viol']['ci_lower']:.1%},{s['omit_viol']['ci_upper']:.1%}]  "
              f"WL={s['omit_wl']['mean']:.1f} ({s['omit_wl_pct']['mean']:.1f}%)  "
              f"Rec={s['omit_recall']['mean']:.1%}  feasible={feas}/{N_RESPLITS}")

    return {
        "dataset": meta["label"],
        "n_total": n,
        "n_cal": n_cal,
        "n_test": n - n_cal,
        "n_resplits": N_RESPLITS,
        "alpha_omit": ALPHA_OMIT,
        "grid_resolution": GRID_OMIT,
        "orderings": orderings,
        "fst_summary": fst_summary,
        "per_resplit": per_resplit,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=str, default=None,
                   help=f"Comma-separated dataset keys. Default: all. Valid: {list(DATASETS.keys())}")
    p.add_argument("--orderings", type=str, default=",".join(ORDERINGS),
                   help=f"Comma-separated orderings. Valid: {ORDERINGS}")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path. Default: paper/outputs/fst_resplit_cis.json")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else (OUTPUT_DIR / "fst_resplit_cis.json")

    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]
    bad_o = set(orderings) - set(ORDERINGS)
    if bad_o:
        print(f"ERROR: unknown orderings {sorted(bad_o)}. Valid: {ORDERINGS}")
        sys.exit(1)

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
        result = run_resplits_fst(dataset_key, orderings)
        if result:
            all_results[dataset_key] = result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final summary table.
    print(f"\n{'='*100}")
    print(f"  FST 100-resplit summary (α={ALPHA_OMIT})")
    print(f"{'='*100}")
    header = f"  {'Dataset':12s} | "
    for o in orderings:
        header += f"{o + ' V%':>15s} {o + ' WL':>10s} | "
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dk, r in all_results.items():
        s = r["fst_summary"]
        line = f"  {DATASETS[dk]['label']:12s} | "
        for o in orderings:
            v = s[o]["omit_viol"]
            wl = s[o]["omit_wl"]
            line += (f"{v['mean']*100:5.1f} [{v['ci_lower']*100:4.1f},{v['ci_upper']*100:4.1f}]"
                     f" {wl['mean']:>8.1f}".rjust(15) + " | ")
        print(line)


if __name__ == "__main__":
    main()
