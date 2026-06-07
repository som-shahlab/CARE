#!/usr/bin/env python3
"""Compute 95% CIs for prompt ablation table via 100 random resplits.

Runs both generic (phase_2) and domain-specific (phase_2_modified_fact_imp)
prompts across all 5 datasets.
"""
import json
import sys
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR

# Config
SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_FACT = 0.15
ALPHA_OMIT = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05

VARIANTS = OrderedDict([
    ("generic", "phase_2"),
    ("domain", "ablations/prompt/phase_2_modified_fact_imp"),
])


def compute_factuality_loss(doc, threshold):
    labels = np.array(doc["factuality_labels"])
    probs = np.array(doc["factuality_probs"])
    if len(labels) == 0:
        return 0, 0, 0, 0
    flagged = probs <= threshold
    halluc = labels == 0
    loss = int((halluc & ~flagged).any())
    n_flagged = int(flagged.sum())
    tp = int((flagged & halluc).sum())
    total_halluc = int(halluc.sum())
    return loss, n_flagged, tp, total_halluc


def compute_omission_loss_fractional(doc, tau, gamma):
    imp_labels = np.array(doc["importance_labels"])
    cov_labels = np.array(doc["coverage_labels"])
    imp_probs = np.array(doc["importance_probs"])
    cov_probs = np.array(doc["coverage_probs"])
    if len(imp_labels) == 0:
        return 0.0, 0, 0, 0

    true_omit = (imp_labels == 1) & (cov_labels == 0)
    n_true = int(true_omit.sum())
    surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
    n_surfaced = int(surfaced.sum())

    if n_true == 0:
        return 0.0, n_surfaced, 0, 0

    n_missed = int((true_omit & ~surfaced).sum())
    tp = int((surfaced & true_omit).sum())
    loss = n_missed / n_true
    return float(loss), n_surfaced, tp, n_true


def calibrate_factuality(cal_docs, alpha):
    n = len(cal_docs)
    thresholds = np.arange(0.0, 1.0 + GRID_FACT, GRID_FACT)
    for lam in thresholds:
        losses = [compute_factuality_loss(d, lam)[0] for d in cal_docs]
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha:
            return float(lam)
    return 1.0


def calibrate_omission_2d(cal_docs, alpha):
    n = len(cal_docs)
    tau_vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)
    gamma_vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)

    feasible = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            results = [compute_omission_loss_fractional(d, tau, gamma)
                       for d in cal_docs]
            losses = [r[0] for r in results]
            adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
            if adj_risk <= alpha:
                wl = np.mean([r[1] for r in results])
                feasible.append((tau, gamma, wl))

    if not feasible:
        return 0.0, 0.0
    feasible.sort(key=lambda x: (x[2], -x[0], -x[1]))
    return float(feasible[0][0]), float(feasible[0][1])


def evaluate_test(test_docs, lam, tau, gamma):
    fact_losses, fact_flagged = [], []
    omit_losses, omit_surfaced = [], []
    fact_tp_total, fact_halluc_total = 0, 0
    omit_tp_total, omit_true_total = 0, 0

    for doc in test_docs:
        fl, nf, ftp, fh = compute_factuality_loss(doc, lam)
        fact_losses.append(fl)
        fact_flagged.append(nf)
        fact_tp_total += ftp
        fact_halluc_total += fh

        ol, ns, otp, ot = compute_omission_loss_fractional(doc, tau, gamma)
        omit_losses.append(ol)
        omit_surfaced.append(ns)
        omit_tp_total += otp
        omit_true_total += ot

    avg_sum_sents = np.mean([len(d.get("factuality_probs", [])) for d in test_docs])
    avg_src_sents = np.mean([len(d.get("importance_probs", [])) for d in test_docs])

    return {
        "fact_viol": float(np.mean(fact_losses)),
        "fact_wl_pct": float(np.mean(fact_flagged) / avg_sum_sents * 100) if avg_sum_sents > 0 else 0,
        "omit_viol": float(np.mean(omit_losses)),
        "omit_wl_pct": float(np.mean(omit_surfaced) / avg_src_sents * 100) if avg_src_sents > 0 else 0,
    }


def run_resplits(dataset_key, phase2_subdir):
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / phase2_subdir / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"  SKIP {dataset_key}/{phase2_subdir}: not found")
        return None

    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)

    print(f"  {meta['label']} / {phase2_subdir} (N={n}, cal={n_cal}, test={n - n_cal})")

    rng = np.random.default_rng(SEED)
    results = []

    for i in tqdm(range(N_RESPLITS), desc=f"  {meta['label']}", leave=False):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]

        lam = calibrate_factuality(cal_docs, ALPHA_FACT)
        tau, gamma = calibrate_omission_2d(cal_docs, ALPHA_OMIT)
        res = evaluate_test(test_docs, lam, tau, gamma)
        results.append(res)

    summary = {}
    for m in ["fact_viol", "fact_wl_pct", "omit_viol", "omit_wl_pct"]:
        vals = [r[m] for r in results]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }

    s = summary
    print(f"    Fact V: {s['fact_viol']['mean']:.1%} [{s['fact_viol']['ci_lower']:.1%}, {s['fact_viol']['ci_upper']:.1%}]"
          f"  WL: {s['fact_wl_pct']['mean']:.1f}% [{s['fact_wl_pct']['ci_lower']:.1f}, {s['fact_wl_pct']['ci_upper']:.1f}]")
    print(f"    Omit V: {s['omit_viol']['mean']:.1%} [{s['omit_viol']['ci_lower']:.1%}, {s['omit_viol']['ci_upper']:.1%}]"
          f"  WL: {s['omit_wl_pct']['mean']:.1f}% [{s['omit_wl_pct']['ci_lower']:.1f}, {s['omit_wl_pct']['ci_upper']:.1f}]")

    return {
        "dataset": meta["label"],
        "n_total": n,
        "summary": summary,
        "per_resplit": results,
    }


if __name__ == "__main__":
    all_results = OrderedDict()

    for variant_name, phase2_subdir in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"  Variant: {variant_name} ({phase2_subdir})")
        print(f"{'='*70}")
        for dataset_key in DATASETS:
            result = run_resplits(dataset_key, phase2_subdir)
            if result:
                all_results[f"{dataset_key}__{variant_name}"] = result

    # Save
    out_path = OUTPUT_DIR / "prompt_ablation_resplit_cis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # LaTeX table
    print(f"\n{'='*70}")
    print("  LaTeX table rows (mean [CI])")
    print(f"{'='*70}")
    for dataset_key in DATASETS:
        label = DATASETS[dataset_key]["label"]
        for vi, (vname, _) in enumerate(VARIANTS.items()):
            key = f"{dataset_key}__{vname}"
            if key not in all_results:
                continue
            s = all_results[key]["summary"]
            prompt_label = "Generic" if vname == "generic" else "Domain"
            prefix = label if vi == 0 else ""

            fv = f"{s['fact_viol']['mean']*100:.1f}"
            fw = f"{s['fact_wl_pct']['mean']:.1f}"
            ov = f"{s['omit_viol']['mean']*100:.1f}"
            ow = f"{s['omit_wl_pct']['mean']:.1f}"

            fv_ci = f"[{s['fact_viol']['ci_lower']*100:.1f}, {s['fact_viol']['ci_upper']*100:.1f}]"
            fw_ci = f"[{s['fact_wl_pct']['ci_lower']:.1f}, {s['fact_wl_pct']['ci_upper']:.1f}]"
            ov_ci = f"[{s['omit_viol']['ci_lower']*100:.1f}, {s['omit_viol']['ci_upper']*100:.1f}]"
            ow_ci = f"[{s['omit_wl_pct']['ci_lower']:.1f}, {s['omit_wl_pct']['ci_upper']:.1f}]"

            print(f"  {prefix} & {prompt_label} & {fv} {fv_ci} & {fw} {fw_ci} & {ov} {ov_ci} & {ow} {ow_ci} \\\\")
        print(f"  \\midrule")
