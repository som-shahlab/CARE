#!/usr/bin/env python3
"""FST prompt-ablation validity check (100 resplits).

Mirrors the omission portion of compute_prompt_ablation_cis.py but selects
(τ, γ) via LTT-FST (τ+γ descending) instead of 2D Joint argmin-workload.

Runs both generic (phase_2) and domain-specific (phase_2_modified_fact_imp)
prompts across all 5 datasets and confirms FST holds α-validity.

Output: paper/outputs/fst_prompt_ablation.json
"""
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
ALPHA = 0.15
GRID = 0.05

VARIANTS = OrderedDict([
    ("generic", "phase_2"),
    ("domain", "ablations/prompt/phase_2_modified_fact_imp"),
])


def compute_omission(doc, tau, gamma):
    imp = np.asarray(doc["importance_probs"])
    cov = np.asarray(doc["coverage_probs"])
    imp_l = np.asarray(doc["importance_labels"])
    cov_l = np.asarray(doc["coverage_labels"])
    if len(imp) == 0:
        return 0.0, 0
    surfaced = (imp >= tau) & ((1 - cov) >= gamma)
    n_surf = int(surfaced.sum())
    true_omit = (imp_l == 1) & (cov_l == 0)
    n_true = int(true_omit.sum())
    if n_true == 0:
        return 0.0, n_surf
    return float((true_omit & ~surfaced).sum() / n_true), n_surf


def fst_calibrate(cal_docs, alpha=ALPHA, grid=GRID):
    n = len(cal_docs)
    vals = np.arange(0.0, 1.0 + grid, grid)
    cells = [(float(t), float(g)) for t in vals for g in vals]
    cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    for tau, gamma in cells:
        losses = [compute_omission(d, tau, gamma)[0] for d in cal_docs]
        adj = (n / (n + 1)) * np.mean(losses) + 1.0 / (n + 1)
        if adj <= alpha:
            return float(tau), float(gamma), True
    return 0.0, 0.0, False


def evaluate(test_docs, tau, gamma):
    losses, surfaced = [], []
    for d in test_docs:
        l, s = compute_omission(d, tau, gamma)
        losses.append(l); surfaced.append(s)
    avg_src = np.mean([len(d.get("importance_probs", [])) for d in test_docs])
    return {
        "omit_viol": float(np.mean(losses)),
        "omit_wl": float(np.mean(surfaced)),
        "omit_wl_pct": float(np.mean(surfaced) / avg_src * 100) if avg_src > 0 else 0.0,
    }


def run_resplits(dataset_key, phase2_subdir):
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / phase2_subdir / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"  SKIP {dataset_key}/{phase2_subdir}: not found")
        return None
    docs = [json.loads(l) for l in open(path)]
    n = len(docs); n_cal = int(n * CAL_FRAC)
    print(f"  {meta['label']} / {phase2_subdir} (N={n}, cal={n_cal})")
    rng = np.random.default_rng(SEED)
    results = []
    for _ in tqdm(range(N_RESPLITS), desc=f"  {meta['label']}", leave=False):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]
        tau, gamma, _ = fst_calibrate(cal_docs)
        res = evaluate(test_docs, tau, gamma)
        res["tau"] = tau; res["gamma"] = gamma
        results.append(res)

    summary = {}
    for m in ["omit_viol", "omit_wl", "omit_wl_pct"]:
        vals = [r[m] for r in results]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }
    s = summary
    mark = "✓" if s["omit_viol"]["mean"] <= ALPHA else "✗"
    print(f"    Omit V: {s['omit_viol']['mean']:.1%} [{s['omit_viol']['ci_lower']:.1%}, {s['omit_viol']['ci_upper']:.1%}] {mark}"
          f"  WL: {s['omit_wl_pct']['mean']:.1f}%")
    return {"dataset": meta["label"], "n_total": n, "summary": summary, "per_resplit": results}


def main():
    all_results = OrderedDict()
    for variant_name, phase2_subdir in VARIANTS.items():
        print(f"\n{'='*70}\n  Variant: {variant_name} ({phase2_subdir})\n{'='*70}")
        for dataset_key in DATASETS:
            r = run_resplits(dataset_key, phase2_subdir)
            if r:
                all_results[f"{dataset_key}__{variant_name}"] = r

    out_path = OUTPUT_DIR / "fst_prompt_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Validity summary
    print(f"\n{'='*72}\n  FST prompt-ablation validity summary (α={ALPHA})\n{'='*72}")
    header = f"  {'Dataset':<14} {'Generic V%':>13} {'Domain V%':>13}"
    print(header); print("  " + "-" * (len(header) - 2))
    for dataset_key in DATASETS:
        line = f"  {DATASETS[dataset_key]['label']:<14}"
        for vname in VARIANTS:
            key = f"{dataset_key}__{vname}"
            if key in all_results:
                v = all_results[key]["summary"]["omit_viol"]["mean"]
                mark = "!" if v > ALPHA else " "
                line += f"{mark}{v*100:11.2f}%"
            else:
                line += f"{'--':>13}"
        print(line)
    print("  ! = exceeds α")


if __name__ == "__main__":
    main()
