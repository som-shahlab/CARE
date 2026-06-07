#!/usr/bin/env python3
"""LLM-judge swap robustness: Gemini 2.5 as Phase 2 judge.

Runs the same 100-resplit CRC / dev-tuned / Max-F1 calibration pipeline on
Gemini-judge calibrated_scores.jsonl files (ACI-Bench, MIMIC-BHC). Complements
the non-LLM scorer-agnosticism experiment by showing CARE is also robust to
swapping the underlying LLM judge family.

Saves means/stds over 100 resplits to paper/outputs/gemini_judge_results.json.
"""
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, PAPER_OUTPUT_DIR as OUTPUT_DIR
from compute_external_scorer_results import (
    calibrate_factuality,
    calibrate_factuality_devset,
    calibrate_factuality_maxf1,
    calibrate_omission_2d,
    calibrate_omission_2d_devset,
    calibrate_omission_maxf1,
    evaluate_test,
    ALPHA_FACT,
    ALPHA_OMIT,
    CAL_FRAC,
    N_RESPLITS,
    SEED,
)

RESULTS_PATH = OUTPUT_DIR / "gemini_judge_results.json"

DATASETS = {
    "ACI_Bench": {
        "label": "ACI-Bench",
        "path": DATA_ROOT / "ACI_Bench/ablations/judge/phase_2_gemini_judge/calibrated_scores.jsonl",
    },
    "MIMIC_IV_BHC": {
        "label": "MIMIC-BHC",
        "path": DATA_ROOT / "MIMIC_IV_BHC/ablations/judge/phase_2_gemini_judge/calibrated_scores.jsonl",
    },
}


def run_resplits(docs):
    rng = random.Random(SEED)
    n = len(docs)
    n_cal = int(n * CAL_FRAC)
    per_resplit = []
    for _ in tqdm(range(N_RESPLITS), desc="Resplits"):
        idx = list(range(n))
        rng.shuffle(idx)
        cal = [docs[i] for i in idx[:n_cal]]
        test = [docs[i] for i in idx[n_cal:]]
        lam = calibrate_factuality(cal, ALPHA_FACT)
        tau, gamma = calibrate_omission_2d(cal, ALPHA_OMIT)
        lam_dev = calibrate_factuality_devset(cal, ALPHA_FACT)
        tau_dev, gamma_dev = calibrate_omission_2d_devset(cal, ALPHA_OMIT)
        lam_f1 = calibrate_factuality_maxf1(cal)
        tau_f1, gamma_f1 = calibrate_omission_maxf1(cal)
        per_resplit.append(evaluate_test(
            test, lam, tau, gamma, lam_dev, tau_dev, gamma_dev,
            lam_f1, tau_f1, gamma_f1,
        ))
    summary = {}
    for tag in ["crc", "dev", "maxf1"]:
        for metric in ["fact_viol", "fact_wl", "fact_recall",
                       "omit_viol", "omit_wl", "omit_recall"]:
            vals = [r[tag][metric] for r in per_resplit]
            summary[f"{tag}_{metric}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    return summary


def main():
    results = {}
    if RESULTS_PATH.exists():
        results = json.load(open(RESULTS_PATH))
        print(f"Loaded {len(results)} existing results")

    for ds_key, meta in DATASETS.items():
        if ds_key in results:
            print(f"\n{meta['label']}: cached, skipping")
            continue
        print(f"\n{'='*60}\n{meta['label']} (Gemini judge)\n{'='*60}")
        docs = [json.loads(l) for l in open(meta["path"])]
        print(f"Loaded {len(docs)} docs")
        summary = run_resplits(docs)
        results[ds_key] = {
            "dataset": ds_key,
            "label": meta["label"],
            "judge": "gemini-2.5",
            "n_docs": len(docs),
            "summary": summary,
        }
        for tag in ["crc", "dev", "maxf1"]:
            fv = summary[f"{tag}_fact_viol"]["mean"] * 100
            ov = summary[f"{tag}_omit_viol"]["mean"] * 100
            print(f"  {tag.upper():5s}: FactV={fv:5.1f}%  OmitV={ov:5.1f}%")
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
