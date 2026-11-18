#!/usr/bin/env python3
"""Alternative Phase-2 judges (Gemini 2.5 Pro, Qwen3.5-4B) under the camera-ready calibration.

Replaces the GPT-5-mini judge scores with an alternative judge's scores and reruns the same
100 random 70/30 resplits (seed 42) used everywhere else:

  - hallucination: scalar CRC on the binary document loss (lambda grid 0.01)
  - omission: LTT-FST (care.calibration.select_omission_threshold_ltt_fst: HB p-values,
    tau+gamma ascending order, min-workload certified cell; alpha=0.15, delta=0.05)

With --verify-2d it also recomputes the submission-time 2D-Joint omission numbers, which
should reproduce outputs/gemini_judge_results.json (sanity check of data paths and splits).

Output: paper/outputs/fst_alt_judge_results.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from care.calibration import omission_grid_matrices, select_omission_threshold_ltt_fst
from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, PAPER_OUTPUT_DIR as OUTPUT_DIR

SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05

JUDGES = {
    "gemini-2.5-pro": {
        "ACI_Bench": "ACI_Bench/ablations/judge/phase_2_gemini_judge/calibrated_scores.jsonl",
        "MIMIC_IV_BHC": "MIMIC_IV_BHC/ablations/judge/phase_2_gemini_judge/calibrated_scores.jsonl",
    },
    "qwen3.5-4b": {
        "ACI_Bench": "ACI_Bench/{qwen_dir}/calibrated_scores.jsonl",
        "MIMIC_IV_BHC": "MIMIC_IV_BHC/{qwen_dir}/calibrated_scores.jsonl",
    },
}
LABELS = {"ACI_Bench": "ACI-Bench", "MIMIC_IV_BHC": "MIMIC-BHC"}
RESULTS_PATH = OUTPUT_DIR / "fst_alt_judge_results.json"


def fact_stats(doc, lam):
    labels = np.asarray(doc["factuality_labels"])
    probs = np.asarray(doc["factuality_probs"])
    if len(labels) == 0:
        return 0, 0, 0, 0
    flagged = probs <= lam
    halluc = labels == 0
    return int((halluc & ~flagged).any()), int(flagged.sum()), int((flagged & halluc).sum()), int(halluc.sum())


def calibrate_factuality(cal):
    n = len(cal)
    for lam in np.arange(0.0, 1.0 + GRID_FACT, GRID_FACT):
        risk = np.mean([fact_stats(d, lam)[0] for d in cal])
        if (n / (n + 1)) * risk + 1 / (n + 1) <= ALPHA:
            return float(lam)
    return 1.0


def omit_stats(doc, tau, gamma):
    imp = np.asarray(doc["importance_probs"])
    cov = np.asarray(doc["coverage_probs"])
    if len(imp) == 0:
        return 0.0, 0, 0, 0
    surfaced = (imp >= tau) & ((1 - cov) >= gamma)
    true = (np.asarray(doc["importance_labels"]) == 1) & (np.asarray(doc["coverage_labels"]) == 0)
    n_true = int(true.sum())
    if n_true == 0:
        return 0.0, int(surfaced.sum()), 0, 0
    return float((true & ~surfaced).sum() / n_true), int(surfaced.sum()), int((surfaced & true).sum()), n_true


def calibrate_2d_joint(cal):
    """Submission-time 2D-Joint comparator (min workload over the CRC-feasible set). Verification only."""
    n = len(cal)
    vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)
    loss, wl = omission_grid_matrices(cal, vals, vals)
    adj = (n / (n + 1)) * loss.mean(0) + 1 / (n + 1)
    best = None
    for i, t in enumerate(vals):
        for j, g in enumerate(vals):
            if adj[i, j] <= ALPHA:
                key = (wl[:, i, j].mean(), -t, -g)
                if best is None or key < best[0]:
                    best = (key, t, g)
    return (0.0, 0.0) if best is None else (float(best[1]), float(best[2]))


def evaluate(test, lam, tau, gamma):
    f = [fact_stats(d, lam) for d in test]
    o = [omit_stats(d, tau, gamma) for d in test]
    n_sum = np.mean([len(d["factuality_probs"]) for d in test])
    n_src = np.mean([len(d["importance_probs"]) for d in test])
    return {
        "fact_viol": float(np.mean([x[0] for x in f])),
        "fact_wl": float(np.mean([x[1] for x in f])),
        "fact_wl_pct": float(np.mean([x[1] for x in f]) / n_sum * 100),
        "fact_recall": sum(x[2] for x in f) / max(1, sum(x[3] for x in f)),
        "omit_viol": float(np.mean([x[0] for x in o])),
        "omit_wl": float(np.mean([x[1] for x in o])),
        "omit_wl_pct": float(np.mean([x[1] for x in o]) / n_src * 100),
        "omit_recall": sum(x[2] for x in o) / max(1, sum(x[3] for x in o)),
    }


def run(docs, verify_2d=False):
    rng = random.Random(SEED)
    n = len(docs)
    n_cal = int(n * CAL_FRAC)
    per = []
    for _ in range(N_RESPLITS):
        idx = list(range(n))
        rng.shuffle(idx)
        cal = [docs[i] for i in idx[:n_cal]]
        test = [docs[i] for i in idx[n_cal:]]
        lam = calibrate_factuality(cal)
        r = select_omission_threshold_ltt_fst(cal, ALPHA, grid_resolution=GRID_OMIT)
        res = evaluate(test, lam, r["tau"], r["gamma"])
        res.update({"lambda": lam, "tau": r["tau"], "gamma": r["gamma"]})
        if verify_2d:
            t2, g2 = calibrate_2d_joint(cal)
            e2 = evaluate(test, lam, t2, g2)
            res.update({"joint2d_omit_viol": e2["omit_viol"], "joint2d_omit_wl": e2["omit_wl"]})
        per.append(res)
    keys = [k for k in per[0] if k not in ("lambda", "tau", "gamma")]
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in per])
        boot = np.random.default_rng(SEED).choice(v, (10000, len(v))).mean(1)
        summary[k] = {"mean": float(v.mean()), "ci_lower": float(np.percentile(boot, 2.5)),
                      "ci_upper": float(np.percentile(boot, 97.5))}
    return summary, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", default="phase_2_qwen_judge_05",
                    help="Phase-2 directory holding the Qwen3.5-4B judge scores")
    ap.add_argument("--verify-2d", action="store_true")
    ap.add_argument("--judges", default=",".join(JUDGES))
    a = ap.parse_args()
    out = {}
    for judge in a.judges.split(","):
        for ds, rel in JUDGES[judge].items():
            path = DATA_ROOT / rel.format(qwen_dir=a.qwen_dir)
            docs = [json.loads(l) for l in open(path)]
            summary, per = run(docs, a.verify_2d)
            out[f"{ds}__{judge}"] = {"dataset": ds, "label": LABELS[ds], "judge": judge,
                                     "source": str(rel.format(qwen_dir=a.qwen_dir)), "n_docs": len(docs),
                                     "summary": summary, "per_resplit": per}
            s = summary
            msg = (f"{LABELS[ds]:<10} {judge:<15} n={len(docs)}  hall V={s['fact_viol']['mean']*100:.1f}% "
                   f"fl={s['fact_wl_pct']['mean']:.1f}%  omit V={s['omit_viol']['mean']*100:.1f}% "
                   f"fl={s['omit_wl']['mean']:.1f} ({s['omit_wl_pct']['mean']:.1f}%) rec={s['omit_recall']['mean']*100:.1f}")
            if a.verify_2d:
                msg += f"  | 2D-Joint omit V={s['joint2d_omit_viol']['mean']*100:.1f}% fl={s['joint2d_omit_wl']['mean']:.1f}"
            print(msg, flush=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
