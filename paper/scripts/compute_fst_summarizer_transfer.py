#!/usr/bin/env python3
"""FST summarizer-transfer validity check (100 resplits).

Replaces 2D Joint argmin-workload calibration with LTT-FST (τ+γ desc) on
phase-2 calibrated_scores from alternative summarizers (Gemini, Opus,
GPT-5, GPT-5-mini). The existing 2D Joint baseline lives in
gemini_resplit_cis.json / opus_resplit_cis.json — this script saves a
parallel fst_summarizer_resplit_cis.json so the comparison is direct.

Output: paper/outputs/fst_summarizer_resplit_cis.json
"""
import json
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from care.config import (
    CANONICAL_DATA_ROOT as DATA_ROOT,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)

SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_FACT = 0.15
ALPHA_OMIT = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05

# Datasets × alternative summarizers to test.
# (ACI gemini/opus phase-2 scores live under nested phase_1_*_summ dirs.)
SUMMARIZERS = [
    ("ACI_Bench",    "ACI-Bench",  "ablations/summarizer/phase_1_gpt5_mini_summ/phase_2_gemini_summ", "gemini-2.5-pro"),
    ("ACI_Bench",    "ACI-Bench",  "ablations/summarizer/phase_1_gpt5_summ/phase_2_opus_summ",        "claude-opus-4"),
    ("ACI_Bench",    "ACI-Bench",  "ablations/summarizer/phase_2_gpt5_mini_summ",  "gpt5-mini"),
    ("ACI_Bench",    "ACI-Bench",  "ablations/summarizer/phase_2_gpt5_summ",       "gpt5"),
    ("MIMIC_IV_BHC", "MIMIC-BHC",  "ablations/summarizer/phase_2_gemini_summ",     "gemini-2.5-pro"),
    ("MIMIC_IV_BHC", "MIMIC-BHC",  "ablations/summarizer/phase_2_gpt5_mini_summ",  "gpt5-mini"),
    ("MIMIC_IV_BHC", "MIMIC-BHC",  "ablations/summarizer/phase_2_gpt5_summ",       "gpt5"),
    ("MIMIC_IV_BHC", "MIMIC-BHC",  "ablations/summarizer/phase_2_opus_summ",       "claude-opus-4"),
]


def compute_omission(doc, tau, gamma):
    imp = np.asarray(doc["importance_probs"])
    cov = np.asarray(doc["coverage_probs"])
    imp_l = np.asarray(doc["importance_labels"])
    cov_l = np.asarray(doc["coverage_labels"])
    if len(imp) == 0:
        return 0.0, 0, 0, 0
    surfaced = (imp >= tau) & ((1 - cov) >= gamma)
    n_surf = int(surfaced.sum())
    true_omit = (imp_l == 1) & (cov_l == 0)
    n_true = int(true_omit.sum())
    if n_true == 0:
        return 0.0, n_surf, 0, 0
    tp = int((surfaced & true_omit).sum())
    return float((true_omit & ~surfaced).sum() / n_true), n_surf, tp, n_true


def compute_factuality(doc, lam):
    labels = np.asarray(doc["factuality_labels"])
    probs = np.asarray(doc["factuality_probs"])
    if len(labels) == 0:
        return 0, 0, 0, 0
    flagged = probs <= lam
    halluc = labels == 0
    loss = int((halluc & ~flagged).any())
    return loss, int(flagged.sum()), int((flagged & halluc).sum()), int(halluc.sum())


def fst_calibrate_omission(cal_docs, alpha=ALPHA_OMIT, grid=GRID_OMIT):
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


def calibrate_factuality(cal_docs, alpha=ALPHA_FACT, grid=GRID_FACT):
    n = len(cal_docs)
    for lam in np.arange(0.0, 1.0 + grid, grid):
        losses = [compute_factuality(d, lam)[0] for d in cal_docs]
        adj = (n / (n + 1)) * np.mean(losses) + 1.0 / (n + 1)
        if adj <= alpha:
            return float(lam)
    return 1.0


def evaluate(test_docs, lam, tau, gamma):
    fact_losses, fact_flagged, fact_tp, fact_h = [], [], 0, 0
    omit_losses, omit_surf, omit_tp, omit_true = [], [], 0, 0
    for d in test_docs:
        fl, nf, ftp, fh = compute_factuality(d, lam)
        fact_losses.append(fl); fact_flagged.append(nf)
        fact_tp += ftp; fact_h += fh
        ol, ns, otp, ot = compute_omission(d, tau, gamma)
        omit_losses.append(ol); omit_surf.append(ns)
        omit_tp += otp; omit_true += ot
    avg_sum = float(np.mean([len(d.get("factuality_probs", [])) for d in test_docs]))
    avg_src = float(np.mean([len(d.get("importance_probs", [])) for d in test_docs]))
    return {
        "fact_viol": float(np.mean(fact_losses)),
        "fact_wl_pct": float(np.mean(fact_flagged) / avg_sum * 100) if avg_sum > 0 else 0.0,
        "fact_recall": fact_tp / fact_h if fact_h > 0 else 1.0,
        "omit_viol": float(np.mean(omit_losses)),
        "omit_wl_pct": float(np.mean(omit_surf) / avg_src * 100) if avg_src > 0 else 0.0,
        "omit_recall": omit_tp / omit_true if omit_true > 0 else 1.0,
    }


def run_resplits(ds_key, label, phase2_subdir, summarizer):
    path = DATA_ROOT / ds_key / phase2_subdir / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"  SKIP {ds_key}/{phase2_subdir}: not found")
        return None
    docs = [json.loads(l) for l in open(path)]
    n = len(docs); n_cal = int(n * CAL_FRAC)
    print(f"\n{label} / {summarizer}  (N={n}, cal={n_cal}, test={n - n_cal})")

    rng = np.random.default_rng(SEED)
    results = []
    for _ in tqdm(range(N_RESPLITS), desc=f"{label}/{summarizer}", leave=False):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]
        lam = calibrate_factuality(cal_docs)
        tau, gamma, _ = fst_calibrate_omission(cal_docs)
        r = evaluate(test_docs, lam, tau, gamma)
        r["lambda"] = lam; r["tau"] = tau; r["gamma"] = gamma
        results.append(r)

    def summ(vals):
        return {"mean": float(np.mean(vals)),
                "ci_lower": float(np.percentile(vals, 2.5)),
                "ci_upper": float(np.percentile(vals, 97.5))}

    summary = {m: summ([r[m] for r in results])
               for m in ["fact_viol", "fact_wl_pct", "fact_recall",
                         "omit_viol", "omit_wl_pct", "omit_recall"]}
    s = summary
    mark_f = "✓" if s["fact_viol"]["mean"] <= ALPHA_FACT else "✗"
    mark_o = "✓" if s["omit_viol"]["mean"] <= ALPHA_OMIT else "✗"
    print(f"  Fact V: {s['fact_viol']['mean']:.1%} {mark_f}  "
          f"Omit V: {s['omit_viol']['mean']:.1%} {mark_o}  "
          f"Fl(omit)={s['omit_wl_pct']['mean']:.1f}%")

    return {"dataset": ds_key, "label": label, "summarizer": summarizer,
            "n_total": n, "summary": summary, "per_resplit": results}


def main():
    all_results = OrderedDict()
    for ds_key, label, subdir, summarizer in SUMMARIZERS:
        r = run_resplits(ds_key, label, subdir, summarizer)
        if r:
            all_results[f"{ds_key}__{summarizer}"] = r

    out_path = OUTPUT_DIR / "fst_summarizer_resplit_cis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*78}\n  FST summarizer-transfer validity (α=0.15)\n{'='*78}")
    print(f"  {'Dataset':<14} {'Summarizer':<18} {'Fact V%':>10} {'Omit V%':>10}")
    print("  " + "-" * 60)
    for k, r in all_results.items():
        fv = r["summary"]["fact_viol"]["mean"] * 100
        ov = r["summary"]["omit_viol"]["mean"] * 100
        fm = "!" if fv > 15 else " "
        om = "!" if ov > 15 else " "
        print(f"  {r['label']:<14} {r['summarizer']:<18} {fm}{fv:9.2f} {om}{ov:9.2f}")
    print("  ! = exceeds α")


if __name__ == "__main__":
    main()
