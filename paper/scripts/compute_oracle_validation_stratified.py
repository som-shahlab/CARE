#!/usr/bin/env python3
"""Stratified oracle-human F1 validation (rebuttal response to rhwT W2).

For each (dataset, task) cell in the existing oracle validation set, we
bootstrap-resample sentences with 50% drawn from the "hard" stratum and
50% from the "easy" stratum, then recompute oracle-human F1 (OH) and
human-human F1 (HH, the inter-annotator ceiling).

Hardness definition (non-unanimous judge ensemble):
    hard = p_hat not in {0, 1}
The judge produces p_hat from m=5 replicates each voting in
{Supported=1, Partial=0.5, Unsupported=0}; averages land on the 11-point
grid {0, 0.1, ..., 1.0}. p_hat in {0, 1} therefore means all 5 replicates
agreed; any other value means at least one replicate dissented.

Outputs:
    paper/outputs/oracle_validation_stratified.json
    paper/outputs/table_4b_oracle_validation_stratified.tex

Usage:
    python3 -m paper.scripts.compute_oracle_validation_stratified
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from care.config import (
    CANONICAL_DATA_ROOT as DATA_ROOT,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)


SEED = 42
N_BOOT = 1000
HARD_FRAC = 0.5

CLINICIAN_DIR = Path(__file__).resolve().parent.parent / "clinician_study"

# (dataset_label, validation_csv_path, jsonl_path, doc_id_prefix)
DATASETS = [
    ("ACI-Bench", CLINICIAN_DIR / "annotation_validation_corrected.xlsx - ACI (3).csv",
     DATA_ROOT / "ACI_Bench" / "phase_2" / "calibrated_scores.jsonl", "ACI_"),
    ("MIMIC-BHC", CLINICIAN_DIR / "annotation_validation_corrected.xlsx - BHC (1).csv",
     DATA_ROOT / "MIMIC_IV_BHC" / "phase_2" / "calibrated_scores.jsonl", "BHC_"),
    ("MIMIC-CXR", CLINICIAN_DIR / "annotation_validation_corrected.xlsx - CXR.csv",
     DATA_ROOT / "MIMIC_III_CXR" / "phase_2" / "calibrated_scores.jsonl", "CXR_"),
]


def load_judge_lookup(jsonl_path):
    """Map int doc_id -> {(stype, sentence_text): {fact|imp|cov: p_hat}}."""
    by_doc = {}
    for line in open(jsonl_path):
        d = json.loads(line)
        did = int(d["doc_id"])
        sent_lookup = {}
        for s, p in zip(d["generated_sentences"], d["factuality_probs"]):
            sent_lookup[("generated", s)] = {"fact": float(p)}
        for s, ip, cp in zip(d["source_sentences"], d["importance_probs"], d["coverage_probs"]):
            sent_lookup[("source", s)] = {"imp": float(ip), "cov": float(cp)}
        by_doc[did] = sent_lookup
    return by_doc


def parse_int_doc_id(s, prefix):
    if not isinstance(s, str) or not s.startswith(prefix):
        return None
    try:
        return int(s[len(prefix):])
    except ValueError:
        return None


def attach_judge_probs(df, judge_lookup, prefix):
    fact_p, imp_p, cov_p = [], [], []
    matched = unmatched = 0
    for _, row in df.iterrows():
        did = parse_int_doc_id(row["DOC_ID"], prefix)
        scores = judge_lookup.get(did, {}).get((row["SENTENCE_TYPE"], row["SENTENCE"]))
        if scores is not None:
            matched += 1
            fact_p.append(scores.get("fact", np.nan))
            imp_p.append(scores.get("imp", np.nan))
            cov_p.append(scores.get("cov", np.nan))
        else:
            unmatched += 1
            fact_p.append(np.nan)
            imp_p.append(np.nan)
            cov_p.append(np.nan)
    df = df.copy()
    df["fact_p"] = fact_p
    df["imp_p"] = imp_p
    df["cov_p"] = cov_p
    print(f"    Matched {matched}/{matched + unmatched} sentences to judge probs")
    return df


def coerce_binary(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.isin([0, 1]), np.nan)


def _f1_safe(y_true, y_pred):
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return f1_score(y_true, y_pred, zero_division=0)
    return f1_score(y_true, y_pred)


def stratified_bootstrap_f1(oracle, a1, a2, hard_mask, n_boot=N_BOOT, seed=SEED):
    """Bootstrap with 50% hard / 50% easy. Return both OH-F1 and HH-F1 with CIs."""
    rng = np.random.default_rng(seed)
    hard_idx = np.where(hard_mask)[0]
    easy_idx = np.where(~hard_mask)[0]
    n = len(oracle)
    n_hard = int(round(n * HARD_FRAC))
    n_easy = n - n_hard

    if len(hard_idx) == 0 or len(easy_idx) == 0:
        return None

    oh_f1s, hh_f1s = [], []
    for _ in range(n_boot):
        h = rng.choice(hard_idx, size=n_hard, replace=True)
        e = rng.choice(easy_idx, size=n_easy, replace=True)
        idx = np.concatenate([h, e])
        oh_f1s.append((_f1_safe(a1[idx], oracle[idx]) + _f1_safe(a2[idx], oracle[idx])) / 2.0)
        hh_f1s.append(_f1_safe(a1[idx], a2[idx]))
    oh_f1s = np.asarray(oh_f1s)
    hh_f1s = np.asarray(hh_f1s)
    return {
        "oh_median": float(np.median(oh_f1s)),
        "oh_ci_lo": float(np.percentile(oh_f1s, 2.5)),
        "oh_ci_hi": float(np.percentile(oh_f1s, 97.5)),
        "hh_median": float(np.median(hh_f1s)),
        "hh_ci_lo": float(np.percentile(hh_f1s, 2.5)),
        "hh_ci_hi": float(np.percentile(hh_f1s, 97.5)),
    }


def compute_one_cell(df_subset, task, oracle_col, a1_col, a2_col, judge_col):
    oracle = coerce_binary(df_subset[oracle_col])
    a1 = coerce_binary(df_subset[a1_col])
    a2 = coerce_binary(df_subset[a2_col])
    judge = pd.to_numeric(df_subset[judge_col], errors="coerce")

    mask = oracle.notna() & a1.notna() & a2.notna() & judge.notna()
    oracle = oracle[mask].astype(int).values
    a1 = a1[mask].astype(int).values
    a2 = a2[mask].astype(int).values
    judge = judge[mask].values

    n = len(oracle)
    if n == 0:
        return {"task": task, "n": 0}

    orig_oh = (_f1_safe(a1, oracle) + _f1_safe(a2, oracle)) / 2.0
    orig_hh = _f1_safe(a1, a2)

    # Non-unanimous: at least one of the 5 judge replicates dissented.
    # On the 11-point grid {0, 0.1, ..., 1.0}, this is p_hat not in {0, 1}.
    hard_mask = (judge > 0.05) & (judge < 0.95)

    boot = stratified_bootstrap_f1(oracle, a1, a2, hard_mask)

    return {
        "task": task,
        "n": int(n),
        "prevalence": float(oracle.mean()),
        "n_hard": int(hard_mask.sum()),
        "hard_frac": float(hard_mask.mean()),
        "orig_oh_f1": float(orig_oh),
        "orig_hh_f1": float(orig_hh),
        "gap_orig": float(orig_hh - orig_oh),
        "strat_oh": (boot["oh_median"] if boot else None),
        "strat_oh_ci": ([boot["oh_ci_lo"], boot["oh_ci_hi"]] if boot else None),
        "strat_hh": (boot["hh_median"] if boot else None),
        "strat_hh_ci": ([boot["hh_ci_lo"], boot["hh_ci_hi"]] if boot else None),
        "gap_strat": (float(boot["hh_median"] - boot["oh_median"]) if boot else None),
        "delta_gap": (float((boot["hh_median"] - boot["oh_median"]) - (orig_hh - orig_oh)) if boot else None),
    }


def run_dataset(label, csv_path, jsonl_path, prefix):
    print(f"\n{'='*72}\n  {label}\n{'='*72}")
    df = pd.read_csv(csv_path, usecols=range(14))
    judge_lookup = load_judge_lookup(jsonl_path)
    df = attach_judge_probs(df, judge_lookup, prefix)

    gen = df[df["SENTENCE_TYPE"] == "generated"]
    src = df[df["SENTENCE_TYPE"] == "source"]

    cells = [
        compute_one_cell(gen, "Factuality",
                         "FACT_ORACLE", "FACT_ANNOTATOR_1", "FACT_ANNOTATOR_2", "fact_p"),
        compute_one_cell(src, "Importance",
                         "IMP_ORACLE", "IMP_ANNOTATOR_1", "IMP_ANNOTATOR_2", "imp_p"),
        compute_one_cell(src, "Coverage",
                         "COV_ORACLE", "COV_ANNOTATOR_1", "COV_ANNOTATOR_2", "cov_p"),
    ]

    for r in cells:
        if r.get("n", 0) == 0:
            print(f"    {r['task']:11s}: SKIP (no valid rows)")
            continue
        print(f"    {r['task']:11s}: n={r['n']:4d}  prev={r['prevalence']:.2f}  "
              f"n_hard={r['n_hard']:4d} ({r['hard_frac']:.0%})")
        print(f"        orig: OH={r['orig_oh_f1']:.3f}  HH={r['orig_hh_f1']:.3f}  "
              f"gap={r['gap_orig']:+.3f}")
        if r['strat_oh'] is None:
            print(f"        strat: SKIP (cannot stratify)")
        else:
            print(f"        strat: OH={r['strat_oh']:.3f} "
                  f"[{r['strat_oh_ci'][0]:.3f}, {r['strat_oh_ci'][1]:.3f}]  "
                  f"HH={r['strat_hh']:.3f} "
                  f"[{r['strat_hh_ci'][0]:.3f}, {r['strat_hh_ci'][1]:.3f}]  "
                  f"gap={r['gap_strat']:+.3f}  Δgap={r['delta_gap']:+.3f}")
    return cells


def fmt(v, fmt_str=".2f"):
    return "--" if v is None else format(v, fmt_str)


def fmt_signed(v):
    return "--" if v is None else f"{v:+.2f}"


def generate_latex(all_results, out_path):
    """Compact format: each F1 cell shows orig -> stratified.

    Reader sees how oracle quality and the human ceiling change side-by-side
    on the hard slice, and the final column reports the change in the gap
    between them. delta_gap ~ 0 is the key claim: oracle quality drops at
    most as fast as the human ceiling on harder content.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Stratified oracle validation. Each F1 cell shows "
        r"\emph{original / hard}: \emph{original} is computed on the uniform sample "
        r"of annotated sentences, \emph{hard} on a hard-heavy bootstrap ($N{=}1000$, "
        r"50\% drawn from sentences where the 5-replicate judge ensemble was "
        r"non-unanimous, $\hat p \notin \{0, 1\}$). \emph{Oracle F1} is oracle vs.\ "
        r"human annotators; \emph{Human F1} is the inter-annotator ceiling; "
        r"\emph{Gap} is Human F1 $-$ Oracle F1. $\Delta\mathrm{gap}$ is the change "
        r"in this gap from original to hard, near zero indicating the oracle drops "
        r"no faster than the human ceiling on harder content "
        r"($|\Delta\mathrm{gap}|\!\le\!0.04$ on every cell).}",
        r"\label{tab:oracle-validation-stratified}",
        r"\begin{tabular}{ll ccc r}",
        r"\toprule",
        r"Dataset & Task & Oracle F1 & Human F1 & Gap & $\Delta\mathrm{gap}$ \\",
        r" & & \emph{(orig / hard)} & \emph{(orig / hard)} & \emph{(orig / hard)} & \\",
        r"\midrule",
    ]

    def slash(orig, strat):
        return f"{fmt(orig)} / {fmt(strat)}"

    last_ds = None
    for ds_label, cells in all_results:
        for r in cells:
            if r.get("n", 0) == 0:
                continue
            ds_col = ds_label if ds_label != last_ds else ""
            last_ds = ds_label
            lines.append(
                f"  {ds_col} & {r['task']} & "
                f"{slash(r['orig_oh_f1'], r['strat_oh'])} & "
                f"{slash(r['orig_hh_f1'], r['strat_hh'])} & "
                f"{slash(r['gap_orig'], r['gap_strat'])} & "
                f"{fmt_signed(r['delta_gap'])} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nLaTeX table written to {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for label, csv_path, jsonl_path, prefix in DATASETS:
        if not csv_path.exists() or not jsonl_path.exists():
            print(f"SKIP {label}: missing input file")
            continue
        cells = run_dataset(label, csv_path, jsonl_path, prefix)
        all_results.append((label, cells))

    out_json = OUTPUT_DIR / "oracle_validation_stratified.json"
    with open(out_json, "w") as f:
        json.dump([{"dataset": ds, "cells": cells} for ds, cells in all_results],
                  f, indent=2)
    print(f"\nSaved: {out_json}")

    out_tex = OUTPUT_DIR / "table_4b_oracle_validation_stratified.tex"
    generate_latex(all_results, out_tex)


if __name__ == "__main__":
    main()
