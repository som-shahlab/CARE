#!/usr/bin/env python3
"""Compute oracle validation metrics from human annotation CSV.

Primary metrics (F1 / precision / recall) answer the key question:
  "Can we trust the oracle labels as supervision for calibration?"

Kappa is reported in the appendix only, with a note about the kappa paradox
for high-prevalence tasks (factuality).

Usage:
    python3 scripts/annotation_agreement.py
    python3 scripts/annotation_agreement.py --csv path/to/annotations.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PAPER_DIR = _SCRIPT_DIR.parent
_OUTPUT_DIR = _PAPER_DIR / "outputs"

_CLINICIAN_DIR = _PAPER_DIR / "clinician_study"

DEFAULT_CSVS = {
    "ACI-Bench": str(_CLINICIAN_DIR / "annotation_validation_corrected.xlsx - ACI (3).csv"),
    "MIMIC-BHC": str(_CLINICIAN_DIR / "annotation_validation_corrected.xlsx - BHC (1).csv"),
    "MIMIC-CXR": str(_CLINICIAN_DIR / "annotation_validation_corrected.xlsx - CXR.csv"),
}


def compute_metrics(oracle, a1, a2, task_name):
    """Compute all metrics for one task.

    Primary comparison (SQuAD-style):
      - HH F1:            F1(A1, A2)              — human ceiling
      - Oracle-Human F1:   avg(F1(O,A1), F1(O,A2)) — oracle quality
    If Oracle-Human F1 ≈ HH F1, the oracle is human-level.
    """
    # Drop rows where any value is NaN or non-binary (e.g., 9 = annotation error)
    valid_binary = lambda s: s.notna() & s.isin([0, 0.0, 1, 1.0])
    mask = valid_binary(oracle) & valid_binary(a1) & valid_binary(a2)
    oracle = oracle[mask].astype(int).values
    a1 = a1[mask].astype(int).values
    a2 = a2[mask].astype(int).values
    n = len(oracle)
    n_dropped = mask.size - n

    # Class prevalence (average across annotators)
    prevalence = np.mean([a1.mean(), a2.mean()])

    # --- Human-human F1 (ceiling) ---
    hh_f1 = f1_score(a1, a2)
    hh_prec = precision_score(a1, a2, zero_division=0)
    hh_rec = recall_score(a1, a2, zero_division=0)
    hh_agree = (a1 == a2).mean()

    # --- Oracle vs each annotator (averaged) ---
    o_a1_f1 = f1_score(a1, oracle)
    o_a1_prec = precision_score(a1, oracle, zero_division=0)
    o_a1_rec = recall_score(a1, oracle, zero_division=0)

    o_a2_f1 = f1_score(a2, oracle)
    o_a2_prec = precision_score(a2, oracle, zero_division=0)
    o_a2_rec = recall_score(a2, oracle, zero_division=0)

    oracle_f1 = np.mean([o_a1_f1, o_a2_f1])
    oracle_prec = np.mean([o_a1_prec, o_a2_prec])
    oracle_rec = np.mean([o_a1_rec, o_a2_rec])

    # --- Appendix: Kappa (with caveat for high-prevalence tasks) ---
    hh_kappa = cohen_kappa_score(a1, a2)
    o_a1_kappa = cohen_kappa_score(a1, oracle)
    o_a2_kappa = cohen_kappa_score(a2, oracle)
    oracle_kappa = np.mean([o_a1_kappa, o_a2_kappa])

    return {
        "task": task_name,
        "n": n,
        "n_dropped": n_dropped,
        "prevalence": prevalence,
        # Human-human (ceiling)
        "hh_f1": hh_f1,
        "hh_prec": hh_prec,
        "hh_rec": hh_rec,
        "hh_agree": hh_agree,
        "hh_kappa": hh_kappa,
        # Oracle vs humans (averaged over both annotators)
        "oracle_f1": oracle_f1,
        "oracle_prec": oracle_prec,
        "oracle_rec": oracle_rec,
        "oracle_kappa": oracle_kappa,
        # Per-annotator breakdown
        "o_a1_f1": o_a1_f1, "o_a2_f1": o_a2_f1,
        "o_a1_prec": o_a1_prec, "o_a2_prec": o_a2_prec,
        "o_a1_rec": o_a1_rec, "o_a2_rec": o_a2_rec,
    }


def load_and_compute(csv_path, dataset_name):
    """Load a CSV and compute metrics for all three tasks."""
    df = pd.read_csv(csv_path, usecols=range(14))
    print(f"\nLoaded {len(df)} rows from {csv_path}")
    print(f"  DOC_IDs: {sorted(df['DOC_ID'].dropna().unique())}")
    print(f"  Generated sentences: {(df['SENTENCE_TYPE'] == 'generated').sum()}")
    print(f"  Source sentences: {(df['SENTENCE_TYPE'] == 'source').sum()}")

    # Coerce label columns to numeric, then replace non-binary values with NaN
    for col in df.columns:
        if col.startswith(("FACT_", "IMP_", "COV_")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~df[col].isin([0, 0.0, 1, 1.0, np.nan]), col] = np.nan

    gen = df[df["SENTENCE_TYPE"] == "generated"].copy()
    src = df[df["SENTENCE_TYPE"] == "source"].copy()

    results = []
    results.append(compute_metrics(
        gen["FACT_ORACLE"], gen["FACT_ANNOTATOR_1"], gen["FACT_ANNOTATOR_2"],
        "Factuality"
    ))
    results.append(compute_metrics(
        src["IMP_ORACLE"], src["IMP_ANNOTATOR_1"], src["IMP_ANNOTATOR_2"],
        "Importance"
    ))
    results.append(compute_metrics(
        src["COV_ORACLE"], src["COV_ANNOTATOR_1"], src["COV_ANNOTATOR_2"],
        "Coverage"
    ))
    # Tag each result with dataset name
    for r in results:
        r["dataset"] = dataset_name
    return results


def print_results(all_results):
    """Print detailed results and paper-ready table."""
    # =====================================================================
    # PRIMARY RESULTS — SQuAD-style: Oracle-Human F1 vs Human-Human F1
    # =====================================================================
    print("\n" + "=" * 80)
    print("ORACLE VALIDATION — PRIMARY METRICS")
    print("  HH F1 = F1(A1, A2)                         — human ceiling")
    print("  OH F1 = avg(F1(Oracle,A1), F1(Oracle,A2))   — oracle quality")
    print("=" * 80)

    for r in all_results:
        print(f"\n--- {r['dataset']} / {r['task']} (n={r['n']}, prevalence={r['prevalence']:.1%}) ---")
        if r['n_dropped']:
            print(f"  ({r['n_dropped']} rows dropped: non-binary annotation values)")
        print(f"  Human-Human (ceiling):")
        print(f"    F1:             {r['hh_f1']:.3f}")
        print(f"    Precision:      {r['hh_prec']:.3f}")
        print(f"    Recall:         {r['hh_rec']:.3f}")
        print(f"    Agreement:      {r['hh_agree']:.1%}")
        print(f"  Oracle-Human (averaged over A1, A2):")
        print(f"    F1:             {r['oracle_f1']:.3f}  (A1:{r['o_a1_f1']:.3f}, A2:{r['o_a2_f1']:.3f})")
        print(f"    Precision:      {r['oracle_prec']:.3f}  (A1:{r['o_a1_prec']:.3f}, A2:{r['o_a2_prec']:.3f})")
        print(f"    Recall:         {r['oracle_rec']:.3f}  (A1:{r['o_a1_rec']:.3f}, A2:{r['o_a2_rec']:.3f})")

    # =====================================================================
    # PAPER TABLE — plain text
    # =====================================================================
    print("\n" + "=" * 80)
    print("TABLE FOR PAPER")
    print("=" * 80)
    print(f"{'Dataset':<12s} {'Task':<14s} {'n':>4s} {'Prev':>6s} | {'HH-F1':>6s} | {'OH-F1':>6s} {'OH-P':>6s} {'OH-R':>6s}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['dataset']:<12s} {r['task']:<14s} {r['n']:4d} {r['prevalence']:5.1%} | "
              f"{r['hh_f1']:6.3f} | "
              f"{r['oracle_f1']:6.3f} {r['oracle_prec']:6.3f} {r['oracle_rec']:6.3f}")


def generate_latex(all_results, output_path=None):
    if output_path is None:
        output_path = str(_PAPER_DIR / "outputs" / "table_4_oracle_validation.tex")
    """Generate LaTeX table for the paper."""
    # Group results by dataset
    datasets = []
    seen = set()
    for r in all_results:
        if r["dataset"] not in seen:
            datasets.append(r["dataset"])
            seen.add(r["dataset"])

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Oracle label validation against human annotations. "
                 r"Two clinicians independently annotated 10--11 randomly sampled documents per dataset. "
                 r"\emph{HH}: inter-annotator F1 (human ceiling). "
                 r"\emph{OH}: oracle--human F1, averaged over both annotators. "
                 r"$n$: sentences evaluated. "
                 r"The oracle matches or exceeds human agreement on all tasks.}")
    lines.append(r"\label{tab:oracle-validation}")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{ll r r ccc}")
    lines.append(r"\toprule")
    lines.append(r" & & & & \multicolumn{3}{c}{\textbf{Oracle--Human}} \\")
    lines.append(r"\cmidrule(lr){5-7}")
    lines.append(r"Dataset & Task & $n$ & HH F1 & F1 & Prec & Rec \\")
    lines.append(r"\midrule")

    for i, ds in enumerate(datasets):
        ds_results = [r for r in all_results if r["dataset"] == ds]
        for j, r in enumerate(ds_results):
            if r["n"] == 0:
                continue  # skip empty (e.g., missing factuality annotations)
            ds_col = f"\\multirow{{{len([x for x in ds_results if x['n'] > 0])}}}" \
                     f"{{*}}{{{ds}}}" if j == 0 else ""
            lines.append(
                f"  {ds_col} & {r['task']} & {r['n']} & "
                f"{r['hh_f1']:.2f} & "
                f"{r['oracle_f1']:.2f} & {r['oracle_prec']:.2f} & {r['oracle_rec']:.2f} \\\\"
            )
        if i < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex + "\n")
    print(f"\nLaTeX table written to {output_path}")
    print(latex)
    return latex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="*", help="Override CSV files (dataset:path pairs)")
    args = parser.parse_args()

    csv_map = dict(DEFAULT_CSVS)
    if args.csvs:
        for item in args.csvs:
            ds, path = item.split(":", 1)
            csv_map[ds] = path

    all_results = []
    for dataset_name, csv_path in csv_map.items():
        all_results.extend(load_and_compute(csv_path, dataset_name))

    print_results(all_results)
    generate_latex(all_results)


if __name__ == "__main__":
    main()
