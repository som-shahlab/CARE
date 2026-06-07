#!/usr/bin/env python3
"""
Table A5: Distribution Shift Ablation (100 resplits)

Renders the distribution-shift table from the precomputed LTT-FST results in
``paper/outputs/fst_distribution_shift.json`` (produced by
``compute_fst_distribution_shift.py``). This script does NOT touch the clinical
Phase-2 data — it is a pure renderer, so it reproduces Table A5 without PHI
access.

Two experiments (see the compute script for details):
  (a) Cross-dataset transfer: calibrate on dataset A, evaluate on dataset B.
  (b) Within-dataset length shift: calibrate on short docs, test on long
      (and vice versa); "random" is the standard 70/30 resplit.
"""
import json
from pathlib import Path
from care.config import DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR

ALPHA = 0.15


# ─── LaTeX generation ───

def generate_latex(omit_matrix, length_results):
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Distribution shift experiments at $\alpha=0.15$, "
                 r"averaged over 100 random resplits. "
                 r"\textbf{(a)}~Cross-dataset threshold transfer: rows = calibration dataset, "
                 r"columns = test dataset. Diagonal (shaded) = in-distribution. "
                 r"\textbf{(b)}~Within-dataset length shift: calibrate on short documents, "
                 r"test on long (and vice versa).}")
    lines.append(r"\label{tab:distribution-shift}")
    lines.append(r"\vspace{4pt}")

    # Part (a): Cross-dataset
    ds_keys = list(DATASETS.keys())
    lines.append(r"\textbf{(a) Cross-dataset transfer (omission frac.\ violation \%)}")
    lines.append(r"\vspace{2pt}")
    n_ds = len(DATASETS)
    lines.append(r"\begin{tabular}{l " + "c" * n_ds + "}")
    lines.append(r"\toprule")

    header = r"Cal $\backslash$ Test"
    for ds in ds_keys:
        header += f" & {DATASETS[ds]['short']}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for cal_ds in ds_keys:
        row = DATASETS[cal_ds]["short"]
        for test_ds in ds_keys:
            v = omit_matrix[cal_ds][test_ds]["violation"] * 100
            if cal_ds == test_ds:
                cell = f"\\cellcolor{{gray!20}}\\textbf{{{v:.1f}}}"
            elif v > ALPHA * 100:
                cell = f"{v:.1f}$^\\dagger$"
            else:
                cell = f"{v:.1f}"
            row += f" & {cell}"
        row += r" \\"
        lines.append(f"  {row}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{8pt}")

    # Part (b): Length shift
    lines.append("")
    lines.append(r"\textbf{(b) Within-dataset length shift (violation \%)}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\begin{tabular}{l ccc ccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{Factuality} & \multicolumn{3}{c}{Omission (frac.)} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"Dataset & Short$\to$Long & Long$\to$Short & Random "
                 r"& Short$\to$Long & Long$\to$Short & Random \\")
    lines.append(r"\midrule")

    for ds in ds_keys:
        label = DATASETS[ds]["short"]
        r = length_results[ds]
        sl = r.get("short→long", {})
        ls = r.get("long→short", {})
        rand = r.get("random→random", {})

        def fmt(val):
            if val is None:
                return "---"
            s = f"{val*100:.1f}"
            if val > ALPHA:
                s += r"$^\dagger$"
            return s

        row = f"  {label}"
        row += f" & {fmt(sl.get('fact_violation'))}"
        row += f" & {fmt(ls.get('fact_violation'))}"
        row += f" & {fmt(rand.get('fact_violation'))}"
        row += f" & {fmt(sl.get('omit_violation'))}"
        row += f" & {fmt(ls.get('omit_violation'))}"
        row += f" & {fmt(rand.get('omit_violation'))}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\par\small $^\dagger$Exceeds $\alpha=0.15$. Shaded = in-distribution (diagonal).")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    src = OUTPUT_DIR / "fst_distribution_shift.json"
    if not src.exists():
        raise SystemExit(
            f"Missing {src}. Run compute_fst_distribution_shift.py first "
            "(it reads the Phase-2 clinical scores)."
        )
    with open(src) as f:
        data = json.load(f)

    omit_matrix = data["cross_dataset"]
    length_results = data["length_shift"]

    latex = generate_latex(omit_matrix, length_results)

    out_path = OUTPUT_DIR / "table_12_distribution_shift.tex"
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX to {out_path}")
    print("\n" + latex)


if __name__ == "__main__":
    main()
