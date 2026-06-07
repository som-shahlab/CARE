#!/usr/bin/env python3
"""Render Table B (2D Joint vs. LTT-FST) from resplit_cis.json + fst_resplit_cis.json.

Produces `paper/outputs/table_fst_vs_2djoint.tex`, joining the existing 2D Joint
heuristic CIs (resplit_cis.json, produced by compute_resplit_cis.py) with the
LTT-FST tau_plus_gamma ordering CIs (fst_resplit_cis.json, produced by
compute_fst_resplit_cis.py). Both runs use the same seeds and 70/30 splits, so
per-dataset rows are directly comparable.

The output mirrors the inlined `tab:fst-vs-2djoint` tabular in paper.tex
byte-for-byte (LTT-FST primary, 2D Joint comparator, CI on the mean), so the
two stay reconciled: regenerate and diff against the paper, or `\input` this
file to make the script the single source of truth.

Usage:
    python3 -m paper.scripts.render_table_fst_vs_2djoint
"""
import json

import numpy as np

from care.config import PAPER_OUTPUT_DIR as OUTPUT_DIR

# 95% bootstrap CI on the mean (matches Table 2 / Table 3 convention).
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 42


def bootstrap_mean_ci(values, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """95% bootstrap CI on the MEAN of per-resplit values (fractions)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(iters, n))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

JOINT_PATH = OUTPUT_DIR / "resplit_cis.json"
FST_PATH = OUTPUT_DIR / "fst_resplit_cis.json"
OUT_PATH = OUTPUT_DIR / "table_fst_vs_2djoint.tex"

DATASETS = [
    ("ACI_Bench",     "ACI-Bench"),
    ("MIMIC_IV_BHC",  "MIMIC-BHC"),
    ("MIMIC_III_CXR", "MIMIC-CXR"),
    ("OMOP",          "Priv-DS"),
    ("SumPubMed",     "SumPubMed"),
]

FST_ORDERING = "tau_plus_gamma"


def fmt_viol_ci_from_resplits(per_viol):
    """Format 'mean [lo, hi]' using a 95% bootstrap CI on the mean (in %)."""
    vals = np.asarray(per_viol, dtype=float) * 100
    lo, hi = bootstrap_mean_ci(vals)
    return f"{vals.mean():.1f} [{lo:.1f}, {hi:.1f}]"


def fmt_wl(stats):
    return f"{stats['mean']:.1f}"


def render(joint_data, fst_data):
    # Column order, caption, and styling match the inlined `tab:fst-vs-2djoint`
    # in paper.tex exactly (LTT-FST primary, 2D Joint as comparator) so the
    # generated table can be diffed against — or \input into — the paper.
    rows = []
    ratios = []
    for ds_key, ds_label in DATASETS:
        joint = joint_data[ds_key]["summary"]
        fst = fst_data[ds_key]["fst_summary"][FST_ORDERING]

        # CI on the mean is computed from per-resplit omission violations.
        joint_per = [r["omit_viol"] for r in joint_data[ds_key]["per_resplit"]]
        fst_per = [r["fst"][FST_ORDERING]["omit_viol"]
                   for r in fst_data[ds_key]["per_resplit"]]

        joint_viol = fmt_viol_ci_from_resplits(joint_per)
        joint_wl_mean = joint["omit_wl"]["mean"]
        fst_viol = fmt_viol_ci_from_resplits(fst_per)
        fst_wl_mean = fst["omit_wl"]["mean"]

        ratio = fst_wl_mean / joint_wl_mean if joint_wl_mean > 0 else float("nan")
        ratios.append(ratio)

        # LTT-FST columns first, then 2D Joint comparator.
        rows.append(
            f"{ds_label:<10} & {fst_viol:<18} & {fst_wl_mean:>5.1f} & "
            f"{joint_viol:<18} & {joint_wl_mean:>5.1f} & {ratio:.2f}$\\times$ \\\\"
        )

    ratio_lo, ratio_hi = min(ratios), max(ratios)

    L = []
    L.append(r"\begin{table}[h]")
    L.append(r"\centering")
    L.append(r"\tiny")
    L.append(
        r"\caption{LTT-FST versus the unconstrained workload-minimizing "
        r"comparator at $\alpha=0.15$, averaged over 100 random 70/30 "
        r"calibration/test resplits per dataset. LTT-FST follows a pre-specified "
        r"ordering and carries the fixed-sequence guarantee; the comparator "
        r"minimizes workload over the empirical feasible set without an explicit "
        r"multiple-testing correction. Ratio denotes LTT-FST workload divided by "
        r"comparator workload. Brackets denote a 95\% bootstrap CI on the mean "
        r"violation rate across resplits.}"
    )
    L.append(r"\label{tab:fst-vs-2djoint}")
    L.append(r"\resizebox{\columnwidth}{!}{%")
    L.append(r"\begin{tabular}{l cc cc r}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{2}{c}{\textbf{LTT-FST}} & "
             r"\multicolumn{2}{c}{\textbf{2D Joint (comparator)}} & \\")
    L.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    L.append(r"Dataset & V\% [95\% CI] & WL & V\% [95\% CI] & WL & Ratio \\")
    L.append(r"\midrule")
    L.extend(rows)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L), ratio_lo, ratio_hi


def main():
    for path in (JOINT_PATH, FST_PATH):
        if not path.exists():
            raise SystemExit(
                f"ERROR: {path} not found. Run "
                f"{'compute_resplit_cis.py' if path == JOINT_PATH else 'compute_fst_resplit_cis.py'} "
                f"first."
            )
    joint = json.loads(JOINT_PATH.read_text())
    fst = json.loads(FST_PATH.read_text())
    tex, ratio_lo, ratio_hi = render(joint, fst)
    OUT_PATH.write_text(tex + "\n")
    print(f"Saved: {OUT_PATH}")
    print(f"FST/Joint workload ratio range: {ratio_lo:.2f}x to {ratio_hi:.2f}x")


if __name__ == "__main__":
    main()
