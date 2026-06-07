#!/usr/bin/env python3
"""Figure 3: α sweep with 2.5-97.5 percentile intervals across resplits.

Reads paper/outputs/alpha_sweep_cis.json (produced by compute_alpha_sweep_cis.py,
100 random 70/30 resplits per α) and plots mean violation / workload / recall
vs. α for each dataset.

Top row: Hallucination Controller (factuality, from alpha_sweep_cis.json).
Bottom row: Omission Controller (LTT-FST, from fst_alpha_sweep_cis.json).

Violation bands are 2.5–97.5 percentile intervals across resplits, i.e. the
spread of per-split outcomes, NOT a confidence interval on the mean (the tables
report the latter). Mean trajectories are the primary signal; bands are
secondary context. The CRC guarantee bounds E[L] ≤ α (mean), so occasional
per-split excursions above α are expected and do not violate the guarantee.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

from care.config import DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR


COLORS = {
    "ACI-Bench":  "#1f77b4",
    "MIMIC-CXR":  "#ff7f0e",
    "MIMIC-BHC":  "#2ca02c",
    "SumPubMed":  "#d62728",
    "Priv-DS":    "#9467bd",
}


def load_sweep():
    fact_path = OUTPUT_DIR / "alpha_sweep_cis.json"
    fst_path = OUTPUT_DIR / "fst_alpha_sweep_cis.json"
    for p in (fact_path, fst_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")
    return json.load(open(fact_path)), json.load(open(fst_path))


def series(ds_data, field, method=None):
    """Return (alphas, mean, ci_lo, ci_hi) arrays for `field`.

    If `method` is given (e.g. 'factuality'), index into that method block
    (2D-style schema); otherwise read the field directly (FST schema).
    CIs are the 2.5/97.5 percentiles across resplits.
    """
    alphas = sorted(float(a) for a in ds_data["by_alpha"].keys())
    mean, lo, hi = [], [], []
    for a in alphas:
        cell = ds_data["by_alpha"][f"{a:.2f}"]
        if method is not None:
            cell = cell[method]
        agg = cell[field]
        mean.append(agg["mean"])
        lo.append(agg["ci_lower"])
        hi.append(agg["ci_upper"])
    return np.array(alphas), np.array(mean), np.array(lo), np.array(hi)


def plot(fact_data, fst_data):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

    # (data source, method key, row title) — factuality is 2D-schema,
    # the omission row is the FST-schema file.
    rows = [(fact_data, "factuality", "Hallucination Controller"),
            (fst_data,  None,         "Omission Controller (LTT-FST)")]
    cols = [("violation",    "Violation Rate (%)", True),
            ("workload_pct", "Flagged (%)",        False),
            ("recall",       "Recall (%)",         False)]

    for r, (data, method, row_title) in enumerate(rows):
        for c, (field, ylabel, is_violation) in enumerate(cols):
            ax = axes[r, c]

            # Paint safety backdrop FIRST so mean lines & bands sit on top
            if is_violation:
                a_lo, a_hi = 0.0, 0.55
                x_fill = np.linspace(a_lo, a_hi, 100)
                # very soft red wash in violation zone
                ax.fill_between(x_fill, x_fill * 100, 100,
                                color="#d62728", alpha=0.05,
                                linewidth=0, zorder=0)
                # prominent dashed y=α line
                ax.plot([a_lo, a_hi], [a_lo * 100, a_hi * 100],
                        color="#d62728", linestyle="--", linewidth=2.0,
                        alpha=0.85, zorder=2)
                # small annotation in the zone
                ax.annotate("violation\nzone (y > α)", xy=(0.475, 52),
                            color="#a11", fontsize=8, alpha=0.7,
                            ha="center", va="top", style="italic")
                ax.set_ylim(0, 60)

            # Data layer — mean lines only (per-dataset CIs in appendix Fig 5)
            for ds_key, ds_data in data.items():
                label = DATASETS[ds_key]["label"]
                alphas, m, _, _ = series(ds_data, field, method)
                color = COLORS.get(label, "gray")
                scale = 100.0 if field != "workload_pct" else 1.0

                ax.plot(alphas, m * scale, "-", color=color,
                        linewidth=2.2, zorder=4, label=label)
                ax.plot(alphas, m * scale, "o", color=color,
                        markersize=5, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=5)

            ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel(r"$\alpha$")
            ax.grid(True, alpha=0.25, zorder=0)
            ax.set_xlim(0.03, 0.52)

        # Row title over the middle panel
        axes[r, 1].set_title(row_title, fontsize=12, fontweight="bold")

    # Single combined legend outside plot area, top right of figure
    dataset_handles = [
        Line2D([0], [0], color=COLORS[DATASETS[k]["label"]], linewidth=2.2,
               marker="o", markersize=5, label=DATASETS[k]["label"])
        for k in data.keys()
    ]
    safety_handles = [
        Line2D([0], [0], color="#d62728", linestyle="--", linewidth=2.0,
               label=r"$y=\alpha$ (safety bound)"),
    ]
    axes[0, 2].legend(handles=dataset_handles + safety_handles,
                      fontsize=8, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fact_data, fst_data = load_sweep()
    fig = plot(fact_data, fst_data)
    out = OUTPUT_DIR / "figure_3_alpha_sweep.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")
    plt.close(fig)
