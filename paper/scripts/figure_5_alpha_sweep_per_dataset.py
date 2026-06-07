#!/usr/bin/env python3
"""Appendix figure: α-sweep validity, per-dataset small multiples.

Complements Figure 3 (main body) by showing each dataset's safety curve in
isolation. 2 rows × 5 columns: each small panel plots one dataset's mean
violation + 2.5-97.5 percentile band (spread of per-split outcomes across
resplits, NOT a CI on the mean) vs α for one controller, with the y=α
dashed line and violation-zone wash. Eliminates color overlap so each
dataset's guarantee is visible independently.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from care.config import DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR


COLORS = {
    "ACI-Bench":  "#1f77b4",
    "MIMIC-CXR":  "#ff7f0e",
    "MIMIC-BHC":  "#2ca02c",
    "SumPubMed":  "#d62728",
    "Priv-DS":    "#9467bd",
}


def load_sweep():
    """(factuality sweep [2D-schema], omission sweep [FST-schema])."""
    fact = json.load(open(OUTPUT_DIR / "alpha_sweep_cis.json"))
    fst = json.load(open(OUTPUT_DIR / "fst_alpha_sweep_cis.json"))
    return fact, fst


def series(ds_data, method=None):
    """Violation series. method='factuality' for 2D-schema; None for FST."""
    alphas = sorted(float(a) for a in ds_data["by_alpha"].keys())
    mean, lo, hi = [], [], []
    for a in alphas:
        cell = ds_data["by_alpha"][f"{a:.2f}"]
        if method is not None:
            cell = cell[method]
        agg = cell["violation"]
        mean.append(agg["mean"])
        lo.append(agg["ci_lower"])
        hi.append(agg["ci_upper"])
    return np.array(alphas), np.array(mean), np.array(lo), np.array(hi)


def plot(fact_data, fst_data):
    ds_keys = list(fact_data.keys())
    n = len(ds_keys)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.5),
                             sharex=True, sharey=True)
    rows = [(fact_data, "factuality", "Hallucination"),
            (fst_data,  None,         "Omission (LTT-FST)")]

    for r, (data, method, row_label) in enumerate(rows):
        for c, ds_key in enumerate(ds_keys):
            ax = axes[r, c]
            label = DATASETS[ds_key]["label"]
            color = COLORS.get(label, "gray")

            # Safety backdrop
            a_lo, a_hi = 0.0, 0.55
            x_fill = np.linspace(a_lo, a_hi, 100)
            ax.fill_between(x_fill, x_fill * 100, 100,
                            color="#d62728", alpha=0.06, linewidth=0, zorder=0)
            ax.plot([a_lo, a_hi], [a_lo * 100, a_hi * 100],
                    color="#d62728", linestyle="--", linewidth=1.6,
                    alpha=0.85, zorder=2)

            # Data
            alphas, m, lo, hi = series(data[ds_key], method)
            ax.fill_between(alphas, lo * 100, hi * 100, color=color,
                            alpha=0.18, linewidth=0, zorder=3)
            ax.plot(alphas, m * 100, "-", color=color, linewidth=2.0, zorder=4)
            ax.plot(alphas, m * 100, "o", color=color, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=5)

            ax.set_xlim(0.03, 0.52)
            ax.set_ylim(0, 60)
            ax.grid(True, alpha=0.25)

            if r == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", color=color)
            if r == 1:
                ax.set_xlabel(r"$\alpha$")
            if c == 0:
                ax.set_ylabel(f"{row_label}\nViolation (%)", fontsize=10)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fact_data, fst_data = load_sweep()
    fig = plot(fact_data, fst_data)
    out = OUTPUT_DIR / "figure_5_alpha_sweep_per_dataset.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")
    plt.close(fig)
