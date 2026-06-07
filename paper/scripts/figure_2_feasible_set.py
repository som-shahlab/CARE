#!/usr/bin/env python3
"""Figure 2: omission calibration feasible set on ACI-Bench.

Shows the feasible set F(alpha) in (tau, gamma) space and the operating
points of 1D-Imp, Union Bound, 2D Joint, and the LTT-FST controller.

Usage:
    python3 paper/scripts/figure_2_feasible_set.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, PAPER_OUTPUT_DIR as OUTPUT_DIR

ALPHA = 0.15
GRID = 0.05
DATASET = "ACI_Bench"
CAL_FRAC = 0.70
SEED = 42


def load_cal_docs():
    """Load calibration split of ACI-Bench Phase 2 data using fixed split."""
    path = DATA_ROOT / DATASET / "phase_2" / "calibrated_scores.jsonl"
    docs = [json.loads(l) for l in open(path)]
    split_path = DATA_ROOT / DATASET / "split" / "split_indices.json"
    split = json.load(open(split_path))
    cal_ids = set(split["calibration_indices"])
    return [d for d in docs if d["doc_id"] in cal_ids]


def compute_risk_and_workload(cal_docs, tau, gamma):
    """Compute adjusted fractional omission risk and workload % at (tau, gamma)."""
    losses = []
    total_surfaced = 0
    total_source = 0

    for d in cal_docs:
        imp_probs = np.array(d["importance_probs"])
        cov_probs = np.array(d["coverage_probs"])
        imp_labels = np.array(d["importance_labels"])
        cov_labels = np.array(d["coverage_labels"])

        if len(imp_labels) == 0:
            losses.append(0.0)
            continue

        surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
        true_omit = (imp_labels == 1) & (cov_labels == 0)
        n_true = true_omit.sum()

        total_surfaced += surfaced.sum()
        total_source += len(imp_labels)

        if n_true == 0:
            losses.append(0.0)
        else:
            n_missed = (true_omit & ~surfaced).sum()
            losses.append(float(n_missed / n_true))

    n = len(losses)
    adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
    workload = total_surfaced / total_source * 100 if total_source > 0 else 0

    # Mean surfaced count per doc (used for workload-minimizing selection in 2D Joint)
    mean_surfaced = total_surfaced / n if n > 0 else 0

    return adj_risk, workload, mean_surfaced


def find_methods(cal_docs):
    """Find operating points for 1D-Imp, Union Bound, and 2D Joint."""
    tau_vals = np.arange(0.0, 1.0 + GRID, GRID)
    gamma_vals = np.arange(0.0, 1.0 + GRID, GRID)

    # 1D-Imp: largest tau on gamma=0 axis that still satisfies guarantee
    best_1d_tau = 0.0
    for tau in reversed(tau_vals):
        risk, _, _ = compute_risk_and_workload(cal_docs, tau, 0.0)
        if risk <= ALPHA:
            best_1d_tau = tau
            break
    _, wl_1d, _ = compute_risk_and_workload(cal_docs, best_1d_tau, 0.0)

    # Union Bound: each dimension calibrated at alpha/2
    ub_tau = 0.0
    for tau in reversed(tau_vals):
        risk, _, _ = compute_risk_and_workload(cal_docs, tau, 0.0)
        if risk <= ALPHA / 2:
            ub_tau = tau
            break
    ub_gamma = 0.0
    for gamma in reversed(gamma_vals):
        risk, _, _ = compute_risk_and_workload(cal_docs, 0.0, gamma)
        if risk <= ALPHA / 2:
            ub_gamma = gamma
            break
    _, wl_ub, _ = compute_risk_and_workload(cal_docs, ub_tau, ub_gamma)

    # 2D Joint: search full grid, minimize mean surfaced per doc
    # Tie-break: favor larger thresholds (same as calibration.py line 568)
    feasible_pairs = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            risk, wl, ms = compute_risk_and_workload(cal_docs, tau, gamma)
            if risk <= ALPHA:
                feasible_pairs.append((ms, -tau, -gamma, tau, gamma, wl))
    feasible_pairs.sort()  # min surfaced, then max tau, max gamma
    _, _, _, best_2d_tau, best_2d_gamma, best_2d_wl = feasible_pairs[0]

    # LTT-FST: walk the pre-specified tau+gamma-descending ordering and stop
    # at the first feasible cell (Angelopoulos et al. 2021, Fixed-Sequence).
    cells = [(float(t), float(g)) for t in tau_vals for g in gamma_vals]
    cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    fst_tau, fst_gamma, fst_wl = 0.0, 0.0, 0.0
    for tau, gamma in cells:
        risk, wl, _ = compute_risk_and_workload(cal_docs, tau, gamma)
        if risk <= ALPHA:
            fst_tau, fst_gamma, fst_wl = tau, gamma, wl
            break

    return {
        "1d_imp": (best_1d_tau, 0.0, wl_1d),
        "union_bnd": (ub_tau, ub_gamma, wl_ub),
        "2d_joint": (best_2d_tau, best_2d_gamma, best_2d_wl),
        "fst": (fst_tau, fst_gamma, fst_wl),
    }


def plot_feasible_set(cal_docs, methods):
    """Create the feasible set figure."""
    tau_vals = np.arange(0.0, 1.0 + GRID, GRID)
    gamma_vals = np.arange(0.0, 1.0 + GRID, GRID)

    # Compute feasibility and workload grids
    feasible = np.zeros((len(gamma_vals), len(tau_vals)))
    workload = np.zeros((len(gamma_vals), len(tau_vals)))

    for i, gamma in enumerate(gamma_vals):
        for j, tau in enumerate(tau_vals):
            risk, wl, _ = compute_risk_and_workload(cal_docs, tau, gamma)
            feasible[i, j] = 1 if risk <= ALPHA else 0
            workload[i, j] = wl

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))

    # Gray background for infeasible region
    ax.set_facecolor("#e0e0e0")

    # Feasible region (green fill on top of gray)
    ax.contourf(
        tau_vals, gamma_vals, feasible,
        levels=[-0.5, 0.5, 1.5],
        colors=["#e0e0e0", "#c8e6c9"],
    )

    # Feasible boundary (dashed dark green, thick)
    ax.contour(
        tau_vals, gamma_vals, feasible,
        levels=[0.5],
        colors=["#1b5e20"],
        linestyles="dashed",
        linewidths=3,
    )

    # Method operating points (larger markers)
    t1, g1, w1 = methods["1d_imp"]
    t2, g2, w2 = methods["union_bnd"]
    t4, g4, w4 = methods["fst"]

    ax.plot(t1, g1, "o", color="#1565C0", markersize=9, zorder=5,
            label=f"1D-Imp ({w1:.0f}% flagged)")
    ax.plot(t2, g2, "o", color="#FF8F00", markersize=9, zorder=5,
            label=f"Union Bnd ({w2:.0f}% flagged)")
    ax.plot(t4, g4, "*", color="#6A1B9A", markersize=18, zorder=6,
            markeredgecolor="white", markeredgewidth=0.8,
            label=f"LTT-FST ({w4:.0f}% flagged)")

    # Blue dotted line along gamma=0 (1D search restriction)
    ax.axhline(y=0.0, color="#1565C0", linestyle="dotted", linewidth=1.5, alpha=0.7)

    # Labels
    ax.text(0.78, 0.88, "Infeasible", fontsize=13, fontstyle="italic",
            color="#777", ha="center", transform=ax.transAxes)
    ax.text(0.18, 0.22, r"$\mathcal{F}(\alpha)$", fontsize=20,
            fontstyle="italic", color="#2e7d32", transform=ax.transAxes)
    ax.set_xlabel(r"$\tau$", fontsize=14)
    ax.text(0.72, 0.05, r"1D search restricted to $\gamma = 0$",
            fontsize=8, color="#D32F2F", fontstyle="italic",
            ha="center", transform=ax.transAxes)
    ax.set_ylabel(r"$\gamma$", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95,
              edgecolor="#ccc", fancybox=True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Loading ACI-Bench calibration data...")
    cal_docs = load_cal_docs()
    print(f"  {len(cal_docs)} calibration documents")

    print("Finding method operating points...")
    methods = find_methods(cal_docs)
    for name, (tau, gamma, wl) in methods.items():
        print(f"  {name}: tau={tau:.2f}, gamma={gamma:.2f}, workload={wl:.1f}%")

    print("Generating figure...")
    fig = plot_feasible_set(cal_docs, methods)

    out_path = OUTPUT_DIR / "figure_2_feasible_set.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)
