#!/usr/bin/env python3
"""Per-document fractional omission loss histograms (5 panels, one per dataset).

Companion visualization to Table 15 (tail-risk diagnostics): the table reports
percentiles, this figure shows the full empirical distribution of per-document
fractional omission loss pooled across all 100 cal/test resplits per dataset.

Reuses the saved (tau, gamma) thresholds in tail_risk_cis.json so calibration
is not re-run; only the test-set per-doc loss vectors are recomputed from the
calibrated scores. The cal/test splits are reproduced using the same SEED and
permutation order as paper/scripts/compute_tail_risk.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from care.config import (
    CANONICAL_DATA_ROOT as DATA_ROOT,
    DATASETS,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)


SEED = 42  # must match compute_tail_risk.py
CAL_FRAC = 0.70
ALPHA = 0.15

COLORS = {
    "ACI-Bench":  "#1f77b4",
    "MIMIC-CXR":  "#ff7f0e",
    "MIMIC-BHC":  "#2ca02c",
    "SumPubMed":  "#d62728",
    "Priv-DS":    "#9467bd",
}

# Order panels left-to-right by document length / omission count, so the
# bimodal-vs-smooth tail story reads naturally.
PANEL_ORDER = ["MIMIC_III_CXR", "ACI_Bench", "MIMIC_IV_BHC", "OMOP", "SumPubMed"]


def omit_frac_loss_doc(doc, tau, gamma):
    imp_labels = np.asarray(doc["importance_labels"])
    cov_labels = np.asarray(doc["coverage_labels"])
    imp_probs = np.asarray(doc["importance_probs"])
    cov_probs = np.asarray(doc["coverage_probs"])
    if len(imp_labels) == 0:
        return 0.0
    true_omit = (imp_labels == 1) & (cov_labels == 0)
    n_true = int(true_omit.sum())
    if n_true == 0:
        return 0.0
    surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
    n_missed = int((true_omit & ~surfaced).sum())
    return float(n_missed / n_true)


def pooled_losses_for_dataset(dataset_key, per_resplit):
    path = DATA_ROOT / dataset_key / "phase_2" / "calibrated_scores.jsonl"
    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)

    rng = np.random.default_rng(SEED)
    pooled = []
    for rec in per_resplit:
        perm = rng.permutation(n)
        test_docs = [docs[j] for j in perm[n_cal:]]
        tau, gamma = rec["tau"], rec["gamma"]
        pooled.extend(omit_frac_loss_doc(d, tau, gamma) for d in test_docs)
    return np.asarray(pooled, dtype=float)


def plot(pooled_by_dataset):
    n_panels = len(pooled_by_dataset)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 2.8),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    bins = np.linspace(0.0, 1.0, 21)  # 0.05-wide bins on [0, 1]

    for ax, (dataset_key, losses) in zip(axes, pooled_by_dataset.items()):
        label = DATASETS[dataset_key]["label"]
        color = COLORS.get(label, "gray")

        ax.hist(losses, bins=bins, density=True, color=color, alpha=0.75,
                edgecolor="white", linewidth=0.4, zorder=2)

        ax.axvline(ALPHA, color="#d62728", linestyle="--", linewidth=1.4,
                   alpha=0.85, zorder=3)
        ax.axvline(losses.mean(), color="black", linestyle="-", linewidth=1.4,
                   alpha=0.9, zorder=4)

        ax.set_title(f"{label}\nmean={losses.mean():.3f}  P95={np.percentile(losses, 95):.2f}",
                     fontsize=10, color=color, fontweight="bold")
        ax.set_xlabel(r"Per-doc fractional omission loss $L$")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, axis="y")

    axes[0].set_ylabel("Density")

    # Single legend for vertical-line semantics.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.4,
               label="empirical mean"),
        Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.4,
               label=r"$\alpha=0.15$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9)

    plt.tight_layout()
    return fig


def main():
    tail_risk_path = OUTPUT_DIR / "tail_risk_cis.json"
    saved = json.load(open(tail_risk_path))

    pooled_by_dataset = {}
    for dataset_key in PANEL_ORDER:
        if dataset_key not in saved:
            continue
        per_resplit = saved[dataset_key]["per_resplit"]
        print(f"  rebuilding pooled losses for {DATASETS[dataset_key]['label']}...")
        pooled = pooled_losses_for_dataset(dataset_key, per_resplit)

        # Sanity check: pooled mean / percentiles should match the saved
        # pooled_summary; the splits are reproduced with the same SEED.
        ref = saved[dataset_key]["pooled_summary"]["omit_fractional"]
        for key, getter in [("mean", lambda x: x.mean()),
                            ("P50", lambda x: np.percentile(x, 50)),
                            ("P95", lambda x: np.percentile(x, 95))]:
            got = getter(pooled)
            want = ref[key]
            if abs(got - want) > 1e-6:
                print(f"    WARN {dataset_key} {key}: got={got:.6f} want={want:.6f}")
        pooled_by_dataset[dataset_key] = pooled

    fig = plot(pooled_by_dataset)
    out = OUTPUT_DIR / "figure_tail_risk_histogram.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
