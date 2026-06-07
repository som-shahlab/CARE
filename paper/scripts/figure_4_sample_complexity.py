#!/usr/bin/env python3
"""Figure A1: Sample Complexity Ablation.

Shows how CRC guarantees and workload change as a function of
calibration set size. Demonstrates finite-sample behavior.

Reads pre-computed results from the sample complexity ablation.
If not found, instructs user how to generate them.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR


# Where ablation results are stored
from care.config import PROJECT_ROOT
ABLATION_DIR = PROJECT_ROOT / "data" / "experiment_v2_outputs"

COLORS = {
    "ACI-Bench":  "#1f77b4",
    "MIMIC-CXR":  "#ff7f0e",
    "MIMIC-BHC":  "#2ca02c",
    "SumPubMed":  "#d62728",
    "Priv-DS":    "#9467bd",
}

ALPHA = 0.15


def load_results():
    """Return (fact_results, omit_results), each keyed by dataset label.

    Omission panels use the LTT-FST sample-complexity run (paper/outputs/
    fst_sample_complexity.json). The hallucination panel is the unchanged
    1D scalar CRC controller, read from the original ablation.
    """
    fst = json.load(open(OUTPUT_DIR / "fst_sample_complexity.json"))
    omit_results = {fst[k]["dataset"]: fst[k] for k in fst}

    fact_results = {}
    for dataset, meta in DATASETS.items():
        candidates = [
            DATA_ROOT / dataset / "ablations" / "sample" / "ablation_sample_complexity" / "sample_complexity_results.json",
            ABLATION_DIR / dataset / "ablation_sample_complexity" / "sample_complexity_results.json",
            DATA_ROOT / dataset / "ablation_sample_complexity" / "sample_complexity_results.json",
        ]
        for path in candidates:
            if path.exists():
                fact_results[meta["label"]] = json.load(open(path))
                break
    return fact_results, omit_results


def plot_sample_complexity(fact_results, omit_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        ("fact_violation_rate", "Halluc. Violation Rate (%)", True),
        ("omit_frac_violation_rate", "Omit. Violation Rate (%)", True),
        ("omit_workload_pct", "Omission Flagged (%)", False),
    ]

    for col, (field, ylabel, show_alpha) in enumerate(metrics):
        ax = axes[col]

        all_results = fact_results if col == 0 else omit_results
        for ds_label, data in all_results.items():
            sizes = []
            means = []
            lo_err = []
            hi_err = []

            for entry in data.get("results", []):
                n = entry.get("n_sub", entry.get("n_cal"))
                if n is None:
                    continue

                # Map field names to actual keys in results
                field_map = {
                    "fact_violation_rate": ("fact_violation_mean", "fact_violation_all"),
                    "omit_frac_violation_rate": ("omit_violation_mean", "omit_violation_all"),
                    "omit_workload_pct": ("omit_workload_mean", None),
                }

                if field not in field_map:
                    continue

                mean_key, all_key = field_map[field]
                m_val = entry.get(mean_key)
                if m_val is None:
                    continue

                sizes.append(n)
                m = m_val * 100 if m_val <= 1 else m_val
                means.append(m)

                # Use all values for error bars if available
                if all_key and all_key in entry:
                    vals = np.array(entry[all_key])
                    vals_pct = vals * 100 if np.max(vals) <= 1 else vals
                    lo_err.append(max(0, m - np.percentile(vals_pct, 2.5)))
                    hi_err.append(max(0, np.percentile(vals_pct, 97.5) - m))
                else:
                    std = entry.get(mean_key.replace("_mean", "_std"), 0)
                    std_pct = std * 100 if std <= 1 else std
                    lo_err.append(std_pct)
                    hi_err.append(std_pct)

            if sizes:
                ax.errorbar(sizes, means, yerr=[lo_err, hi_err],
                           fmt="o-", label=ds_label, color=COLORS.get(ds_label, "gray"),
                           markersize=4, linewidth=1.5, capsize=3)

        if show_alpha:
            ax.axhline(y=ALPHA * 100, color="red", linestyle="--", alpha=0.5, label=f"α={ALPHA}")

        ax.set_xscale("log")
        ax.set_xticks([15, 25, 50, 100, 200, 300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Calibration Set Size")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Single shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    return fig


if __name__ == "__main__":
    fact_results, omit_results = load_results()

    if not omit_results:
        print("No FST sample complexity results found.")
        print("Run: python3 paper/scripts/compute_fst_sample_complexity.py")
        sys.exit(1)

    print(f"Omission (FST): {list(omit_results.keys())}")
    print(f"Hallucination:  {list(fact_results.keys())}")

    fig = plot_sample_complexity(fact_results, omit_results)

    png_path = OUTPUT_DIR / "figure_4_sample_complexity.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {png_path}")
    plt.close(fig)
