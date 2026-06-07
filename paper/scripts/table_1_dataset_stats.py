#!/usr/bin/env python3
"""Table 1: Dataset Statistics.

Generates a LaTeX table with per-dataset statistics:
- Total documents, calibration/test split
- Avg source sentences, avg summary sentences
- Avg source length (tokens)
- Error base rates (factuality error %, omission %)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR, get_phase1_labels, get_split_file


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_stats():
    rows = []
    for dataset, meta in DATASETS.items():
        # Load Phase 1 oracle labels for sentence counts and error rates
        labels = load_jsonl(get_phase1_labels(dataset))

        # Get cal/test split sizes — try split files first, fall back to split_indices.json
        cal_path = get_split_file(dataset, "calibration")
        test_path = get_split_file(dataset, "test")
        if cal_path.exists() and test_path.exists():
            n_cal = len(load_jsonl(cal_path))
            n_test = len(load_jsonl(test_path))
        else:
            idx_path = DATA_ROOT / dataset / "split" / "split_indices.json"
            if idx_path.exists():
                idx = json.load(open(idx_path))
                n_cal = len(idx.get("calibration_indices", idx.get("calibration", [])))
                n_test = len(idx.get("test_indices", idx.get("test", [])))
            else:
                n_cal = int(len(labels) * 0.7)
                n_test = len(labels) - n_cal
        n_total = n_cal + n_test

        src_sents = [len(d["source_sentences"]) for d in labels]
        sum_sents = [len(d["generated_sentences"]) for d in labels]

        # Error base rates from oracle labels
        fact_error_rates = []
        omission_rates = []
        for d in labels:
            fact_labels = d.get("factuality_labels", [])
            if fact_labels:
                fact_error_rates.append(1 - np.mean(fact_labels))  # % unfactual

            imp_labels = d.get("importance_labels", [])
            cov_labels = d.get("coverage_labels", [])
            if imp_labels and cov_labels:
                omissions = sum(
                    1 for imp, cov in zip(imp_labels, cov_labels)
                    if imp == 1 and cov == 0
                )
                n_important = sum(imp_labels)
                if n_important > 0:
                    omission_rates.append(omissions / n_important)

        rows.append({
            "label": meta["label"],
            "n_total": n_total,
            "n_cal": n_cal,
            "n_test": n_test,
            "avg_src_sents": np.mean(src_sents),
            "avg_sum_sents": np.mean(sum_sents),
            "avg_fact_error": np.mean(fact_error_rates) * 100 if fact_error_rates else 0,
            "avg_omission": np.mean(omission_rates) * 100 if omission_rates else 0,
        })

    return rows


def generate_latex(rows):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Dataset statistics. Error rates are oracle-labeled averages across all documents.}")
    lines.append(r"\label{tab:dataset-stats}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & $N$ & Cal / Test & Avg Src & Avg Sum & Fact Err\% & Omit\% \\")
    lines.append(r"\midrule")

    for r in rows:
        lines.append(
            f"{r['label']} & {r['n_total']} & {r['n_cal']}/{r['n_test']} & "
            f"{r['avg_src_sents']:.1f} & {r['avg_sum_sents']:.1f} & "
            f"{r['avg_fact_error']:.1f} & {r['avg_omission']:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_text(rows):
    lines = ["Dataset Statistics", "=" * 60]
    for r in rows:
        lines.append(
            f"{r['label']:12s}  N={r['n_total']:>4d} (cal={r['n_cal']}, test={r['n_test']})  "
            f"src={r['avg_src_sents']:.1f}  sum={r['avg_sum_sents']:.1f}  "
            f"fact_err={r['avg_fact_error']:.1f}%  omit={r['avg_omission']:.1f}%"
        )
    return "\n".join(lines)


def render_table_png(rows, out_path):
    """Render the table as a clean PNG image using matplotlib."""
    headers = ["Dataset", "N", "Cal / Test", "Avg Src", "Avg Sum", "Fact Err%", "Omit%"]
    cell_data = []
    for r in rows:
        cell_data.append([
            r["label"],
            str(r["n_total"]),
            f"{r['n_cal']}/{r['n_test']}",
            f"{r['avg_src_sents']:.1f}",
            f"{r['avg_sum_sents']:.1f}",
            f"{r['avg_fact_error']:.1f}",
            f"{r['avg_omission']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4 * len(cell_data)))
    ax.axis("off")
    ax.set_title("Table 1: Dataset Statistics", fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(cell_data)):
        for j in range(len(headers)):
            cell = table[i + 1, j]
            if i % 2 == 0:
                cell.set_facecolor("#D9E2F3")
            else:
                cell.set_facecolor("#FFFFFF")

    fig.savefig(out_path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    rows = compute_stats()

    # Print to console
    print(generate_text(rows))
    print()

    # Save LaTeX
    latex = generate_latex(rows)
    out_path = OUTPUT_DIR / "table_1_dataset_stats.tex"
    out_path.write_text(latex)
    print(f"Saved: {out_path}")

