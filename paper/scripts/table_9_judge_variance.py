#!/usr/bin/env python3
"""Table A2: Judge Self-Consistency.

Computes per-sentence variance, unanimity rate, and Fleiss' kappa across
m=5 vote replicates from Phase 2 calibrated_scores.jsonl.

For ACI-Bench: reads from phase_2/with_raw_scores.jsonl (primary data).
For other datasets: reads from phase_2_redo/calibrated_scores.jsonl (replicates).
"""
import json
import numpy as np
from pathlib import Path
from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR


def fleiss_kappa(ratings_matrix):
    """Compute Fleiss' kappa for m raters, k categories.

    ratings_matrix: (n_subjects, k_categories) counts.
    """
    N, k = ratings_matrix.shape
    m = ratings_matrix.sum(axis=1)[0]  # raters per subject

    # Proportion in each category
    p_j = ratings_matrix.sum(axis=0) / (N * m)

    # Per-subject agreement
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - m) / (m * (m - 1))
    P_bar = np.mean(P_i)

    # Expected agreement
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def compute_stats(all_raw_scores):
    """Compute variance, unanimity, and kappa from raw vote scores.

    all_raw_scores: list of lists, each inner list is m=5 votes in {0.0, 0.5, 1.0}.
    """
    n = len(all_raw_scores)
    if n == 0:
        return 0, np.nan, np.nan, np.nan

    variances = []
    unanimous = 0

    # For Fleiss' kappa: convert to category counts (3 categories: 0.0, 0.5, 1.0)
    cat_map = {0.0: 0, 0.5: 1, 1.0: 2}
    ratings_matrix = np.zeros((n, 3), dtype=int)

    for i, votes in enumerate(all_raw_scores):
        votes = np.array(votes)
        variances.append(np.var(votes))
        if np.all(votes == votes[0]):
            unanimous += 1
        for v in votes:
            ratings_matrix[i, cat_map.get(v, 1)] += 1

    mean_var = np.mean(variances)
    unan_pct = unanimous / n * 100
    kappa = fleiss_kappa(ratings_matrix)

    return n, mean_var, unan_pct, kappa


def load_raw_scores(ds_folder):
    """Load raw vote scores for a dataset, trying multiple paths."""
    # Try phase_2 first (ACI-Bench has raw scores in primary phase_2)
    candidates = [
        DATA_ROOT / ds_folder / "phase_2" / "calibrated_scores.jsonl",
        DATA_ROOT / ds_folder / "phase_2" / "with_raw_scores.jsonl",
        DATA_ROOT / ds_folder / "phase_2_redo" / "calibrated_scores.jsonl",
    ]

    for path in candidates:
        if not path.exists():
            continue
        fact_raw, imp_raw, cov_raw = [], [], []
        with open(path) as f:
            for line in f:
                doc = json.loads(line)
                if "factuality_raw_scores" not in doc:
                    break
                fact_raw.extend(doc["factuality_raw_scores"])
                imp_raw.extend(doc["importance_raw_scores"])
                cov_raw.extend(doc["coverage_raw_scores"])
        if fact_raw:
            return fact_raw, imp_raw, cov_raw, path
    return None, None, None, None


def generate_table():
    # Collect per-dataset stats
    rows = []
    for ds_folder, ds_info in DATASETS.items():
        fact_raw, imp_raw, cov_raw, path = load_raw_scores(ds_folder)
        if fact_raw is None:
            print(f"  SKIP {ds_info['label']}: no raw scores found")
            continue

        fact_n, fact_var, fact_unan, fact_kappa = compute_stats(fact_raw)
        imp_n, imp_var, imp_unan, imp_kappa = compute_stats(imp_raw)
        cov_n, cov_var, cov_unan, cov_kappa = compute_stats(cov_raw)

        print(f"  {ds_info['label']:12s}  fact: n={fact_n:>6,} var={fact_var:.3f} unan={fact_unan:.0f}% κ={fact_kappa:.2f}  "
              f"imp: n={imp_n:>6,} κ={imp_kappa:.2f}  cov: n={cov_n:>6,} κ={cov_kappa:.2f}  [{path.parent.name}]")

        rows.append({
            "label": ds_info["label"],
            "fact_n": fact_n, "fact_var": fact_var, "fact_unan": fact_unan, "fact_kappa": fact_kappa,
            "imp_n": imp_n, "imp_var": imp_var, "imp_unan": imp_unan, "imp_kappa": imp_kappa,
            "cov_n": cov_n, "cov_var": cov_var, "cov_unan": cov_unan, "cov_kappa": cov_kappa,
        })

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{LLM judge self-consistency across $m{=}5$ vote replicates. "
        r"We report Fleiss' $\kappa$ and unanimity rate (\% of sentences where all 5 votes agree) "
        r"for each annotation task.}"
    )
    lines.append(r"\label{tab:judge-variance}")
    lines.append(r"\begin{tabular}{l cc cc cc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{Factuality} & \multicolumn{2}{c}{Importance} & \multicolumn{2}{c}{Coverage} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Dataset & $\kappa$ & Unan.\% & $\kappa$ & Unan.\% & $\kappa$ & Unan.\% \\")
    lines.append(r"\midrule")

    for r in rows:
        lines.append(
            f"  {r['label']} & {r['fact_kappa']:.2f} & {r['fact_unan']:.0f} "
            f"& {r['imp_kappa']:.2f} & {r['imp_unan']:.0f} "
            f"& {r['cov_kappa']:.2f} & {r['cov_unan']:.0f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    table = generate_table()
    out_path = OUTPUT_DIR / "table_9_judge_variance.tex"
    with open(out_path, "w") as f:
        f.write(table)
    print(f"\nSaved: {out_path}")
    print()
    print(table)


if __name__ == "__main__":
    main()
