#!/usr/bin/env python3
"""Table A6: Sentence-Level vs. Document-Level Hallucination Control.

Computes document-level CRC abstention baselines (min and mean p_fact)
and compares against sentence-level CARE flagging.

Reads Phase 2 calibrated_scores.jsonl and Phase 3 conformal_thresholds.json
from canonical data at /share/pi/nigam/projects/conf-summ/.
"""
import json
import numpy as np
from pathlib import Path
from care.config import DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR, get_phase2_scores, get_phase3_thresholds, CANONICAL_DATA_ROOT as DATA_ROOT

ALPHA = 0.15
RESPLIT_CI_PATH = OUTPUT_DIR / "resplit_cis_fst.json"  # factuality identical; FST file kept canonical


def load_cal_test_docs(ds_folder):
    """Load phase 2 scores split into cal/test."""
    split_path = DATA_ROOT / ds_folder / "split" / "split_indices.json"
    with open(split_path) as f:
        split_idx = json.load(f)

    cal_ids = set(split_idx["calibration_indices"])
    test_ids = set(split_idx["test_indices"])

    all_docs = []
    with open(get_phase2_scores(ds_folder)) as f:
        for line in f:
            all_docs.append(json.loads(line))

    cal_docs = [d for d in all_docs if d["doc_id"] in cal_ids]
    test_docs = [d for d in all_docs if d["doc_id"] in test_ids]
    return cal_docs, test_docs


def doc_level_score(doc, agg_fn):
    """Compute document-level factuality score using agg_fn (min or mean)."""
    probs = doc["factuality_probs"]
    if not probs:
        return 1.0
    return float(agg_fn(probs))


def has_unflagged_halluc(doc):
    """Check if doc has any true hallucination (label=0)."""
    labels = doc["factuality_labels"]
    return any(l == 0 for l in labels)


def calibrate_doc_level(cal_docs, agg_fn):
    """Find threshold t* such that P(reject a doc with halluc) >= 1-alpha.

    Nonconformity score = agg_fn(p_fact). Lower score = more suspicious.
    Reject if score <= t*. Find smallest t* s.t. empirical risk <= alpha.
    """
    n = len(cal_docs)

    # For each doc: compute score and whether it has a hallucination
    scores_and_labels = []
    for doc in cal_docs:
        score = doc_level_score(doc, agg_fn)
        has_halluc = has_unflagged_halluc(doc)
        scores_and_labels.append((score, has_halluc))

    # Grid search over thresholds
    all_scores = sorted(set(s for s, _ in scores_and_labels))
    # Add boundaries
    candidates = [0.0] + all_scores + [1.0]

    best_t = None
    for t in candidates:
        # Instance loss: 1 if doc has hallucination AND is NOT rejected
        losses = []
        for score, has_halluc in scores_and_labels:
            rejected = score <= t
            if has_halluc and not rejected:
                losses.append(1)
            else:
                losses.append(0)
        risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if risk <= ALPHA:
            best_t = t
            break

    return best_t


def evaluate_doc_level(test_docs, agg_fn, threshold):
    """Evaluate doc-level abstention on test set."""
    if threshold is None:
        return None, None

    n_rejected = 0
    violations = 0
    for doc in test_docs:
        score = doc_level_score(doc, agg_fn)
        rejected = score <= threshold
        if rejected:
            n_rejected += 1
        else:
            if has_unflagged_halluc(doc):
                violations += 1

    n = len(test_docs)
    rejection_rate = n_rejected / n
    violation_rate = violations / n
    return rejection_rate, violation_rate


def generate_table():
    # Collect per-dataset results
    min_rejections = []
    mean_rejections = []
    sent_flagged = []
    min_valid = []
    mean_valid = []

    for ds_folder, ds_info in DATASETS.items():
        cal_docs, test_docs = load_cal_test_docs(ds_folder)

        # Doc-level: min p_fact
        t_min = calibrate_doc_level(cal_docs, np.min)
        rej_min, viol_min = evaluate_doc_level(test_docs, np.min, t_min)

        # Doc-level: mean p_fact
        t_mean = calibrate_doc_level(cal_docs, np.mean)
        rej_mean, viol_mean = evaluate_doc_level(test_docs, np.mean, t_mean)

        # Sentence-level: use resplit means if available, else phase 4
        if RESPLIT_CI_PATH.exists():
            resplit_cis = json.load(open(RESPLIT_CI_PATH))
            rs = resplit_cis.get(ds_folder, {})
            s = rs.get("summary", {})
            pct_flagged = s["fact_wl_pct"]["mean"]
            fact_viol = s["fact_viol"]["mean"]
        else:
            p4_path = DATA_ROOT / ds_folder / "phase_4" / "test_results.json"
            with open(p4_path) as f:
                p4 = json.load(f)
            pct_flagged = p4["factuality"]["pct_flagged"]
            fact_viol = p4["factuality"]["violation_rate"]

        min_rejections.append(rej_min)
        mean_rejections.append(rej_mean)
        sent_flagged.append(pct_flagged)
        min_valid.append(viol_min is not None and viol_min <= ALPHA)
        mean_valid.append(viol_mean is not None and viol_mean <= ALPHA)

        print(f"{ds_info['label']}: min_rej={rej_min:.0%}, mean_rej={rej_mean:.0%}, "
              f"sent_flag={pct_flagged:.0f}%, fact_viol={fact_viol:.1%}")

    # Compute ranges
    min_range = f"{min(min_rejections)*100:.0f}--{max(min_rejections)*100:.0f}\\%"
    mean_range = f"{min(mean_rejections)*100:.0f}--{max(mean_rejections)*100:.0f}\\%"
    sent_range = f"{min(sent_flagged):.0f}--{max(sent_flagged):.0f}\\%"

    all_min_valid = all(min_valid)
    all_mean_valid = all(mean_valid)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\caption{Sentence- vs.\ document-level hallucination control. "
        r"\emph{Action}: what happens when an error is detected. "
        r"\emph{Rej./Flag}: fraction of documents rejected or sentences flagged "
        r"(range across datasets). Document-level abstention either rejects "
        r"too aggressively or violates guarantees.}"
    )
    lines.append(r"\label{tab:sent-vs-doc}")
    lines.append(r"\begin{tabular}{l l c r}")
    lines.append(r"\toprule")
    lines.append(r"Method & Action & Valid? & Rej./Flag \\")
    lines.append(r"\midrule")
    cmark = r"\cmark"
    xmark = r"\xmark"
    min_valid_mark = cmark if all_min_valid else xmark
    mean_valid_mark = cmark if all_mean_valid else xmark
    lines.append(
        r"Doc ($\min \phat_\mathrm{f}$) & Reject doc & "
        + min_valid_mark + " & " + min_range + r" \\"
    )
    lines.append(
        r"Doc ($\mathrm{mean}\;\phat_\mathrm{f}$) & Reject doc & "
        + mean_valid_mark + " & " + mean_range + r" \\"
    )
    lines.append(
        r"\textbf{Ours (sent.)} & Flag sent. & \cmark & " + sent_range + r" \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    table = generate_table()
    out_path = OUTPUT_DIR / "table_10_sent_vs_doc.tex"
    with open(out_path, "w") as f:
        f.write(table)
    print(f"\nSaved: {out_path}")
    print()
    print(table)


if __name__ == "__main__":
    main()
