#!/usr/bin/env python3
"""Table: Scorer-agnostic risk control.

Shows CARE maintains α-validity across three external scorer families
(DeBERTa NLI, BERT cosine, AlignScore) on all 5 datasets, while Max-F1
thresholding (the standard empirical operating point) catastrophically
violates.

CARE (CRC) columns are the LTT-FST controller, read from
fst_external_scorer_results.json (compute_fst_external_scorers.py); the
Max-F1 comparator is method-agnostic and read from external_scorer_results.json
(compute_external_scorer_results.py). Both runs share seeds/resplits.
"""
import json
import math
import numpy as np
from care.config import DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR

TWOD_PATH = OUTPUT_DIR / "external_scorer_results.json"
FST_PATH = OUTPUT_DIR / "fst_external_scorer_results.json"
OUT_PATH = OUTPUT_DIR / "table_external_scorers.tex"
ALPHA = 0.15
N_RESPLITS = 100
Z_95 = 1.96

SCORER_ORDER = ["nli_embed", "embed_only", "alignscore"]
SCORER_LABELS = {
    "nli_embed": "NLI (DeBERTa)",
    "embed_only": "Embed (BERT)",
    "alignscore": "AlignScore",
}


def ci_halfwidth(std_pct, n=N_RESPLITS, z=Z_95):
    return z * std_pct / math.sqrt(n)


def fmt(mean_pct, std_pct, alpha_pct=ALPHA * 100):
    hw = ci_halfwidth(std_pct)
    lo = mean_pct - hw
    mean_str = f"{mean_pct:.1f}"
    ci_str = rf"{{\scriptsize $\pm${hw:.1f}}}"
    if round(lo, 1) > alpha_pct:
        return rf"\textbf{{{mean_str}}}$^\dagger${ci_str}"
    return f"{mean_str}{ci_str}"


def load_results():
    """Return key -> {crc_fact, crc_omit, maxf1_fact, maxf1_omit} as (mean%, std%).

    CARE (CRC) factuality and omission are the LTT-FST controller (per-resplit
    means from fst_external_scorer_results.json). Max-F1 is method-agnostic
    and taken from external_scorer_results.json.
    """
    if not TWOD_PATH.exists() or not FST_PATH.exists():
        raise SystemExit(f"ERROR: need {TWOD_PATH.name} and {FST_PATH.name}.")
    twod = json.load(open(TWOD_PATH))
    fst = json.load(open(FST_PATH))
    out = {}
    for key, fentry in fst.items():
        if key not in twod:
            continue
        fpr = fentry["per_resplit"]
        f_fact = np.array([r["fact_viol"] for r in fpr]) * 100
        f_omit = np.array([r["omit_viol"] for r in fpr]) * 100
        s2 = twod[key]["summary"]
        out[key] = {
            "crc_fact": (float(f_fact.mean()), float(f_fact.std())),
            "crc_omit": (float(f_omit.mean()), float(f_omit.std())),
            "maxf1_fact": (s2["maxf1_fact_viol"]["mean"] * 100,
                           s2["maxf1_fact_viol"]["std"] * 100),
            "maxf1_omit": (s2["maxf1_omit_viol"]["mean"] * 100,
                           s2["maxf1_omit_viol"]["std"] * 100),
        }
    return out


def generate_latex(data):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{Scorer-agnostic risk control ($\alpha = 0.15$, fractional "
        r"omission loss, 100 random 70/30 cal/test resplits). The LLM-judge "
        r"Phase 2 factuality and coverage signals are replaced by the three "
        r"canonical external scorer families from the faithfulness-evaluation "
        r"literature: NLI (\textsc{DeBERTa-v3} cross-encoder, \textsc{SummaC}-"
        r"style), embedding cosine (\textsc{BERT-base}, \textsc{BERTScore}-"
        r"style), and \textsc{AlignScore}. Importance is held fixed at the "
        r"LLM-judge signal. Each cell shows mean V\% $\pm$ half-width of a "
        r"95\% normal-approximation CI on the resplit mean "
        r"($1.96 \cdot \text{SD}/\sqrt{100}$). "
        r"$\dagger$ marks cells whose CI lies entirely above $\alpha$ at "
        r"display precision. "
        r"\textsc{Care} preserves $\alpha$-validity marginally "
        r"(mean $V=13.3\%$ across all 30 cells), consistent with CRC's "
        r"finite-sample guarantee being marginal over "
        r"(calibration, test) draws \citep{angelopoulos-crc-2024}. "
        r"Per-cell realizations fluctuate around $\alpha$ as predicted "
        r"by the Beta distribution of per-calibration coverage; the "
        r"largest realization at $V=15.7\%$ (\textsc{AlignScore} "
        r"$\times$ Omit on ACI-Bench) lies well within one standard "
        r"deviation ($\sqrt{\alpha(1-\alpha)/n_{\mathrm{cal}}} "
        r"\approx 3.9$\,pp at $n_{\mathrm{cal}}=86$) of $\alpha$. "
        r"Max-F1 thresholding (the standard empirical operating point, "
        r"oracle F1-optimal on cal using labels, ignores $\alpha$) "
        r"significantly violates on 22/30 cells, with factuality violation "
        r"up to 71\% on Priv-DS.}"
    )
    lines.append(r"\label{tab:external-scorers}")
    lines.append(r"\begin{tabular}{ll rr rr}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{Factuality V\%} & \multicolumn{2}{c}{Omission V\%} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(r"Dataset & Scorer & \textsc{Care} & Max-F1 & \textsc{Care} & Max-F1 \\")
    lines.append(r"\midrule")

    for i, (ds_key, meta) in enumerate(DATASETS.items()):
        if i > 0:
            lines.append(r"\midrule")
        for j, scorer in enumerate(SCORER_ORDER):
            key = f"{ds_key}__{scorer}"
            if key not in data:
                continue
            d = data[key]
            cf, cf_s = d["crc_fact"]
            mf, mf_s = d["maxf1_fact"]
            co, co_s = d["crc_omit"]
            mo, mo_s = d["maxf1_omit"]
            ds_cell = meta["label"] if j == 0 else ""
            lines.append(
                f"{ds_cell} & {SCORER_LABELS[scorer]} & "
                f"{fmt(cf, cf_s)} & {fmt(mf, mf_s)} & "
                f"{fmt(co, co_s)} & {fmt(mo, mo_s)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_text(data):
    out = [
        f"Scorer-agnostic CRC results (alpha={ALPHA}, 100 resplits, mean±1.96·SD/√100)",
        "=" * 100,
        f"{'Dataset':<10} {'Scorer':<14} {'CARE F (CI)':>20} {'CARE O (CI)':>20} {'MaxF1 F (CI)':>20} {'MaxF1 O (CI)':>20}",
        "-" * 100,
    ]
    a = ALPHA * 100
    def cell(m, sd):
        hw = ci_halfwidth(sd)
        flag = "!" if (m - hw) > a else (" " if m <= a else "~")
        return f"{m:5.1f}±{hw:4.1f}{flag}"
    for ds_key, meta in DATASETS.items():
        for scorer in SCORER_ORDER:
            key = f"{ds_key}__{scorer}"
            if key not in data:
                continue
            d = data[key]
            cf = cell(*d["crc_fact"])
            co = cell(*d["crc_omit"])
            mf = cell(*d["maxf1_fact"])
            mo = cell(*d["maxf1_omit"])
            out.append(
                f"{meta['label']:<10} {SCORER_LABELS[scorer]:<14} "
                f"{cf:>20} {co:>20} {mf:>20} {mo:>20}"
            )
    out.append("")
    out.append("Legend: '!' = CI lower bound > α (significant violation)")
    out.append("        '~' = mean > α but CI contains α (within Monte Carlo noise)")
    return "\n".join(out)


if __name__ == "__main__":
    data = load_results()
    print(generate_text(data))
    print()
    OUT_PATH.write_text(generate_latex(data))
    print(f"Saved: {OUT_PATH}")
