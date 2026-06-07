#!/usr/bin/env python3
"""Render Table 13 (cost analysis) from `table_13_cost_analysis.csv`.

Produces `paper/outputs/table_13_cost_analysis.tex` with three panels:
  (a) API costs (USD): calibration totals + per-doc deployment costs
      (LLM judge + GPT-5 oracle pipeline)
  (b) Token usage: calibration (M tokens) + deployment (K tokens per doc).
      Panel (b) needs per-phase token breakdowns (Fact/Imp/Cov), which the
      cost-analysis script computes but does not currently dump to the CSV.
      Until the CSV is extended, panel (b) values are pulled from the
      hardcoded dict `PANEL_B_TOKENS` below — regenerate by extending
      `table_13_cost_analysis.py` to include per-phase token counts.
  (c) Deployment-time judge cost (USD per doc) across scoring functions:
      GPT-5-mini, Gemini 2.5 Pro (where run), external scorers (\$0 API).

Read by paper.tex via `\input{../outputs/table_13_cost_analysis.tex}`.
"""
import csv
from collections import defaultdict
from pathlib import Path

from care.config import PAPER_OUTPUT_DIR as OUTPUT_DIR

CSV_PATH = OUTPUT_DIR / "table_13_cost_analysis.csv"
OUT_PATH = OUTPUT_DIR / "table_13_cost_analysis.tex"

# Display order + short labels (match the original Table 13 row layout).
DATASETS = [
    ("ACI_Bench",     "ACI$^\\ddagger$"),
    ("MIMIC_IV_BHC",  "BHC"),
    ("MIMIC_III_CXR", "CXR"),
    ("OMOP",          "Priv-DS"),
    ("SumPubMed",     "PubMed"),
]

# Panel (b) token usage is reported as: judge_tokens / m_replicates + oracle_tokens.
# The cost-analysis script computes per-phase (Fact/Imp/Cov) input+output
# token counts but does not write them to the CSV. Until that's plumbed
# through, we keep these as constants. Regenerate by adding per-phase token
# columns to the CSV writer.
PANEL_B_TOKENS = {
    # dataset_key: (cal_summ, cal_fact, cal_imp, cal_cov, cal_total,
    #               dep_summ, dep_fact, dep_imp, dep_cov, dep_total)
    "ACI_Bench":     (0.21, 0.96,  1.19,   0.92,  3.28,   2.1,  4.8,  8.7,   4.6,  20.2),
    "MIMIC_IV_BHC":  (0.37, 1.14,  4.38,   1.72,  7.61,   3.7,  5.7,  31.9,  8.6,  49.9),
    "MIMIC_III_CXR": (0.03, 0.10,  0.07,   0.08,  0.28,   0.3,  0.5,  0.4,   0.4,  1.6),
    "OMOP":          (1.82, 10.24, 64.89,  7.48,  84.43,  18.2, 51.2, 557.0, 37.4, 663.8),
    "SumPubMed":     (0.64, 1.40,  10.10,  2.82,  14.96,  6.4,  7.0,  87.9,  14.1, 115.4),
}


# ---------- CSV loading ----------

def load_csv(path):
    """Group rows by (dataset, section, model)."""
    by_key = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["section"], row["model"])
            by_key[key] = row
    return by_key


def cost(rows, dataset, section, model, field):
    r = rows.get((dataset, section, model))
    if r is None or r.get(field) in (None, ""):
        return None
    return float(r[field])


# ---------- Panel A: API costs (calibration + deployment) ----------

def panel_a_row(rows, ds_key, ds_label):
    """Calibration uses BOTH judge and oracle for fact/imp/cov; deployment
    uses only the judge. Summ runs once on the calibration set."""
    summ_total   = cost(rows, ds_key, "summarization", "llama3.3", "total_cost_usd") or 0.0
    judge_fact   = cost(rows, ds_key, "judge",  "gpt-5o-mini", "fact_cost_usd") or 0.0
    judge_imp    = cost(rows, ds_key, "judge",  "gpt-5o-mini", "imp_cost_usd")  or 0.0
    judge_cov    = cost(rows, ds_key, "judge",  "gpt-5o-mini", "cov_cost_usd")  or 0.0
    judge_total  = cost(rows, ds_key, "judge",  "gpt-5o-mini", "total_cost_usd") or 0.0
    oracle_fact  = cost(rows, ds_key, "oracle", "gpt-5",       "fact_cost_usd") or 0.0
    oracle_imp   = cost(rows, ds_key, "oracle", "gpt-5",       "imp_cost_usd")  or 0.0
    oracle_cov   = cost(rows, ds_key, "oracle", "gpt-5",       "cov_cost_usd")  or 0.0
    oracle_total = cost(rows, ds_key, "oracle", "gpt-5",       "total_cost_usd") or 0.0

    cal = (
        summ_total,
        judge_fact + oracle_fact,
        judge_imp  + oracle_imp,
        judge_cov  + oracle_cov,
        summ_total + judge_total + oracle_total,
    )
    n = 100  # all rows normalized to n=100 docs
    dep = (
        summ_total / n,
        judge_fact / n,
        judge_imp  / n,
        judge_cov  / n,
        (summ_total + judge_total) / n,
    )
    return ds_label, cal, dep


def fmt_cal(v):
    return f"{v:.2f}"


def fmt_dep(v):
    return f"<0.001" if v < 0.0005 else f"{v:.3f}"


def render_panel_a(rows):
    L = []
    L.append(r"\textbf{(a)} API costs (USD).")
    L.append(r"\vspace{0.3em}")
    L.append("")
    L.append(r"\begin{tabular}{l rrrrr rrrrr}")
    L.append(r"\toprule")
    L.append(r"& \multicolumn{5}{c}{\textbf{Calibration (100 docs)}} & \multicolumn{5}{c}{\textbf{Deployment (per doc)}} \\")
    L.append(r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}")
    L.append(r"\textbf{Dataset} & Summ. & Fact & Imp & Cov & \textbf{Total} & Summ. & Fact & Imp & Cov & \textbf{Total} \\")
    L.append(r"\midrule")
    for ds_key, ds_label in DATASETS:
        label, cal, dep = panel_a_row(rows, ds_key, ds_label)
        cal_str = " & ".join(fmt_cal(v) for v in cal)
        dep_str = " & ".join(fmt_dep(v) for v in dep)
        L.append(f"{label} & {cal_str} & {dep_str} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


# ---------- Panel B: Token usage (constants, see note above) ----------

def render_panel_b():
    L = []
    L.append(r"\par\vspace{1.5em}")
    L.append(r"\textbf{(b)} Token usage.")
    L.append(r"Calibration: millions of tokens for 100 documents. Deployment: thousands of tokens per document.")
    L.append(r"\vspace{0.3em}")
    L.append("")
    L.append(r"\begin{tabular}{l rrrrr rrrrr}")
    L.append(r"\toprule")
    L.append(r"& \multicolumn{5}{c}{\textbf{Calibration (M tokens, 100 docs)}} & \multicolumn{5}{c}{\textbf{Deployment (K tokens, per doc)}} \\")
    L.append(r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}")
    L.append(r"\textbf{Dataset} & Summ. & Fact & Imp & Cov & \textbf{Total} & Summ. & Fact & Imp & Cov & \textbf{Total} \\")
    L.append(r"\midrule")
    for ds_key, ds_label in DATASETS:
        vals = PANEL_B_TOKENS[ds_key]
        cal_str = " & ".join(f"{v:.2f}" for v in vals[:5])
        dep_str = " & ".join(f"{v:.1f}" for v in vals[5:])
        L.append(f"{ds_label} & {cal_str} & {dep_str} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


# ---------- Panel C: Deployment-time judge cost across scoring functions ----------

def render_panel_c(rows):
    L = []
    L.append(r"\par\vspace{1.5em}")
    L.append(r"\textbf{(c)} Deployment-time judge cost (USD per document) across scoring functions.")
    L.append(
        r"External scorers (NLI cross-encoder, BERT embedding cosine, AlignScore) "
        r"run as local GPU inference and incur no per-token API cost. "
        r"Gemini 2.5 Pro was run on the two datasets where its outputs are "
        r"available (App.~\ref{app:scorer-agnostic}); ``--'' marks datasets where "
        r"the experiment was not run."
    )
    L.append(r"\vspace{0.3em}")
    L.append("")
    L.append(r"\begin{tabular}{l c c c}")
    L.append(r"\toprule")
    L.append(r"\textbf{Dataset} & GPT-5-mini & Gemini 2.5 Pro & External$^{\dagger}$ \\")
    L.append(r"\midrule")
    n = 100
    for ds_key, ds_label in DATASETS:
        gpt = cost(rows, ds_key, "judge", "gpt-5o-mini", "total_cost_usd")
        gem = cost(rows, ds_key, "judge_gemini", "gemini-2.5-pro", "total_cost_usd")
        gpt_str = fmt_dep(gpt / n) if gpt is not None else "--"
        gem_str = f"{gem / n:.3f}" if gem is not None else "--"
        L.append(f"{ds_label} & {gpt_str} & {gem_str} & 0.000 \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\par\vspace{0.2em}")
    L.append(r"\footnotesize $^{\dagger}$Local GPU inference for NLI (DeBERTa), Embed (BERT), and AlignScore; \$0 API cost.")
    return "\n".join(L)


# ---------- Top-level table wrapper ----------

def render_table(rows):
    L = []
    L.append(r"\begin{table*}[!t]")
    L.append(r"\centering")
    L.append(r"\small")
    L.append(
        r"\caption{Estimated API costs and token usage for the CARE pipeline. "
        r"Calibration costs are reported for 100 documents; deployment costs "
        r"are per document. \textbf{Fact}, \textbf{Imp}, \textbf{Cov}: factuality, "
        r"importance, and coverage scoring, respectively. $^\ddagger$ACI-Bench "
        r"contains fewer than 100 calibration documents; calibration totals are "
        r"mean-padded to 100 documents for comparability.}"
    )
    L.append(r"\label{tab:cost}")
    L.append(r"\vspace{0.5em}")
    L.append(render_panel_a(rows))
    L.append(render_panel_b())
    L.append(render_panel_c(rows))
    L.append(r"\end{table*}")
    return "\n".join(L)


# ---------- main ----------

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"ERROR: {CSV_PATH} not found. Run table_13_cost_analysis.py first.")
    rows = load_csv(CSV_PATH)
    tex = render_table(rows)
    OUT_PATH.write_text(tex + "\n")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
