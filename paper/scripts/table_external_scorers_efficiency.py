#!/usr/bin/env python3
"""Table: Scorer-agnostic CRC — workload + recall comparison.

Companion to table_external_scorers.py (which shows V%-only). This table
shows that while CRC validity is scorer-agnostic, *efficiency* is not:
LLM judges flag substantially fewer sentences for the same alpha, with
comparable or better recall — especially on radiology (MIMIC-CXR), where
external scorers' factuality/omission recall collapses.

Omission columns are the LTT-FST controller. LLM judges:
  - GPT-5-mini (primary, all 5 datasets)        — paper/outputs/resplit_cis_fst.json
  - Gemini 2.5 Pro (alt, ACI-Bench + MIMIC-BHC) — paper/outputs/gemini_resplit_cis.json
    (Gemini-judge omission shown at the 2D Joint operating point — the
    Gemini-judge phase-2 data was not regenerated for FST; factuality is
    unchanged and the BHC omission operating point is identical under FST.)
And three external scorers (NLI, Embed, AlignScore) for all 5 datasets:
  paper/outputs/fst_external_scorer_results.json
"""
import json
import numpy as np

from care.config import DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR

GPT5_PATH = OUTPUT_DIR / "resplit_cis_fst.json"
GEMINI_PATH = OUTPUT_DIR / "gemini_resplit_cis.json"
EXT_PATH = OUTPUT_DIR / "fst_external_scorer_results.json"
OUT_PATH = OUTPUT_DIR / "table_external_scorers_efficiency.tex"

ALPHA = 0.15
N_RESPLITS = 100

EXT_ORDER = ["nli_embed", "embed_only", "alignscore"]
EXT_LABELS = {
    "nli_embed": "NLI (DeBERTa)",
    "embed_only": "Embed (BERT)",
    "alignscore": "AlignScore",
}


# ---------- helpers ----------

def fmt_flagged(mean, pct):
    return f"{mean:.1f} ({pct:.1f}\\%)"


def fmt_rec(mean_pct):
    return f"{mean_pct:.1f}"


def load():
    if not GPT5_PATH.exists():
        raise SystemExit(f"ERROR: {GPT5_PATH} not found.")
    if not EXT_PATH.exists():
        raise SystemExit(f"ERROR: {EXT_PATH} not found.")
    if not GEMINI_PATH.exists():
        raise SystemExit(f"ERROR: {GEMINI_PATH} not found.")
    return (
        json.load(open(GPT5_PATH)),
        json.load(open(GEMINI_PATH)),
        json.load(open(EXT_PATH)),
    )


def judge_denominators(ds_key, judge_data):
    """Recover total-sentences-per-doc denominators from judge `wl_pct`."""
    s = judge_data[ds_key]["summary"]
    fact_total = s["fact_wl"]["mean"] * 100.0 / s["fact_wl_pct"]["mean"]
    omit_total = s["omit_wl"]["mean"] * 100.0 / s["omit_wl_pct"]["mean"]
    return fact_total, omit_total


def judge_row(ds_key, judge_data):
    s = judge_data[ds_key]["summary"]
    return (
        s["fact_wl"]["mean"],
        s["fact_wl_pct"]["mean"],
        s["fact_recall"]["mean"] * 100,
        s["omit_wl"]["mean"],
        s["omit_wl_pct"]["mean"],
        s["omit_recall"]["mean"] * 100,
    )


def ext_row(key, ext):
    """External-scorer row from fst_external_scorer_results.json (LTT-FST).

    Absolute flagged counts are per-resplit means; flagged-% and recall come
    from the summary block.
    """
    e = ext[key]
    s = e["summary"]
    pr = e["per_resplit"]
    fwl_m = float(np.mean([r["fact_wl"] for r in pr]))
    owl_m = float(np.mean([r["omit_wl"] for r in pr]))
    return (
        fwl_m,
        s["fact_wl_pct"]["mean"],
        s["fact_recall"]["mean"] * 100,
        owl_m,
        s["omit_wl_pct"]["mean"],
        s["omit_recall"]["mean"] * 100,
    )


# ---------- LaTeX ----------

def generate_latex(gpt5, gemini, ext):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{\textbf{Validity is scorer-agnostic; efficiency is not.} "
        r"Companion to Table~\ref{tab:external-scorers}. Each dataset block "
        r"groups rows into \emph{LLM-based judges} (\textsc{GPT-5-mini}, the "
        r"main pipeline; \textsc{Gemini~2.5~Pro} as a second LLM family on the "
        r"two datasets we ran it on) and \emph{External scorers} from the "
        r"faithfulness-evaluation literature (NLI cross-encoder, BERT embedding "
        r"cosine, AlignScore). All scorers run \textsc{Care} at $\alpha=0.15$ "
        r"over the same 100 random 70/30 cal/test resplits. \emph{Flagged}: "
        r"mean sentences flagged per document (and \% of total), matching "
        r"Tables~\ref{tab:main-results}--\ref{tab:baselines}. \emph{Rec\%}: "
        r"mean fraction of true hallucinations / true omissions surfaced. "
        r"All scorers maintain $\alpha$-validity (Table~\ref{tab:external-scorers}). "
        r"LLM-based judges achieve it with substantially fewer flags at "
        r"comparable or better recall, and are the only scorer family that "
        r"retains usable recall on \textsc{MIMIC-CXR} radiology reports "
        r"(83\% / 59\% factuality / omission recall for \textsc{GPT-5-mini}, "
        r"vs.\ 53--60\% / 38--43\% for external scorers).}"
    )
    lines.append(r"\label{tab:external-scorers-efficiency}")
    lines.append(r"\begin{tabular}{lll rr rr}")
    lines.append(r"\toprule")
    lines.append(r" & & & \multicolumn{2}{c}{Factuality} & \multicolumn{2}{c}{Omission} \\")
    lines.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"Dataset & Type & Scorer & Flagged & Rec\% & Flagged & Rec\% \\")
    lines.append(r"\midrule")

    first_dataset = True
    for ds_key, meta in DATASETS.items():
        if ds_key not in gpt5:
            continue
        if not first_dataset:
            lines.append(r"\midrule")
        first_dataset = False

        # Build LLM-judge rows
        llm_rows = []
        fwl_m, fwl_pct, fr_m, owl_m, owl_pct, or_m = judge_row(ds_key, gpt5)
        llm_rows.append(
            (
                r"\textbf{GPT-5-mini}",
                rf"\textbf{{{fwl_m:.1f}}} ({fwl_pct:.1f}\%)",
                rf"\textbf{{{fr_m:.1f}}}",
                rf"\textbf{{{owl_m:.1f}}} ({owl_pct:.1f}\%)",
                rf"\textbf{{{or_m:.1f}}}",
            )
        )
        if ds_key in gemini:
            fwl_m, fwl_pct, fr_m, owl_m, owl_pct, or_m = judge_row(ds_key, gemini)
            llm_rows.append(
                (
                    "Gemini 2.5 Pro",
                    fmt_flagged(fwl_m, fwl_pct),
                    fmt_rec(fr_m),
                    fmt_flagged(owl_m, owl_pct),
                    fmt_rec(or_m),
                )
            )

        # Build external rows
        ext_rows = []
        for scorer in EXT_ORDER:
            key = f"{ds_key}__{scorer}"
            if key not in ext:
                continue
            fwl_m, fwl_pct, fr_m, owl_m, owl_pct, or_m = ext_row(key, ext)
            ext_rows.append(
                (
                    EXT_LABELS[scorer],
                    fmt_flagged(fwl_m, fwl_pct),
                    fmt_rec(fr_m),
                    fmt_flagged(owl_m, owl_pct),
                    fmt_rec(or_m),
                )
            )

        n_llm = len(llm_rows)
        n_ext = len(ext_rows)
        n_block = n_llm + n_ext
        ds_cell = rf"\multirow{{{n_block}}}{{*}}{{\textbf{{{meta['label']}}}}}"

        def type_cell(label, n, i):
            if i != 0:
                return ""
            return label if n == 1 else rf"\multirow{{{n}}}{{*}}{{{label}}}"

        # Emit LLM judge rows.
        for i, (scorer, ff, fr, of, orr) in enumerate(llm_rows):
            ds_field = ds_cell if i == 0 else ""
            lines.append(
                f"  {ds_field} & {type_cell('LLM', n_llm, i)} & {scorer} & "
                f"{ff} & {fr} & {of} & {orr} \\\\"
            )

        # Light separator between LLM and external groups.
        lines.append(r"\addlinespace[2pt]")

        # Emit external scorer rows.
        for i, (scorer, ff, fr, of, orr) in enumerate(ext_rows):
            lines.append(
                f"   & {type_cell('External', n_ext, i)} & {scorer} & "
                f"{ff} & {fr} & {of} & {orr} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------- text dump ----------

def generate_text(gpt5, gemini, ext):
    out = [
        f"Scorer-agnostic CRC: Flagged + Recall (alpha={ALPHA}, 100 resplits)",
        "=" * 100,
        f"{'Dataset':<11} {'Type':<8} {'Scorer':<16} | "
        f"{'FactFlagged':>15} {'FactRec%':>9} | {'OmitFlagged':>15} {'OmitRec%':>9}",
        "-" * 100,
    ]
    for ds_key, meta in DATASETS.items():
        if ds_key not in gpt5:
            continue
        fwl_m, fwl_pct, fr_m, owl_m, owl_pct, or_m = judge_row(ds_key, gpt5)
        out.append(
            f"{meta['label']:<11} {'LLM':<8} {'GPT-5-mini':<16} | "
            f"{fwl_m:6.1f} ({fwl_pct:5.1f}%) {fr_m:8.1f} | "
            f"{owl_m:6.1f} ({owl_pct:5.1f}%) {or_m:8.1f}"
        )
        if ds_key in gemini:
            fwl_m, fwl_pct, fr_m, owl_m, owl_pct, or_m = judge_row(ds_key, gemini)
            out.append(
                f"{'':<11} {'':<8} {'Gemini 2.5 Pro':<16} | "
                f"{fwl_m:6.1f} ({fwl_pct:5.1f}%) {fr_m:8.1f} | "
                f"{owl_m:6.1f} ({owl_pct:5.1f}%) {or_m:8.1f}"
            )
        first_ext = True
        for scorer in EXT_ORDER:
            key = f"{ds_key}__{scorer}"
            if key not in ext:
                continue
            fwl_m, fwl_pct, fr_m, owl_m, owl_pct, or_m = ext_row(key, ext)
            type_lbl = "External" if first_ext else ""
            first_ext = False
            out.append(
                f"{'':<11} {type_lbl:<8} {EXT_LABELS[scorer]:<16} | "
                f"{fwl_m:6.1f} ({fwl_pct:5.1f}%) {fr_m:8.1f} | "
                f"{owl_m:6.1f} ({owl_pct:5.1f}%) {or_m:8.1f}"
            )
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    gpt5, gemini, ext = load()
    print(generate_text(gpt5, gemini, ext))
    OUT_PATH.write_text(generate_latex(gpt5, gemini, ext))
    print(f"Saved: {OUT_PATH}")
