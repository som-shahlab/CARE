#!/usr/bin/env python3
"""Worked CARE example on one ACI-Bench test-split document (camera-ready appendix).

Applies the canonical-split thresholds (outputs/fst_thresholds.json) to one ACI-Bench document
from the canonical test split and renders every summary sentence and every source turn with
its judge scores, oracle labels and CARE decision.

Usage: python example_aci.py [--doc-id 73]
Output: paper/outputs/camera_ready/example_aci.tex (two table* environments)
"""
import argparse
import json
import re
from pathlib import Path

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, PAPER_OUTPUT_DIR as OUTPUT_DIR


def tex(s):
    s = s.replace("**", "").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s.replace(r"\textbackslash\{\}", r"\textbackslash{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", type=int, default=73)
    a = ap.parse_args()
    rows = {json.loads(l)["doc_id"]: json.loads(l) for l in open(DATA_ROOT / "ACI_Bench/phase_2/calibrated_scores.jsonl")}
    test = set(json.load(open(DATA_ROOT / "ACI_Bench/split/split_indices.json"))["test_indices"])
    assert a.doc_id in test, "example must come from the canonical test split"
    th = json.load(open(OUTPUT_DIR / "fst_thresholds.json"))["ACI_Bench"]
    lam, tau, gam = th["lambda"], th["tau"], th["gamma"]
    r = rows[a.doc_id]

    # ---------------- summary sentences (hallucination controller)
    S = []
    n_hal = n_flag = n_tp = 0
    for i, (s, p, y) in enumerate(zip(r["generated_sentences"], r["factuality_probs"], r["factuality_labels"]), 1):
        flag = p <= lam
        hal = y == 0
        n_hal += hal; n_flag += flag; n_tp += flag and hal
        mark = (r"\textcolor{redflag}{\textbf{flag}}" if flag else "") + (r" \textbf{(TP)}" if flag and hal else
                (r" (FP)" if flag else (r" \textbf{missed}" if hal else "")))
        txt = tex(s)
        if hal:
            txt = r"\cellcolor{lightred}" + txt
        S.append(f"{i} & {txt} & {p:.2f} & {'Unsupp.' if hal else 'Supp.'} & {mark} \\\\")

    # ---------------- source turns (omission controller)
    U = []
    n_om = n_surf = n_hit = 0
    for j, (u, pi, pc, yi, yc) in enumerate(zip(r["source_sentences"], r["importance_probs"], r["coverage_probs"],
                                               r["importance_labels"], r["coverage_labels"]), 1):
        surf = pi >= tau and (1 - pc) >= gam
        om = yi == 1 and yc == 0
        n_om += om; n_surf += surf; n_hit += surf and om
        mark = (r"\textcolor{blueflag}{\textbf{surface}}" if surf else "") + (r" \textbf{(TP)}" if surf and om else
                (r" \textbf{missed}" if om else ""))
        lab = ("Imp" if yi == 1 else "--") + "/" + ("Cov" if yc == 1 else "Omit")
        txt = tex(u)
        if om:
            txt = r"\cellcolor{purplebg}" + txt
        U.append(f"{j} & {txt} & {pi:.2f} & {1 - pc:.2f} & {lab} & {mark} \\\\")

    hdr = (f"Document {a.doc_id} (canonical test split). Thresholds from the canonical calibration split: "
           f"$\\lambda^*={lam:.2f}$, $(\\tau^*,\\gamma^*)=({tau:.2f},{gam:.2f})$.")
    out = []
    out += [r"\begin{table*}[!p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Worked example, part 1: generated summary with hallucination flags. " + hdr +
            f" A summary sentence is flagged when $\\hat p_{{\\mathrm{{sup}}}}\\le\\lambda^*$. "
            f"The oracle labels {n_hal} sentence(s) unsupported (shaded); \\textsc{{CARE}} flags {n_flag} sentences, "
            f"{n_tp} of which are the unsupported ones (TP); the rest are false positives (FP) routed to review.}}",
            r"\label{tab:example-summary}",
            r"\begin{tabular}{r p{11.4cm} c c l}", r"\toprule",
            r"\# & Summary sentence & $\hat p_{\mathrm{sup}}$ & Oracle & \textsc{CARE} \\", r"\midrule"]
    out += S + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out += [r"\begin{table*}[!p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Worked example, part 2: source dialogue with omission surfacing for the same document. "
            r"A source turn is surfaced when $\hat p_{\mathrm{imp}}\ge\tau^*$ and $\hat p_{\mathrm{ncov}}\ge\gamma^*$. "
            f"The oracle labels {n_om} turn(s) as true omissions (important and not covered; shaded); \\textsc{{CARE}} surfaces "
            f"{n_surf} of {len(U)} turns, including {n_hit} of the {n_om} true omissions. Other surfaced turns are "
            r"important content the judge was unsure the summary covered.}",
            r"\label{tab:example-source}",
            r"\begin{tabular}{r p{10.6cm} c c c l}", r"\toprule",
            r"\# & Source turn & $\hat p_{\mathrm{imp}}$ & $\hat p_{\mathrm{ncov}}$ & Oracle & \textsc{CARE} \\", r"\midrule"]
    out += U + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    dst = OUTPUT_DIR / "camera_ready"
    dst.mkdir(exist_ok=True)
    (dst / "example_aci.tex").write_text("\n".join(out) + "\n")
    print(f"doc {a.doc_id}: {n_hal} halluc, {n_flag} flagged ({n_tp} TP); {n_om} omissions, {n_surf} surfaced ({n_hit} TP)")


if __name__ == "__main__":
    main()
