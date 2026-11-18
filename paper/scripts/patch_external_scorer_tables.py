#!/usr/bin/env python3
"""Patch Tables 11/12 (external scorers) in ML4H_2026___CARE/main.tex from LTT-FST results.

Reads outputs/fst_external_scorer_results.json (and outputs/external_scorer_results.json for the
Max-F1 columns), and rewrites, for every dataset x scorer that has finished:
  - Table 11 (tab:external-scorers): the full row (hallucination CARE, Max-F1, omission CARE, Max-F1)
  - Table 12 (tab:external-scorers-efficiency): the omission Flagged / Rec% cells
Rows without results are left untouched (still marked \\stale{}). Safe to re-run.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_ready_tables as crt  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
TEX = PAPER / "ML4H_2026___CARE" / "main.tex"
SC = {"nli_embed": "NLI (DeBERTa)", "embed_only": "Embed (BERT)", "alignscore": "AlignScore"}
DSL = dict(crt.DS)


def block(s, label):
    i = s.index(f"\\label{{{label}}}")
    return i, s.index("\\end{table}", i)


COMPACT_BEGIN, COMPACT_END = "% BEGIN-COMPACT-SCORERS", "% END-COMPACT-SCORERS"
SCORER_ORDER = ["GPT-5-mini", "Gemini 2.5 Pro", "Qwen3.5-4B", "NLI (DeBERTa)", "Embed (BERT)", "AlignScore"]


def sync_compact(s):
    """Regenerate the main-text compact scorer table (tab:scorers-compact) from the full
    appendix table (tab:external-scorers-efficiency): % of sentences flagged per controller."""
    if COMPACT_BEGIN not in s:
        return s
    a, b = block(s, "tab:external-scorers-efficiency")
    body = s[s.rindex("\\begin{table}", 0, a):b]
    dsl = [lab for _, lab in crt.DS]
    cells, ds = {}, None
    for line in body.splitlines():
        m = re.search(r"\\textbf\{([^}]*)\}\}", line)
        if m and m.group(1) in dsl:
            ds = m.group(1)
        if ds is None or line.count("&") < 6:
            continue
        cols = [c.strip() for c in line.rstrip("\\ ").split("&")]
        name = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cols[2])
        if name not in SCORER_ORDER:
            continue
        pct = [re.search(r"\(([\d.]+)\\%\)", cols[i]) for i in (3, 5)]
        cells[(name, ds)] = (pct[0].group(1) if pct[0] else "--",
                             pct[1].group(1) if pct[1] else "--", "\\stale" in cols[5])
    best = {}
    for d in dsl:
        for k in (0, 1):
            v = [float(c[k]) for (n, dd), c in cells.items()
                 if dd == d and c[k] != "--" and not (k == 1 and c[2])]
            best[(d, k)] = min(v) if v else None

    def fmt(n, d):
        c = cells.get((n, d))
        if c is None:
            return "--"
        out = []
        for k in (0, 1):
            x = c[k]
            if x != "--" and best[(d, k)] is not None and float(x) == best[(d, k)]:
                x = f"\\textbf{{{x}}}"
            if k == 1 and c[2]:
                x = f"\\stale{{{x}}}"
            out.append(x)
        return " / ".join(out)

    rows = [f"{n} & " + " & ".join(fmt(n, d) for d in dsl) + " \\\\" for n in SCORER_ORDER]
    rows.insert(3, "\\midrule")
    short = {"ACI-Bench": "ACI", "MIMIC-BHC": "BHC", "MIMIC-CXR": "CXR", "Priv-DS": "Priv-DS", "SumPubMed": "PubMed"}
    tab = "\n".join([
        COMPACT_BEGIN,
        "\\begin{table}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\caption{Review workload by scorer at $\\alpha=0.15$ after \\textsc{CARE} calibration (100 resplits). "
        "Each cell: \\% of summary sentences flagged (hallucination) / \\% of source sentences surfaced (omission); "
        "\\textbf{bold}: lowest per dataset and controller. Top: LLM judges; bottom: external scorers, with importance "
        "held fixed to the LLM judge. Validity is in Table~\\ref{tab:external-scorers}; counts and recall are in "
        "Table~\\ref{tab:external-scorers-efficiency}.}",
        "\\label{tab:scorers-compact}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l ccccc}", "\\toprule",
        "Scorer & " + " & ".join(short[d] for d in dsl) + " \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}}", "\\end{table}", COMPACT_END])
    i, j = s.index(COMPACT_BEGIN), s.index(COMPACT_END) + len(COMPACT_END)
    return s[:i] + tab + s[j:]


def main():
    if not TEX.exists():
        print(f"skip: {TEX} not present (paper source is not distributed)")
        return
    fst = json.load(open(PAPER / "outputs" / "fst_external_scorer_results.json"))
    old = json.load(open(PAPER / "outputs" / "external_scorer_results.json"))
    s = TEX.read_text()
    n11 = n12 = 0
    for key, r in fst.items():
        ds, sc = key.split("__")
        if sc not in SC or "per_resplit" not in r:
            continue
        per = r["per_resplit"]
        lab, name = DSL[ds], SC[sc]

        def pm(m):
            v = np.array([x[m] for x in per]) * 100
            return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))

        def mf(m):
            o = old.get(key)
            if o is None:
                return None
            if "per_resplit" in o:
                v = np.array([x["maxf1"][m] for x in o["per_resplit"]]) * 100
                return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            q = o["summary"][f"maxf1_{m}"]
            return q["mean"] * 100, 1.96 * q["std"] * 100 / 10

        def cell(x, flag=False):
            m, h = x
            if flag and m - h > 15:
                return f"\\textbf{{{m:.1f}}}$^\\dagger${{\\scriptsize $\\pm${h:.1f}}}"
            return f"{m:.1f}{{\\scriptsize $\\pm${h:.1f}}}"

        # ---- Table 11: locate the dataset's rows, then the scorer row inside that block
        a, b = block(s, "tab:external-scorers")
        body = s[a:b]
        start = body.index(f"{lab} & ")
        nxt = body.find("\\midrule", start)
        nxt = len(body) if nxt == -1 else nxt
        seg = body[start:nxt]
        m = re.search(r"(?m)^(?P<pre>[^\n&]*)& " + re.escape(name) + r" &[^\n]*\\\\$", seg)
        if m:
            row = (f"{m.group('pre')}& {name} & {cell(pm('fact_viol'))} & {cell(mf('fact_viol'), True)} & "
                   f"{cell(pm('omit_viol'))} & {cell(mf('omit_viol'), True)} \\\\")
            seg = seg[:m.start()] + row + seg[m.end():]
            body = body[:start] + seg + body[nxt:]
            s = s[:a] + body + s[b:]
            n11 += 1

        # ---- Table 12: omission cells of the external-scorer row in this dataset's block
        a, b = block(s, "tab:external-scorers-efficiency")
        body = s[a:b]
        start = body.index(f"{{\\textbf{{{lab}}}}}")
        nxt = body.find("\\midrule", start)
        nxt = len(body) if nxt == -1 else nxt
        seg = body[start:nxt]
        m = re.search(r"(?m)^(?P<pre>.*?& " + re.escape(name) + r" & [^&]+ & [^&]+ & ).*\\\\$", seg)
        if m:
            owl = np.mean([x["omit_wl"] for x in per])
            owlp = np.mean([x["omit_wl_pct"] for x in per])
            orc = np.mean([x["omit_recall"] for x in per]) * 100
            row = f"{m.group('pre')}{owl:.1f} ({owlp:.1f}\\%) & {orc:.1f} \\\\"
            seg = seg[:m.start()] + row + seg[m.end():]
            body = body[:start] + seg + body[nxt:]
            s = s[:a] + body + s[b:]
            n12 += 1
    s = sync_compact(s)
    TEX.write_text(s)
    left = s.count("\\stale{")
    print(f"patched Table 11 rows: {n11}, Table 12 rows: {n12}; \\stale occurrences left: {left}")


if __name__ == "__main__":
    main()
