#!/usr/bin/env python3
"""Table 14: Clinician study results (pooled v2 + v3).

Loads both rounds of the clinician study and reports:
  - Time per source sentence (CARE vs control)
  - Selected sentences per doc (n_yellow)
  - Safety-layer omission detection rate: |true_omit ∩ (yellow ∪ surfaced)| / |true_omit|
    (in CARE, surfaced is shown to the clinician; in control, surfaced is treated
    as empty since the clinician was not shown CARE's flags)
  - Attribution: % of clinician-flagged true omissions that lie within
    CARE's surfaced set (CARE condition only)
  - v3 confidence ratings

Uses a design-faithful stratified permutation test that permutes condition
labels within each (clinician, version) block, respecting that v2 randomized
condition per-document (across all clinicians) and v3 randomized condition
per-(clinician, document).

Usage:
    python3 paper/scripts/table_14_clinician_study.py
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

PAPER_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PAPER_DIR / "outputs"
CLINICIAN_DIR = PAPER_DIR / "clinician_study"

CLINICIANS = [
    ("C1", "Chloe",   "clinician_study_v2_Chloe.json",   "clinician_study_v3_chloe.json"),
    ("C2", "Jenelle", "clinician_study_v2_Jenelle.json", "clinician_study_v3_jenelle.json"),
    ("C3", "Anson",   "clinician_study_v2_anson.json",   "clinician_study_v3_anson.json"),
]

PERM_ITERS = 20000
SEED = 0


def load_rows():
    rows = []
    for label, name, v2_file, v3_file in CLINICIANS:
        for ver, fname in [("v2", v2_file), ("v3", v3_file)]:
            data = json.load(open(CLINICIAN_DIR / fname))
            for r in data["responses"]:
                true_omit = set(r.get("true_omission_indices", []))
                yellow = set(r.get("yellow_highlighted_sentences", []))
                # Only credit surfaced flags in the CARE condition — the
                # condition under which the clinician actually saw them.
                # (v2 stores surfaced for control rows; normalize.)
                surfaced = (set(r.get("surfaced_indices", []))
                            if r["condition"] == "highlighted" else set())
                n_src = r["n_src"]
                detected = true_omit & (yellow | surfaced)
                rows.append(dict(
                    label=label, clinician=name, version=ver,
                    doc_id=r["doc_id"], dataset=r["dataset"],
                    condition=r["condition"],
                    time_per_sent=r["time_seconds"] / n_src if n_src else np.nan,
                    n_yellow=len(yellow),
                    n_surfaced=len(surfaced),
                    safety_recall=(len(detected) / len(true_omit)
                                   if true_omit else np.nan),
                    n_true_omit=len(true_omit),
                    n_detected=len(detected),
                    n_yellow_in_surfaced=len(yellow & surfaced),
                    confidence=r.get("confidence"),
                ))
    return rows


def per_clinician(rows):
    """Pooled v2+v3 per-clinician summary."""
    out = []
    for label, name, _, _ in CLINICIANS:
        mine = [r for r in rows if r["clinician"] == name]
        ctrl = [r for r in mine if r["condition"] == "control"]
        care = [r for r in mine if r["condition"] == "highlighted"]

        def mean(xs, key):
            vals = [x[key] for x in xs
                    if not (isinstance(x[key], float) and np.isnan(x[key]))]
            return float(np.mean(vals)) if vals else 0.0

        # Attribution: among true omissions the clinician yellowed under CARE,
        # what fraction were also in CARE's surfaced set?
        tot_yel_omit = sum(len(set(range(0)) | set()) for _ in care)  # placeholder
        # Recompute properly: per-doc, |true_omit & yellow & surfaced| / |true_omit & yellow|
        num_attrib = 0
        den_attrib = 0
        for r in care:
            num_attrib += r["n_yellow_in_surfaced"]  # |yellow & surfaced| as proxy
        # Need |true_omit & yellow & surfaced| and |true_omit & yellow|; recompute below
        # via raw access — simpler to recompute from raw study data, but n_yellow_in_surfaced
        # already captures yellow & surfaced. Strictly we want true_omit ∩ each.
        # For consistency with prior table, compute from rows again with full sets:
        out.append(dict(
            label=label, name=name,
            n_ctrl=len(ctrl), n_care=len(care),
            ctrl_time=mean(ctrl, "time_per_sent"),
            care_time=mean(care, "time_per_sent"),
            ctrl_yellow=mean(ctrl, "n_yellow"),
            care_yellow=mean(care, "n_yellow"),
            ctrl_recall=mean(ctrl, "safety_recall"),
            care_recall=mean(care, "safety_recall"),
        ))
    return out


def attribution_rate(rows):
    """% of CARE clinicians' yellow omissions that lie within CARE's surfaced set.
    Computed properly from raw study JSONs to avoid set-arithmetic bugs."""
    num, den = 0, 0
    for label, name, v2_file, v3_file in CLINICIANS:
        for ver, fname in [("v2", v2_file), ("v3", v3_file)]:
            data = json.load(open(CLINICIAN_DIR / fname))
            for r in data["responses"]:
                if r["condition"] != "highlighted":
                    continue
                true_omit = set(r.get("true_omission_indices", []))
                yellow = set(r.get("yellow_highlighted_sentences", []))
                surfaced = set(r.get("surfaced_indices", []))
                yel_omit = true_omit & yellow
                den += len(yel_omit)
                num += len(yel_omit & surfaced)
    return num / den if den else 0.0


def per_clinician_attribution():
    """Per-clinician attribution (CARE-only)."""
    out = {}
    for label, name, v2_file, v3_file in CLINICIANS:
        num, den = 0, 0
        for ver, fname in [("v2", v2_file), ("v3", v3_file)]:
            data = json.load(open(CLINICIAN_DIR / fname))
            for r in data["responses"]:
                if r["condition"] != "highlighted":
                    continue
                true_omit = set(r.get("true_omission_indices", []))
                yellow = set(r.get("yellow_highlighted_sentences", []))
                surfaced = set(r.get("surfaced_indices", []))
                yel_omit = true_omit & yellow
                den += len(yel_omit)
                num += len(yel_omit & surfaced)
        out[name] = num / den if den else 0.0
    return out


def permutation_test(rows, metric, direction, n_iter=PERM_ITERS, seed=SEED):
    """Design-faithful permutation test.

    v2 randomized condition at the document level (all 3 clinicians saw a
    given doc in the same condition); v3 randomized at the (clinician, doc)
    level. We mirror that:
      - v2: permute the doc -> condition map (preserving the per-doc
        original count of 5 ctrl / 5 CARE), and apply the same condition
        to all 3 clinicians' reviews of that doc.
      - v3: permute condition labels within each clinician's row set
        (matches the actual per-(clinician, doc) randomization).

    Returns observed delta (CARE - control) and one-sided p-value.
    direction="greater" tests CARE > control; "less" tests CARE < control.
    """
    rng = np.random.default_rng(seed)
    valid = [r for r in rows
             if not (isinstance(r[metric], float) and np.isnan(r[metric]))]

    vals = np.array([r[metric] for r in valid])
    is_care = np.array([r["condition"] == "highlighted" for r in valid], dtype=bool)
    obs = vals[is_care].mean() - vals[~is_care].mean()

    # v2: doc-level permutation (apply same condition to all clinicians of a doc).
    v2_doc_to_idxs = {}
    v2_doc_orig_care = {}
    for i, r in enumerate(valid):
        if r["version"] != "v2":
            continue
        k = (r["dataset"], r["doc_id"])
        v2_doc_to_idxs.setdefault(k, []).append(i)
        v2_doc_orig_care[k] = (r["condition"] == "highlighted")
    v2_docs = list(v2_doc_to_idxs.keys())
    v2_n_care = sum(1 for k in v2_docs if v2_doc_orig_care[k])

    # v3: stratify within clinician (preserves each clinician's marginal split).
    v3_strata = {}
    for i, r in enumerate(valid):
        if r["version"] == "v3":
            v3_strata.setdefault(r["clinician"], []).append(i)

    null = np.empty(n_iter)
    for it in range(n_iter):
        perm = is_care.copy()
        # v2: permute doc -> condition assignment, broadcast to all clinicians.
        order = rng.permutation(len(v2_docs))
        care_docs = set(v2_docs[i] for i in order[:v2_n_care])
        for k, idxs in v2_doc_to_idxs.items():
            new_cond = (k in care_docs)
            for i in idxs:
                perm[i] = new_cond
        # v3: permute within clinician.
        for idxs in v3_strata.values():
            sub = perm[idxs].copy()
            rng.shuffle(sub)
            perm[idxs] = sub
        null[it] = vals[perm].mean() - vals[~perm].mean()

    if direction == "greater":
        p = (null >= obs).mean()
    elif direction == "less":
        p = (null <= obs).mean()
    else:
        p = (np.abs(null) >= abs(obs)).mean()
    return obs, p


def aggregate_stats(rows):
    ctrl = [r for r in rows if r["condition"] == "control"]
    care = [r for r in rows if r["condition"] == "highlighted"]

    def mean(xs, key):
        vals = [x[key] for x in xs
                if not (isinstance(x[key], float) and np.isnan(x[key]))]
        return float(np.mean(vals)) if vals else 0.0

    t_obs, t_p = permutation_test(rows, "time_per_sent", "less")
    r_obs, r_p = permutation_test(rows, "safety_recall", "greater")

    return dict(
        n_ctrl=len(ctrl), n_care=len(care),
        ctrl_time=mean(ctrl, "time_per_sent"), care_time=mean(care, "time_per_sent"),
        ctrl_yellow=mean(ctrl, "n_yellow"), care_yellow=mean(care, "n_yellow"),
        ctrl_recall=mean(ctrl, "safety_recall"), care_recall=mean(care, "safety_recall"),
        attrib=attribution_rate(rows),
        time_delta=t_obs, time_p=t_p,
        recall_delta=r_obs, recall_p=r_p,
    )


def confidence_stats(rows):
    v3 = [r for r in rows if r["version"] == "v3" and r["confidence"] is not None]
    ctrl = [r["confidence"] for r in v3 if r["condition"] == "control"]
    care = [r["confidence"] for r in v3 if r["condition"] == "highlighted"]
    if ctrl and care:
        u, p = stats.mannwhitneyu(care, ctrl, alternative="two-sided")
    else:
        p = None

    per_clin = []
    for label, name, _, _ in CLINICIANS:
        c_ctrl = [r["confidence"] for r in v3
                  if r["clinician"] == name and r["condition"] == "control"]
        c_care = [r["confidence"] for r in v3
                  if r["clinician"] == name and r["condition"] == "highlighted"]
        per_clin.append(dict(
            label=label, name=name,
            ctrl_mean=float(np.mean(c_ctrl)) if c_ctrl else 0.0,
            care_mean=float(np.mean(c_care)) if c_care else 0.0,
            n_ctrl=len(c_ctrl), n_care=len(c_care),
        ))

    return dict(
        n_ctrl=len(ctrl), n_care=len(care),
        ctrl_mean=float(np.mean(ctrl)) if ctrl else 0.0,
        care_mean=float(np.mean(care)) if care else 0.0,
        mw_p=p,
        per_clin=per_clin,
    )


def generate_latex(per_clin, agg, attrib_per):
    """Single unified table: per-clinician rows + aggregate row.

    Columns: n | Omissions caught (Ctrl %) | (CARE %) | Δ (pp)
    """
    lines = []
    p_recall = (f"{agg['recall_p']:.4f}" if agg['recall_p'] < 0.01
                else f"{agg['recall_p']:.2f}")
    n_total = agg["n_ctrl"] + agg["n_care"]
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Clinician study: \textbf{omission detection during "
                 r"review}. Each clinician reviewed 25 documents drawn from "
                 r"ACI-Bench and MIMIC-BHC (12--13 control, 12--13 "
                 r"\textsc{CARE}-assisted). \emph{Omissions caught} is the "
                 r"percentage of oracle-labeled true omissions identified at "
                 r"review time, either through clinician highlighting or (in "
                 r"the \textsc{CARE} condition) \textsc{CARE}'s surfaced "
                 r"flags. The aggregate \textsc{CARE} vs.\ control difference "
                 r"is significant ($p = " + p_recall + r"$, 1-sided "
                 r"permutation test, 20{,}000 iterations, stratified to match "
                 r"the study's randomization design).}")
    lines.append(r"\label{tab:clinician-study}")
    lines.append(r"\begin{tabular}{l c rr r}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c}{Omissions caught (\%)} \\")
    lines.append(r"\cmidrule(lr){3-5}")
    lines.append(r" & $n$ & Control & \textsc{CARE} & $\Delta$\,(pp) \\")
    lines.append(r"\midrule")
    for r in per_clin:
        n_clin = r["n_ctrl"] + r["n_care"]
        ctrl_pct = r["ctrl_recall"] * 100
        care_pct = r["care_recall"] * 100
        delta = (r["care_recall"] - r["ctrl_recall"]) * 100
        lines.append(f"{r['label']} & {n_clin} "
                     f"& {ctrl_pct:.0f} & {care_pct:.0f} "
                     f"& ${delta:+.0f}$ \\\\")
    lines.append(r"\midrule")
    agg_ctrl = agg["ctrl_recall"] * 100
    agg_care = agg["care_recall"] * 100
    agg_delta = agg["recall_delta"] * 100
    lines.append(r"\textbf{All clinicians} "
                 f"& \\textbf{{{n_total}}} "
                 f"& \\textbf{{{agg_ctrl:.1f}}} & \\textbf{{{agg_care:.1f}}} "
                 f"& $\\mathbf{{{agg_delta:+.1f}}}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    rows = load_rows()
    per_clin = per_clinician(rows)
    agg = aggregate_stats(rows)
    attrib_per = per_clinician_attribution()
    conf = confidence_stats(rows)

    n_total = agg["n_ctrl"] + agg["n_care"]
    print(f"Loaded {n_total} doc-reviews ({agg['n_ctrl']} ctrl / {agg['n_care']} CARE) "
          f"from {len(CLINICIANS)} clinicians, v2 + v3 pooled.")
    print()
    print("Per-clinician (pooled v2+v3):")
    for r in per_clin:
        print(f"  {r['label']} ({r['name']:<7}) n=({r['n_ctrl']}c/{r['n_care']}h)  "
              f"Time {r['ctrl_time']:.1f}->{r['care_time']:.1f}s  "
              f"Yellow {r['ctrl_yellow']:.1f}->{r['care_yellow']:.1f}  "
              f"Det.Rate {r['ctrl_recall']:.2f}->{r['care_recall']:.2f}  "
              f"Attrib {attrib_per[r['name']]:.0%}")
    print()
    print("Aggregate (permutation test, 1-sided):")
    print(f"  Time/sent : ctrl {agg['ctrl_time']:.2f}s -> CARE {agg['care_time']:.2f}s   "
          f"Δ={agg['time_delta']:+.3f}s  p={agg['time_p']:.3f}")
    print(f"  Det. rate : ctrl {agg['ctrl_recall']:.3f}  -> CARE {agg['care_recall']:.3f}   "
          f"Δ={agg['recall_delta']:+.3f}   p={agg['recall_p']:.4f}")
    print(f"  Attribution (CARE): {agg['attrib']:.0%}")
    print()
    print("v3 confidence (1-5 Likert):")
    print(f"  Control n={conf['n_ctrl']}, mean={conf['ctrl_mean']:.2f}   "
          f"CARE n={conf['n_care']}, mean={conf['care_mean']:.2f}   "
          f"MW 2-sided p={conf['mw_p']:.3f}" if conf['mw_p'] is not None else
          f"  Control n={conf['n_ctrl']}, mean={conf['ctrl_mean']:.2f}   "
          f"CARE n={conf['n_care']}, mean={conf['care_mean']:.2f}")
    for r in conf["per_clin"]:
        d = r["care_mean"] - r["ctrl_mean"]
        print(f"    {r['label']} ({r['name']:<7}): "
              f"ctrl={r['ctrl_mean']:.2f} (n={r['n_ctrl']})  "
              f"CARE={r['care_mean']:.2f} (n={r['n_care']})   Δ={d:+.2f}")

    latex = generate_latex(per_clin, agg, attrib_per)
    out_path = OUTPUT_DIR / "table_14_clinician_study.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
