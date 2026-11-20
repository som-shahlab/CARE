"""Evaluate CARE's submitted thresholds (paper Table 9) against clinician labels instead of
GPT-5 oracle labels, on the independently annotated documents (R3 W1, R4, R5).

Thresholds are the submitted ones (alpha = 0.15). Annotated sentences are matched to the
phase_2 judge scores within each document. Primary matching ("normalized") undoes spreadsheet
artifacts: strip a leading quote (Excel text escape), collapse whitespace, lowercase; then a
one-to-one fuzzy match (difflib ratio >= 0.9) for the rest, which only strips list numbering
("2. ") or whitespace splits. Still-unmatched sentences (ACI: 32 empty section headers such as
"2. **Diabetes:**." that the current pipeline drops, plus 1 sentence) are excluded and counted.
Exact-text matching is reported as a sensitivity check.
"""
import difflib, json, re
import numpy as np
import pandas as pd
from pathlib import Path

from care.config import CANONICAL_DATA_ROOT as ROOT  # honours CARE_DATA_ROOT
D = Path(__file__).resolve().parents[1] / "clinician_study"
SETS = {"ACI-Bench": ("ACI_Bench", "ACI (3)"), "MIMIC-BHC": ("MIMIC_IV_BHC", "BHC (1)"), "MIMIC-CXR": ("MIMIC_III_CXR", "CXR")}
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from care_thresholds import thresholds, mode
THRESH = {k: v for k, v in thresholds().items() if k in SETS}  # lambda, tau, gamma (camera-ready by default; CARE_THRESHOLDS=submitted for Table 9 as reviewed)
LABELERS = {"Oracle (GPT-5)": "ORACLE", "Clinician 1": "ANNOTATOR_1", "Clinician 2": "ANNOTATOR_2"}
FUZZY_MIN = 0.9
rng = np.random.default_rng(0)


def boot_ci(x, B=10000):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lstrip("'\"").strip()).lower()


def match(annotated, current, mode):
    """Index into `current` for each annotated sentence (None if unmatched)."""
    if mode == "exact":
        idx = {c.strip(): i for i, c in enumerate(current)}
        return [idx.get(str(a).strip()) for a in annotated]
    idx = {}
    for i, c in enumerate(current):
        idx.setdefault(_norm(c), i)
    out = [idx.get(_norm(a)) for a in annotated]
    used = {i for i in out if i is not None}
    for k, a in enumerate(annotated):
        if out[k] is None:
            cands = [(difflib.SequenceMatcher(None, _norm(a), _norm(c)).ratio(), i)
                     for i, c in enumerate(current) if i not in used]
            r, i = max(cands, default=(0, None))
            if r >= FUZZY_MIN:
                out[k] = i; used.add(i)
    return out


def run(mode):
    out = {}
    for ds, (folder, f) in SETS.items():
        lam, tau, gam = THRESH[ds]
        df = pd.read_csv(D / f"annotation_validation_corrected.xlsx - {f}.csv", usecols=range(14))
        for c in df.columns:
            if c.startswith(("FACT_", "IMP_", "COV_")):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        scores = {d["doc_id"]: d for d in map(json.loads, open(ROOT / folder / "phase_2" / "calibrated_scores.jsonl"))}
        split = json.load(open(ROOT / folder / "split" / "split_indices.json"))
        test_ids = set(split["test_indices"])

        per = {name: dict(doc_hall_viol=[], hall_tot=0, hall_flagged=0, doc_omit_loss=[], omit_tot=0, omit_surf=0) for name in LABELERS}
        matched = dict(gen=0, gen_tot=0, src=0, src_tot=0, n_test_docs=0, unmatched_hall={n: 0 for n in LABELERS})
        flagged_per_doc, surfaced_per_doc = [], []
        for did in df.DOC_ID.dropna().unique():
            d = scores[int(re.search(r"\d+", did).group())]
            matched["n_test_docs"] += d["doc_id"] in test_ids
            sub = df[df.DOC_ID == did]
            gen, src = sub[sub.SENTENCE_TYPE == "generated"], sub[sub.SENTENCE_TYPE == "source"]
            matched["gen_tot"] += len(gen); matched["src_tot"] += len(src)
            gm = match(gen.SENTENCE.tolist(), d["generated_sentences"], mode)
            sm = match(src.SENTENCE.tolist(), d["source_sentences"], mode)
            gkeep, skeep = np.array([i is not None for i in gm], bool), np.array([i is not None for i in sm], bool)
            for name, col in LABELERS.items():
                matched["unmatched_hall"][name] += int((gen[f"FACT_{col}"].values[~gkeep] == 0).sum())
            gen, src = gen[gkeep], src[skeep]
            gm, sm = [i for i in gm if i is not None], [i for i in sm if i is not None]
            matched["gen"] += len(gen); matched["src"] += len(src)
            pf = np.array([d["factuality_probs"][i] for i in gm])
            pi = np.array([d["importance_probs"][i] for i in sm])
            pc = np.array([d["coverage_probs"][i] for i in sm])
            flag = pf <= lam
            surf = (pi >= tau) & ((1 - pc) >= gam)
            flagged_per_doc.append(int(flag.sum())); surfaced_per_doc.append(int(surf.sum()))
            for name, col in LABELERS.items():
                hall = gen[f"FACT_{col}"].values == 0
                per[name]["hall_tot"] += int(hall.sum()); per[name]["hall_flagged"] += int((hall & flag).sum())
                per[name]["doc_hall_viol"].append(int((hall & ~flag).any()))
                iv, cv = src[f"IMP_{col}"].values, src[f"COV_{col}"].values
                omit = (iv == 1) & (cv == 0)
                per[name]["omit_tot"] += int(omit.sum()); per[name]["omit_surf"] += int((omit & surf).sum())
                per[name]["doc_omit_loss"].append(float((omit & ~surf).sum() / omit.sum()) if omit.sum() else 0.0)

        res = dict(n_docs=len(flagged_per_doc), matched=matched, lam=lam, tau=tau, gamma=gam,
                   flagged_per_doc=float(np.mean(flagged_per_doc)), surfaced_per_doc=float(np.mean(surfaced_per_doc)), by_labeler={})
        for name, p in per.items():
            res["by_labeler"][name] = dict(
                hall_doc_violation=float(np.mean(p["doc_hall_viol"])), hall_doc_violation_ci=boot_ci(p["doc_hall_viol"]),
                hall_n=p["hall_tot"], hall_recall=p["hall_flagged"] / p["hall_tot"] if p["hall_tot"] else np.nan,
                omit_frac_loss=float(np.mean(p["doc_omit_loss"])), omit_frac_loss_ci=boot_ci(p["doc_omit_loss"]),
                omit_n=p["omit_tot"], omit_recall=p["omit_surf"] / p["omit_tot"] if p["omit_tot"] else np.nan)
        out[ds] = res
    return out


out = {"normalized": run("normalized"), "exact": run("exact")}
json.dump(out, open(Path(__file__).with_suffix(".json") if mode() != "submitted" else Path(__file__).with_name(Path(__file__).stem + "_submitted.json"), "w"), indent=1, default=float)
for mode in ("normalized", "exact"):
    print(f"\n===== matching: {mode} =====")
    for ds, r in out[mode].items():
        m = r["matched"]
        print(f"\n{ds}: {r['n_docs']} docs ({m['n_test_docs']} in canonical test split); matched {m['gen']}/{m['gen_tot']} summary, "
              f"{m['src']}/{m['src_tot']} source sents; lambda={r['lam']}, (tau,gamma)=({r['tau']},{r['gamma']}); "
              f"flagged/doc={r['flagged_per_doc']:.1f}, surfaced/doc={r['surfaced_per_doc']:.1f}; "
              f"labeled hallucinations left unmatched O/A1/A2 = {'/'.join(str(v) for v in m['unmatched_hall'].values())}")
        print(f"  {'Ground truth':<16}{'#hall':>6}{'hall recall':>12}{'doc viol (any missed hall)':>30}{'#omit':>7}{'omit recall':>12}{'frac omit loss':>24}")
        for name, b in r["by_labeler"].items():
            print(f"  {name:<16}{b['hall_n']:>6}{100*b['hall_recall']:>11.0f}%{100*b['hall_doc_violation']:>12.0f}% [{100*b['hall_doc_violation_ci'][0]:.0f},{100*b['hall_doc_violation_ci'][1]:.0f}]"
                  f"{b['omit_n']:>13}{100*b['omit_recall']:>11.0f}%{100*b['omit_frac_loss']:>12.1f}% [{100*b['omit_frac_loss_ci'][0]:.0f},{100*b['omit_frac_loss_ci'][1]:.0f}]")
