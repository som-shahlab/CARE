"""R3 Q4 / R3 W5 / R4: patient grouping and target-summary exclusion (E6).

1. Target exclusion: (a) section-header check of each source; (b) verbatim-leak test: share of reference
   sentences (>= 8 tokens) with a near-identical (token Jaccard >= 0.9) sentence in the source.
2. Patient grouping: the released derived datasets carry no subject/admission IDs, so we use a content proxy:
   pairs of documents that share >= 3 rare (appearing in < 1% of docs), long (>= 8 tokens) near-identical source
   sentences, as would happen for readmissions of the same patient with copied history.
   Paper results use 100 random 70/30 resplits, so any such pair is split across calibration/test in ~42% of resplits.
   The broad proxy is dominated by shared templates (discharge instructions, lab/consult boilerplate), so we also
   report a STRICT, patient-specific proxy:
     - MIMIC-BHC: >= 2 shared rare long sentences inside HISTORY OF PRESENT ILLNESS / PAST MEDICAL HISTORY /
       SOCIAL HISTORY, or past-medical-history token Jaccard >= 0.6 (PMH >= 10 tokens).
     - Priv-DS: a shared rare long sentence that is a patient introduction (age + "y.o."/"year old").
   Manual review (local only) showed the BHC shared-sentence criterion still catches review-of-systems templates,
   so the final LIKELY same-patient set is: BHC pairs with PMH Jaccard >= 0.6 and the same <SEX>; Priv-DS strict pairs.
3. Impact on the canonical split (seed 42, Table 9 thresholds, submitted loss definitions from care/calibration.py):
   test hallucination violation and mean fractional omission loss, with and without test docs that have a likely
   same-patient partner in calibration.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from care.config import CANONICAL_DATA_ROOT as ROOT  # honours CARE_DATA_ROOT
DATASETS = {"ACI-Bench": "ACI_Bench", "MIMIC-BHC": "MIMIC_IV_BHC", "MIMIC-CXR": "MIMIC_III_CXR", "Priv-DS": "PRIV_DS"}
MIN_TOK, JACC, RARE_FRAC, MIN_SHARED = 8, 0.9, 0.01, 3


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def best_match_mask(queries, pool):
    """For each query token set: does some pool set have Jaccard >= JACC?"""
    index = defaultdict(list)
    for j, p in enumerate(pool):
        for w in p:
            index[w].append(j)
    out = []
    for q in queries:
        cnt = Counter()
        for w in q:
            cnt.update(index.get(w, ()))
        out.append(any(c / (len(q) + len(pool[j]) - c) >= JACC for j, c in cnt.items()))
    return np.array(out, bool)


def leak_stats(rows):
    per_doc = []
    for r in rows:
        ref = [frozenset(toks(s)) for s in re.split(r"(?<=[.!?])\s+|\n+", r["reference_text"]) if len(toks(s)) >= MIN_TOK]
        src = [frozenset(toks(s)) for s in r["source_sentences"] if len(toks(s)) >= MIN_TOK]
        if ref and src:
            per_doc.append(best_match_mask(ref, src).mean())
    a = np.array(per_doc)
    return dict(n_docs=len(a), mean_ref_sent_verbatim_in_source=float(a.mean()),
                median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                docs_over_50pct=int((a > 0.5).sum()), docs_over_25pct=int((a > 0.25).sum()))


def patient_proxy(rows):
    sets = [{frozenset(toks(s)) for s in r["source_sentences"] if len(toks(s)) >= MIN_TOK} for r in rows]
    # exact-normalized long sentences; rare = in < RARE_FRAC of docs
    df = Counter(s for ss in sets for s in ss)
    max_df = max(2, int(RARE_FRAC * len(rows)))
    owners = defaultdict(list)
    for d, ss in enumerate(sets):
        for s in ss:
            if 2 <= df[s] <= max_df:
                owners[s].append(d)
    shared = Counter()
    for s, ds in owners.items():
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                shared[(ds[i], ds[j])] += 1
    pairs = {k: v for k, v in shared.items() if v >= MIN_SHARED}
    docs = {d for p in pairs for d in p}
    return dict(n_docs=len(rows), n_pairs=len(pairs), n_docs_in_pairs=len(docs),
                frac_docs_in_pairs=len(docs) / len(rows),
                pairs=sorted([[int(rows[a]["doc_id"]), int(rows[b]["doc_id"]), int(v)] for (a, b), v in pairs.items()],
                             key=lambda x: -x[2]))  # [doc_id, doc_id, n shared rare sentences]


def canonical_cross_split(ds_dir, rows, pairs):
    p = ROOT / ds_dir / "split" / "split_indices.json"
    if not p.exists():
        p = next((ROOT / ds_dir / "split").glob("*/split_indices.json"), p)
    if not p.exists():
        return None
    idx = json.load(open(p))
    cal = set(idx.get("calibration_indices", []))  # indices are doc_ids (care/config.py)
    return sum((a in cal) != (b in cal) for a, b, _ in pairs)


SECTION_RE = re.compile(r"<([A-Z][A-Z /&()\-]{2,60})>")
AGE_RE = re.compile(r"\b\d{1,3}\s*(y\.?o\.?|yo|year[- ]old|years old|yr old)\b", re.I)
PATIENT_SECTIONS = {"HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY", "SOCIAL HISTORY"}


def sentences(txt):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", txt) if len(toks(s)) >= MIN_TOK]


def bhc_sections(txt):
    parts = SECTION_RE.split(txt)
    d = defaultdict(str)
    for i in range(1, len(parts) - 1, 2):
        d[parts[i]] += " " + parts[i + 1]
    return d


def strict_pairs(name, rows):
    by_id = {r["doc_id"]: r for r in rows}
    if name == "MIMIC-BHC":
        secs = {i: bhc_sections(r["source_text"]) for i, r in by_id.items()}
        pat = {i: {frozenset(toks(x)) for k in PATIENT_SECTIONS for x in sentences(s.get(k, ""))} for i, s in secs.items()}
        pmh = {i: set(toks(s.get("PAST MEDICAL HISTORY", ""))) for i, s in secs.items()}
    elif name == "Priv-DS":
        pat = {i: {frozenset(toks(x)) for x in sentences(r["source_text"]) if AGE_RE.search(x)} for i, r in by_id.items()}
        pmh = None
    else:
        return []
    df = Counter(x for v in pat.values() for x in v)
    max_df = max(2, int(RARE_FRAC * len(rows)))
    owners = defaultdict(list)
    for i, v in pat.items():
        for x in v:
            if 2 <= df[x] <= max_df:
                owners[x].append(i)
    shared = Counter()
    for ds in owners.values():
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                shared[tuple(sorted((ds[a], ds[b])))] += 1
    need = 2 if name == "MIMIC-BHC" else 1
    pairs = {k for k, v in shared.items() if v >= need}
    if pmh is not None:
        ids = sorted(pmh)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                A, B = pmh[ids[a]], pmh[ids[b]]
                if len(A) >= 10 and len(B) >= 10 and len(A & B) / len(A | B) >= 0.6:
                    pairs.add((ids[a], ids[b]))
    return sorted([int(a), int(b), int(shared.get((a, b), 0))] for a, b in pairs)


import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from care_thresholds import thresholds, mode
TABLE9 = {k: v for k, v in thresholds().items() if k != "SumPubMed"}  # (lambda, tau, gamma); camera-ready by default


def likely_pairs(name, rows, strict):
    if name == "MIMIC-BHC":
        by_id = {r["doc_id"]: bhc_sections(r["source_text"]) for r in rows}
        keep = []
        for a, b, n in strict:
            A, B = set(toks(by_id[a].get("PAST MEDICAL HISTORY", ""))), set(toks(by_id[b].get("PAST MEDICAL HISTORY", "")))
            same_sex = by_id[a].get("SEX", "").strip() == by_id[b].get("SEX", "").strip()
            if len(A) >= 10 and len(B) >= 10 and len(A & B) / len(A | B) >= 0.6 and same_sex:
                keep.append([a, b, n])
        return keep
    return strict


def split_ids(ds_dir):
    p = ROOT / ds_dir / "split" / "split_indices.json"
    if not p.exists():
        p = next((ROOT / ds_dir / "split").glob("*/split_indices.json"), p)
    idx = json.load(open(p))
    return set(idx["calibration_indices"]), set(idx["test_indices"])


def test_risks(rows, lam, tau, gamma, test_ids):
    hall, omit = [], []
    for r in rows:
        if r["doc_id"] not in test_ids:
            continue
        f, pf = np.array(r["factuality_labels"]), np.array(r["factuality_probs"])
        hall.append(int(((f == 0) & (pf > lam)).any()))
        t = (np.array(r["importance_labels"]) == 1) & (np.array(r["coverage_labels"]) == 0)
        surf = (np.array(r["importance_probs"]) >= tau) & (1 - np.array(r["coverage_probs"]) >= gamma)
        omit.append(float((t & ~surf).sum() / t.sum()) if t.sum() else 0.0)
    return dict(n_test=len(hall), hall_violation=float(np.mean(hall)), omit_frac_loss=float(np.mean(omit)))


out = {}
for name, d in DATASETS.items():
    rows = [json.loads(l) for l in open(ROOT / d / "phase_2" / "calibrated_scores.jsonl")]
    rows.sort(key=lambda r: r["doc_id"])
    assert len({r["doc_id"] for r in rows}) == len(rows), name
    leak = leak_stats(rows)
    prox = patient_proxy(rows)
    prox["canonical_split_cross_pairs"] = canonical_cross_split(d, rows, prox["pairs"])
    sp = strict_pairs(name, rows)
    strict = dict(n_pairs=len(sp), n_docs=len({x for a, b, _ in sp for x in (a, b)}), pairs=sp,
                  canonical_split_cross_pairs=canonical_cross_split(d, rows, sp))
    lp = likely_pairs(name, rows, sp)
    cal, test = split_ids(d)
    excl = {b if b in test else a for a, b, _ in lp if (a in cal) != (b in cal)}
    likely = dict(n_pairs=len(lp), n_docs=len({x for a, b, _ in lp for x in (a, b)}), pairs=lp,
                  canonical_split_cross_pairs=canonical_cross_split(d, rows, lp), excluded_test_docs=sorted(excl),
                  canonical_all=test_risks(rows, *TABLE9[name], test),
                  canonical_excluding=test_risks(rows, *TABLE9[name], test - excl))
    out[name] = dict(target_leak=leak, patient_proxy=prox, patient_proxy_strict=strict, likely_same_patient=likely)
    print(f"{name:<10} ref sents verbatim in source: mean {leak['mean_ref_sent_verbatim_in_source']:.1%}, "
          f"median {leak['median']:.1%}, p95 {leak['p95']:.1%}, docs >25% {leak['docs_over_25pct']}, >50% {leak['docs_over_50pct']} "
          f"| proxy same-patient pairs {prox['n_pairs']} ({prox['n_docs_in_pairs']} docs, {prox['frac_docs_in_pairs']:.1%}); "
          f"cross canonical split {prox['canonical_split_cross_pairs']} | STRICT pairs {strict['n_pairs']} "
          f"({strict['n_docs']} docs), cross canonical split {strict['canonical_split_cross_pairs']}")
json.dump(out, open(Path(__file__).with_suffix(".json") if mode() != "submitted" else Path(__file__).with_name(Path(__file__).stem + "_submitted.json"), "w"), indent=1)

print("\nLikely same-patient pairs and canonical-split impact (Table 9 thresholds)")
for name, o in out.items():
    l = o["likely_same_patient"]; a, e = l["canonical_all"], l["canonical_excluding"]
    print(f"{name:<10} likely pairs {l['n_pairs']} ({l['n_docs']} docs), cross canonical split {l['canonical_split_cross_pairs']} | "
          f"test hall viol {a['hall_violation']:.3f} (n={a['n_test']}) -> {e['hall_violation']:.3f} (n={e['n_test']}); "
          f"omit frac {a['omit_frac_loss']:.3f} -> {e['omit_frac_loss']:.3f}")
json.dump(out, open(Path(__file__).with_suffix(".json") if mode() != "submitted" else Path(__file__).with_name(Path(__file__).stem + "_submitted.json"), "w"), indent=1)
