"""R3 W4: document-level abstention baselines (paper Table 15), calibrated with CRC on the
actual binary document loss and evaluated over the same 100 resplits as the CARE row.

Loss for a document-abstention rule with aggregate score s(x) = agg_i p_sup(v_i) and threshold t:
    L(t) = 1{doc contains >=1 oracle-unsupported sentence AND s(x) > t}   (reject doc iff s(x) <= t)
L(t) is non-increasing in t for any aggregate (raising t can only add rejections), bounded by B = 1,
and L(1) = 0 because every aggregate of scores in [0, 1] is <= 1. So CRC (Angelopoulos et al. 2024)
applies to min AND mean support:
    t* = min{t : n/(n+1) * R_hat(t) + 1/(n+1) <= alpha}   =>   E[L_test(t*)] <= alpha.

Same calibration code path as paper/scripts/table_10_sent_vs_doc.py (submitted version on `main`),
same resplit protocol as paper/scripts/compute_resplit_cis.py (SEED 42, 100 x 70/30), alpha = 0.15.

Run: CARE_DATA_ROOT=/path/to/data python r3_w4_doc_abstention.py
"""
import json
import os
from pathlib import Path

import numpy as np

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT  # honours CARE_DATA_ROOT
DATASETS = {"ACI_Bench": "ACI-Bench", "MIMIC_IV_BHC": "MIMIC-BHC", "MIMIC_III_CXR": "MIMIC-CXR",
            "PRIV_DS": "Priv-DS", "SumPubMed": "SumPubMed"}
ALPHA, SEED, N_RESPLITS, CAL_FRAC, GRID_FACT = 0.15, 42, 100, 0.70, 0.01
AGGS = {"min": np.min, "mean": np.mean}


# ---- document-level abstention (identical logic to table_10_sent_vs_doc.py on main) ----
def doc_score(doc, agg):
    p = doc["factuality_probs"]
    return 1.0 if not p else float(agg(p))


def has_halluc(doc):
    return any(l == 0 for l in doc["factuality_labels"])


def calibrate_doc(cal, agg):
    s = np.array([doc_score(d, agg) for d in cal]); h = np.array([has_halluc(d) for d in cal])
    n = len(cal)
    for t in [0.0] + sorted(set(s.tolist())) + [1.0]:
        if (n / (n + 1)) * np.mean(h & (s > t)) + 1 / (n + 1) <= ALPHA:
            return t
    return None


def eval_doc(test, agg, t):
    s = np.array([doc_score(d, agg) for d in test]); h = np.array([has_halluc(d) for d in test])
    n_sent = np.array([len(d["factuality_probs"]) for d in test])
    rej = s <= t
    return dict(viol=float(np.mean(h & ~rej)), rej=float(rej.mean()),
                rej_given_halluc=float(rej[h].mean()) if h.any() else np.nan,
                rej_given_clean=float(rej[~h].mean()) if (~h).any() else np.nan,
                sent_discarded_pct=float(100 * n_sent[rej].sum() / n_sent.sum()))


# ---- CARE sentence-level hallucination controller (identical to compute_resplit_cis.py on main) ----
def fact_loss(doc, lam):
    lab, p = np.array(doc["factuality_labels"]), np.array(doc["factuality_probs"])
    if len(lab) == 0:
        return 0, 0
    flagged = p <= lam
    return int(((lab == 0) & ~flagged).any()), int(flagged.sum())


def calibrate_care(cal):
    n = len(cal)
    for lam in np.arange(0.0, 1.0 + GRID_FACT, GRID_FACT):
        if (n / (n + 1)) * np.mean([fact_loss(d, lam)[0] for d in cal]) + 1 / (n + 1) <= ALPHA:
            return float(lam)
    return 1.0


def eval_care(test, lam):
    L, F = zip(*[fact_loss(d, lam) for d in test])
    avg_sents = np.mean([len(d["factuality_probs"]) for d in test])
    return dict(viol=float(np.mean(L)), flag_pct=float(np.mean(F) / avg_sents * 100),
                docs_with_flag=float(np.mean(np.array(F) > 0)))


def monotonicity_violations(docs, agg):
    """Empirical check: per-document loss must never increase as t increases."""
    grid = np.linspace(0, 1, 201)
    bad = 0
    for d in docs:
        s, h = doc_score(d, agg), has_halluc(d)
        L = np.array([int(h and s > t) for t in grid])
        bad += int((np.diff(L) > 0).any())
    return bad


def summ(v):
    v = np.asarray(v, float)
    return dict(mean=float(np.nanmean(v)), ci=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))])


out = {}
for key, label in DATASETS.items():
    docs = [json.loads(l) for l in open(DATA_ROOT / key / "phase_2" / "calibrated_scores.jsonl")]
    by_id = {d["doc_id"]: d for d in docs}
    r = dict(n_docs=len(docs), halluc_doc_rate=float(np.mean([has_halluc(d) for d in docs])))

    # (a) canonical split: reproduces the submitted Table 15 doc rows
    split = json.load(open(DATA_ROOT / key / "split" / "split_indices.json"))
    cal = [by_id[i] for i in split["calibration_indices"] if i in by_id]
    test = [by_id[i] for i in split["test_indices"] if i in by_id]
    r["canonical"] = {}
    for a, f in AGGS.items():
        t = calibrate_doc(cal, f)
        r["canonical"][a] = dict(t=t, **eval_doc(test, f, t))

    # (b) 100 resplits, identical permutation stream to compute_resplit_cis.py
    n = len(docs); n_cal = int(n * CAL_FRAC)
    rng = np.random.default_rng(SEED)
    per = {a: [] for a in AGGS}; care = []
    for _ in range(N_RESPLITS):
        perm = rng.permutation(n)
        cal = [docs[j] for j in perm[:n_cal]]; test = [docs[j] for j in perm[n_cal:]]
        for a, f in AGGS.items():
            t = calibrate_doc(cal, f)
            per[a].append(dict(t=t, **eval_doc(test, f, t)))
        lam = calibrate_care(cal)
        care.append(dict(lam=lam, **eval_care(test, lam)))
    def agg_runs(v):
        d = {k: summ([x[k] for x in v]) for k in v[0]}
        d["frac_resplits_viol_gt_alpha"] = float(np.mean([x["viol"] > ALPHA for x in v]))
        return d
    r["resplits"] = {a: agg_runs(v) for a, v in per.items()}
    r["resplits"]["care_sentence"] = agg_runs(care)
    r["monotonicity_violations"] = {a: monotonicity_violations(docs, f) for a, f in AGGS.items()}
    out[key] = r

json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)

print("(a) Canonical split (submitted Table 15 doc rows): rejection % / test violation %")
for k, r in out.items():
    c = r["canonical"]
    print(f"  {DATASETS[k]:<10} min {100*c['min']['rej']:5.1f} / {100*c['min']['viol']:5.1f}   "
          f"mean {100*c['mean']['rej']:5.1f} / {100*c['mean']['viol']:5.1f}")
print("\n(b) 100 resplits, alpha = 0.15: mean [2.5, 97.5 pct]")
for k, r in out.items():
    R = r["resplits"]; f = lambda d, m: f"{100*d[m]['mean']:5.1f} [{100*d[m]['ci'][0]:4.1f}, {100*d[m]['ci'][1]:4.1f}]"
    print(f"  {DATASETS[k]:<10} (docs w/ halluc {100*r['halluc_doc_rate']:.0f}%, monotonicity violations "
          f"{r['monotonicity_violations']})")
    for a in AGGS:
        print(f"     doc {a:<4}: viol {f(R[a], 'viol')}  reject docs {f(R[a], 'rej')}  "
              f"sentences discarded {R[a]['sent_discarded_pct']['mean']:5.1f}%  "
              f"reject clean docs {100*R[a]['rej_given_clean']['mean']:5.1f}%")
    C = R["care_sentence"]
    print(f"     CARE sent: viol {f(C, 'viol')}  flag {C['flag_pct']['mean']:5.1f}% of sentences  "
          f"docs w/ >=1 flag {100*C['docs_with_flag']['mean']:5.1f}%")
