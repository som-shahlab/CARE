"""R4 / R5: sensitivity of the hallucination controller to oracle-label errors.

Setup. Treat the current oracle factuality labels as ground truth. Corrupt the CALIBRATION labels with
a noise model, recalibrate lambda with the submitted CRC procedure (paper/scripts/compute_resplit_cis.py:
calibrate_factuality, alpha = 0.15, grid 0.01, infimum rule), and score the deployed lambda on the TEST
split against the clean labels. Same 100 resplits (seed 42, 70/30) and five datasets as paper Table 2.

Noise models, PRE-SPECIFIED before any noisy run (rates anchored to E1, r3_q1_label_agreement.json:
oracle miss rate vs clinicians 0.17-0.54, oracle false-positive rate on supported sentences 1.4-4.1%):
  (a) symmetric:  every sentence label flipped independently w.p. eps in {0.01, 0.02, 0.05, 0.10}
  (b) missed:     every true hallucination relabeled "supported" w.p. m in {0.1, ..., 0.7}
                  (oracle recall 0.9 ... 0.3; E1 range is recall 0.46-0.83)
  (c) false pos.: every supported sentence relabeled "hallucinated" w.p. f in {0.01, 0.02, 0.04}
  (d) E1-matched: missed + false positives at the E1 rates (averaged over the two clinicians):
                    ACI-matched m = 0.52, f = 0.017;  BHC-matched m = 0.33, f = 0.039;
                    pessimistic m = 0.54, f = 0.014 (worst miss, fewest compensating FPs)
                  each applied to every dataset.
  (e) correlated stress test (not a realistic model; worst case for CRC): same rates as (d), but the
      missed hallucinations are the ones the scorer rates MOST supported (highest p_fact) and the false
      positives are the supported sentences the scorer rates LEAST supported (lowest p_fact), i.e. oracle
      and scorer make the same mistakes. Counts per document are drawn as in (d); only WHICH sentences
      are corrupted changes.
Noise is redrawn independently for each resplit (seeded per setting and resplit).

POST-HOC additions (added after seeing (a)-(e) and the error structure on the clinician-annotated docs,
see noise_structure below; reported as such):
  (f) E1-structured: rates as in (d); a share rho of the missed hallucinations is placed on the
      highest-p_fact hallucinations (observed on matched annotation sentences: 7/19 oracle misses on ACI had
      p_fact = 1.0 -> rho = 0.37; BHC 1/7 -> rho = 0.14), the rest are random; false positives are placed on
      supported sentences with p_fact in [0.4, 0.6] (where 26/34 observed oracle false positives lie).
  (g) Noise-adjusted calibration: calibrate at alpha' = alpha * (1 - m). If each hallucination label is missed
      independently w.p. <= m, then E[noisy doc loss | doc, lambda] >= (1 - m) * clean doc loss (a doc that
      fails has >= 1 unflagged hallucination, each kept w.p. >= 1 - m; false positives only raise the noisy
      loss), so CRC on noisy labels at alpha' gives E[clean loss] <= alpha' / (1 - m) = alpha.

Outputs per (dataset, setting): mean / 2.5-97.5 percentile across resplits of the clean-label test
violation, lambda, flagged per doc, and clean-label recall; plus the fraction of resplits whose test
violation exceeds alpha.

Mechanism check: the per-document loss is 1{max p_fact over hallucinations > lambda}; missed labels
only lower a calibration document's loss when they hit its highest-scored hallucination (or its only one).
We report, per dataset, the share of hallucinated documents with >= 2 hallucinations.

Run:  CARE_DATA_ROOT=/path/to/data python paper/rebuttal/r4_oracle_noise.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from care.config import CANONICAL_DATA_ROOT as DATA_ROOT  # noqa: E402
from paper.scripts.compute_resplit_cis import (  # noqa: E402  submitted procedure (unchanged vs main)
    ALPHA_FACT, CAL_FRAC, GRID_FACT, N_RESPLITS, SEED, calibrate_factuality, compute_factuality_loss)

DATASETS = ["ACI_Bench", "MIMIC_IV_BHC", "MIMIC_III_CXR", "PRIV_DS", "SumPubMed"]
THRESHOLDS = np.arange(0.0, 1.0 + GRID_FACT, GRID_FACT)  # identical grid to calibrate_factuality

E1_RATES = {"ACI-matched": (0.52, 0.017), "BHC-matched": (0.33, 0.039), "pessimistic": (0.54, 0.014)}
SETTINGS = [("clean", "none", 0.0, 0.0)]
SETTINGS += [(f"symmetric eps={e}", "sym", e, e) for e in (0.01, 0.02, 0.05, 0.10)]
SETTINGS += [(f"missed m={m}", "iid", m, 0.0) for m in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
SETTINGS += [(f"false-pos f={f}", "iid", 0.0, f) for f in (0.01, 0.02, 0.04)]
SETTINGS += [(f"E1 {k} (m={m}, f={f})", "iid", m, f) for k, (m, f) in E1_RATES.items()]
SETTINGS += [(f"correlated {k} (m={m}, f={f})", "corr", m, f) for k, (m, f) in E1_RATES.items()]
# --- post hoc (see docstring) ---
RHO = {"ACI-matched": 0.37, "BHC-matched": 0.14, "pessimistic": 0.37}
SETTINGS += [(f"structured {k} (m={m}, f={f}, rho={RHO[k]})", f"struct{RHO[k]}", m, f) for k, (m, f) in E1_RATES.items()]
ADJUSTED = [("missed m=0.3", 0.3), ("missed m=0.5", 0.5), ("E1 ACI-matched (m=0.52, f=0.017)", 0.52),
            ("E1 pessimistic (m=0.54, f=0.014)", 0.54), ("correlated pessimistic (m=0.54, f=0.014)", 0.54),
            ("structured ACI-matched (m=0.52, f=0.017, rho=0.37)", 0.52),
            ("structured pessimistic (m=0.54, f=0.014, rho=0.37)", 0.54)]


def corrupt(y, p, kind, m, f, rng):
    """y: 0/1 labels (0 = hallucinated), p: p_fact. Returns the noisy label array."""
    y = np.asarray(y).copy()
    if kind == "none" or len(y) == 0:
        return y
    hal, sup = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    k_miss = rng.binomial(len(hal), m) if m > 0 else 0   # same count distribution for iid / sym / corr
    k_fp = rng.binomial(len(sup), f) if f > 0 else 0
    if kind == "corr":
        miss = hal[np.argsort(-p[hal], kind="stable")[:k_miss]]
        fp = sup[np.argsort(p[sup], kind="stable")[:k_fp]]
    elif kind.startswith("struct"):
        k_top = rng.binomial(k_miss, float(kind[6:]))
        top = hal[np.argsort(-p[hal], kind="stable")[:k_top]]
        rest = np.setdiff1d(hal, top)
        miss = np.concatenate([top, rng.choice(rest, k_miss - k_top, replace=False)]).astype(int)
        band = sup[(p[sup] >= 0.4) & (p[sup] <= 0.6)]
        fp = rng.choice(band, min(k_fp, len(band)), replace=False)
    else:
        miss = rng.choice(hal, k_miss, replace=False)
        fp = rng.choice(sup, k_fp, replace=False)
    y[miss], y[fp] = 1, 0
    return y


def crit(y, p):
    """Doc loss at lambda is 1{crit > lambda}: crit = max p_fact over hallucinated sentences (-1 if none)."""
    h = p[np.asarray(y) == 0]
    return h.max() if h.size else -1.0


def calibrate_fast(crits, alpha=ALPHA_FACT):
    """Vectorized calibrate_factuality: smallest grid lambda with (n/(n+1)) R + 1/(n+1) <= alpha, else 1.0."""
    n = len(crits)
    risk = (crits[None, :] > THRESHOLDS[:, None]).mean(1)
    ok = np.flatnonzero((n / (n + 1)) * risk + 1 / (n + 1) <= alpha)
    return float(THRESHOLDS[ok[0]]) if ok.size else 1.0


def summ(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))


def run(ds):
    docs = [json.loads(l) for l in open(DATA_ROOT / ds / "phase_2" / "calibrated_scores.jsonl")]
    Y = [np.asarray(d["factuality_labels"]) for d in docs]
    P = [np.asarray(d["factuality_probs"], float) for d in docs]
    n, n_cal = len(docs), int(len(docs) * CAL_FRAC)
    perms, rng = [], np.random.default_rng(SEED)  # replay the paper's split sequence exactly
    for _ in range(N_RESPLITS):
        perms.append(rng.permutation(n))
    nh = np.array([(y == 0).sum() for y in Y])
    out = dict(n_docs=n, halluc_doc_share=float((nh > 0).mean()),
               multi_halluc_share_of_halluc_docs=float((nh >= 2).sum() / max((nh > 0).sum(), 1)), settings={})
    n_checked = 0
    jobs = [(si, name, kind, m, f, ALPHA_FACT, name) for si, (name, kind, m, f) in enumerate(SETTINGS)]
    by_name = {name: (si, kind, m, f) for si, (name, kind, m, f) in enumerate(SETTINGS)}
    for base, m_adj in ADJUSTED:  # same noise draws as the base setting (same seed), alpha' = alpha (1 - m)
        si, kind, m, f = by_name[base]
        jobs.append((si, base, kind, m, f, ALPHA_FACT * (1 - m_adj), f"{base} | adjusted alpha'={ALPHA_FACT * (1 - m_adj):.4f}"))
    for si, name, kind, m, f, alpha, label in jobs:
        viol, lam_l, flag_l, rec_l = [], [], [], []
        for i, perm in enumerate(perms):
            cal, test = perm[:n_cal], perm[n_cal:]
            nrng = np.random.default_rng([SEED, si, i])
            y_cal = {j: corrupt(Y[j], P[j], kind, m, f, nrng) for j in cal}
            lam = calibrate_fast(np.array([crit(y_cal[j], P[j]) for j in cal]), alpha)
            if i == 0 or (kind == "none"):  # check against the submitted implementation
                ref = calibrate_factuality([dict(factuality_labels=y_cal[j], factuality_probs=P[j]) for j in cal],
                                           alpha)
                assert ref == lam, (ds, name, i, ref, lam)
                n_checked += 1
            losses, flagged, tp, tot = zip(*[compute_factuality_loss(docs[j], lam) for j in test])  # clean labels
            viol.append(np.mean(losses)); lam_l.append(lam); flag_l.append(np.mean(flagged))
            rec_l.append(sum(tp) / sum(tot) if sum(tot) else 1.0)
        out["settings"][label] = dict(kind=kind, m=m, f=f, alpha_cal=alpha, viol=summ(viol), lam=summ(lam_l), flagged=summ(flag_l),
                                     recall=summ(rec_l), frac_resplits_viol_gt_alpha=float(np.mean(np.array(viol) > ALPHA_FACT)),
                                     per_resplit_viol=[float(v) for v in viol])
    out["n_calibrations_checked_against_submitted_code"] = n_checked
    return out


def noise_structure():
    """Where do oracle-clinician disagreements fall on p_fact? Annotation sentences matched to phase_2 by text
    (as in r4_human_ground_truth.py); lambda = submitted Table 9 value."""
    import re
    import pandas as pd
    sets = {"ACI-Bench": ("ACI_Bench", "ACI (3)", 0.70), "MIMIC-BHC": ("MIMIC_IV_BHC", "BHC (1)", 0.60),
            "MIMIC-CXR": ("MIMIC_III_CXR", "CXR", 0.50)}
    cd = ROOT / "paper" / "clinician_study"
    if not cd.exists():  # clinician annotations are not distributed (may contain PHI)
        print(f"skip noise_structure: {cd} not present")
        return None
    out = {}
    for ds, (folder, f, lam) in sets.items():
        df = pd.read_csv(cd / f"annotation_validation_corrected.xlsx - {f}.csv", usecols=range(14))
        for c in [c for c in df.columns if c.startswith("FACT_")]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        scores = {d["doc_id"]: d for d in map(json.loads, open(DATA_ROOT / folder / "phase_2" / "calibrated_scores.jsonl"))}
        rows = []
        for did in df.DOC_ID.dropna().unique():
            d = scores[int(re.search(r"\d+", did).group())]
            gidx = {s.strip(): i for i, s in enumerate(d["generated_sentences"])}
            g = df[(df.DOC_ID == did) & (df.SENTENCE_TYPE == "generated")]
            g = g[g.SENTENCE.astype(str).str.strip().isin(gidx)]
            rows += [(r.FACT_ORACLE, r.FACT_ANNOTATOR_1, r.FACT_ANNOTATOR_2, d["factuality_probs"][gidx[str(r.SENTENCE).strip()]])
                     for r in g.itertuples()]
        a = np.array(rows, float)
        a = a[~np.isnan(a[:, :3]).any(1)]
        o, p = a[:, 0], a[:, 3]
        cats = {}
        for name, k in (("A1", 1), ("A2", 2)):
            c = a[:, k]
            for cat, mk in (("agreed_halluc", (o == 0) & (c == 0)), ("oracle_miss", (o == 1) & (c == 0)),
                            ("oracle_fp", (o == 0) & (c == 1)), ("agreed_supported", (o == 1) & (c == 1))):
                cats[f"{cat}_vs_{name}"] = dict(n=int(mk.sum()), flagged_at_lambda=float(np.mean(p[mk] <= lam)) if mk.sum() else None,
                                                p_values=sorted(np.round(p[mk], 2).tolist()) if cat != "agreed_supported" else None,
                                                median_p=float(np.median(p[mk])) if mk.sum() else None)
        out[ds] = dict(lam=lam, n_matched=len(a), categories=cats)
    return out


if __name__ == "__main__":
    res = {ds: run(ds) for ds in DATASETS}
    struct = noise_structure()
    json.dump(dict(alpha=ALPHA_FACT, n_resplits=N_RESPLITS, settings=list(res[DATASETS[0]]["settings"]), results=res,
                   noise_structure=struct),
              open(Path(__file__).with_suffix(".json"), "w"), indent=1)
    short = {"ACI_Bench": "ACI", "MIMIC_IV_BHC": "BHC", "MIMIC_III_CXR": "CXR", "PRIV_DS": "PrivDS", "SumPubMed": "PubMed"}
    print(f"Clean-label test violation, mean over {N_RESPLITS} resplits (alpha = {ALPHA_FACT}); "
          f"[2.5, 97.5] percentile of resplits; flagged/doc")
    print(f"{'setting':<60}" + "".join(f"{short[d]:>24}" for d in DATASETS))
    for name in res[DATASETS[0]]["settings"]:
        row = ""
        for d in DATASETS:
            s = res[d]["settings"][name]
            row += f"{s['viol']['mean']*100:6.1f} [{s['viol']['lo']*100:4.1f},{s['viol']['hi']*100:5.1f}] {s['flagged']['mean']:4.1f}"
        print(f"{name[:60]:<60}{row}")
    for d in DATASETS:
        r = res[d]
        print(f"{short[d]}: hallucinated docs {r['halluc_doc_share']:.0%}; of those with >=2 hallucinations "
              f"{r['multi_halluc_share_of_halluc_docs']:.0%}; calibrations checked vs submitted code: "
              f"{r['n_calibrations_checked_against_submitted_code']}")
    print("\nWhere oracle-clinician disagreements fall (summary sentences matched to phase_2; lambda = Table 9):")
    for ds, r in (struct or {}).items():
        for cat, v in r["categories"].items():
            if v["n"]:
                print(f"  {ds:<10} {cat:<24} n={v['n']:>4}  flagged at lambda={r['lam']}: {v['flagged_at_lambda']:.2f}  "
                      f"median p_fact {v['median_p']:.2f}")
