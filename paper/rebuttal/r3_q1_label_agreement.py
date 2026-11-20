"""R3 Q1: oracle-clinician agreement with the clinically relevant class as positive,
including the joint omission label (important AND not covered).

Labels are evaluated at the granularity each CARE loss uses:
  - omission loss: fraction of missed (important AND not covered) source sentences -> sentence level
  - hallucination loss: 1{>=1 unflagged unsupported sentence in the doc} -> also document level
Also reports agreement on clinician-agreed sentences and document-bootstrap 95% CIs.

Uses the same annotation CSVs and the same row filtering as table_4_oracle_validation.py
(rows with a non-binary label from the oracle or either annotator are dropped per task).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, confusion_matrix

D = Path(__file__).resolve().parents[1] / "clinician_study"
FILES = {"ACI-Bench": "ACI (3)", "MIMIC-BHC": "BHC (1)", "MIMIC-CXR": "CXR"}
WHO = ["ORACLE", "ANNOTATOR_1", "ANNOTATOR_2"]
N_BOOT, SEED = 2000, 0


def load(name):
    df = pd.read_csv(D / f"annotation_validation_corrected.xlsx - {name}.csv", usecols=range(14))
    for c in df.columns:
        if c.startswith(("FACT_", "IMP_", "COV_")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[~df[c].isin([0, 1]), c] = np.nan
    return df


def pair_stats(y_ref, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_ref, y_pred, labels=[0, 1]).ravel()
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                f1=f1_score(y_ref, y_pred, zero_division=0),
                prec=precision_score(y_ref, y_pred, zero_division=0),
                rec=recall_score(y_ref, y_pred, zero_division=0),
                kappa=cohen_kappa_score(y_ref, y_pred))


def _f1_rec(y_ref, y_pred):
    tp = int((y_ref & y_pred).sum()); fp = int((~y_ref & y_pred).sum()); fn = int((y_ref & ~y_pred).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return f1, (tp / (tp + fn) if tp + fn else np.nan)


def doc_bootstrap(labels, docs):
    """95% CIs for OH F1, OH recall and HH F1, resampling documents with replacement."""
    a1, a2, o = (labels[w].astype(bool) for w in ("ANNOTATOR_1", "ANNOTATOR_2", "ORACLE"))
    uniq = np.unique(docs)
    idx = {d: np.flatnonzero(docs == d) for d in uniq}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
        (f1a, ra), (f1b, rb) = _f1_rec(a1[rows], o[rows]), _f1_rec(a2[rows], o[rows])
        draws.append(((f1a + f1b) / 2, (ra + rb) / 2, _f1_rec(a1[rows], a2[rows])[0]))
    draws = np.array(draws)
    ci = lambda c: [float(np.nanpercentile(c, 2.5)), float(np.nanpercentile(c, 97.5))]
    return dict(oh_f1=ci(draws[:, 0]), oh_rec=ci(draws[:, 1]), hh_f1=ci(draws[:, 2]))


def consensus(labels):
    """Oracle vs the clinician label on sentences where both clinicians agree."""
    a1, a2, o = labels["ANNOTATOR_1"], labels["ANNOTATOR_2"], labels["ORACLE"]
    m = a1 == a2
    s = pair_stats(a1[m], o[m])
    either = (a1 == 1) | (a2 == 1)
    return dict(n_agreed=int(m.sum()), n_agreed_pos=int(a1[m].sum()), **s,
                oracle_pos_flagged_by_a_clinician=int((o[either] == 1).sum()),
                oracle_pos_flagged_by_neither=int(((o == 1) & ~either).sum()))


def doc_level(labels, docs):
    """Agreement on 'document contains >=1 positive' (the unit of the hallucination loss)."""
    d = pd.DataFrame({w: labels[w] for w in WHO}).groupby(docs).max()
    a1, a2, o = d["ANNOTATOR_1"].values, d["ANNOTATOR_2"].values, d["ORACLE"].values
    both = (a1 == 1) & (a2 == 1)
    return dict(n_docs=len(d), n_pos_docs={w: int(d[w].sum()) for w in WHO},
                hh_agree=int((a1 == a2).sum()), oh_agree_a1=int((o == a1).sum()), oh_agree_a2=int((o == a2).sum()),
                both_clinicians_pos=int(both.sum()), oracle_flags_of_both=int(o[both].sum()))


def task_table(labels, docs):
    """labels: dict who -> 0/1 array with POSITIVE = clinically relevant class; docs: DOC_ID per row."""
    a1, a2, o = labels["ANNOTATOR_1"], labels["ANNOTATOR_2"], labels["ORACLE"]
    hh = pair_stats(a1, a2)
    oa1, oa2 = pair_stats(a1, o), pair_stats(a2, o)
    avg = {k: (oa1[k] + oa2[k]) / 2 for k in ("f1", "prec", "rec", "kappa")}
    return dict(n=len(o), n_pos={w: int(v.sum()) for w, v in labels.items()},
                hh=hh, oracle_vs_a1=oa1, oracle_vs_a2=oa2, oracle_avg=avg,
                boot_ci=doc_bootstrap(labels, docs), consensus=consensus(labels), doc_level=doc_level(labels, docs))


out = {}
for ds, f in FILES.items():
    df = load(f)
    gen, src = df[df.SENTENCE_TYPE == "generated"], df[df.SENTENCE_TYPE == "source"]
    res = {}
    # Hallucination: positive = unsupported (FACT == 0)
    g = gen.dropna(subset=[f"FACT_{w}" for w in WHO])
    res["hallucination (FACT=0)"] = task_table({w: (g[f"FACT_{w}"] == 0).astype(int).values for w in WHO}, g.DOC_ID.values)
    # Importance: positive = important
    s = src.dropna(subset=[f"IMP_{w}" for w in WHO])
    res["important (IMP=1)"] = task_table({w: (s[f"IMP_{w}"] == 1).astype(int).values for w in WHO}, s.DOC_ID.values)
    # Non-coverage: positive = not covered
    s = src.dropna(subset=[f"COV_{w}" for w in WHO])
    res["not covered (COV=0)"] = task_table({w: (s[f"COV_{w}"] == 0).astype(int).values for w in WHO}, s.DOC_ID.values)
    # Joint omission: positive = important AND not covered (each labeler's own IMP and COV)
    s = src.dropna(subset=[f"{t}_{w}" for t in ("IMP", "COV") for w in WHO])
    res["omission (IMP=1 & COV=0)"] = task_table(
        {w: ((s[f"IMP_{w}"] == 1) & (s[f"COV_{w}"] == 0)).astype(int).values for w in WHO}, s.DOC_ID.values)
    out[ds] = dict(n_docs=int(df.DOC_ID.nunique()), tasks=res)

json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1, default=float)

print(f"{'Dataset':<10} {'Task (positive class)':<26} {'n':>5} {'#pos O/A1/A2':>14} | {'HH F1':>6} {'HH κ':>6} | "
      f"{'OH F1':>6} {'OH P':>6} {'OH R':>6} {'OH κ':>6} | oracle-vs-A1 TP/FP/FN")
for ds, r in out.items():
    for t, s in r["tasks"].items():
        p = s["n_pos"]; a = s["oracle_avg"]; c = s["oracle_vs_a1"]
        print(f"{ds:<10} {t:<26} {s['n']:>5} {p['ORACLE']:>4}/{p['ANNOTATOR_1']:>3}/{p['ANNOTATOR_2']:>3}  | "
              f"{s['hh']['f1']:6.2f} {s['hh']['kappa']:6.2f} | {a['f1']:6.2f} {a['prec']:6.2f} {a['rec']:6.2f} {a['kappa']:6.2f} | "
              f"{c['tp']}/{c['fp']}/{c['fn']}")
    print(f"{'':10} docs = {r['n_docs']}")

print("\nControlled labels: document-bootstrap 95% CIs, clinician-agreed sentences, document level")
for ds, r in out.items():
    for t in ("hallucination (FACT=0)", "omission (IMP=1 & COV=0)"):
        s = r["tasks"][t]; b = s["boot_ci"]; c = s["consensus"]; d = s["doc_level"]
        fmt = lambda v, ci: f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
        print(f"{ds:<10} {t:<26} OH F1 {fmt(s['oracle_avg']['f1'], b['oh_f1'])}  OH R {fmt(s['oracle_avg']['rec'], b['oh_rec'])}  "
              f"HH F1 {fmt(s['hh']['f1'], b['hh_f1'])}")
        print(f"{'':37} agreed n={c['n_agreed']}, pos={c['n_agreed_pos']}: oracle F1 {c['f1']:.2f}, "
              f"recall {c['tp']}/{c['n_agreed_pos']}, FP {c['fp']}")
        print(f"{'':37} docs with >=1 positive A1/A2/O = {d['n_pos_docs']['ANNOTATOR_1']}/{d['n_pos_docs']['ANNOTATOR_2']}/"
              f"{d['n_pos_docs']['ORACLE']} of {d['n_docs']}; agree HH {d['hh_agree']}, OH {d['oh_agree_a1']} & {d['oh_agree_a2']}; "
              f"oracle flags {d['oracle_flags_of_both']}/{d['both_clinicians_pos']} docs both clinicians call positive")
