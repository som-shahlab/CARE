"""E10 / R1 W2: do off-the-shelf hallucination detectors at their default threshold control the
document-level hallucination risk that CARE certifies? Pre-registration: notes/E10_prereg.md.

Convention (identical to the paper): score high = supported; a summary sentence is flagged iff score <= threshold.
Loss per doc = 1{any oracle-hallucinated sentence is unflagged}. alpha = 0.15.

Rows:
  default : fixed threshold 0.5, no calibration (NLI entailment prob, AlignScore prob, LLM-judge vote fraction).
  CRC     : the same scorer calibrated by CARE (from the submitted outputs), plus our recomputation as a check.
Test sets: the paper's 100 resplits (70/30). External scorers use the external-scorer script's RNG
(random.Random(42).shuffle) and the LLM judge uses compute_resplit_cis.py's RNG (np default_rng(42).permutation),
so each default row is evaluated on exactly the test sets of the CRC row it is compared with.

Scores: LLM judge from <data>/<ds>/phase_2/calibrated_scores.jsonl; NLI / AlignScore from
cache/<ds>__{nli,alignscore}_fact.json (r1_w2_score_factuality.py, submitted scoring code);
BERT CLS cosine from <data>/<ds>/external_scores/embed_only.json (range only: not a probability).
"""
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path(os.environ.get("CARE_DATA_ROOT", REPO.parent))
DATASETS = {"ACI_Bench": "ACI-Bench", "MIMIC_III_CXR": "MIMIC-CXR", "MIMIC_IV_BHC": "MIMIC-BHC"}
ALPHA, DEFAULT, SEED, N_RESPLITS, CAL_FRAC, GRID = 0.15, 0.5, 42, 100, 0.70, 0.01


def git_json(path):
    return json.loads(subprocess.run(["git", "show", f"main:{path}"], cwd=REPO, capture_output=True,
                                     text=True, check=True).stdout)


def doc_stats(probs, labels, lam):
    flagged, halluc = probs <= lam, labels == 0
    return int((halluc & ~flagged).any()), int(flagged.sum()), int((flagged & halluc).sum()), int(halluc.sum()), len(probs)


def evaluate(docs, lam):
    s = np.array([doc_stats(p, y, lam) for p, y in docs])
    return dict(viol=s[:, 0].mean(), flagged_per_doc=s[:, 1].mean(),
                flagged_pct=100 * s[:, 1].sum() / s[:, 4].sum(),
                recall=s[:, 2].sum() / s[:, 3].sum() if s[:, 3].sum() else np.nan)


def crc_lambda(cal):
    """CARE's CRC rule for the hallucination controller (paper / submitted code)."""
    n = len(cal)
    for lam in np.arange(0, 1 + GRID / 2, GRID):
        if (n / (n + 1)) * np.mean([doc_stats(p, y, lam)[0] for p, y in cal]) + 1 / (n + 1) <= ALPHA:
            return float(lam)
    return 1.0


def splits(n, rng_kind):
    n_cal = int(n * CAL_FRAC)
    if rng_kind == "random":
        rng = random.Random(SEED)
        for _ in range(N_RESPLITS):
            idx = list(range(n)); rng.shuffle(idx)
            yield idx[:n_cal], idx[n_cal:]
    else:
        rng = np.random.default_rng(SEED)
        for _ in range(N_RESPLITS):
            perm = rng.permutation(n)
            yield list(perm[:n_cal]), list(perm[n_cal:])


def run(docs, rng_kind, recompute_crc):
    rows = {"default": [], "crc_recomputed": []}
    for cal_idx, test_idx in splits(len(docs), rng_kind):
        test = [docs[i] for i in test_idx]
        rows["default"].append(evaluate(test, DEFAULT))
        if recompute_crc:
            lam = crc_lambda([docs[i] for i in cal_idx])
            rows["crc_recomputed"].append(dict(evaluate(test, lam), lam=lam))
    out = {}
    for k, v in rows.items():
        if v:
            out[k] = {m: float(np.mean([r[m] for r in v])) for m in v[0]}
            viol = np.array([r["viol"] for r in v])
            out[k]["viol_split_p95"] = float(np.percentile(viol, 95))
            out[k]["frac_splits_viol_gt_alpha"] = float((viol > ALPHA).mean())
    full = evaluate(docs, DEFAULT)
    out["default_full_data"] = {m: float(x) for m, x in full.items()}
    return out


def main():
    ext = json.load(open(REPO / "paper/outputs/external_scorer_results.json"))
    assert ext == git_json("paper/outputs/external_scorer_results.json"), "external results differ from main"
    judge = git_json("paper/outputs/resplit_cis.json")
    results = {}
    for ds, label in DATASETS.items():
        base = [json.loads(l) for l in open(DATA / ds / "phase_2/calibrated_scores.jsonl")]
        labels = {str(d["doc_id"]): np.array(d["factuality_labels"]) for d in base}
        res = {}
        # LLM judge (CARE's own scorer), vote fraction in {0, .1, ..., 1}
        jd = [(np.array(d["factuality_probs"]), labels[str(d["doc_id"])]) for d in base]
        r = run(jd, "numpy", recompute_crc=True)
        s = judge[ds]["summary"]
        r["crc_submitted"] = dict(viol=s["fact_viol"]["mean"], flagged_per_doc=s["fact_wl"]["mean"],
                                  flagged_pct=s["fact_wl_pct"]["mean"], recall=s["fact_recall"]["mean"])
        res["llm_judge"] = r
        for scorer, key in (("nli", "nli_embed"), ("alignscore", "alignscore")):
            f = HERE / "cache" / f"{ds}__{scorer}_fact.json"
            if not f.exists():
                continue
            sc = json.load(open(f))
            if len(sc) < len(base):
                print(f"{ds} {scorer}: only {len(sc)}/{len(base)} docs scored, skipping"); continue
            docs = [(np.array(sc[str(d["doc_id"])]), labels[str(d["doc_id"])]) for d in base]
            assert all(len(p) == len(y) for p, y in docs)
            r = run(docs, "random", recompute_crc=True)
            s = ext[f"{ds}__{key}"]["summary"]
            r["crc_submitted"] = dict(viol=s["crc_fact_viol"]["mean"], flagged_per_doc=s["crc_fact_wl"]["mean"],
                                      recall=s["crc_fact_recall"]["mean"], maxf1_viol=s["maxf1_fact_viol"]["mean"])
            allp = np.concatenate([p for p, _ in docs])
            r["score_range"] = [float(allp.min()), float(np.median(allp)), float(allp.max())]
            res[scorer] = r
        emb = DATA / ds / "external_scores/embed_only.json"
        if emb.exists():
            e = json.load(open(emb))
            allp = np.concatenate([v["factuality_probs"] for v in e.values()])
            res["bert_cosine"] = dict(note="not a probability; no default threshold", n_docs=len(e),
                                      score_min_p1_median_max=[float(allp.min()), float(np.percentile(allp, 1)),
                                                               float(np.median(allp)), float(allp.max())])
        results[label] = res

    json.dump(results, open(HERE / "r1_w2_default_thresholds.json", "w"), indent=1)
    print(f"alpha = {ALPHA}; default threshold = {DEFAULT}; flag iff score <= threshold. Mean over 100 test splits.")
    print(f"{'Dataset':<10} {'Scorer':<11} {'Rule':<16} {'Viol%':>6} {'P95':>6} {'>a':>5} {'Flag/doc':>8} {'Flag%':>6} {'Rec%':>5}")
    for label, res in results.items():
        for scorer, r in res.items():
            if scorer == "bert_cosine":
                print(f"{label:<10} {scorer:<11} range min/p1/median/max = {r['score_min_p1_median_max']}"); continue
            for rule in ("default", "crc_recomputed", "crc_submitted"):
                x = r[rule]
                print(f"{label:<10} {scorer:<11} {rule:<16} {100*x['viol']:6.1f} "
                      f"{100*x.get('viol_split_p95', np.nan):6.1f} {100*x.get('frac_splits_viol_gt_alpha', np.nan):5.0f} "
                      f"{x['flagged_per_doc']:8.2f} {x.get('flagged_pct', np.nan):6.1f} {100*x['recall']:5.1f}"
                      + (f"  mean lambda {x['lam']:.3f}" if "lam" in x else ""))
            if "score_range" in r:
                print(f"{'':22} score min/median/max = {[round(v, 3) for v in r['score_range']]}; "
                      f"submitted Max-F1 viol = {100*r['crc_submitted']['maxf1_viol']:.1f}%")


if __name__ == "__main__":
    main()
