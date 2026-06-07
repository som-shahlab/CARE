#!/usr/bin/env python3
"""FST external-scorer validity check.

Mirrors compute_external_scorer_results.py but runs FST calibration in
place of the existing crc/dev/maxf1 trio. Re-scores docs via the same
external scorer families (NLI, BERT cosine, AlignScore) — this is the
GPU-heavy part. Then runs 100 resplits per (dataset, scorer) with FST
τ+γ-descending calibration.

Output: paper/outputs/fst_external_scorer_results.json
"""
import json
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR

# Import scoring functions from the existing script (avoids duplicating
# the GPU pipelines).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_external_scorer_results import (
    load_nli_model, load_embed_model, load_alignscore,
    score_nli_factuality, score_nli_coverage,
    score_embedding_factuality, score_embedding_coverage,
    score_alignscore_factuality, score_alignscore_coverage,
    device,
)

SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_FACT = 0.15
ALPHA_OMIT = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05

RESULTS_PATH = OUTPUT_DIR / "fst_external_scorer_results.json"


def compute_factuality(doc, lam):
    labels = np.array(doc["factuality_labels"])
    probs = np.array(doc["factuality_probs"])
    if len(labels) == 0:
        return 0, 0, 0, 0
    flagged = probs <= lam
    halluc = labels == 0
    return (int((halluc & ~flagged).any()), int(flagged.sum()),
            int((flagged & halluc).sum()), int(halluc.sum()))


def compute_omission(doc, tau, gamma):
    imp = np.array(doc["importance_probs"])
    cov = np.array(doc["coverage_probs"])
    imp_l = np.array(doc["importance_labels"])
    cov_l = np.array(doc["coverage_labels"])
    if len(imp) == 0:
        return 0.0, 0, 0, 0
    surfaced = (imp >= tau) & ((1 - cov) >= gamma)
    n_surf = int(surfaced.sum())
    true_omit = (imp_l == 1) & (cov_l == 0)
    n_true = int(true_omit.sum())
    if n_true == 0:
        return 0.0, n_surf, 0, 0
    return (float((true_omit & ~surfaced).sum() / n_true), n_surf,
            int((surfaced & true_omit).sum()), n_true)


def calibrate_factuality(cal_docs, alpha=ALPHA_FACT):
    n = len(cal_docs)
    for lam in np.arange(0.0, 1.0 + GRID_FACT, GRID_FACT):
        losses = [compute_factuality(d, lam)[0] for d in cal_docs]
        adj = (n / (n + 1)) * np.mean(losses) + 1.0 / (n + 1)
        if adj <= alpha:
            return float(lam)
    return 1.0


def fst_calibrate_omission(cal_docs, alpha=ALPHA_OMIT):
    n = len(cal_docs)
    vals = np.arange(0.0, 1.0 + GRID_OMIT, GRID_OMIT)
    cells = [(float(t), float(g)) for t in vals for g in vals]
    cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    for tau, gamma in cells:
        losses = [compute_omission(d, tau, gamma)[0] for d in cal_docs]
        adj = (n / (n + 1)) * np.mean(losses) + 1.0 / (n + 1)
        if adj <= alpha:
            return float(tau), float(gamma)
    return 0.0, 0.0


def evaluate(test_docs, lam, tau, gamma):
    fl, fwl, ftp, fh = 0, 0, 0, 0
    ol, owl, otp, ot = [], [], 0, 0
    fact_losses = []
    for d in test_docs:
        a, b, c, e = compute_factuality(d, lam)
        fact_losses.append(a); fwl += b; ftp += c; fh += e
        x, y, z, w = compute_omission(d, tau, gamma)
        ol.append(x); owl.append(y); otp += z; ot += w
    n = len(test_docs)
    avg_sum = np.mean([len(d.get("factuality_probs", [])) for d in test_docs])
    avg_src = np.mean([len(d.get("importance_probs", [])) for d in test_docs])
    return {
        "fact_viol": float(np.mean(fact_losses)),
        "fact_wl": fwl / n,
        "fact_wl_pct": (fwl / n) / avg_sum * 100 if avg_sum > 0 else 0.0,
        "fact_recall": ftp / fh if fh > 0 else 1.0,
        "omit_viol": float(np.mean(ol)),
        "omit_wl": float(np.mean(owl)),
        "omit_wl_pct": float(np.mean(owl)) / avg_src * 100 if avg_src > 0 else 0.0,
        "omit_recall": otp / ot if ot > 0 else 1.0,
    }


def load_docs(dataset):
    path = DATA_ROOT / dataset / "phase_2" / "calibrated_scores.jsonl"
    return [json.loads(line) for line in open(path)]


def apply_external(docs, scorer, nli_tok, nli_model, emb_tok, emb_model, align):
    scored = []
    for doc in tqdm(docs, desc=f"Scoring ({scorer})", leave=False):
        d = dict(doc)
        if scorer == "nli_embed":
            d["factuality_probs"] = score_nli_factuality(d, nli_tok, nli_model)
            d["coverage_probs"] = score_nli_coverage(d, nli_tok, nli_model)
        elif scorer == "embed_only":
            d["factuality_probs"] = score_embedding_factuality(d, emb_tok, emb_model)
            d["coverage_probs"] = score_embedding_coverage(d, emb_tok, emb_model)
        elif scorer == "alignscore":
            d["factuality_probs"] = score_alignscore_factuality(d, align)
            d["coverage_probs"] = score_alignscore_coverage(d, align)
        else:
            raise ValueError(scorer)
        scored.append(d)
    return scored


def run_resplits(docs):
    rng = random.Random(SEED)
    n = len(docs); n_cal = int(n * CAL_FRAC)
    per_resplit = []
    for _ in tqdm(range(N_RESPLITS), desc="FST resplits", leave=False):
        idx = list(range(n)); rng.shuffle(idx)
        cal = [docs[i] for i in idx[:n_cal]]
        test = [docs[i] for i in idx[n_cal:]]
        lam = calibrate_factuality(cal)
        tau, gamma = fst_calibrate_omission(cal)
        r = evaluate(test, lam, tau, gamma)
        r["lambda"] = lam; r["tau"] = tau; r["gamma"] = gamma
        per_resplit.append(r)
    summary = {}
    for m in ["fact_viol", "fact_wl_pct", "fact_recall",
              "omit_viol", "omit_wl_pct", "omit_recall"]:
        vals = np.array([r[m] for r in per_resplit])
        summary[m] = {
            "mean": float(vals.mean()),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }
    return summary, per_resplit


def main():
    print(f"Device: {device}")
    scorers = os.environ.get("EXTERNAL_SCORERS", "nli_embed,embed_only,alignscore").split(",")
    scorers = [s.strip() for s in scorers if s.strip()]
    print(f"Scorers: {scorers}")

    nli_tok = nli_model = emb_tok = emb_model = align = None
    if "nli_embed" in scorers:
        nli_tok, nli_model = load_nli_model()
    if "embed_only" in scorers:
        emb_tok, emb_model = load_embed_model()
    if "alignscore" in scorers:
        align = load_alignscore()

    all_results = {}
    if RESULTS_PATH.exists():
        all_results = json.load(open(RESULTS_PATH))
        print(f"Loaded {len(all_results)} cached results")

    for ds_key, meta in DATASETS.items():
        print(f"\n{'='*60}\n{meta['label']} ({ds_key})\n{'='*60}")
        docs = load_docs(ds_key)
        for scorer in scorers:
            key = f"{ds_key}__{scorer}"
            if key in all_results and "per_resplit" in all_results[key]:
                print(f"  [cached] {scorer}")
                continue
            print(f"  --- {scorer} ---")
            scored = apply_external(docs, scorer, nli_tok, nli_model, emb_tok, emb_model, align)
            summary, per_resplit = run_resplits(scored)
            all_results[key] = {
                "dataset": ds_key, "label": meta["label"], "scorer": scorer,
                "n_docs": len(docs), "summary": summary, "per_resplit": per_resplit,
            }
            fv = summary["fact_viol"]["mean"] * 100
            ov = summary["omit_viol"]["mean"] * 100
            print(f"    FST: FactV={fv:.1f}% OmitV={ov:.1f}% "
                  f"OmitFl={summary['omit_wl_pct']['mean']:.1f}%")
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")

    # Validity summary
    print(f"\n{'='*78}\n  FST external-scorer validity (α=0.15)\n{'='*78}")
    print(f"  {'Dataset':<14} {'Scorer':<12} {'Fact V%':>10} {'Omit V%':>10}")
    for ds in DATASETS:
        for sc in scorers:
            k = f"{ds}__{sc}"
            if k not in all_results:
                continue
            r = all_results[k]
            fv = r["summary"]["fact_viol"]["mean"] * 100
            ov = r["summary"]["omit_viol"]["mean"] * 100
            print(f"  {DATASETS[ds]['label']:<14} {sc:<12} "
                  f"{'!' if fv>15 else ' '}{fv:9.2f} "
                  f"{'!' if ov>15 else ' '}{ov:9.2f}")


if __name__ == "__main__":
    main()
