#!/usr/bin/env python3
"""Compute 95% CIs via random resplits (Angelopoulos-style).

For each dataset:
  1. Load all Phase 2 scored documents
  2. For N_RESPLITS random 70/30 cal/test splits:
     a. Calibrate thresholds on cal set (all methods)
     b. Evaluate on test set (all methods)
  3. Report distribution of violation, workload, recall across resplits

Methods: 2D Joint, 1D-Importance, Product, Score-Gated, Union Bound,
Dev-tuned, Max-F1, Fixed-threshold (τ=γ=0.5 on LLM-judge signal; non-conformal).
Also computes factuality (shared across all omission methods).

Usage:
  # Full recompute (default; regenerates resplit_cis.json from scratch)
  python3 compute_resplit_cis.py

  # Partial recompute: only run named method(s), merge into existing JSON.
  # Seed and split logic are replayed identically, so merged values match
  # what a full recompute would produce.
  python3 compute_resplit_cis.py --methods fixed_threshold
  python3 compute_resplit_cis.py --methods fixed_threshold,dev_tuned
  python3 compute_resplit_cis.py --methods factuality   # recompute factuality only
  python3 compute_resplit_cis.py --datasets ACI_Bench   # restrict to one dataset
"""
import argparse
import json
import sys
import numpy as np
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

from care.config import CANONICAL_DATA_ROOT as DATA_ROOT, DATASETS, PAPER_OUTPUT_DIR as OUTPUT_DIR


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70
ALPHA_FACT = 0.15
ALPHA_OMIT = 0.15
GRID_FACT = 0.01
GRID_OMIT = 0.05
COVERAGE_GATE = 0.5  # For score-gated baseline
FIXED_TAU = 0.5      # Fixed-threshold baseline (non-conformal, label-free)
FIXED_GAMMA = 0.5


# ---------------------------------------------------------------------------
# Loss / surfacing helpers
# ---------------------------------------------------------------------------
def compute_factuality_loss(doc, threshold):
    """Binary factuality loss: 1 if any unflagged hallucination."""
    labels = np.array(doc["factuality_labels"])
    probs = np.array(doc["factuality_probs"])
    if len(labels) == 0:
        return 0, 0, 0, 0
    flagged = probs <= threshold
    halluc = labels == 0
    loss = int((halluc & ~flagged).any())
    n_flagged = int(flagged.sum())
    tp = int((flagged & halluc).sum())
    total_halluc = int(halluc.sum())
    return loss, n_flagged, tp, total_halluc


def _omission_stats(doc, surfaced_mask):
    """Given a surfaced mask, compute fractional loss, binary loss, workload, recall stats."""
    imp_labels = np.array(doc["importance_labels"])
    cov_labels = np.array(doc["coverage_labels"])
    if len(imp_labels) == 0:
        return 0.0, 0, 0, 0, 0

    true_omit = (imp_labels == 1) & (cov_labels == 0)
    n_true = int(true_omit.sum())
    n_surfaced = int(surfaced_mask.sum())

    if n_true == 0:
        return 0.0, n_surfaced, 0, 0, 0

    n_missed = int((true_omit & ~surfaced_mask).sum())
    tp = int((surfaced_mask & true_omit).sum())
    frac_loss = n_missed / n_true
    binary_loss = 1 if n_missed > 0 else 0
    return float(frac_loss), n_surfaced, tp, n_true, binary_loss


def compute_omission_2d(doc, tau, gamma):
    imp_probs = np.array(doc["importance_probs"])
    cov_probs = np.array(doc["coverage_probs"])
    surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
    return _omission_stats(doc, surfaced)


def compute_omission_1d(doc, tau):
    """1D importance only (no coverage gate, gamma=0)."""
    imp_probs = np.array(doc["importance_probs"])
    surfaced = imp_probs >= tau
    return _omission_stats(doc, surfaced)


def compute_omission_product(doc, beta):
    """Product composite: s_prod = p_imp * (1 - p_cov) >= beta."""
    imp_probs = np.array(doc["importance_probs"])
    cov_probs = np.array(doc["coverage_probs"])
    s_prod = imp_probs * (1 - cov_probs)
    surfaced = s_prod >= beta
    return _omission_stats(doc, surfaced)


def compute_omission_score_gated(doc, tau, gate=COVERAGE_GATE):
    """Score-gated: p_imp >= tau AND p_cov < gate."""
    imp_probs = np.array(doc["importance_probs"])
    cov_probs = np.array(doc["coverage_probs"])
    surfaced = (imp_probs >= tau) & (cov_probs < gate)
    return _omission_stats(doc, surfaced)


def compute_omission_union(doc, tau, gamma):
    """Union bound: same surfacing as 2D joint (tau AND gamma), but calibrated independently."""
    return compute_omission_2d(doc, tau, gamma)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------
def calibrate_factuality(cal_docs, alpha, grid=GRID_FACT):
    n = len(cal_docs)
    thresholds = np.arange(0.0, 1.0 + grid, grid)
    for lam in thresholds:
        losses = [compute_factuality_loss(d, lam)[0] for d in cal_docs]
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha:
            return float(lam)
    return 1.0


def calibrate_omission_2d(cal_docs, alpha, grid=GRID_OMIT):
    n = len(cal_docs)
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)

    feasible = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            losses = [compute_omission_2d(d, tau, gamma)[0] for d in cal_docs]
            adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
            if adj_risk <= alpha:
                wl = np.mean([compute_omission_2d(d, tau, gamma)[1] for d in cal_docs])
                feasible.append((tau, gamma, wl))

    if not feasible:
        return 0.0, 0.0

    feasible.sort(key=lambda x: (x[2], -x[0], -x[1]))
    return float(feasible[0][0]), float(feasible[0][1])


def calibrate_omission_1d(cal_docs, alpha, grid=GRID_OMIT):
    """1D importance-only CRC: find largest tau s.t. risk <= alpha (min workload)."""
    n = len(cal_docs)
    thresholds = np.arange(0.0, 1.0 + grid, grid)
    best_tau = 0.0
    for tau in thresholds:
        losses = [compute_omission_1d(d, tau)[0] for d in cal_docs]
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha:
            best_tau = float(tau)
    return best_tau


def calibrate_omission_product(cal_docs, alpha, grid=GRID_OMIT):
    """Product composite CRC: find largest beta s.t. risk <= alpha."""
    n = len(cal_docs)
    thresholds = np.arange(0.0, 1.0 + grid, grid)
    best_beta = 0.0
    for beta in thresholds:
        losses = [compute_omission_product(d, beta)[0] for d in cal_docs]
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha:
            best_beta = float(beta)
    return best_beta


def calibrate_omission_devtuned(cal_docs, alpha, grid=GRID_OMIT):
    """Dev-tuned (no finite-sample correction): minimize workload subject to
    empirical loss <= alpha on cal. Same objective as 2d_joint, but without
    the n/(n+1) correction term. Tie-break: max tau, max gamma."""
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)

    feasible = []
    for tau in tau_vals:
        for gamma in gamma_vals:
            losses = [compute_omission_2d(d, tau, gamma)[0] for d in cal_docs]
            emp_risk = np.mean(losses)
            if emp_risk <= alpha:
                wl = np.mean([compute_omission_2d(d, tau, gamma)[1] for d in cal_docs])
                feasible.append((tau, gamma, wl))

    if not feasible:
        return 0.0, 0.0

    feasible.sort(key=lambda x: (x[2], -x[0], -x[1]))
    return float(feasible[0][0]), float(feasible[0][1])


def calibrate_omission_maxf1(cal_docs, grid=GRID_OMIT):
    """Max-F1 threshold: pick (tau, gamma) that maximizes pooled F1 of
    surfaced sentences against true omissions on calibration data. No
    alpha constraint -- a pure ML thresholding baseline."""
    tau_vals = np.arange(0.0, 1.0 + grid, grid)
    gamma_vals = np.arange(0.0, 1.0 + grid, grid)

    # Precompute true omission masks and probs once per cal doc
    cache = []
    for d in cal_docs:
        imp_labels = np.array(d["importance_labels"])
        cov_labels = np.array(d["coverage_labels"])
        imp_probs = np.array(d["importance_probs"])
        cov_probs = np.array(d["coverage_probs"])
        if len(imp_labels) == 0:
            continue
        true_omit = (imp_labels == 1) & (cov_labels == 0)
        cache.append((imp_probs, cov_probs, true_omit))

    if not cache:
        return 0.0, 0.0

    best_f1 = -1.0
    best_pair = (0.0, 0.0)
    for tau in tau_vals:
        for gamma in gamma_vals:
            tp = fp = fn = 0
            for imp_probs, cov_probs, true_omit in cache:
                surfaced = (imp_probs >= tau) & ((1 - cov_probs) >= gamma)
                tp += int((surfaced & true_omit).sum())
                fp += int((surfaced & ~true_omit).sum())
                fn += int((~surfaced & true_omit).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if prec + rec == 0:
                continue
            f1 = 2 * prec * rec / (prec + rec)
            # Tie-break on higher tau, higher gamma (lower workload) at equal F1
            if (f1 > best_f1) or (f1 == best_f1 and (tau, gamma) > best_pair):
                best_f1 = f1
                best_pair = (float(tau), float(gamma))
    return best_pair


def calibrate_omission_union(cal_docs, alpha, grid=GRID_OMIT):
    """Union bound: calibrate tau at alpha/2, gamma at alpha/2 independently."""
    tau = calibrate_omission_1d(cal_docs, alpha / 2, grid)
    # Calibrate gamma independently
    n = len(cal_docs)
    thresholds = np.arange(0.0, 1.0 + grid, grid)
    best_gamma = 0.0
    for gamma in thresholds:
        losses = []
        for d in cal_docs:
            imp_labels = np.array(d["importance_labels"])
            cov_labels = np.array(d["coverage_labels"])
            cov_probs = np.array(d["coverage_probs"])
            true_omit = (imp_labels == 1) & (cov_labels == 0)
            n_true = true_omit.sum()
            if n_true == 0:
                losses.append(0.0)
            else:
                surfaced = (1 - cov_probs) >= gamma
                n_missed = (true_omit & ~surfaced).sum()
                losses.append(float(n_missed / n_true))
        adj_risk = (n / (n + 1)) * np.mean(losses) + 1 / (n + 1)
        if adj_risk <= alpha / 2:
            best_gamma = float(gamma)
    return tau, best_gamma


# ---------------------------------------------------------------------------
# Method registry: each entry maps calibration + evaluation for one baseline.
# Adding a new baseline = adding one entry here.
# ---------------------------------------------------------------------------
METHODS = OrderedDict([
    ("2d_joint", {
        "calibrate": lambda cal: calibrate_omission_2d(cal, ALPHA_OMIT),
        "surface":   lambda doc, p: compute_omission_2d(doc, p[0], p[1]),
    }),
    ("1d_imp", {
        "calibrate": lambda cal: (calibrate_omission_1d(cal, ALPHA_OMIT),),
        "surface":   lambda doc, p: compute_omission_1d(doc, p[0]),
    }),
    ("product", {
        "calibrate": lambda cal: (calibrate_omission_product(cal, ALPHA_OMIT),),
        "surface":   lambda doc, p: compute_omission_product(doc, p[0]),
    }),
    ("score_gated", {
        "calibrate": lambda cal: (calibrate_omission_1d(cal, ALPHA_OMIT),),
        "surface":   lambda doc, p: compute_omission_score_gated(doc, p[0]),
    }),
    ("union_bound", {
        "calibrate": lambda cal: calibrate_omission_union(cal, ALPHA_OMIT),
        "surface":   lambda doc, p: compute_omission_union(doc, p[0], p[1]),
    }),
    ("dev_tuned", {
        "calibrate": lambda cal: calibrate_omission_devtuned(cal, ALPHA_OMIT),
        "surface":   lambda doc, p: compute_omission_2d(doc, p[0], p[1]),
    }),
    ("max_f1", {
        "calibrate": lambda cal: calibrate_omission_maxf1(cal),
        "surface":   lambda doc, p: compute_omission_2d(doc, p[0], p[1]),
    }),
    ("fixed_threshold", {
        "calibrate": lambda cal: (FIXED_TAU, FIXED_GAMMA),
        "surface":   lambda doc, p: compute_omission_2d(doc, p[0], p[1]),
    }),
])

BASELINE_METRICS = ["omit_viol", "omit_binary_viol", "omit_wl", "omit_wl_pct", "omit_recall"]


def _evaluate_method_on_test(method_name, test_docs, params, avg_src_sents):
    """Surface sentences on test set with (params) and aggregate per-method metrics."""
    surface_fn = METHODS[method_name]["surface"]
    frac_losses, bin_losses, surfaced, tp_total, true_total = [], [], [], 0, 0
    for doc in test_docs:
        fl, ns, tp, nt, bl = surface_fn(doc, params)
        frac_losses.append(fl); bin_losses.append(bl); surfaced.append(ns)
        tp_total += tp; true_total += nt
    rec = tp_total / true_total if true_total > 0 else 1.0
    return {
        "omit_viol": float(np.mean(frac_losses)),
        "omit_binary_viol": float(np.mean(bin_losses)),
        "omit_wl": float(np.mean(surfaced)),
        "omit_wl_pct": float(np.mean(surfaced) / avg_src_sents * 100) if avg_src_sents > 0 else 0,
        "omit_recall": float(rec),
    }


def _evaluate_factuality_on_test(test_docs, lam, avg_sum_sents):
    f_losses, f_flagged, f_tp, f_halluc = [], [], 0, 0
    for doc in test_docs:
        fl, nf, ftp, fh = compute_factuality_loss(doc, lam)
        f_losses.append(fl); f_flagged.append(nf); f_tp += ftp; f_halluc += fh
    f_rec = f_tp / f_halluc if f_halluc > 0 else 1.0
    return {
        "fact_viol": float(np.mean(f_losses)),
        "fact_wl": float(np.mean(f_flagged)),
        "fact_wl_pct": float(np.mean(f_flagged) / avg_sum_sents * 100) if avg_sum_sents > 0 else 0,
        "fact_recall": float(f_rec),
        "lambda": float(lam),
    }


def _summarize(vals):
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
        "median": float(np.median(vals)),
    }


# ---------------------------------------------------------------------------
# Main per-dataset driver
# ---------------------------------------------------------------------------
def run_resplits(dataset_key, methods_to_run=None, include_factuality=True,
                 existing=None):
    """
    methods_to_run: list of METHODS keys to (re)compute. Defaults to all.
    include_factuality: if True, (re)compute factuality fields.
    existing: if not None, merge new values into this dict (must have per_resplit
              of length N_RESPLITS).
    """
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / "phase_2" / "calibrated_scores.jsonl"
    if not path.exists():
        print(f"SKIP {dataset_key}: {path} not found")
        return None

    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_cal = int(n * CAL_FRAC)

    if methods_to_run is None:
        methods_to_run = list(METHODS.keys())
    merging = existing is not None
    if merging:
        assert len(existing.get("per_resplit", [])) == N_RESPLITS, \
            f"{dataset_key}: --methods requires existing per_resplit of length {N_RESPLITS}"
        results = existing["per_resplit"]
    else:
        results = [{} for _ in range(N_RESPLITS)]

    print(f"\n{'='*70}")
    print(f"  {meta['label']} (N={n}, cal={n_cal}, test={n - n_cal})")
    print(f"  {N_RESPLITS} resplits, alpha_fact={ALPHA_FACT}, alpha_omit={ALPHA_OMIT}")
    if merging:
        tag = f"merge: methods={methods_to_run}" + (" +factuality" if include_factuality else "")
        print(f"  {tag}")
    print(f"{'='*70}")

    rng = np.random.default_rng(SEED)

    for i in tqdm(range(N_RESPLITS), desc=meta["label"]):
        perm = rng.permutation(n)
        cal_docs = [docs[j] for j in perm[:n_cal]]
        test_docs = [docs[j] for j in perm[n_cal:]]

        avg_sum_sents = np.mean([len(d.get("factuality_probs", [])) for d in test_docs])
        avg_src_sents = np.mean([len(d.get("importance_probs", [])) for d in test_docs])

        res = results[i]

        if include_factuality:
            lam = calibrate_factuality(cal_docs, ALPHA_FACT)
            res.update(_evaluate_factuality_on_test(test_docs, lam, avg_sum_sents))

        baselines = res.setdefault("baselines", {})
        params_by_method = {}
        for mname in methods_to_run:
            params = METHODS[mname]["calibrate"](cal_docs)
            params_by_method[mname] = params
            baselines[mname] = _evaluate_method_on_test(mname, test_docs, params, avg_src_sents)

        # Flat 2D Joint fields (backward compat) — only update if 2d_joint recomputed
        if "2d_joint" in methods_to_run:
            for k, v in baselines["2d_joint"].items():
                res[k] = v
            p2d = params_by_method["2d_joint"]
            res["tau"] = float(p2d[0])
            res["gamma"] = float(p2d[1])

        results[i] = res

    # Re-aggregate summary + baselines_summary from final per_resplit.
    # (We always re-aggregate so merged JSONs reflect any changes.)
    top_metrics = ["fact_viol", "fact_wl", "fact_wl_pct", "fact_recall",
                   "omit_viol", "omit_wl", "omit_wl_pct", "omit_recall",
                   "omit_binary_viol", "lambda", "tau", "gamma"]
    summary = {}
    for m in top_metrics:
        vals = [r.get(m) for r in results if m in r]
        if vals:
            summary[m] = _summarize(vals)

    all_baselines = set()
    for r in results:
        all_baselines.update(r.get("baselines", {}).keys())

    baselines_summary = {}
    for bl in all_baselines:
        baselines_summary[bl] = {}
        for m in BASELINE_METRICS:
            vals = [r["baselines"][bl][m] for r in results
                    if bl in r.get("baselines", {}) and m in r["baselines"][bl]]
            if vals:
                baselines_summary[bl][m] = _summarize(vals)

    # Print
    if "fact_viol" in summary:
        s = summary
        print(f"\n  Factuality:")
        print(f"    Viol:   {s['fact_viol']['mean']:.1%} [{s['fact_viol']['ci_lower']:.1%}, {s['fact_viol']['ci_upper']:.1%}]  target <= {ALPHA_FACT:.0%}")
        print(f"    WL:     {s['fact_wl']['mean']:.1f} ({s['fact_wl_pct']['mean']:.1f}%)")
        print(f"    Recall: {s['fact_recall']['mean']:.1%}")

    print(f"\n  Omission baselines (mean V% / WL / Rec%):")
    for bl in METHODS.keys():
        if bl not in baselines_summary:
            continue
        bs = baselines_summary[bl]
        print(f"    {bl:15s}: V={bs['omit_viol']['mean']:.1%}  WL={bs['omit_wl']['mean']:.1f} ({bs['omit_wl_pct']['mean']:.1f}%)  Rec={bs['omit_recall']['mean']:.1%}")

    return {
        "dataset": meta["label"],
        "n_total": n,
        "n_cal": n_cal,
        "n_test": n - n_cal,
        "n_resplits": N_RESPLITS,
        "alpha_fact": ALPHA_FACT,
        "alpha_omit": ALPHA_OMIT,
        "summary": summary,
        "baselines_summary": baselines_summary,
        "per_resplit": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated method names to (re)compute. "
                             "Include 'factuality' to recompute factuality fields. "
                             "When specified, merges into existing resplit_cis.json "
                             f"instead of overwriting. Valid: {list(METHODS.keys()) + ['factuality']}")
    parser.add_argument("--datasets", type=str, default=None,
                        help=f"Comma-separated dataset keys to process. Default: all. "
                             f"Valid: {list(DATASETS.keys())}")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_path = OUTPUT_DIR / "resplit_cis.json"

    if args.methods is None:
        methods_to_run = None            # => all methods
        include_factuality = True
        existing = None                  # full recompute, do not merge
    else:
        tokens = {t.strip() for t in args.methods.split(",") if t.strip()}
        include_factuality = "factuality" in tokens
        tokens.discard("factuality")
        bad = tokens - set(METHODS.keys())
        if bad:
            parser_valid = list(METHODS.keys()) + ["factuality"]
            print(f"ERROR: unknown method(s) {sorted(bad)}. Valid: {parser_valid}")
            sys.exit(1)
        methods_to_run = [m for m in METHODS.keys() if m in tokens]
        if not methods_to_run and not include_factuality:
            print("ERROR: --methods must include at least one valid method")
            sys.exit(1)
        if not out_path.exists():
            print(f"ERROR: {out_path} does not exist; --methods requires existing JSON. "
                  f"Run without --methods first.")
            sys.exit(1)
        existing = OrderedDict(json.load(open(out_path)))

    dataset_filter = None
    if args.datasets is not None:
        dataset_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}
        bad = dataset_filter - set(DATASETS.keys())
        if bad:
            print(f"ERROR: unknown dataset(s) {sorted(bad)}. Valid: {list(DATASETS.keys())}")
            sys.exit(1)

    all_results = existing if existing is not None else OrderedDict()

    for dataset_key in DATASETS:
        if dataset_filter is not None and dataset_key not in dataset_filter:
            continue
        ds_existing = all_results.get(dataset_key) if args.methods is not None else None
        result = run_resplits(
            dataset_key,
            methods_to_run=methods_to_run,
            include_factuality=include_factuality,
            existing=ds_existing,
        )
        if result:
            all_results[dataset_key] = result

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*90}")
    print(f"  SUMMARY ({N_RESPLITS} resplits, alpha={ALPHA_FACT})")
    print(f"{'='*90}")
    print(f"  {'Dataset':12s} | {'Fact Viol':>20s} | {'Omit Viol (2D)':>20s} | {'Fact Rec':>20s} | {'Omit Rec':>20s}")
    print(f"  {'-'*100}")
    for dk, r in all_results.items():
        s = r.get("summary", {})
        def _ci(k):
            if k not in s: return "n/a"
            return f"{s[k]['mean']:5.1%} [{s[k]['ci_lower']:5.1%},{s[k]['ci_upper']:5.1%}]"
        print(f"  {DATASETS[dk]['label']:12s} | {_ci('fact_viol'):>20s} | "
              f"{_ci('omit_viol'):>20s} | {_ci('fact_recall'):>20s} | {_ci('omit_recall'):>20s}")
