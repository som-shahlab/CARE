#!/usr/bin/env python3
"""FST distribution-shift validity check (100 resplits).

Mirrors the omission portion of table_12_distribution_shift.py but uses
LTT-FST (τ+γ descending) calibration in place of 2D Joint argmin-workload.

Two experiments:
  1. Cross-dataset transfer: calibrate FST on dataset A's cal set, evaluate
     test loss on dataset B's test set, for all 5×5 pairs.
  2. Within-dataset length shift: cal on short docs, test on long; cal on
     long, test on short; and random→random.

Output: paper/outputs/fst_distribution_shift.json
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from care.config import (
    CANONICAL_DATA_ROOT as DATA_ROOT,
    DATASETS,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)

ALPHA = 0.15
GRID = 0.05
SEED = 42
N_RESPLITS = 100
CAL_FRAC = 0.70


def load_scored(dataset):
    path = DATA_ROOT / dataset / "phase_2" / "calibrated_scores.jsonl"
    return [json.loads(l) for l in open(path)]


def compute_omission(doc, tau, gamma):
    imp = np.asarray(doc["importance_probs"])
    cov = np.asarray(doc["coverage_probs"])
    imp_l = np.asarray(doc["importance_labels"])
    cov_l = np.asarray(doc["coverage_labels"])
    if len(imp) == 0:
        return 0.0
    surfaced = (imp >= tau) & ((1 - cov) >= gamma)
    true_omit = (imp_l == 1) & (cov_l == 0)
    n_true = int(true_omit.sum())
    if n_true == 0:
        return 0.0
    return float((true_omit & ~surfaced).sum() / n_true)


# ─── Factuality (1D scalar CRC, unchanged by FST) ───

def fact_calibrate(cal_docs, alpha=ALPHA, resolution=0.01):
    """1D CRC factuality threshold: smallest λ whose finite-sample bound holds."""
    n = len(cal_docs)
    for lam in np.arange(0.0, 1.0 + resolution, resolution):
        losses = []
        for doc in cal_docs:
            probs = np.asarray(doc["factuality_probs"])
            labels = np.asarray(doc["factuality_labels"])
            flagged = probs <= lam
            errors = labels == 0
            losses.append(1 if (errors.any() and (errors & ~flagged).any()) else 0)
        if (sum(losses) + 1) / (n + 1) <= alpha:
            return float(lam)
    return 1.0


def evaluate_factuality(test_docs, lam):
    viol = 0
    for doc in test_docs:
        probs = np.asarray(doc["factuality_probs"])
        labels = np.asarray(doc["factuality_labels"])
        flagged = probs <= lam
        errors = labels == 0
        if errors.any() and (errors & ~flagged).any():
            viol += 1
    return viol / len(test_docs)


def fst_calibrate(cal_docs, alpha=ALPHA, grid=GRID):
    """LTT-FST τ+γ descending. Returns (tau, gamma, is_feasible)."""
    n = len(cal_docs)
    vals = np.arange(0.0, 1.0 + grid, grid)
    cells = [(float(t), float(g)) for t in vals for g in vals]
    cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    for tau, gamma in cells:
        losses = [compute_omission(d, tau, gamma) for d in cal_docs]
        adj = (n / (n + 1)) * np.mean(losses) + 1.0 / (n + 1)
        if adj <= alpha:
            return float(tau), float(gamma), True
    return 0.0, 0.0, False


def evaluate(test_docs, tau, gamma):
    return float(np.mean([compute_omission(d, tau, gamma) for d in test_docs]))


def run_cross_dataset():
    print("=" * 72)
    print(f"Experiment 1: Cross-Dataset FST Transfer ({N_RESPLITS} resplits)")
    print("=" * 72)

    all_docs = {}
    for ds in DATASETS:
        all_docs[ds] = load_scored(ds)
        print(f"  Loaded {DATASETS[ds]['label']}: {len(all_docs[ds])} docs")

    ds_keys = list(DATASETS.keys())
    rng = np.random.default_rng(SEED)
    omit_viols = {cal: {test: [] for test in ds_keys} for cal in ds_keys}

    for _ in tqdm(range(N_RESPLITS), desc="Cross-dataset resplits"):
        splits = {}
        for ds in ds_keys:
            docs = all_docs[ds]
            n = len(docs); n_cal = int(n * CAL_FRAC)
            perm = rng.permutation(n)
            splits[ds] = {
                "cal": [docs[j] for j in perm[:n_cal]],
                "test": [docs[j] for j in perm[n_cal:]],
            }
        thresholds = {}
        for ds in ds_keys:
            tau, gamma, _ = fst_calibrate(splits[ds]["cal"])
            thresholds[ds] = (tau, gamma)
        for cal_ds in ds_keys:
            tau, gamma = thresholds[cal_ds]
            for test_ds in ds_keys:
                omit_viols[cal_ds][test_ds].append(evaluate(splits[test_ds]["test"], tau, gamma))

    matrix = {cal: {test: {"violation": float(np.mean(omit_viols[cal][test]))}
                    for test in ds_keys} for cal in ds_keys}

    cal_test = "Cal\\Test"
    header = f"{cal_test:<14}" + "".join(f"{DATASETS[d]['short']:>10}" for d in ds_keys)
    print(f"\nOmission frac. violation (mean over {N_RESPLITS} resplits)")
    print(header)
    print("-" * len(header))
    for cal in ds_keys:
        row = f"{DATASETS[cal]['short']:<14}"
        for test in ds_keys:
            v = matrix[cal][test]["violation"] * 100
            mark = " ✓" if v <= ALPHA * 100 else " ✗"
            row += f"{v:>8.1f}%{mark}"
        print(row)

    return matrix


def run_length_shift():
    print("\n" + "=" * 72)
    print(f"Experiment 2: Within-Dataset Length Shift ({N_RESPLITS} resplits)")
    print("=" * 72)

    rng = np.random.default_rng(SEED + 1)
    out = {}
    for ds in DATASETS:
        docs = load_scored(ds)
        n = len(docs)
        docs.sort(key=lambda d: len(d.get("source_sentences", d.get("importance_labels", []))))
        mid = n // 2
        short_docs = docs[:mid]
        long_docs = docs[mid:]

        label = DATASETS[ds]["label"]
        print(f"\n{label}: n={n}  short={len(short_docs)}  long={len(long_docs)}")

        conditions = {
            "short→long": {"fact": [], "omit": []},
            "long→short": {"fact": [], "omit": []},
            "random→random": {"fact": [], "omit": []},
        }
        for _ in range(N_RESPLITS):
            # short→long
            n_s_cal = int(len(short_docs) * CAL_FRAC)
            perm_s = rng.permutation(len(short_docs))
            cal_s = [short_docs[j] for j in perm_s[:n_s_cal]]
            tau, gamma, _ = fst_calibrate(cal_s)
            lam = fact_calibrate(cal_s)
            conditions["short→long"]["omit"].append(evaluate(long_docs, tau, gamma))
            conditions["short→long"]["fact"].append(evaluate_factuality(long_docs, lam))

            # long→short
            n_l_cal = int(len(long_docs) * CAL_FRAC)
            perm_l = rng.permutation(len(long_docs))
            cal_l = [long_docs[j] for j in perm_l[:n_l_cal]]
            tau, gamma, _ = fst_calibrate(cal_l)
            lam = fact_calibrate(cal_l)
            conditions["long→short"]["omit"].append(evaluate(short_docs, tau, gamma))
            conditions["long→short"]["fact"].append(evaluate_factuality(short_docs, lam))

            # random→random
            n_cal = int(n * CAL_FRAC)
            perm_r = rng.permutation(n)
            cal_r = [docs[j] for j in perm_r[:n_cal]]
            test_r = [docs[j] for j in perm_r[n_cal:]]
            tau, gamma, _ = fst_calibrate(cal_r)
            lam = fact_calibrate(cal_r)
            conditions["random→random"]["omit"].append(evaluate(test_r, tau, gamma))
            conditions["random→random"]["fact"].append(evaluate_factuality(test_r, lam))

        ds_res = {}
        for cond, vals in conditions.items():
            omit_m = float(np.mean(vals["omit"]))
            fact_m = float(np.mean(vals["fact"]))
            ds_res[cond] = {"omit_violation": omit_m, "fact_violation": fact_m}
            om = "✓" if omit_m <= ALPHA else "✗"
            fm = "✓" if fact_m <= ALPHA else "✗"
            print(f"  {cond:<15}: fact={fact_m*100:.1f}% {fm}  omit={omit_m*100:.1f}% {om}")
        out[ds] = ds_res
    return out


def main():
    cross = run_cross_dataset()
    length = run_length_shift()

    out_path = OUTPUT_DIR / "fst_distribution_shift.json"
    with open(out_path, "w") as f:
        json.dump({
            "alpha": ALPHA, "n_resplits": N_RESPLITS, "ordering": "tau_plus_gamma",
            "cross_dataset": cross, "length_shift": length,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
