#!/usr/bin/env python3
"""FST sample-complexity validation.

For each dataset and each calibration sample size n ∈ {15, 25, 50, 75, 100},
draw 20 random subsamples, calibrate via LTT-FST (τ+γ desc) at α=0.15,
evaluate on the held-out test set. Reports mean violation rate per (dataset, n)
and checks whether validity holds.

Output: paper/outputs/fst_sample_complexity.json
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
SIZES = [15, 25, 50, 75, 100]
N_REPEATS = 20
TEST_FRAC = 0.30
SEED = 42


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


def fst_calibrate(cal_docs, alpha=ALPHA, grid=GRID):
    """LTT-FST with the finite-sample correction (the CARE controller)."""
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


def fst_calibrate_devset(cal_docs, alpha=ALPHA, grid=GRID):
    """LTT-FST WITHOUT the finite-sample correction (dev-set tuning).

    Walks the identical tau+gamma-descending ordering but stops at the first
    cell whose raw empirical risk R̂ <= alpha, i.e. it omits the n/(n+1)
    correction term. This isolates the role of the correction at small n.
    """
    vals = np.arange(0.0, 1.0 + grid, grid)
    cells = [(float(t), float(g)) for t in vals for g in vals]
    cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    for tau, gamma in cells:
        losses = [compute_omission(d, tau, gamma) for d in cal_docs]
        if np.mean(losses) <= alpha:
            return float(tau), float(gamma)
    return 0.0, 0.0


def run_dataset(dataset_key):
    meta = DATASETS[dataset_key]
    path = DATA_ROOT / dataset_key / "phase_2" / "calibrated_scores.jsonl"
    if not path.exists():
        return None
    docs = [json.loads(l) for l in open(path)]
    n = len(docs)
    n_test = max(int(n * TEST_FRAC), 1)

    print(f"\n{'='*70}")
    print(f"  {meta['label']}  (N={n}, test={n_test}, sizes={SIZES}, repeats={N_REPEATS})")
    print(f"{'='*70}")

    rng = np.random.default_rng(SEED)
    results = []
    for n_sub in SIZES:
        if n_sub + n_test > n:
            print(f"  SKIP n_sub={n_sub} (n_sub+n_test={n_sub+n_test} > N={n})")
            continue
        viols, workloads, feas_count = [], [], 0
        devset_viols = []
        for _ in range(N_REPEATS):
            perm = rng.permutation(n)
            test = [docs[j] for j in perm[:n_test]]
            cal = [docs[j] for j in perm[n_test:n_test + n_sub]]
            tau, gamma, is_feas = fst_calibrate(cal)
            if is_feas:
                feas_count += 1
            v = float(np.mean([compute_omission(d, tau, gamma) for d in test]))
            wl = float(np.mean([
                int(((np.asarray(d["importance_probs"]) >= tau) &
                     ((1 - np.asarray(d["coverage_probs"])) >= gamma)).sum())
                for d in test
            ]))
            viols.append(v); workloads.append(wl)
            # Dev-set tuning: same FST ordering, no n/(n+1) correction.
            dt, dg = fst_calibrate_devset(cal)
            devset_viols.append(
                float(np.mean([compute_omission(d, dt, dg) for d in test])))
        v_mean = float(np.mean(viols))
        wl_mean = float(np.mean(workloads))
        dev_mean = float(np.mean(devset_viols))
        holds = v_mean <= ALPHA
        mark = "✓" if holds else "✗"
        dev_mark = "✗" if dev_mean > ALPHA else "✓"
        print(f"  n={n_sub:>3}: omit V={v_mean*100:5.2f}% {mark}  "
              f"devset V={dev_mean*100:5.2f}% {dev_mark}  WL={wl_mean:6.2f}  "
              f"feasible={feas_count}/{N_REPEATS}")
        results.append({
            "n_sub": n_sub,
            "n_repeats": N_REPEATS,
            "omit_violation_mean": v_mean,
            "omit_violation_std": float(np.std(viols)),
            "omit_violation_all": viols,
            "omit_pct_holds": float(np.mean([v <= ALPHA for v in viols])),
            "omit_workload_mean": wl_mean,
            "n_feasible": feas_count,
            "devset_omit_violation_mean": dev_mean,
            "devset_omit_violation_std": float(np.std(devset_viols)),
            "devset_omit_violation_all": devset_viols,
            "devset_omit_pct_holds": float(np.mean([v <= ALPHA for v in devset_viols])),
        })
    return {
        "dataset": meta["label"],
        "n_total": n,
        "n_test": n_test,
        "alpha_omit": ALPHA,
        "results": results,
    }


def main():
    out = {}
    for ds in DATASETS:
        r = run_dataset(ds)
        if r is not None:
            out[ds] = r
    out_path = OUTPUT_DIR / "fst_sample_complexity.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*80}")
    print(f"  FST sample-complexity summary (mean omit V%)")
    print(f"{'='*80}")
    header = f"  {'Dataset':<14}" + "".join(f"{n:>10}" for n in SIZES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ds, r in out.items():
        row = {x["n_sub"]: x["omit_violation_mean"] for x in r["results"]}
        line = f"  {DATASETS[ds]['label']:<14}"
        for n in SIZES:
            if n in row:
                v = row[n] * 100
                mark = "!" if v > ALPHA * 100 else " "
                line += f"{mark}{v:8.2f} "
            else:
                line += f"{'--':>9} "
        print(line)
    print("  ! = exceeds α")


if __name__ == "__main__":
    main()
