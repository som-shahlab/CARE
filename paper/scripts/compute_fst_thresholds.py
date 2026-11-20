#!/usr/bin/env python3
"""Compute canonical LTT-FST omission thresholds on the calibration split.

Mirrors Phase 3 (care/calibration.py) but uses the LTT-FST procedure
(HB p-values, tau+gamma ascending, delta=0.05, min-workload certified cell) for the omission controller. The factuality
threshold lambda* is 1D scalar CRC and is unaffected by the FST migration, so
it is carried over verbatim from phase_3/conformal_thresholds.json.

Output: paper/outputs/fst_thresholds.json — per-dataset {lambda, tau, gamma,
adjusted_risk, workload, is_feasible}. Consumed by table_7_thresholds.py.

Usage:
    python3 paper/scripts/compute_fst_thresholds.py
"""
import json
import logging
from pathlib import Path

from care.config import (
    DATASETS, CANONICAL_DATA_ROOT, get_phase2_scores, get_phase3_thresholds,
    PAPER_OUTPUT_DIR as OUTPUT_DIR,
)
from care.calibration import select_omission_threshold_ltt_fst

logging.disable(logging.CRITICAL)

ALPHA = 0.15
GRID = 0.05
ORDERING = "tau_plus_gamma"
DELTA = 0.05
OUT_PATH = OUTPUT_DIR / "fst_thresholds.json"


def calibration_split_path(ds: str) -> Path:
    """Canonical calibration split file (handles the nested PRIV_DS layout)."""
    flat = CANONICAL_DATA_ROOT / ds / "split" / "calibration.jsonl"
    if flat.exists():
        return flat
    nested = CANONICAL_DATA_ROOT / ds / "split" / ds / "calibration.jsonl"
    if nested.exists():
        return nested
    # Nested under another folder name (split/<other>/calibration.jsonl).
    for other in sorted((CANONICAL_DATA_ROOT / ds / "split").glob("*/calibration.jsonl")):
        return other
    raise FileNotFoundError(f"no calibration split for {ds}")


def main():
    out = {}
    print(f"Canonical LTT-FST thresholds ({ORDERING}, alpha={ALPHA})")
    print("=" * 72)
    for ds in DATASETS:
        cal_ids = {json.loads(l)["doc_id"]
                   for l in open(calibration_split_path(ds))}
        docs = [json.loads(l) for l in open(get_phase2_scores(ds))]
        cal = [d for d in docs if d["doc_id"] in cal_ids]

        fst = select_omission_threshold_ltt_fst(cal, ALPHA, DELTA, GRID, ORDERING)

        # lambda* (factuality) is unchanged 1D scalar CRC — carry from phase_3.
        p3 = json.load(open(get_phase3_thresholds(ds)))
        lam = p3["factuality"]["threshold"]

        out[ds] = {
            "lambda": lam,
            "tau": fst["tau"],
            "gamma": fst["gamma"],
            "n_cal": len(cal),
            "empirical_risk": fst["empirical_risk"],
            "hb_pvalue": fst["hb_pvalue"],
            "n_certified": fst["n_certified"],
            "workload": fst["workload"],
            "is_feasible": fst["is_feasible"],
            "alpha": ALPHA,
            "delta": DELTA,
            "method": fst["threshold_type"],
        }
        print(f"{ds:<16} lambda*={lam:.2f}  "
              f"tau*={fst['tau']:.2f}  gamma*={fst['gamma']:.2f}  "
              f"risk={fst['empirical_risk']:.4f}  p_HB={fst['hb_pvalue']:.4f}  "
              f"feasible={fst['is_feasible']}")

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("=" * 72)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
