#!/usr/bin/env python3
"""Merge 2D-Joint and LTT-FST resplit CIs into one canonical file.

Produces ``resplit_cis_fst.json`` with the SAME schema as ``resplit_cis.json``,
so downstream table scripts need only a one-line path swap.

What changes vs. resplit_cis.json:
  - ``summary`` omission fields (omit_viol/omit_wl/omit_wl_pct/omit_recall/
    omit_binary_viol, tau, gamma) are replaced by LTT-FST (tau+gamma ordering).
    Factuality fields and ``lambda`` are untouched (factuality is 1D scalar CRC,
    unaffected by the FST migration).
  - ``per_resplit[i]`` omission fields are replaced by FST values for the same
    resplit i (the two files share seeds; verified by index pairing below).
  - ``baselines`` / ``baselines_summary`` gain an ``fst`` entry (= primary CARE
    omission controller). The ``2d_joint`` entry is PRESERVED as a comparator.

The FST omission summary (incl. 95% CIs) is taken directly from
``fst_resplit_cis.json``'s ``fst_summary['tau_plus_gamma']``, which compute_
fst_resplit_cis.py computes with the same percentile convention as
compute_resplit_cis.py.

Usage:
    python3 paper/scripts/merge_resplit_cis_fst.py
"""
import copy
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SRC_2DJ = OUTPUT_DIR / "resplit_cis.json"
SRC_FST = OUTPUT_DIR / "fst_resplit_cis.json"
DST = OUTPUT_DIR / "resplit_cis_fst.json"

ORDERING = "tau_plus_gamma"

# Omission fields that the FST controller overwrites.
OMIT_FIELDS = ["omit_viol", "omit_wl", "omit_wl_pct", "omit_recall",
               "omit_binary_viol"]
THRESH_FIELDS = ["tau", "gamma"]
# Per-resplit baseline record fields (no tau/gamma in baseline records).
BASELINE_FIELDS = OMIT_FIELDS


def main():
    if not SRC_2DJ.exists() or not SRC_FST.exists():
        sys.exit(f"ERROR: need both {SRC_2DJ.name} and {SRC_FST.name}")

    d2 = json.load(open(SRC_2DJ))
    df = json.load(open(SRC_FST))

    merged = copy.deepcopy(d2)

    print(f"Merging FST ({ORDERING}) omission results into 2D-Joint schema")
    print("=" * 72)

    for ds in d2:
        if ds not in df:
            sys.exit(f"ERROR: dataset {ds} missing from {SRC_FST.name}")

        v2, vf = merged[ds], df[ds]

        # --- pairing sanity checks -----------------------------------------
        pr2, prf = v2["per_resplit"], vf["per_resplit"]
        if len(pr2) != len(prf):
            sys.exit(f"ERROR: {ds} resplit count mismatch "
                     f"({len(pr2)} vs {len(prf)})")
        if v2.get("n_resplits") != vf.get("n_resplits"):
            sys.exit(f"ERROR: {ds} n_resplits mismatch")
        # FST per_resplit carries resplit_idx 0..n-1; confirm it is positional.
        for i, r in enumerate(prf):
            if r.get("resplit_idx") != i:
                sys.exit(f"ERROR: {ds} FST resplit_idx not positional at {i}")
        if v2.get("n_total") != vf.get("n_total"):
            sys.exit(f"ERROR: {ds} n_total mismatch "
                     f"({v2.get('n_total')} vs {vf.get('n_total')}) "
                     f"-- files are not on the same data snapshot")

        fst_summary = vf["fst_summary"][ORDERING]

        # --- summary: overwrite omission, keep factuality + lambda ---------
        for f in OMIT_FIELDS + THRESH_FIELDS:
            v2["summary"][f] = copy.deepcopy(fst_summary[f])
        # carry FST-specific diagnostics (harmless extras)
        v2["summary"]["fst_steps_walked"] = copy.deepcopy(
            fst_summary["fst_steps_walked"])
        v2["summary"]["fst_n_feasible"] = fst_summary["n_feasible"]
        v2["omission_method"] = f"fst_{ORDERING}"

        # --- baselines_summary: add 'fst', keep '2d_joint' -----------------
        v2["baselines_summary"]["fst"] = {
            f: copy.deepcopy(fst_summary[f]) for f in OMIT_FIELDS
        }

        # --- per_resplit: overwrite omission, add 'fst' baseline -----------
        for i in range(len(pr2)):
            fst_cell = prf[i]["fst"][ORDERING]
            for f in OMIT_FIELDS + THRESH_FIELDS:
                pr2[i][f] = fst_cell[f]
            pr2[i].setdefault("baselines", {})["fst"] = {
                f: fst_cell[f] for f in BASELINE_FIELDS
            }

        # --- report --------------------------------------------------------
        o2 = d2[ds]["summary"]["omit_viol"]["mean"] * 100
        of = fst_summary["omit_viol"]["mean"] * 100
        w2 = d2[ds]["summary"]["omit_wl"]["mean"]
        wf = fst_summary["omit_wl"]["mean"]
        print(f"{ds:<16} omit V%: 2DJ {o2:5.2f} -> FST {of:5.2f}   "
              f"workload: 2DJ {w2:6.2f} -> FST {wf:6.2f}   "
              f"feasible {fst_summary['n_feasible']}/100")

    with open(DST, "w") as f:
        json.dump(merged, f, indent=2)
    print("=" * 72)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
