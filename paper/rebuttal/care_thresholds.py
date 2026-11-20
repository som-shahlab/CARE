"""Deployed CARE thresholds for the rebuttal / camera-ready analyses.

Default: the camera-ready thresholds from the canonical seed-42 calibration split
(paper/outputs/fst_thresholds.json: CRC lambda, LTT-FST (tau, gamma) with min-workload deploy).
Set CARE_THRESHOLDS=submitted to reproduce the reviewed-version (submission-time) analyses.
Values are returned exactly as stored (unrounded floats), matching the calibration pipeline.
"""
import json
import os
from pathlib import Path

LABEL_TO_KEY = {"ACI-Bench": "ACI_Bench", "MIMIC-BHC": "MIMIC_IV_BHC", "MIMIC-CXR": "MIMIC_III_CXR",
                "Priv-DS": "PRIV_DS", "SumPubMed": "SumPubMed"}
SUBMITTED = {"ACI-Bench": (0.70, 0.50, 0.30), "MIMIC-BHC": (0.60, 0.40, 0.50),
             "MIMIC-CXR": (0.50, 1.00, 0.50), "Priv-DS": (0.80, 0.50, 0.50), "SumPubMed": (0.50, 0.50, 0.40)}


def mode():
    return os.environ.get("CARE_THRESHOLDS", "camera_ready")


def thresholds():
    """{label: (lambda, tau, gamma)}"""
    if mode() == "submitted":
        return dict(SUBMITTED)
    d = json.load(open(Path(__file__).resolve().parents[1] / "outputs" / "fst_thresholds.json"))
    return {lab: (d[k]["lambda"], d[k]["tau"], d[k]["gamma"]) for lab, k in LABEL_TO_KEY.items() if k in d}
