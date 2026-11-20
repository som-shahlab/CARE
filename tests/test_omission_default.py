"""The pipeline's deployed omission controller must be LTT-FST (the only omission
rule here with a formal guarantee over a 2D grid); 2D-Joint is a comparator."""
import json

import numpy as np
import pytest

from care.calibration import (
    deployed_omission_thresholds,
    run_alpha_sweep,
    run_phase3,
    select_omission_threshold_2d,
    select_omission_threshold_2d_fractional,
    select_omission_threshold_ltt_fst,
)
from care.evaluation import run_alpha_sweep_test

ALPHA, DELTA = 0.15, 0.05


def _docs(n, seed=0):
    rng = np.random.default_rng(seed)
    docs = []
    for k in range(n):
        m = 8
        imp_lab = rng.integers(0, 2, m)
        cov_lab = rng.integers(0, 2, m)
        imp = np.clip(0.6 * imp_lab + 0.4 * rng.random(m), 0, 1)
        cov = np.clip(0.6 * cov_lab + 0.4 * rng.random(m), 0, 1)
        fact_lab = rng.integers(0, 2, 5)
        fact = np.clip(0.6 * fact_lab + 0.4 * rng.random(5), 0, 1)
        docs.append({
            "doc_id": f"d{k}",
            "importance_probs": imp.tolist(), "coverage_probs": cov.tolist(),
            "importance_labels": imp_lab.tolist(), "coverage_labels": cov_lab.tolist(),
            "factuality_probs": fact.tolist(), "factuality_labels": fact_lab.tolist(),
        })
    return docs


def test_2d_joint_does_not_claim_a_formal_guarantee():
    docs = _docs(40)
    assert select_omission_threshold_2d_fractional(docs, ALPHA)["has_formal_guarantee"] is False
    assert select_omission_threshold_2d(docs, ALPHA)["has_formal_guarantee"] is False


def test_phase3_omission_key_is_ltt_fst(tmp_path):
    docs = _docs(60)
    inp = tmp_path / "scores.jsonl"
    inp.write_text("\n".join(json.dumps(d) for d in docs))
    run_phase3(input_file=inp, output_dir=tmp_path / "p3", alpha_fact=ALPHA, alpha_omit=ALPHA,
               split="all", delta=DELTA)
    out = json.load(open(tmp_path / "p3" / "conformal_thresholds.json"))
    expected = select_omission_threshold_ltt_fst(docs, ALPHA, DELTA)
    assert out["omission"]["threshold_type"].startswith("omission_ltt_fst")
    assert (out["omission"]["tau"], out["omission"]["gamma"]) == (expected["tau"], expected["gamma"])
    assert out["omission_ltt_fst"] == out["omission"]  # back-compat alias
    assert out["omission_2d_joint"]["threshold_type"] == "omission_2d_fractional"


def test_calibration_alpha_sweep_uses_ltt_fst():
    docs = _docs(60)
    r = run_alpha_sweep(docs, [ALPHA], compute_ci=False)[0]
    expected = select_omission_threshold_ltt_fst(docs, ALPHA, DELTA)
    assert (r["omission"]["tau"], r["omission"]["gamma"]) == (expected["tau"], expected["gamma"])
    assert "omission_2d_joint" in r


def test_test_alpha_sweep_uses_ltt_fst():
    cal, test = _docs(60, seed=1), _docs(30, seed=2)
    r = run_alpha_sweep_test(cal, test, [ALPHA], compute_ci=False)[0]
    expected = select_omission_threshold_ltt_fst(cal, ALPHA, DELTA)
    assert (r["omission"]["tau"], r["omission"]["gamma"]) == (expected["tau"], expected["gamma"])


def test_deployed_thresholds_prefer_ltt_fst_in_old_files():
    ltt = {"tau": 0.3, "gamma": 0.4, "threshold_type": "omission_ltt_fst_tau_plus_gamma_min_workload"}
    joint = {"tau": 0.6, "gamma": 0.7, "threshold_type": "omission_2d_fractional"}
    assert deployed_omission_thresholds({"omission": ltt, "omission_ltt_fst": ltt}) == ltt
    # files written before this change: 'omission' held 2D-Joint
    assert deployed_omission_thresholds({"omission": joint, "omission_ltt_fst": ltt}) == ltt


def test_deployed_thresholds_refuse_2d_joint_only_files():
    joint = {"tau": 0.6, "gamma": 0.7, "threshold_type": "omission_2d_fractional"}
    with pytest.raises(ValueError, match="re-run Phase 3"):
        deployed_omission_thresholds({"omission": joint})
