"""Tests for the LTT-FST omission controller (care.calibration)."""
import numpy as np
import pytest

from care.calibration import (
    hb_pvalue,
    ltt_fst_ordering,
    ltt_fst_select_from_grid,
    omission_grid_matrices,
    select_omission_threshold_ltt_fst,
)

ALPHA, DELTA = 0.15, 0.05


# ---------------------------------------------------------------- hb_pvalue
def test_hb_pvalue_zero_risk_is_one_minus_alpha_to_the_n():
    for n in (10, 50, 200):
        assert hb_pvalue(0.0, n, ALPHA) == pytest.approx((1 - ALPHA) ** n, rel=1e-6)


def test_hb_pvalue_minimum_n_for_certification():
    # (0.85)^18 = 0.0537 > 0.05 ; (0.85)^19 = 0.0456 <= 0.05
    assert hb_pvalue(0.0, 18, ALPHA) > DELTA
    assert hb_pvalue(0.0, 19, ALPHA) <= DELTA


def test_hb_pvalue_is_one_when_risk_at_or_above_alpha():
    assert hb_pvalue(ALPHA, 100, ALPHA) == 1.0
    assert hb_pvalue(0.30, 100, ALPHA) == 1.0


def test_hb_pvalue_nondecreasing_in_risk():
    risks = np.linspace(0.0, 0.3, 61)
    ps = [hb_pvalue(float(r), 100, ALPHA) for r in risks]
    assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:]))


def test_hb_pvalue_in_unit_interval():
    for r in (0.0, 0.01, 0.1, 0.149, 0.5, 1.0):
        p = hb_pvalue(r, 80, ALPHA)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------- ordering
VALS3 = np.array([0.0, 0.5, 1.0])


def test_tau_plus_gamma_order_is_permissive_first_with_tau_tiebreak():
    order = ltt_fst_ordering(VALS3, VALS3, "tau_plus_gamma")
    assert order == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (1, 2), (2, 1), (2, 2)]


def test_diagonal_order():
    assert ltt_fst_ordering(VALS3, VALS3, "diagonal") == [(0, 0), (1, 1), (2, 2)]


def test_lex_tau_order():
    assert ltt_fst_ordering(VALS3, VALS3, "lex_tau")[:3] == [(0, 0), (0, 1), (0, 2)]


def test_unknown_ordering_raises():
    with pytest.raises(ValueError):
        ltt_fst_ordering(VALS3, VALS3, "nope")


# ---------------------------------------------------------------- fixed-sequence selection
def _grid(n_docs, bad_cells, workload_fn):
    loss = np.zeros((n_docs, 3, 3))
    for (i, j) in bad_cells:
        loss[:, i, j] = 1.0
    wl = np.zeros_like(loss)
    for i in range(3):
        for j in range(3):
            wl[:, i, j] = workload_fn(i, j)
    return loss, wl


ORDER3 = ltt_fst_ordering(VALS3, VALS3, "tau_plus_gamma")


def test_certifies_prefix_and_deploys_min_workload():
    # cells with i+j >= 3 have loss 1 -> walk certifies the 6 cells with i+j <= 2
    loss, wl = _grid(100, [(1, 2), (2, 1), (2, 2)], lambda i, j: 10 - (i + j) + 0.1 * i)
    i, j, n_cert, ok = ltt_fst_select_from_grid(loss, wl, ORDER3, ALPHA, DELTA, "min_workload")
    assert ok and n_cert == 6
    assert (i, j) == (0, 2)  # workload 8.0 < (1,1)=8.1 < (2,0)=8.2


def test_deploy_last_returns_final_certified_cell():
    loss, wl = _grid(100, [(1, 2), (2, 1), (2, 2)], lambda i, j: 10 - (i + j) + 0.1 * i)
    i, j, n_cert, ok = ltt_fst_select_from_grid(loss, wl, ORDER3, ALPHA, DELTA, "last")
    assert (i, j) == (2, 0) and n_cert == 6


def test_stops_at_first_non_rejection_even_if_later_cells_pass():
    loss, wl = _grid(100, [(0, 1)], lambda i, j: 10 - (i + j))
    i, j, n_cert, ok = ltt_fst_select_from_grid(loss, wl, ORDER3, ALPHA, DELTA)
    assert ok and n_cert == 1 and (i, j) == (0, 0)


def test_nothing_certified_with_tiny_n():
    loss, wl = _grid(10, [], lambda i, j: 1.0)
    assert ltt_fst_select_from_grid(loss, wl, ORDER3, ALPHA, DELTA) == (0, 0, 0, False)


def test_min_workload_tie_breaks_by_test_order():
    loss, wl = _grid(100, [], lambda i, j: 5.0)  # all certified, all tied
    i, j, n_cert, ok = ltt_fst_select_from_grid(loss, wl, ORDER3, ALPHA, DELTA)
    assert n_cert == 9 and (i, j) == ORDER3[0]


def test_unknown_deploy_raises():
    loss, wl = _grid(100, [], lambda i, j: 1.0)
    with pytest.raises(ValueError):
        ltt_fst_select_from_grid(loss, wl, ORDER3, ALPHA, DELTA, "bogus")


# ---------------------------------------------------------------- end-to-end selector
def _doc(n_extra=3, omitted=True):
    """One important+uncovered sentence (if omitted) plus n_extra unimportant covered ones."""
    imp = [0.92] + [0.10] * n_extra
    cov = [0.08] + [0.90] * n_extra
    imp_lab = [1 if omitted else 0] + [0] * n_extra
    cov_lab = [0] + [1] * n_extra
    return {"importance_probs": imp, "coverage_probs": cov,
            "importance_labels": imp_lab, "coverage_labels": cov_lab}


def test_end_to_end_surfaces_only_the_omission():
    r = select_omission_threshold_ltt_fst([_doc() for _ in range(40)], ALPHA, DELTA)
    assert r["is_feasible"] and r["empirical_risk"] == 0.0
    assert r["workload"] == 1.0
    assert r["n_docs"] == 40 and r["grid_size"] == 21 * 21
    assert r["hb_pvalue"] <= DELTA
    assert r["threshold_type"] == "omission_ltt_fst_tau_plus_gamma_min_workload"


def test_small_n_falls_back_to_surface_everything():
    r = select_omission_threshold_ltt_fst([_doc() for _ in range(18)], ALPHA, DELTA)
    assert not r["is_feasible"] and r["n_certified"] == 0
    assert (r["tau"], r["gamma"]) == (0.0, 0.0)
    assert r["workload"] == 4.0  # every sentence surfaced


def test_docs_without_true_omissions_count_as_zero_loss():
    docs = [_doc(omitted=False) for _ in range(40)]
    loss, _ = omission_grid_matrices(docs, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.all(loss == 0.0) and not np.any(np.isnan(loss))


def test_reported_thresholds_are_grid_values():
    r = select_omission_threshold_ltt_fst([_doc() for _ in range(40)], ALPHA, DELTA)
    grid = np.arange(0.0, 1.0 + 0.05, 0.05)
    assert any(r["tau"] == float(g) for g in grid)
    assert any(r["gamma"] == float(g) for g in grid)
