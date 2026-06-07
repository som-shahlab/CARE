#!/usr/bin/env python3
"""
Phase 4: Test-Time Evaluation with Binary Instance Losses

This script evaluates the conformal risk control system on test documents:
1. Load test documents with oracle labels and vote-rate probabilities
2. Load conformal thresholds (λ*, τ*, γ*) from Phase 3
3. Apply both controllers using binary instance losses
4. Measure test-time violation rates and compare to guarantees
5. Generate visualizations

Key metrics (plan.tex Section 4):
- Violation rate: Fraction of docs with at least one missed error/omission
- Load: Average number of flagged/surfaced sentences per doc
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports

from care import config
from care.utils import (
    load_jsonl,
    load_json,
    save_json,
    setup_logging,
)
from care.calibration import (
    select_factuality_threshold,
    select_factuality_threshold_uncalibrated,
    select_factuality_threshold_devset,
    select_omission_threshold_2d,
    select_omission_threshold_2d_fractional,
    select_omission_threshold_2d_devset,
    select_omission_threshold_1d,
    select_omission_threshold_product,
    select_omission_threshold_score_gated,
    select_omission_threshold_union_bound,
    compute_product_loss_fractional,
    compute_omission_loss_fractional,
)

logger = setup_logging('Phase4')


# ============================================================================
# Binary Instance Loss Functions (matching Phase 3)
# ============================================================================

def compute_factuality_loss_binary(
    factuality_labels: List[int],
    factuality_probs: List[float],
    threshold: float,
) -> int:
    """
    Compute binary instance-level factuality loss.

    L_fact^inst(λ; X) = 1{∃v: Y_fact(v)=0 AND p̂_fact(v) > λ}

    Returns: 0 or 1
    """
    if len(factuality_labels) == 0:
        return 0

    labels = np.array(factuality_labels)
    probs = np.array(factuality_probs)

    # Flagged: p̂_fact ≤ λ
    flagged = probs <= threshold

    # Unflagged hallucinations
    hallucinations = (labels == 0)
    unflagged_hallucinations = hallucinations & (~flagged)

    return int(unflagged_hallucinations.any())


def compute_omission_loss_binary(
    importance_labels: List[int],
    coverage_labels: List[int],
    importance_probs: List[float],
    coverage_probs: List[float],
    tau: float,
    gamma: float,
) -> int:
    """
    Compute binary instance-level omission loss with 2D thresholds.

    L_omit^inst(τ,γ; X) = 1{∃u ∈ O_true: u ∉ O_{τ,γ}(X)}

    Returns: 0 or 1
    """
    if len(importance_labels) == 0:
        return 0

    imp_labels = np.array(importance_labels)
    cov_labels = np.array(coverage_labels)
    imp_probs = np.array(importance_probs)
    cov_probs = np.array(coverage_probs)

    # True omissions: important AND NOT covered
    true_omissions = (imp_labels == 1) & (cov_labels == 0)

    if not true_omissions.any():
        return 0

    # Surfaced: p̂_imp ≥ τ AND p̂_non-cov ≥ γ
    non_cov_probs = 1.0 - cov_probs
    surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)

    # Unsurfaced true omissions
    unsurfaced = true_omissions & (~surfaced)

    return int(unsurfaced.any())


# ============================================================================
# Controller Application
# ============================================================================

def apply_factuality_controller(
    documents: List[Dict[str, Any]],
    lambda_threshold: float,
) -> Dict[str, Any]:
    """
    Apply factuality controller to test documents.

    Flagged set: F_λ(X) = {v : p̂_fact(v) ≤ λ}

    Args:
        documents: Test documents with factuality_probs and factuality_labels
        lambda_threshold: λ* from Phase 3

    Returns:
        Dictionary with test-time performance metrics
    """
    n_docs = len(documents)
    total_sents = 0
    n_flagged = 0
    n_errors_total = 0
    n_errors_flagged = 0
    n_errors_unflagged = 0

    instance_losses = []  # Binary losses per document

    for doc in documents:
        probs = np.array(doc['factuality_probs'])
        labels = np.array(doc['factuality_labels'])

        # Flagged set: p̂_fact ≤ λ
        flagged = probs <= lambda_threshold
        not_flagged = ~flagged

        # Errors (hallucinations)
        errors = (labels == 0)

        # Counts
        total_sents += len(probs)
        n_flagged += flagged.sum()
        n_errors_total += errors.sum()
        n_errors_flagged += (errors & flagged).sum()
        n_errors_unflagged += (errors & not_flagged).sum()

        # Binary instance loss
        loss = compute_factuality_loss_binary(
            doc['factuality_labels'],
            doc['factuality_probs'],
            lambda_threshold
        )
        instance_losses.append(loss)

    # Violation rate (primary metric)
    violation_rate = np.mean(instance_losses)

    # Sentence-level diagnostics (optional per plan.tex)
    # Precision: What fraction of flagged sentences are actual errors?
    flagging_precision = n_errors_flagged / n_flagged if n_flagged > 0 else 0.0
    # Recall: What fraction of errors are flagged?
    flagging_recall = n_errors_flagged / n_errors_total if n_errors_total > 0 else 1.0

    return {
        'lambda': float(lambda_threshold),
        'n_docs': int(n_docs),
        'total_sentences': int(total_sents),
        'flagged': int(n_flagged),
        'flagged_per_doc': float(n_flagged / n_docs),
        'pct_flagged': float(100 * n_flagged / total_sents) if total_sents > 0 else 0,
        'errors_total': int(n_errors_total),
        'errors_flagged': int(n_errors_flagged),
        'errors_unflagged': int(n_errors_unflagged),
        'errors_per_doc': float(n_errors_total / n_docs),
        'unflagged_errors_per_doc': float(n_errors_unflagged / n_docs),
        'violation_rate': float(violation_rate),  # Primary: fraction of docs with missed error
        'instance_losses': [int(x) for x in instance_losses],
        # Sentence-level diagnostics (plan.tex optional metrics)
        'flagging_precision': float(flagging_precision),  # errors_flagged / flagged
        'flagging_recall': float(flagging_recall),        # errors_flagged / errors_total
    }


def apply_omission_controller(
    documents: List[Dict[str, Any]],
    tau: float,
    gamma: float,
) -> Dict[str, Any]:
    """
    Apply 2D omission controller to test documents.

    Surfaced set: O_{τ,γ}(X) = {u : p̂_imp(u) ≥ τ AND p̂_non-cov(u) ≥ γ}

    Args:
        documents: Test documents with importance/coverage probs and labels
        tau: τ* from Phase 3
        gamma: γ* from Phase 3

    Returns:
        Dictionary with test-time performance metrics
    """
    n_docs = len(documents)
    total_source = 0
    n_surfaced = 0
    n_true_omissions_total = 0
    n_true_omissions_surfaced = 0
    n_true_omissions_unsurfaced = 0

    instance_losses = []  # Binary losses per document
    fractional_losses = []  # Fractional losses per document

    for doc in documents:
        imp_labels = np.array(doc['importance_labels'])
        cov_labels = np.array(doc['coverage_labels'])
        imp_probs = np.array(doc['importance_probs'])
        cov_probs = np.array(doc['coverage_probs'])

        # Surfaced set
        non_cov_probs = 1.0 - cov_probs
        surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)

        # True omissions (per Oracle)
        true_omissions = (imp_labels == 1) & (cov_labels == 0)

        # Counts
        total_source += len(imp_probs)
        n_surfaced += surfaced.sum()
        n_true_omissions_total += true_omissions.sum()
        n_true_omissions_surfaced += (true_omissions & surfaced).sum()
        n_true_omissions_unsurfaced += (true_omissions & ~surfaced).sum()

        # Binary instance loss
        loss = compute_omission_loss_binary(
            doc['importance_labels'],
            doc['coverage_labels'],
            doc['importance_probs'],
            doc['coverage_probs'],
            tau,
            gamma
        )
        instance_losses.append(loss)

        # Fractional instance loss
        frac_loss = compute_omission_loss_fractional(
            doc['importance_labels'],
            doc['coverage_labels'],
            doc['importance_probs'],
            doc['coverage_probs'],
            tau,
            gamma
        )
        fractional_losses.append(frac_loss)

    # Violation rates
    violation_rate = np.mean(instance_losses)  # Binary (secondary)
    fractional_violation_rate = np.mean(fractional_losses)  # Fractional (primary)

    # Sentence-level diagnostics (optional per plan.tex)
    # Precision: What fraction of surfaced sentences are true omissions?
    surfacing_precision = (
        n_true_omissions_surfaced / n_surfaced
        if n_surfaced > 0 else 0.0
    )
    # Recall: What fraction of true omissions are surfaced?
    surfacing_recall = (
        n_true_omissions_surfaced / n_true_omissions_total
        if n_true_omissions_total > 0 else 1.0
    )

    return {
        'tau': float(tau),
        'gamma': float(gamma),
        'n_docs': int(n_docs),
        'total_source': int(total_source),
        'surfaced': int(n_surfaced),
        'surfaced_per_doc': float(n_surfaced / n_docs),
        'pct_surfaced': float(100 * n_surfaced / total_source) if total_source > 0 else 0,
        'true_omissions_total': int(n_true_omissions_total),
        'true_omissions_per_doc': float(n_true_omissions_total / n_docs),
        'true_omissions_surfaced': int(n_true_omissions_surfaced),
        'true_omissions_unsurfaced': int(n_true_omissions_unsurfaced),
        'violation_rate': float(violation_rate),  # Binary: fraction of docs with ANY missed omission
        'fractional_violation_rate': float(fractional_violation_rate),  # Fractional: mean fraction missed per doc
        'instance_losses': [int(x) for x in instance_losses],
        'fractional_losses': [float(x) for x in fractional_losses],
        # Sentence-level diagnostics (plan.tex optional metrics)
        'surfacing_precision': float(surfacing_precision),  # true_omissions_surfaced / surfaced
        'surfacing_recall': float(surfacing_recall),        # true_omissions_surfaced / true_omissions_total
    }


# ============================================================================
# Guarantee Evaluation
# ============================================================================

def evaluate_guarantees(
    factuality_results: Dict[str, Any],
    omission_results: Dict[str, Any],
    alpha_fact: float,
    alpha_omit: float,
) -> Dict[str, Any]:
    """
    Evaluate whether CRC guarantees hold on test set.

    CRC guarantees: E[L^inst] ≤ α

    Args:
        factuality_results: Results from apply_factuality_controller
        omission_results: Results from apply_omission_controller
        alpha_fact: Factuality risk budget
        alpha_omit: Omission risk budget

    Returns:
        Dictionary with guarantee evaluation
    """
    fact_violation_rate = factuality_results['violation_rate']
    omit_fractional_rate = omission_results['fractional_violation_rate']
    omit_binary_rate = omission_results['violation_rate']

    fact_holds = fact_violation_rate <= alpha_fact
    omit_holds = omit_fractional_rate <= alpha_omit  # Fractional is the primary guarantee

    return {
        'factuality': {
            'alpha': alpha_fact,
            'violation_rate': fact_violation_rate,
            'guarantee_holds': fact_holds,
            'slack': alpha_fact - fact_violation_rate,
        },
        'omission': {
            'alpha': alpha_omit,
            'violation_rate': omit_fractional_rate,  # Fractional (primary)
            'binary_violation_rate': omit_binary_rate,  # Binary (secondary)
            'guarantee_holds': omit_holds,
            'slack': alpha_omit - omit_fractional_rate,
        },
        'both_hold': fact_holds and omit_holds,
    }


# ============================================================================
# Alpha Sweep and Visualization
# ============================================================================

def wilson_score_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    This is the recommended method for CI on proportions (Agresti & Coull 1998).
    Unlike Wald (normal approximation), Wilson:
    - Never gives bounds outside [0, 1]
    - Has correct coverage even for small n or extreme p
    - Works well when k=0 or k=n

    Args:
        k: Number of successes (documents with loss=1)
        n: Total trials (documents)
        z: Z-score for confidence level (1.96 for 95% CI)

    Returns:
        (lower, upper) bounds of CI
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = k / n

    # Wilson score formula
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (float(lower), float(upper))


def run_alpha_sweep_test(
    calibration_docs: List[Dict[str, Any]],
    test_docs: List[Dict[str, Any]],
    alpha_values: List[float],
    grid_resolution: float = 0.05,
    compute_ci: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run proper out-of-sample test evaluation for multiple alpha values.

    For each alpha:
    1. Calibrate thresholds on CALIBRATION set (proper CRC)
    2. Evaluate on TEST set (held-out)

    This gives true out-of-sample validation of the CRC guarantee.

    Uses Wilson score CI for violation rates (binomial proportions).

    Args:
        calibration_docs: Documents for threshold calibration
        test_docs: Held-out documents for evaluation
        alpha_values: List of alpha values to sweep
        grid_resolution: Grid resolution for 2D omission search
        compute_ci: Whether to compute Wilson score CIs

    Returns:
        List of results for each alpha value
    """
    from tqdm import tqdm

    n_test = len(test_docs)
    n_calib = len(calibration_docs)
    results = []

    logger.info(f"Alpha sweep: calibrating on {n_calib} docs, testing on {n_test} docs")

    for alpha in tqdm(alpha_values, desc="Alpha sweep"):
        # CALIBRATE thresholds on CALIBRATION set
        fact_result = select_factuality_threshold(calibration_docs, alpha)
        omit_result = select_omission_threshold_2d_fractional(calibration_docs, alpha, grid_resolution)

        # Calibrate Controller A baselines
        fact_uncal = select_factuality_threshold_uncalibrated(calibration_docs, alpha)
        fact_devset = select_factuality_threshold_devset(calibration_docs, alpha, grid_resolution=0.01)

        # Calibrate Controller B baselines
        omit_1d = select_omission_threshold_1d(calibration_docs, alpha, grid_resolution=0.01)
        omit_product = select_omission_threshold_product(calibration_docs, alpha, grid_resolution=0.01)
        omit_score_gated = select_omission_threshold_score_gated(calibration_docs, alpha, grid_resolution=0.01)
        omit_union_bound = select_omission_threshold_union_bound(calibration_docs, alpha, grid_resolution=0.01)
        omit_devset = select_omission_threshold_2d_devset(calibration_docs, alpha, grid_resolution=0.05)

        lambda_thresh = fact_result['threshold']
        tau_thresh = omit_result['tau']
        gamma_thresh = omit_result['gamma']

        # EVALUATE on TEST set (held-out)
        fact_violations = 0
        fact_flagged = 0
        fact_errors_total = 0
        fact_errors_flagged = 0
        omit_violations = 0
        omit_surfaced = 0
        omit_true_omissions_total = 0
        omit_true_omissions_surfaced = 0
        omit_frac_losses = []

        # Controller A baseline counters
        fact_uncal_violations = 0
        fact_uncal_flagged = 0
        fact_devset_violations = 0
        fact_devset_flagged = 0

        # Controller B baseline counters
        omit_1d_violations = 0
        omit_1d_surfaced = 0
        omit_1d_frac_losses = []
        omit_product_violations = 0
        omit_product_surfaced = 0
        omit_product_frac_losses = []
        omit_sg_violations = 0
        omit_sg_surfaced = 0
        omit_sg_frac_losses = []
        omit_ub_violations = 0
        omit_ub_surfaced = 0
        omit_ub_frac_losses = []
        omit_devset_surfaced = 0
        omit_devset_frac_losses = []

        for doc in test_docs:
            # Factuality
            probs = np.array(doc['factuality_probs'])
            labels = np.array(doc['factuality_labels'])
            flagged = probs <= lambda_thresh
            fact_flagged += flagged.sum()
            errors = (labels == 0)
            fact_errors_total += errors.sum()
            fact_errors_flagged += (errors & flagged).sum()
            if errors.any() and (errors & ~flagged).any():
                fact_violations += 1

            # Controller A baselines
            flagged_uncal = probs <= fact_uncal['threshold']
            fact_uncal_flagged += flagged_uncal.sum()
            if errors.any() and (errors & ~flagged_uncal).any():
                fact_uncal_violations += 1

            flagged_devset = probs <= fact_devset['threshold']
            fact_devset_flagged += flagged_devset.sum()
            if errors.any() and (errors & ~flagged_devset).any():
                fact_devset_violations += 1

            # Omission (2D Joint)
            imp_probs = np.array(doc['importance_probs'])
            cov_probs = np.array(doc['coverage_probs'])
            imp_labels = np.array(doc['importance_labels'])
            cov_labels = np.array(doc['coverage_labels'])
            non_cov_probs = 1.0 - cov_probs
            true_omissions = (imp_labels == 1) & (cov_labels == 0)

            surfaced_2d = (imp_probs >= tau_thresh) & (non_cov_probs >= gamma_thresh)
            omit_surfaced += surfaced_2d.sum()
            omit_true_omissions_total += true_omissions.sum()
            omit_true_omissions_surfaced += (true_omissions & surfaced_2d).sum()
            if true_omissions.any() and (true_omissions & ~surfaced_2d).any():
                omit_violations += 1
            # Fractional loss
            n_true = true_omissions.sum()
            if n_true == 0:
                omit_frac_losses.append(0.0)
            else:
                omit_frac_losses.append(float((true_omissions & ~surfaced_2d).sum() / n_true))

            # 1D Importance baseline
            surfaced_1d = imp_probs >= omit_1d['threshold']
            omit_1d_surfaced += surfaced_1d.sum()
            if true_omissions.any() and (true_omissions & ~surfaced_1d).any():
                omit_1d_violations += 1
            if n_true == 0:
                omit_1d_frac_losses.append(0.0)
            else:
                omit_1d_frac_losses.append(float((true_omissions & ~surfaced_1d).sum() / n_true))

            # Product baseline
            s_prod = imp_probs * non_cov_probs
            surfaced_prod = s_prod >= omit_product['threshold']
            omit_product_surfaced += surfaced_prod.sum()
            if true_omissions.any() and (true_omissions & ~surfaced_prod).any():
                omit_product_violations += 1
            if n_true == 0:
                omit_product_frac_losses.append(0.0)
            else:
                omit_product_frac_losses.append(float((true_omissions & ~surfaced_prod).sum() / n_true))

            # Score-gated baseline
            surfaced_sg = (imp_probs >= omit_score_gated['threshold']) & (cov_probs < omit_score_gated['coverage_gate'])
            omit_sg_surfaced += surfaced_sg.sum()
            if true_omissions.any() and (true_omissions & ~surfaced_sg).any():
                omit_sg_violations += 1
            if n_true == 0:
                omit_sg_frac_losses.append(0.0)
            else:
                omit_sg_frac_losses.append(float((true_omissions & ~surfaced_sg).sum() / n_true))

            # Union bound baseline
            surfaced_ub = (imp_probs >= omit_union_bound['tau']) & (non_cov_probs >= omit_union_bound['gamma'])
            omit_ub_surfaced += surfaced_ub.sum()
            if true_omissions.any() and (true_omissions & ~surfaced_ub).any():
                omit_ub_violations += 1
            if n_true == 0:
                omit_ub_frac_losses.append(0.0)
            else:
                omit_ub_frac_losses.append(float((true_omissions & ~surfaced_ub).sum() / n_true))

            # Devset-tuned 2D baseline (no CRC correction)
            surfaced_devset = (imp_probs >= omit_devset['tau']) & (non_cov_probs >= omit_devset['gamma'])
            omit_devset_surfaced += surfaced_devset.sum()
            if n_true == 0:
                omit_devset_frac_losses.append(0.0)
            else:
                omit_devset_frac_losses.append(float((true_omissions & ~surfaced_devset).sum() / n_true))

        result = {
            'alpha': float(alpha),
            'n_calibration': n_calib,
            'n_test': n_test,
            'factuality': {
                'violation': float(fact_violations / n_test),
                'holds': bool((fact_violations / n_test) <= alpha),
                'workload': float(fact_flagged / n_test),
                'threshold': float(lambda_thresh),
                'calib_risk': float(fact_result['adjusted_risk']),
                'precision': float(fact_errors_flagged / fact_flagged) if fact_flagged > 0 else 0.0,
                'recall': float(fact_errors_flagged / fact_errors_total) if fact_errors_total > 0 else 1.0,
            },
            'factuality_uncalibrated': {
                'violation': float(fact_uncal_violations / n_test),
                'workload': float(fact_uncal_flagged / n_test),
                'threshold': float(fact_uncal['threshold']),
            },
            'factuality_devset': {
                'violation': float(fact_devset_violations / n_test),
                'workload': float(fact_devset_flagged / n_test),
                'threshold': float(fact_devset['threshold']),
            },
            'omission': {
                'violation': float(np.mean(omit_frac_losses)),  # Fractional loss (primary)
                'binary_violation': float(omit_violations / n_test),  # Binary (secondary)
                'holds': bool(np.mean(omit_frac_losses) <= alpha),  # Test-set fractional check
                'workload': float(omit_surfaced / n_test),
                'tau': float(tau_thresh),
                'gamma': float(gamma_thresh),
                'calib_risk': float(omit_result['adjusted_risk']),
                'loss_type': 'fractional',
                'precision': float(omit_true_omissions_surfaced / omit_surfaced) if omit_surfaced > 0 else 0.0,
                'recall': float(omit_true_omissions_surfaced / omit_true_omissions_total) if omit_true_omissions_total > 0 else 1.0,
            },
            'omission_1d': {
                'violation': float(np.mean(omit_1d_frac_losses)),
                'binary_violation': float(omit_1d_violations / n_test),
                'holds': bool(np.mean(omit_1d_frac_losses) <= alpha),
                'workload': float(omit_1d_surfaced / n_test),
                'tau': float(omit_1d['threshold']),
            },
            'omission_product': {
                'violation': float(np.mean(omit_product_frac_losses)),
                'binary_violation': float(omit_product_violations / n_test),
                'holds': bool(np.mean(omit_product_frac_losses) <= alpha),
                'workload': float(omit_product_surfaced / n_test),
                'beta': float(omit_product['threshold']),
            },
            'omission_score_gated': {
                'violation': float(np.mean(omit_sg_frac_losses)),
                'binary_violation': float(omit_sg_violations / n_test),
                'workload': float(omit_sg_surfaced / n_test),
                'tau': float(omit_score_gated['threshold']),
                'gate': float(omit_score_gated['coverage_gate']),
            },
            'omission_union_bound': {
                'violation': float(np.mean(omit_ub_frac_losses)),
                'binary_violation': float(omit_ub_violations / n_test),
                'holds': bool(np.mean(omit_ub_frac_losses) <= alpha),
                'workload': float(omit_ub_surfaced / n_test),
                'tau': float(omit_union_bound['tau']),
                'gamma': float(omit_union_bound['gamma']),
            },
            'omission_devset': {
                'violation': float(np.mean(omit_devset_frac_losses)),
                'holds': bool(np.mean(omit_devset_frac_losses) <= alpha),
                'workload': float(omit_devset_surfaced / n_test),
                'tau': float(omit_devset['tau']),
                'gamma': float(omit_devset['gamma']),
                'has_formal_guarantee': False,
            },
        }

        # Compute Wilson score CI on violation rates (binomial proportions)
        if compute_ci:
            fact_ci = wilson_score_ci(fact_violations, n_test)
            result['factuality']['violation_ci'] = fact_ci
            # Wilson CI only valid for binary (binomial) proportions
            # Store binary omission CI separately — not on the fractional 'omission' dict
            omit_binary_ci = wilson_score_ci(omit_violations, n_test)
            result['omission']['binary_violation_ci'] = omit_binary_ci

        results.append(result)

    return results


def plot_calibration_curves(
    sweep_results: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Plot Risk Budget vs Workload curves for factuality and omission.

    Shows how workload (sentences flagged/surfaced per note) varies with α.
    Thresholds are calibrated on calibration set; workload is measured on test set.
    """
    summary_sents = [len(d['factuality_probs']) for d in documents]
    source_sents = [len(d['importance_probs']) for d in documents]

    mean_summary = np.mean(summary_sents)
    median_summary = np.median(summary_sents)
    mean_source = np.mean(source_sents)
    median_source = np.median(source_sents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Factuality
    alphas_f = [r['alpha'] * 100 for r in sweep_results]
    workloads_f = [r['factuality']['workload'] for r in sweep_results]
    holds_f = [r['factuality']['holds'] for r in sweep_results]

    ax1.step(alphas_f, workloads_f, where='post', color='#3498db', linewidth=2)
    for a, w, h in zip(alphas_f, workloads_f, holds_f):
        color = 'green' if h else 'red'
        ax1.scatter([a], [w], c=color, s=60, zorder=5)

    ax1.set_xlabel('Risk Budget α (%)', fontsize=12)
    ax1.set_ylabel('Sentences Flagged per Note', fontsize=12)
    ax1.set_title('Hallucination: Risk Budget vs Workload (Test)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max(alphas_f) + 2)
    ax1.set_ylim(0, max(workloads_f) * 1.1 if workloads_f else 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    legend_text = f'Summary sentences per note:\nMean: {mean_summary:.1f}  |  Median: {median_summary:.0f}\n\n● green = guarantee holds\n● red = guarantee violated'
    ax1.text(0.98, 0.98, legend_text, transform=ax1.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    # Omission
    alphas_o = [r['alpha'] * 100 for r in sweep_results]
    workloads_o = [r['omission']['workload'] for r in sweep_results]
    holds_o = [r['omission']['holds'] for r in sweep_results]

    ax2.step(alphas_o, workloads_o, where='post', color='#9b59b6', linewidth=2)
    for a, w, h in zip(alphas_o, workloads_o, holds_o):
        color = 'green' if h else 'red'
        ax2.scatter([a], [w], c=color, s=60, zorder=5)

    ax2.set_xlabel('Risk Budget α (%)', fontsize=12)
    ax2.set_ylabel('Sentences Surfaced per Note', fontsize=12)
    ax2.set_title('Omission (Fractional): Risk Budget vs Workload (Test)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(alphas_o) + 2)
    ax2.set_ylim(0, max(workloads_o) * 1.1 if workloads_o else 1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    legend_text = f'Source sentences per note:\nMean: {mean_source:.1f}  |  Median: {median_source:.0f}\n\n● green = guarantee holds\n● red = guarantee violated'
    ax2.text(0.98, 0.98, legend_text, transform=ax2.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()
    output_file = output_dir / 'calibration_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Calibration curves saved to {output_file}")


def plot_alpha_vs_violation(
    sweep_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Plot alpha vs test violation rate with safe/violated regions and Wilson score CI.

    This plot shows TRUE out-of-sample validation:
    - For each α: thresholds are calibrated on CALIBRATION set
    - Violation rate is measured on HELD-OUT TEST set

    If CRC guarantees hold, violation rate should be ≤ α (below diagonal).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    alphas = [r['alpha'] for r in sweep_results]

    # Check if CI data is available
    has_ci = 'violation_ci' in sweep_results[0].get('factuality', {})

    # Factuality
    violations_f = [r['factuality']['violation'] for r in sweep_results]
    holds_f = [r['factuality']['holds'] for r in sweep_results]

    ax1.fill_between([0, 0.55], [0, 0.55], [0.55, 0.55], alpha=0.15, color='red', label='Violated region')
    ax1.fill_between([0, 0.55], [0, 0], [0, 0.55], alpha=0.15, color='green', label='Safe region')
    ax1.plot([0, 0.55], [0, 0.55], 'k--', alpha=0.5, linewidth=1.5, label='y = α')

    # Plot CI band if available
    if has_ci:
        ci_lower_f = [r['factuality']['violation_ci'][0] for r in sweep_results]
        ci_upper_f = [r['factuality']['violation_ci'][1] for r in sweep_results]
        ax1.fill_between(alphas, ci_lower_f, ci_upper_f, alpha=0.25, color='#3498db', label='95% Wilson CI')

    ax1.plot(alphas, violations_f, 'o-', color='#3498db', linewidth=2, markersize=8)

    for a, v, h in zip(alphas, violations_f, holds_f):
        color = 'green' if h else 'red'
        ax1.scatter([a], [v], c=color, s=100, zorder=5, edgecolors='white', linewidths=1)

    ax1.set_xlabel('Target α', fontsize=12)
    ax1.set_ylabel('Test Violation Rate', fontsize=12)
    ax1.set_title('Hallucination', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 0.55)
    ax1.set_ylim(0, 0.55)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Omission (fractional loss is the primary metric)
    violations_o = [r['omission']['violation'] for r in sweep_results]  # Already fractional
    holds_o = [r['omission']['holds'] for r in sweep_results]

    ax2.fill_between([0, 0.55], [0, 0.55], [0.55, 0.55], alpha=0.15, color='red', label='Violated region')
    ax2.fill_between([0, 0.55], [0, 0], [0, 0.55], alpha=0.15, color='green', label='Safe region')
    ax2.plot([0, 0.55], [0, 0.55], 'k--', alpha=0.5, linewidth=1.5, label='y = α')

    # No Wilson CI for fractional loss (not binomial) — omit CI band for omission

    ax2.plot(alphas, violations_o, 'o-', color='#9b59b6', linewidth=2, markersize=8)

    for a, v, h in zip(alphas, violations_o, holds_o):
        color = 'green' if h else 'red'
        ax2.scatter([a], [v], c=color, s=100, zorder=5, edgecolors='white', linewidths=1)

    ax2.set_xlabel('Target α', fontsize=12)
    ax2.set_ylabel('Test Fractional Loss', fontsize=12)
    ax2.set_title('Omission (Fractional Loss)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 0.55)
    ax2.set_ylim(0, 0.55)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'alpha_vs_violation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Alpha vs violation plot saved to {output_file}")


def plot_precision_recall_workload(
    sweep_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Plot precision, recall, and workload vs alpha for both controllers.

    Creates a 2x3 grid:
    - Top row: Factuality (precision, recall, workload)
    - Bottom row: Omission (precision, recall, workload)
    """
    alphas = [r['alpha'] for r in sweep_results]
    alphas_pct = [a * 100 for a in alphas]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Top row: Factuality ---
    fact_precision = [r['factuality']['precision'] * 100 for r in sweep_results]
    fact_recall = [r['factuality']['recall'] * 100 for r in sweep_results]
    fact_workload = [r['factuality']['workload'] for r in sweep_results]

    # Factuality Precision
    ax = axes[0, 0]
    ax.plot(alphas_pct, fact_precision, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax.set_xlabel('Risk Budget α (%)', fontsize=11)
    ax.set_ylabel('Precision (%)', fontsize=11)
    ax.set_title('Hallucination Precision', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Factuality Recall
    ax = axes[0, 1]
    ax.plot(alphas_pct, fact_recall, 's-', color='#e74c3c', linewidth=2, markersize=6)
    ax.set_xlabel('Risk Budget α (%)', fontsize=11)
    ax.set_ylabel('Recall (%)', fontsize=11)
    ax.set_title('Hallucination Recall', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Factuality Workload
    ax = axes[0, 2]
    ax.plot(alphas_pct, fact_workload, 'D-', color='#e74c3c', linewidth=2, markersize=6)
    ax.set_xlabel('Risk Budget α (%)', fontsize=11)
    ax.set_ylabel('Sentences Flagged / Note', fontsize=11)
    ax.set_title('Hallucination Workload', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(fact_workload) * 1.15 if fact_workload else 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Bottom row: Omission ---
    omit_precision = [r['omission']['precision'] * 100 for r in sweep_results]
    omit_recall = [r['omission']['recall'] * 100 for r in sweep_results]
    omit_workload = [r['omission']['workload'] for r in sweep_results]

    # Omission Precision
    ax = axes[1, 0]
    ax.plot(alphas_pct, omit_precision, 'o-', color='#9b59b6', linewidth=2, markersize=6)
    ax.set_xlabel('Risk Budget α (%)', fontsize=11)
    ax.set_ylabel('Precision (%)', fontsize=11)
    ax.set_title('Omission Precision', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Omission Recall
    ax = axes[1, 1]
    ax.plot(alphas_pct, omit_recall, 's-', color='#9b59b6', linewidth=2, markersize=6)
    ax.set_xlabel('Risk Budget α (%)', fontsize=11)
    ax.set_ylabel('Recall (%)', fontsize=11)
    ax.set_title('Omission Recall', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Omission Workload
    ax = axes[1, 2]
    ax.plot(alphas_pct, omit_workload, 'D-', color='#9b59b6', linewidth=2, markersize=6)
    ax.set_xlabel('Risk Budget α (%)', fontsize=11)
    ax.set_ylabel('Sentences Surfaced / Note', fontsize=11)
    ax.set_title('Omission Workload', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(omit_workload) * 1.15 if omit_workload else 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_file = output_dir / 'precision_recall_workload.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Precision/recall/workload plot saved to {output_file}")


def generate_test_summary(
    factuality_results: Dict[str, Any],
    omission_results: Dict[str, Any],
    guarantee_eval: Dict[str, Any],
    thresholds: Dict[str, Any],
    output_dir: Path,
):
    """
    Generate a markdown summary of test results.
    """
    fact_holds = '✓' if guarantee_eval['factuality']['guarantee_holds'] else '✗'
    omit_holds = '✓' if guarantee_eval['omission']['guarantee_holds'] else '✗'
    both_hold = '✓ Both guarantees hold!' if guarantee_eval['both_hold'] else '✗ At least one guarantee violated'

    summary = f"""# Test Evaluation Summary

## Guarantee Check

| Controller | Target | Actual | Status |
|------------|--------|--------|--------|
| **Factuality** | ≤{guarantee_eval['factuality']['alpha']*100:.0f}% notes with missed error | {guarantee_eval['factuality']['violation_rate']*100:.1f}% | {fact_holds} |
| **Omission** | ≤{guarantee_eval['omission']['alpha']*100:.0f}% mean fractional omission loss | {guarantee_eval['omission']['violation_rate']*100:.1f}% | {omit_holds} |

**Overall: {both_hold}**

## Thresholds Used

- **Factuality**: λ* = {thresholds['lambda']:.2f} (flag sentences with p_fact ≤ λ*)
- **Omission**: τ* = {thresholds['tau']:.2f}, γ* = {thresholds['gamma']:.2f} (surface if p_imp ≥ τ* AND p_non_cov ≥ γ*)

## Factuality Results (n={factuality_results['n_docs']} notes)

| Metric | Value |
|--------|-------|
| Sentences flagged | {factuality_results['flagged']} ({factuality_results['pct_flagged']:.1f}%) |
| **Flagged per note** | **{factuality_results['flagged_per_doc']:.1f}** |
| Total errors (hallucinations) | {factuality_results['errors_total']} |
| Errors caught | {factuality_results['errors_flagged']} |
| Errors missed | {factuality_results['errors_unflagged']} |
| **Precision** | **{factuality_results['flagging_precision']*100:.0f}%** |
| **Recall** | **{factuality_results['flagging_recall']*100:.0f}%** |
| Violation rate | {factuality_results['violation_rate']*100:.1f}% ({sum(factuality_results['instance_losses'])}/{factuality_results['n_docs']} notes) |

## Omission Results (n={omission_results['n_docs']} notes)

| Metric | Value |
|--------|-------|
| Sentences surfaced | {omission_results['surfaced']} ({omission_results['pct_surfaced']:.1f}%) |
| **Surfaced per note** | **{omission_results['surfaced_per_doc']:.1f}** |
| True omissions (total) | {omission_results['true_omissions_total']} |
| True omissions surfaced | {omission_results['true_omissions_surfaced']} |
| True omissions missed | {omission_results['true_omissions_unsurfaced']} |
| **Precision** | **{omission_results['surfacing_precision']*100:.1f}%** |
| **Recall** | **{omission_results['surfacing_recall']*100:.0f}%** |
| **Fractional loss** | **{omission_results['fractional_violation_rate']*100:.1f}%** (primary guarantee) |
| Binary violation rate | {omission_results['violation_rate']*100:.1f}% ({sum(omission_results['instance_losses'])}/{omission_results['n_docs']} notes) |

## What This Means for Clinicians

### Factuality (Red Flags)
- A clinician reviewing a note will see **{factuality_results['flagged_per_doc']:.1f} flagged sentences** on average
- **{factuality_results['flagging_precision']*100:.0f}%** of these flags are actual errors (precision)
- **{factuality_results['flagging_recall']*100:.0f}%** of all errors are caught by the flags (recall)
- Only **{factuality_results['violation_rate']*100:.1f}%** of notes will have an error slip through unflagged

### Omissions (Purple Surfaces)
- A clinician reviewing a note will see **{omission_results['surfaced_per_doc']:.1f} surfaced source sentences** to consider
- **{omission_results['surfacing_precision']*100:.1f}%** of these are truly important omissions (precision)
- **{omission_results['surfacing_recall']*100:.0f}%** of all important omissions are surfaced (recall)
- On average, **{omission_results['fractional_violation_rate']*100:.1f}%** of important omissions per note are missed (fractional loss)

## Files Generated
- `guarantee_check.png` - Visual guarantee verification
- `workload_summary.png` - Clinician workload metrics
- `test_results.json` - Machine-readable results
"""

    summary_file = output_dir / 'test_summary.md'
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"Test summary saved to {summary_file}")




# ============================================================================
# Main Pipeline
# ============================================================================

def run_phase4(
    input_file: Path = config.PHASE2_DIR / 'calibrated_scores.jsonl',
    thresholds_file: Path = config.PHASE3_DIR / 'conformal_thresholds.json',
    output_dir: Path = config.PHASE4_DIR,
    split: str = 'test',
):
    """
    Run Phase 4: Test-time evaluation.

    Args:
        input_file: Path to Phase 2 scored documents
        thresholds_file: Path to Phase 3 conformal thresholds
        output_dir: Directory for output files
        split: Which split to use ('test' or 'all'). Default: 'test'
    """
    logger.info("="*80)
    logger.info("Phase 4: Test-Time Evaluation (Binary Instance Losses)")
    logger.info("="*80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ALL documents (we need both calibration and test for proper alpha sweep)
    logger.info(f"Loading documents from {input_file}...")
    all_documents = load_jsonl(input_file)
    all_documents = [d for d in all_documents if d is not None]

    # Get calibration documents (for alpha sweep threshold selection)
    try:
        calibration_docs = config.filter_documents_by_split(all_documents, 'calibration')
        logger.info(f"Calibration split: {len(calibration_docs)} documents")
    except FileNotFoundError:
        logger.warning("split_indices.json not found. Cannot separate calibration/test.")
        calibration_docs = all_documents

    # Get test documents (for evaluation)
    if split == 'test':
        try:
            test_docs = config.filter_documents_by_split(all_documents, 'test')
            logger.info(f"Test split: {len(test_docs)} documents")
        except FileNotFoundError:
            logger.warning("split_indices.json not found. Using all documents as test.")
            test_docs = all_documents
    elif split == 'all':
        test_docs = all_documents
        logger.info(f"Using all {len(test_docs)} documents for evaluation")
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'test' or 'all'")

    # For backward compatibility, use test_docs as 'documents' in the rest of the function
    documents = test_docs
    logger.info(f"Evaluating on {len(documents)} test documents")

    # Load thresholds
    logger.info(f"\nLoading conformal thresholds from {thresholds_file}...")
    thresholds = load_json(thresholds_file)

    # Extract thresholds
    lambda_threshold = thresholds['factuality']['threshold']
    tau_threshold = thresholds['omission']['tau']
    gamma_threshold = thresholds['omission']['gamma']
    alpha_fact = thresholds['config']['alpha_fact']
    alpha_omit = thresholds['config']['alpha_omit']

    logger.info(f"Factuality threshold: λ* = {lambda_threshold:.3f} (α = {alpha_fact})")
    logger.info(f"Omission thresholds: (τ*, γ*) = ({tau_threshold:.3f}, {gamma_threshold:.3f}) (α = {alpha_omit})")

    # Apply controllers
    logger.info("\n" + "="*80)
    logger.info("Applying Controllers to Test Documents")
    logger.info("="*80)

    logger.info("\nController A: Factuality Flagging...")
    factuality_results = apply_factuality_controller(documents, lambda_threshold)

    logger.info("\nController B: Omission Surfacing (2D)...")
    omission_results = apply_omission_controller(documents, tau_threshold, gamma_threshold)

    # Evaluate Controller A baselines
    logger.info("\nEvaluating Controller A baselines on test set...")
    factuality_baseline_results = {}

    if 'factuality_uncalibrated_baseline' in thresholds:
        uncal_lambda = thresholds['factuality_uncalibrated_baseline']['threshold']
        uncal_results = apply_factuality_controller(documents, uncal_lambda)
        factuality_baseline_results['uncalibrated'] = uncal_results
        logger.info(f"  Uncalibrated (lambda={uncal_lambda:.3f}): "
                    f"violation={uncal_results['violation_rate']:.3f}, "
                    f"workload={uncal_results['flagged_per_doc']:.1f}")

    if 'factuality_devset_baseline' in thresholds:
        devset_lambda = thresholds['factuality_devset_baseline']['threshold']
        devset_results = apply_factuality_controller(documents, devset_lambda)
        factuality_baseline_results['devset_tuned'] = devset_results
        logger.info(f"  Dev-set tuned (lambda={devset_lambda:.3f}): "
                    f"violation={devset_results['violation_rate']:.3f}, "
                    f"workload={devset_results['flagged_per_doc']:.1f}")

    # Evaluate Controller B baselines
    logger.info("\nEvaluating Controller B baselines on test set...")
    baseline_results = {}

    # 1D Importance baseline
    if 'omission_1d_baseline' in thresholds:
        tau_1d = thresholds['omission_1d_baseline']['threshold']
        omit_1d_results = apply_omission_controller(documents, tau_1d, 0.0)
        baseline_results['omission_1d'] = omit_1d_results
        logger.info(f"  1D-Imp (tau={tau_1d:.3f}): violation={omit_1d_results['violation_rate']:.3f}, "
                    f"workload={omit_1d_results['surfaced_per_doc']:.1f}")

    # Product composite baseline
    if 'omission_product_baseline' in thresholds:
        beta_prod = thresholds['omission_product_baseline']['threshold']
        # Evaluate product baseline: surface = {u : p_imp * (1-p_cov) >= beta}
        prod_losses = []
        prod_binary_losses = []
        prod_surfaced = 0
        prod_total_source = 0
        prod_true_omissions_total = 0
        prod_true_omissions_surfaced = 0
        for doc in documents:
            imp_probs = np.array(doc['importance_probs'])
            cov_probs = np.array(doc['coverage_probs'])
            imp_labels = np.array(doc['importance_labels'])
            cov_labels = np.array(doc['coverage_labels'])
            s_prod = imp_probs * (1.0 - cov_probs)
            surfaced = s_prod >= beta_prod
            prod_surfaced += surfaced.sum()
            prod_total_source += len(imp_probs)
            true_omissions = (imp_labels == 1) & (cov_labels == 0)
            prod_true_omissions_total += true_omissions.sum()
            prod_true_omissions_surfaced += (true_omissions & surfaced).sum()
            loss = compute_product_loss_fractional(
                doc['importance_labels'], doc['coverage_labels'],
                doc['importance_probs'], doc['coverage_probs'], beta_prod)
            prod_losses.append(loss)
            prod_binary_losses.append(1 if loss > 0 else 0)
        prod_recall = float(prod_true_omissions_surfaced / prod_true_omissions_total) if prod_true_omissions_total > 0 else 1.0
        baseline_results['omission_product'] = {
            'beta': float(beta_prod),
            'n_docs': len(documents),
            'fractional_violation_rate': float(np.mean(prod_losses)),
            'violation_rate': float(np.mean(prod_binary_losses)),
            'surfaced_per_doc': float(prod_surfaced / len(documents)),
            'pct_surfaced': float(100 * prod_surfaced / prod_total_source) if prod_total_source > 0 else 0,
            'true_omissions_total': int(prod_true_omissions_total),
            'true_omissions_surfaced': int(prod_true_omissions_surfaced),
            'surfacing_recall': prod_recall,
            'instance_losses': [float(x) for x in prod_losses],
            'loss_type': 'fractional',
        }
        logger.info(f"  Product (beta={beta_prod:.3f}): violation={np.mean(prod_losses):.3f}, "
                    f"workload={prod_surfaced/len(documents):.1f}")

    # Score-gated baseline
    if 'omission_score_gated_baseline' in thresholds:
        sg_tau = thresholds['omission_score_gated_baseline']['threshold']
        sg_gate = thresholds['omission_score_gated_baseline']['coverage_gate']
        sg_losses = []
        sg_binary_losses = []
        sg_surfaced = 0
        sg_total_source = 0
        sg_true_omissions_total = 0
        sg_true_omissions_surfaced = 0
        for doc in documents:
            imp_probs = np.array(doc['importance_probs'])
            cov_probs = np.array(doc['coverage_probs'])
            imp_labels = np.array(doc['importance_labels'])
            cov_labels = np.array(doc['coverage_labels'])
            surfaced = (imp_probs >= sg_tau) & (cov_probs < sg_gate)
            sg_surfaced += surfaced.sum()
            sg_total_source += len(imp_probs)
            true_omissions = (imp_labels == 1) & (cov_labels == 0)
            n_true = true_omissions.sum()
            sg_true_omissions_total += n_true
            sg_true_omissions_surfaced += (true_omissions & surfaced).sum()
            if n_true == 0:
                sg_losses.append(0.0)
                sg_binary_losses.append(0)
            else:
                n_missed = (true_omissions & ~surfaced).sum()
                sg_losses.append(float(n_missed / n_true))
                sg_binary_losses.append(1 if n_missed > 0 else 0)
        sg_recall = float(sg_true_omissions_surfaced / sg_true_omissions_total) if sg_true_omissions_total > 0 else 1.0
        baseline_results['omission_score_gated'] = {
            'tau': float(sg_tau),
            'coverage_gate': float(sg_gate),
            'n_docs': len(documents),
            'fractional_violation_rate': float(np.mean(sg_losses)),
            'violation_rate': float(np.mean(sg_binary_losses)),
            'surfaced_per_doc': float(sg_surfaced / len(documents)),
            'pct_surfaced': float(100 * sg_surfaced / sg_total_source) if sg_total_source > 0 else 0,
            'true_omissions_total': int(sg_true_omissions_total),
            'true_omissions_surfaced': int(sg_true_omissions_surfaced),
            'surfacing_recall': sg_recall,
            'instance_losses': [float(x) for x in sg_losses],
            'loss_type': 'fractional',
        }
        logger.info(f"  Score-Gated (tau={sg_tau:.3f}, gate={sg_gate}): "
                    f"violation={np.mean(sg_losses):.3f}, "
                    f"workload={sg_surfaced/len(documents):.1f}")

    # Union bound baseline
    if 'omission_union_bound_baseline' in thresholds:
        ub_tau = thresholds['omission_union_bound_baseline']['tau']
        ub_gamma = thresholds['omission_union_bound_baseline']['gamma']
        omit_ub_results = apply_omission_controller(documents, ub_tau, ub_gamma)
        baseline_results['omission_union_bound'] = omit_ub_results
        logger.info(f"  Union Bound (tau={ub_tau:.3f}, gamma={ub_gamma:.3f}): "
                    f"violation={omit_ub_results['violation_rate']:.3f}, "
                    f"workload={omit_ub_results['surfaced_per_doc']:.1f}")

    # Evaluate guarantees
    logger.info("\n" + "="*80)
    logger.info("Evaluating CRC Guarantees")
    logger.info("="*80)
    guarantee_eval = evaluate_guarantees(
        factuality_results,
        omission_results,
        alpha_fact,
        alpha_omit,
    )

    # Save results
    results = {
        'thresholds': {
            'lambda': lambda_threshold,
            'tau': tau_threshold,
            'gamma': gamma_threshold,
            'alpha_fact': alpha_fact,
            'alpha_omit': alpha_omit,
        },
        'factuality': factuality_results,
        'factuality_baselines': factuality_baseline_results,
        'omission': omission_results,
        'omission_baselines': baseline_results,
        'guarantees': guarantee_eval,
        'n_test_docs': len(documents),
    }

    output_file = output_dir / 'test_results.json'
    logger.info(f"\nSaving results to {output_file}...")
    save_json(results, output_file)

    # Generate visualizations via alpha sweep
    # IMPORTANT: Calibrate on calibration_docs, evaluate on test documents
    logger.info("\nRunning alpha sweep for visualizations...")
    logger.info(f"  Calibrating thresholds on {len(calibration_docs)} calibration docs")
    logger.info(f"  Evaluating on {len(documents)} test docs")
    alpha_values = list(np.arange(0.05, 0.55, 0.05))
    sweep_results = run_alpha_sweep_test(calibration_docs, documents, alpha_values)

    # Save sweep results
    sweep_file = output_dir / 'alpha_sweep_results.json'
    save_json(sweep_results, sweep_file)

    # Generate plots
    plot_calibration_curves(sweep_results, documents, output_dir)
    plot_alpha_vs_violation(sweep_results, output_dir)
    plot_precision_recall_workload(sweep_results, output_dir)
    generate_test_summary(factuality_results, omission_results, guarantee_eval, results['thresholds'], output_dir)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Test Evaluation Summary")
    logger.info("="*80)

    logger.info("\nFactuality Controller (Red Flagging):")
    logger.info(f"  Flagged: {factuality_results['flagged']} sentences "
                f"({factuality_results['flagged_per_doc']:.1f} per doc, "
                f"{factuality_results['pct_flagged']:.1f}%)")
    logger.info(f"  Errors total: {factuality_results['errors_total']} "
                f"({factuality_results['errors_per_doc']:.2f} per doc)")
    logger.info(f"  Errors flagged: {factuality_results['errors_flagged']} "
                f"(precision: {100*factuality_results['flagging_precision']:.1f}%, "
                f"recall: {100*factuality_results['flagging_recall']:.1f}%)")
    logger.info(f"  VIOLATION RATE: {factuality_results['violation_rate']:.3f} "
                f"({sum(factuality_results['instance_losses'])}/{len(documents)} docs)")

    logger.info("\nOmission Controller (Purple Surfacing):")
    logger.info(f"  Surfaced: {omission_results['surfaced']} sentences "
                f"({omission_results['surfaced_per_doc']:.1f} per doc, "
                f"{omission_results['pct_surfaced']:.1f}%)")
    logger.info(f"  True omissions: {omission_results['true_omissions_total']} "
                f"({omission_results['true_omissions_per_doc']:.1f} per doc)")
    logger.info(f"  True omissions surfaced: {omission_results['true_omissions_surfaced']} "
                f"(precision: {100*omission_results['surfacing_precision']:.1f}%, "
                f"recall: {100*omission_results['surfacing_recall']:.1f}%)")
    logger.info(f"  FRACTIONAL LOSS: {omission_results['fractional_violation_rate']:.3f} (primary)")
    logger.info(f"  BINARY VIOLATION: {omission_results['violation_rate']:.3f} "
                f"({sum(omission_results['instance_losses'])}/{len(documents)} docs) (secondary)")

    logger.info("\nGuarantee Evaluation:")
    logger.info(f"  Factuality: Violation rate {guarantee_eval['factuality']['violation_rate']:.3f} "
                f"≤ α {guarantee_eval['factuality']['alpha']:.2f}? "
                f"{'✓ YES' if guarantee_eval['factuality']['guarantee_holds'] else '✗ NO'}")
    logger.info(f"  Omission: Fractional loss {guarantee_eval['omission']['violation_rate']:.3f} "
                f"≤ α {guarantee_eval['omission']['alpha']:.2f}? "
                f"{'✓ YES' if guarantee_eval['omission']['guarantee_holds'] else '✗ NO'}")
    logger.info(f"\n  Overall: {'✓ Both guarantees hold' if guarantee_eval['both_hold'] else '✗ At least one guarantee violated'}")

    logger.info("\n" + "="*80)
    logger.info("Phase 4 Complete!")
    logger.info("="*80)
    logger.info(f"Output: {output_dir}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Test-time evaluation (binary instance losses)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Input file with test documents (Phase 2 output)'
    )
    parser.add_argument(
        '--thresholds',
        type=Path,
        default=None,
        help='Conformal thresholds file (Phase 3 output)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--dataset',
        choices=["aci", "meq", "bhc", "cxr", "pubmed", "omop"],
        required=True,
        help='Define the dataset name'
    )
    parser.add_argument(
        '--split',
        choices=['test', 'all'],
        default='test',
        help='Which data split to use (default: test)'
    )

    args = parser.parse_args()
    config.configure_dataset(args.dataset)

    # Set defaults after configure_dataset
    # Default to PHASE2_DIR (combined data) - will filter by split
    input_file = args.input if args.input else config.PHASE2_DIR / 'calibrated_scores.jsonl'
    thresholds_file = args.thresholds if args.thresholds else config.PHASE3_DIR / 'conformal_thresholds.json'
    output_dir = args.output_dir if args.output_dir else config.PHASE4_DIR

    run_phase4(
        input_file=input_file,
        thresholds_file=thresholds_file,
        output_dir=output_dir,
        split=args.split,
    )


if __name__ == '__main__':
    main()
