#!/usr/bin/env python3
"""
Phase 3: Conformal Risk Control with Instance-Level Binary Losses

This script implements Phase 3 of the conformal risk control pipeline (plan.tex):
Two independent CRC controllers using instance-level binary losses.

Controller A (Factuality - Red Flagging):
- Flagged set: F_λ(X) = {v ∈ V(X) : p̂_fact(v) ≤ λ}
- Instance loss: L_fact^inst(λ; X) = 1{∃v: Y_fact(v)=0 AND v ∉ F_λ(X)}
- Loss is monotone non-increasing in λ (larger λ flags more → lower risk)
- Find INFIMUM (smallest) λ* satisfying: (n/(n+1)) * R̂_fact(λ) + 1/(n+1) ≤ α_fact

Controller B (Omission - Purple Surfacing):
- True omissions: O_true(X) = {u : Y_imp(u)=1 AND Y_cov(u)=0}
- Surfaced set: O_{τ,γ}(X) = {u : p̂_imp(u) ≥ τ AND p̂_non-cov(u) ≥ γ}
- Instance loss: L_omit^inst(τ,γ; X) = 1{∃u ∈ O_true: u ∉ O_{τ,γ}(X)}
- 2D joint calibration: Find (τ*, γ*) = lexicographic max over feasible set

Guarantees (plan.tex):
- With probability 1-δ, test-time risk is controlled at level α
- Finite-sample, distribution-free guarantees
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add parent directory to path for imports

from care import config
from care.utils import (
    load_jsonl,
    save_json,
    setup_logging,
)


logger = setup_logging('Phase3', config.LOG_LEVEL)


# ============================================================================
# Controller A: Factuality Filtering (Binary Instance Loss)
# ============================================================================

def compute_factuality_loss_binary(
    factuality_labels: List[int],
    factuality_probs: List[float],
    threshold: float,
) -> int:
    """
    Compute binary instance-level factuality loss (plan.tex Eq. L_fact^inst).

    L_fact^inst(λ; X) = 1{∃v: Y_fact(v)=0 AND p̂_fact(v) > λ}

    Flagged set: F_λ = {v : p̂_fact(v) ≤ λ}
    Loss = 1 iff there exists an unflagged hallucination.

    Args:
        factuality_labels: Y_fact(v) ∈ {0, 1} for each sentence v
        factuality_probs: p̂_fact(v) ∈ [0, 1] for each sentence v
        threshold: λ ∈ [0, 1]

    Returns:
        Binary loss (0 or 1)
    """
    if len(factuality_labels) == 0:
        return 0

    labels = np.array(factuality_labels)
    probs = np.array(factuality_probs)

    # Flagged: p̂_fact(v) ≤ λ (plan.tex uses ≤)
    flagged = probs <= threshold

    # Hallucinations (Y_fact = 0) that are NOT flagged
    hallucinations = (labels == 0)
    unflagged_hallucinations = hallucinations & (~flagged)

    # Binary loss: 1 if ANY unflagged hallucination exists
    return int(unflagged_hallucinations.any())


def compute_factuality_risk_binary(
    documents: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[float, List[int]]:
    """
    Compute empirical binary factuality risk over all calibration documents.

    R̂_fact(λ) = (1/n) Σ_i L_fact^inst(λ; X_i)

    Args:
        documents: List of calibration documents
        threshold: λ ∈ [0, 1]

    Returns:
        (mean_risk, individual_losses)
    """
    losses = []
    for doc in documents:
        loss = compute_factuality_loss_binary(
            doc['factuality_labels'],
            doc['factuality_probs'],
            threshold,
        )
        losses.append(loss)

    mean_risk = np.mean(losses)
    return mean_risk, losses


def select_factuality_threshold(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Select factuality threshold using CRC with binary instance loss.

    Find INFIMUM (smallest) λ* such that:
        (n/(n+1)) * R̂_fact(λ) + 1/(n+1) ≤ α_fact

    Loss is monotone non-increasing in λ (larger λ flags more sentences).
    We select the smallest valid λ to minimize clinician burden.

    Args:
        documents: List of calibration documents
        alpha: Risk level α_fact
        grid_resolution: Grid step size (default: 0.01)

    Returns:
        Dictionary with threshold, risk curve, and statistics
    """
    n = len(documents)
    logger.info(f"Selecting factuality threshold (binary loss) with n={n}, α={alpha}")

    # Define threshold grid
    thresholds = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    # Compute binary risk for each threshold
    risks = []
    all_losses = []

    for threshold in tqdm(thresholds, desc="Factuality grid search"):
        risk, losses = compute_factuality_risk_binary(documents, threshold)
        risks.append(risk)
        all_losses.append(losses)

    risks = np.array(risks)

    # Apply CRC adjustment: (n/(n+1)) * R̂(λ) + 1/(n+1)
    adjusted_risks = (n / (n + 1)) * risks + 1 / (n + 1)

    # Find SMALLEST λ satisfying constraint (infimum)
    # Loss is monotone non-increasing in λ, so valid set is [λ*, 1]
    valid_indices = np.where(adjusted_risks <= alpha)[0]

    if len(valid_indices) == 0:
        logger.warning(f"No threshold satisfies α={alpha}. Using λ=1.0 (all sentences flagged).")
        selected_idx = len(thresholds) - 1
        selected_threshold = 1.0
        is_feasible = False
    else:
        # Select smallest valid λ (infimum) - first index in valid_indices
        selected_idx = valid_indices[0]
        selected_threshold = float(thresholds[selected_idx])
        is_feasible = True

    logger.info(f"Selected λ_fact = {selected_threshold:.3f}")
    logger.info(f"  Empirical risk: {risks[selected_idx]:.4f}")
    logger.info(f"  Adjusted risk: {adjusted_risks[selected_idx]:.4f}")
    logger.info(f"  Constraint: {adjusted_risks[selected_idx]:.4f} ≤ {alpha}")

    return {
        'threshold': selected_threshold,
        'threshold_type': 'factuality',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(risks[selected_idx]),
        'adjusted_risk': float(adjusted_risks[selected_idx]),
        'is_feasible': is_feasible,
        'calibration_curve': {
            'thresholds': thresholds.tolist(),
            'empirical_risks': risks.tolist(),
            'adjusted_risks': adjusted_risks.tolist(),
        },
    }


# ============================================================================
# Controller A Baselines: Uncalibrated + Dev-Set Tuned
# ============================================================================

def select_factuality_threshold_uncalibrated(
    documents: List[Dict[str, Any]],
    alpha: float,
    fixed_lambda: float = 0.5,
) -> Dict[str, Any]:
    """
    Uncalibrated fixed threshold baseline for factuality.

    Uses a fixed lambda (default 0.5, the natural midpoint) without any
    calibration. Reports the resulting violation rate.

    Args:
        documents: Calibration documents (used only for evaluation)
        alpha: Target risk level (for comparison only)
        fixed_lambda: Fixed threshold to use

    Returns:
        Dictionary with threshold and evaluation results
    """
    n = len(documents)

    risk, losses = compute_factuality_risk_binary(documents, fixed_lambda)

    # Compute workload
    n_flagged = 0
    for doc in documents:
        probs = np.array(doc['factuality_probs'])
        n_flagged += (probs <= fixed_lambda).sum()
    avg_workload = n_flagged / n

    return {
        'threshold': fixed_lambda,
        'threshold_type': 'uncalibrated_fixed',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(risk),
        'workload': float(avg_workload),
        'has_formal_guarantee': False,
    }


def select_factuality_threshold_devset(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Dev-set tuned threshold baseline for factuality.

    Picks the smallest lambda such that empirical risk <= alpha on the
    calibration set, WITHOUT the CRC finite-sample correction.

    This tests whether the (n/(n+1)) correction matters.

    Args:
        documents: Calibration documents
        alpha: Target risk level
        grid_resolution: Grid step size

    Returns:
        Dictionary with threshold and evaluation results
    """
    n = len(documents)

    thresholds = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    risks = []
    for threshold in thresholds:
        risk, _ = compute_factuality_risk_binary(documents, threshold)
        risks.append(risk)

    risks = np.array(risks)

    # Find smallest lambda where empirical risk <= alpha (NO CRC adjustment)
    valid_indices = np.where(risks <= alpha)[0]

    if len(valid_indices) == 0:
        selected_idx = len(thresholds) - 1
        is_feasible = False
    else:
        selected_idx = valid_indices[0]
        is_feasible = True

    selected_threshold = float(thresholds[selected_idx])

    # Compute workload
    n_flagged = 0
    for doc in documents:
        probs = np.array(doc['factuality_probs'])
        n_flagged += (probs <= selected_threshold).sum()
    avg_workload = n_flagged / n

    return {
        'threshold': selected_threshold,
        'threshold_type': 'devset_tuned',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(risks[selected_idx]),
        'workload': float(avg_workload),
        'is_feasible': is_feasible,
        'has_formal_guarantee': False,
    }


# ============================================================================
# Controller B: Omission Surfacing (2D Binary Instance Loss)
# ============================================================================

def compute_omission_loss_binary(
    importance_labels: List[int],
    coverage_labels: List[int],
    importance_probs: List[float],
    coverage_probs: List[float],
    tau: float,
    gamma: float,
) -> int:
    """
    Compute binary instance-level omission loss with 2D thresholds (plan.tex Eq. L_omit^inst).

    True omissions: O_true = {u : Y_imp(u)=1 AND Y_cov(u)=0}
    Surfaced: O_{τ,γ} = {u : p̂_imp(u) ≥ τ AND p̂_non-cov(u) ≥ γ}
    Loss = 1 if ANY true omission is NOT surfaced

    Args:
        importance_labels: Y_imp(u) ∈ {0, 1}
        coverage_labels: Y_cov(u) ∈ {0, 1}
        importance_probs: p̂_imp(u) ∈ [0, 1]
        coverage_probs: p̂_cov(u) ∈ [0, 1]
        tau: Importance threshold τ
        gamma: Non-coverage threshold γ

    Returns:
        Binary loss (0 or 1)
    """
    if len(importance_labels) == 0:
        return 0

    imp_labels = np.array(importance_labels)
    cov_labels = np.array(coverage_labels)
    imp_probs = np.array(importance_probs)
    cov_probs = np.array(coverage_probs)

    # True omissions: important AND NOT covered (per Oracle)
    true_omissions = (imp_labels == 1) & (cov_labels == 0)

    if not true_omissions.any():
        return 0  # No true omissions → loss is 0

    # Surfaced: p̂_imp ≥ τ AND p̂_non-cov ≥ γ
    non_cov_probs = 1.0 - cov_probs
    surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)

    # Unsurfaced true omissions
    unsurfaced = true_omissions & (~surfaced)

    # Binary loss: 1 if ANY true omission is unsurfaced
    return int(unsurfaced.any())


def compute_omission_risk_binary(
    documents: List[Dict[str, Any]],
    tau: float,
    gamma: float,
) -> Tuple[float, List[int]]:
    """
    Compute empirical binary omission risk over all calibration documents.

    R̂_omit(τ,γ) = (1/n) Σ_i L_omit^inst(τ,γ; X_i)

    Args:
        documents: List of calibration documents
        tau: Importance threshold τ
        gamma: Non-coverage threshold γ

    Returns:
        (mean_risk, individual_losses)
    """
    losses = []
    for doc in documents:
        loss = compute_omission_loss_binary(
            doc['importance_labels'],
            doc['coverage_labels'],
            doc['importance_probs'],
            doc['coverage_probs'],
            tau,
            gamma,
        )
        losses.append(loss)

    mean_risk = np.mean(losses)
    return mean_risk, losses


def compute_omission_loss_fractional(
    importance_labels: List[int],
    coverage_labels: List[int],
    importance_probs: List[float],
    coverage_probs: List[float],
    tau: float,
    gamma: float,
) -> float:
    """
    Compute fractional instance-level omission loss with 2D thresholds.

    L_omit^frac(τ,γ; X) = |{u ∈ O_true : u ∉ O_{τ,γ}(X)}| / |O_true(X)|

    Returns fraction of true omissions that are NOT surfaced.
    When there are no true omissions, returns 0.

    This loss is bounded in [0, 1] so CRC guarantees hold.
    When |O_true| = 1, this equals the binary loss.

    Args:
        importance_labels: Y_imp(u) ∈ {0, 1}
        coverage_labels: Y_cov(u) ∈ {0, 1}
        importance_probs: p̂_imp(u) ∈ [0, 1]
        coverage_probs: p̂_cov(u) ∈ [0, 1]
        tau: Importance threshold τ
        gamma: Non-coverage threshold γ

    Returns:
        Fractional loss in [0, 1]
    """
    if len(importance_labels) == 0:
        return 0.0

    imp_labels = np.array(importance_labels)
    cov_labels = np.array(coverage_labels)
    imp_probs = np.array(importance_probs)
    cov_probs = np.array(coverage_probs)

    # True omissions: important AND NOT covered (per Oracle)
    true_omissions = (imp_labels == 1) & (cov_labels == 0)

    n_true = true_omissions.sum()
    if n_true == 0:
        return 0.0  # No true omissions → loss is 0

    # Surfaced: p̂_imp ≥ τ AND p̂_non-cov ≥ γ
    non_cov_probs = 1.0 - cov_probs
    surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)

    # Unsurfaced true omissions
    n_missed = (true_omissions & (~surfaced)).sum()

    return float(n_missed / n_true)


def compute_omission_risk_fractional(
    documents: List[Dict[str, Any]],
    tau: float,
    gamma: float,
) -> Tuple[float, List[float]]:
    """
    Compute empirical fractional omission risk over all calibration documents.

    R̂_omit^frac(τ,γ) = (1/n) Σ_i L_omit^frac(τ,γ; X_i)

    Args:
        documents: List of calibration documents
        tau: Importance threshold τ
        gamma: Non-coverage threshold γ

    Returns:
        (mean_risk, individual_losses)
    """
    losses = []
    for doc in documents:
        loss = compute_omission_loss_fractional(
            doc['importance_labels'],
            doc['coverage_labels'],
            doc['importance_probs'],
            doc['coverage_probs'],
            tau,
            gamma,
        )
        losses.append(loss)

    mean_risk = np.mean(losses)
    return mean_risk, losses


def select_omission_threshold_2d_fractional(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.05,
) -> Dict[str, Any]:
    """
    2D joint calibration for omissions using FRACTIONAL loss.

    Same grid search as select_omission_threshold_2d, but uses fractional loss:
    L = (# missed omissions) / (# total omissions per doc)

    This is the default for Controller B. Fractional loss:
    - Equals binary loss when k=1 (single omission per doc)
    - Degrades gracefully for high-k docs (many omissions)
    - Is bounded in [0,1] so CRC guarantees hold

    Guarantee: E[fraction of missed omissions per doc] ≤ α

    Args:
        documents: List of calibration documents
        alpha: Risk level α_omit
        grid_resolution: Grid step size (default: 0.05 for 2D)

    Returns:
        Dictionary with thresholds, risk surface, and statistics
    """
    n = len(documents)
    logger.info(f"Selecting omission thresholds (2D fractional) with n={n}, α={alpha}")
    logger.info(f"Grid resolution: {grid_resolution}")

    # Define threshold grids
    tau_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)
    gamma_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    logger.info(f"Grid size: {len(tau_vals)} × {len(gamma_vals)} = {len(tau_vals) * len(gamma_vals)} pairs")

    # Compute risk surface AND workload for each (τ, γ)
    risk_surface = {}
    for tau in tqdm(tau_vals, desc="Omission 2D fractional grid search"):
        for gamma in gamma_vals:
            risk, losses = compute_omission_risk_fractional(documents, tau, gamma)

            # Also compute binary risk for secondary reporting
            binary_risk, binary_losses = compute_omission_risk_binary(documents, tau, gamma)

            # Compute workload (sentences surfaced)
            n_surfaced = 0
            for doc in documents:
                imp_probs = np.array(doc['importance_probs'])
                cov_probs = np.array(doc['coverage_probs'])
                non_cov_probs = 1.0 - cov_probs
                surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)
                n_surfaced += surfaced.sum()
            avg_surfaced = n_surfaced / n

            risk_surface[(tau, gamma)] = {
                'fractional_risk': risk,
                'binary_risk': binary_risk,
                'losses': losses,
                'binary_losses': binary_losses,
                'avg_surfaced': avg_surfaced,
            }

    # Find feasible set using FRACTIONAL loss
    feasible = []
    for (tau, gamma), data in risk_surface.items():
        adj_risk = (n / (n + 1)) * data['fractional_risk'] + 1 / (n + 1)
        if adj_risk <= alpha:
            feasible.append((tau, gamma, data['fractional_risk'], adj_risk,
                           data['avg_surfaced'], data['binary_risk']))

    logger.info(f"Feasible set size: {len(feasible)} / {len(risk_surface)}")

    if not feasible:
        logger.warning(f"No threshold pair satisfies α={alpha}. Using (0, 0) (surface all).")
        selected_tau, selected_gamma = 0.0, 0.0
        frac_risk = risk_surface[(0.0, 0.0)]['fractional_risk']
        bin_risk = risk_surface[(0.0, 0.0)]['binary_risk']
        adj_risk = (n / (n + 1)) * frac_risk + 1 / (n + 1)
        workload = risk_surface[(0.0, 0.0)]['avg_surfaced']
        is_feasible = False
    else:
        # Select minimum workload from feasible set
        # Tie-break: favor larger thresholds (max τ, then max γ)
        feasible.sort(key=lambda x: (x[4], -x[0], -x[1]))  # min workload, max tau, max gamma
        selected_tau, selected_gamma, frac_risk, adj_risk, workload, bin_risk = feasible[0]
        is_feasible = True

    # Compute binary adjusted risk at selected thresholds (secondary metric)
    binary_adj_risk = (n / (n + 1)) * bin_risk + 1 / (n + 1)

    logger.info(f"Selected (τ*, γ*) = ({selected_tau:.3f}, {selected_gamma:.3f})")
    logger.info(f"  Fractional risk: {frac_risk:.4f} (adjusted: {adj_risk:.4f})")
    logger.info(f"  Binary risk: {bin_risk:.4f} (adjusted: {binary_adj_risk:.4f})")
    logger.info(f"  Workload: {workload:.1f} sentences/doc")
    logger.info(f"  Constraint: {adj_risk:.4f} ≤ {alpha}")

    return {
        'tau': float(selected_tau),
        'gamma': float(selected_gamma),
        'threshold_type': 'omission_2d_fractional',
        'loss_type': 'fractional',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(frac_risk),
        'adjusted_risk': float(adj_risk),
        'binary_empirical_risk': float(bin_risk),
        'binary_adjusted_risk': float(binary_adj_risk),
        'workload': float(workload),
        'is_feasible': is_feasible,
        'feasible_set_size': len(feasible),
        'grid_size': len(tau_vals) * len(gamma_vals),
    }


def select_omission_threshold_fst_fractional(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.05,
    ordering: str = "tau_plus_gamma",
) -> Dict[str, Any]:
    """
    LTT-FST omission calibration — the PRIMARY omission controller.

    Rigorous variant of the 2D-Joint heuristic (select_omission_threshold_2d_
    fractional, kept as a comparator). Walks a pre-specified, data-independent
    ordering of grid cells and selects the first cell whose finite-sample CRC
    bound holds at level alpha.

    Theory: by Learn-Then-Test (Angelopoulos et al., 2021, Thm 1) with the
    fixed-sequence multiple-testing procedure over a data-independent ordering,
    the output (tau*, gamma*) satisfies E[L(tau*,gamma*; X_{n+1})] <= alpha with
    the same finite-sample (n+1) correction as scalar CRC. No Bonferroni
    correction is required because the testing order is data-independent. The
    omission loss is monotone non-increasing along the (tau+gamma)-descending
    sequence, so early-stop at the first feasible cell is well-defined.

    Args:
        documents: calibration documents (same format as the 2D-Joint variant).
        alpha: omission risk budget alpha_omit in (0, 1].
        grid_resolution: grid step (default 0.05 -> 21x21 = 441 cells).
        ordering: deterministic ordering to walk:
            - "tau_plus_gamma": (tau + gamma) descending, ties -> tau desc,
              gamma desc. Walks from the most-conservative corner (1,1) outward.
            - "lex_tau": (tau desc, gamma desc). Rows then columns.
            - "diagonal": tau = gamma chain only (1D restriction).

    Returns:
        Same dict shape as select_omission_threshold_2d_fractional, plus
        'fst_steps_walked' and 'fst_ordering' for diagnostics.
    """
    n = len(documents)

    tau_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)
    gamma_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    # Build the data-independent ordering.
    cells = [(float(t), float(g)) for t in tau_vals for g in gamma_vals]
    if ordering == "tau_plus_gamma":
        # Descending tau+gamma; ties broken by tau desc then gamma desc.
        cells.sort(key=lambda c: (-(c[0] + c[1]), -c[0], -c[1]))
    elif ordering == "lex_tau":
        cells.sort(key=lambda c: (-c[0], -c[1]))
    elif ordering == "diagonal":
        cells = [(float(v), float(v)) for v in tau_vals[::-1]]
    else:
        raise ValueError(f"unknown ordering: {ordering}")

    # Walk the sequence; stop at first feasible cell.
    selected = None
    steps_walked = 0
    fractional_losses = None
    binary_losses = None
    workload_at_selected = None
    frac_risk = None
    bin_risk = None

    for (tau, gamma) in tqdm(cells, desc=f"FST scan ({ordering})"):
        steps_walked += 1
        risk, losses = compute_omission_risk_fractional(documents, tau, gamma)
        adj_risk = (n / (n + 1)) * risk + 1.0 / (n + 1)
        if adj_risk <= alpha:
            selected = (tau, gamma)
            frac_risk = risk
            fractional_losses = losses
            # Compute workload and binary risk at the selected cell only.
            n_surfaced = 0
            for doc in documents:
                imp_probs = np.array(doc['importance_probs'])
                cov_probs = np.array(doc['coverage_probs'])
                non_cov = 1.0 - cov_probs
                surfaced = (imp_probs >= tau) & (non_cov >= gamma)
                n_surfaced += surfaced.sum()
            workload_at_selected = n_surfaced / n
            bin_risk, binary_losses = compute_omission_risk_binary(documents, tau, gamma)
            break

    if selected is None:
        # No cell along the sequence is feasible. Fall back to (0, 0)
        # (surface everything) — guaranteed to give zero loss in expectation.
        logger.warning(
            f"FST ordering '{ordering}' found no feasible cell at alpha={alpha}. "
            f"Falling back to (0.0, 0.0)."
        )
        selected = (0.0, 0.0)
        frac_risk, fractional_losses = compute_omission_risk_fractional(documents, 0.0, 0.0)
        bin_risk, binary_losses = compute_omission_risk_binary(documents, 0.0, 0.0)
        adj_risk = (n / (n + 1)) * frac_risk + 1.0 / (n + 1)
        n_surfaced = 0
        for doc in documents:
            n_surfaced += len(doc['importance_probs'])
        workload_at_selected = n_surfaced / n
        is_feasible = False
    else:
        is_feasible = True

    selected_tau, selected_gamma = selected
    binary_adj_risk = (n / (n + 1)) * bin_risk + 1.0 / (n + 1)

    logger.info(f"FST selected (tau*, gamma*) = ({selected_tau:.3f}, {selected_gamma:.3f})")
    logger.info(f"  Walked {steps_walked} cells of {len(cells)} before stopping")
    logger.info(f"  Fractional risk: {frac_risk:.4f} (adjusted: {adj_risk:.4f})")
    logger.info(f"  Workload: {workload_at_selected:.1f} sentences/doc")

    return {
        'tau': float(selected_tau),
        'gamma': float(selected_gamma),
        'threshold_type': f'omission_fst_{ordering}',
        'loss_type': 'fractional',
        'alpha': float(alpha),
        'n_docs': int(n),
        'empirical_risk': float(frac_risk),
        'adjusted_risk': float(adj_risk),
        'binary_empirical_risk': float(bin_risk),
        'binary_adjusted_risk': float(binary_adj_risk),
        'workload': float(workload_at_selected),
        'is_feasible': bool(is_feasible),
        'fst_steps_walked': int(steps_walked),
        'fst_ordering': ordering,
        'grid_size': int(len(cells)),
    }


def select_omission_threshold_2d(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.05,
) -> Dict[str, Any]:
    """
    2D joint calibration for omissions (plan.tex proposed method).

    1. Compute risk for each (τ, γ) pair on 2D grid
    2. Find feasible set F(α) = {(τ,γ): adjusted_risk ≤ α}
    3. Select (τ*, γ*) that MINIMIZES workload (sentences surfaced)

    Args:
        documents: List of calibration documents
        alpha: Risk level α_omit
        grid_resolution: Grid step size (default: 0.05 for 2D)

    Returns:
        Dictionary with thresholds, risk surface, and statistics
    """
    n = len(documents)
    logger.info(f"Selecting omission thresholds (2D joint) with n={n}, α={alpha}")
    logger.info(f"Grid resolution: {grid_resolution}")

    # Define threshold grids
    tau_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)
    gamma_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    logger.info(f"Grid size: {len(tau_vals)} × {len(gamma_vals)} = {len(tau_vals) * len(gamma_vals)} pairs")

    # Compute risk surface AND workload for each (τ, γ)
    risk_surface = {}
    for tau in tqdm(tau_vals, desc="Omission 2D grid search"):
        for gamma in gamma_vals:
            risk, losses = compute_omission_risk_binary(documents, tau, gamma)

            # Compute workload (sentences surfaced)
            n_surfaced = 0
            for doc in documents:
                imp_probs = np.array(doc['importance_probs'])
                cov_probs = np.array(doc['coverage_probs'])
                non_cov_probs = 1.0 - cov_probs
                surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)
                n_surfaced += surfaced.sum()
            avg_surfaced = n_surfaced / n

            risk_surface[(tau, gamma)] = {
                'empirical_risk': risk,
                'losses': losses,
                'avg_surfaced': avg_surfaced,
            }

    # Find feasible set with workload
    feasible = []
    for (tau, gamma), data in risk_surface.items():
        adj_risk = (n / (n + 1)) * data['empirical_risk'] + 1 / (n + 1)
        if adj_risk <= alpha:
            feasible.append((tau, gamma, data['empirical_risk'], adj_risk, data['avg_surfaced']))

    logger.info(f"Feasible set size: {len(feasible)} / {len(risk_surface)}")

    if not feasible:
        logger.warning(f"No threshold pair satisfies α={alpha}. Using (0, 0) (surface all).")
        # Use tau=0, gamma=0 which surfaces all sentences
        selected_tau, selected_gamma = 0.0, 0.0
        emp_risk = risk_surface[(0.0, 0.0)]['empirical_risk']
        adj_risk = (n / (n + 1)) * emp_risk + 1 / (n + 1)
        workload = risk_surface[(0.0, 0.0)]['avg_surfaced']
        is_feasible = False
    else:
        # Select minimum workload from feasible set
        # Tie-break: favor larger thresholds (max τ, then max γ) per plan.tex
        feasible.sort(key=lambda x: (x[4], -x[0], -x[1]))  # min workload, max tau, max gamma
        selected_tau, selected_gamma, emp_risk, adj_risk, workload = feasible[0]
        is_feasible = True

    logger.info(f"Selected (τ*, γ*) = ({selected_tau:.3f}, {selected_gamma:.3f})")
    logger.info(f"  Empirical risk: {emp_risk:.4f}")
    logger.info(f"  Adjusted risk: {adj_risk:.4f}")
    logger.info(f"  Workload: {workload:.1f} sentences/doc")
    logger.info(f"  Constraint: {adj_risk:.4f} ≤ {alpha}")

    # Build calibration curve for visualization (marginal over gamma at selected tau)
    tau_marginal_risks = []
    for tau in tau_vals:
        # At each tau, use gamma=0 for comparison with 1D
        risk = risk_surface[(tau, 0.0)]['empirical_risk']
        tau_marginal_risks.append(risk)

    return {
        'tau': float(selected_tau),
        'gamma': float(selected_gamma),
        'threshold_type': 'omission_2d',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(emp_risk),
        'adjusted_risk': float(adj_risk),
        'workload': float(workload),
        'is_feasible': is_feasible,
        'feasible_set_size': len(feasible),
        'grid_size': len(tau_vals) * len(gamma_vals),
        'calibration_curve': {
            'tau_values': tau_vals.tolist(),
            'gamma_values': gamma_vals.tolist(),
            'tau_marginal_risks': tau_marginal_risks,
        },
        'risk_surface': {str(k): v for k, v in risk_surface.items()},  # For plotting
    }


# ============================================================================
# Backward-compatible 1D Omission (importance-only)
# ============================================================================

def compute_importance_loss_fractional(
    importance_labels: List[int],
    coverage_labels: List[int],
    importance_probs: List[float],
    threshold: float,
) -> float:
    """
    Compute fractional instance-level importance loss (1D, importance-only).

    For comparison with 2D method. Uses same omission target but only
    thresholds on importance.

    Surfaced: P_τ = {u : p̂_imp(u) ≥ τ}
    Loss = fraction of true omissions NOT surfaced.
    Returns 0 when there are no true omissions.
    """
    if len(importance_labels) == 0:
        return 0.0

    imp_labels = np.array(importance_labels)
    cov_labels = np.array(coverage_labels)
    imp_probs = np.array(importance_probs)

    # True omissions: important AND NOT covered
    true_omissions = (imp_labels == 1) & (cov_labels == 0)

    n_true = true_omissions.sum()
    if n_true == 0:
        return 0.0

    # Surfaced by importance only
    surfaced = imp_probs >= threshold
    n_missed = (true_omissions & (~surfaced)).sum()

    return float(n_missed / n_true)


def select_omission_threshold_1d(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    1D importance-only calibration for comparison.

    Find LARGEST τ* satisfying CRC bound.
    """
    n = len(documents)
    logger.info(f"Selecting importance threshold (1D) with n={n}, α={alpha}")

    thresholds = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    risks = []
    for threshold in tqdm(thresholds, desc="Importance 1D grid search"):
        losses = []
        for doc in documents:
            loss = compute_importance_loss_fractional(
                doc['importance_labels'],
                doc['coverage_labels'],
                doc['importance_probs'],
                threshold,
            )
            losses.append(loss)
        risks.append(np.mean(losses))

    risks = np.array(risks)
    adjusted_risks = (n / (n + 1)) * risks + 1 / (n + 1)

    valid_indices = np.where(adjusted_risks <= alpha)[0]

    if len(valid_indices) == 0:
        logger.warning(f"No threshold satisfies α={alpha}. Using τ=0.")
        selected_idx = 0
        is_feasible = False
    else:
        # Largest valid τ
        selected_idx = valid_indices[-1]
        is_feasible = True

    selected_threshold = float(thresholds[selected_idx])

    logger.info(f"Selected τ_1D = {selected_threshold:.3f}")
    logger.info(f"  Empirical risk: {risks[selected_idx]:.4f}")
    logger.info(f"  Adjusted risk: {adjusted_risks[selected_idx]:.4f}")

    # Compute workload at selected threshold
    n_surfaced = 0
    for doc in documents:
        imp_probs = np.array(doc['importance_probs'])
        surfaced = imp_probs >= selected_threshold
        n_surfaced += surfaced.sum()
    avg_workload = n_surfaced / n

    return {
        'threshold': selected_threshold,
        'threshold_type': 'importance_1d',
        'loss_type': 'fractional',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(risks[selected_idx]),
        'adjusted_risk': float(adjusted_risks[selected_idx]),
        'is_feasible': is_feasible,
        'workload': float(avg_workload),
        'calibration_curve': {
            'thresholds': thresholds.tolist(),
            'empirical_risks': risks.tolist(),
            'adjusted_risks': adjusted_risks.tolist(),
        },
    }


# ============================================================================
# Controller B Baselines: Product Composite
# ============================================================================

def compute_product_loss_fractional(
    importance_labels: List[int],
    coverage_labels: List[int],
    importance_probs: List[float],
    coverage_probs: List[float],
    threshold: float,
) -> float:
    """
    Compute fractional instance-level omission loss using product composite score.

    s_prod(u) = p_imp(u) * (1 - p_cov(u))
    Surfaced: {u : s_prod(u) >= beta}
    Loss = fraction of true omissions NOT surfaced.
    Returns 0 when there are no true omissions.

    Args:
        importance_labels: Y_imp(u) in {0, 1}
        coverage_labels: Y_cov(u) in {0, 1}
        importance_probs: p_imp(u) in [0, 1]
        coverage_probs: p_cov(u) in [0, 1]
        threshold: beta in [0, 1]

    Returns:
        Fractional loss in [0, 1]
    """
    if len(importance_labels) == 0:
        return 0.0

    imp_labels = np.array(importance_labels)
    cov_labels = np.array(coverage_labels)
    imp_probs = np.array(importance_probs)
    cov_probs = np.array(coverage_probs)

    # True omissions: important AND NOT covered
    true_omissions = (imp_labels == 1) & (cov_labels == 0)

    n_true = true_omissions.sum()
    if n_true == 0:
        return 0.0

    # Product composite score
    s_prod = imp_probs * (1.0 - cov_probs)
    surfaced = s_prod >= threshold

    n_missed = (true_omissions & (~surfaced)).sum()
    return float(n_missed / n_true)


def select_omission_threshold_product(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Product composite baseline for omission detection.

    Single 1D threshold on s_prod(u) = p_imp(u) * (1 - p_cov(u)).
    Find LARGEST beta satisfying CRC bound.

    Args:
        documents: List of calibration documents
        alpha: Risk level
        grid_resolution: Grid step size

    Returns:
        Dictionary with threshold and statistics
    """
    n = len(documents)
    logger.info(f"Selecting product composite threshold with n={n}, alpha={alpha}")

    thresholds = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    risks = []
    for threshold in tqdm(thresholds, desc="Product composite grid search"):
        losses = []
        for doc in documents:
            loss = compute_product_loss_fractional(
                doc['importance_labels'],
                doc['coverage_labels'],
                doc['importance_probs'],
                doc['coverage_probs'],
                threshold,
            )
            losses.append(loss)
        risks.append(np.mean(losses))

    risks = np.array(risks)
    adjusted_risks = (n / (n + 1)) * risks + 1 / (n + 1)

    valid_indices = np.where(adjusted_risks <= alpha)[0]

    if len(valid_indices) == 0:
        logger.warning(f"No threshold satisfies alpha={alpha}. Using beta=0.")
        selected_idx = 0
        is_feasible = False
    else:
        # Largest valid beta (maximize threshold = minimize workload)
        selected_idx = valid_indices[-1]
        is_feasible = True

    selected_threshold = float(thresholds[selected_idx])

    # Compute workload
    n_surfaced = 0
    for doc in documents:
        imp_probs = np.array(doc['importance_probs'])
        cov_probs = np.array(doc['coverage_probs'])
        s_prod = imp_probs * (1.0 - cov_probs)
        n_surfaced += (s_prod >= selected_threshold).sum()
    avg_workload = n_surfaced / n

    logger.info(f"Selected beta_product = {selected_threshold:.3f}")
    logger.info(f"  Empirical risk: {risks[selected_idx]:.4f}")
    logger.info(f"  Adjusted risk: {adjusted_risks[selected_idx]:.4f}")
    logger.info(f"  Workload: {avg_workload:.1f} sentences/doc")

    return {
        'threshold': selected_threshold,
        'threshold_type': 'product_composite',
        'loss_type': 'fractional',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(risks[selected_idx]),
        'adjusted_risk': float(adjusted_risks[selected_idx]),
        'is_feasible': is_feasible,
        'workload': float(avg_workload),
        'calibration_curve': {
            'thresholds': thresholds.tolist(),
            'empirical_risks': risks.tolist(),
            'adjusted_risks': adjusted_risks.tolist(),
        },
    }


# ============================================================================
# Controller B Baselines: Score-Gated (1D-Imp + fixed coverage gate)
# ============================================================================

def select_omission_threshold_score_gated(
    documents: List[Dict[str, Any]],
    alpha: float,
    coverage_gate: float = 0.5,
    grid_resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Score-gated baseline: CRC-calibrated tau + fixed coverage gate.

    1. Calibrate tau via 1D importance CRC
    2. Apply fixed coverage gate: surface = {u : p_imp >= tau AND p_cov < coverage_gate}
    3. Compute empirical loss at this operating point (no CRC guarantee on coverage)

    Args:
        documents: List of calibration documents
        alpha: Risk level (used for 1D importance calibration)
        coverage_gate: Fixed coverage threshold (default: 0.5)
        grid_resolution: Grid step size

    Returns:
        Dictionary with threshold and statistics
    """
    n = len(documents)
    logger.info(f"Selecting score-gated threshold with n={n}, alpha={alpha}, gate={coverage_gate}")

    # Step 1: Get calibrated tau from 1D importance
    imp_1d = select_omission_threshold_1d(documents, alpha, grid_resolution)
    tau = imp_1d['threshold']

    # Step 2: Compute fractional loss at (tau, coverage_gate) operating point
    losses = []
    n_surfaced = 0
    for doc in documents:
        imp_labels = np.array(doc['importance_labels'])
        cov_labels = np.array(doc['coverage_labels'])
        imp_probs = np.array(doc['importance_probs'])
        cov_probs = np.array(doc['coverage_probs'])

        # True omissions
        true_omissions = (imp_labels == 1) & (cov_labels == 0)

        # Surfaced: p_imp >= tau AND p_cov < gate
        surfaced = (imp_probs >= tau) & (cov_probs < coverage_gate)
        n_surfaced += surfaced.sum()

        n_true = true_omissions.sum()
        if n_true == 0:
            losses.append(0.0)
        else:
            n_missed = (true_omissions & (~surfaced)).sum()
            losses.append(float(n_missed / n_true))

    empirical_risk = float(np.mean(losses))
    avg_workload = n_surfaced / n

    logger.info(f"Score-gated: tau={tau:.3f}, gate={coverage_gate}")
    logger.info(f"  Empirical risk (fractional): {empirical_risk:.4f}")
    logger.info(f"  Workload: {avg_workload:.1f} sentences/doc")
    logger.info(f"  NOTE: No formal CRC guarantee on coverage gate")

    return {
        'threshold': tau,
        'coverage_gate': coverage_gate,
        'threshold_type': 'score_gated',
        'loss_type': 'fractional',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': empirical_risk,
        'is_feasible': True,  # tau is always feasible from 1D
        'workload': float(avg_workload),
        'has_formal_guarantee': False,
    }


# ============================================================================
# Controller B Baselines: Union Bound (independent tau, gamma)
# ============================================================================

def compute_noncoverage_loss_fractional(
    importance_labels: List[int],
    coverage_labels: List[int],
    coverage_probs: List[float],
    threshold: float,
) -> float:
    """
    Compute fractional instance-level non-coverage loss.

    Surfaced by coverage gate: {u : (1 - p_cov(u)) >= gamma}
    Loss = fraction of true omissions NOT surfaced by coverage gate.
    Returns 0 when there are no true omissions.

    This is the coverage-dimension analog of compute_importance_loss_fractional().

    Args:
        importance_labels: Y_imp(u) in {0, 1}
        coverage_labels: Y_cov(u) in {0, 1}
        coverage_probs: p_cov(u) in [0, 1]
        threshold: gamma in [0, 1]

    Returns:
        Fractional loss in [0, 1]
    """
    if len(importance_labels) == 0:
        return 0.0

    imp_labels = np.array(importance_labels)
    cov_labels = np.array(coverage_labels)
    cov_probs = np.array(coverage_probs)

    # True omissions: important AND NOT covered
    true_omissions = (imp_labels == 1) & (cov_labels == 0)

    n_true = true_omissions.sum()
    if n_true == 0:
        return 0.0

    # Surfaced by coverage gate
    non_cov_probs = 1.0 - cov_probs
    surfaced = non_cov_probs >= threshold

    n_missed = (true_omissions & (~surfaced)).sum()
    return float(n_missed / n_true)


def select_omission_threshold_union_bound(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Union bound baseline: independent CRC calibration on tau and gamma.

    Split budget: alpha_imp = alpha/2, alpha_cov = alpha/2.
    Calibrate tau and gamma independently, surface = intersection.

    More conservative than 2D joint (splits alpha budget).

    Args:
        documents: List of calibration documents
        alpha: Risk level
        grid_resolution: Grid step size

    Returns:
        Dictionary with thresholds and statistics
    """
    n = len(documents)
    alpha_imp = alpha / 2.0
    alpha_cov = alpha / 2.0
    logger.info(f"Selecting union bound thresholds with n={n}, alpha={alpha}")
    logger.info(f"  Split budget: alpha_imp={alpha_imp:.3f}, alpha_cov={alpha_cov:.3f}")

    # Step 1: Calibrate tau using 1D importance CRC at alpha/2
    imp_result = select_omission_threshold_1d(documents, alpha_imp, grid_resolution)
    tau = imp_result['threshold']

    # Step 2: Calibrate gamma independently using non-coverage CRC at alpha/2
    thresholds = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    risks = []
    for threshold in tqdm(thresholds, desc="Non-coverage grid search"):
        losses = []
        for doc in documents:
            loss = compute_noncoverage_loss_fractional(
                doc['importance_labels'],
                doc['coverage_labels'],
                doc['coverage_probs'],
                threshold,
            )
            losses.append(loss)
        risks.append(np.mean(losses))

    risks = np.array(risks)
    adjusted_risks = (n / (n + 1)) * risks + 1 / (n + 1)

    valid_indices = np.where(adjusted_risks <= alpha_cov)[0]

    if len(valid_indices) == 0:
        logger.warning(f"No gamma satisfies alpha_cov={alpha_cov}. Using gamma=0.")
        gamma_idx = 0
        gamma_feasible = False
    else:
        gamma_idx = valid_indices[-1]
        gamma_feasible = True

    gamma = float(thresholds[gamma_idx])

    # Step 3: Compute full omission loss at (tau, gamma) intersection
    full_losses = []
    n_surfaced = 0
    for doc in documents:
        imp_probs = np.array(doc['importance_probs'])
        cov_probs = np.array(doc['coverage_probs'])
        non_cov_probs = 1.0 - cov_probs

        # Intersection: p_imp >= tau AND (1 - p_cov) >= gamma
        surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)
        n_surfaced += surfaced.sum()

        # Full omission loss (fractional)
        loss = compute_omission_loss_fractional(
            doc['importance_labels'],
            doc['coverage_labels'],
            doc['importance_probs'],
            doc['coverage_probs'],
            tau,
            gamma,
        )
        full_losses.append(loss)

    empirical_risk = float(np.mean(full_losses))
    adjusted_risk = (n / (n + 1)) * empirical_risk + 1 / (n + 1)
    avg_workload = n_surfaced / n

    is_feasible = imp_result['is_feasible'] and gamma_feasible

    logger.info(f"Union bound: tau={tau:.3f}, gamma={gamma:.3f}")
    logger.info(f"  Importance risk (fractional): {imp_result['adjusted_risk']:.4f} <= {alpha_imp}")
    logger.info(f"  Non-coverage risk (fractional): {adjusted_risks[gamma_idx]:.4f} <= {alpha_cov}")
    logger.info(f"  Full omission risk (fractional): {adjusted_risk:.4f}")
    logger.info(f"  Workload: {avg_workload:.1f} sentences/doc")

    return {
        'tau': tau,
        'gamma': gamma,
        'threshold_type': 'union_bound',
        'loss_type': 'fractional',
        'alpha': alpha,
        'alpha_imp': alpha_imp,
        'alpha_cov': alpha_cov,
        'n_docs': n,
        'empirical_risk': empirical_risk,
        'adjusted_risk': adjusted_risk,
        'is_feasible': is_feasible,
        'workload': float(avg_workload),
        'has_formal_guarantee': True,
        'imp_adjusted_risk': float(imp_result['adjusted_risk']),
        'cov_adjusted_risk': float(adjusted_risks[gamma_idx]),
    }


# ============================================================================
# Controller B Baseline: Devset-Tuned 2D (no CRC correction)
# ============================================================================

def select_omission_threshold_2d_devset(
    documents: List[Dict[str, Any]],
    alpha: float,
    grid_resolution: float = 0.05,
) -> Dict[str, Any]:
    """
    Devset-tuned 2D omission threshold — NO CRC correction.

    Same grid search as select_omission_threshold_2d_fractional, but picks
    (τ, γ) where the empirical fractional risk ≤ α directly, WITHOUT the
    finite-sample CRC adjustment ((n/(n+1)) * R̂ + 1/(n+1)).

    This is the non-conformal baseline that answers: "Does the CRC correction
    matter, or can you just pick thresholds on a held-out validation set?"

    Args:
        documents: List of calibration documents
        alpha: Risk level α_omit
        grid_resolution: Grid step size (default: 0.05 for 2D)

    Returns:
        Dictionary with thresholds, risk, and statistics
    """
    n = len(documents)
    logger.info(f"Selecting devset-tuned omission thresholds (2D, NO CRC) with n={n}, α={alpha}")

    tau_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)
    gamma_vals = np.arange(0.0, 1.0 + grid_resolution, grid_resolution)

    # Compute risk surface
    risk_surface = {}
    for tau in tau_vals:
        for gamma in gamma_vals:
            risk, losses = compute_omission_risk_fractional(documents, tau, gamma)

            n_surfaced = 0
            for doc in documents:
                imp_probs = np.array(doc['importance_probs'])
                cov_probs = np.array(doc['coverage_probs'])
                non_cov_probs = 1.0 - cov_probs
                surfaced = (imp_probs >= tau) & (non_cov_probs >= gamma)
                n_surfaced += surfaced.sum()
            avg_surfaced = n_surfaced / n

            risk_surface[(tau, gamma)] = {
                'fractional_risk': risk,
                'avg_surfaced': avg_surfaced,
            }

    # Find feasible set: empirical risk ≤ α (NO CRC adjustment)
    feasible = []
    for (tau, gamma), data in risk_surface.items():
        if data['fractional_risk'] <= alpha:
            feasible.append((tau, gamma, data['fractional_risk'], data['avg_surfaced']))

    logger.info(f"Devset feasible set size: {len(feasible)} / {len(risk_surface)}")

    if not feasible:
        logger.warning(f"No threshold pair satisfies α={alpha} (devset). Using (0, 0).")
        selected_tau, selected_gamma = 0.0, 0.0
        frac_risk = risk_surface[(0.0, 0.0)]['fractional_risk']
        workload = risk_surface[(0.0, 0.0)]['avg_surfaced']
        is_feasible = False
    else:
        # Select minimum workload from feasible set (same tie-breaking as CRC)
        feasible.sort(key=lambda x: (x[3], -x[0], -x[1]))
        selected_tau, selected_gamma, frac_risk, workload = feasible[0]
        is_feasible = True

    logger.info(f"Devset-tuned (τ*, γ*) = ({selected_tau:.3f}, {selected_gamma:.3f})")
    logger.info(f"  Empirical risk (fractional): {frac_risk:.4f}")
    logger.info(f"  Workload: {workload:.1f} sentences/doc")

    return {
        'tau': float(selected_tau),
        'gamma': float(selected_gamma),
        'threshold_type': 'omission_2d_devset',
        'loss_type': 'fractional',
        'alpha': alpha,
        'n_docs': n,
        'empirical_risk': float(frac_risk),
        'is_feasible': is_feasible,
        'workload': float(workload),
        'has_formal_guarantee': False,
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


def run_alpha_sweep(
    documents: List[Dict[str, Any]],
    alpha_values: List[float],
    grid_resolution: float = 0.05,
    compute_ci: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run calibration for multiple alpha values to generate calibration curves.

    Uses Wilson score CI for the empirical risk (binomial proportion).

    Returns list of results for each alpha value.
    """
    n = len(documents)
    results = []

    for alpha in tqdm(alpha_values, desc="Alpha sweep"):
        # Factuality threshold selection
        fact_result = select_factuality_threshold(documents, alpha, grid_resolution=0.01)

        # Omission threshold selection (2D Fractional - default)
        omit_result = select_omission_threshold_2d_fractional(documents, alpha, grid_resolution=grid_resolution)

        # Omission (2D Binary - secondary)
        omit_binary = select_omission_threshold_2d(documents, alpha, grid_resolution=grid_resolution)

        # Omission baselines
        omit_1d = select_omission_threshold_1d(documents, alpha, grid_resolution=0.01)
        omit_product = select_omission_threshold_product(documents, alpha, grid_resolution=0.01)
        omit_score_gated = select_omission_threshold_score_gated(documents, alpha, grid_resolution=0.01)
        omit_union_bound = select_omission_threshold_union_bound(documents, alpha, grid_resolution=0.01)

        # Compute workloads and per-document losses at selected thresholds
        lambda_thresh = fact_result['threshold']
        tau_thresh = omit_result['tau']
        gamma_thresh = omit_result['gamma']

        # Binary 2D thresholds (secondary)
        tau_bin = omit_binary['tau']
        gamma_bin = omit_binary['gamma']

        fact_flagged = 0
        omit_surfaced = 0
        omit_bin_surfaced = 0
        fact_losses = []
        omit_frac_losses = []
        omit_bin_losses = []

        for doc in documents:
            # Factuality
            probs = np.array(doc['factuality_probs'])
            fact_flagged += (probs <= lambda_thresh).sum()
            fact_loss = compute_factuality_loss_binary(
                doc['factuality_labels'], doc['factuality_probs'], lambda_thresh
            )
            fact_losses.append(fact_loss)

            # Omission (fractional - default)
            imp_probs = np.array(doc['importance_probs'])
            cov_probs = np.array(doc['coverage_probs'])
            non_cov = 1.0 - cov_probs
            omit_surfaced += ((imp_probs >= tau_thresh) & (non_cov >= gamma_thresh)).sum()
            frac_loss = compute_omission_loss_fractional(
                doc['importance_labels'], doc['coverage_labels'],
                doc['importance_probs'], doc['coverage_probs'],
                tau_thresh, gamma_thresh
            )
            omit_frac_losses.append(frac_loss)

            # Omission (binary - secondary)
            omit_bin_surfaced += ((imp_probs >= tau_bin) & (non_cov >= gamma_bin)).sum()
            bin_loss = compute_omission_loss_binary(
                doc['importance_labels'], doc['coverage_labels'],
                doc['importance_probs'], doc['coverage_probs'],
                tau_bin, gamma_bin
            )
            omit_bin_losses.append(bin_loss)

        fact_k = sum(fact_losses)
        omit_bin_k = sum(omit_bin_losses)

        result = {
            'alpha': float(alpha),
            'factuality': {
                'risk': float(fact_result['adjusted_risk']),
                'empirical_risk': float(fact_k / n),
                'holds': bool(fact_result['adjusted_risk'] <= alpha),
                'workload': float(fact_flagged / n),
                'threshold': float(lambda_thresh),
            },
            'omission': {
                'risk': float(omit_result['adjusted_risk']),
                'empirical_risk': float(np.mean(omit_frac_losses)),
                'holds': bool(omit_result['adjusted_risk'] <= alpha),
                'workload': float(omit_surfaced / n),
                'tau': float(tau_thresh),
                'gamma': float(gamma_thresh),
                'loss_type': 'fractional',
            },
            'omission_binary': {
                'risk': float(omit_binary['adjusted_risk']),
                'empirical_risk': float(omit_bin_k / n),
                'holds': bool(omit_binary['adjusted_risk'] <= alpha),
                'workload': float(omit_bin_surfaced / n),
                'tau': float(tau_bin),
                'gamma': float(gamma_bin),
                'loss_type': 'binary',
            },
            'omission_1d': {
                'tau': float(omit_1d['threshold']),
                'workload': float(omit_1d.get('workload', 0)),
                'risk': float(omit_1d['adjusted_risk']),
                'holds': bool(omit_1d['adjusted_risk'] <= alpha),
            },
            'omission_product': {
                'beta': float(omit_product['threshold']),
                'workload': float(omit_product['workload']),
                'risk': float(omit_product['adjusted_risk']),
                'holds': bool(omit_product['adjusted_risk'] <= alpha),
            },
            'omission_score_gated': {
                'tau': float(omit_score_gated['threshold']),
                'gate': float(omit_score_gated['coverage_gate']),
                'workload': float(omit_score_gated['workload']),
                'risk': float(omit_score_gated['empirical_risk']),
            },
            'omission_union_bound': {
                'tau': float(omit_union_bound['tau']),
                'gamma': float(omit_union_bound['gamma']),
                'workload': float(omit_union_bound['workload']),
                'risk': float(omit_union_bound['adjusted_risk']),
                'holds': bool(omit_union_bound['adjusted_risk'] <= alpha),
            },
        }

        # Compute Wilson score CI on empirical risk, then transform to adjusted risk
        # Adjusted risk = (n/(n+1)) * empirical_risk + 1/(n+1)
        if compute_ci:
            fact_emp_ci = wilson_score_ci(fact_k, n)
            # For omission binary (secondary), use binary count
            omit_bin_emp_ci = wilson_score_ci(omit_bin_k, n)

            # Transform to adjusted risk CI (monotonic transformation preserves CI)
            crc_transform = lambda r: (n / (n + 1)) * r + 1 / (n + 1)
            fact_ci = (crc_transform(fact_emp_ci[0]), crc_transform(fact_emp_ci[1]))
            omit_bin_ci = (crc_transform(omit_bin_emp_ci[0]), crc_transform(omit_bin_emp_ci[1]))

            result['factuality']['risk_ci'] = fact_ci
            result['factuality']['empirical_risk_ci'] = fact_emp_ci
            result['omission_binary']['risk_ci'] = omit_bin_ci
            result['omission_binary']['empirical_risk_ci'] = omit_bin_emp_ci

        results.append(result)

    return results


def plot_calibration_curves(
    sweep_results: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Plot Risk Budget vs Workload curves for factuality and omission.
    Shows which alpha values result in feasible calibration (green) or infeasible (red).
    """
    # Compute document statistics for legend
    summary_sents = [len(d['factuality_probs']) for d in documents]
    source_sents = [len(d['importance_probs']) for d in documents]

    mean_summary = np.mean(summary_sents)
    median_summary = np.median(summary_sents)
    mean_source = np.mean(source_sents)
    median_source = np.median(source_sents)

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Factuality Plot =====
    alphas_f = [r['alpha'] * 100 for r in sweep_results]
    workloads_f = [r['factuality']['workload'] for r in sweep_results]
    holds_f = [r['factuality']['holds'] for r in sweep_results]

    ax1.step(alphas_f, workloads_f, where='post', color='#3498db', linewidth=2)
    for a, w, h in zip(alphas_f, workloads_f, holds_f):
        color = 'green' if h else 'red'
        ax1.scatter([a], [w], c=color, s=60, zorder=5)

    ax1.set_xlabel('Risk Budget α (%)', fontsize=12)
    ax1.set_ylabel('Sentences Flagged per Note', fontsize=12)
    ax1.set_title('Factuality: Risk Budget vs Workload (Calibration)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max(alphas_f) + 2)
    ax1.set_ylim(0, max(workloads_f) * 1.1 if workloads_f else 1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    legend_text = f'Summary sentences per note:\nMean: {mean_summary:.1f}  |  Median: {median_summary:.0f}\n\n● green = feasible\n● red = infeasible'
    ax1.text(0.98, 0.98, legend_text, transform=ax1.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    # ===== Omission Plot =====
    alphas_o = [r['alpha'] * 100 for r in sweep_results]
    workloads_o = [r['omission']['workload'] for r in sweep_results]
    holds_o = [r['omission']['holds'] for r in sweep_results]

    ax2.step(alphas_o, workloads_o, where='post', color='#9b59b6', linewidth=2)
    for a, w, h in zip(alphas_o, workloads_o, holds_o):
        color = 'green' if h else 'red'
        ax2.scatter([a], [w], c=color, s=60, zorder=5)

    ax2.set_xlabel('Risk Budget α (%)', fontsize=12)
    ax2.set_ylabel('Sentences Surfaced per Note', fontsize=12)
    ax2.set_title('Omission: Risk Budget vs Workload (Calibration)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(alphas_o) + 2)
    ax2.set_ylim(0, max(workloads_o) * 1.1 if workloads_o else 1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    legend_text = f'Source sentences per note:\nMean: {mean_source:.1f}  |  Median: {median_source:.0f}\n\n● green = feasible\n● red = infeasible'
    ax2.text(0.98, 0.98, legend_text, transform=ax2.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    plt.tight_layout()
    output_file = output_dir / 'calibration_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Calibration curves saved to {output_file}")


def plot_alpha_vs_risk(
    sweep_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Plot alpha vs calibration risk with safe/violated regions and bootstrap CI.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    alphas = [r['alpha'] for r in sweep_results]

    # Check if CI data is available
    has_ci = 'risk_ci' in sweep_results[0].get('factuality', {})

    # ===== Factuality Plot =====
    risks_f = [r['factuality']['risk'] for r in sweep_results]
    holds_f = [r['factuality']['holds'] for r in sweep_results]

    ax1.fill_between([0, 0.55], [0, 0.55], [0.55, 0.55], alpha=0.15, color='red', label='Violated region')
    ax1.fill_between([0, 0.55], [0, 0], [0, 0.55], alpha=0.15, color='green', label='Safe region')
    ax1.plot([0, 0.55], [0, 0.55], 'k--', alpha=0.5, linewidth=1.5, label='y = α')

    # Plot CI band if available
    if has_ci:
        ci_lower_f = [r['factuality']['risk_ci'][0] for r in sweep_results]
        ci_upper_f = [r['factuality']['risk_ci'][1] for r in sweep_results]
        ax1.fill_between(alphas, ci_lower_f, ci_upper_f, alpha=0.25, color='#3498db', label='95% Wilson CI')

    ax1.plot(alphas, risks_f, 'o-', color='#3498db', linewidth=2, markersize=8)

    for a, r, h in zip(alphas, risks_f, holds_f):
        color = 'green' if h else 'red'
        ax1.scatter([a], [r], c=color, s=100, zorder=5, edgecolors='white', linewidths=1)

    ax1.set_xlabel('Target α', fontsize=12)
    ax1.set_ylabel('Calibration Risk', fontsize=12)
    ax1.set_title('Factuality', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 0.55)
    ax1.set_ylim(0, 0.55)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ===== Omission Plot =====
    risks_o = [r['omission']['risk'] for r in sweep_results]
    holds_o = [r['omission']['holds'] for r in sweep_results]

    ax2.fill_between([0, 0.55], [0, 0.55], [0.55, 0.55], alpha=0.15, color='red', label='Violated region')
    ax2.fill_between([0, 0.55], [0, 0], [0, 0.55], alpha=0.15, color='green', label='Safe region')
    ax2.plot([0, 0.55], [0, 0.55], 'k--', alpha=0.5, linewidth=1.5, label='y = α')

    # Plot CI band if available (use binary CI if present)
    if has_ci and 'omission_binary' in sweep_results[0] and 'risk_ci' in sweep_results[0].get('omission_binary', {}):
        ci_lower_o = [r['omission_binary']['risk_ci'][0] for r in sweep_results]
        ci_upper_o = [r['omission_binary']['risk_ci'][1] for r in sweep_results]
        ax2.fill_between(alphas, ci_lower_o, ci_upper_o, alpha=0.25, color='#9b59b6', label='95% Wilson CI (binary)')

    ax2.plot(alphas, risks_o, 'o-', color='#9b59b6', linewidth=2, markersize=8)

    for a, r, h in zip(alphas, risks_o, holds_o):
        color = 'green' if h else 'red'
        ax2.scatter([a], [r], c=color, s=100, zorder=5, edgecolors='white', linewidths=1)

    ax2.set_xlabel('Target α', fontsize=12)
    ax2.set_ylabel('Calibration Risk', fontsize=12)
    ax2.set_title('Omission', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 0.55)
    ax2.set_ylim(0, 0.55)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    output_file = output_dir / 'alpha_vs_risk.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Alpha vs risk plot saved to {output_file}")


def generate_calibration_summary(
    factuality_result: Dict[str, Any],
    omission_result: Dict[str, Any],
    documents: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Generate a markdown summary of calibration results.
    """
    n_docs = len(documents)
    total_sents = sum(len(d['factuality_probs']) for d in documents)
    total_source = sum(len(d['importance_probs']) for d in documents)

    # Compute workload metrics at selected thresholds
    selected_lambda = factuality_result['threshold']
    selected_tau = omission_result['tau']
    selected_gamma = omission_result['gamma']

    n_flagged = 0
    n_errors = 0
    n_errors_flagged = 0
    for doc in documents:
        probs = np.array(doc['factuality_probs'])
        labels = np.array(doc['factuality_labels'])
        flagged = probs <= selected_lambda
        n_flagged += flagged.sum()
        errors = (labels == 0)
        n_errors += errors.sum()
        n_errors_flagged += (errors & flagged).sum()

    n_surfaced = 0
    n_true_omissions = 0
    n_omissions_surfaced = 0
    for doc in documents:
        imp_probs = np.array(doc['importance_probs'])
        cov_probs = np.array(doc['coverage_probs'])
        imp_labels = np.array(doc['importance_labels'])
        cov_labels = np.array(doc['coverage_labels'])

        non_cov_probs = 1.0 - cov_probs
        surfaced = (imp_probs >= selected_tau) & (non_cov_probs >= selected_gamma)
        n_surfaced += surfaced.sum()

        true_omissions = (imp_labels == 1) & (cov_labels == 0)
        n_true_omissions += true_omissions.sum()
        n_omissions_surfaced += (true_omissions & surfaced).sum()

    # Write markdown summary
    summary = f"""# Calibration Summary

## Thresholds Selected

| Controller | Threshold | Target α | Calibration Risk |
|------------|-----------|----------|------------------|
| **Factuality** | λ* = {selected_lambda:.2f} | ≤{factuality_result['alpha']*100:.0f}% | {factuality_result['adjusted_risk']*100:.1f}% |
| **Omission** | τ* = {selected_tau:.2f}, γ* = {selected_gamma:.2f} | ≤{omission_result['alpha']*100:.0f}% | {omission_result['adjusted_risk']*100:.1f}% |

## Expected Workload

### Factuality (Red Flags)
- **{n_flagged/n_docs:.1f} sentences flagged per note** ({100*n_flagged/total_sents:.1f}% of all sentences)
- Precision: {100*n_errors_flagged/n_flagged:.0f}% of flags are actual errors
- Recall: {100*n_errors_flagged/n_errors:.0f}% of errors are flagged

### Omission (Purple Surfaces)
- **{n_surfaced/n_docs:.1f} sentences surfaced per note** ({100*n_surfaced/total_source:.1f}% of source)
- Precision: {100*n_omissions_surfaced/n_surfaced:.1f}% of surfaces are true omissions
- Recall: {100*n_omissions_surfaced/n_true_omissions:.0f}% of true omissions are surfaced

## Interpretation

- **Factuality**: Expect ≤{factuality_result['alpha']*100:.0f}% of notes to have a missed hallucination
- **Omission**: Expect ≤{omission_result['alpha']*100:.0f}% of notes to have a missed important omission

## Files Generated
- `alpha_impact.png` - How α affects thresholds and workload
- `factuality_calibration_curve.png` - Factuality: α → sentences flagged
- `conformal_thresholds.json` - Machine-readable results
"""

    summary_file = output_dir / 'calibration_summary.md'
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"Calibration summary saved to {summary_file}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_phase3(
    input_file: Path = config.PHASE2_DIR / 'calibrated_scores.jsonl',
    output_dir: Path = config.PHASE3_DIR,
    alpha_fact: float = 0.10,
    alpha_omit: float = 0.35,
    grid_resolution: float = 0.01,
    omission_2d_resolution: float = 0.05,
    split: str = 'calibration',
):
    """
    Run Phase 3: Conformal risk control threshold selection.

    Args:
        input_file: Path to Phase 2 calibrated scores (JSONL)
        output_dir: Directory for output files
        alpha_fact: Factuality risk level (default: 0.10)
        alpha_omit: Omission risk level (default: 0.35)
        grid_resolution: Threshold grid resolution for 1D (default: 0.01)
        omission_2d_resolution: Grid resolution for 2D omission (default: 0.05)
        split: Which split to use ('calibration' or 'all'). Default: 'calibration'
    """
    logger.info("="*80)
    logger.info("Phase 3: Conformal Risk Control (Binary Instance Losses)")
    logger.info("="*80)

    # Setup
    config.setup_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 2 data
    logger.info(f"Loading Phase 2 scored documents from {input_file}...")
    documents = load_jsonl(input_file)

    # Filter out None docs
    n_before = len(documents)
    documents = [d for d in documents if d is not None]
    if len(documents) < n_before:
        logger.warning(f"Filtered out {n_before - len(documents)} None documents")

    # Filter to calibration split if requested
    if split == 'calibration':
        try:
            n_before = len(documents)
            documents = config.filter_documents_by_split(documents, 'calibration')
            logger.info(f"Filtered to calibration split: {len(documents)}/{n_before} documents")
        except FileNotFoundError:
            logger.warning("split_indices.json not found. Using all documents.")
    elif split == 'all':
        logger.info(f"Using all {len(documents)} documents (no split filtering)")
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'calibration' or 'all'")

    logger.info(f"Using {len(documents)} calibration documents")

    # Check binary loss floor
    n = len(documents)
    min_alpha = 1 / (n + 1)
    logger.info(f"Minimum feasible α for binary loss: {min_alpha:.4f} (n={n})")
    if alpha_fact < min_alpha:
        logger.warning(f"α_fact={alpha_fact} < min_α={min_alpha}. May have no feasible solution.")
    if alpha_omit < min_alpha:
        logger.warning(f"α_omit={alpha_omit} < min_α={min_alpha}. May have no feasible solution.")

    # Controller A: Factuality (binary instance loss)
    logger.info("\n" + "="*80)
    logger.info("Controller A: Factuality Filtering (Binary Instance Loss)")
    logger.info("="*80)
    factuality_result = select_factuality_threshold(
        documents,
        alpha=alpha_fact,
        grid_resolution=grid_resolution,
    )

    # Controller A baselines
    logger.info("\n" + "-"*40)
    logger.info("Computing Controller A baselines...")
    factuality_uncalibrated_result = select_factuality_threshold_uncalibrated(
        documents, alpha=alpha_fact)
    factuality_devset_result = select_factuality_threshold_devset(
        documents, alpha=alpha_fact, grid_resolution=grid_resolution)

    # Controller B: Omission (2D joint calibration with fractional loss - DEFAULT)
    logger.info("\n" + "="*80)
    logger.info("Controller B: Omission Surfacing (2D Fractional Loss - Default)")
    logger.info("="*80)
    omission_result = select_omission_threshold_2d_fractional(
        documents,
        alpha=alpha_omit,
        grid_resolution=omission_2d_resolution,
    )

    # Also compute binary loss thresholds (secondary metric)
    logger.info("\n" + "-"*40)
    logger.info("Computing 2D binary loss thresholds (secondary)...")
    omission_binary_result = select_omission_threshold_2d(
        documents,
        alpha=alpha_omit,
        grid_resolution=omission_2d_resolution,
    )

    # Also compute 1D baseline for comparison
    logger.info("\n" + "-"*40)
    logger.info("Computing 1D importance-only baseline...")
    omission_1d_result = select_omission_threshold_1d(
        documents,
        alpha=alpha_omit,
        grid_resolution=grid_resolution,
    )

    # Product composite baseline
    logger.info("\n" + "-"*40)
    logger.info("Computing product composite baseline...")
    omission_product_result = select_omission_threshold_product(
        documents,
        alpha=alpha_omit,
        grid_resolution=grid_resolution,
    )

    # Score-gated baseline
    logger.info("\n" + "-"*40)
    logger.info("Computing score-gated baseline...")
    omission_score_gated_result = select_omission_threshold_score_gated(
        documents,
        alpha=alpha_omit,
        grid_resolution=grid_resolution,
    )

    # Union bound baseline
    logger.info("\n" + "-"*40)
    logger.info("Computing union bound baseline...")
    omission_union_bound_result = select_omission_threshold_union_bound(
        documents,
        alpha=alpha_omit,
        grid_resolution=grid_resolution,
    )

    # Combine results
    results = {
        'factuality': factuality_result,
        'factuality_uncalibrated_baseline': factuality_uncalibrated_result,
        'factuality_devset_baseline': factuality_devset_result,
        'omission': omission_result,  # Fractional loss (default)
        'omission_binary': omission_binary_result,  # Binary loss (secondary)
        'omission_1d_baseline': omission_1d_result,
        'omission_product_baseline': omission_product_result,
        'omission_score_gated_baseline': omission_score_gated_result,
        'omission_union_bound_baseline': omission_union_bound_result,
        'config': {
            'alpha_fact': alpha_fact,
            'alpha_omit': alpha_omit,
            'grid_resolution': grid_resolution,
            'omission_2d_resolution': omission_2d_resolution,
            'n_calibration_docs': len(documents),
            'min_feasible_alpha': min_alpha,
            'omission_loss_type': 'fractional',
        },
    }

    # Save results
    output_file = output_dir / 'conformal_thresholds.json'
    logger.info(f"\nSaving results to {output_file}...")
    save_json(results, output_file)

    # Generate visualizations via alpha sweep
    logger.info("\nRunning alpha sweep for visualizations...")
    alpha_values = list(np.arange(0.05, 0.55, 0.05))
    sweep_results = run_alpha_sweep(documents, alpha_values)

    # Save sweep results
    sweep_file = output_dir / 'alpha_sweep_results.json'
    save_json(sweep_results, sweep_file)

    # Generate plots
    plot_calibration_curves(sweep_results, documents, output_dir)
    plot_alpha_vs_risk(sweep_results, output_dir)
    generate_calibration_summary(factuality_result, omission_result, documents, output_dir)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Phase 3 Complete!")
    logger.info("="*80)
    logger.info(f"Factuality threshold λ* = {factuality_result['threshold']:.3f}")
    logger.info(f"  Empirical risk: {factuality_result['empirical_risk']:.4f}")
    logger.info(f"  Adjusted risk: {factuality_result['adjusted_risk']:.4f} ≤ {alpha_fact}")
    logger.info(f"  Feasible: {factuality_result['is_feasible']}")
    logger.info(f"  Uncalibrated (λ=0.5): risk={factuality_uncalibrated_result['empirical_risk']:.4f}, "
                f"workload={factuality_uncalibrated_result['workload']:.1f}")
    logger.info(f"  Dev-set tuned: λ={factuality_devset_result['threshold']:.3f}, "
                f"risk={factuality_devset_result['empirical_risk']:.4f}, "
                f"workload={factuality_devset_result['workload']:.1f}")
    logger.info("")
    logger.info(f"Omission thresholds (τ*, γ*) = ({omission_result['tau']:.3f}, {omission_result['gamma']:.3f}) [fractional loss]")
    logger.info(f"  Fractional risk: {omission_result['empirical_risk']:.4f} (adjusted: {omission_result['adjusted_risk']:.4f} ≤ {alpha_omit})")
    logger.info(f"  Binary risk: {omission_result['binary_empirical_risk']:.4f} (adjusted: {omission_result['binary_adjusted_risk']:.4f})")
    logger.info(f"  Feasible: {omission_result['is_feasible']}")
    logger.info(f"  Feasible set size: {omission_result['feasible_set_size']}")
    logger.info(f"Binary 2D thresholds (τ*, γ*) = ({omission_binary_result['tau']:.3f}, {omission_binary_result['gamma']:.3f}) [secondary]")
    logger.info(f"  Binary adjusted risk: {omission_binary_result['adjusted_risk']:.4f}")
    logger.info(f"  Binary workload: {omission_binary_result['workload']:.1f}")
    logger.info("")
    logger.info(f"1D Baseline τ* = {omission_1d_result['threshold']:.3f} "
                f"(workload: {omission_1d_result.get('workload', 'N/A')})")
    logger.info(f"Product Baseline β* = {omission_product_result['threshold']:.3f} "
                f"(workload: {omission_product_result['workload']:.1f})")
    logger.info(f"Score-Gated: τ={omission_score_gated_result['threshold']:.3f}, "
                f"gate={omission_score_gated_result['coverage_gate']} "
                f"(workload: {omission_score_gated_result['workload']:.1f}, "
                f"no formal guarantee)")
    logger.info(f"Union Bound: τ={omission_union_bound_result['tau']:.3f}, "
                f"γ={omission_union_bound_result['gamma']:.3f} "
                f"(workload: {omission_union_bound_result['workload']:.1f})")
    logger.info("")
    logger.info(f"Output: {output_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Conformal Risk Control (Binary Instance Losses)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Input Phase 2 scored documents file (JSONL)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--alpha-fact',
        type=float,
        default=0.10,
        help='Factuality risk level (default: 0.10)'
    )
    parser.add_argument(
        '--alpha-omit',
        type=float,
        default=0.35,
        help='Omission risk level (default: 0.35)'
    )
    parser.add_argument(
        '--grid-resolution',
        type=float,
        default=0.01,
        help='1D grid resolution (default: 0.01)'
    )
    parser.add_argument(
        '--omission-2d-resolution',
        type=float,
        default=0.05,
        help='2D grid resolution for omission (default: 0.05)'
    )
    parser.add_argument(
        '--dataset',
        choices=["aci", "meq", "bhc", "cxr", "pubmed", "omop"],  # All datasets
        required=True,
        help='Define the dataset name'
    )
    parser.add_argument(
        '--split',
        choices=['calibration', 'all'],
        default='calibration',
        help='Which data split to use (default: calibration)'
    )

    args = parser.parse_args()

    config.configure_dataset(args.dataset)

    # Set defaults after configure_dataset
    input_file = args.input if args.input else config.PHASE2_DIR / 'calibrated_scores.jsonl'
    output_dir = args.output_dir if args.output_dir else config.PHASE3_DIR

    run_phase3(
        input_file=input_file,
        output_dir=output_dir,
        alpha_fact=args.alpha_fact,
        alpha_omit=args.alpha_omit,
        grid_resolution=args.grid_resolution,
        omission_2d_resolution=args.omission_2d_resolution,
        split=args.split,
    )


if __name__ == '__main__':
    main()
