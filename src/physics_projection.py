"""
Physics-Constrained Conformal Projection
=========================================
src/physics_projection.py -- Team PVAV

Implements Step 3 of the proposal: projecting conformal prediction sets
onto physical constraint manifolds with provable coverage guarantees.

This module handles the molecular dynamics case (energy conservation,
quadratic constraint). The weather case (mass conservation, linear
constraint) and equivariance case live in separate modules.

Energy conservation constraint:
    The surrogate's predicted forces must satisfy ||F||^2 = E_target,
    where E_target is the mean energy proxy computed from the calibration
    set. This is a quadratic (sphere) constraint in force space with an
    exact closed-form projection:

        F_proj = F * sqrt(E_target / ||F||^2)

    This scales each prediction to lie on the energy manifold while
    minimizing the L2 distance from the original prediction.

Exports:
    compute_energy_from_forces(F, R)
    compute_energy_violation(F_pred, E_target)
    compute_energy_violation_batched(F_pred, E_target, batch_size)
    project_forces_energy_constraint(F_pred, E_target)
    project_forces_energy_constraint_batched(F_pred, E_target, batch_size)
    compute_epsilon_relaxation_bound(get_predictions_fn, cal_R, cal_F, cal_E, alpha, projection_results)
"""

import numpy as np
import torch


# ============================================================
# Energy conservation constraint
# ============================================================

def compute_energy_from_forces(F: torch.Tensor, R=None) -> torch.Tensor:
    """
    Approximate total energy proxy from force magnitudes.

    Uses ||F||^2 summed over atoms, which is proportional to kinetic
    energy in the harmonic approximation.

    Args:
        F: (n, n_atoms, 3)  predicted or true forces
        R: unused, kept for API consistency

    Returns:
        energy_proxy: (n,)  estimated energy per snapshot
    """
    return (F ** 2).sum(dim=[1, 2])


def compute_energy_violation(
    F_pred: torch.Tensor,
    E_target: float,
) -> torch.Tensor:
    """
    Compute per-snapshot energy conservation violation.

    Violation = | ||F_pred||^2 - E_target | / E_target

    This is the relative deviation from the target energy manifold.

    Args:
        F_pred:   (n, n_atoms, 3)  surrogate force predictions
        E_target: scalar           target energy (e.g. calibration mean)

    Returns:
        violations: (n,)  relative energy violations per snapshot
    """
    E_proxy = compute_energy_from_forces(F_pred)
    return (E_proxy - E_target).abs() / (abs(E_target) + 1e-8)


def compute_energy_violation_batched(
    F_pred: torch.Tensor,
    E_target: float,
    batch_size: int = 2000,
) -> torch.Tensor:
    """
    Batched version of compute_energy_violation for large datasets.

    Args:
        F_pred:     (n, n_atoms, 3)
        E_target:   scalar target energy
        batch_size: chunk size

    Returns:
        violations: (n,)
    """
    results = []
    for i in range(0, len(F_pred), batch_size):
        results.append(compute_energy_violation(F_pred[i:i + batch_size], E_target))
    return torch.cat(results, dim=0)


def project_forces_energy_constraint(
    F_pred: torch.Tensor,
    E_target: float,
) -> torch.Tensor:
    """
    Closed-form projection onto the energy conservation manifold.

    The energy conservation constraint is:
        ||F||^2 = E_target

    This defines a sphere in force space. The closest point on this
    sphere to F_pred is:

        F_proj = F_pred * sqrt(E_target / ||F_pred||^2)

    This is the exact projection -- no gradient descent needed.
    Coverage preservation follows directly: if the true output already
    satisfies the constraint (||F_true||^2 = E_target), projection
    leaves it unchanged and it remains inside the conformal set.

    Args:
        F_pred:   (n, n_atoms, 3)  surrogate predictions to project
        E_target: scalar           target energy value

    Returns:
        F_projected: (n, n_atoms, 3)  projected forces
    """
    E_current = compute_energy_from_forces(F_pred)  # (n,)
    scale = (E_target / (E_current + 1e-8)).sqrt()  # (n,)
    scale = scale.unsqueeze(-1).unsqueeze(-1)        # (n, 1, 1) for broadcasting
    return F_pred * scale


def project_forces_energy_constraint_batched(
    F_pred: torch.Tensor,
    E_target: float,
    batch_size: int = 2000,
) -> torch.Tensor:
    """
    Batched wrapper around project_forces_energy_constraint.
    Processes F_pred in chunks to avoid OOM on large test sets.

    Args:
        F_pred:     (n, n_atoms, 3)
        E_target:   scalar target energy
        batch_size: chunk size

    Returns:
        F_projected: (n, n_atoms, 3)
    """
    results = []
    n = len(F_pred)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        results.append(project_forces_energy_constraint(F_pred[start:end], E_target))
        print(f"  Projected {end}/{n} ({100*end/n:.0f}%)", end="\r")
    print()
    return torch.cat(results, dim=0)


# ============================================================
# Epsilon relaxation bound (Task 3)
# ============================================================

def compute_epsilon_relaxation_bound(
    get_predictions_fn,
    cal_R: torch.Tensor,
    cal_F: torch.Tensor,
    cal_E: torch.Tensor,
    alpha: float,
    projection_results: dict,
) -> dict:
    """
    Compute the epsilon relaxation bound on coverage loss (Task 3).

    The energy target is the mean ||F||^2 over the calibration set.
    Epsilon is the 95th percentile of relative violations on the
    calibration set.

    Predicted coverage loss = P(violation > epsilon) ~ 5% by construction.

    Args:
        get_predictions_fn: callable(R) -> F_pred tensor
        cal_R, cal_F, cal_E: calibration set
        alpha:               miscoverage level
        projection_results:  dict from run_conformal_with_projection

    Returns:
        dict with epsilon stats, predicted/empirical coverage loss, and gap
    """
    print(f"\n--- Epsilon Relaxation Bound (alpha={alpha}) ---")

    # Batch predictions to avoid OOM on large calibration sets
    batch_size = 2000
    cal_pred_chunks = []
    for i in range(0, len(cal_R), batch_size):
        cal_pred_chunks.append(get_predictions_fn(cal_R[i:i + batch_size]))
    cal_pred = torch.cat(cal_pred_chunks, dim=0)

    # Energy target = mean calibration energy proxy
    E_target = compute_energy_from_forces(cal_pred).mean().item()

    violations = compute_energy_violation_batched(cal_pred, E_target)

    viol_np        = violations.numpy()
    epsilon_mean   = float(viol_np.mean())
    epsilon_median = float(np.median(viol_np))
    epsilon_95     = float(np.percentile(viol_np, 95))
    epsilon_max    = float(viol_np.max())

    print(f"  Energy target (cal mean proxy): {E_target:.4f}")
    print(f"  Energy violation distribution (calibration set):")
    print(f"    Mean   epsilon : {epsilon_mean:.6f}")
    print(f"    Median epsilon : {epsilon_median:.6f}")
    print(f"    95th pct       : {epsilon_95:.6f}")
    print(f"    Max    epsilon : {epsilon_max:.6f}")

    epsilon_threshold       = epsilon_95
    frac_violated           = (violations > epsilon_threshold).float().mean().item()
    predicted_coverage_loss = frac_violated
    empirical_coverage_loss = projection_results["coverage_loss_empirical"]

    print(f"\n  Epsilon bound analysis:")
    print(f"    Epsilon threshold (95th pct)   : {epsilon_threshold:.6f}")
    print(f"    Predicted coverage loss bound  : {predicted_coverage_loss:.1%}")
    print(f"    Empirical coverage loss        : {empirical_coverage_loss:.1%}")
    gap = abs(predicted_coverage_loss - abs(empirical_coverage_loss))
    print(f"    Gap (bound tightness)          : {gap:.1%}")

    if gap <= 0.03:
        print(f"    ✓ Bound is tight (gap <= 3pp) -- theory validated")
    else:
        print(f"    ✗ Bound is loose (gap > 3pp) -- report honestly in paper")

    return {
        "molecule":                 projection_results["molecule"],
        "alpha":                    alpha,
        "epsilon_mean":             epsilon_mean,
        "epsilon_median":           epsilon_median,
        "epsilon_95":               epsilon_95,
        "epsilon_max":              epsilon_max,
        "epsilon_threshold":        epsilon_threshold,
        "E_target":                 E_target,
        "predicted_coverage_loss":  predicted_coverage_loss,
        "empirical_coverage_loss":  empirical_coverage_loss,
        "bound_gap":                gap,
        "bound_tight":              gap <= 0.03,
        "cal_violations":           violations,
        "coverage_before":          projection_results["coverage_before"],
        "coverage_after":           projection_results["coverage_after"],
    }
