"""
Physics-Constrained Conformal Projection
=========================================
src/physics_projection.py -- Team PVAV

Implements Step 3 of the proposal: projecting conformal prediction sets
onto physical constraint manifolds with provable coverage guarantees.

This module handles the molecular dynamics case (energy conservation,
quadratic constraint). The weather case (mass conservation, linear
constraint) and equivariance case live in separate modules.

Exports:
    compute_energy_from_forces(F, R)
    compute_energy_violation(F_pred, R, E_true)
    project_forces_energy_constraint(F_pred, E_target, n_steps, lr)
    compute_epsilon_relaxation_bound(model, cal_R, cal_F, cal_E, alpha, projection_results)
"""

import numpy as np
import torch


# ============================================================
# Energy conservation constraint
# ============================================================

def compute_energy_from_forces(F: torch.Tensor, R=None) -> torch.Tensor:
    """
    Approximate total energy proxy from force magnitudes.

    In a real MD system, the work-energy theorem states:
        W = integral of F dot dR

    For our surrogate we use ||F||^2 summed over atoms as a proxy,
    which is proportional to kinetic energy in the harmonic approximation.

    Args:
        F: (n, n_atoms, 3)  predicted or true forces
        R: unused, kept for API consistency with position-based energy functions

    Returns:
        energy_proxy: (n,)  estimated energy per snapshot
    """
    return (F ** 2).sum(dim=[1, 2])


def compute_energy_violation(
    F_pred: torch.Tensor,
    R: torch.Tensor,
    E_true: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-snapshot energy conservation violation (epsilon).

    Normalizes both the proxy and DFT energies to a common scale before
    comparing, since they are in different units (force^2 vs kcal/mol).

    epsilon_i = | E_proxy_norm(F_pred_i) - E_true_norm_i |

    Args:
        F_pred: (n, n_atoms, 3)  surrogate force predictions
        R:      (n, n_atoms, 3)  atomic positions (unused, for API consistency)
        E_true: (n,)             DFT reference energies (kcal/mol)

    Returns:
        violations: (n,)  normalized energy violations per snapshot
    """
    E_proxy = compute_energy_from_forces(F_pred, R)

    E_true_norm  = (E_true  - E_true.mean())  / (E_true.std()  + 1e-8)
    E_proxy_norm = (E_proxy - E_proxy.mean()) / (E_proxy.std() + 1e-8)

    return (E_proxy_norm - E_true_norm).abs()


def project_forces_energy_constraint(
    F_pred: torch.Tensor,
    E_target: torch.Tensor,
    n_steps: int = 10,
    lr: float = 0.01,
) -> torch.Tensor:
    """
    Gradient-based projection onto the energy conservation manifold.

    The energy conservation constraint is quadratic in the forces:
        g(F) = ||F||^2_F - E_target = 0

    This is a sphere in force space. Unlike the linear (mass conservation)
    case, there is no closed-form projection, so we use gradient descent
    on the Lagrangian relaxation:

        min_{F} ||F - F_pred||^2 + lambda * g(F)^2

    Args:
        F_pred:   (n, n_atoms, 3)  surrogate predictions to project
        E_target: (n,)             target energy values
        n_steps:  number of gradient steps (10 is fast and sufficient)
        lr:       SGD step size

    Returns:
        F_projected: (n, n_atoms, 3)  projected forces (detached)
    """
    F = F_pred.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([F], lr=lr)

    # Scale E_target to match force-squared units
    E_proxy_orig = compute_energy_from_forces(F_pred.detach())
    scale = E_proxy_orig.mean().item() / (E_target.abs().mean().item() + 1e-8)
    E_scaled = E_target * scale

    for _ in range(n_steps):
        optimizer.zero_grad()
        recon_loss      = ((F - F_pred.detach()) ** 2).sum(dim=[1, 2]).mean()
        E_current       = compute_energy_from_forces(F)
        constraint_loss = ((E_current - E_scaled) ** 2).mean()
        total_loss      = recon_loss + 1.0 * constraint_loss
        total_loss.backward()
        optimizer.step()

    return F.detach()


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

    Theory:
        Let epsilon = 95th percentile of calibration energy violations.
        Predicted coverage loss = P(violation > epsilon) ~ 5% by construction.

        If the bound is tight:
            empirical_coverage_loss ≈ predicted_coverage_loss

        If the bound is loose, report the gap honestly -- it means the
        surrogate's energy violations are larger on test than calibration,
        which motivates tighter quadratic-specific bounds (see proof doc).

    Args:
        get_predictions_fn: callable(R) -> F_pred tensor
        cal_R, cal_F, cal_E: calibration set
        alpha:               miscoverage level
        projection_results:  dict from run_conformal_with_projection

    Returns:
        dict with epsilon stats, predicted/empirical coverage loss, and gap
    """
    print(f"\n--- Epsilon Relaxation Bound (alpha={alpha}) ---")

    cal_pred = get_predictions_fn(cal_R)
    violations = compute_energy_violation(cal_pred, cal_R, cal_E)

    viol_np        = violations.numpy()
    epsilon_mean   = float(viol_np.mean())
    epsilon_median = float(np.median(viol_np))
    epsilon_95     = float(np.percentile(viol_np, 95))
    epsilon_max    = float(viol_np.max())

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
        "predicted_coverage_loss":  predicted_coverage_loss,
        "empirical_coverage_loss":  empirical_coverage_loss,
        "bound_gap":                gap,
        "bound_tight":              gap <= 0.03,
        "cal_violations":           violations,
        "coverage_before":          projection_results["coverage_before"],
        "coverage_after":           projection_results["coverage_after"],
    }
