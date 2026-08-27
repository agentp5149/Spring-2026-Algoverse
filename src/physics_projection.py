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
    where E_target is the mean ||F||^2 computed from the TRUE calibration
    forces (not the surrogate's own predictions -- using predictions here
    is self-referential and was a real bug in an earlier version of this
    module). This is a quadratic (sphere) constraint in force space with
    an exact closed-form projection:

        F_proj = F * sqrt(E_target / ||F||^2)

    This scales each prediction to lie on the energy manifold while
    minimizing the L2 distance from the original prediction.

    IMPORTANT CAVEAT: ||F||^2 is a proxy for true energy, justified by a
    harmonic approximation (force magnitude squared is proportional to
    energy above a local minimum, under a quadratic potential). This is
    not exact. fit_energy_force_relationship() empirically checks how well
    this proxy actually tracks true energy E on calibration data, via a
    linear fit and R^2. Across MD17 molecules this comes out to roughly
    R^2 = 0.35-0.42, meaning the proxy is real but loose, not a tight
    physical law. Report R^2 alongside any energy-conservation claim
    rather than presenting it as exact.

REVISION (post-Kiran feedback, round 2):
    compute_epsilon_relaxation_bound() previously computed epsilon_95 as
    the 95th percentile of violations on cal_F, then checked what
    fraction of that SAME cal_F violations array exceeded that SAME
    threshold. That is circular: a 95th-percentile threshold will show
    ~5% "exceedance" against the data it was computed from almost by
    definition, regardless of whether the underlying bound holds on new
    data. The function now requires test_F and validates the
    calibration-derived epsilon threshold against held-out TEST
    violations instead, which is what the paper's claim
    "P(V(Y_test) > epsilon) ~ delta" and Theorem 1 actually assert.

Exports:
    compute_energy_from_forces(F, R)
    fit_energy_force_relationship(cal_F, cal_E)
    compute_energy_violation(F_pred, E_target)
    compute_energy_violation_batched(F_pred, E_target, batch_size)
    project_forces_energy_constraint(F_pred, E_target)
    project_forces_energy_constraint_batched(F_pred, E_target, batch_size)
    compute_epsilon_relaxation_bound(cal_F, cal_E, test_F, E_target, alpha, projection_results)
"""

import numpy as np
import torch


# ============================================================
# Energy conservation constraint
# ============================================================

def compute_energy_from_forces(F: torch.Tensor, R=None) -> torch.Tensor:
    """
    Approximate total energy proxy from force magnitudes.

    Uses ||F||^2 summed over atoms, which is proportional to energy above
    a local minimum under a harmonic (quadratic) approximation of the
    potential. See fit_energy_force_relationship() for an empirical check
    of how well this approximation actually holds on real data.

    Args:
        F: (n, n_atoms, 3)  predicted or true forces
        R: unused, kept for API consistency

    Returns:
        energy_proxy: (n,)  estimated energy per snapshot
    """
    return (F ** 2).sum(dim=[1, 2])


def fit_energy_force_relationship(cal_F: torch.Tensor, cal_E: torch.Tensor) -> dict:
    """
    Empirically test whether ||F||^2 actually tracks true energy E.

    Fits E_centered = a + b * ||F||^2 by ordinary least squares, where
    E_centered = E - mean(E). Centering matters: raw MD17 energies carry a
    huge DFT baseline offset (on the order of -1e5 to -1e6 depending on
    molecule) that swamps the chemically relevant fluctuation if left in,
    so any relative-violation comparison must be made after centering, not
    against the raw absolute energy.

    This must be run on TRUE forces and TRUE energies only. Running it on
    model predictions would conflate "is this proxy physically valid" with
    "is the model accurate," which are different questions.

    Args:
        cal_F: (n_cal, n_atoms, 3) true calibration forces
        cal_E: (n_cal,) true calibration energies, already flattened to 1D

    Returns:
        dict with slope, intercept, r_squared, correlation, e_mean, e_std
    """
    fsq = compute_energy_from_forces(cal_F).numpy()
    e = cal_E.numpy().reshape(-1)

    if len(e) != len(fsq):
        raise ValueError(
            f"cal_F and cal_E length mismatch: {len(fsq)} forces vs {len(e)} "
            f"energies. Check that cal_E was flattened to 1D after loading "
            f"(MD17 .npz files store E with shape (N, 1), not (N,))."
        )

    e_mean = float(e.mean())
    e_std = float(e.std())
    e_centered = e - e_mean

    slope, intercept = np.polyfit(fsq, e_centered, 1)
    pred = slope * fsq + intercept
    ss_res = float(((e_centered - pred) ** 2).sum())
    ss_tot = float((e_centered ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    correlation = float(np.corrcoef(fsq, e_centered)[0, 1])

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "correlation": correlation,
        "e_mean": e_mean,
        "e_std": e_std,
    }


def compute_energy_violation(
    F_pred: torch.Tensor,
    E_target: float,
) -> torch.Tensor:
    """
    Compute per-snapshot energy conservation violation.

    Violation = | ||F_pred||^2 - E_target | / E_target

    This is the relative deviation from the target energy manifold.
    E_target and ||F_pred||^2 are both in force-squared units here (not
    raw kcal/mol energy units), so this relative comparison is well-scaled
    regardless of which molecule's force magnitudes are involved.

    Args:
        F_pred:   (n, n_atoms, 3)  surrogate force predictions, or true forces
        E_target: scalar           target ||F||^2 (from TRUE calibration data)

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
        E_target: scalar           target energy value, from TRUE calibration data

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
    cal_F: torch.Tensor,
    cal_E: torch.Tensor,
    test_F: torch.Tensor,
    E_target: float,
    alpha: float,
    projection_results: dict,
) -> dict:
    """
    Compute the epsilon relaxation bound on coverage loss (Task 3).

    Epsilon is the 95th percentile of relative ||F||^2 violations measured
    on the TRUE calibration forces against E_target. E_target should be
    computed from TRUE calibration forces upstream (see
    run_conformal_with_projection in molecular_week3.py). Deriving
    epsilon from calibration data is legitimate -- that is exactly what
    conformal calibration is supposed to do.

    FIXED (post-Kiran feedback, round 2): the bound used to be "validated"
    by checking what fraction of the SAME cal_F violations array exceeded
    the SAME 95th-percentile threshold computed from it. That is circular
    -- a 95th-percentile threshold shows ~5% "exceedance" against the data
    it was computed from almost by construction, regardless of whether
    the bound generalizes. This function now requires test_F and checks
    the calibration-derived threshold against held-out TEST violations,
    which is what the paper's claim "P(V(Y_test) > epsilon) ~ delta" and
    Theorem 1 actually assert. The old circular number is still returned
    (as cal_exceedance_rate_OLD_CIRCULAR) purely so you can compare it to
    the fixed number and confirm this change actually did something --
    do not report that field in the paper.

    This also runs fit_energy_force_relationship() on the same calibration
    data and reports R^2 / correlation, so the validity of treating ||F||^2
    as an energy proxy is checked and recorded every time this runs, not
    just in a one-off diagnostic.

    Args:
        cal_F:               (n_cal, n_atoms, 3) TRUE calibration forces
        cal_E:                (n_cal,) TRUE calibration energies, flattened to 1D
        test_F:              (n_test, n_atoms, 3) TRUE held-out test forces
        E_target:            scalar target ||F||^2, from TRUE calibration data
        alpha:                miscoverage level
        projection_results:  dict from run_conformal_with_projection

    Returns:
        dict with epsilon stats, target delta, the HONEST test-set
        exceedance rate (this is the number to report in the paper), the
        old circular cal-vs-cal number (debugging only, do not report),
        empirical coverage loss, gap, and the energy-force proxy
        validation (r_squared, correlation)
    """
    print(f"\n--- Epsilon Relaxation Bound (alpha={alpha}) ---")

    fit = fit_energy_force_relationship(cal_F, cal_E)
    print(f"  Energy-force proxy validation (TRUE calibration data):")
    print(f"    Correlation(||F||^2, E_centered) : {fit['correlation']:.4f}")
    print(f"    R^2 of linear fit                : {fit['r_squared']:.4f}")
    if fit["r_squared"] < 0.5:
        print(f"    NOTE: R^2 below 0.5. This proxy is real but loose --")
        print(f"    report it as an approximate energy-consistency constraint,")
        print(f"    not as exact energy conservation.")

    # Epsilon is legitimately derived from calibration data.
    cal_violations = compute_energy_violation_batched(cal_F, E_target)
    cal_viol_np    = cal_violations.numpy()
    epsilon_mean   = float(cal_viol_np.mean())
    epsilon_median = float(np.median(cal_viol_np))
    epsilon_95     = float(np.percentile(cal_viol_np, 95))
    epsilon_max    = float(cal_viol_np.max())

    print(f"  Energy target (TRUE cal mean ||F||^2): {E_target:.4f}")
    print(f"  Energy violation distribution (TRUE calibration forces):")
    print(f"    Mean   epsilon : {epsilon_mean:.6f}")
    print(f"    Median epsilon : {epsilon_median:.6f}")
    print(f"    95th pct       : {epsilon_95:.6f}")
    print(f"    Max    epsilon : {epsilon_max:.6f}")

    epsilon_threshold = epsilon_95

    # OLD (circular) check -- kept only so you can see the old number
    # alongside the fixed one and confirm the fix changed something.
    # DO NOT report this number in the paper.
    frac_violated_circular = (cal_violations > epsilon_threshold).float().mean().item()

    # FIXED check: validate the calibration-derived threshold against
    # held-out TEST violations. This is the number that actually tests
    # whether the epsilon-relaxed bound generalizes.
    test_violations = compute_energy_violation_batched(test_F, E_target)
    frac_violated_test = (test_violations > epsilon_threshold).float().mean().item()

    target_delta = 0.05  # nominal delta used to pick the 95th percentile
    empirical_coverage_loss = projection_results["coverage_loss_empirical"]

    print(f"\n  Epsilon bound analysis:")
    print(f"    Epsilon threshold (95th pct, from CAL)      : {epsilon_threshold:.6f}")
    print(f"    Target delta                                : {target_delta:.1%}")
    print(f"    [OLD, circular] cal-vs-cal exceedance        : {frac_violated_circular:.1%}")
    print(f"    [FIXED] TEST-set exceedance rate             : {frac_violated_test:.1%}")
    print(f"    Empirical coverage loss (before vs after)    : {empirical_coverage_loss:.1%}")

    gap = abs(frac_violated_test - target_delta)
    print(f"    Gap (bound tightness, test-validated)        : {gap:.1%}")

    if gap <= 0.03:
        print(f"    Bound is tight (gap <= 3pp) -- theory validated on held-out data")
    else:
        print(f"    Bound is loose (gap > 3pp) -- report honestly in paper")

    return {
        "molecule":                          projection_results["molecule"],
        "alpha":                             alpha,
        "epsilon_mean":                      epsilon_mean,
        "epsilon_median":                    epsilon_median,
        "epsilon_95":                        epsilon_95,
        "epsilon_max":                       epsilon_max,
        "epsilon_threshold":                 epsilon_threshold,
        "E_target":                          E_target,
        "target_delta":                      target_delta,
        "test_exceedance_rate":              frac_violated_test,      # <-- report THIS in the paper
        "cal_exceedance_rate_OLD_CIRCULAR":  frac_violated_circular,  # debugging only, do not report
        "empirical_coverage_loss":           empirical_coverage_loss,
        "bound_gap":                         gap,
        "bound_tight":                       gap <= 0.03,
        "cal_violations":                    cal_violations,
        "test_violations":                   test_violations,
        "coverage_before":                   projection_results["coverage_before"],
        "coverage_after":                    projection_results["coverage_after"],
        "energy_force_r_squared":            fit["r_squared"],
        "energy_force_correlation":          fit["correlation"],
        "energy_force_slope":                fit["slope"],
        "energy_force_intercept":            fit["intercept"],
    }
