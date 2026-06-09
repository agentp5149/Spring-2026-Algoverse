"""
Week 5: Epsilon Relaxed Projection Ablation (Molecular Surrogate)
=================================================================

Vijay's Week 5 tasks:
    1. Run epsilon relaxed projection ablation: compare exact projection vs
       epsilon-relaxed projection on outputs with known constraint violations.
    2. Validate coverage bound tightness: report predicted vs empirical
       coverage loss and the gap between them.
    3. Finalize all molecular surrogate results and confirm reproducibility.

Epsilon-relaxed projection:
    Instead of projecting every prediction unconditionally (exact projection),
    only project predictions whose energy violation exceeds epsilon. Predictions
    already within the epsilon-relaxed constraint manifold are left unchanged.

Results saved to:
    results/molecular/aspirin_week5_ablation.pt

Reads from (must already exist):
    results/molecular/week3_projection_results.pt
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.physics_projection import (
    compute_energy_violation,
    project_forces_energy_constraint,
    compute_energy_from_forces,
)

# ============================================================
# Paths
# ============================================================

RESULTS_DIR   = "results/molecular"
FIGURES_DIR   = "results/figures"
WEEK3_RESULTS = "results/molecular/week3_projection_results.pt"
OUT_ABLATION  = "results/molecular/aspirin_week5_ablation.pt"
OUT_FIGURE    = "results/figures/molecular_week5_bound_tightness.png"


# ============================================================
# Nonconformity score (matches Week 2/3 calibration exactly)
# ============================================================

def compute_nonconformity_scores(F_pred: torch.Tensor, F_true: torch.Tensor) -> torch.Tensor:
    """
    TrajectoryNormScore(normalize_by_length=True) from src/conformal.py:
        flatten (n, n_atoms, 3) -> (n, 63)
        score = ||F_pred - F_true||_2 / sqrt(63)

    This exactly reproduces the saved scores_before values.
    """
    flat = (F_pred - F_true).reshape(len(F_pred), -1)
    return flat.norm(dim=1) / np.sqrt(flat.shape[1])


def measure_coverage(F_pred: torch.Tensor, F_true: torch.Tensor, threshold: float) -> float:
    scores = compute_nonconformity_scores(F_pred, F_true)
    return (scores <= threshold).float().mean().item()


# ============================================================
# Epsilon-relaxed projection
# ============================================================

def project_epsilon_relaxed(
    F_pred: torch.Tensor,
    E_true_proxy: torch.Tensor,
    epsilon_threshold: float,
    n_steps: int = 10,
    lr: float = 0.01,
) -> tuple:
    """
    Only project snapshots whose energy violation exceeds epsilon_threshold.
    Snapshots already within the relaxed manifold are returned unchanged.
    """
    violations = compute_energy_violation(F_pred, None, E_true_proxy)
    projected_mask = violations > epsilon_threshold

    n_projected = projected_mask.sum().item()
    print(f"  Epsilon threshold: {epsilon_threshold:.4f}")
    print(f"  Snapshots requiring projection: {n_projected} / {len(F_pred)} "
          f"({100 * n_projected / len(F_pred):.1f}%)")

    F_relaxed = F_pred.clone()
    if n_projected > 0:
        F_sub = F_pred[projected_mask]
        E_sub = E_true_proxy[projected_mask]
        F_relaxed[projected_mask] = project_forces_energy_constraint(
            F_sub, E_sub, n_steps=n_steps, lr=lr
        )

    return F_relaxed, projected_mask


# ============================================================
# Main ablation
# ============================================================

def run_ablation(week3: dict, alpha: float) -> dict:
    print(f"\n{'='*60}")
    print(f"Ablation at alpha={alpha}")
    print(f"{'='*60}")

    proj_result = next(
        r for r in week3["projection_results"] if abs(r["alpha"] - alpha) < 1e-6
    )
    eps_result = next(
        r for r in week3["epsilon_results"] if abs(r["alpha"] - alpha) < 1e-6
    )

    F_raw       = proj_result["test_pred_raw"]
    F_projected = proj_result["test_pred_projected"]
    F_true      = proj_result["ground_truth"]
    threshold   = proj_result["threshold"]
    epsilon_95  = eps_result["epsilon_95"]

    E_true_proxy = compute_energy_from_forces(F_true)

    # ----------------------------------------------------------
    # Step 1: Unprojected
    # ----------------------------------------------------------
    print(f"\nStep 1: Unprojected coverage")
    cov_unprojected = measure_coverage(F_raw, F_true, threshold)
    print(f"  Coverage (unprojected): {cov_unprojected:.4f}")
    print(f"  Threshold:              {threshold:.4f}")

    # ----------------------------------------------------------
    # Step 2: Exact projection (saved + verified)
    # ----------------------------------------------------------
    print(f"\nStep 2: Exact projection coverage (from saved Week 3 results)")
    cov_exact_saved    = proj_result["coverage_after"]
    cov_exact_verified = measure_coverage(F_projected, F_true, threshold)
    width_exact        = proj_result["mean_width_after"]
    print(f"  Coverage (exact proj, saved):    {cov_exact_saved:.4f}")
    print(f"  Coverage (exact proj, verified): {cov_exact_verified:.4f}")
    print(f"  Mean width: {width_exact:.4f}")

    # ----------------------------------------------------------
    # Step 3: Epsilon-relaxed projection
    # ----------------------------------------------------------
    print(f"\nStep 3: Epsilon-relaxed projection (epsilon={epsilon_95:.4f})")
    F_relaxed, proj_mask = project_epsilon_relaxed(
        F_raw, E_true_proxy, epsilon_threshold=epsilon_95
    )
    cov_relaxed = measure_coverage(F_relaxed, F_true, threshold)
    print(f"  Coverage (relaxed proj): {cov_relaxed:.4f}")

    # ----------------------------------------------------------
    # Step 4: Bound tightness
    # ----------------------------------------------------------
    print(f"\nStep 4: Bound tightness analysis")
    predicted_loss         = eps_result["predicted_coverage_loss"]
    empirical_loss_exact   = abs(cov_unprojected - cov_exact_verified)
    empirical_loss_relaxed = abs(cov_unprojected - cov_relaxed)
    bound_gap_exact        = abs(predicted_loss - empirical_loss_exact)
    bound_gap_relaxed      = abs(predicted_loss - empirical_loss_relaxed)

    print(f"  Predicted coverage loss (bound):        {predicted_loss:.4f}  ({predicted_loss:.1%})")
    print(f"  Empirical coverage loss (exact proj):   {empirical_loss_exact:.4f}  ({empirical_loss_exact:.1%})")
    print(f"  Empirical coverage loss (relaxed proj): {empirical_loss_relaxed:.4f}  ({empirical_loss_relaxed:.1%})")
    print(f"  Bound gap (exact):   {bound_gap_exact:.4f}  ({'TIGHT' if bound_gap_exact <= 0.03 else 'LOOSE'})")
    print(f"  Bound gap (relaxed): {bound_gap_relaxed:.4f}  ({'TIGHT' if bound_gap_relaxed <= 0.03 else 'LOOSE'})")

    return {
        "alpha":                      alpha,
        "threshold":                  threshold,
        "epsilon_95":                 epsilon_95,
        "n_test":                     len(F_raw),
        "n_projected_relaxed":        int(proj_mask.sum().item()),
        "frac_projected_relaxed":     proj_mask.float().mean().item(),
        "coverage_unprojected":       cov_unprojected,
        "coverage_exact":             cov_exact_verified,
        "coverage_relaxed":           cov_relaxed,
        "width_exact":                width_exact,
        "coverage_loss_exact":        empirical_loss_exact,
        "coverage_loss_relaxed":      empirical_loss_relaxed,
        "predicted_coverage_loss":    predicted_loss,
        "bound_gap_exact":            bound_gap_exact,
        "bound_gap_relaxed":          bound_gap_relaxed,
        "bound_tight_exact":          bound_gap_exact <= 0.03,
        "bound_tight_relaxed":        bound_gap_relaxed <= 0.03,
    }


# ============================================================
# Summary table
# ============================================================

def print_summary(results: list):
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    header = (f"{'alpha':>6}  {'Unprojected':>12}  {'Exact proj':>11}"
              f"  {'Relaxed proj':>13}  {'Pred loss':>10}"
              f"  {'Gap (exact)':>12}  {'Gap (relaxed)':>13}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['alpha']:>6.2f}  "
            f"{r['coverage_unprojected']:>12.4f}  "
            f"{r['coverage_exact']:>11.4f}  "
            f"{r['coverage_relaxed']:>13.4f}  "
            f"{r['predicted_coverage_loss']:>10.4f}  "
            f"{r['bound_gap_exact']:>12.4f}  "
            f"{r['bound_gap_relaxed']:>13.4f}"
        )


# ============================================================
# Figure: bound tightness bar chart
# ============================================================

def plot_bound_tightness(results: list, out_path: str):
    labels      = [f"alpha={r['alpha']}" for r in results]
    predicted   = [r["predicted_coverage_loss"]  for r in results]
    exact_emp   = [r["coverage_loss_exact"]       for r in results]
    relaxed_emp = [r["coverage_loss_relaxed"]     for r in results]

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, predicted,   width, label="Predicted (bound)",        color="#4C72B0")
    ax.bar(x,         exact_emp,   width, label="Empirical (exact proj)",   color="#DD8452")
    ax.bar(x + width, relaxed_emp, width, label="Empirical (relaxed proj)", color="#55A868")
    ax.axhline(0.03, color="gray", linestyle="--", linewidth=0.8,
               label="3pp tightness threshold")

    ax.set_ylabel("Coverage loss")
    ax.set_title("Bound tightness: predicted vs empirical coverage loss\n"
                 "Molecular surrogate (aspirin, MD17)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(exact_emp) * 1.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nBound tightness figure saved to {out_path}")


# ============================================================
# Reproducibility check
# ============================================================

def check_reproducibility(results: list):
    print(f"\n{'='*60}")
    print("REPRODUCIBILITY CHECK")
    print(f"{'='*60}")
    all_pass = True
    for r in results:
        nominal = 1.0 - r["alpha"]
        cov     = r["coverage_unprojected"]
        gap     = abs(cov - nominal)
        status  = "PASS" if gap <= 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  alpha={r['alpha']}: unprojected coverage={cov:.4f}, "
              f"nominal={nominal:.2f}, gap={gap:.4f}  [{status}]")
    if all_pass:
        print("  All coverage checks passed (within 2pp of nominal).")
    else:
        print("  WARNING: one or more coverage checks failed.")
    return all_pass


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Week 5: Epsilon relaxed projection ablation"
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10],
                        help="Miscoverage levels to run (default: 0.05 0.10)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading saved results...")
    week3 = torch.load(WEEK3_RESULTS, weights_only=False)
    print(f"  Week 3 results loaded: {len(week3['projection_results'])} alpha levels")

    all_results = []
    for alpha in args.alphas:
        result = run_ablation(week3, alpha)
        all_results.append(result)

    print_summary(all_results)
    check_reproducibility(all_results)
    plot_bound_tightness(all_results, OUT_FIGURE)

    torch.save({"ablation_results": all_results, "molecule": "aspirin"}, OUT_ABLATION)
    print(f"\nAblation results saved to {OUT_ABLATION}")

    print("\nFiles to commit:")
    print("  molecular_week5.py")


if __name__ == "__main__":
    main()
