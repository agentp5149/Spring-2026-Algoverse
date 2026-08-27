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
    only project predictions whose energy violation exceeds epsilon_95.
    Uses the closed-form projection: F_proj = F * sqrt(E_target / ||F||^2).

Results saved to:
    results/molecular/<molecule>_week5_ablation.pt
    results/figures/molecular_week5_bound_tightness.png

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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from physics_projection import (
    compute_energy_violation,
    compute_energy_from_forces,
    project_forces_energy_constraint,
)

# ============================================================
# Paths
# ============================================================

RESULTS_DIR   = "results/molecular"
FIGURES_DIR   = "results/figures"
WEEK3_RESULTS = "results/molecular/week3_projection_results.pt"

MOLECULES = ["aspirin", "ethanol", "uracil", "malonaldehyde"]


# ============================================================
# Nonconformity score (matches Week 2/3 calibration exactly)
# ============================================================

def compute_nonconformity_scores(F_pred: torch.Tensor, F_true: torch.Tensor) -> torch.Tensor:
    """
    TrajectoryNormScore(normalize_by_length=True):
        score = ||F_pred - F_true||_2 / sqrt(n_atoms * 3)
    """
    flat = (F_pred - F_true).reshape(len(F_pred), -1)
    return flat.norm(dim=1) / np.sqrt(flat.shape[1])


def measure_coverage(F_pred: torch.Tensor, F_true: torch.Tensor, threshold: float) -> float:
    scores = compute_nonconformity_scores(F_pred, F_true)
    return (scores <= threshold).float().mean().item()


# ============================================================
# Epsilon-relaxed projection (closed-form)
# ============================================================

def project_epsilon_relaxed(
    F_pred: torch.Tensor,
    E_target: float,
    epsilon_threshold: float,
) -> tuple:
    """
    Only project snapshots whose relative energy violation exceeds epsilon_threshold.
    Uses closed-form projection: F_proj = F * sqrt(E_target / ||F||^2).

    Args:
        F_pred:            (n, n_atoms, 3)
        E_target:          scalar -- calibration mean energy proxy
        epsilon_threshold: scalar -- 95th percentile of calibration violations

    Returns:
        F_relaxed:      (n, n_atoms, 3)
        projected_mask: (n,) bool
    """
    violations = compute_energy_violation(F_pred, E_target)
    projected_mask = violations > epsilon_threshold

    n_projected = projected_mask.sum().item()
    print(f"  Epsilon threshold: {epsilon_threshold:.4f}")
    print(f"  Snapshots requiring projection: {n_projected} / {len(F_pred)} "
          f"({100 * n_projected / len(F_pred):.1f}%)")

    F_relaxed = F_pred.clone()
    if n_projected > 0:
        F_sub = F_pred[projected_mask]
        F_relaxed[projected_mask] = project_forces_energy_constraint(F_sub, E_target)

    return F_relaxed, projected_mask


# ============================================================
# Main ablation per molecule per alpha
# ============================================================

def run_ablation(proj_result: dict, eps_result: dict) -> dict:
    molecule = proj_result["molecule"]
    alpha    = proj_result["alpha"]

    print(f"\n{'='*60}")
    print(f"Ablation: {molecule}, alpha={alpha}")
    print(f"{'='*60}")

    F_raw       = proj_result["test_pred_raw"]
    F_projected = proj_result["test_pred_projected"]
    F_true      = proj_result["ground_truth"]
    threshold   = proj_result["threshold"]
    epsilon_95  = eps_result["epsilon_95"]
    E_target    = proj_result.get("E_target_scalar",
                                   compute_energy_from_forces(F_raw).mean().item())

    # Step 1: Unprojected
    print(f"\nStep 1: Unprojected coverage")
    cov_unprojected = measure_coverage(F_raw, F_true, threshold)
    print(f"  Coverage (unprojected): {cov_unprojected:.4f}  threshold={threshold:.4f}")

    # Step 2: Exact projection (saved)
    print(f"\nStep 2: Exact projection (saved + verified)")
    cov_exact_saved    = proj_result["coverage_after"]
    cov_exact_verified = measure_coverage(F_projected, F_true, threshold)
    width_exact        = proj_result["mean_width_after"]
    print(f"  Coverage (saved):    {cov_exact_saved:.4f}")
    print(f"  Coverage (verified): {cov_exact_verified:.4f}")
    print(f"  Mean width: {width_exact:.4f}")

    # Step 3: Epsilon-relaxed projection
    print(f"\nStep 3: Epsilon-relaxed projection (epsilon_95={epsilon_95:.4f})")
    F_relaxed, proj_mask = project_epsilon_relaxed(F_raw, E_target, epsilon_95)
    cov_relaxed = measure_coverage(F_relaxed, F_true, threshold)
    print(f"  Coverage (relaxed proj): {cov_relaxed:.4f}")

    # Step 4: Bound tightness
    print(f"\nStep 4: Bound tightness")
    predicted_loss         = eps_result["test_exceedance_rate"]
    empirical_loss_exact   = abs(cov_unprojected - cov_exact_verified)
    empirical_loss_relaxed = abs(cov_unprojected - cov_relaxed)
    bound_gap_exact        = abs(predicted_loss - empirical_loss_exact)
    bound_gap_relaxed      = abs(predicted_loss - empirical_loss_relaxed)

    print(f"  Predicted loss (bound):        {predicted_loss:.4f} ({predicted_loss:.1%})")
    print(f"  Empirical loss (exact proj):   {empirical_loss_exact:.4f} ({empirical_loss_exact:.1%})")
    print(f"  Empirical loss (relaxed proj): {empirical_loss_relaxed:.4f} ({empirical_loss_relaxed:.1%})")
    print(f"  Bound gap (exact):   {bound_gap_exact:.4f}  ({'TIGHT' if bound_gap_exact <= 0.03 else 'LOOSE'})")
    print(f"  Bound gap (relaxed): {bound_gap_relaxed:.4f}  ({'TIGHT' if bound_gap_relaxed <= 0.03 else 'LOOSE'})")

    return {
        "molecule":                   molecule,
        "alpha":                      alpha,
        "threshold":                  threshold,
        "epsilon_95":                 epsilon_95,
        "E_target":                   E_target,
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

def print_summary(all_results: list):
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    header = (f"{'molecule':<16} {'alpha':>6}  {'Unproj':>8}  {'Exact':>8}  "
              f"{'Relaxed':>8}  {'PredLoss':>9}  {'Gap(ex)':>8}  {'Gap(rx)':>8}  "
              f"{'Tight(ex)':>10}  {'Tight(rx)':>10}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['molecule']:<16} {r['alpha']:>6.2f}  "
            f"{r['coverage_unprojected']:>8.4f}  "
            f"{r['coverage_exact']:>8.4f}  "
            f"{r['coverage_relaxed']:>8.4f}  "
            f"{r['predicted_coverage_loss']:>9.4f}  "
            f"{r['bound_gap_exact']:>8.4f}  "
            f"{r['bound_gap_relaxed']:>8.4f}  "
            f"{'Y' if r['bound_tight_exact'] else 'N':>10}  "
            f"{'Y' if r['bound_tight_relaxed'] else 'N':>10}"
        )


# ============================================================
# Figure: bound tightness bar chart (one per molecule)
# ============================================================

def plot_bound_tightness(results_by_molecule: dict, out_path: str):
    molecules = list(results_by_molecule.keys())
    n = len(molecules)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, mol in zip(axes, molecules):
        results = results_by_molecule[mol]
        labels      = [f"a={r['alpha']}" for r in results]
        predicted   = [r["predicted_coverage_loss"]  for r in results]
        exact_emp   = [r["coverage_loss_exact"]       for r in results]
        relaxed_emp = [r["coverage_loss_relaxed"]     for r in results]

        x     = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, predicted,   width, label="Predicted",     color="#4C72B0")
        ax.bar(x,         exact_emp,   width, label="Exact proj",    color="#DD8452")
        ax.bar(x + width, relaxed_emp, width, label="Relaxed proj",  color="#55A868")
        ax.axhline(0.03, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(mol, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Coverage loss")
        if max(exact_emp) > 0:
            ax.set_ylim(0, max(exact_emp) * 1.4)
        ax.legend(fontsize=8)

    plt.suptitle("Bound tightness: predicted vs empirical coverage loss\nMolecular surrogate (MD17)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nBound tightness figure saved to {out_path}")


# ============================================================
# Reproducibility check
# ============================================================

def check_reproducibility(all_results: list):
    print(f"\n{'='*60}")
    print("REPRODUCIBILITY CHECK")
    print(f"{'='*60}")
    all_pass = True
    for r in all_results:
        nominal = 1.0 - r["alpha"]
        cov     = r["coverage_unprojected"]
        gap     = abs(cov - nominal)
        status  = "PASS" if gap <= 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {r['molecule']} alpha={r['alpha']}: coverage={cov:.4f}, "
              f"nominal={nominal:.2f}, gap={gap:.4f}  [{status}]")
    if all_pass:
        print("  All coverage checks passed (within 2pp of nominal).")
    else:
        print("  WARNING: one or more checks failed.")
    return all_pass


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Week 5: Epsilon relaxed projection ablation")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--molecules", nargs="+", type=str, default=MOLECULES)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading Week 3 results...")
    week3 = torch.load(WEEK3_RESULTS, weights_only=False)
    print(f"  {len(week3['projection_results'])} projection results loaded")

    all_results = []
    results_by_molecule = {}

    for molecule in args.molecules:
        mol_results = []
        for alpha in args.alphas:
            # Find matching projection and epsilon results
            try:
                proj_result = next(
                    r for r in week3["projection_results"]
                    if r["molecule"] == molecule and abs(r["alpha"] - alpha) < 1e-6
                )
                eps_result = next(
                    r for r in week3["epsilon_results"]
                    if r["molecule"] == molecule and abs(r["alpha"] - alpha) < 1e-6
                )
            except StopIteration:
                print(f"  Skipping {molecule} alpha={alpha} -- not found in Week 3 results")
                continue

            result = run_ablation(proj_result, eps_result)
            all_results.append(result)
            mol_results.append(result)

        if mol_results:
            results_by_molecule[molecule] = mol_results
            out_path = f"{RESULTS_DIR}/{molecule}_week5_ablation.pt"
            torch.save({"ablation_results": mol_results, "molecule": molecule}, out_path)
            print(f"\n  Ablation results saved to {out_path}")

    print_summary(all_results)
    check_reproducibility(all_results)

    out_figure = f"{FIGURES_DIR}/molecular_week5_bound_tightness.png"
    plot_bound_tightness(results_by_molecule, out_figure)

    print("\nFiles to commit:")
    print("  molecular_week5.py")


if __name__ == "__main__":
    main()
