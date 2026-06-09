"""
Week 4: Adaptive Shift Detection & OOD Testing (Molecular Surrogate)
======================================================================

Vijay's tasks:
    1. Construct OOD test sets from MD22 DHA for each MD17 molecule.
    2. Apply shift detection and interval widening. Measure coverage
       before and after widening.
    3. Run shift detection threshold sweep (90th to 99th percentile)
       and plot the Pareto curve.

Shift score design:
    Input-space L2 distance from each OOD configuration to the mean of
    the per-molecule calibration positions, normalised by calibration std.
    MD22 DHA (56-atom fatty acid) is used as the OOD source for all
    molecules -- it is chemically distinct from all MD17 small organics.

Outputs:
    results/molecular/<molecule>_ood_shift_results.pt
    results/molecular/<molecule>_shift_threshold_sweep.pt
    results/figures/molecular_shift_pareto_<molecule>.png

Usage:
    python molecular_week4.py
    python molecular_week4.py --molecule aspirin
    python molecular_week4.py --all
    python molecular_week4.py --percentile 95 --max-ood-samples 2000
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from conformal import ShiftDetector, worst_case_coverage_bound


# ============================================================
# Paths
# ============================================================

MOLECULE_FILES = {
    "aspirin":       "data/md17/md17_aspirin.npz",
    "ethanol":       "data/md17/md17_ethanol.npz",
    "uracil":        "data/md17/md17_uracil.npz",
    "malonaldehyde": "data/md17/md17_malonaldehyde.npz",
}

MD22_DHA    = "data/md22/md22_DHA.npz"
RESULTS_DIR = "results/molecular"
FIGURES_DIR = "results/figures"


# ============================================================
# Input-space shift score
# ============================================================

def compute_input_distance_scores(query_positions, ref_mean, ref_std):
    """
    Normalised L2 distance from each query config to the reference
    calibration distribution.

    Args:
        query_positions: (n, n_atoms, 3)
        ref_mean:        (n_atoms, 3) mean of calibration positions
        ref_std:         scalar std of calibration positions

    Returns:
        scores: (n,)
    """
    flat_query = query_positions.reshape(query_positions.shape[0], -1)
    flat_mean  = ref_mean.reshape(1, -1)
    normalised = (flat_query - flat_mean) / (ref_std + 1e-8)
    return normalised.norm(dim=1) / np.sqrt(flat_query.shape[1])


def build_calibration_scores(molecule_path, n_atoms, seed=42):
    """
    Reproduce the 70/15/15 split and compute input-space distance scores
    for the calibration split of a given molecule's positions.

    Returns:
        cal_scores:    (n_cal,) tensor
        ref_mean:      (n_atoms, 3) tensor
        ref_std:       scalar
    """
    data = np.load(molecule_path)
    R    = torch.tensor(data["R"], dtype=torch.float32)

    N   = len(R)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)

    n_train = int(0.70 * N)
    n_cal   = int(0.15 * N)

    cal_idx       = idx[n_train : n_train + n_cal]
    cal_positions = R[cal_idx, :n_atoms, :]

    ref_mean = cal_positions.mean(dim=0)
    ref_std  = cal_positions.std().item()

    cal_scores = compute_input_distance_scores(cal_positions, ref_mean, ref_std)

    print(f"  Calibration split: n_cal={len(cal_idx)}")
    print(f"  Input-space score range: [{cal_scores.min():.3f}, {cal_scores.max():.3f}]")

    return cal_scores, ref_mean, ref_std


# ============================================================
# Load OOD configurations from MD22 DHA
# ============================================================

def load_md22_positions(npz_path, max_samples=2000, n_atoms_target=21, seed=42):
    """
    Load atomic positions from MD22 DHA as the OOD test set.
    Truncates to first n_atoms_target atoms to match surrogate input size.
    """
    print(f"Loading MD22 DHA from {npz_path}")
    data = np.load(npz_path)
    R    = torch.tensor(data["R"], dtype=torch.float32)

    n_total = R.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)[:max_samples]
    R_ood = R[idx, :n_atoms_target, :]

    print(f"  Total MD22 DHA frames: {n_total}")
    print(f"  Sampled: {len(R_ood)}, truncated to {n_atoms_target} atoms")

    return R_ood


# ============================================================
# Shift detection + coverage
# ============================================================

def run_shift_detection(week3_results, ood_positions, cal_scores_input,
                        ref_mean, ref_std, molecule, percentile=95):
    threshold       = week3_results["threshold"]
    alpha           = week3_results["alpha"]
    coverage_before = week3_results["coverage_before"]
    coverage_after  = week3_results["coverage_after"]
    target_coverage = week3_results["target_coverage"]

    print(f"\n  Shift detection at {percentile}th percentile threshold")

    ood_scores = compute_input_distance_scores(ood_positions, ref_mean, ref_std)
    print(f"  OOD score range: [{ood_scores.min():.3f}, {ood_scores.max():.3f}]")

    detector = ShiftDetector(cal_scores_input, percentile_threshold=percentile)
    shift_flags, shift_magnitudes = detector.detect(ood_scores)

    flag_rate = shift_flags.float().mean().item()
    print(f"  Flag rate: {flag_rate:.1%}")
    print(f"  Shift threshold (cal {percentile}th pct): {detector.shift_threshold:.4f}")

    widened_thresholds = detector.widen_intervals(threshold, shift_magnitudes)
    mean_widened = widened_thresholds.mean().item()
    width_ratio  = mean_widened / threshold
    print(f"  Base threshold: {threshold:.4f}  ->  Mean widened: {mean_widened:.4f}  ({width_ratio:.2f}x)")

    shift_scale = detector.shift_threshold if detector.shift_threshold > 0 else 1.0
    wcb = worst_case_coverage_bound(
        ood_scores, alpha=alpha,
        shift_threshold=detector.shift_threshold,
        shift_scale=shift_scale,
    )
    mean_wcb = wcb.mean().item()
    min_wcb  = wcb.min().item()
    print(f"  Worst-case coverage bound: mean={mean_wcb:.1%}, min={min_wcb:.1%}")
    print(f"  In-dist coverage before/after projection: {coverage_before:.1%} / {coverage_after:.1%}")

    return {
        "surrogate":                      f"molecular_mlp_{molecule}",
        "molecule":                       molecule,
        "method":                         "conformal_shift_detection_input_space",
        "ood_source":                     "MD22_DHA",
        "percentile_used":                percentile,
        "alpha":                          alpha,
        "n_cal":                          len(cal_scores_input),
        "n_ood":                          len(ood_scores),
        "shift_threshold":                detector.shift_threshold,
        "flag_rate":                      flag_rate,
        "ood_scores":                     ood_scores,
        "cal_scores_input":               cal_scores_input,
        "shift_flags":                    shift_flags,
        "shift_magnitudes":               shift_magnitudes,
        "widened_thresholds":             widened_thresholds,
        "mean_widened_threshold":         mean_widened,
        "base_threshold":                 threshold,
        "width_ratio":                    width_ratio,
        "coverage_indist_before":         coverage_before,
        "coverage_indist_after":          coverage_after,
        "target_coverage":                target_coverage,
        "worst_case_coverage_bound_mean": mean_wcb,
        "worst_case_coverage_bound_min":  min_wcb,
        "wcb_per_sample":                 wcb,
    }


# ============================================================
# Threshold sweep
# ============================================================

def run_threshold_sweep(week3_results, ood_positions, cal_scores_input,
                        ref_mean, ref_std, molecule):
    threshold = week3_results["threshold"]
    alpha     = week3_results["alpha"]

    ood_scores = compute_input_distance_scores(ood_positions, ref_mean, ref_std)

    percentiles = list(range(90, 100))
    flag_rates, mean_widths, mean_wcbs = [], [], []

    print(f"\n  Threshold sweep (90th-99th percentile)")
    print(f"  {'Pct':>4}  {'FlagRate':>9}  {'MeanWidth':>10}  {'MeanWCB':>9}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*9}")

    for pct in percentiles:
        detector = ShiftDetector(cal_scores_input, percentile_threshold=pct)
        shift_flags, shift_magnitudes = detector.detect(ood_scores)

        flag_rate  = shift_flags.float().mean().item()
        widened    = detector.widen_intervals(threshold, shift_magnitudes)
        mean_width = (2 * widened).mean().item()

        shift_scale = detector.shift_threshold if detector.shift_threshold > 0 else 1.0
        wcb = worst_case_coverage_bound(
            ood_scores, alpha=alpha,
            shift_threshold=detector.shift_threshold,
            shift_scale=shift_scale,
        )
        mean_wcb = wcb.mean().item()

        flag_rates.append(flag_rate)
        mean_widths.append(mean_width)
        mean_wcbs.append(mean_wcb)

        print(f"  {pct:>4}  {flag_rate:>9.1%}  {mean_width:>10.4f}  {mean_wcb:>9.1%}")

    return {
        "surrogate":      f"molecular_mlp_{molecule}",
        "molecule":       molecule,
        "ood_source":     "MD22_DHA",
        "alpha":          alpha,
        "percentiles":    percentiles,
        "flag_rates":     flag_rates,
        "mean_widths":    mean_widths,
        "mean_wcbs":      mean_wcbs,
        "base_threshold": threshold,
        "base_width":     2 * threshold,
    }


# ============================================================
# Pareto plot
# ============================================================

def plot_pareto(sweep_results, out_path):
    widths     = sweep_results["mean_widths"]
    wcbs       = [v * 100 for v in sweep_results["mean_wcbs"]]
    pcts       = sweep_results["percentiles"]
    base_width = sweep_results["base_width"]
    alpha      = sweep_results["alpha"]
    molecule   = sweep_results["molecule"]
    target     = (1 - alpha) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(widths, wcbs, c=pcts, cmap="viridis", s=80, zorder=3)
    ax.plot(widths, wcbs, color="gray", linewidth=1, zorder=2)

    for i, pct in enumerate(pcts):
        ax.annotate(f"{pct}th", (widths[i], wcbs[i]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    ax.axvline(base_width, color="steelblue", linestyle="--",
               linewidth=1.2, label=f"In-dist width ({base_width:.2f})")
    ax.axhline(target, color="tomato", linestyle="--",
               linewidth=1.2, label=f"Target coverage ({target:.0f}%)")

    ax.set_xlabel("Mean Interval Width (widened)", fontsize=11)
    ax.set_ylabel("Worst-Case Coverage Bound (%)", fontsize=11)
    ax.set_title(
        f"Pareto Curve: Coverage Robustness vs Interval Width\n"
        f"Molecular Surrogate ({molecule}, MD22 DHA OOD)", fontsize=11)
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="Shift threshold percentile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Pareto curve saved to {out_path}")


# ============================================================
# Per-molecule pipeline
# ============================================================

def run_molecule_week4(molecule, percentile=95, max_ood_samples=2000):
    print(f"\n{'='*65}")
    print(f"Week 4 Pipeline: {molecule.upper()}")
    print(f"{'='*65}")

    proj_path = f"results/molecular/{molecule}_conformal_projected_indist.pt"
    if not os.path.exists(proj_path):
        print(f"  WARNING: {proj_path} not found, skipping {molecule}")
        return

    week3 = torch.load(proj_path, weights_only=False)
    n_atoms = week3["test_pred_raw"].shape[1]
    print(f"  Loaded Week 3 results: threshold={week3['threshold']:.4f}, "
          f"alpha={week3['alpha']}, n_atoms={n_atoms}")

    mol_path = MOLECULE_FILES[molecule]
    print(f"\nBuilding calibration scores for {molecule}")
    cal_scores_input, ref_mean, ref_std = build_calibration_scores(mol_path, n_atoms)

    print(f"\nTask 1: Loading OOD set from MD22 DHA")
    ood_positions = load_md22_positions(MD22_DHA, max_samples=max_ood_samples,
                                        n_atoms_target=n_atoms)

    shift_results = run_shift_detection(
        week3, ood_positions, cal_scores_input, ref_mean, ref_std,
        molecule, percentile=percentile,
    )
    out_shift = f"{RESULTS_DIR}/{molecule}_ood_shift_results.pt"
    torch.save(shift_results, out_shift)
    print(f"  Shift results saved to {out_shift}")

    sweep_results = run_threshold_sweep(
        week3, ood_positions, cal_scores_input, ref_mean, ref_std, molecule,
    )
    out_sweep = f"{RESULTS_DIR}/{molecule}_shift_threshold_sweep.pt"
    torch.save(sweep_results, out_sweep)
    print(f"  Sweep results saved to {out_sweep}")

    out_pareto = f"{FIGURES_DIR}/molecular_shift_pareto_{molecule}.png"
    plot_pareto(sweep_results, out_pareto)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Week 4: Molecular OOD shift detection")
    parser.add_argument("--molecule", type=str, default="aspirin",
                        choices=list(MOLECULE_FILES.keys()))
    parser.add_argument("--all", action="store_true",
                        help="Run on all four MD17 molecules")
    parser.add_argument("--percentile", type=int, default=95)
    parser.add_argument("--max-ood-samples", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    molecules = list(MOLECULE_FILES.keys()) if args.all else [args.molecule]

    for mol in molecules:
        run_molecule_week4(mol, percentile=args.percentile,
                           max_ood_samples=args.max_ood_samples)

    print("\nWeek 4 complete.")
    print("Files to commit: molecular_week4.py")


if __name__ == "__main__":
    main()
