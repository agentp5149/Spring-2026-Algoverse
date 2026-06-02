"""
Week 4: Adaptive Shift Detection & OOD Testing (Molecular Surrogate)
======================================================================

Vijay's tasks:
    1. Construct OOD test sets from MD22 DHA (larger, chemically distinct
       molecules underrepresented relative to MD17 aspirin training data).
       MD22 DHA has 56 atoms vs aspirin's 21 and covers a fatty acid chemical
       family with different bonding environments -- a strong OOD source.
    2. Apply shift detection and interval widening to the molecular surrogate
       on OOD inputs. Measure coverage before and after widening.
    3. Run shift detection threshold sweep (90th to 99th percentile) and
       measure coverage vs. interval width tradeoff. Plot the Pareto curve.

Shift score design:
    We compute shift scores in input space rather than output space, since
    no trained surrogate checkpoint is available. For each test configuration
    we compute the L2 distance between its flattened position vector and the
    mean of the aspirin calibration positions. This is a principled input-space
    distance: DHA configurations span a much larger spatial range than aspirin
    (different chain length, bonding environment, coordinate scale), so their
    distances from the aspirin calibration cloud will fall in the extreme tail
    of the calibration distance distribution and get correctly flagged.

    The calibration distance distribution is built from the aspirin cal split
    positions using the same 70/15/15 split as Week 2/3.

Outputs (all committed to git):
    results/molecular/aspirin_ood_shift_results.pt
    results/molecular/aspirin_shift_threshold_sweep.pt
    results/figures/molecular_shift_pareto.png

Usage:
    python molecular_week4.py
    python molecular_week4.py --percentile 95
    python molecular_week4.py --max-ood-samples 500
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

WEEK3_RESULTS  = "results/molecular/aspirin_conformal_projected_indist.pt"
MD17_ASPIRIN   = "data/md17/md17_aspirin.npz"
MD22_DHA       = "data/md22/md22_DHA.npz"
RESULTS_DIR    = "results/molecular"
FIGURES_DIR    = "results/figures"
OUT_SHIFT      = "results/molecular/aspirin_ood_shift_results.pt"
OUT_SWEEP      = "results/molecular/aspirin_shift_threshold_sweep.pt"
OUT_PARETO     = "results/figures/molecular_shift_pareto.png"


# ============================================================
# Input-space shift score
# ============================================================

def compute_input_distance_scores(query_positions, ref_mean, ref_std):
    """
    Compute a normalised L2 distance from each query configuration to
    the reference distribution (aspirin calibration set).

    We standardise the query positions using the reference mean and std
    before computing the norm. This makes the score scale-invariant and
    directly comparable across in-distribution and OOD inputs.

    Args:
        query_positions: (n, n_atoms, 3) tensor
        ref_mean: (n_atoms, 3) tensor -- mean of calibration positions
        ref_std:  scalar -- std of calibration positions

    Returns:
        scores: (n,) tensor of distances
    """
    flat_query = query_positions.reshape(query_positions.shape[0], -1)
    flat_mean  = ref_mean.reshape(1, -1)
    normalised = (flat_query - flat_mean) / (ref_std + 1e-8)
    return normalised.norm(dim=1) / np.sqrt(flat_query.shape[1])


def build_calibration_scores(aspirin_path, seed=42):
    """
    Reproduce the 70/15/15 split from Week 2/3 and compute input-space
    distance scores for the calibration split of aspirin positions.

    Returns:
        cal_scores:   (n_cal,) tensor
        ref_mean:     (21, 3) tensor  -- calibration mean positions
        ref_std:      scalar          -- calibration position std
        cal_positions (n_cal, 21, 3) tensor
    """
    data = np.load(aspirin_path)
    R    = torch.tensor(data["R"], dtype=torch.float32)   # (211762, 21, 3)

    N   = len(R)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)

    n_train = int(0.70 * N)
    n_cal   = int(0.15 * N)

    cal_idx      = idx[n_train : n_train + n_cal]
    cal_positions = R[cal_idx]                             # (n_cal, 21, 3)

    ref_mean = cal_positions.mean(dim=0)                  # (21, 3)
    ref_std  = cal_positions.std().item()

    cal_scores = compute_input_distance_scores(cal_positions, ref_mean, ref_std)

    print(f"Aspirin calibration split:")
    print(f"  n_cal = {len(cal_idx)}")
    print(f"  Input-space score range: [{cal_scores.min():.3f}, {cal_scores.max():.3f}]")
    print(f"  Score mean: {cal_scores.mean():.3f}, std: {cal_scores.std():.3f}")

    return cal_scores, ref_mean, ref_std, cal_positions


# ============================================================
# Task 1: Load OOD configurations from MD22 DHA
# ============================================================

def load_md22_positions(npz_path, max_samples=2000, n_atoms_target=21, seed=42):
    """
    Load atomic positions from MD22 DHA as the OOD test set.

    MD22 DHA (docosahexaenoic acid) is a 56-atom fatty acid -- chemically
    distinct from MD17's small organics in chain length, bonding pattern,
    and conformational space. Truncating to 21 atoms still produces OOD
    inputs because the coordinate scale and local bonding environment
    differ fundamentally from aspirin.
    """
    print(f"Loading MD22 DHA from {npz_path}")
    data = np.load(npz_path)
    R    = torch.tensor(data["R"], dtype=torch.float32)   # (69753, 56, 3)

    n_total = R.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)[:max_samples]
    R_ood = R[idx, :n_atoms_target, :]                    # (n_samples, 21, 3)

    print(f"  Total MD22 DHA frames: {n_total}")
    print(f"  Sampled: {len(R_ood)} configurations")
    print(f"  Truncated to first {n_atoms_target} atoms per config")
    print(f"  OOD set shape: {R_ood.shape}  (n_configs, n_atoms, 3)")

    return R_ood


# ============================================================
# Task 2: Shift detection + interval widening + coverage
# ============================================================

def run_shift_detection(week3_results, ood_positions, cal_scores_input,
                        ref_mean, ref_std, percentile=95):
    """
    Apply ShiftDetector to OOD inputs using input-space distance scores.

    The ShiftDetector is initialised with the aspirin calibration input-space
    scores (not the conformal nonconformity scores from Week 3). OOD scores
    are computed the same way. This correctly detects when test inputs are
    far from the calibration distribution in input space.

    Coverage numbers reported are from Week 3 (in-distribution baseline).
    The worst-case coverage bound shows how much coverage would degrade
    on these OOD inputs if we used the base conformal threshold without widening.
    """
    threshold       = week3_results["threshold"]
    alpha           = week3_results["alpha"]
    coverage_before = week3_results["coverage_before"]
    coverage_after  = week3_results["coverage_after"]
    target_coverage = week3_results["target_coverage"]

    print(f"\nTask 2: Shift detection at {percentile}th percentile threshold")
    print(f"  Input-space cal scores: n={len(cal_scores_input)}, "
          f"range=[{cal_scores_input.min():.3f}, {cal_scores_input.max():.3f}]")
    print(f"  In-dist conformal threshold: {threshold:.4f}")

    ood_scores = compute_input_distance_scores(ood_positions, ref_mean, ref_std)
    print(f"  OOD input-space scores: n={len(ood_scores)}, "
          f"range=[{ood_scores.min():.3f}, {ood_scores.max():.3f}]")

    detector = ShiftDetector(cal_scores_input, percentile_threshold=percentile)
    shift_flags, shift_magnitudes = detector.detect(ood_scores)

    flag_rate = shift_flags.float().mean().item()
    print(f"  Shift flag rate: {flag_rate:.1%} of OOD inputs flagged")
    print(f"  Shift threshold (cal {percentile}th pct): {detector.shift_threshold:.4f}")

    widened_thresholds = detector.widen_intervals(threshold, shift_magnitudes)
    mean_widened = widened_thresholds.mean().item()
    width_ratio  = mean_widened / threshold
    print(f"  Mean base threshold:        {threshold:.4f}")
    print(f"  Mean widened threshold:     {mean_widened:.4f}")
    print(f"  Width ratio (widened/base): {width_ratio:.2f}x")

    print(f"\n  In-dist coverage (from Week 3):")
    print(f"    Before projection: {coverage_before:.1%}")
    print(f"    After projection:  {coverage_after:.1%}")
    print(f"    Target:            {target_coverage:.1%}")

    shift_scale = detector.shift_threshold if detector.shift_threshold > 0 else 1.0
    wcb = worst_case_coverage_bound(
        ood_scores,
        alpha=alpha,
        shift_threshold=detector.shift_threshold,
        shift_scale=shift_scale,
    )
    mean_wcb = wcb.mean().item()
    min_wcb  = wcb.min().item()
    print(f"\n  Worst-case coverage bound (OOD):")
    print(f"    Mean: {mean_wcb:.1%}")
    print(f"    Min:  {min_wcb:.1%}")

    return {
        "surrogate":                      "molecular_mlp_aspirin",
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
# Task 3: Threshold sweep 90th to 99th percentile
# ============================================================

def run_threshold_sweep(week3_results, ood_positions, cal_scores_input, ref_mean, ref_std):
    """
    Sweep shift detection threshold from 90th to 99th percentile.
    For each threshold, record flag rate, mean widened width, and
    worst-case coverage bound. This produces the Pareto curve.
    """
    threshold = week3_results["threshold"]
    alpha     = week3_results["alpha"]

    ood_scores = compute_input_distance_scores(ood_positions, ref_mean, ref_std)

    percentiles = list(range(90, 100))
    flag_rates  = []
    mean_widths = []
    mean_wcbs   = []

    print(f"\nTask 3: Threshold sweep ({percentiles[0]}th to {percentiles[-1]}th percentile)")
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
            ood_scores,
            alpha=alpha,
            shift_threshold=detector.shift_threshold,
            shift_scale=shift_scale,
        )
        mean_wcb = wcb.mean().item()

        flag_rates.append(flag_rate)
        mean_widths.append(mean_width)
        mean_wcbs.append(mean_wcb)

        print(f"  {pct:>4}  {flag_rate:>9.1%}  {mean_width:>10.4f}  {mean_wcb:>9.1%}")

    return {
        "surrogate":      "molecular_mlp_aspirin",
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
# Plot Pareto curve
# ============================================================

def plot_pareto(sweep_results, out_path):
    widths     = sweep_results["mean_widths"]
    wcbs       = [v * 100 for v in sweep_results["mean_wcbs"]]
    pcts       = sweep_results["percentiles"]
    base_width = sweep_results["base_width"]
    alpha      = sweep_results["alpha"]
    target     = (1 - alpha) * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(widths, wcbs, c=pcts, cmap="viridis", s=80, zorder=3)
    ax.plot(widths, wcbs, color="gray", linewidth=1, zorder=2)

    for i, pct in enumerate(pcts):
        ax.annotate(
            f"{pct}th",
            (widths[i], wcbs[i]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
        )

    ax.axvline(base_width, color="steelblue", linestyle="--",
               linewidth=1.2, label=f"In-dist width ({base_width:.2f})")
    ax.axhline(target, color="tomato", linestyle="--",
               linewidth=1.2, label=f"Target coverage ({target:.0f}%)")

    ax.set_xlabel("Mean Interval Width (widened)", fontsize=11)
    ax.set_ylabel("Worst-Case Coverage Bound (%)", fontsize=11)
    ax.set_title(
        "Pareto Curve: Coverage Robustness vs Interval Width\n"
        "Molecular Surrogate (MD22 DHA OOD)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="Shift threshold percentile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Pareto curve saved to {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Week 4: Molecular OOD shift detection")
    parser.add_argument("--percentile", type=int, default=95,
                        help="Shift detection threshold percentile for task 2 (default 95)")
    parser.add_argument("--max-ood-samples", type=int, default=2000,
                        help="Max MD22 configs to load (default 2000)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Loading Week 3 results from {WEEK3_RESULTS}")
    week3 = torch.load(WEEK3_RESULTS, weights_only=False)
    print(f"  threshold={week3['threshold']:.4f}, alpha={week3['alpha']}, "
          f"n_cal={week3['n_cal']}")

    # Build input-space calibration scores from aspirin cal split
    print(f"\nBuilding input-space calibration scores from aspirin")
    cal_scores_input, ref_mean, ref_std, _ = build_calibration_scores(MD17_ASPIRIN)

    # Task 1
    print(f"\nTask 1: Constructing OOD test set from MD22 DHA")
    ood_positions = load_md22_positions(
        MD22_DHA,
        max_samples=args.max_ood_samples,
        n_atoms_target=week3["test_pred_raw"].shape[1],
    )

    # Task 2
    shift_results = run_shift_detection(
        week3, ood_positions, cal_scores_input, ref_mean, ref_std,
        percentile=args.percentile,
    )
    torch.save(shift_results, OUT_SHIFT)
    print(f"\n  Shift results saved to {OUT_SHIFT}")

    # Task 3
    sweep_results = run_threshold_sweep(
        week3, ood_positions, cal_scores_input, ref_mean, ref_std,
    )
    torch.save(sweep_results, OUT_SWEEP)
    print(f"  Sweep results saved to {OUT_SWEEP}")

    plot_pareto(sweep_results, OUT_PARETO)

    print("\nWeek 4 complete. Files to commit:")
    print(f"  molecular_week4.py")
    print(f"  {OUT_SHIFT}")
    print(f"  {OUT_SWEEP}")
    print(f"  {OUT_PARETO}")


if __name__ == "__main__":
    main()
