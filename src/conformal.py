"""
Conformal Prediction Framework for Scientific ML Surrogates
=============================================================

Implements the conformal prediction pipeline from Proposal 2:
    1. Basic split conformal prediction (scalar and structured)
    2. Nonconformity scores for different output types
    3. Physics-constrained projection (future)
    4. Adaptive shift detection (future)

This module handles Step 2 of the methods. Steps 3 and 4 will be
added as separate modules once the basics are validated.

Usage:
    from conformal import SplitConformal, SupNormScore, TrajectoryNormScore

    # Set up conformal wrapper
    score_fn = SupNormScore()  # for spatial fields
    cp = SplitConformal(score_fn, alpha=0.05)

    # Calibrate on held-out simulator runs
    cp.calibrate(cal_predictions, cal_ground_truth)

    # Get prediction intervals for new inputs
    intervals = cp.predict(new_predictions)
    print(f"Coverage threshold: {cp.threshold}")
"""

import argparse
import os

import numpy as np
import torch
from nonconformity_scores import ScalarAbsoluteErrorScore


# ============================================================
# Nonconformity score functions
# ============================================================

class ScalarAbsoluteError(ScalarAbsoluteErrorScore):
    """Absolute error for scalar predictions. The classic baseline."""


class SupNormScore:
    """
    Supremum norm for spatial field predictions.
    Returns the maximum absolute error across all grid points.

    For a weather field of shape (n_samples, channels, lat, lon),
    this gives the single worst error anywhere on the map.
    This is conservative but gives uniform coverage over the field.
    """

    def __call__(self, prediction, ground_truth):
        """
        Args:
            prediction: (n_samples, ...) any spatial shape
            ground_truth: same shape

        Returns:
            scores: (n_samples,) max absolute error per sample
        """
        residuals = (prediction - ground_truth).abs()
        # Flatten everything except batch dimension
        flat = residuals.view(residuals.shape[0], -1)
        return flat.max(dim=1).values


class TrajectoryNormScore:
    """
    L2 norm of trajectory residuals for time-series predictions.
    Captures cumulative deviation over the trajectory.

    For a PK curve of shape (n_samples, n_timepoints), this gives
    the integrated trajectory distance.
    """

    def __init__(self, normalize_by_length=True):
        self.normalize = normalize_by_length

    def __call__(self, prediction, ground_truth):
        """
        Args:
            prediction: (n_samples, n_timepoints) or (n_samples, n_timepoints, state_dim)
            ground_truth: same shape

        Returns:
            scores: (n_samples,) trajectory distance per sample
        """
        residuals = prediction - ground_truth
        flat = residuals.view(residuals.shape[0], -1)
        norms = flat.norm(dim=1)
        if self.normalize:
            n_points = flat.shape[1]
            norms = norms / np.sqrt(n_points)
        return norms


class WeightedFunctionalScore:
    """
    Weighted functional norm for PK curves.
    Emphasizes clinically relevant time windows (absorption, elimination).

    The weight function assigns higher importance to:
    - Early timepoints (absorption phase, 0-2h)
    - Late timepoints (elimination phase, 12-24h)
    """

    def __init__(self, times, absorption_end=2.0, elimination_start=12.0):
        weights = torch.ones_like(times)
        weights[times <= absorption_end] = 2.0
        weights[times >= elimination_start] = 1.5
        self.weights = weights / weights.sum() * len(weights)

    def __call__(self, prediction, ground_truth):
        residuals = (prediction - ground_truth).abs()
        if residuals.dim() == 2:
            weighted = residuals * self.weights.unsqueeze(0)
        else:
            weighted = residuals * self.weights.view(1, -1, *([1] * (residuals.dim() - 2)))
        flat = weighted.view(weighted.shape[0], -1)
        return flat.norm(dim=1) / np.sqrt(flat.shape[1])


# ============================================================
# Split conformal prediction
# ============================================================

class SplitConformal:
    """
    Split conformal prediction for any output type.

    Given a calibration set of (prediction, ground_truth) pairs and
    a nonconformity score function, computes a threshold such that
    new predictions are covered with probability >= 1 - alpha.

    The coverage guarantee is distribution-free and finite-sample valid
    under the assumption that calibration and test data are exchangeable.
    """

    def __init__(self, score_fn, alpha=0.05):
        """
        Args:
            score_fn: callable that takes (prediction, ground_truth) and
                      returns (n_samples,) nonconformity scores
            alpha: miscoverage level. 0.05 means 95% coverage target.
        """
        self.score_fn = score_fn
        self.alpha = alpha
        self.cal_scores = None
        self.threshold = None

    def calibrate(self, cal_predictions, cal_ground_truth):
        """
        Compute conformal threshold from calibration data.

        Args:
            cal_predictions: model predictions on calibration inputs
            cal_ground_truth: true simulator outputs for calibration inputs
        """
        self.cal_scores = self.score_fn(cal_predictions, cal_ground_truth)
        n = len(self.cal_scores)

        # Conformal quantile: ceil((1 - alpha)(n + 1)) / n
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)

        sorted_scores = self.cal_scores.sort().values
        q_index = int(np.ceil(q_level * n)) - 1
        q_index = min(q_index, n - 1)
        self.threshold = sorted_scores[q_index].item()

        print(f"Calibration complete:")
        print(f"  n_cal = {n}")
        print(f"  alpha = {self.alpha}")
        print(f"  threshold = {self.threshold:.6f}")
        print(f"  score range = [{self.cal_scores.min():.6f}, {self.cal_scores.max():.6f}]")

    def predict(self, test_predictions):
        """
        Return conformal prediction sets for new predictions.

        For scalar predictions, this is an interval [pred - q, pred + q].
        For structured predictions, the set is {y : score(pred, y) <= q}.

        Args:
            test_predictions: model predictions on new inputs

        Returns:
            dict with threshold and prediction bounds
        """
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() first")

        return {
            "predictions": test_predictions,
            "threshold": self.threshold,
            "lower": test_predictions - self.threshold,
            "upper": test_predictions + self.threshold,
        }

    def evaluate(self, test_predictions, test_ground_truth):
        """
        Evaluate empirical coverage and interval width on test data.

        Args:
            test_predictions: model predictions on test inputs
            test_ground_truth: true simulator outputs for test inputs

        Returns:
            dict with coverage, mean_width, and per-sample scores
        """
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() first")

        test_scores = self.score_fn(test_predictions, test_ground_truth)
        covered = (test_scores <= self.threshold).float()
        coverage = covered.mean().item()
        mean_width = 2 * self.threshold  # for symmetric intervals

        results = {
            "coverage": coverage,
            "target_coverage": 1 - self.alpha,
            "coverage_gap": coverage - (1 - self.alpha),
            "mean_width": mean_width,
            "threshold": self.threshold,
            "test_scores": test_scores,
            "covered": covered,
            "n_test": len(test_scores),
        }

        print(f"Evaluation results:")
        print(f"  Target coverage: {1 - self.alpha:.1%}")
        print(f"  Empirical coverage: {coverage:.1%}")
        print(f"  Gap: {results['coverage_gap']:+.1%}")
        print(f"  Interval width: {mean_width:.6f}")
        print(f"  n_test = {len(test_scores)}")

        return results


# ============================================================
# Shift detection (Step 4 skeleton)
# ============================================================

class ShiftDetector:
    """
    Detects when test inputs are far from calibration distribution.
    Widens prediction intervals accordingly.

    This is the Step 4 adaptive mechanism. Currently a skeleton
    to be filled in once Steps 2-3 are validated.
    """

    def __init__(self, cal_scores, percentile_threshold=95):
        self.cal_scores = cal_scores
        self.threshold_percentile = percentile_threshold
        sorted_scores = cal_scores.sort().values
        idx = int(len(sorted_scores) * percentile_threshold / 100)
        idx = min(idx, len(sorted_scores) - 1)
        self.shift_threshold = sorted_scores[idx].item()

    def detect(self, test_scores):
        """
        Flag test points that look unlike calibration data.

        Returns:
            shift_flags: (n_test,) boolean tensor
            shift_magnitudes: (n_test,) how far beyond threshold
        """
        shift_flags = test_scores > self.shift_threshold
        shift_magnitudes = (test_scores / self.shift_threshold).clamp(min=1.0)
        return shift_flags, shift_magnitudes

    def widen_intervals(self, base_threshold, shift_magnitudes):
        """
        Compute per-sample widened thresholds.

        For in-distribution points: use base_threshold.
        For shifted points: multiply by shift_magnitude.
        """
        return base_threshold * shift_magnitudes


# ============================================================
# Demo / testing
# ============================================================

def demo_scalar():
    """Quick demo with scalar predictions."""
    print("=" * 50)
    print("Demo: Scalar conformal prediction")
    print("=" * 50)

    torch.manual_seed(42)
    n_cal, n_test = 500, 200

    # Simulate a model that predicts with some noise
    cal_true = torch.randn(n_cal)
    cal_pred = cal_true + 0.3 * torch.randn(n_cal)
    test_true = torch.randn(n_test)
    test_pred = test_true + 0.3 * torch.randn(n_test)

    cp = SplitConformal(ScalarAbsoluteError(), alpha=0.1)
    cp.calibrate(cal_pred, cal_true)
    results = cp.evaluate(test_pred, test_true)
    print()


def demo_spatial():
    """Quick demo with spatial field predictions."""
    print("=" * 50)
    print("Demo: Spatial field conformal prediction")
    print("=" * 50)

    torch.manual_seed(42)
    n_cal, n_test = 300, 100
    h, w = 16, 32

    cal_true = torch.randn(n_cal, 2, h, w)
    cal_pred = cal_true + 0.2 * torch.randn(n_cal, 2, h, w)
    test_true = torch.randn(n_test, 2, h, w)
    test_pred = test_true + 0.2 * torch.randn(n_test, 2, h, w)

    cp = SplitConformal(SupNormScore(), alpha=0.1)
    cp.calibrate(cal_pred, cal_true)
    results = cp.evaluate(test_pred, test_true)
    print()


def demo_trajectory():
    """Quick demo with trajectory predictions."""
    print("=" * 50)
    print("Demo: Trajectory conformal prediction")
    print("=" * 50)

    torch.manual_seed(42)
    n_cal, n_test = 400, 150
    t_steps = 50

    cal_true = torch.randn(n_cal, t_steps)
    cal_pred = cal_true + 0.15 * torch.randn(n_cal, t_steps)
    test_true = torch.randn(n_test, t_steps)
    test_pred = test_true + 0.15 * torch.randn(n_test, t_steps)

    cp = SplitConformal(TrajectoryNormScore(), alpha=0.1)
    cp.calibrate(cal_pred, cal_true)
    results = cp.evaluate(test_pred, test_true)
    print()


def main():
    parser = argparse.ArgumentParser(description="Conformal prediction framework")
    parser.add_argument("--demo", action="store_true", help="Run demos on synthetic data")
    parser.add_argument("--surrogate", type=str, help="Path to trained surrogate model")
    parser.add_argument("--calibration", type=str, help="Path to calibration data")
    parser.add_argument("--test", type=str, help="Path to test data")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage level")
    args = parser.parse_args()

    if args.demo:
        demo_scalar()
        demo_spatial()
        demo_trajectory()
        print("All demos complete.")
    elif args.surrogate and args.calibration and args.test:
        print("Full pipeline not yet implemented.")
        print("Use --demo to verify the conformal framework works.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
