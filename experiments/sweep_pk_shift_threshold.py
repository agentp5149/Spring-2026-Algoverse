"""
Sweep PK OOD shift-detection thresholds.

The sweep varies the calibration shift-score percentile used as the OOD threshold,
then reports the resulting flag rate, widening factor, widened OOD coverage, and
worst-case coverage lower bound.
"""

import argparse
import csv
import json
import os
import sys

import torch

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo, "src"))

from conformal import SplitConformal, WeightedFunctionalScore, worst_case_coverage_bound  # noqa: E402
from neural_ode_pk import PKNeuralODE  # noqa: E402


def predict_trajectories(model, params, times):
    y0 = torch.stack([torch.tensor([100.0, 0.0])] * len(params))
    with torch.no_grad():
        return model(params, y0, times).permute(1, 0, 2)


def standardized_distance(params, center, scale):
    z = (params - center) / scale.clamp_min(1e-6)
    return z.norm(dim=1)


def parse_percentiles(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Sweep PK OOD shift threshold percentiles")
    parser.add_argument("--model", default="models/pk_surrogate.pt")
    parser.add_argument("--id-data", default="data/pk_task1/pk_population.pt")
    parser.add_argument("--ood-data", default="data/ood/pk_ood.pt")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--percentiles",
        default="50,60,70,75,80,85,90,92.5,95,97.5,99",
        help="Comma-separated calibration shift-score percentiles to sweep.",
    )
    parser.add_argument("--json-output", default="results/pk/pk_shift_threshold_sweep.json")
    parser.add_argument("--csv-output", default="results/pk/pk_shift_threshold_sweep.csv")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, weights_only=False)
    id_data = torch.load(args.id_data, weights_only=False)
    ood_data = torch.load(args.ood_data, weights_only=False)

    cal_idx = checkpoint["splits"]["cal"]
    id_params = id_data["params"]
    id_times = id_data["times"][::5]
    id_traj = id_data["trajectories"][:, ::5, :]
    ood_params = ood_data["params"]
    ood_traj = ood_data["trajectories"][:, ::5, :]

    model = PKNeuralODE()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cal_params = id_params[cal_idx]
    cal_true = id_traj[cal_idx]
    cal_pred = predict_trajectories(model, cal_params, id_times)
    ood_pred = predict_trajectories(model, ood_params, id_times)

    score_fn = WeightedFunctionalScore(id_times)
    cp = SplitConformal(score_fn, alpha=args.alpha)
    cp.calibrate(cal_pred, cal_true)

    ood_scores = score_fn(ood_pred, ood_traj)
    base_coverage = (ood_scores <= cp.threshold).float().mean().item()

    center = cal_params.mean(dim=0)
    scale = cal_params.std(dim=0)
    cal_shift_scores = standardized_distance(cal_params, center, scale)
    ood_shift_scores = standardized_distance(ood_params, center, scale)

    rows = []
    for percentile in parse_percentiles(args.percentiles):
        shift_threshold = torch.quantile(cal_shift_scores, percentile / 100.0).item()
        safe_threshold = max(shift_threshold, 1e-8)
        widening_factors = (ood_shift_scores / safe_threshold).clamp(min=1.0)
        widened_thresholds = cp.threshold * widening_factors
        widened_covered = ood_scores <= widened_thresholds
        shift_flags = ood_shift_scores > shift_threshold
        coverage_bounds = worst_case_coverage_bound(
            ood_shift_scores,
            alpha=args.alpha,
            shift_threshold=shift_threshold,
            shift_scale=safe_threshold,
        )

        widened_coverage = widened_covered.float().mean().item()
        rows.append(
            {
                "shift_percentile": percentile,
                "shift_threshold": float(shift_threshold),
                "shift_flag_rate": float(shift_flags.float().mean().item()),
                "base_coverage": float(base_coverage),
                "widened_coverage": float(widened_coverage),
                "widened_coverage_gap": float(widened_coverage - (1 - args.alpha)),
                "mean_widening_factor": float(widening_factors.mean().item()),
                "median_widening_factor": float(widening_factors.median().item()),
                "p90_widening_factor": float(torch.quantile(widening_factors, 0.90).item()),
                "max_widening_factor": float(widening_factors.max().item()),
                "mean_min_expected_coverage": float(coverage_bounds.mean().item()),
                "median_min_expected_coverage": float(coverage_bounds.median().item()),
                "min_expected_coverage": float(coverage_bounds.min().item()),
            }
        )

    results = {
        "method": "pk_shift_threshold_sweep",
        "surrogate": "neural_ode_pk",
        "alpha": args.alpha,
        "target_coverage": 1 - args.alpha,
        "n_cal": int(len(cal_idx)),
        "n_ood": int(len(ood_params)),
        "base_threshold": float(cp.threshold),
        "shift_metric": "standardized_parameter_distance_to_calibration_centroid",
        "cal_shift_score_mean": float(cal_shift_scores.mean().item()),
        "ood_shift_score_mean": float(ood_shift_scores.mean().item()),
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    with open(args.json_output, "w") as f:
        json.dump(results, f, indent=2)

    with open(args.csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("PK shift threshold sweep complete")
    print(f"  target coverage   : {1 - args.alpha:.1%}")
    print(f"  base OOD coverage : {base_coverage:.1%}")
    print(f"  rows              : {len(rows)}")
    print(f"  json              : {args.json_output}")
    print(f"  csv               : {args.csv_output}")
    print()
    print("percentile  flag_rate  widened_cov  mean_width_x  mean_bound")
    for row in rows:
        print(
            f"{row['shift_percentile']:>10.1f}  "
            f"{row['shift_flag_rate']:>9.1%}  "
            f"{row['widened_coverage']:>11.1%}  "
            f"{row['mean_widening_factor']:>12.3f}  "
            f"{row['mean_min_expected_coverage']:>10.1%}"
        )


if __name__ == "__main__":
    main()
