"""
Evaluate PK OOD coverage with input-shift detection and interval widening.

Shift score:
    Standardized Euclidean distance from the in-distribution calibration
    parameter centroid, using calibration-set parameter standard deviations.

Widening:
    q_i = q_base * max(1, shift_score_i / q_shift_95)

where q_base is the conformal threshold calibrated on in-distribution PK
trajectories and q_shift_95 is the 95th percentile of calibration shift scores.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

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


def summarize_by_phenotype(labels, scores, base_covered, widened_covered, thresholds, coverage_bounds):
    grouped = defaultdict(list)
    for i, label in enumerate(labels):
        grouped[label].append(i)

    summary = {}
    for label, idxs in sorted(grouped.items()):
        idx = torch.tensor(idxs)
        summary[label] = {
            "n": len(idxs),
            "base_coverage": float(base_covered[idx].float().mean().item()),
            "widened_coverage": float(widened_covered[idx].float().mean().item()),
            "mean_score": float(scores[idx].mean().item()),
            "mean_widened_threshold": float(thresholds[idx].mean().item()),
            "mean_min_expected_coverage": float(coverage_bounds[idx].mean().item()),
            "min_expected_coverage": float(coverage_bounds[idx].min().item()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="PK OOD shift detection and widening")
    parser.add_argument("--model", default="models/pk_surrogate.pt")
    parser.add_argument("--id-data", default="data/pk_task1/pk_population.pt")
    parser.add_argument("--ood-data", default="data/ood/pk_ood.pt")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--shift-percentile", type=float, default=95.0)
    parser.add_argument("--output", default="results/pk/pk_ood_shift_widening.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, weights_only=False)
    id_data = torch.load(args.id_data, weights_only=False)
    ood_data = torch.load(args.ood_data, weights_only=False)

    splits = checkpoint["splits"]
    cal_idx = splits["cal"]

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
    base_covered = ood_scores <= cp.threshold
    base_coverage = base_covered.float().mean().item()

    center = cal_params.mean(dim=0)
    scale = cal_params.std(dim=0)
    cal_shift_scores = standardized_distance(cal_params, center, scale)
    ood_shift_scores = standardized_distance(ood_params, center, scale)
    shift_threshold = torch.quantile(cal_shift_scores, args.shift_percentile / 100.0).item()

    widening_factors = (ood_shift_scores / max(shift_threshold, 1e-8)).clamp(min=1.0)
    widened_thresholds = cp.threshold * widening_factors
    widened_covered = ood_scores <= widened_thresholds
    widened_coverage = widened_covered.float().mean().item()
    shift_flags = ood_shift_scores > shift_threshold
    coverage_bounds = worst_case_coverage_bound(
        ood_shift_scores,
        alpha=args.alpha,
        shift_threshold=shift_threshold,
        shift_scale=max(shift_threshold, 1e-8),
    )

    labels = ood_data.get("phenotype", ["unknown"] * len(ood_params))
    results = {
        "method": "pk_conformal_shift_widening",
        "surrogate": "neural_ode_pk",
        "split": "ood",
        "alpha": args.alpha,
        "target_coverage": 1 - args.alpha,
        "n_cal": int(len(cal_idx)),
        "n_ood": int(len(ood_params)),
        "base_threshold": float(cp.threshold),
        "base_coverage": float(base_coverage),
        "base_coverage_gap": float(base_coverage - (1 - args.alpha)),
        "widened_coverage": float(widened_coverage),
        "widened_coverage_gap": float(widened_coverage - (1 - args.alpha)),
        "mean_widening_factor": float(widening_factors.mean().item()),
        "median_widening_factor": float(widening_factors.median().item()),
        "p90_widening_factor": float(torch.quantile(widening_factors, 0.90).item()),
        "max_widening_factor": float(widening_factors.max().item()),
        "shift_metric": "standardized_parameter_distance_to_calibration_centroid",
        "shift_percentile": args.shift_percentile,
        "shift_threshold": float(shift_threshold),
        "shift_flag_rate": float(shift_flags.float().mean().item()),
        "cal_shift_score_mean": float(cal_shift_scores.mean().item()),
        "ood_shift_score_mean": float(ood_shift_scores.mean().item()),
        "coverage_bound": {
            "derivation": "If TV(Q_shift, P_cal) <= Delta(s), then coverage_Q >= 1 - alpha - Delta(s).",
            "delta_of_shift": "clip((shift_score - shift_threshold) / shift_threshold, 0, 1)",
            "mean_min_expected_coverage": float(coverage_bounds.mean().item()),
            "median_min_expected_coverage": float(coverage_bounds.median().item()),
            "p10_min_expected_coverage": float(torch.quantile(coverage_bounds, 0.10).item()),
            "min_expected_coverage": float(coverage_bounds.min().item()),
        },
        "phenotype_counts": dict(Counter(labels)),
        "by_phenotype": summarize_by_phenotype(
            labels,
            ood_scores,
            base_covered,
            widened_covered,
            widened_thresholds,
            coverage_bounds,
        ),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("PK OOD shift detection + widening complete")
    print(f"  target coverage       : {1 - args.alpha:.1%}")
    print(f"  base OOD coverage     : {base_coverage:.1%}")
    print(f"  widened OOD coverage  : {widened_coverage:.1%}")
    print(f"  mean min coverage bd. : {coverage_bounds.mean().item():.1%}")
    print(f"  shift flag rate       : {results['shift_flag_rate']:.1%}")
    print(f"  mean widening factor  : {results['mean_widening_factor']:.3f}")
    print(f"  saved                 : {args.output}")


if __name__ == "__main__":
    main()
