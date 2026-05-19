"""
Log Week 2 molecular results through Vasilisa's eval harness.
Run this from the repo root after molecular_week2.py has completed.

Output:
    results/molecular/aspirin_conformal_indist_90.json
    results/molecular/aspirin_conformal_indist_95.json
    results/molecular/aspirin_mc_dropout_indist_90.json
    results/molecular/aspirin_mc_dropout_indist_95.json

Usage:
    python log_molecular_results.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from baselines.eval_harness import (
    compute_empirical_coverage,
    compute_interval_widths,
    log_results,
    compare_methods,
)

RESULT_FILES = [
    "results/molecular/aspirin_conformal_indist.pt",
    "results/molecular/aspirin_mc_dropout_indist.pt",
]


def process_result(pt_path: str):
    r = torch.load(pt_path, weights_only=False)

    # Reconstruct lower/upper bounds from predictions + threshold
    preds     = r["predictions"]       # (n, n_atoms, 3)
    threshold = r["threshold"]         # float or None (MC dropout)
    alpha     = r["alpha"]
    method    = r["method"]
    molecule  = r["molecule"]

    if method == "conformal":
        lower = preds - threshold
        upper = preds + threshold
        surrogate_name = f"mace_md17_{molecule}"
        method_name    = "conformal_split"
    else:
        # MC dropout: threshold is None, width stored as mean_width/2
        half_width = r["mean_width"] / 2
        lower = preds - half_width
        upper = preds + half_width
        surrogate_name = f"mace_md17_{molecule}"
        method_name    = "mc_dropout"

    targets = r["ground_truth"]   # (n, n_atoms, 3)

    cov = compute_empirical_coverage(lower, upper, targets, reduction="trajectory")
    wid = compute_interval_widths(lower, upper)

    coverage_pct = int((1 - alpha) * 100)
    out_path = pt_path.replace(".pt", f"_{coverage_pct}.json")

    record = log_results(
        method=method_name,
        surrogate=surrogate_name,
        alpha=alpha,
        coverage=cov,
        widths=wid,
        runtime={"train_s": None, "inference_s": None},
        output_path=out_path,
        extra={
            "molecule": molecule,
            "n_cal": r["n_cal"],
            "n_test": r["n_test"],
        },
    )

    print(f"Logged: {out_path}")
    print(f"  Coverage : {record['empirical_coverage']:.1%}  (target {1-alpha:.1%})")
    print(f"  Gap      : {record['coverage_gap']:+.1%}")
    print(f"  Width    : {record['mean_width']:.4f}")
    return out_path


def main():
    json_paths = []
    for pt_path in RESULT_FILES:
        if not os.path.exists(pt_path):
            print(f"Missing: {pt_path} — run molecular_week2.py first")
            continue
        json_paths.append(process_result(pt_path))

    if len(json_paths) >= 2:
        print("\n" + "=" * 60)
        print("Comparison table (paste into Slack for Vasilisa):")
        print("=" * 60)
        print(compare_methods(json_paths))


if __name__ == "__main__":
    main()
