"""
Conformalize PK Deep Ensemble
===============================
experiments/conformalize_pk_ensemble.py -- Team PVAV

WHY THIS FILE EXISTS:
    The paper's Table 3 reports a "Conformalized deep ensemble" row
    (88.7% coverage, 67.57 mg mean width) that turned out not to
    correspond to any function anywhere in the committed codebase.
    DeepEnsemblePK (src/baselines/deep_ensemble_pk.py) only ever produces
    raw Gaussian mean +/- z*std intervals -- it is never passed through
    SplitConformal anywhere in this repo. This script fills that gap.

    Whatever numbers this script produces are the REAL, reproducible
    replacement for that row. They are not expected to match 88.7% /
    67.57 mg exactly (there is no way to verify those old numbers, since
    no code produced them) -- update the paper with whatever this script
    actually prints.

METHOD (explicit, so this doesn't end up ambiguous again):
    1. Train the M-member deep ensemble exactly as in deep_ensemble_pk.py.
    2. For each patient, get the ensemble MEAN prediction and the
       ensemble STD (spread across members) on the central compartment.
    3. Define a per-patient scale sigma_i = mean over timepoints of the
       ensemble's own predicted std for that patient's central-compartment
       trajectory. This is "the ensemble's own per-sample scale" referred
       to in the paper -- now unambiguous and printed explicitly below.
    4. Nonconformity score for patient i:
           raw_i         = MAX absolute residual between ensemble mean
                            and true trajectory, over timepoints (sup-norm,
                            same pattern as the weather domain's
                            SupNormScore -- NOT a mean or an L2 norm).
                            Both of those were tried first and failed:
                            a mean lets one bad timepoint hide behind
                            several good ones (collapsed to 0% trajectory
                            coverage); an L2 norm can still have all its
                            "energy" concentrated in a single timepoint
                            while staying numerically small, so bounding
                            the L2 norm does NOT bound any individual
                            timepoint either (collapsed to 0.7%).
                            A MAX is the only score type for which
                            "score <= threshold" mathematically implies
                            EVERY timepoint's residual <= threshold,
                            which is exactly what trajectory-level
                            all-timepoints-covered evaluation requires.
           score_i       = raw_i / sigma_i
       Clinical weighting (absorption/elimination emphasis) is NOT
       applied to this score: combining a max with per-timepoint weights
       would require dividing each timepoint's bound by its own weight
       before checking coverage, adding complexity without changing the
       core fix. If clinical weighting matters for the final paper, it
       can be added back explicitly once this baseline version is
       confirmed to produce valid coverage.
       Dividing by the ensemble's own uncertainty estimate means patients
       the ensemble was already unsure about get more slack before being
       flagged nonconforming -- this is standard "normalized" / locally-
       weighted conformal prediction, not something invented ad hoc for
       this script.
    5. Calibrate a single scalar threshold q on this STANDARDIZED score
       using split conformal (same SplitConformal class used elsewhere).
    6. At test time, each patient's ACTUAL interval half-width in mg is
       q * sigma_i (different patients get different physical widths,
       reflecting their own ensemble uncertainty). Evaluate trajectory
       coverage and mean width on the central compartment, using the
       exact same eval_intervals() convention as run_pk_comparison.py,
       so this row is directly comparable to the others in Table 3.

Usage:
    cd experiments
    python conformalize_pk_ensemble.py --n-patients 1000 --n-members 5 --alpha 0.10

Output:
    results/conformalized_ensemble_pk.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo, "src"))

from neural_ode_pk import generate_population_data
from baselines.deep_ensemble_pk import DeepEnsemblePK
from conformal import SplitConformal


# ============================================================
# Splits (identical convention to run_pk_comparison.py)
# ============================================================

def make_splits(n_total: int, seed: int = 42) -> dict:
    n_train = int(0.70 * n_total)
    n_cal = int(0.15 * n_total)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=gen)
    return {
        "train": idx[:n_train],
        "cal": idx[n_train:n_train + n_cal],
        "test": idx[n_train + n_cal:],
    }


def eval_intervals(lower, upper, true_traj, label="test") -> dict:
    """Identical to run_pk_comparison.py's eval_intervals, for direct
    comparability with the rest of Table 3. lower/upper/true_traj are
    (n, T) tensors for the CENTRAL COMPARTMENT ONLY."""
    covered_pt = (true_traj >= lower) & (true_traj <= upper)
    traj_cov = covered_pt.all(dim=1).float().mean().item()
    pt_cov = covered_pt.float().mean().item()
    width = (upper - lower).mean().item()
    print(f"  {label:4s}: trajectory coverage={traj_cov:.3f}  "
          f"pointwise coverage={pt_cov:.3f}  mean width={width:.4f}")
    return {"traj_coverage": traj_cov, "pointwise_coverage": pt_cov, "mean_width": width}


# ============================================================
# Clinical weighting (same absorption/elimination emphasis as
# WeightedFunctionalScore in src/conformal.py, applied here to the
# central compartment only, matching the coverage metric)
# ============================================================

def clinical_weights(times: torch.Tensor, absorption_end=2.0, elimination_start=12.0,
                      absorption_weight=2.0, elimination_weight=1.5) -> torch.Tensor:
    w = torch.ones_like(times)
    w[times <= absorption_end] = absorption_weight
    w[times >= elimination_start] = elimination_weight
    return w / w.sum() * len(w)


class EnsembleStandardizedScore:
    """
    Nonconformity score for the deep ensemble, standardized by the
    ensemble's own per-patient uncertainty (sigma).

    Uses a MAX (sup-norm) over timepoints, not a mean or L2 norm. This
    is the only score type for which "score <= threshold" mathematically
    guarantees every individual timepoint's residual stays within
    threshold * sigma -- which is exactly the trajectory-level
    all-timepoints-covered criterion this script evaluates against.
    Mirrors the weather domain's SupNormScore pattern.

    Must be constructed once per split (cal or test) since sigma differs
    between them -- this mirrors how SplitConformal expects a plain
    (prediction, ground_truth) -> scores callable, so the per-sample
    sigma is captured via closure at construction time rather than
    passed as a third argument.
    """

    def __init__(self, sigma: torch.Tensor):
        self.sigma = sigma      # (n,) ensemble's own per-patient scale

    def __call__(self, mean_pred_central: torch.Tensor, true_central: torch.Tensor) -> torch.Tensor:
        residual = (mean_pred_central - true_central).abs()   # (n, T)
        raw = residual.max(dim=1).values                       # (n,) sup-norm over time
        return raw / (self.sigma + 1e-8)                        # (n,) standardized score


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Conformalize the PK deep ensemble")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None,
                         help="Path to pk_population.pt; generates fresh data if omitted")
    parser.add_argument("--output", type=str, default="../results/conformalized_ensemble_pk.json")
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        data = torch.load(args.data, weights_only=False)
    else:
        print(f"Generating PK data for {args.n_patients} patients ...")
        data = generate_population_data(args.n_patients, seed=args.seed)

    n_total = len(data["params"])
    splits = make_splits(n_total, seed=args.seed)
    print(f"Split: {len(splits['train'])} train / {len(splits['cal'])} cal / {len(splits['test'])} test")

    times = data["times"]
    t_sub = times[::5]
    traj_sub = data["trajectories"][:, ::5, :]
    # Note: clinical weighting (absorption/elimination emphasis) is not
    # used in this sup-norm score -- see docstring for why.

    # --- Train ensemble (same class, same convention as deep_ensemble_pk.py) ---
    ensemble = DeepEnsemblePK(n_members=args.n_members, base_seed=args.seed)
    print(f"\nTraining {args.n_members} ensemble members ({args.epochs} epochs each) ...")
    ensemble.train(data, splits, epochs=args.epochs)

    cal_idx, test_idx = splits["cal"], splits["test"]
    cal_params, test_params = data["params"][cal_idx], data["params"][test_idx]
    cal_true_full, test_true_full = traj_sub[cal_idx], traj_sub[test_idx]
    cal_true = cal_true_full[:, :, 0]     # central compartment only
    test_true = test_true_full[:, :, 0]

    print("\n--- Ensemble predictions (mean, std) on cal and test ---")
    cal_mean_full, cal_std_full = ensemble.predict(cal_params, t_sub)
    test_mean_full, test_std_full = ensemble.predict(test_params, t_sub)
    cal_mean, cal_std = cal_mean_full[:, :, 0], cal_std_full[:, :, 0]
    test_mean, test_std = test_mean_full[:, :, 0], test_std_full[:, :, 0]

    # Per-patient scale = mean over timepoints of the ensemble's own std
    # on the central compartment (see docstring for why this compartment).
    cal_sigma = cal_std.mean(dim=1)      # (n_cal,)
    test_sigma = test_std.mean(dim=1)    # (n_test,)

    print(f"\nEnsemble's own per-patient uncertainty scale (sigma):")
    print(f"  Cal  sigma: mean={cal_sigma.mean():.4f}  std={cal_sigma.std():.4f}  "
          f"min={cal_sigma.min():.4f}  max={cal_sigma.max():.4f}")
    print(f"  Test sigma: mean={test_sigma.mean():.4f}  std={test_sigma.std():.4f}  "
          f"min={test_sigma.min():.4f}  max={test_sigma.max():.4f}")

    # --- Calibrate ---
    cal_score_fn = EnsembleStandardizedScore(cal_sigma)
    cp = SplitConformal(cal_score_fn, alpha=args.alpha)
    cp.calibrate(cal_mean, cal_true)

    # --- Evaluate on test: build ACTUAL mg-scale intervals using each
    #     patient's own sigma, then check central-compartment trajectory
    #     coverage exactly like the rest of Table 3. ---
    threshold = cp.threshold
    half_width_mg = threshold * test_sigma          # (n_test,) -- per-patient physical width
    lower = test_mean - half_width_mg.unsqueeze(1)  # broadcast across T
    upper = test_mean + half_width_mg.unsqueeze(1)

    print(f"\n--- Test evaluation (central compartment, alpha={args.alpha}) ---")
    result = eval_intervals(lower, upper, test_true, label="test")

    out = {
        "method": "conformalized_deep_ensemble",
        "surrogate": "neural_ode_pk",
        "n_members": args.n_members,
        "epochs_per_member": args.epochs,
        "n_patients_total": n_total,
        "alpha": args.alpha,
        "target_coverage": 1 - args.alpha,
        "split_seed": args.seed,
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
        "score_definition": (
            "clinically-weighted L2 norm of residual on central compartment "
            "(divided by sqrt(n_timepoints)), standardized by ensemble's own "
            "per-patient mean predicted std"
        ),
        "conformal_threshold_standardized": float(threshold),
        "test_sigma_mean": float(test_sigma.mean()),
        "test_sigma_std": float(test_sigma.std()),
        "test_traj_coverage": result["traj_coverage"],
        "test_pointwise_coverage": result["pointwise_coverage"],
        "test_mean_width_mg": result["mean_width"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("\nUse test_traj_coverage and test_mean_width_mg above to replace the")
    print("'Conformalized deep ensemble' row in the paper's Table 3.")


if __name__ == "__main__":
    main()
