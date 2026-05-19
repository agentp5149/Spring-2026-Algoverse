"""
Deep Ensemble Baseline for PK Neural ODE
=========================================

Trains M independent PKNeuralODE surrogates with different random seeds.
Ensemble predictive mean = average of member predictions.
Ensemble predictive std  = standard deviation across members.
Prediction intervals: mean ± z * std  (z chosen to match nominal alpha).

Usage (standalone):
    cd src
    python baselines/deep_ensemble_pk.py \\
        --n-patients 1000 --n-members 5 --epochs 100 \\
        --output ../results/deep_ensemble_pk.json

Usage (import):
    from baselines.deep_ensemble_pk import DeepEnsemblePK
    ens = DeepEnsemblePK(n_members=5)
    ens.train(data, split_indices, epochs=100)
    mean_pred, std_pred = ens.predict(params, t_subsample)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as scipy_norm

# Allow running from src/ or from project root
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
from neural_ode_pk import PKNeuralODE, generate_population_data


# ============================================================
# Ensemble class
# ============================================================

class DeepEnsemblePK:
    """
    Wraps M independently trained PKNeuralODE instances.

    After calling train(), use predict() to get mean/std predictions and
    make_intervals() to produce lower/upper bounds at a given alpha.
    """

    def __init__(self, n_members: int = 5, base_seed: int = 42):
        self.n_members = n_members
        self.base_seed = base_seed
        self.members: list[PKNeuralODE] = []
        self.member_seeds: list[int] = []
        self.member_runtimes: list[float] = []

    def train(self, data: dict, split_indices: dict, epochs: int = 100,
              lr: float = 1e-3, batch_size: int = 64) -> None:
        """Train all M members. Populates self.members and self.member_runtimes."""
        self.members = []
        self.member_seeds = []
        self.member_runtimes = []

        params = data["params"]
        times = data["times"]
        trajectories = data["trajectories"]
        train_idx = split_indices["train"]

        t_sub = times[::5]
        traj_sub = trajectories[:, ::5, :]

        for m in range(self.n_members):
            seed = self.base_seed + m * 100
            self.member_seeds.append(seed)
            print(f"  Member {m + 1}/{self.n_members} (seed={seed}) ... ", end="", flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = PKNeuralODE()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            t0 = time.time()
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(len(train_idx))[:batch_size]
                batch_idx = train_idx[perm]
                batch_params = params[batch_idx]
                batch_traj = traj_sub[batch_idx]
                y0 = torch.stack([torch.tensor([100.0, 0.0])] * batch_size)

                pred = model(batch_params, y0, t_sub).permute(1, 0, 2)
                loss = loss_fn(pred, batch_traj)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elapsed = time.time() - t0
            self.members.append(model)
            self.member_runtimes.append(elapsed)
            print(f"done in {elapsed:.1f}s")

    def predict(self, params: torch.Tensor, t_sub: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run all members and return (mean, std) of their predictions.

        Returns:
            mean_pred: (n, T, state_dim)
            std_pred:  (n, T, state_dim)
        """
        if not self.members:
            raise RuntimeError("Call train() before predict()")

        n = len(params)
        y0 = torch.stack([torch.tensor([100.0, 0.0])] * n)
        preds = []

        for model in self.members:
            model.eval()
            with torch.no_grad():
                pred = model(params, y0, t_sub).permute(1, 0, 2)  # (n, T, 2)
            preds.append(pred)

        stack = torch.stack(preds)  # (M, n, T, 2)
        return stack.mean(dim=0), stack.std(dim=0)

    def make_intervals(self, params: torch.Tensor, t_sub: torch.Tensor,
                       alpha: float = 0.1) -> dict:
        """
        Predict and construct symmetric Gaussian intervals at level 1 - alpha.

        Returns dict with keys: mean, std, lower, upper, z
        """
        z = float(scipy_norm.ppf(1 - alpha / 2))
        mean_pred, std_pred = self.predict(params, t_sub)
        return {
            "mean": mean_pred,
            "std": std_pred,
            "lower": mean_pred - z * std_pred,
            "upper": mean_pred + z * std_pred,
            "z": z,
            "alpha": alpha,
        }


# ============================================================
# CLI entry point
# ============================================================

def _make_splits(n_total: int, seed: int = 42) -> dict:
    n_train = int(0.70 * n_total)
    n_cal = int(0.15 * n_total)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=gen)
    return {
        "train": idx[:n_train],
        "cal": idx[n_train:n_train + n_cal],
        "test": idx[n_train + n_cal:],
    }


def main():
    parser = argparse.ArgumentParser(description="Deep Ensemble PK baseline")
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to pk_population.pt; generates fresh data if omitted")
    parser.add_argument("--output", type=str, default="../results/deep_ensemble_pk.json")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ------ data ------------------------------------------
    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        data = torch.load(args.data, weights_only=False)
    else:
        print(f"Generating PK data for {args.n_patients} patients ...")
        data = generate_population_data(args.n_patients, seed=args.seed)

    n_total = len(data["params"])
    splits = _make_splits(n_total, seed=args.seed)
    print(f"Split: {len(splits['train'])} train / "
          f"{len(splits['cal'])} cal / {len(splits['test'])} test")

    times = data["times"]
    t_sub = times[::5]
    traj_sub = data["trajectories"][:, ::5, :]

    # ------ train -----------------------------------------
    ensemble = DeepEnsemblePK(n_members=args.n_members, base_seed=args.seed)
    print(f"\nTraining {args.n_members} members ({args.epochs} epochs each) ...")
    wall_start = time.time()
    ensemble.train(data, splits, epochs=args.epochs)
    total_train_s = time.time() - wall_start

    print(f"\nTotal wall time: {total_train_s:.1f}s")
    print(f"Per-member:  mean={np.mean(ensemble.member_runtimes):.1f}s  "
          f"std={np.std(ensemble.member_runtimes):.1f}s")

    # ------ evaluate on cal + test ------------------------
    def _eval_split(idx, label):
        p = data["params"][idx]
        t = traj_sub[idx]
        t0 = time.time()
        intervals = ensemble.make_intervals(p, t_sub, alpha=args.alpha)
        inf_s = time.time() - t0

        lower, upper = intervals["lower"], intervals["upper"]
        # Trajectory-level coverage: all timepoints in central compartment must be covered
        covered = ((t[:, :, 0] >= lower[:, :, 0]) & (t[:, :, 0] <= upper[:, :, 0]))
        traj_cov = covered.all(dim=1).float().mean().item()
        # Mean interval width on central compartment
        width = (upper[:, :, 0] - lower[:, :, 0]).mean().item()
        print(f"  {label:4s}: coverage={traj_cov:.3f} (target={1-args.alpha:.2f})  "
              f"width={width:.4f}  inf_time={inf_s:.2f}s")
        return traj_cov, width, inf_s

    print(f"\n--- Evaluation (alpha={args.alpha}, z={scipy_norm.ppf(1-args.alpha/2):.3f}) ---")
    cal_cov, cal_width, _ = _eval_split(splits["cal"], "cal")
    test_cov, test_width, test_inf_s = _eval_split(splits["test"], "test")

    # ------ save results ----------------------------------
    out = {
        "method": "deep_ensemble",
        "surrogate": "neural_ode_pk",
        "n_members": args.n_members,
        "epochs_per_member": args.epochs,
        "n_patients_total": n_total,
        "alpha": args.alpha,
        "target_coverage": 1 - args.alpha,
        "z": float(scipy_norm.ppf(1 - args.alpha / 2)),
        "split_seed": args.seed,
        "n_train": len(splits["train"]),
        "n_cal": len(splits["cal"]),
        "n_test": len(splits["test"]),
        "cal_empirical_coverage": float(cal_cov),
        "cal_mean_interval_width": float(cal_width),
        "test_empirical_coverage": float(test_cov),
        "test_mean_interval_width": float(test_width),
        "runtime_train_total_s": float(total_train_s),
        "runtime_train_per_member_s": [float(t) for t in ensemble.member_runtimes],
        "runtime_train_mean_s": float(np.mean(ensemble.member_runtimes)),
        "runtime_train_std_s": float(np.std(ensemble.member_runtimes)),
        "runtime_inference_test_s": float(test_inf_s),
        "member_seeds": ensemble.member_seeds,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
