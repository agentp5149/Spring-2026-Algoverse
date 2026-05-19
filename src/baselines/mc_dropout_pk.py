"""
MC Dropout Baseline for PK Neural ODE
======================================

Adds dropout after each hidden Tanh layer. At inference time, dropout is kept
active so that T stochastic forward passes give an approximate posterior
predictive distribution (Gal & Ghahramani, 2016).

Training is identical to the deterministic surrogate (MSE loss); no additional
hyperparameters beyond the dropout rate p and the number of MC samples T.

Usage (standalone):
    cd src
    python baselines/mc_dropout_pk.py \\
        --n-patients 1000 --epochs 200 --dropout-p 0.1 --n-samples 50 \\
        --output ../results/mc_dropout_pk.json

Usage (import):
    from baselines.mc_dropout_pk import MCDropoutPK
    mcd = MCDropoutPK(dropout_p=0.1)
    mcd.train(data, split_indices, epochs=200)
    mean_pred, std_pred = mcd.predict(params, t_sub, n_samples=50)
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
from torchdiffeq import odeint

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
from neural_ode_pk import generate_population_data


# ============================================================
# Model
# ============================================================

class MCDropoutPKNeuralODE(nn.Module):
    """
    PKNeuralODE augmented with dropout after each hidden activation.
    Keeping self.train() active at inference provides MC Dropout uncertainty.
    """

    def __init__(self, param_dim=4, hidden_dim=64, state_dim=2, dropout_p=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.dropout_p = dropout_p

        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + param_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, state_dim),
        )
        self._params = None

    def ode_func(self, t, y):
        inp = torch.cat([y, self._params.expand(y.shape[0], -1)], dim=-1)
        return self.dynamics_net(inp)

    def forward(self, params, y0, t_span):
        self._params = params
        return odeint(self.ode_func, y0, t_span, method="euler",
                      options={"step_size": 0.1})

    def mc_predict(self, params, t_span, n_samples=50):
        """
        Run n_samples stochastic forward passes with dropout active.
        Returns (mean, std) over samples; shapes (n, T, state_dim).
        """
        n = len(params)
        y0 = torch.stack([torch.tensor([100.0, 0.0])] * n)
        preds = []
        self.train()  # keep dropout active
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(params, y0, t_span).permute(1, 0, 2)
                preds.append(pred)
        stack = torch.stack(preds)  # (S, n, T, 2)
        return stack.mean(dim=0), stack.std(dim=0)


# ============================================================
# Training wrapper
# ============================================================

class MCDropoutPK:
    """Wraps training and inference for MCDropoutPKNeuralODE."""

    def __init__(self, dropout_p=0.1):
        self.dropout_p = dropout_p
        self.model: MCDropoutPKNeuralODE | None = None
        self.train_runtime_s: float = 0.0

    def train(self, data: dict, split_indices: dict, epochs: int = 200,
              lr: float = 1e-3, batch_size: int = 64, seed: int = 42) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        params = data["params"]
        times = data["times"]
        trajectories = data["trajectories"]
        train_idx = split_indices["train"]

        t_sub = times[::5]
        traj_sub = trajectories[:, ::5, :]

        self.model = MCDropoutPKNeuralODE(dropout_p=self.dropout_p)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        t0 = time.time()
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(train_idx))[:batch_size]
            batch_idx = train_idx[perm]
            batch_params = params[batch_idx]
            batch_traj = traj_sub[batch_idx]
            y0 = torch.stack([torch.tensor([100.0, 0.0])] * batch_size)

            pred = self.model(batch_params, y0, t_sub).permute(1, 0, 2)
            loss = loss_fn(pred, batch_traj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.train_runtime_s = time.time() - t0

    def predict(self, params: torch.Tensor, t_sub: torch.Tensor,
                n_samples: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Call train() first")
        return self.model.mc_predict(params, t_sub, n_samples=n_samples)

    def make_intervals(self, params: torch.Tensor, t_sub: torch.Tensor,
                       alpha: float = 0.1, n_samples: int = 50) -> dict:
        z = float(scipy_norm.ppf(1 - alpha / 2))
        mean_pred, std_pred = self.predict(params, t_sub, n_samples)
        return {
            "mean": mean_pred,
            "std": std_pred,
            "lower": mean_pred - z * std_pred,
            "upper": mean_pred + z * std_pred,
            "z": z,
            "alpha": alpha,
        }


# ============================================================
# CLI
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
    parser = argparse.ArgumentParser(description="MC Dropout PK baseline")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default="../results/mc_dropout_pk.json")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        print(f"Loading data from {args.data}")
        data = torch.load(args.data, weights_only=False)
    else:
        print(f"Generating PK data for {args.n_patients} patients ...")
        data = generate_population_data(args.n_patients, seed=args.seed)

    splits = _make_splits(len(data["params"]), seed=args.seed)
    t_sub = data["times"][::5]
    traj_sub = data["trajectories"][:, ::5, :]

    mcd = MCDropoutPK(dropout_p=args.dropout_p)
    print(f"Training MC Dropout model (dropout_p={args.dropout_p}, "
          f"{args.epochs} epochs) ...")
    mcd.train(data, splits, epochs=args.epochs, seed=args.seed)
    print(f"  Done in {mcd.train_runtime_s:.1f}s")

    def _eval(idx, label):
        p = data["params"][idx]
        t = traj_sub[idx]
        t0 = time.time()
        ivs = mcd.make_intervals(p, t_sub, alpha=args.alpha, n_samples=args.n_samples)
        inf_s = time.time() - t0
        covered = ((t[:, :, 0] >= ivs["lower"][:, :, 0]) &
                   (t[:, :, 0] <= ivs["upper"][:, :, 0]))
        cov = covered.all(dim=1).float().mean().item()
        width = (ivs["upper"][:, :, 0] - ivs["lower"][:, :, 0]).mean().item()
        print(f"  {label:4s}: coverage={cov:.3f}  width={width:.4f}  "
              f"inf_time={inf_s:.2f}s")
        return cov, width, inf_s

    z = float(scipy_norm.ppf(1 - args.alpha / 2))
    print(f"\n--- Evaluation (alpha={args.alpha}, z={z:.3f}) ---")
    cal_cov, cal_width, _ = _eval(splits["cal"], "cal")
    test_cov, test_width, test_inf_s = _eval(splits["test"], "test")

    out = {
        "method": "mc_dropout",
        "surrogate": "neural_ode_pk",
        "dropout_p": args.dropout_p,
        "n_mc_samples": args.n_samples,
        "epochs": args.epochs,
        "n_patients_total": len(data["params"]),
        "alpha": args.alpha,
        "target_coverage": 1 - args.alpha,
        "z": z,
        "split_seed": args.seed,
        "n_train": len(splits["train"]),
        "n_cal": len(splits["cal"]),
        "n_test": len(splits["test"]),
        "cal_empirical_coverage": float(cal_cov),
        "cal_mean_interval_width": float(cal_width),
        "test_empirical_coverage": float(test_cov),
        "test_mean_interval_width": float(test_width),
        "runtime_train_s": float(mcd.train_runtime_s),
        "runtime_inference_test_s": float(test_inf_s),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
