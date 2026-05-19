"""
Bayesian Variational Inference (Mean-Field) Baseline for PK Neural ODE
=======================================================================

Replaces each Linear layer in the dynamics network with a BayesianLinear layer
whose weights follow a mean-field Gaussian approximate posterior
q(W) = N(mu_W, softplus(rho_W)^2).

Training minimises the ELBO:
    L = E_q[MSE(f_W(x), y)] + (beta / N_train) * KL(q || p)
where p = N(0, prior_sigma^2) is an isotropic Gaussian prior.

Because odeint calls ode_func multiple times per step, stochastic weight
sampling inside ode_func would make the dynamics inconsistent within one
trajectory. Strategy used here:
    - Training:   use MAP weights (mu_W) inside odeint; add KL as regulariser.
    - Inference:  sample one weight vector per trajectory from q(W), run a
                  deterministic odeint, repeat T times, aggregate.

This is a valid approximation (used in practice for neural ODEs) that decouples
the variational posterior fitting from the trajectory integration.

Usage (standalone):
    cd src
    python baselines/bayesian_vi_pk.py \\
        --n-patients 1000 --epochs 300 --prior-sigma 0.5 \\
        --output ../results/bayesian_vi_pk.json

Usage (import):
    from baselines.bayesian_vi_pk import BayesianVIPK
    vi = BayesianVIPK(prior_sigma=0.5)
    vi.train(data, split_indices, epochs=300)
    mean_pred, std_pred = vi.predict(params, t_sub, n_samples=50)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm
from torchdiffeq import odeint

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
from neural_ode_pk import PKNeuralODE, generate_population_data


# ============================================================
# BayesianLinear layer
# ============================================================

class BayesianLinear(nn.Module):
    """
    Linear layer with mean-field Gaussian weight posterior.

    Parameters:
        mu_W, mu_b:   variational mean (same role as weight/bias in nn.Linear)
        rho_W, rho_b: log-scale parameters; sigma = softplus(rho) > 0

    Two forward modes (controlled by use_map):
        use_map=True  → deterministic MAP prediction via mu_W, mu_b
        use_map=False → sample one W ~ q(W) and use it (reparameterisation)
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.mu_W = nn.Parameter(torch.empty(out_features, in_features))
        self.rho_W = nn.Parameter(torch.full((out_features, in_features), -4.0))
        self.mu_b = nn.Parameter(torch.zeros(out_features))
        self.rho_b = nn.Parameter(torch.full((out_features,), -4.0))

        nn.init.kaiming_uniform_(self.mu_W, nonlinearity="tanh")

        self.use_map = True          # set to False for posterior sampling
        self._fixed_W: torch.Tensor | None = None
        self._fixed_b: torch.Tensor | None = None

    def sigma_W(self) -> torch.Tensor:
        return F.softplus(self.rho_W) + 1e-6

    def sigma_b(self) -> torch.Tensor:
        return F.softplus(self.rho_b) + 1e-6

    def kl_divergence(self) -> torch.Tensor:
        """Analytical KL(q || N(0, prior_sigma^2)) summed over all parameters."""
        log_p = np.log(self.prior_sigma)
        sW = self.sigma_W()
        sb = self.sigma_b()
        kl_W = 0.5 * ((sW / self.prior_sigma) ** 2
                       + (self.mu_W / self.prior_sigma) ** 2
                       - 1 + 2 * (log_p - sW.log())).sum()
        kl_b = 0.5 * ((sb / self.prior_sigma) ** 2
                       + (self.mu_b / self.prior_sigma) ** 2
                       - 1 + 2 * (log_p - sb.log())).sum()
        return kl_W + kl_b

    def set_sampled_weights(self) -> None:
        """Draw one sample from q(W) and cache it for inference mode."""
        with torch.no_grad():
            self._fixed_W = self.mu_W + self.sigma_W() * torch.randn_like(self.mu_W)
            self._fixed_b = self.mu_b + self.sigma_b() * torch.randn_like(self.mu_b)

    def clear_sampled_weights(self) -> None:
        self._fixed_W = None
        self._fixed_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fixed_W is not None:
            # inference: use cached sample
            return F.linear(x, self._fixed_W, self._fixed_b)
        if self.use_map:
            # training: MAP (mean) weights
            return F.linear(x, self.mu_W, self.mu_b)
        # stochastic forward for explicit reparameterisation (not used in odeint path)
        W = self.mu_W + self.sigma_W() * torch.randn_like(self.mu_W)
        b = self.mu_b + self.sigma_b() * torch.randn_like(self.mu_b)
        return F.linear(x, W, b)


# ============================================================
# Bayesian neural ODE
# ============================================================

class BayesianPKNeuralODE(nn.Module):
    """
    PK neural ODE with BayesianLinear dynamics layers.

    Training: MAP forward (mu weights) + KL regulariser.
    Inference: sample weight sets from q(W), run deterministic odeint per sample.
    """

    def __init__(self, param_dim=4, hidden_dim=64, state_dim=2,
                 prior_sigma=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim

        self.dynamics_net = nn.Sequential(
            BayesianLinear(state_dim + param_dim, hidden_dim, prior_sigma),
            nn.Tanh(),
            BayesianLinear(hidden_dim, hidden_dim, prior_sigma),
            nn.Tanh(),
            BayesianLinear(hidden_dim, state_dim, prior_sigma),
        )
        self._params: torch.Tensor | None = None

    def _bayes_layers(self) -> list[BayesianLinear]:
        return [m for m in self.dynamics_net if isinstance(m, BayesianLinear)]

    def ode_func(self, t, y):
        inp = torch.cat([y, self._params.expand(y.shape[0], -1)], dim=-1)
        return self.dynamics_net(inp)

    def forward(self, params, y0, t_span):
        """MAP forward pass (mu weights) — used during training."""
        self._params = params
        return odeint(self.ode_func, y0, t_span, method="euler",
                      options={"step_size": 0.1})

    def kl_total(self) -> torch.Tensor:
        return sum(layer.kl_divergence() for layer in self._bayes_layers())

    def vi_predict(self, params: torch.Tensor, t_span: torch.Tensor,
                   n_samples: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n_samples weight vectors from q(W), run a deterministic odeint
        for each, return (mean, std) over trajectories.
        """
        n = len(params)
        y0 = torch.stack([torch.tensor([100.0, 0.0])] * n)
        preds = []

        self.eval()
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample one weight vector and freeze it
                for layer in self._bayes_layers():
                    layer.set_sampled_weights()

                pred = self.forward(params, y0, t_span).permute(1, 0, 2)
                preds.append(pred)

            # Restore MAP mode
            for layer in self._bayes_layers():
                layer.clear_sampled_weights()

        stack = torch.stack(preds)  # (S, n, T, 2)
        return stack.mean(dim=0), stack.std(dim=0)


# ============================================================
# Training wrapper
# ============================================================

class BayesianVIPK:
    """Wraps ELBO training and posterior-predictive inference."""

    def __init__(self, prior_sigma: float = 0.5):
        self.prior_sigma = prior_sigma
        self.model: BayesianPKNeuralODE | None = None
        self.train_runtime_s: float = 0.0

    def train(self, data: dict, split_indices: dict, epochs: int = 300,
              lr: float = 1e-3, batch_size: int = 64,
              kl_weight: float = 1.0, seed: int = 42) -> None:
        """
        Train with ELBO = MSE_loss + (kl_weight / n_train) * KL(q || p).

        kl_weight anneals from 0 → kl_weight over the first 20% of epochs to
        prevent posterior collapse early in training.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        params = data["params"]
        times = data["times"]
        trajectories = data["trajectories"]
        train_idx = split_indices["train"]
        n_train = len(train_idx)

        t_sub = times[::5]
        traj_sub = trajectories[:, ::5, :]

        self.model = BayesianPKNeuralODE(prior_sigma=self.prior_sigma)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        warmup_epochs = max(1, int(0.20 * epochs))

        t0 = time.time()
        for epoch in range(epochs):
            self.model.train()

            # KL annealing: ramp beta from 0 → kl_weight over first warmup_epochs
            beta = kl_weight * min(1.0, (epoch + 1) / warmup_epochs) / n_train

            perm = torch.randperm(len(train_idx))[:batch_size]
            batch_idx = train_idx[perm]
            batch_params = params[batch_idx]
            batch_traj = traj_sub[batch_idx]
            y0 = torch.stack([torch.tensor([100.0, 0.0])] * batch_size)

            pred = self.model(batch_params, y0, t_sub).permute(1, 0, 2)
            mse = loss_fn(pred, batch_traj)
            kl = self.model.kl_total()
            loss = mse + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.train_runtime_s = time.time() - t0

    def predict(self, params: torch.Tensor, t_sub: torch.Tensor,
                n_samples: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Call train() first")
        return self.model.vi_predict(params, t_sub, n_samples=n_samples)

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
    parser = argparse.ArgumentParser(description="Bayesian VI PK baseline")
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--prior-sigma", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default="../results/bayesian_vi_pk.json")
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

    vi = BayesianVIPK(prior_sigma=args.prior_sigma)
    print(f"Training Bayesian VI model (prior_sigma={args.prior_sigma}, "
          f"{args.epochs} epochs) ...")
    vi.train(data, splits, epochs=args.epochs,
             kl_weight=args.kl_weight, seed=args.seed)
    print(f"  Done in {vi.train_runtime_s:.1f}s")

    def _eval(idx, label):
        p = data["params"][idx]
        t = traj_sub[idx]
        t0 = time.time()
        ivs = vi.make_intervals(p, t_sub, alpha=args.alpha,
                                n_samples=args.n_samples)
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
        "method": "bayesian_vi",
        "surrogate": "neural_ode_pk",
        "prior_sigma": args.prior_sigma,
        "kl_weight": args.kl_weight,
        "n_vi_samples": args.n_samples,
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
        "runtime_train_s": float(vi.train_runtime_s),
        "runtime_inference_test_s": float(test_inf_s),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
