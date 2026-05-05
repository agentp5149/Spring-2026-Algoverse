"""
Neural ODE Pharmacokinetics Surrogate
=======================================

Implements a two-compartment PK model as both a traditional ODE (ground truth)
and a neural ODE surrogate. This is the simplest of our three scientific
domains and serves as the development testbed for conformal methods.

The ground truth model:
    dA_central/dt = -k_el * A_central - k_12 * A_central + k_21 * A_peripheral + (dose input)
    dA_peripheral/dt = k_12 * A_central - k_21 * A_peripheral

    where A = amount in compartment, k = rate constants

We vary patient parameters (k_el, k_12, k_21, volume) to create a
population of PK curves. The neural ODE learns to predict concentration
trajectories from patient parameters.

Usage:
    # Generate synthetic PK data
    python neural_ode_pk.py --generate-data --n-patients 5000 --output ../data/pk/

    # Train surrogate
    python neural_ode_pk.py --train --data ../data/pk/ --epochs 200 --output ../models/pk_surrogate.pt

    # Run inference
    python neural_ode_pk.py --predict --model ../models/pk_surrogate.pt --params 0.5 0.3 0.2 1.0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


# ============================================================
# Ground truth: two-compartment PK model
# ============================================================

class TwoCompartmentPK:
    """Traditional two-compartment PK model (ground truth simulator)."""

    def __init__(self, k_el, k_12, k_21, V_central, dose=100.0, dt=0.1, t_max=24.0):
        self.k_el = k_el        # elimination rate
        self.k_12 = k_12        # central -> peripheral rate
        self.k_21 = k_21        # peripheral -> central rate
        self.V_central = V_central  # central volume of distribution
        self.dose = dose
        self.dt = dt
        self.t_max = t_max

    def simulate(self):
        """Run forward simulation, return (times, concentrations)."""
        t_span = torch.linspace(0, self.t_max, int(self.t_max / self.dt) + 1)

        # Initial state: full dose in central compartment, nothing in peripheral
        y0 = torch.tensor([self.dose, 0.0])

        k_el = self.k_el
        k_12 = self.k_12
        k_21 = self.k_21

        def pk_ode(t, y):
            A_c, A_p = y[0], y[1]
            dA_c = -k_el * A_c - k_12 * A_c + k_21 * A_p
            dA_p = k_12 * A_c - k_21 * A_p
            return torch.stack([dA_c, dA_p])

        with torch.no_grad():
            trajectory = odeint(pk_ode, y0, t_span, method="dopri5")

        # Convert amount to concentration
        concentrations = trajectory[:, 0] / self.V_central

        return t_span, concentrations, trajectory


def generate_population_data(n_patients, seed=42):
    """Generate PK data for a population of patients with varying parameters."""
    rng = np.random.RandomState(seed)

    all_params = []
    all_times = []
    all_concentrations = []
    all_trajectories = []

    # Sample patient parameters from realistic ranges
    k_els = rng.lognormal(mean=np.log(0.3), sigma=0.4, size=n_patients)
    k_12s = rng.lognormal(mean=np.log(0.2), sigma=0.3, size=n_patients)
    k_21s = rng.lognormal(mean=np.log(0.15), sigma=0.3, size=n_patients)
    V_centrals = rng.lognormal(mean=np.log(10.0), sigma=0.3, size=n_patients)

    print(f"Generating PK data for {n_patients} patients...")
    for i in range(n_patients):
        params = {
            "k_el": float(k_els[i]),
            "k_12": float(k_12s[i]),
            "k_21": float(k_21s[i]),
            "V_central": float(V_centrals[i]),
        }
        model = TwoCompartmentPK(**params)
        times, conc, traj = model.simulate()

        all_params.append(torch.tensor([params["k_el"], params["k_12"],
                                         params["k_21"], params["V_central"]]))
        all_times.append(times)
        all_concentrations.append(conc)
        all_trajectories.append(traj)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{n_patients}")

    params_tensor = torch.stack(all_params)
    times_tensor = all_times[0]  # all same time grid
    conc_tensor = torch.stack(all_concentrations)
    traj_tensor = torch.stack(all_trajectories)

    return {
        "params": params_tensor,        # (n_patients, 4)
        "times": times_tensor,           # (n_timepoints,)
        "concentrations": conc_tensor,   # (n_patients, n_timepoints)
        "trajectories": traj_tensor,     # (n_patients, n_timepoints, 2)
    }


# ============================================================
# Neural ODE surrogate
# ============================================================

class PKNeuralODE(nn.Module):
    """
    Neural ODE surrogate for PK prediction.

    Takes patient parameters (k_el, k_12, k_21, V_central) as input
    and predicts the concentration trajectory over time.
    """

    def __init__(self, param_dim=4, hidden_dim=64, state_dim=2):
        super().__init__()
        self.param_dim = param_dim
        self.state_dim = state_dim

        # Encode patient parameters into ODE dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + param_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.params = None  # set per-forward

    def ode_func(self, t, y):
        # Concatenate state with patient parameters
        inp = torch.cat([y, self.params.expand(y.shape[0], -1)], dim=-1)
        return self.dynamics_net(inp)

    def forward(self, params, y0, t_span):
        self.params = params
        # odeint expects y0 shape (batch, state_dim) for batched solving
        trajectory = odeint(self.ode_func, y0, t_span, method="euler",
                           options={"step_size": 0.1})
        # trajectory shape: (n_times, batch, state_dim)
        return trajectory


def train_surrogate(data, epochs=200, lr=1e-3, batch_size=64):
    """Train the neural ODE surrogate on population PK data."""
    params = data["params"]
    times = data["times"]
    trajectories = data["trajectories"]

    n_total = len(params)
    n_train = int(0.7 * n_total)
    n_cal = int(0.15 * n_total)

    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    test_idx = indices[n_train + n_cal:]

    print(f"Split: {len(train_idx)} train, {len(cal_idx)} calibration, {len(test_idx)} test")

    model = PKNeuralODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Use a subset of time points for faster training
    t_subsample = times[::5]
    traj_subsample = trajectories[:, ::5, :]

    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()

        # Mini-batch
        perm = torch.randperm(len(train_idx))[:batch_size]
        batch_idx = train_idx[perm]
        batch_params = params[batch_idx]
        batch_traj = traj_subsample[batch_idx]

        y0 = batch_traj[0]  # initial state is first timepoint
        # For simplicity, use first patient's y0 pattern
        y0 = torch.stack([torch.tensor([100.0, 0.0])] * batch_size)

        pred_traj = model(batch_params, y0, t_subsample)
        pred_traj = pred_traj.permute(1, 0, 2)  # (batch, time, state)

        loss = loss_fn(pred_traj, batch_traj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_params = params[cal_idx[:20]]
                val_traj = traj_subsample[cal_idx[:20]]
                val_y0 = torch.stack([torch.tensor([100.0, 0.0])] * 20)
                val_pred = model(val_params, val_y0, t_subsample).permute(1, 0, 2)
                val_loss = loss_fn(val_pred, val_traj).item()
            print(f"  Epoch {epoch+1}/{epochs} - Train: {loss.item():.6f}, Val: {val_loss:.6f}")

    return model, {"train": train_idx, "cal": cal_idx, "test": test_idx}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Neural ODE PK Surrogate")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic PK data")
    parser.add_argument("--train", action="store_true", help="Train surrogate")
    parser.add_argument("--predict", action="store_true", help="Run prediction")
    parser.add_argument("--n-patients", type=int, default=5000, help="Number of patients")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--data", type=str, default="../data/pk/", help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--model", type=str, default=None, help="Model path for prediction")
    parser.add_argument("--params", nargs=4, type=float, default=None,
                       help="Patient params: k_el k_12 k_21 V_central")
    args = parser.parse_args()

    if args.generate_data:
        data = generate_population_data(args.n_patients)
        out_dir = args.output or args.data
        os.makedirs(out_dir, exist_ok=True)
        torch.save(data, os.path.join(out_dir, "pk_population.pt"))
        print(f"Saved population data to {out_dir}/pk_population.pt")
        print(f"  Params shape: {data['params'].shape}")
        print(f"  Times shape: {data['times'].shape}")
        print(f"  Concentrations shape: {data['concentrations'].shape}")

    elif args.train:
        data_path = os.path.join(args.data, "pk_population.pt")
        data = torch.load(data_path, weights_only=False)
        print(f"Loaded data from {data_path}")
        model, splits = train_surrogate(data, epochs=args.epochs)
        out_path = args.output or "../models/pk_surrogate.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(), "splits": splits}, out_path)
        print(f"Saved model to {out_path}")

    elif args.predict:
        if args.params is None:
            print("Provide --params k_el k_12 k_21 V_central")
            return
        print(f"Predicting for params: {args.params}")
        # Load and run (simplified)
        model = PKNeuralODE()
        checkpoint = torch.load(args.model, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        params = torch.tensor(args.params).unsqueeze(0)
        y0 = torch.tensor([[100.0, 0.0]])
        t_span = torch.linspace(0, 24, 241)
        with torch.no_grad():
            pred = model(params, y0, t_span)
        conc = pred[:, 0, 0] / args.params[3]
        print(f"Peak concentration: {conc.max().item():.2f}")
        print(f"Concentration at 24h: {conc[-1].item():.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
