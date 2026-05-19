"""
Week 2: Molecular Conformal Prediction + MC Dropout Baseline
=============================================================
Vijay - Team PVAV

Tasks:
    1. Implement conformal prediction with integrated trajectory distance
       nonconformity score for molecular dynamics trajectories.
    2. Calibrate on held-out MD17 data and run in-distribution coverage
       check at 90% and 95% nominal levels.
    3. Compare coverage and interval width against MC dropout on the same
       data. Log results in shared results template.

Usage:
    # Run conformal + MC dropout on aspirin (default)
    python molecular_week2.py

    # Run on a different molecule
    python molecular_week2.py --molecule ethanol

    # Run on all molecules
    python molecular_week2.py --all

    # Run with a specific alpha
    python molecular_week2.py --alpha 0.05

Output:
    results/molecular/molecular_conformal_indist.pt
    results/molecular/molecular_mc_dropout_indist.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Make sure src/ is on the path so we can import conformal.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from conformal import SplitConformal, TrajectoryNormScore


# ============================================================
# Data loading
# ============================================================

MOLECULE_FILES = {
    "aspirin":      "data/md17/md17_aspirin.npz",
    "ethanol":      "data/md17/md17_ethanol.npz",
    "uracil":       "data/md17/md17_uracil.npz",
    "malonaldehyde": "data/md17/md17_malonaldehyde.npz",
}


def load_md17(molecule: str, seed: int = 42):
    """
    Load MD17 data for a given molecule and apply the 70/15/15 split
    policy from docs/split_policy.md.

    MD17 .npz files contain:
        - R: (N, n_atoms, 3)  atomic positions (Angstrom)
        - F: (N, n_atoms, 3)  atomic forces (kcal/mol/Angstrom)
        - E: (N,)             total energies (kcal/mol)

    We treat per-atom forces along a snapshot as the "trajectory" the
    surrogate must predict, matching the integrated trajectory distance
    nonconformity score.

    Returns:
        train_F, cal_F, test_F  -- force tensors, each (n, n_atoms, 3)
        train_R, cal_R, test_R  -- position tensors (inputs to surrogate)
    """
    path = MOLECULE_FILES[molecule]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MD17 file not found: {path}\n"
            f"Download from http://www.sgdml.org and place in data/md17/"
        )

    data = np.load(path)
    R = torch.tensor(data["R"], dtype=torch.float32)   # positions
    F = torch.tensor(data["F"], dtype=torch.float32)   # forces (target)

    N = len(R)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)

    n_train = int(0.70 * N)
    n_cal   = int(0.15 * N)
    # n_test  = N - n_train - n_cal  (remainder)

    train_idx = idx[:n_train]
    cal_idx   = idx[n_train : n_train + n_cal]
    test_idx  = idx[n_train + n_cal :]

    print(f"Molecule: {molecule}")
    print(f"  Total snapshots : {N}")
    print(f"  Train           : {len(train_idx)}")
    print(f"  Cal             : {len(cal_idx)}")
    print(f"  Test            : {len(test_idx)}")

    return (
        R[train_idx], R[cal_idx], R[test_idx],
        F[train_idx], F[cal_idx], F[test_idx],
    )


# ============================================================
# Surrogate model  (lightweight MLP over flattened positions)
# ============================================================

class MolecularSurrogate(nn.Module):
    """
    Lightweight MLP surrogate that maps flattened atomic positions
    to per-atom forces.

    In a real run you would load a pretrained MACE/NequIP checkpoint.
    This MLP is used here so the Week 2 conformal/MC-dropout pipeline
    can be validated end-to-end without a GPU cluster.

    Replace load_surrogate() below with your actual MACE loader once
    you have the checkpoint.
    """

    def __init__(self, n_atoms: int, hidden: int = 256, dropout_p: float = 0.1):
        super().__init__()
        self.n_atoms = n_atoms
        self.dropout_p = dropout_p
        in_dim  = n_atoms * 3
        out_dim = n_atoms * 3

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """
        Args:
            R: (batch, n_atoms, 3) atomic positions
        Returns:
            F_pred: (batch, n_atoms, 3) predicted forces
        """
        batch = R.shape[0]
        x = R.reshape(batch, -1)
        out = self.net(x)
        return out.reshape(batch, self.n_atoms, 3)


def train_surrogate(
    model: MolecularSurrogate,
    train_R: torch.Tensor,
    train_F: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> MolecularSurrogate:
    """Train the MLP surrogate on force prediction."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    N = len(train_R)

    model.train()
    # Disable dropout during training — keep it only for MC inference
    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            idx   = perm[start : start + batch_size]
            R_b   = train_R[idx]
            F_b   = train_F[idx]
            optimizer.zero_grad()
            F_pred = model(R_b)
            loss   = loss_fn(F_pred, F_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/N:.6f}")

    model.eval()
    return model


def get_predictions(model: MolecularSurrogate, R: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Run deterministic inference (dropout off)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(R), batch_size):
            preds.append(model(R[start : start + batch_size]))
    return torch.cat(preds, dim=0)


# ============================================================
# Conformal prediction  (Step 1 & 2 of your Week 2 tasks)
# ============================================================

def run_conformal(
    model: MolecularSurrogate,
    cal_R: torch.Tensor,
    cal_F: torch.Tensor,
    test_R: torch.Tensor,
    test_F: torch.Tensor,
    alpha: float,
    molecule: str,
) -> dict:
    """
    Calibrate conformal predictor and evaluate in-distribution coverage.

    Uses TrajectoryNormScore (integrated trajectory distance) from conformal.py.
    Forces tensor shape: (n, n_atoms, 3) — treated as a "trajectory" over atoms.
    """
    print(f"\n--- Conformal prediction (alpha={alpha}, target={1-alpha:.0%}) ---")

    cal_pred  = get_predictions(model, cal_R)
    test_pred = get_predictions(model, test_R)

    score_fn = TrajectoryNormScore(normalize_by_length=True)
    cp = SplitConformal(score_fn, alpha=alpha)
    cp.calibrate(cal_pred, cal_F)
    eval_results = cp.evaluate(test_pred, test_F)

    results = {
        "surrogate":      "molecular",
        "method":         "conformal",
        "split":          "indist",
        "molecule":       molecule,
        "alpha":          alpha,
        "coverage":       eval_results["coverage"],
        "target_coverage": 1 - alpha,
        "coverage_gap":   eval_results["coverage_gap"],
        "mean_width":     eval_results["mean_width"],
        "threshold":      cp.threshold,
        "n_cal":          len(cal_R),
        "n_test":         len(test_R),
        "predictions":    test_pred,
        "ground_truth":   test_F,
        "scores":         eval_results["test_scores"],
        "cal_scores":     cp.cal_scores,
    }

    return results


# ============================================================
# MC Dropout baseline  (Step 3 of your Week 2 tasks)
# ============================================================

def mc_dropout_predict(
    model: MolecularSurrogate,
    R: torch.Tensor,
    n_samples: int = 30,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run MC dropout inference.

    Keeps dropout active at inference time. Returns mean and std
    over n_samples stochastic forward passes.

    Args:
        model: MolecularSurrogate with dropout layers
        R:     (n, n_atoms, 3) input positions
        n_samples: number of MC forward passes

    Returns:
        mean_pred: (n, n_atoms, 3)
        std_pred:  (n, n_atoms, 3)  — per-output std across MC samples
    """
    # Enable dropout at inference time
    model.train()

    all_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            for start in range(0, len(R), batch_size):
                preds.append(model(R[start : start + batch_size]))
            all_preds.append(torch.cat(preds, dim=0))

    model.eval()

    stacked   = torch.stack(all_preds, dim=0)   # (n_samples, n, n_atoms, 3)
    mean_pred = stacked.mean(dim=0)              # (n, n_atoms, 3)
    std_pred  = stacked.std(dim=0)               # (n, n_atoms, 3)

    return mean_pred, std_pred


def evaluate_mc_dropout(
    mean_pred: torch.Tensor,
    std_pred: torch.Tensor,
    ground_truth: torch.Tensor,
    n_sigma: float = 2.0,
) -> dict:
    """
    Evaluate MC dropout coverage using mean ± n_sigma*std intervals.

    Coverage: fraction of test samples where the integrated force error
    is within n_sigma standard deviations of the MC mean.

    n_sigma=2.0 targets ~95% coverage for Gaussian uncertainty.
    n_sigma=1.645 targets ~90%.

    Args:
        mean_pred:    (n, n_atoms, 3)
        std_pred:     (n, n_atoms, 3)
        ground_truth: (n, n_atoms, 3)
        n_sigma:      interval half-width in units of std

    Returns:
        dict with coverage, mean_width, and per-sample errors
    """
    # Use same TrajectoryNormScore metric for apples-to-apples comparison
    score_fn = TrajectoryNormScore(normalize_by_length=True)

    # Error of mean prediction vs ground truth
    errors = score_fn(mean_pred, ground_truth)

    # Interval width: n_sigma * mean per-sample std (flattened over atoms/xyz)
    flat_std  = std_pred.reshape(std_pred.shape[0], -1)
    mean_std  = flat_std.norm(dim=1) / np.sqrt(flat_std.shape[1])
    widths    = 2 * n_sigma * mean_std

    # Coverage: error < half-width
    half_width = n_sigma * mean_std
    covered    = (errors <= half_width).float()
    coverage   = covered.mean().item()
    mean_width = widths.mean().item()

    return {
        "coverage":   coverage,
        "mean_width": mean_width,
        "errors":     errors,
        "widths":     widths,
        "covered":    covered,
    }


def run_mc_dropout(
    model: MolecularSurrogate,
    cal_R: torch.Tensor,
    cal_F: torch.Tensor,
    test_R: torch.Tensor,
    test_F: torch.Tensor,
    alpha: float,
    molecule: str,
    n_mc_samples: int = 30,
) -> dict:
    """
    Run MC dropout baseline and return results dict matching the
    shared storage spec from docs/data_storage_spec.md.
    """
    # n_sigma chosen to match alpha target:
    #   alpha=0.05 -> 95% -> 1.96 sigma
    #   alpha=0.10 -> 90% -> 1.645 sigma
    from scipy import stats as sp_stats
    n_sigma = sp_stats.norm.ppf(1 - alpha / 2)

    print(f"\n--- MC Dropout baseline (alpha={alpha}, n_sigma={n_sigma:.3f}, n_mc={n_mc_samples}) ---")
    t0 = time.time()

    # Cal set: used to report n_cal in results (MC dropout has no calibration step)
    cal_mean, cal_std = mc_dropout_predict(model, cal_R, n_samples=n_mc_samples)

    # Test set
    test_mean, test_std = mc_dropout_predict(model, test_R, n_samples=n_mc_samples)
    runtime = time.time() - t0

    eval_r = evaluate_mc_dropout(test_mean, test_std, test_F, n_sigma=n_sigma)

    print(f"  Target coverage : {1-alpha:.1%}")
    print(f"  Empirical coverage: {eval_r['coverage']:.1%}")
    print(f"  Gap             : {eval_r['coverage'] - (1-alpha):+.1%}")
    print(f"  Mean width      : {eval_r['mean_width']:.6f}")
    print(f"  Runtime         : {runtime:.1f}s")

    results = {
        "surrogate":      "molecular",
        "method":         "mc_dropout",
        "split":          "indist",
        "molecule":       molecule,
        "alpha":          alpha,
        "coverage":       eval_r["coverage"],
        "target_coverage": 1 - alpha,
        "coverage_gap":   eval_r["coverage"] - (1 - alpha),
        "mean_width":     eval_r["mean_width"],
        "threshold":      None,    # MC dropout has no conformal threshold
        "n_cal":          len(cal_R),
        "n_test":         len(test_R),
        "predictions":    test_mean,
        "ground_truth":   test_F,
        "scores":         eval_r["errors"],
        "n_mc_samples":   n_mc_samples,
        "n_sigma":        n_sigma,
    }

    return results


# ============================================================
# Comparison table
# ============================================================

def print_comparison(conformal_results: list[dict], dropout_results: list[dict]):
    """
    Print a side-by-side comparison table of conformal vs MC dropout,
    matching the format Vasilisa needs for the aggregation table.
    """
    print("\n" + "=" * 75)
    print("COMPARISON: Conformal Prediction vs MC Dropout — Molecular (MD17)")
    print("=" * 75)
    print(f"{'Molecule':<16} {'Method':<14} {'Alpha':<8} {'Target':<10} {'Coverage':<12} {'Gap':<10} {'Width'}")
    print("-" * 75)

    all_results = [(r, "conformal") for r in conformal_results] + \
                  [(r, "mc_dropout") for r in dropout_results]

    # Sort by molecule then method
    all_results.sort(key=lambda x: (x[0]["molecule"], x[0]["alpha"], x[1]))

    for r, method in all_results:
        gap_str = f"{r['coverage_gap']:+.1%}"
        flag    = " ✓" if abs(r["coverage_gap"]) <= 0.02 else " ✗ (>2pp)"
        print(
            f"{r['molecule']:<16} {method:<14} {r['alpha']:<8.2f} "
            f"{r['target_coverage']:<10.1%} {r['coverage']:<12.1%} "
            f"{gap_str:<10} {r['mean_width']:.6f}{flag}"
        )

    print("=" * 75)
    print("✓ = within 2 percentage points of target (pass)")
    print("✗ = coverage gap > 2pp (flag for Vasilisa's diagnosis note)")


# ============================================================
# Save results
# ============================================================

def save_results(results: dict, out_dir: str = "results/molecular"):
    """Save results dict to .pt file per data_storage_spec.md."""
    os.makedirs(out_dir, exist_ok=True)
    molecule = results["molecule"]
    method   = results["method"]
    split    = results["split"]
    fname    = f"{molecule}_{method}_{split}.pt"
    path     = os.path.join(out_dir, fname)
    torch.save(results, path)
    print(f"  Saved -> {path}")


# ============================================================
# Main
# ============================================================

def run_molecule(molecule: str, alphas: list[float], n_mc_samples: int):
    """Full Week 2 pipeline for one molecule."""
    print(f"\n{'='*60}")
    print(f"Running Week 2 pipeline: {molecule.upper()}")
    print(f"{'='*60}")

    # 1. Load data
    train_R, cal_R, test_R, train_F, cal_F, test_F = load_md17(molecule)

    n_atoms = train_R.shape[1]

    # 2. Train surrogate
    print(f"\nTraining MLP surrogate ({n_atoms} atoms)...")
    model = MolecularSurrogate(n_atoms=n_atoms, hidden=256, dropout_p=0.1)
    model = train_surrogate(model, train_R, train_F, epochs=30)

    conformal_results_all = []
    dropout_results_all   = []

    for alpha in alphas:
        # 3. Conformal prediction
        conf_r = run_conformal(model, cal_R, cal_F, test_R, test_F, alpha, molecule)
        save_results(conf_r)
        conformal_results_all.append(conf_r)

        # 4. MC dropout baseline
        drop_r = run_mc_dropout(model, cal_R, cal_F, test_R, test_F, alpha, molecule, n_mc_samples)
        save_results(drop_r)
        dropout_results_all.append(drop_r)

    return conformal_results_all, dropout_results_all


def main():
    parser = argparse.ArgumentParser(description="Week 2 Molecular Conformal + MC Dropout")
    parser.add_argument(
        "--molecule", type=str, default="aspirin",
        choices=list(MOLECULE_FILES.keys()),
        help="Which MD17 molecule to run (default: aspirin)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run on all four MD17 molecules"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Single miscoverage level (default: runs both 0.10 and 0.05)"
    )
    parser.add_argument(
        "--n-mc-samples", type=int, default=30,
        help="Number of MC dropout forward passes (default: 30)"
    )
    args = parser.parse_args()

    alphas = [args.alpha] if args.alpha is not None else [0.10, 0.05]
    molecules = list(MOLECULE_FILES.keys()) if args.all else [args.molecule]

    all_conformal = []
    all_dropout   = []

    for mol in molecules:
        c, d = run_molecule(mol, alphas, args.n_mc_samples)
        all_conformal.extend(c)
        all_dropout.extend(d)

    # Print final comparison table
    print_comparison(all_conformal, all_dropout)

    print("\nWeek 2 complete. Results saved to results/molecular/")
    print("Share results with Vasilisa for the aggregation table.")


if __name__ == "__main__":
    main()
