"""
Minimal PK smoke test:
1. Generate a small synthetic PK dataset in-memory.
2. Train PK neural ODE for 1 epoch.
3. Verify deterministic split sizes and seed metadata.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_ode_pk import generate_population_data, train_surrogate


def main():
    n_patients = 64
    split_seed = 42

    data = generate_population_data(n_patients=n_patients, seed=123)
    _, splits = train_surrogate(
        data,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        split_seed=split_seed,
    )

    n_train = len(splits["train"])
    n_cal = len(splits["cal"])
    n_test = len(splits["test"])

    assert n_train == int(0.7 * n_patients), f"Unexpected train size: {n_train}"
    assert n_cal == int(0.15 * n_patients), f"Unexpected cal size: {n_cal}"
    assert n_test == n_patients - n_train - n_cal, f"Unexpected test size: {n_test}"
    assert splits["split_seed"] == split_seed, "Split seed metadata mismatch"

    print("PK smoke test passed.")
    print(f"Split sizes: train={n_train}, cal={n_cal}, test={n_test}")


if __name__ == "__main__":
    main()
