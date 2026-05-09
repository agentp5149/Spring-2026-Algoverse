# Train/Calibration/Test Split Policy

This document defines a single in-distribution split policy for all three surrogates:

- Weather surrogate (`weather`)
- Molecular surrogate (`molecular`)
- Pharmacokinetic surrogate (`pk`)

## Split Ratios

Use a fixed `70/15/15` split by sample count:

- `train`: 70% (fit surrogate parameters)
- `cal`: 15% (fit conformal threshold only)
- `test`: 15% (final in-distribution reporting only)

For total sample count `N`:

- `n_train = int(0.70 * N)`
- `n_cal = int(0.15 * N)`
- `n_test = N - n_train - n_cal`

This matches existing weather and PK preprocessing/training code and is now the project-wide default.

## Example Sizes

| Surrogate | Example `N` | Train | Cal | Test |
|---|---:|---:|---:|---:|
| PK (`pk`) | 500 trajectories | 350 | 75 | 75 |
| Weather (`weather`) | 10,000 forecast pairs | 7,000 | 1,500 | 1,500 |
| Molecular (`molecular`) | 20,000 structures | 14,000 | 3,000 | 3,000 |

The weather and molecular totals depend on what is downloaded/constructed; apply the same formula.

## Rationale

1. `70%` training provides enough data for model fitting across all three domains.
2. `15%` calibration is large enough for stable conformal quantiles at common targets (e.g., 90%/95% coverage).
3. `15%` testing gives an unbiased holdout for final coverage and width metrics.
4. A dedicated calibration split prevents leakage from model fitting into uncertainty calibration.

## Reproducibility Rules

Use deterministic shuffling with a fixed seed before slicing:

- Default seed: `42`
- Persist split indices (or seed + index order) with artifacts where practical.

Current implementation status:

- Weather preprocessing: deterministic (`numpy.RandomState(42)`).
- PK training split: deterministic via a fixed split seed (`42` by default).
- Molecular: implement the same deterministic seed policy in the molecular data pipeline.

## Data Leakage Guardrails

1. Never mix `cal` into surrogate training.
2. Never tune hyperparameters on `test`.
3. Reserve OOD datasets for separate OOD evaluation (`data/ood/*`), not for in-distribution split counts.
4. If doing temporal weather evaluation, enforce chronological split variants and document dates explicitly.
