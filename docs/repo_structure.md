# Repository Structure, PR Conventions, and Integration Plan

## Folder Layout

```
Spring-2026-Algoverse/
│
├── src/                          # Core library (importable)
│   ├── neural_ode_pk.py          # PK ground-truth + surrogate model + training
│   ├── conformal.py              # Split conformal prediction framework
│   ├── nonconformity_scores.py   # Nonconformity score interfaces + implementations
│   └── baselines/                # Baseline UQ methods (NEW, Week 2)
│       ├── __init__.py
│       ├── deep_ensemble_pk.py   # Deep ensemble training + evaluation for PK
│       └── eval_harness.py       # Coverage, width, logging utilities (all methods)
│
├── experiments/                  # Runnable experiment scripts (gittracked)
│   └── run_pk_baselines.py       # [to be added] orchestrates all three baselines + conformal
│
├── notebooks/                    # Jupyter notebooks (outputs cleared before commit)
│
├── scripts/                      # One-off data download / environment setup
│   ├── download_era5.py
│   ├── setup_weatherbench2.py
│   ├── prepare_era5_training.py
│   ├── train_weather_surrogate.py
│   └── test_graphcast.py
│
├── docs/                         # Design docs (gittracked)
│   ├── uncertainty_methods_survey.md   # One-page baseline comparison (NEW)
│   ├── repo_structure.md               # This file (NEW)
│   ├── split_policy.md                 # Canonical 70/15/15 split rules
│   ├── data_storage_spec.md            # Data/result format spec
│   ├── compute_estimates.md            # AWS cost estimates per surrogate
│   └── SETUP_GUIDE.md                  # Environment setup guide
│
├── data/                         # Datasets — GITIGNORED, download separately
│   ├── pk/                       # Synthetic PK trajectories (generated locally)
│   ├── era5/                     # ERA5 reanalysis (Copernicus CDS)
│   ├── md17/                     # MD17/MD22 molecular dynamics (sgdml.org)
│   └── weatherbench2/            # WeatherBench 2 evaluation data
│
├── models/                       # Trained model checkpoints — GITIGNORED
│
├── results/                      # Evaluation result JSONs — GITIGNORED
│   └── deep_ensemble_pk.json     # (example, not committed)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## PR Conventions

### Branch naming

```
<type>/<short-description>
```

| Type | Use for |
|---|---|
| `feat/` | New functionality (new surrogate, new score, new method) |
| `baseline/` | Baseline UQ method implementations |
| `exp/` | Experiment runs and result analysis |
| `fix/` | Bug fixes |
| `docs/` | Documentation only changes |
| `refactor/` | Code restructuring with no behaviour change |

Examples: `baseline/deep-ensemble-pk`, `feat/conformal-pk-pipeline`, `exp/coverage-sweep-alpha`.

### PR checklist (author)

Before marking Ready for Review:

- [ ] All new code lives in `src/` or `experiments/` (not `scripts/` or root)
- [ ] No data files, model checkpoints, or result JSONs committed (`data/`, `models/`, `results/` are gitignored)
- [ ] Jupyter notebooks committed with **cleared outputs** (`Cell → Clear All Outputs`)
- [ ] `split_policy.md` seed conventions followed (default seed 42, documented if overridden)
- [ ] New public functions have a one-line docstring at minimum
- [ ] `results/` JSON produced by any new experiment script is attached to the PR description (paste inline or link Gist)

### PR description template

```markdown
## What this PR does
<!-- 1-3 sentences -->

## Key files changed
<!-- bullet list of files with one-line description of change -->

## How to test
<!-- minimal commands to reproduce the key output -->

## Results (if experiment)
<!-- paste or link the result JSON / markdown table from eval_harness.compare_methods() -->

## Reviewer focus
<!-- what you want reviewers to look at closely -->
```

### Review conventions

- Minimum **1 reviewer approval** before merge.
- Reviewer should run the "How to test" commands and verify output matches.
- Do not merge if `empirical_coverage < target_coverage - 0.05` on the test split without a documented explanation.
- Force-push to feature branches is allowed; force-push to `main` is **never** allowed.

---

## Integration Plan

The project integrates three components: surrogate training, baseline UQ, and conformal wrapping.

### Phase 1 (Weeks 1–2) — PK surrogate as testbed ✓

1. Generate synthetic PK population data (`neural_ode_pk.py --generate-data`).
2. Train baseline surrogates — **done**: single surrogate in `neural_ode_pk.py`.
3. Implement and evaluate baseline UQ: deep ensemble, MC dropout, Bayesian VI — **done (ensemble)**: `src/baselines/deep_ensemble_pk.py`.
4. Implement split conformal wrapper — **done**: `src/conformal.py`.
5. Run `eval_harness.compare_methods()` over all four methods on the same test split.

### Phase 2 (Weeks 3–4) — Conformal improvement loop

1. Show that conformal wrapping closes the coverage gap left by baselines.
2. Add physics-constrained nonconformity scores (`WeightedFunctionalScore`) and compare against scalar `AbsoluteError`.
3. Wire up `ShiftDetector` for OOD detection.
4. Produce a results table (`experiments/run_pk_baselines.py`) covering: deep ensemble, MC dropout, Bayesian VI, conformal (all α ∈ {0.05, 0.10, 0.20}).

### Phase 3 (Weeks 5–6) — Transfer to weather and molecular

| Surrogate | Owner | Data source | Status |
|---|---|---|---|
| PK neural ODE | shared | Synthetic | Done |
| GraphCast small (weather) | Prajesh | ERA5 CDS | In progress |
| MACE (molecular) | Vijay | MD17/MD22 sgdml.org | In progress |

Conformal framework in `src/conformal.py` is surrogate-agnostic; transfer requires only:
1. Defining a domain-appropriate nonconformity score (`SupNormScore` for weather, `TrajectoryNormScore` for molecular).
2. Producing `(cal_predictions, cal_ground_truth)` and `(test_predictions, test_ground_truth)` tensors.
3. Calling `SplitConformal.calibrate()` + `evaluate()`.

### Phase 4 (Week 7) — Paper / report

- Collect all result JSONs from `results/`.
- Use `eval_harness.compare_methods(result_paths)` to generate the final comparison table.
- Write up discussion of coverage gaps and conformal advantage.

### Data flow diagram

```
neural_ode_pk.py          baselines/deep_ensemble_pk.py
  generate_population_data  →  DeepEnsemblePK.train()
                               DeepEnsemblePK.predict()
                                    ↓
                           eval_harness.compute_empirical_coverage()
                           eval_harness.compute_interval_widths()
                           eval_harness.log_results()  →  results/<method>_pk.json

conformal.py
  SplitConformal.calibrate()
  SplitConformal.evaluate()   →  results/conformal_pk.json
                                    ↓
                           eval_harness.compare_methods([...])  →  final table
```
