# Data Storage and Naming Format Spec
**Author:** Vijay
**Week:** 1

## Directory Structure

```
data/
├── md17/                        # Raw MD17 molecular dynamics data
│   ├── md17_aspirin.npz
│   ├── md17_ethanol.npz
│   ├── md17_uracil.npz
│   └── md17_malonaldehyde.npz
├── md22/                        # Raw MD22 molecular dynamics data
│   ├── md22_DHA.npz
│   └── md22_stachyose.npz
├── era5/                        # ERA5 weather reanalysis data
│   └── (NetCDF files from Copernicus CDS)
├── weatherbench2/               # WeatherBench 2 evaluation suite
├── pk/                          # Synthetic pharmacokinetics data
│   ├── train.pt
│   ├── cal.pt
│   └── test.pt
└── ood/                         # Out-of-distribution test sets
    ├── weather_ood.pt           # Extreme weather events
    ├── molecular_ood.npz        # OC20 OOD molecular configs
    └── pk_ood.pt                # Rare metabolizer phenotypes

models/
├── weather_surrogate.pt         # Trained neural weather surrogate
├── molecular_surrogate.pt       # Trained MACE/NequIP surrogate
└── pk_surrogate.pt              # Trained neural ODE PK surrogate

results/
├── weather/
│   ├── conformal_indist.pt      # In-distribution conformal results
│   ├── conformal_ood.pt         # OOD conformal results
│   ├── ensemble_indist.pt       # Deep ensemble baseline results
│   ├── ensemble_ood.pt
│   ├── mc_dropout_indist.pt
│   ├── mc_dropout_ood.pt
│   ├── bayesian_vi_indist.pt
│   └── bayesian_vi_ood.pt
├── molecular/
│   └── (same structure as weather/)
└── pk/
    └── (same structure as weather/)
```

## Naming Conventions

- Lowercase with underscores for all filenames
- Always include surrogate name and method name in result filenames
- Always include indist or ood to distinguish in-distribution vs out-of-distribution
- No spaces or special characters in filenames

### Surrogate Names
| Surrogate | Short Name |
|-----------|------------|
| Neural weather model | `weather` |
| MACE interatomic potential | `molecular` |
| Neural ODE PK model | `pk` |

### Method Names
| Method | Short Name |
|--------|------------|
| Conformal prediction (ours) | `conformal` |
| Deep ensembles | `ensemble` |
| Monte Carlo dropout | `mc_dropout` |
| Bayesian variational inference | `bayesian_vi` |

### Result Filename Format
```
{surrogate}_{method}_{split}.pt
```
Examples:
- `weather_conformal_indist.pt`
- `molecular_ensemble_ood.pt`
- `pk_mc_dropout_indist.pt`

## File Formats

| Content | Format | Notes |
|---------|--------|-------|
| Raw molecular data | `.npz` | As downloaded from sgdml.org |
| Raw weather data | `.nc` (NetCDF) | As downloaded from Copernicus CDS |
| Processed PyTorch tensors | `.pt` | Use `torch.save()` / `torch.load()` |
| Trained model weights | `.pt` | Use `torch.save(model.state_dict())` |
| Evaluation metrics | `.pt` | Save as Python dict via `torch.save()` |

## Results Dictionary Format

All result files saved as `.pt` should be Python dicts with the following keys:

```python
{
    "surrogate":        str,    # e.g. "weather"
    "method":           str,    # e.g. "conformal"
    "split":            str,    # "indist" or "ood"
    "alpha":            float,  # miscoverage level (e.g. 0.05 for 95% coverage)
    "coverage":         float,  # empirical coverage (e.g. 0.943)
    "target_coverage":  float,  # nominal coverage (e.g. 0.95)
    "coverage_gap":     float,  # empirical minus target
    "mean_width":       float,  # mean interval width
    "threshold":        float,  # conformal threshold (conformal only)
    "n_cal":            int,    # number of calibration samples
    "n_test":           int,    # number of test samples
    "predictions":      Tensor, # model predictions on test set
    "ground_truth":     Tensor, # true simulator outputs on test set
    "scores":           Tensor, # per-sample nonconformity scores
}
```

Save example:
```python
torch.save(results_dict, "results/weather/weather_conformal_indist.pt")
```

Load example:
```python
results = torch.load("results/weather/weather_conformal_indist.pt")
print(results["coverage"])
```

## Notes

- data/, models/, and results/ are gitignored. Do not push them to GitHub.
- All paths in scripts should be relative to the repo root.
- If you add a new surrogate or method, update this doc and post in Slack.
