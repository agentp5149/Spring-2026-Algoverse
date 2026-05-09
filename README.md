# Conformal Trust Scores for Scientific ML Surrogates

Physics-constrained conformal prediction for neural weather models, molecular dynamics surrogates, and pharmacokinetic simulators.

## Project Structure

```
conformal-trust-scores/
├── src/                    # Core library code
│   ├── conformal.py        # Conformal prediction framework
│   ├── neural_ode_pk.py    # PK surrogate (neural ODE)
│   ├── nonconformity_scores.py # Shared nonconformity score interfaces/stubs
│   └── physics_projection.py   # Constraint projection (future)
├── scripts/                # Data download and setup scripts
│   ├── download_era5.py    # ERA5 reanalysis download
│   ├── setup_weatherbench2.py  # WeatherBench 2 setup
│   └── test_graphcast.py   # GraphCast variant smoke test
├── notebooks/              # Experiments and analysis
├── data/                   # Downloaded datasets (gitignored)
├── docs/                   # Documentation
│   ├── split_policy.md     # Canonical train/cal/test policy
│   └── data_storage_spec.md # Data/result format spec
├── requirements.txt        # Python dependencies
└── README.md
```

## Environment Setup

### 1. Clone and create virtual environment (Alternatively, run on collab using code in part 2)

```bash
git clone <repo-url>
cd conformal-trust-scores
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
pip install --upgrade pip
```

### 2. Install dependencies

For CPU only (setup verification):
```bash
pip install -r requirements.txt
```

For GPU (actual experiments, adjust CUDA version as needed):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torchdiffeq numpy scipy pandas matplotlib seaborn xarray netCDF4 cdsapi scikit-learn tqdm
```

### 3. Verify installation

```bash
python3 -c "
import torch
import torchdiffeq
import numpy as np
import scipy
import xarray
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('All packages OK')
"
```

You should see all packages listed with no import errors. CUDA will show False on CPU-only machines, which is fine for setup verification.

## Data Setup

### ERA5 Reanalysis Data

ERA5 is the ground truth dataset for the weather surrogate. It requires a free Copernicus Climate Data Store account.

**Step 1: Create account**
Go to https://cds.climate.copernicus.eu and register for a free account. After registering, go to your profile page and find your API key.

**Step 2: Configure credentials**
Create the file `~/.cdsapirc` with:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

**Step 3: Run download script**
```bash
python scripts/download_era5.py --variables geopotential temperature --pressure-levels 500 850 --years 2018 2019 2020 --output data/era5/
```

This downloads a small subset for initial testing. The full dataset for experiments will be larger. See `scripts/download_era5.py` for all options.

**Expected output:** NetCDF files in `data/era5/` with shape `(time, level, lat, lon)` for each variable.

### WeatherBench 2

WeatherBench 2 provides standardized evaluation for ML weather models.

```bash
python scripts/setup_weatherbench2.py --output data/weatherbench2/
```

See the script for details. This clones the evaluation code and downloads a small test subset.

### GraphCast Variant

We use a small GraphCast variant for the weather surrogate. The script clones the repo and runs a smoke test.

```bash
python scripts/test_graphcast.py
```

### Molecular Dynamics Data (MD17/MD22)

Download from http://www.sgdml.org. Place files in `data/md17/`.

### Pharmacokinetics Data

Generated synthetically using our neural ODE script. No external download needed.

```bash
python src/neural_ode_pk.py --generate-data --n-patients 5000 --output data/pk/
```

## Quick Start

After setup, verify the full pipeline works on the PK surrogate (smallest and fastest):

```bash
# Generate synthetic PK data
python src/neural_ode_pk.py --generate-data --n-patients 1000 --output data/pk/

# Train a small neural ODE surrogate
python src/neural_ode_pk.py --train --data data/pk/ --epochs 100 --split-seed 42 --output models/pk_surrogate.pt

# Run basic conformal prediction on the surrogate
python src/conformal.py --surrogate models/pk_surrogate.pt --calibration data/pk/cal.pt --test data/pk/test.pt --alpha 0.05
```

## GPU Requirements

- PK surrogate: runs fine on CPU, ~10 min training
- Weather surrogate: needs 1 GPU (A100 recommended), several hours training
- Molecular surrogate: needs 1 GPU, depends on model choice

## Team Notes

- All data goes in `data/` which is gitignored
- Models go in `models/` which is also gitignored
- Push notebooks with outputs cleared
- Document any issues in the GitHub Issues tab
- Split policy for all surrogates (weather/molecular/PK): see `docs/split_policy.md`

## Split Policy

All in-distribution training follows a `70/15/15` split:

- `70%` train
- `15%` calibration
- `15%` test

For reproducibility, use deterministic shuffle seeds (default `42` in current PK training code via `--split-seed`).
