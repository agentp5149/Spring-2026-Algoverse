# Setup Guide: All 5 Tasks Step by Step

This guide walks you through every task assigned to Prajesh for the Conformal Trust Scores project. Each task has exact commands, expected outputs, and what to do when things go wrong.

---

## Task 1: Set Up Python Environment

### What this task accomplishes

You are building the software environment that every other task depends on. By the end you will have a working Python installation with PyTorch for building neural networks, torchdiffeq for solving differential equations with neural networks, and all the scientific computing libraries needed for the project.

### Step by step

Open a terminal and navigate to wherever you keep your research projects.

```bash
cd ~/research    # or wherever you work
```

Clone the project repository. If you have not pushed it to GitHub yet, just create the directory and untar the package I gave you.

```bash
mkdir conformal-trust-scores
cd conformal-trust-scores
# If you have the tar file:
tar -xzf conformal-trust-scores.tar.gz --strip-components=1
```

Create a Python virtual environment. This keeps our project's packages separate from your system Python so nothing conflicts.

```bash
python3 -m venv venv
```

Activate the virtual environment. You need to do this every time you open a new terminal to work on the project.

```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt. If you do not see it, the activation did not work. On Windows the command is `venv\Scripts\activate` instead.

Upgrade pip so package installations go smoothly.

```bash
pip install --upgrade pip
```

Now install PyTorch. This is the most important step and it depends on whether your machine has an NVIDIA GPU. Check first.

```bash
nvidia-smi
```

If that command works and shows you a GPU with CUDA version, install the GPU version of PyTorch. Match the CUDA version shown by nvidia-smi.

```bash
# If nvidia-smi shows CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# If nvidia-smi shows CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# If nvidia-smi shows CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If nvidia-smi does not work or you do not have a GPU, install the CPU version. Everything will still work but training will be slower.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Now install all the other packages we need.

```bash
pip install torchdiffeq numpy scipy pandas matplotlib seaborn xarray netCDF4 cdsapi scikit-learn tqdm
```

Here is what each package does so you know why we need it.

torchdiffeq is the neural ODE solver that lets us build the pharmacokinetics surrogate. numpy and scipy are standard scientific computing libraries for math and optimization. pandas is for tabular data manipulation. matplotlib and seaborn are for making plots and figures. xarray is for working with multi-dimensional labeled data like weather fields, which come as NetCDF files with dimensions like time, latitude, longitude, and pressure level. netCDF4 is the low level library that xarray uses to read and write those NetCDF files. cdsapi is the client for downloading ERA5 weather data from the Copernicus Climate Data Store. scikit-learn gives us basic ML tools we will need for some of the analysis. tqdm gives us progress bars so long computations do not feel like they froze.

Save the exact package versions so your teammates can reproduce your setup.

```bash
pip freeze > requirements.txt
```

Now verify everything works.

```bash
python3 -c "
import torch
import torchdiffeq
import numpy as np
import scipy
import xarray
import cdsapi

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
print('torchdiffeq: OK')
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('xarray:', xarray.__version__)
print('cdsapi: OK')
print()
print('ALL PACKAGES VERIFIED SUCCESSFULLY')
"
```

You should see all packages listed with no errors. If CUDA shows True and lists your GPU name, you are set for fast training. If CUDA shows False, everything still works but you may want to install the GPU version before running the bigger experiments.

Also run the conformal prediction demo to make sure the core framework works.

```bash
python3 src/conformal.py --demo
```

You should see three demos run, one for scalar predictions, one for spatial fields, and one for trajectories. Each should report coverage within a few percentage points of the target. If all three say "Evaluation results" with reasonable coverage numbers, the framework is working.

Finally run the GraphCast smoke test to verify the weather surrogate pipeline.

```bash
python3 scripts/test_graphcast.py
```

This builds a tiny neural network, trains it on fake weather data, and runs conformal prediction on the output. It should end with "ALL TESTS PASSED."

### What to put in the repo README

Document the exact versions you installed. Copy the output of `pip freeze` and the verification script output. Note whether you have GPU access and which CUDA version. This way when your teammates set up their environments they can match yours exactly.

### Troubleshooting

If PyTorch installation fails with an error about wheels not being found, you are probably using the wrong CUDA version URL. Run `nvidia-smi` again and check the CUDA version in the top right corner.

If torchdiffeq installation fails, try `pip install torchdiffeq --no-deps` and then install its dependencies manually. It depends on torch and scipy which you already have.

If xarray gives import errors about missing backends, make sure netCDF4 installed correctly. Run `python3 -c "import netCDF4; print(netCDF4.__version__)"` to check.

---

## Task 2: Download ERA5 Reanalysis Data

### What this task accomplishes

ERA5 is the ground truth dataset for our weather surrogate. It contains the state of the Earth's atmosphere at every point on a global grid, recorded every hour going back to 1940, computed by the European Centre for Medium-Range Weather Forecasts. We use it as both the training data for our small weather model and the ground truth that our conformal prediction calibrates against.

### Step by step

First you need a Copernicus Climate Data Store account. Go to this URL in your browser.

```
https://cds.climate.copernicus.eu
```

Click "Register" and create a free account. Use your university email. You will get a confirmation email. Click the link to verify.

Once logged in, you need to accept the ERA5 license. Go to this page.

```
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
```

Scroll down to the "Terms of use" section and click "Accept." You only have to do this once.

Now get your API credentials. Go to your profile page by clicking your name in the top right corner, then "Your profile." You will see a section labeled "API key" that shows two pieces of information: your user ID (a number) and your API key (a long string). You need both.

Create the credentials file on your machine. Open a text editor and create a file at `~/.cdsapirc` with exactly these two lines, replacing the placeholder with your actual credentials.

```bash
# On Mac/Linux
nano ~/.cdsapirc
```

Put this in the file:

```
url: https://cds.climate.copernicus.eu/api
key: YOUR_UID:YOUR_API_KEY
```

Replace `YOUR_UID:YOUR_API_KEY` with the values from your profile page. It should look something like `12345:abcdef12-3456-7890-abcd-ef1234567890`. Save and close.

Make sure only you can read this file since it contains your credentials.

```bash
chmod 600 ~/.cdsapirc
```

Now verify the credentials work before trying a big download.

```bash
python3 -c "
import cdsapi
client = cdsapi.Client()
print('CDS API client connected successfully.')
print('Server:', client.url)
"
```

If this prints the server URL without errors, your credentials are working. If you get an authentication error, double check the `.cdsapirc` file. The most common mistake is having an extra space or newline in the key.

Now run the download. Start small so you can verify everything works before downloading a lot of data.

```bash
python3 scripts/download_era5.py \
    --variables geopotential temperature \
    --pressure-levels 500 850 \
    --years 2020 \
    --months 1 2 3 \
    --output data/era5/
```

This downloads geopotential and temperature at two pressure levels for the first three months of 2020. It is a small request that should finish in 10 to 30 minutes depending on the CDS queue. The CDS processes requests asynchronously, so the script will show "Requesting..." and then wait. Do not kill it, it is waiting for the server.

Once the download finishes, verify the files look right.

```bash
python3 scripts/download_era5.py --verify-only --output data/era5/
```

This opens each NetCDF file and prints its dimensions and variables. You should see something like `era5_geopotential_2020.nc: variables=['z'], dims={'time': X, 'level': 2, 'latitude': 721, 'longitude': 1440}`.

You can also inspect the data manually in Python to make sure it looks reasonable.

```bash
python3 -c "
import xarray as xr
ds = xr.open_dataset('data/era5/era5_geopotential_2020.nc')
print(ds)
print()
print('First few values:')
print(ds['z'][0, 0, :5, :5].values)
print()
print('Value range:')
z = ds['z'].values
print(f'  Min: {z.min():.0f}')
print(f'  Max: {z.max():.0f}')
print(f'  Mean: {z.mean():.0f}')
"
```

Geopotential at 500 hPa should have values roughly in the range of 48000 to 60000 m²/s². If you see values in that range, the data is correct.

Once the small download checks out, run the full download you will need for experiments.

```bash
python3 scripts/download_era5.py \
    --variables geopotential temperature u_component_of_wind v_component_of_wind \
    --pressure-levels 500 700 850 1000 \
    --years 2016 2017 2018 2019 2020 \
    --output data/era5/
```

This is a bigger request and may take several hours. Kick it off before you go to bed or before a long meeting. The script downloads one variable per year as a separate file, so if it gets interrupted you can restart and it will skip files that already exist.

### What to document for the team

Write down the exact variables and years you downloaded, the file sizes, and any issues with the CDS queue. If the CDS was slow or errored, note that so the team knows it is a CDS problem and not a code problem. Also note the total disk space used so teammates know how much space to reserve.

### Troubleshooting

If the CDS gives you a "request too large" error, break the request into smaller pieces. Download one month at a time instead of a full year.

If the download seems stuck, it is probably waiting in the CDS queue. Check the CDS website under "Your requests" to see the queue position.

If you get a "license not accepted" error, go back to the ERA5 dataset page and explicitly click "Accept" on the terms of use.

---

## Task 3: Clone and Test a GraphCast Variant

### What this task accomplishes

We need a neural weather model to serve as one of our three scientific surrogates. The official GraphCast from DeepMind is powerful but it runs on JAX and needs significant resources. For our project we are better off training our own small weather model because we need full control over the model's internals for conformal calibration, and a smaller model lets us iterate faster during development.

This task verifies that our small weather model architecture works end to end, from taking in atmospheric fields to producing predictions and running conformal calibration on those predictions.

### Step by step

The smoke test script is already in the project. Run it.

```bash
cd conformal-trust-scores
source venv/bin/activate
python3 scripts/test_graphcast.py
```

This does four things automatically. It checks whether any pre-built weather model like GraphCast or Aurora is installed, which it probably is not and that is fine. It then builds a tiny UNet style convolutional network with about 10,000 parameters. It trains this network for 50 epochs on synthetic atmospheric fields, which takes about 10 seconds on a CPU. Finally it runs conformal prediction on the test set and checks whether coverage is in a reasonable range.

You should see output ending with "ALL TESTS PASSED." If it does, the pipeline is verified.

Now we need to check whether we want to use an existing open-weight weather model or train our own from scratch. Let me walk through both options.

Option A is to use an existing model. The most accessible one in PyTorch is Microsoft Aurora.

```bash
pip install microsoft-aurora
```

If that works, you can test it with:

```bash
python3 -c "
import aurora
print('Aurora installed:', aurora.__version__)
"
```

Aurora is nice because it is PyTorch native, relatively small, and has published checkpoints. But we still need to verify we can hook into its internals for conformal calibration.

Option B is to train our own model, which is what I recommend for starting out. The advantage is total control. Here is how to build a proper small weather surrogate once you have ERA5 data.

Create a new file called `src/weather_surrogate.py`. The model takes the atmospheric state at time t, meaning the values of geopotential, temperature, and wind at every grid point, and predicts the state at time t plus 6 hours. The architecture is a UNet, which is a convolutional network with an encoder that compresses the spatial field and a decoder that expands it back, with skip connections so it can learn residual corrections.

```bash
python3 -c "
import torch
import torch.nn as nn

# Verify we can build a reasonable-sized model
class WeatherUNet(nn.Module):
    def __init__(self, in_channels=8, hidden=64):
        super().__init__()
        # 8 channels = 2 variables x 4 pressure levels
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(hidden, hidden*2, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(hidden*2, hidden*4, 3, stride=2, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(hidden*4, hidden*2, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(hidden*4, hidden, 4, stride=2, padding=1), nn.ReLU())
        self.dec1 = nn.Conv2d(hidden*2, in_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return x + out  # residual prediction

model = WeatherUNet()
n_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {n_params:,}')

# Test with a fake ERA5-shaped input
# 4 pressure levels x 2 variables = 8 channels, on a coarse 64x128 grid
x = torch.randn(4, 8, 64, 128)
y = model(x)
print(f'Input shape:  {x.shape}')
print(f'Output shape: {y.shape}')
print('Shapes match:', x.shape == y.shape)
"
```

This should print the parameter count, confirm input and output shapes match, and show no errors. The model has roughly 500K to 1M parameters depending on the hidden dimension, which is small enough to train on a single GPU in a few hours on ERA5 data.

### What to log for the team

Note which option you went with, A or B. If you installed Aurora, note whether it runs inference successfully. If you are training your own model, note the architecture details and parameter count. Also note whether GPU inference works by running `torch.cuda.is_available()` at the start of the test.

### Troubleshooting

If the smoke test fails on the import of torch, your virtual environment is not activated. Run `source venv/bin/activate` again.

If training takes unusually long even on the synthetic data, something may be wrong with your PyTorch installation. Check `torch.backends.mkl.is_available()` which should be True for CPU performance.

---

## Task 4: Download and Verify WeatherBench 2

### What this task accomplishes

WeatherBench 2 is a standardized evaluation suite built by Google Research for comparing ML weather models. We use it for two things. First, it provides pre-processed ERA5 data on standard grids that are already aligned with how ML weather models expect their inputs, which saves us data preprocessing work. Second, it provides standard metrics like RMSE, anomaly correlation coefficient, and CRPS that the weather forecasting community expects, so our results are directly comparable to published work.

### Step by step

Try installing the package first.

```bash
source venv/bin/activate
pip install weatherbench2
```

If that works, verify it:

```bash
python3 -c "
import weatherbench2
print('WeatherBench 2 installed successfully')
"
```

If pip install fails, clone it from GitHub instead.

```bash
git clone https://github.com/google-research/weatherbench2.git /tmp/weatherbench2
pip install -e /tmp/weatherbench2
```

If both fail, that is OK for now. WeatherBench 2 is primarily an evaluation library that we need later, not immediately. You can work with raw ERA5 data for the initial surrogate training.

Now set up access to the WeatherBench 2 data. The data is hosted on Google Cloud Storage. You need the `gcsfs` package to access it from Python.

```bash
pip install gcsfs
```

Test that you can see the data.

```bash
python3 -c "
import xarray as xr

# This opens a cloud-hosted zarr store without downloading everything
print('Opening WeatherBench 2 ERA5 data from Google Cloud...')
try:
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023-6h-1440x721.zarr',
                       chunks=None)
    print('Success!')
    print(f'Variables: {list(ds.data_vars)[:10]}...')
    print(f'Time range: {ds.time.values[0]} to {ds.time.values[-1]}')
    print(f'Spatial grid: {ds.dims}')
except Exception as e:
    print(f'Could not access: {e}')
    print('This is OK. You can use your local ERA5 download instead.')
    print('GCS access sometimes needs authentication or network configuration.')
"
```

If this works, you can load data lazily from the cloud without downloading the full dataset. If it does not work, that is fine because you already have ERA5 data from Task 2.

For local access, you can also download a small subset using gsutil if you have the Google Cloud SDK installed.

```bash
# Install Google Cloud SDK (if not already installed)
# Mac: brew install google-cloud-sdk
# Linux: curl https://sdk.cloud.google.com | bash

# Download a small test dataset
gsutil -m cp -r gs://weatherbench2/datasets/era5-derived/1959-2023-6h-64x32_equiangular_conservative.zarr data/weatherbench2/
```

The 64x32 version is the coarsest resolution and is small enough to download quickly, about 2 GB. This is perfect for initial development and testing.

Run the setup script to generate documentation about data access.

```bash
python3 scripts/setup_weatherbench2.py --output data/weatherbench2/
```

### Verification checklist

After this task you should be able to answer yes to at least one of these: you can import weatherbench2 in Python, you can open a WeatherBench 2 zarr store either from the cloud or locally, or at minimum you have your local ERA5 data from Task 2 and know where the WeatherBench 2 evaluation code lives on GitHub. Any one of these is sufficient to proceed.

### Troubleshooting

If gcsfs fails to connect, it may be a network issue, especially on university networks that block outbound connections to cloud storage. Try from a different network or use a VPN. If nothing works, just use your local ERA5 data from Task 2 and come back to WeatherBench 2 later.

---

## Task 5: Train a Small Neural Weather Surrogate on ERA5

### What this task accomplishes

This is where you actually train the ML model that we will wrap with conformal prediction. You take the ERA5 data you downloaded in Task 2, build a small convolutional neural network, and train it to predict the next weather state from the current one. The model does not need to be as good as GraphCast. It just needs to be good enough to produce meaningful predictions that conformal prediction can calibrate.

### Prerequisites

You need to have completed Tasks 1, 2, and 3 before this one. You need the ERA5 data in `data/era5/`, a working Python environment, and a verified model architecture from the smoke test.

### Step by step

First, prepare the ERA5 data for training. The raw ERA5 files need to be converted into input-output pairs where the input is the atmospheric state at time t and the output is the state at time t plus 6 hours.

Create a data preparation script. Save this as `scripts/prepare_era5_training.py`.

```python
"""
Prepare ERA5 data for weather surrogate training.
Creates (input, target) pairs where input is state at time t
and target is state at time t+6h.
"""

import argparse
import os
import numpy as np
import xarray as xr
import torch
from pathlib import Path


def load_era5_variable(data_dir, variable, year):
    """Load a single ERA5 variable for a given year."""
    filepath = os.path.join(data_dir, f"era5_{variable}_{year}.nc")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    ds = xr.open_dataset(filepath)
    # Get the main data variable (name varies by file)
    var_name = [v for v in ds.data_vars if v not in ['time', 'level']][0]
    return ds[var_name]


def prepare_training_data(data_dir, variables, years, output_dir,
                          coarsen_factor=4, lead_time_hours=6):
    """
    Build training dataset from ERA5.

    Args:
        data_dir: directory with ERA5 NetCDF files
        variables: list of variable names
        years: list of years to include
        output_dir: where to save processed tensors
        coarsen_factor: spatial downsampling (4 = 180x360 grid from 721x1440)
        lead_time_hours: forecast horizon
    """
    os.makedirs(output_dir, exist_ok=True)

    all_inputs = []
    all_targets = []

    for year in years:
        print(f"Processing {year}...")
        channels = []

        for var in variables:
            try:
                data = load_era5_variable(data_dir, var, year)
            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

            # Coarsen spatially for faster training
            if coarsen_factor > 1:
                data = data.coarsen(
                    latitude=coarsen_factor, longitude=coarsen_factor,
                    boundary='trim'
                ).mean()

            values = data.values  # (time, level, lat, lon) or (time, lat, lon)

            if values.ndim == 3:
                # Single level: add level dimension
                values = values[:, np.newaxis, :, :]

            # Each pressure level becomes a separate channel
            n_times, n_levels, n_lat, n_lon = values.shape
            for lev in range(n_levels):
                channels.append(values[:, lev, :, :])

        if not channels:
            print(f"  No data for {year}, skipping")
            continue

        # Stack channels: (time, channels, lat, lon)
        stacked = np.stack(channels, axis=1)
        print(f"  Shape: {stacked.shape}")

        # Create input/target pairs with lead_time_hours offset
        # ERA5 is 6-hourly, so offset = lead_time_hours / 6
        offset = lead_time_hours // 6

        inputs = stacked[:-offset]
        targets = stacked[offset:]

        all_inputs.append(inputs)
        all_targets.append(targets)
        print(f"  Created {len(inputs)} training pairs")

    # Concatenate all years
    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute normalization statistics from training data
    mean = inputs.mean(axis=(0, 2, 3), keepdims=True)
    std = inputs.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1e-6] = 1.0  # avoid division by zero

    # Normalize
    inputs = (inputs - mean) / std
    targets = (targets - mean) / std

    # Split: 70% train, 15% calibration, 15% test
    n = len(inputs)
    n_train = int(0.7 * n)
    n_cal = int(0.15 * n)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)

    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]

    # Save as PyTorch tensors
    torch.save({
        'train_inputs': torch.tensor(inputs[train_idx], dtype=torch.float32),
        'train_targets': torch.tensor(targets[train_idx], dtype=torch.float32),
        'cal_inputs': torch.tensor(inputs[cal_idx], dtype=torch.float32),
        'cal_targets': torch.tensor(targets[cal_idx], dtype=torch.float32),
        'test_inputs': torch.tensor(inputs[test_idx], dtype=torch.float32),
        'test_targets': torch.tensor(targets[test_idx], dtype=torch.float32),
        'mean': torch.tensor(mean, dtype=torch.float32),
        'std': torch.tensor(std, dtype=torch.float32),
        'variables': variables,
    }, os.path.join(output_dir, 'era5_processed.pt'))

    print(f"\nSaved to {output_dir}/era5_processed.pt")
    print(f"  Train: {len(train_idx)}")
    print(f"  Calibration: {len(cal_idx)}")
    print(f"  Test: {len(test_idx)}")
    print(f"  Channels: {inputs.shape[1]}")
    print(f"  Grid: {inputs.shape[2]}x{inputs.shape[3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/era5/")
    parser.add_argument("--variables", nargs="+",
                       default=["geopotential", "temperature"])
    parser.add_argument("--years", nargs="+", type=int, default=[2020])
    parser.add_argument("--output", default="../data/era5_processed/")
    parser.add_argument("--coarsen", type=int, default=4)
    args = parser.parse_args()
    prepare_training_data(args.data_dir, args.variables, args.years,
                         args.output, args.coarsen)
```

Run the data preparation on your downloaded ERA5 data.

```bash
python3 scripts/prepare_era5_training.py \
    --data-dir data/era5/ \
    --variables geopotential temperature \
    --years 2020 \
    --output data/era5_processed/ \
    --coarsen 4
```

This reads the raw NetCDF files, coarsens the spatial grid by a factor of 4 so training is fast, normalizes the data, creates input-target pairs with a 6-hour lead time, splits into train-calibration-test sets, and saves everything as a single PyTorch file.

The coarsening factor of 4 takes the native ERA5 grid from 721x1440 down to about 180x360. If that is still too large for your GPU memory, increase coarsen to 8 or even 16 for initial development. You can always scale up later.

Now create the training script. Save this as `scripts/train_weather_surrogate.py`.

```python
"""
Train a small weather surrogate on processed ERA5 data.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class WeatherUNet(nn.Module):
    """Small UNet for weather prediction."""

    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 4),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        self.dec1 = nn.Conv2d(hidden * 2, in_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        # Handle potential size mismatch from stride/padding
        if d3.shape != e2.shape:
            d3 = d3[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        if d2.shape != e1.shape:
            d2 = d2[:, :, :e1.shape[2], :e1.shape[3]]
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return x + out  # residual: predict the change


def train(data_path, epochs=100, batch_size=32, lr=1e-3, hidden=64,
          output_path="models/weather_surrogate.pt"):
    # Load processed data
    data = torch.load(data_path, weights_only=False)
    train_X = data['train_inputs']
    train_Y = data['train_targets']
    cal_X = data['cal_inputs']
    cal_Y = data['cal_targets']

    in_channels = train_X.shape[1]
    print(f"Training weather surrogate")
    print(f"  Channels: {in_channels}")
    print(f"  Grid: {train_X.shape[2]}x{train_X.shape[3]}")
    print(f"  Train samples: {len(train_X)}")
    print(f"  Cal samples: {len(cal_X)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = WeatherUNet(in_channels, hidden=hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            pred = model(batch_X)
            loss = loss_fn(pred, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches

        # Validation on calibration set
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate in chunks to avoid OOM
                val_losses = []
                for i in range(0, len(cal_X), batch_size):
                    chunk_X = cal_X[i:i+batch_size].to(device)
                    chunk_Y = cal_Y[i:i+batch_size].to(device)
                    val_pred = model(chunk_X)
                    val_losses.append(loss_fn(val_pred, chunk_Y).item())
                val_loss = sum(val_losses) / len(val_losses)

            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save({
                    'model_state': model.state_dict(),
                    'in_channels': in_channels,
                    'hidden': hidden,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch + 1,
                }, output_path)
                print(f"    Saved best model (val_loss={best_val_loss:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {output_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to era5_processed.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", default="models/weather_surrogate.pt")
    args = parser.parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr, args.hidden, args.output)
```

Run the training.

```bash
python3 scripts/train_weather_surrogate.py \
    --data data/era5_processed/era5_processed.pt \
    --epochs 100 \
    --batch-size 32 \
    --hidden 64 \
    --output models/weather_surrogate.pt
```

On a GPU this should take 30 minutes to 2 hours depending on your data size and GPU. On CPU it will be slower, maybe 4 to 8 hours. Watch the validation loss. It should decrease steadily for the first 50 or so epochs and then level off. If it starts going up, the model is overfitting and you should either stop early, reduce the hidden dimension, or add more data.

Once training finishes, verify the model works by running conformal prediction on it.

```bash
python3 -c "
import torch
import sys
sys.path.insert(0, 'src')
from conformal import SplitConformal, SupNormScore

# Load model and data
data = torch.load('data/era5_processed/era5_processed.pt', weights_only=False)
checkpoint = torch.load('models/weather_surrogate.pt', weights_only=False)

from scripts.train_weather_surrogate import WeatherUNet

model = WeatherUNet(checkpoint['in_channels'], checkpoint['hidden'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Run predictions on calibration and test sets
with torch.no_grad():
    cal_pred = model(data['cal_inputs'])
    test_pred = model(data['test_inputs'])

# Conformal prediction
cp = SplitConformal(SupNormScore(), alpha=0.05)
cp.calibrate(cal_pred, data['cal_targets'])
results = cp.evaluate(test_pred, data['test_targets'])

print()
print('If coverage is between 93% and 97%, the pipeline is working correctly.')
print('This is the core result: conformal prediction on a real weather surrogate.')
"
```

If coverage is within a few percentage points of 95%, you have successfully completed the end-to-end pipeline. That is the entire goal. From here the project extends to physics projection and shift detection, but the foundation is built.

### What to document for the team

Record the final training loss and validation loss. Record the coverage and interval width from the conformal evaluation. Note how long training took and on what hardware. Save these numbers because they become the baseline that the rest of the project improves on.

### Troubleshooting

If you get an out of memory error on GPU, reduce the batch size to 16 or 8, or increase the coarsening factor when preparing the data.

If the model seems to not learn at all with loss staying flat, check that the data normalization is correct. Run `print(data['train_inputs'].mean(), data['train_inputs'].std())` and make sure the values are close to 0 and 1 respectively.

If conformal coverage is very far from the target, like 50% or 100%, something is wrong with the calibration split. Make sure the calibration data was not used during training.
