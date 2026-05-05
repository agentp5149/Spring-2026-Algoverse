"""
Prepare ERA5 data for weather surrogate training.
Creates (input, target) pairs where input is state at time t
and target is state at time t+6h.

Usage:
    python prepare_era5_training.py --data-dir ../data/era5/ \
        --variables geopotential temperature \
        --years 2020 \
        --output ../data/era5_processed/ \
        --coarsen 4
"""

import argparse
import os
import numpy as np
import xarray as xr
import torch
from pathlib import Path


def load_era5_variable(data_dir, variable, year):
    filepath = os.path.join(data_dir, f"era5_{variable}_{year}.nc")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing: {filepath}")
    ds = xr.open_dataset(filepath)
    var_name = [v for v in ds.data_vars if v not in ['time', 'level']][0]
    return ds[var_name]


def prepare_training_data(data_dir, variables, years, output_dir,
                          coarsen_factor=4, lead_time_hours=6):
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

            if coarsen_factor > 1:
                data = data.coarsen(
                    latitude=coarsen_factor, longitude=coarsen_factor,
                    boundary='trim'
                ).mean()

            values = data.values
            if values.ndim == 3:
                values = values[:, np.newaxis, :, :]

            n_times, n_levels, n_lat, n_lon = values.shape
            for lev in range(n_levels):
                channels.append(values[:, lev, :, :])

        if not channels:
            print(f"  No data for {year}, skipping")
            continue

        stacked = np.stack(channels, axis=1)
        print(f"  Shape: {stacked.shape}")

        offset = lead_time_hours // 6
        inputs = stacked[:-offset]
        targets = stacked[offset:]
        all_inputs.append(inputs)
        all_targets.append(targets)
        print(f"  Created {len(inputs)} training pairs")

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mean = inputs.mean(axis=(0, 2, 3), keepdims=True)
    std = inputs.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1e-6] = 1.0

    inputs = (inputs - mean) / std
    targets = (targets - mean) / std

    n = len(inputs)
    n_train = int(0.7 * n)
    n_cal = int(0.15 * n)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    test_idx = perm[n_train + n_cal:]

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
    parser.add_argument("--variables", nargs="+", default=["geopotential", "temperature"])
    parser.add_argument("--years", nargs="+", type=int, default=[2020])
    parser.add_argument("--output", default="../data/era5_processed/")
    parser.add_argument("--coarsen", type=int, default=4)
    args = parser.parse_args()
    prepare_training_data(args.data_dir, args.variables, args.years,
                         args.output, args.coarsen)
