"""
WeatherBench 2 Setup Script
============================

Sets up the WeatherBench 2 evaluation suite for standardized ML weather
model evaluation.

WeatherBench 2 provides:
    - Standardized metrics (RMSE, ACC, CRPS) for weather forecasting
    - Ground truth data aligned with common ML weather model grids
    - Scripts for computing deterministic and probabilistic scores

We use it to:
    - Evaluate our weather surrogate against standard baselines
    - Provide the evaluation framework our conformal methods plug into
    - Access pre-processed ERA5 evaluation data on standard grids

Prerequisites:
    pip install weatherbench2  (or clone from GitHub)

Usage:
    python setup_weatherbench2.py --output ../data/weatherbench2/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def install_weatherbench2():
    """Install WeatherBench 2 package."""
    print("Installing WeatherBench 2...")
    try:
        import weatherbench2
        print(f"  Already installed: version {weatherbench2.__version__}")
        return True
    except ImportError:
        pass

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "weatherbench2"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  Installed successfully.")
        return True

    print("  pip install failed. Trying GitHub clone...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/google-research/weatherbench2.git",
         "/tmp/weatherbench2"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "/tmp/weatherbench2"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  Installed from GitHub.")
            return True

    print("  Could not install WeatherBench 2 automatically.")
    print("  Manual steps:")
    print("    git clone https://github.com/google-research/weatherbench2.git")
    print("    cd weatherbench2")
    print("    pip install -e .")
    return False


def setup_evaluation_data(output_dir):
    """Document how to get WeatherBench 2 evaluation data."""
    os.makedirs(output_dir, exist_ok=True)

    info_file = os.path.join(output_dir, "DATA_ACCESS.md")
    with open(info_file, "w") as f:
        f.write("# WeatherBench 2 Data Access\n\n")
        f.write("WeatherBench 2 evaluation data is stored on Google Cloud Storage.\n\n")
        f.write("## Quick access (small subset for testing)\n\n")
        f.write("```bash\n")
        f.write("# Install Google Cloud SDK if needed\n")
        f.write("# https://cloud.google.com/sdk/docs/install\n\n")
        f.write("# List available data\n")
        f.write("gsutil ls gs://weatherbench2/datasets/\n\n")
        f.write("# Download ERA5 ground truth at 1.5 degree resolution (smallest)\n")
        f.write("gsutil -m cp -r gs://weatherbench2/datasets/era5/1959-2023-6h-1440x721.zarr ./\n")
        f.write("```\n\n")
        f.write("## For our project\n\n")
        f.write("We need:\n")
        f.write("- ERA5 ground truth (for calibration and evaluation)\n")
        f.write("- Climatology baselines (for ACC computation)\n")
        f.write("- Pre-computed GraphCast forecasts (optional, for comparison)\n\n")
        f.write("## Alternative: use xarray + zarr directly\n\n")
        f.write("```python\n")
        f.write("import xarray as xr\n")
        f.write("ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023-6h-1440x721.zarr')\n")
        f.write("```\n\n")
        f.write("This requires `gcsfs` package: `pip install gcsfs`\n")

    print(f"Data access instructions written to {info_file}")


def verify_installation():
    """Quick smoke test of WeatherBench 2."""
    try:
        import weatherbench2
        print(f"WeatherBench 2 version: {weatherbench2.__version__}")
        print("Smoke test passed.")
        return True
    except ImportError:
        print("WeatherBench 2 not importable.")
        print("This is OK for now. You can still proceed with ERA5 data directly.")
        print("WeatherBench 2 is mainly needed for standardized evaluation metrics later.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Set up WeatherBench 2")
    parser.add_argument("--output", default="../data/weatherbench2/", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("WeatherBench 2 Setup")
    print("=" * 60)
    print()

    installed = install_weatherbench2()
    print()
    setup_evaluation_data(args.output)
    print()
    verify_installation()

    print()
    print("Next steps:")
    print("  1. Follow instructions in data/weatherbench2/DATA_ACCESS.md")
    print("  2. Download a small ERA5 subset for initial testing")
    print("  3. Run test_graphcast.py to verify the weather surrogate")


if __name__ == "__main__":
    main()
