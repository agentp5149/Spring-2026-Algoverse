"""
ERA5 Reanalysis Data Download Script
=====================================

Downloads ERA5 data from the Copernicus Climate Data Store (CDS).

Prerequisites:
    1. Create a free account at https://cds.climate.copernicus.eu
    2. Accept the license for ERA5 data
    3. Create ~/.cdsapirc with your credentials:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR_UID:YOUR_API_KEY

Usage:
    # Small test download (2 variables, 2 levels, 1 year)
    python download_era5.py --variables geopotential temperature \
                            --pressure-levels 500 850 \
                            --years 2020 \
                            --output ../data/era5/

    # Larger download for experiments (multiple years)
    python download_era5.py --variables geopotential temperature \
                                       u_component_of_wind v_component_of_wind \
                            --pressure-levels 500 700 850 1000 \
                            --years 2016 2017 2018 2019 2020 \
                            --output ../data/era5/

Variables we care about for the weather surrogate:
    - geopotential (z500 is the classic benchmark variable)
    - temperature
    - u_component_of_wind, v_component_of_wind
    - specific_humidity (optional, for mass conservation tests)

The download can be slow (hours for multi-year requests). CDS queues
requests and processes them asynchronously. The cdsapi client handles
waiting automatically.
"""

import argparse
import os
from pathlib import Path


def check_credentials():
    """Verify CDS API credentials are configured."""
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        print("ERROR: No ~/.cdsapirc file found.")
        print()
        print("To fix this:")
        print("1. Go to https://cds.climate.copernicus.eu and create a free account")
        print("2. Go to your profile page and copy your API key")
        print("3. Create the file ~/.cdsapirc with these two lines:")
        print()
        print("   url: https://cds.climate.copernicus.eu/api")
        print("   key: YOUR_UID:YOUR_API_KEY")
        print()
        print("Replace YOUR_UID:YOUR_API_KEY with the values from your profile.")
        return False

    with open(cdsapirc) as f:
        content = f.read()
    if "YOUR_UID" in content or "YOUR_API_KEY" in content:
        print("ERROR: ~/.cdsapirc still has placeholder values.")
        print("Replace YOUR_UID:YOUR_API_KEY with your actual credentials.")
        return False

    print("CDS API credentials found.")
    return True


def download_era5_pressure_levels(variables, pressure_levels, years, months, output_dir):
    """Download ERA5 pressure level data."""
    import cdsapi

    client = cdsapi.Client()
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        for var in variables:
            filename = f"era5_{var}_{year}.nc"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"Already exists, skipping: {filename}")
                continue

            print(f"Requesting {var} for {year}...")
            print(f"  Pressure levels: {pressure_levels}")
            print(f"  Months: {months}")
            print(f"  This may take a while (CDS queues requests)...")

            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": var,
                    "pressure_level": [str(p) for p in pressure_levels],
                    "year": str(year),
                    "month": [f"{m:02d}" for m in months],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "data_format": "netcdf",
                },
                filepath,
            )
            print(f"  Saved to {filepath}")

    print()
    print("Download complete.")


def verify_download(output_dir):
    """Verify downloaded files can be opened and have expected structure."""
    import xarray as xr

    nc_files = list(Path(output_dir).glob("*.nc"))
    if not nc_files:
        print("No NetCDF files found in output directory.")
        return False

    print(f"Found {len(nc_files)} NetCDF files:")
    all_ok = True
    for f in sorted(nc_files):
        try:
            ds = xr.open_dataset(f)
            dims = dict(ds.dims)
            var_names = list(ds.data_vars)
            print(f"  {f.name}: variables={var_names}, dims={dims}")
            ds.close()
        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis data")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["geopotential", "temperature"],
        help="ERA5 variable names to download",
    )
    parser.add_argument(
        "--pressure-levels",
        nargs="+",
        type=int,
        default=[500, 850],
        help="Pressure levels in hPa",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020],
        help="Years to download",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(range(1, 13)),
        help="Months to download (default: all 12)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/era5/",
        help="Output directory",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, do not download",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_download(args.output)
        return

    if not check_credentials():
        return

    print()
    print(f"Variables: {args.variables}")
    print(f"Pressure levels: {args.pressure_levels}")
    print(f"Years: {args.years}")
    print(f"Output: {args.output}")
    print()

    download_era5_pressure_levels(
        variables=args.variables,
        pressure_levels=args.pressure_levels,
        years=args.years,
        months=args.months,
        output_dir=args.output,
    )

    print("Verifying downloads...")
    verify_download(args.output)


if __name__ == "__main__":
    main()
