# Compute Time Estimates for Calibration Runs
**Author:** Vijay
**Week:** 1

## Overview

Estimates for running surrogate inference on calibration sets to produce nonconformity scores for conformal prediction. DFT ground truth energies and forces are already available in MD17/MD22 and do not require new DFT calculations.

## Split Sizes

Based on the 70/15/15 split policy in docs/split_policy.md:

| Surrogate | Total Samples | Cal Samples (15%) |
|-----------|--------------|-------------------|
| MD17 aspirin (molecular) | 211,762 | ~31,764 |
| MD22 DHA (molecular OOD) | ~122,000 | ~18,300 |
| ERA5 weather (2018-2020) | ~26,280 timesteps | ~3,942 |
| PK synthetic | 5,000 | 750 |

## Molecular Surrogate (MACE on MD17)

MACE inference on CPU for a single aspirin configuration takes roughly 50-100ms. On a GPU (A100) this drops to under 5ms per configuration.

| Hardware | Time per config | 31,764 configs | 
|----------|----------------|----------------|
| CPU only | ~75ms | ~40 min |
| 1x A100 GPU | ~5ms | ~3 min |

Recommended: 1x A100 GPU instance. Total calibration inference including MD22: ~10 min.

## ERA5 Weather Surrogate

ERA5 data processing involves loading NetCDF files and running the neural weather surrogate on each timestep. ERA5 at 0.25 degree resolution is ~1.4GB per variable per year.

| Hardware | Time per timestep | 3,942 timesteps |
|----------|------------------|-----------------|
| CPU only | ~2s | ~2.2 hours |
| 1x A100 GPU | ~0.1s | ~7 min |

Recommended: 1x A100 GPU instance.

## PK Surrogate (Neural ODE)

Neural ODE inference is fast even on CPU. 750 calibration trajectories at ~1s each on CPU.

| Hardware | Time per trajectory | 750 trajectories |
|----------|-------------------|-----------------|
| CPU only | ~1s | ~12 min |
| GPU | ~0.05s | ~1 min |

## AWS Instance Recommendations

| Task | Instance Type | Hourly Cost | Est. Runtime | Est. Cost |
|------|--------------|-------------|--------------|-----------|
| Molecular (MACE) | g5.xlarge (A100) | ~$1.01/hr | 30 min | ~$0.51 |
| Weather (ERA5) | g5.xlarge (A100) | ~$1.01/hr | 1 hour | ~$1.01 |
| PK (Neural ODE) | t3.medium (CPU) | ~$0.04/hr | 30 min | ~$0.02 |
| **Total** | | | | **~$1.54** |

## Notes

- All estimates assume pre-downloaded data in data/ directory
- ERA5 estimate assumes the small subset downloaded by download_era5.py (2018-2020, geopotential + temperature, 500/850 hPa levels)
- MACE inference uses float32 as set in test_mace.py
- These are calibration-only estimates. Training the surrogates is a separate and more expensive step handled by Prajesh (weather) and the team
- Queue jobs once surrogate training is complete in Week 2-3
