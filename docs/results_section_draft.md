# Results Section Draft (ID + OOD + Projection)

This draft uses only measured outputs available in this checkout at run time (2026-06-01).  
Primary visualization artifact for team review: the canvas `ood-results-summary.canvas.tsx`.

## In-distribution coverage

On PK (`alpha=0.10`, target 90%), split conformal reached 93.3% trajectory coverage (+3.3 pp), while deep ensemble, MC dropout, and Bayesian VI each remained at 0.0% trajectory coverage under the strict all-timepoint metric. This confirms that baseline posterior spread remained too narrow relative to trajectory error in this run, whereas conformal calibration recovered near-nominal coverage with finite-sample validity.

## OOD coverage under shift

For PK OOD, the adaptive shift mechanism (input-shift detection + interval widening) improved coverage from 32.7% to 82.0% (+49.3 pp). In the same OOD setting, deep ensemble, MC dropout, and Bayesian VI each measured 0.0% trajectory coverage. Adaptive conformal is therefore the best-performing method under the tested PK shift, but still remains below the 90% target by 8.0 pp.

### OOD artifact status

PK OOD baseline results are available and included:

- `results/pk/pk_ood_baselines_vs_adaptive.json`
- `results/pk/pk_ood_shift_widening.json`
- `results/pk/pk_shift_threshold_sweep.json`

The cross-domain OOD aggregation remains blocked for molecular and weather because analogous OOD baseline artifacts were not found under `results/` or elsewhere in the workspace scan. These should be added before claiming “all three surrogates” for the OOD baseline comparison.

Blocked inputs:

- Molecular OOD baselines: deep ensemble, MC dropout, Bayesian VI, adaptive/conformal.
- Weather OOD baselines: deep ensemble, MC dropout, Bayesian VI, adaptive/conformal.
- Molecular ensemble width artifact for the conformal-width ≤ 2× ensemble-width criterion.

## Pareto trade-off for adaptive shift threshold

Sweeping the shift threshold percentile in PK shows the expected coverage-width trade-off: lower thresholds widen intervals more aggressively and recover higher coverage (up to 100% at 50th percentile, width multiplier 3.384x), while higher thresholds reduce width but sacrifice OOD coverage (72.8% at 99th percentile, width multiplier 1.671x). Around the 70th percentile, coverage first exceeds nominal (91.3%) with a 2.720x width multiplier.

## Physics projection ablation

Across the three physics constraint domains (linear weather proxy, quadratic molecular proxy, equivariance molecular proxy), projection with recalibration reduced conformal interval width by 0.5%-9.1%. Coverage under recalibration was 86.5% (linear), 90.0% (quadratic), and 88.5% (equivariance), compared with 87.0%, 90.0%, and 82.5% for no projection. Projection therefore improves efficiency in all three domains and improves or preserves coverage in two of three.

## Interval-width criterion check (conformal <= 2x ensemble width)

The criterion is currently not satisfied where measurable:

- PK (alpha=0.10): conformal/ensemble width ratio = 20.37x.
- Weather (alpha=0.10, from `docs/week2results_prajesh`): ratio = 19.59x.
- Molecular: cannot be evaluated yet because no molecular ensemble width artifact is present in this checkout.

## Team-share note

For team review, share the canvas plus these source files:

- `results/pk/pk_ood_baselines_vs_adaptive.json`
- `results/pk/pk_ood_shift_widening.json`
- `results/pk/pk_shift_threshold_sweep.json`
- `results/conformal_pk.json`
- `results/deep_ensemble_pk.json`
- `results/mc_dropout_pk.json`
- `results/bayesian_vi_pk.json`
- `results/ablation_projection.json`
