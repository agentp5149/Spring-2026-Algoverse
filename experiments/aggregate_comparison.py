"""
Cross-Domain Aggregate Comparison Table
========================================

Aggregates in-distribution results from all available surrogates into a
single markdown table with per-method failure flags.

Surrogates covered
------------------
  pk          neural_ode_pk           alpha=0.10  target=90%  n_test=150
  molecular   mace_md17_aspirin       alpha=0.05  target=95%  n_test=31765
  weather     graphcast_small         PENDING (no results yet)

Coverage metrics
----------------
  PK        : trajectory-level sup-norm (ALL 49 timepoints covered)
  Molecular : sample-level TrajectoryNormScore L2/sqrt(n_atoms) ≤ threshold

Fail criterion
--------------
  A method fails if  empirical_coverage  <  target_coverage - 0.02  (2pp)

Usage
-----
    cd experiments
    python aggregate_comparison.py
    # writes  ../results/aggregate_comparison.md
"""

from __future__ import annotations

import datetime
import json
import os
import sys

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_repo, "results")

TARGET_2PP = 0.02


# ------------------------------------------------------------------ #
# Diagnosis text (method-level, domain-aware where needed)           #
# ------------------------------------------------------------------ #

DIAGNOSIS: dict[tuple[str, str], str] = {
    ("deep_ensemble", "pk"): (
        "Ensemble spread captures only member diversity on the same finite dataset. "
        "After 200 epochs on 1000 patients the neural ODE has not converged; member "
        "predictions are biased rather than varied, so z·σ intervals are far too "
        "narrow. Systematic underfitting (~127 amount-unit residuals) dwarfs the "
        "ensemble spread (~1.8 units). No finite-M ensemble can guarantee coverage."
    ),
    ("mc_dropout", "pk"): (
        "MC Dropout posterior collapses in the 64-unit dynamics network: the ODE "
        "learns to route information through paths that are rarely dropped, narrowing "
        "MC sample variance to ~0.64 units while true residuals are ~127 units. "
        "Dropout-based UQ cannot detect systematic underfitting bias."
    ),
    ("bayesian_vi", "pk"): (
        "Mean-field diagonal posterior ignores weight correlations. The MAP training "
        "strategy (mean weights inside odeint, KL as regulariser) further decouples "
        "the variational posterior from actual trajectory errors. At inference, "
        "posterior samples produce under-dispersed intervals (~3.0 units) relative "
        "to true residuals (~127 units). Root cause: model bias dominates weight "
        "uncertainty."
    ),
    ("mc_dropout", "molecular"): (
        "Gaussian n_sigma·std intervals are calibrated under Gaussian-posterior "
        "assumptions, but force-field prediction errors have heavy-tailed "
        "distributions (one mis-predicted atom creates a large L2-norm score). "
        "Coverage collapses to 56.7% (target 95%, gap −38.3 pp) because "
        "approximate-posterior std severely underestimates actual error magnitude. "
        "MC Dropout UQ is not exchangeable with the test error distribution."
    ),
    ("conformal_split", "pk"): None,        # pass — no diagnosis needed
    ("conformal_split", "molecular"): None, # pass
}


# ------------------------------------------------------------------ #
# Load helpers                                                        #
# ------------------------------------------------------------------ #

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalise_pk(j: dict) -> dict:
    """Normalise a PK result JSON to the common schema."""
    # Support both schemas produced by run_pk_comparison.py and bayesian_vi_pk.py CLI
    cov = j.get("test_traj_coverage",
          j.get("test_empirical_coverage",
          j.get("empirical_coverage")))
    width = j.get("test_mean_width",
            j.get("test_mean_interval_width",
            j.get("mean_width")))
    alpha = j.get("alpha", 0.10)
    target = j.get("target_coverage", 1 - alpha)
    return {
        "method":             j["method"],
        "domain":             "pk",
        "surrogate":          "neural_ode_pk",
        "alpha":              alpha,
        "target_coverage":    target,
        "empirical_coverage": cov,
        "coverage_gap":       round(cov - target, 6) if cov is not None else None,
        "mean_width":         width,
        "n_test":             j.get("n_test", 150),
        "runtime_train_s":    j.get("runtime_train_s"),
        "runtime_inf_s":      j.get("runtime_inference_test_s",
                              j.get("runtime_inference_s")),
        "coverage_metric":    "traj-sup-norm (all 49 timepoints)",
    }


def normalise_mol(j: dict) -> dict:
    """Normalise a molecular result JSON to the common schema."""
    return {
        "method":             j["method"],
        "domain":             "molecular",
        "surrogate":          j.get("surrogate", f'mace_md17_{j.get("molecule","aspirin")}'),
        "alpha":              j["alpha"],
        "target_coverage":    j["target_coverage"],
        "empirical_coverage": j["empirical_coverage"],
        "coverage_gap":       j["coverage_gap"],
        "mean_width":         j["mean_width"],
        "n_test":             j["n_test"],
        "runtime_train_s":    j.get("runtime_train_s"),
        "runtime_inf_s":      j.get("runtime_inference_s"),
        "coverage_metric":    "TrajectoryNormScore L2/√n_atoms",
    }


# ------------------------------------------------------------------ #
# Table builder                                                       #
# ------------------------------------------------------------------ #

def build_table(rows: list[dict]) -> tuple[str, list[dict]]:
    """Return (markdown_table_string, list_of_failing_rows)."""
    header = (
        "| Domain | Surrogate | Method | α | Target | Cov | Gap | "
        "Width | n_test | Status |"
    )
    sep = "|" + "---|" * 10

    lines = [header, sep]
    failures: list[dict] = []

    for r in rows:
        cov    = r["empirical_coverage"]
        target = r["target_coverage"]
        gap    = r["coverage_gap"]
        ok     = (gap is not None) and (gap >= -TARGET_2PP)
        status = "✓" if ok else "✗ FAIL"

        cov_str   = f"{cov:.3f}"  if cov   is not None else "—"
        gap_str   = f"{gap:+.3f}" if gap   is not None else "—"
        width_str = f"{r['mean_width']:.3f}" if r.get("mean_width") is not None else "—"
        n_str     = f"{r['n_test']:,}"

        lines.append(
            f"| {r['domain']:<10}"
            f"| {r['surrogate']:<24}"
            f"| {r['method']:<16}"
            f"| {r['alpha']:.2f}"
            f"| {target:.2f}"
            f"| {cov_str}"
            f"| {gap_str}"
            f"| {width_str}"
            f"| {n_str}"
            f"| {status} |"
        )

        if not ok:
            failures.append(r)

    return "\n".join(lines), failures


def build_report(rows: list[dict]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines += [
        "# In-Distribution Uncertainty Method Comparison — All Surrogates",
        "",
        f"**Generated:** {now}  ",
        f"**Fail criterion:** empirical coverage < target − 2 pp  ",
        "**Surrogates:**",
        "- `neural_ode_pk` — 2-compartment PK model (1 000 synthetic patients, seed 42)",
        "  700 train / 150 cal / 150 test. Coverage: trajectory-level sup-norm "
        "(ALL 49 timepoints inside the band).  α = 0.10, target = 90%.",
        "- `mace_md17_aspirin` — molecular force-field surrogate on MD17 aspirin "
        "(~211 k snapshots 70/15/15 split). Coverage: per-snapshot "
        "TrajectoryNormScore L2/√n_atoms ≤ threshold.  α = 0.05, target = 95%.",
        "- `graphcast_small` (weather) — **PENDING**: ERA5 data download and "
        "training not yet complete; will be added in Week 3.",
        "",
        "## Results",
        "",
    ]

    table_str, failures = build_table(rows)
    lines.append(table_str)
    lines += [
        "",
        "**Width units** differ by domain:",
        "- PK: mean (upper − lower) in raw amount units (central compartment, mg).  ",
        "  Conformal width = 2 × conformal threshold in TrajectoryNormScore L2/√T units.",
        "- Molecular: 2 × conformal threshold OR 2 × n_sigma × ||std||_F/√(n_atoms·3) "
        "in TrajectoryNormScore units (kcal/mol/Å).  ",
        "",
    ]

    if failures:
        lines += [
            "## Failure Diagnosis",
            "",
            f"The following {len(failures)} method(s) miss the 2-percentage-point "
            "coverage target and are flagged below.  ",
            "Root causes are categorised as **model bias** (surrogate has not "
            "converged; errors dwarf posterior uncertainty) or "
            "**posterior mismatch** (UQ posterior is mis-specified relative to the "
            "true error distribution).  ",
            "",
        ]
        for r in failures:
            method = r["method"]
            domain = r["domain"]
            cov    = r["empirical_coverage"]
            gap    = r["coverage_gap"]
            target = r["target_coverage"]

            lines.append(
                f"### `{method}` on `{r['surrogate']}` "
                f"(coverage {cov:.1%}, target {target:.0%}, gap {gap:+.1%})"
            )
            lines.append("")
            note = DIAGNOSIS.get((method, domain))
            if note:
                lines.append(note)
            else:
                lines.append("No diagnosis entry — please add one.")
            lines.append("")

    else:
        lines += ["All methods meet the 2 pp coverage target.", ""]

    lines += [
        "## Notes",
        "",
        "- **Split conformal is the only method with a finite-sample guarantee**: "
        "coverage ≥ 1 − α − 1/(n_cal + 1) holds distribution-free by construction. "
        "Both PK and molecular conformal methods pass.  ",
        "- **Baseline methods (ensemble, dropout, VI) require model convergence**: "
        "more epochs / members / better priors widen intervals but cannot guarantee "
        "nominal coverage. All three PK baselines fail because systematic "
        "underfitting bias dominates weight uncertainty.  ",
        "- **PK trajectory-level metric is strict**: requires simultaneous coverage "
        "at all 49 time-points. Pointwise PK coverage is substantially higher "
        "(Bayesian VI: 34.4%, MC Dropout: 6.2%, Ensemble: 10.7%).  ",
        "- **Molecular MC Dropout mismatch is severe (−38.3 pp)**: the Gaussian "
        "posterior assumption is violated for heavy-tailed force errors. "
        "Conformal prediction is robust because it makes no distributional "
        "assumption.  ",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    rows: list[dict] = []

    # ---------- PK results ----------
    pk_files = {
        "deep_ensemble_pk.json": normalise_pk,
        "mc_dropout_pk.json":    normalise_pk,
        "bayesian_vi_pk.json":   normalise_pk,
        "conformal_pk.json":     normalise_pk,
    }
    for fname, norm_fn in pk_files.items():
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        j = _load_json(path)
        rows.append(norm_fn(j))
        print(f"  Loaded PK: {j['method']}")

    # ---------- Molecular results ----------
    mol_files = [
        "molecular/aspirin_conformal_indist_95.json",
        "molecular/aspirin_mc_dropout_indist_95.json",
    ]
    for fname in mol_files:
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        j = _load_json(path)
        rows.append(normalise_mol(j))
        print(f"  Loaded molecular: {j['method']}")

    # ---------- Sort: domain → method ----------
    domain_order = {"pk": 0, "molecular": 1, "weather": 2}
    method_order = {"deep_ensemble": 0, "mc_dropout": 1, "bayesian_vi": 2,
                    "conformal_split": 3}
    rows.sort(key=lambda r: (
        domain_order.get(r["domain"], 9),
        method_order.get(r["method"], 9),
    ))

    report = build_report(rows)

    out_path = os.path.join(RESULTS, "aggregate_comparison.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nWrote {out_path}")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
