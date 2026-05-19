"""
Evaluation Harness for Uncertainty Quantification Methods
==========================================================

Provides three capabilities:
  1. compute_empirical_coverage  -- trajectory-level and pointwise coverage
  2. compute_interval_widths     -- mean, median, std of interval widths
  3. log_results / compare_methods -- standardised JSON logging and tabular comparison

All functions accept torch.Tensor inputs so they compose directly with model
output from deep_ensemble_pk, conformal.SplitConformal, etc.

Usage:
    from baselines.eval_harness import (
        compute_empirical_coverage,
        compute_interval_widths,
        log_results,
        compare_methods,
    )

    cov = compute_empirical_coverage(lower, upper, targets)
    wid = compute_interval_widths(lower, upper)
    log_results("deep_ensemble", "neural_ode_pk", alpha=0.1,
                coverage=cov, widths=wid, runtime={"train_s": 312, "inference_s": 4.1},
                output_path="results/deep_ensemble_pk.json")
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from typing import Optional

import numpy as np
import torch


# ============================================================
# 1. Empirical coverage
# ============================================================

def compute_empirical_coverage(
    lower: torch.Tensor,
    upper: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "trajectory",
) -> dict:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        lower:     (n, ...) lower bounds of prediction intervals
        upper:     (n, ...) upper bounds of prediction intervals
        targets:   (n, ...) ground-truth values; same shape as lower/upper
        reduction: "trajectory"  -> a sample is covered iff ALL its elements
                                    fall inside the interval (conservative,
                                    matches the sup-norm conformal score)
                   "pointwise"   -> fraction of individual elements covered
                   "both"        -> returns both

    Returns:
        dict with keys:
            coverage           -- scalar float, primary coverage statistic
            n_covered          -- int, number of covered samples (trajectory mode)
            n_total            -- int
            coverage_pointwise -- (optional) per-position coverage tensor
    """
    covered_elem = (targets >= lower) & (targets <= upper)  # (n, ...)
    flat = covered_elem.view(covered_elem.shape[0], -1)     # (n, D)

    result: dict = {"n_total": len(flat)}

    if reduction in ("trajectory", "both"):
        traj_covered = flat.all(dim=1).float()
        result["coverage"] = float(traj_covered.mean().item())
        result["n_covered"] = int(traj_covered.sum().item())

    if reduction in ("pointwise", "both"):
        pw = covered_elem.float().mean(dim=0)
        result["coverage_pointwise"] = pw.tolist()
        if reduction == "pointwise":
            result["coverage"] = float(pw.mean().item())
            result["n_covered"] = int((flat.all(dim=1)).sum().item())

    return result


# ============================================================
# 2. Interval widths
# ============================================================

def compute_interval_widths(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict:
    """
    Compute summary statistics of interval widths.

    Args:
        lower: (n, ...) lower bounds
        upper: (n, ...) upper bounds

    Returns:
        dict with:
            mean_width          -- mean over all positions and samples
            median_width        -- median over all positions and samples
            width_std           -- std over all positions and samples
            mean_max_width      -- mean of per-sample supremum width
                                   (width of the tightest band that covers
                                   the whole trajectory)
            median_max_width    -- median of per-sample supremum width
            per_position_mean   -- (optional) mean width at each position
    """
    widths = (upper - lower).abs()              # (n, ...)
    flat = widths.view(widths.shape[0], -1)     # (n, D)
    max_per_sample = flat.max(dim=1).values     # (n,)

    return {
        "mean_width": float(flat.mean().item()),
        "median_width": float(flat.median().item()),
        "width_std": float(flat.std().item()),
        "mean_max_width": float(max_per_sample.mean().item()),
        "median_max_width": float(max_per_sample.median().item()),
    }


# ============================================================
# 3. Results logging
# ============================================================

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def log_results(
    method: str,
    surrogate: str,
    alpha: float,
    coverage: dict,
    widths: dict,
    runtime: dict,
    output_path: str,
    extra: Optional[dict] = None,
) -> dict:
    """
    Write a standardised JSON result log for one (method, alpha) evaluation run.

    Canonical schema
    ----------------
    method               -- e.g. "deep_ensemble", "mc_dropout", "conformal_split"
    surrogate            -- e.g. "neural_ode_pk", "graphcast_small", "mace_md17"
    alpha                -- miscoverage level (float)
    target_coverage      -- 1 - alpha
    timestamp            -- UTC ISO-8601
    git_commit           -- short hash
    empirical_coverage   -- primary coverage statistic (trajectory-level)
    n_covered, n_total   -- raw counts
    coverage_gap         -- empirical - target (positive = over-covered)
    mean_width           -- mean interval width
    median_width
    width_std
    mean_max_width       -- mean sup-norm width per trajectory
    runtime_train_s      -- wall-clock training time (None if not trained here)
    runtime_inference_s  -- wall-clock inference time on the evaluated split
    notes                -- free-text, empty by default

    Args:
        method:       method name string
        surrogate:    surrogate name string
        alpha:        miscoverage level
        coverage:     output of compute_empirical_coverage()
        widths:       output of compute_interval_widths()
        runtime:      dict with optional keys "train_s", "inference_s"
        output_path:  path to write JSON (parent dirs created automatically)
        extra:        any additional fields to include in the record

    Returns:
        The full result dict (also written to disk).
    """
    record = {
        "method": method,
        "surrogate": surrogate,
        "alpha": alpha,
        "target_coverage": round(1 - alpha, 6),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        # Coverage
        "empirical_coverage": coverage.get("coverage"),
        "n_covered": coverage.get("n_covered"),
        "n_total": coverage.get("n_total"),
        "coverage_gap": (
            round(coverage["coverage"] - (1 - alpha), 6)
            if coverage.get("coverage") is not None else None
        ),
        # Width
        "mean_width": widths.get("mean_width"),
        "median_width": widths.get("median_width"),
        "width_std": widths.get("width_std"),
        "mean_max_width": widths.get("mean_max_width"),
        # Runtime
        "runtime_train_s": runtime.get("train_s"),
        "runtime_inference_s": runtime.get("inference_s"),
        # Notes
        "notes": "",
    }
    if extra:
        record.update(extra)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


# ============================================================
# 4. Method comparison
# ============================================================

def compare_methods(result_paths: list[str]) -> str:
    """
    Load multiple result JSON files and return a GitHub-flavoured markdown table.

    Columns: Method · Surrogate · Target · Empirical · Gap · Mean Width ·
             Max Width · Train (s) · Inference (s)

    Args:
        result_paths: list of paths to JSON files produced by log_results()

    Returns:
        Markdown table string (printable or writable to a .md file).
    """
    rows = []
    for p in result_paths:
        with open(p) as f:
            rows.append(json.load(f))

    cols = [
        ("Method",       lambda r: r["method"]),
        ("Surrogate",    lambda r: r["surrogate"]),
        ("Target",       lambda r: f"{r['target_coverage']:.2f}"),
        ("Coverage",     lambda r: f"{r['empirical_coverage']:.3f}"
                                    if r.get("empirical_coverage") is not None else "—"),
        ("Gap",          lambda r: f"{r['coverage_gap']:+.3f}"
                                    if r.get("coverage_gap") is not None else "—"),
        ("Mean Width",   lambda r: f"{r['mean_width']:.4f}"
                                    if r.get("mean_width") is not None else "—"),
        ("Max Width",    lambda r: f"{r['mean_max_width']:.4f}"
                                    if r.get("mean_max_width") is not None else "—"),
        ("Train (s)",    lambda r: f"{r['runtime_train_s']:.0f}"
                                    if r.get("runtime_train_s") is not None else "—"),
        ("Inference (s)",lambda r: f"{r['runtime_inference_s']:.1f}"
                                    if r.get("runtime_inference_s") is not None else "—"),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "|" + "|".join([" --- "] * len(cols)) + "|"
    lines  = [header, sep]

    for r in rows:
        cells = " | ".join(fn(r) for _, fn in cols)
        lines.append(f"| {cells} |")

    return "\n".join(lines)


# ============================================================
# Quick smoke-test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    n, T = 200, 50

    # Simulate a model with mild over-coverage
    true = torch.randn(n, T)
    lower = true - 0.5
    upper = true + 0.5

    cov = compute_empirical_coverage(lower, upper, true, reduction="both")
    wid = compute_interval_widths(lower, upper)

    print("Coverage:", cov["coverage"])
    print("Widths:", wid)

    record = log_results(
        method="smoke_test",
        surrogate="synthetic",
        alpha=0.1,
        coverage=cov,
        widths=wid,
        runtime={"train_s": 10.0, "inference_s": 0.5},
        output_path="/tmp/smoke_test.json",
    )
    print("Logged to /tmp/smoke_test.json")
    print(json.dumps(record, indent=2))
