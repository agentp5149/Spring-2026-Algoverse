"""
Nonconformity score stubs for conformal prediction.

Goal:
    Provide clear, typed score-function interfaces that can be extended
    as additional surrogate output types are added.
"""

from __future__ import annotations

from typing import Protocol

import torch


class NonconformityScore(Protocol):
    """
    Interface for nonconformity score functions.

    Input:
        prediction: torch.Tensor with leading dimension n_samples
        ground_truth: torch.Tensor with same shape as prediction

    Output:
        torch.Tensor of shape (n_samples,) where each entry is the
        nonconformity score for one sample.
    """

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        ...


class ScalarAbsoluteErrorScore:
    """
    Absolute error nonconformity for scalar/regression outputs.

    Input:
        prediction: (n_samples,) or (n_samples, 1)
        ground_truth: same shape as prediction

    Output:
        scores: (n_samples,) where scores[i] = |prediction[i] - ground_truth[i]|
    """

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: prediction {tuple(prediction.shape)} "
                f"!= ground_truth {tuple(ground_truth.shape)}"
            )
        return (prediction - ground_truth).abs().reshape(prediction.shape[0], -1).squeeze(-1)


class SpatialFieldScoreStub:
    """
    Stub for future spatial nonconformity functions.

    Expected input shape (example):
        prediction, ground_truth: (n_samples, channels, lat, lon)

    Expected output shape:
        scores: (n_samples,)
    """

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement spatial nonconformity score.")


class TrajectoryScoreStub:
    """
    Stub for future trajectory/time-series nonconformity functions.

    Expected input shape (example):
        prediction, ground_truth: (n_samples, n_timepoints[, state_dim])

    Expected output shape:
        scores: (n_samples,)
    """

    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: implement trajectory nonconformity score.")

