from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Summary metrics for lifetime regression."""

    mae_ns: float
    median_absolute_error_ns: float
    mean_relative_error: float
    median_relative_error: float
    r2: float


def normalize_histograms(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize every TCSPC histogram by its total count."""
    # TODO: Once Day 41's new representation API is complete, one can decide whether to:
    #       delegate old normalize_histograms()
    #         ↓
    #       normalize_histogram_batch()
    #       or deprecate/remove the older helper
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if np.any(X < 0):
        raise ValueError("Photon counts cannot be negative.")

    totals = X.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(totals == 0):
        raise ValueError(
            "Cannot normalize a histogram with zero total counts."
        )

    return X / totals


def evaluate_regression(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> RegressionMetrics:
    """Evaluate predicted lifetimes."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if np.any(y_true <= 0):
        raise ValueError(
            "True lifetimes must be strictly positive."
        )

    relative_errors = (
        np.abs(y_pred - y_true) / y_true
    )

    return RegressionMetrics(
        mae_ns=float(
            mean_absolute_error(y_true, y_pred)
        ),
        median_absolute_error_ns=float(
            median_absolute_error(y_true, y_pred)
        ),
        mean_relative_error=float(
            np.mean(relative_errors)
        ),
        median_relative_error=float(
            np.median(relative_errors)
        ),
        r2=float(
            r2_score(y_true, y_pred)
        ),
    )
