"""Baseline estimators for TCSPC lifetime benchmarking."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.dummy import DummyRegressor


FloatArray = NDArray[np.float64]


def predict_constant_mean_baseline(
    *,
    y_train: ArrayLike,
    n_predictions: int,
) -> FloatArray:
    """Predict the training-set mean lifetime for every sample.

    Parameters
    ----------
    y_train:
        Training-set fluorescence lifetimes.
    n_predictions:
        Number of predictions to generate.

    Returns
    -------
    numpy.ndarray
        Constant lifetime predictions with shape
        ``(n_predictions,)``.

    Notes
    -----
    This baseline uses only the training-set lifetime distribution.
    It does not use TCSPC histogram features.
    """
    y_train_array = np.asarray(
        y_train,
        dtype=np.float64,
    )

    if y_train_array.ndim != 1:
        raise ValueError(
            "y_train must be one-dimensional."
        )

    if y_train_array.size == 0:
        raise ValueError(
            "y_train must contain at least one lifetime."
        )

    if not np.all(
        np.isfinite(y_train_array)
    ):
        raise ValueError(
            "y_train must contain only finite values."
        )

    if np.any(y_train_array <= 0.0):
        raise ValueError(
            "Training lifetimes must be strictly positive."
        )

    if type(n_predictions) is not int:
        raise ValueError(
            "n_predictions must be an integer."
        )

    if n_predictions < 1:
        raise ValueError(
            "n_predictions must be at least 1."
        )

    X_train_dummy = np.zeros(
        (y_train_array.size, 1),
        dtype=np.float64,
    )

    X_prediction_dummy = np.zeros(
        (n_predictions, 1),
        dtype=np.float64,
    )

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        X_train_dummy,
        y_train_array,
    )

    predictions = model.predict(
        X_prediction_dummy
    )

    return np.asarray(
        predictions,
        dtype=np.float64,
    )


def estimate_lifetime_from_mean_arrival(
    *,
    mean_arrival_time_ns: ArrayLike,
    peak_time_ns: ArrayLike,
) -> FloatArray:
    """Estimate lifetime from mean and peak photon-arrival times.

    The estimator is defined as

    ``lifetime = mean_arrival_time - peak_time``.

    For an ideal mono-exponential decay beginning at ``t0``,
    the mean photon-arrival time approaches ``t0 + lifetime``,
    while the histogram peak occurs near ``t0``.

    Parameters
    ----------
    mean_arrival_time_ns:
        Count-weighted mean photon-arrival times.
    peak_time_ns:
        Observed histogram peak times.

    Returns
    -------
    numpy.ndarray
        Estimated lifetimes with the same shape as the input arrays.

    Notes
    -----
    No background, IRF, or simulation-metadata correction is applied.
    Therefore background, finite acquisition windows, IRF broadening,
    temporal discretization, and noise may bias the estimates.

    Non-positive estimates are not clipped so that failure modes remain
    visible during benchmarking.
    """
    mean_arrival_array = np.asarray(
        mean_arrival_time_ns,
        dtype=np.float64,
    )

    peak_time_array = np.asarray(
        peak_time_ns,
        dtype=np.float64,
    )

    if mean_arrival_array.ndim != 1:
        raise ValueError(
            "mean_arrival_time_ns must be one-dimensional."
        )

    if peak_time_array.ndim != 1:
        raise ValueError(
            "peak_time_ns must be one-dimensional."
        )

    if mean_arrival_array.shape != peak_time_array.shape:
        raise ValueError(
            "mean_arrival_time_ns and peak_time_ns "
            "must have identical shapes."
        )

    if mean_arrival_array.size == 0:
        raise ValueError(
            "Input arrays must contain at least one sample."
        )

    if not np.all(
        np.isfinite(mean_arrival_array)
    ):
        raise ValueError(
            "mean_arrival_time_ns must contain only finite values."
        )

    if not np.all(
        np.isfinite(peak_time_array)
    ):
        raise ValueError(
            "peak_time_ns must contain only finite values."
        )

    lifetime_estimates = (
        mean_arrival_array
        - peak_time_array
    )

    if not np.all(
        np.isfinite(lifetime_estimates)
    ):
        raise RuntimeError(
            "Mean-arrival lifetime estimates are not finite."
        )

    return lifetime_estimates
