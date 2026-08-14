"""Validation and preprocessing utilities for TCSPC histograms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import InvalidHistogramError


_TIME_BIN_RTOL = 1e-7
_TIME_BIN_ATOL = 1e-12
_INTEGER_COUNT_ATOL = 1e-9


def _validate_raw_counts(
    counts: ArrayLike,
) -> NDArray[np.float64]:
    """Validate raw measured photon counts and return them as an array."""
    try:
        counts_array = np.asarray(counts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidHistogramError(
            "counts must contain numeric values."
        ) from exc

    if counts_array.ndim != 1:
        raise InvalidHistogramError(
            "counts must be a one-dimensional array."
        )

    if not np.all(np.isfinite(counts_array)):
        raise InvalidHistogramError(
            "counts must contain only finite values."
        )

    if np.any(counts_array < 0.0):
        raise InvalidHistogramError(
            "counts must be non-negative."
        )

    if not np.allclose(
        counts_array,
        np.rint(counts_array),
        rtol=0.0,
        atol=_INTEGER_COUNT_ATOL,
    ):
        raise InvalidHistogramError(
            "raw histogram counts must be integer-valued."
        )

    return counts_array


def validate_histogram(
    time: ArrayLike,
    counts: ArrayLike,
) -> None:
    """Validate a raw measured TCSPC histogram.

    Parameters
    ----------
    time
        One-dimensional, finite, strictly increasing, and approximately
        uniformly spaced time-bin coordinates.
    counts
        One-dimensional, finite, non-negative, integer-valued or
        integer-like photon counts.

    Raises
    ------
    InvalidHistogramError
        If the supplied arrays do not represent a valid raw TCSPC
        histogram.

    Notes
    -----
    This function is intended for raw measured photon-count histograms.
    Model predictions and processed signals may legitimately contain
    non-integer floating-point values and should not be validated with
    this function.
    """
    try:
        time_array = np.asarray(time, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidHistogramError(
            "time must contain numeric values."
        ) from exc

    counts_array = _validate_raw_counts(counts)

    if time_array.ndim != 1:
        raise InvalidHistogramError(
            "time must be a one-dimensional array."
        )

    if time_array.size != counts_array.size:
        raise InvalidHistogramError(
            "time and counts must have the same length."
        )

    if time_array.size < 2:
        raise InvalidHistogramError(
            "histogram must contain at least two bins."
        )

    if not np.all(np.isfinite(time_array)):
        raise InvalidHistogramError(
            "time must contain only finite values."
        )

    bin_widths = np.diff(time_array)

    if np.any(bin_widths <= 0.0):
        raise InvalidHistogramError(
            "time must be strictly increasing."
        )

    if not np.allclose(
        bin_widths,
        bin_widths[0],
        rtol=_TIME_BIN_RTOL,
        atol=_TIME_BIN_ATOL,
    ):
        raise InvalidHistogramError(
            "time bins must be approximately uniform."
        )


def estimate_background(
    counts: ArrayLike,
    start_bin: int,
    stop_bin: int,
) -> float:
    """Estimate a stationary background level from a selected bin interval.

    The background is estimated as the arithmetic mean count per bin over
    the half-open interval ``[start_bin, stop_bin)``.

    Parameters
    ----------
    counts:
        One-dimensional array containing non-negative histogram counts.
    start_bin:
        Index of the first bin included in the background region.
    stop_bin:
        Index immediately after the last bin included in the background
        region.

    Returns
    -------
    float
        Estimated mean background count per bin.

    Raises
    ------
    ValueError
        If the selected bin interval is empty, reversed, or outside the
        histogram bounds.

    Notes
    -----
    The background region must be selected explicitly. This function does
    not attempt to determine automatically which histogram bins contain
    only background.

    For Poisson maximum-likelihood fitting, background-subtracted counts
    should generally not be used as the observed data. Instead, the raw
    photon counts should be fitted with a model that includes the
    background explicitly. The estimate returned here can provide an
    initial guess for that background parameter.
    """
    counts_array = _validate_raw_counts(counts)

    if start_bin < 0:
        raise ValueError("start_bin must be greater than or equal to zero")

    if stop_bin > counts_array.size:
        raise ValueError(
            "stop_bin must not exceed the number of histogram bins"
        )

    if start_bin >= stop_bin:
        raise ValueError("start_bin must be smaller than stop_bin")

    return float(
        np.mean(counts_array[start_bin:stop_bin])
    )


def subtract_background(
    counts: ArrayLike,
    background: float,
) -> NDArray[np.float64]:
    """Subtract a constant background level from histogram counts.

    Parameters
    ----------
    counts:
        One-dimensional array containing non-negative histogram counts.
    background:
        Constant background level in counts per bin.

    Returns
    -------
    NDArray[np.float64]
        Background-corrected histogram. Negative values are preserved.

    Raises
    ------
    ValueError
        If ``background`` is negative or non-finite.

    Notes
    -----
    The input array is not modified.

    Negative values produced by background subtraction are intentionally
    retained because they represent statistical fluctuations around the
    estimated background level.

    Background subtraction should generally not be applied before Poisson
    maximum-likelihood fitting. Poisson reconvolution fitting should use
    the original photon counts and include the background directly in the
    expected-count model.
    """
    counts_array = _validate_raw_counts(counts)

    if not np.isfinite(background):
        raise ValueError("background must be finite")

    if background < 0:
        raise ValueError("background must be non-negative")

    corrected = counts_array.copy()
    corrected -= background

    return corrected
