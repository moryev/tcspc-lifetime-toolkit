"""Validation and preprocessing utilities for TCSPC histograms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from tcspc_toolkit.exceptions import InvalidHistogramError


_TIME_BIN_RTOL = 1e-7
_TIME_BIN_ATOL = 1e-12
_INTEGER_COUNT_ATOL = 1e-9


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

    try:
        counts_array = np.asarray(counts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidHistogramError(
            "counts must contain numeric values."
        ) from exc

    if time_array.ndim != 1:
        raise InvalidHistogramError(
            "time must be a one-dimensional array."
        )

    if counts_array.ndim != 1:
        raise InvalidHistogramError(
            "counts must be a one-dimensional array."
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

    if not np.all(np.isfinite(counts_array)):
        raise InvalidHistogramError(
            "counts must contain only finite values."
        )

    if np.any(counts_array < 0.0):
        raise InvalidHistogramError(
            "counts must be non-negative."
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

    if not np.allclose(
        counts_array,
        np.rint(counts_array),
        rtol=0.0,
        atol=_INTEGER_COUNT_ATOL,
    ):
        raise InvalidHistogramError(
            "raw histogram counts must be integer-valued."
        )
