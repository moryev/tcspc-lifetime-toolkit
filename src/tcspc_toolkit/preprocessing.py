"""Validation and preprocessing utilities for TCSPC histograms."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from tcspc_toolkit.config import CountNormalization
from tcspc_toolkit.exceptions import InvalidHistogramError


logger = logging.getLogger(__name__)


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

    background = float(
        np.mean(counts_array[start_bin:stop_bin])
    )

    logger.info(
        "Estimated background from bins %d-%d",
        start_bin,
        stop_bin - 1,
    )

    logger.debug(
        "Estimated background level: %.6g counts/bin",
        background,
    )

    return background


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


def detect_peak(
    counts: ArrayLike,
) -> int:
    """Return the index of the maximum histogram count.

    Parameters
    ----------
    counts:
        One-dimensional array containing non-negative raw photon counts.

    Returns
    -------
    int
        Index of the first bin containing the maximum count.

    Raises
    ------
    InvalidHistogramError
        If ``counts`` does not represent valid raw histogram counts or is
        empty.

    Notes
    -----
    Peak detection is currently based on a simple discrete maximum. No
    smoothing, interpolation, or noise-aware peak estimation is applied.

    If multiple bins share the same maximum count, the first maximum is
    returned, following ``numpy.argmax`` behaviour.
    """
    counts_array = _validate_raw_counts(counts)

    if counts_array.size == 0:
        raise InvalidHistogramError(
            "counts must not be empty."
        )

    peak_index = int(np.argmax(counts_array))

    logger.info(
        "Peak detected at bin %d",
        peak_index,
    )

    return peak_index


def align_to_irf(
    time: ArrayLike,
    irf: ArrayLike,
) -> NDArray[np.float64]:
    """Align the time coordinate to the discrete IRF peak.

    The time axis is translated so that the bin containing the maximum IRF
    value occurs at time zero. The IRF itself is not shifted or
    interpolated.

    Parameters
    ----------
    time:
        One-dimensional, finite, strictly increasing time coordinates.
    irf:
        One-dimensional, finite, non-negative instrument response function
        evaluated at the supplied time coordinates.

    Returns
    -------
    NDArray[np.float64]
        New time coordinates with the IRF peak located at zero.

    Raises
    ------
    ValueError
        If the input arrays are invalid, have different shapes, or if the
        IRF has no positive values.

    Notes
    -----
    This operation changes only the time-coordinate origin. It does not
    modify or interpolate measured photon counts or the IRF.

    This nominal alignment is distinct from the ``temporal_shift`` parameter
    used in reconvolution fitting. The latter may still estimate a residual
    offset between the measured IRF and fluorescence histogram.
    """
    try:
        time_array = np.asarray(time, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time must contain numeric values."
        ) from exc

    try:
        irf_array = np.asarray(irf, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "irf must contain numeric values."
        ) from exc

    if time_array.ndim != 1:
        raise ValueError(
            "time must be a one-dimensional array."
        )

    if irf_array.ndim != 1:
        raise ValueError(
            "irf must be a one-dimensional array."
        )

    if time_array.shape != irf_array.shape:
        raise ValueError(
            "time and irf must have the same shape."
        )

    if time_array.size < 2:
        raise ValueError(
            "time and irf must contain at least two values."
        )

    if not np.all(np.isfinite(time_array)):
        raise ValueError(
            "time must contain only finite values."
        )

    if not np.all(np.isfinite(irf_array)):
        raise ValueError(
            "irf must contain only finite values."
        )

    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError(
            "time must be strictly increasing."
        )

    if np.any(irf_array < 0.0):
        raise ValueError(
            "irf values must be non-negative."
        )

    if not np.any(irf_array > 0.0):
        raise ValueError(
            "irf must contain at least one positive value."
        )

    peak_index = int(np.argmax(irf_array))
    peak_time = time_array[peak_index]

    return time_array - peak_time


def crop_time_window(
    time: ArrayLike,
    counts: ArrayLike,
    start_time: float,
    stop_time: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Crop a raw TCSPC histogram to a selected time window.

    The selected interval follows half-open semantics:
    ``[start_time, stop_time)``. A bin is retained when its time
    coordinate is greater than or equal to ``start_time`` and strictly
    smaller than ``stop_time``.

    Parameters
    ----------
    time:
        One-dimensional array of histogram time-bin coordinates.
    counts:
        One-dimensional array of raw photon counts.
    start_time:
        Lower boundary of the selected time window. This boundary is
        included.
    stop_time:
        Upper boundary of the selected time window. This boundary is
        excluded.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Cropped time coordinates and corresponding photon counts.

    Raises
    ------
    InvalidHistogramError
        If ``time`` and ``counts`` do not represent a valid raw TCSPC
        histogram.
    ValueError
        If either boundary is non-finite, if ``start_time`` is not smaller
        than ``stop_time``, or if the selected interval contains no bins.

    Notes
    -----
    The input arrays are not modified.

    Cropping a measured histogram is useful for visualization, exporting
    subsets, exploratory analysis, and preparation of machine-learning
    inputs.

    For reconvolution fitting, the convolution model should generally be
    constructed on the full time axis and the fitting window applied only
    when comparing observed and expected counts. Cropping before
    convolution can introduce edge effects.
    """
    validate_histogram(
        time=time,
        counts=counts,
    )

    if not np.isfinite(start_time):
        raise ValueError("start_time must be finite")

    if not np.isfinite(stop_time):
        raise ValueError("stop_time must be finite")

    if start_time >= stop_time:
        raise ValueError(
            "start_time must be smaller than stop_time"
        )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    mask = (
        (time_array >= start_time)
        & (time_array < stop_time)
    )

    if not np.any(mask):
        raise ValueError(
            "selected time window contains no histogram bins"
        )

    return (
        time_array[mask].copy(),
        counts_array[mask].copy(),
    )


def rebin_histogram(
    time: ArrayLike,
    counts: ArrayLike,
    factor: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Rebin a raw TCSPC histogram by summing neighboring bins.

    Parameters
    ----------
    time:
        One-dimensional array of uniformly spaced histogram time-bin
        coordinates.
    counts:
        One-dimensional array of raw photon counts.
    factor:
        Number of consecutive original bins combined into each new bin.
        Must be a positive integer and must divide the original number of
        bins exactly.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Rebinned time coordinates and photon counts. Each new time
        coordinate is the centre of the corresponding group of original
        bins.

    Raises
    ------
    InvalidHistogramError
        If ``time`` and ``counts`` do not represent a valid raw TCSPC
        histogram.
    ValueError
        If ``factor`` is not a positive integer or if the number of
        histogram bins is not divisible by ``factor``.

    Notes
    -----
    Photon counts are rebinned by summation, so the total photon count is
    preserved.

    If the original histogram contains independent Poisson-distributed
    counts, summing neighboring bins preserves the Poisson count-statistics
    interpretation.

    This function is intended for photon-count histograms. Rebinning a
    normalized continuous IRF for reconvolution fitting requires additional
    consideration of bin integration, time-bin width, and convolution
    normalization and is not handled here.
    """
    validate_histogram(
        time=time,
        counts=counts,
    )

    if (
        isinstance(factor, (bool, np.bool_))
        or not isinstance(factor, (int, np.integer))
    ):
        raise ValueError(
            "factor must be a positive integer"
        )

    if factor < 1:
        raise ValueError(
            "factor must be a positive integer"
        )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    if time_array.size % factor != 0:
        raise ValueError(
            "number of histogram bins must be divisible by factor"
        )

    n_rebinned_bins = time_array.size // factor

    rebinned_time = (
        time_array
        .reshape(n_rebinned_bins, factor)
        .mean(axis=1)
    )

    rebinned_counts = (
        counts_array
        .reshape(n_rebinned_bins, factor)
        .sum(axis=1)
    )

    return rebinned_time, rebinned_counts


def normalize_counts(
    counts: ArrayLike,
    mode: CountNormalization = CountNormalization.TOTAL,
) -> NDArray[np.float64]:
    """Normalize histogram counts using the selected normalization mode.

    Parameters
    ----------
    counts
        One-dimensional array containing histogram counts or processed
        count values.
    mode
        Normalization strategy to apply.

    Returns
    -------
    np.ndarray
        Normalized count values as a new float64 array.

    Raises
    ------
    ValueError
        If counts cannot be converted to a one-dimensional finite numeric
        array, if counts are empty, or if the normalization factor is not
        positive.
    TypeError
        If mode is not a CountNormalization value.
    """
    try:
        counts_array = np.asarray(counts, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "counts must contain numeric values."
        ) from exc

    if counts_array.ndim != 1:
        raise ValueError("counts must be one-dimensional.")

    if counts_array.size == 0:
        raise ValueError("counts must not be empty.")

    if not np.all(np.isfinite(counts_array)):
        raise ValueError("counts must contain only finite values.")

    if not isinstance(mode, CountNormalization):
        raise TypeError(
            "mode must be a CountNormalization value."
        )

    if mode is CountNormalization.TOTAL:
        normalization_factor = np.sum(counts_array)

    elif mode is CountNormalization.PEAK:
        normalization_factor = np.max(counts_array)

    else:
        raise ValueError(
            f"Unsupported normalization mode: {mode!r}."
        )

    if normalization_factor <= 0.0:
        raise ValueError(
            "normalization factor must be positive."
        )

    return counts_array / normalization_factor
