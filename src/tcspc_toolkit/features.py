"""Feature extraction for TCSPC photon-count histograms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.exceptions import FeatureExtractionError
from tcspc_toolkit.preprocessing import (
    detect_peak,
    validate_histogram,
)


FEATURE_NAMES = (
    "total_counts",
    "peak_height",
    "peak_time_ns",
    "mean_arrival_time_ns",
    "arrival_time_variance_ns2",
    "arrival_time_skewness",
    "t10_ns",
    "t25_ns",
    "t50_ns",
    "t75_ns",
    "t90_ns",
    "half_decay_time_ns",
    "tail_log_slope_per_ns",
    "integrated_tail_fraction",
    "early_late_count_ratio",
)


def _photon_arrival_moments(
    time: NDArray[np.float64],
    counts: NDArray[np.float64],
    total_counts: float,
) -> tuple[float, float, float]:
    """Calculate count-weighted photon-arrival moments.

    Parameters
    ----------
    time:
        One-dimensional array of time-bin coordinates.
    counts:
        One-dimensional array of non-negative photon counts.
    total_counts:
        Sum of all histogram counts.

    Returns
    -------
    tuple[float, float, float]
        Mean arrival time, arrival-time variance, and standardized
        arrival-time skewness.

    Raises
    ------
    FeatureExtractionError
        If the photon-arrival distribution or one of its requested
        moments is mathematically undefined.
    """
    if not np.isfinite(total_counts):
        raise FeatureExtractionError(
            "total photon count must be finite."
        )

    if total_counts <= 0.0:
        raise FeatureExtractionError(
            "photon-arrival moments are undefined for an all-zero histogram."
        )

    probabilities = counts / total_counts

    mean_arrival_time = float(
        np.sum(probabilities * time)
    )

    centered_time = time - mean_arrival_time

    variance = float(
        np.sum(
            probabilities * centered_time**2
        )
    )

    if not np.isfinite(mean_arrival_time):
        raise FeatureExtractionError(
            "mean photon-arrival time is not finite."
        )

    if not np.isfinite(variance):
        raise FeatureExtractionError(
            "photon-arrival variance is not finite."
        )

    if variance == 0.0:
        raise FeatureExtractionError(
            "photon-arrival skewness is undefined when variance is zero."
        )

    standard_deviation = np.sqrt(variance)

    third_central_moment = float(
        np.sum(
            probabilities * centered_time**3
        )
    )

    skewness = float(
        third_central_moment / standard_deviation**3
    )

    if not np.isfinite(skewness):
        raise FeatureExtractionError(
            "photon-arrival skewness is not finite."
        )

    return (
        mean_arrival_time,
        variance,
        skewness,
    )


def extract_features(
    time: ArrayLike,
    counts: ArrayLike,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Extract physically interpretable features from a TCSPC histogram.

    Parameters
    ----------
    time:
        One-dimensional array of uniformly spaced time-bin coordinates.
    counts:
        One-dimensional array of raw non-negative photon counts.
    config:
        Configuration defining the temporal regions and minimum
        number of valid tail points used by configurable features.

    Returns
    -------
    pandas.DataFrame
        One-row table containing the extracted histogram features.

    Raises
    ------
    InvalidHistogramError
        If ``time`` and ``counts`` do not represent a valid raw TCSPC
        histogram.
    FeatureExtractionError
        If one or more requested features are mathematically undefined,
        for example for an all-zero histogram or zero arrival-time
        variance.

    Notes
    -----
    Feature extraction does not perform preprocessing automatically.
    In particular, this function does not subtract background, normalize
    counts, crop the histogram, align the time axis, or rebin the data.

    Any preprocessing required for a particular analysis should be
    performed explicitly before feature extraction so that the scientific
    meaning of the resulting features remains traceable.

    The mean photon-arrival time is lifetime-related but should not
    generally be interpreted directly as the fluorescence lifetime.
    Instrument-response broadening and temporal shift, background counts,
    finite acquisition windows, and multi-component decays can all alter
    the photon-arrival moments.

    Tail and early/late features are defined using the temporal
    boundaries supplied through FeatureConfig. These boundaries are
    not inferred automatically from the histogram.
    """
    validate_histogram(
        time=time,
        counts=counts,
    )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    total_counts = float(
        np.sum(counts_array)
    )

    (
        mean_arrival_time,
        arrival_time_variance,
        arrival_time_skewness,
    ) = _photon_arrival_moments(
        time=time_array,
        counts=counts_array,
        total_counts=total_counts,
    )

    peak_index = detect_peak(counts_array)

    tail_slope = tail_log_slope(
        time=time_array,
        counts=counts_array,
        tail_start_ns=config.tail_start_ns,
        min_points=config.min_tail_points,
    )

    tail_fraction = integrated_tail_fraction(
        time=time_array,
        counts=counts_array,
        tail_start_ns=config.tail_start_ns,
    )

    early_late_ratio = early_late_count_ratio(
        time=time_array,
        counts=counts_array,
        early_stop_ns=config.early_stop_ns,
        late_start_ns=config.late_start_ns,
    )

    features = {
        "total_counts": total_counts,
        "peak_height": float(counts_array[peak_index]),
        "peak_time_ns": float(time_array[peak_index]),
        "mean_arrival_time_ns": mean_arrival_time,
        "arrival_time_variance_ns2": arrival_time_variance,
        "arrival_time_skewness": arrival_time_skewness,
        "t10_ns": quantile_arrival_time(
            time_array,
            counts_array,
            0.10,
        ),
        "t25_ns": quantile_arrival_time(
            time_array,
            counts_array,
            0.25,
        ),
        "t50_ns": quantile_arrival_time(
            time_array,
            counts_array,
            0.50,
        ),
        "t75_ns": quantile_arrival_time(
            time_array,
            counts_array,
            0.75,
        ),
        "t90_ns": quantile_arrival_time(
            time_array,
            counts_array,
            0.90,
        ),
        "half_decay_time_ns": half_decay_time(
            time_array,
            counts_array,
        ),
        "tail_log_slope_per_ns": tail_slope,
        "integrated_tail_fraction": tail_fraction,
        "early_late_count_ratio": early_late_ratio,
    }

    return pd.DataFrame(
        [features],
        columns=list(FEATURE_NAMES),
    )


def extract_feature_table(
    histograms: ArrayLike,
    time: ArrayLike,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Extract engineered features from multiple TCSPC histograms.

    Parameters
    ----------
    histograms:
        Two-dimensional array of raw non-negative photon counts.
        Each row represents one TCSPC histogram and each column
        corresponds to one time bin.
    time:
        One-dimensional array of time-bin coordinates shared by
        all histograms.
    config:
        Configuration defining the temporal regions and minimum
        number of valid tail points used by configurable features.

    Returns
    -------
    pandas.DataFrame
        Feature table with one row per histogram and columns ordered
        according to FEATURE_NAMES.

    Raises
    ------
    ValueError
        If ``histograms`` is not two-dimensional, contains no
        histograms, or its number of bins does not match ``time``.
    InvalidHistogramError
        If an individual histogram is not a valid raw TCSPC
        histogram.
    FeatureExtractionError
        If one or more requested features are mathematically
        undefined for an individual histogram.

    Notes
    -----
    All histograms must share the same time axis.

    This function delegates feature calculation to ``extract_features``
    so that single-histogram and batch extraction use exactly the same
    feature definitions.

    The returned table contains engineered histogram features only.
    Target variables and dataset metadata should be stored separately.

    For example, a simulated ground-truth lifetime must not be included
    in the feature table because doing so would introduce direct target
    leakage.

    Other simulation parameters, such as the true background level,
    instrument-response width, or configured photon-count level, should
    also not be treated automatically as model features. Such quantities
    may be useful for evaluation, stratification, and error analysis, but
    they may not be available for experimental measurements.

    Only quantities that are legitimately available at inference time
    should be considered candidate model inputs.
    """
    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    histograms_array = np.asarray(
        histograms,
        dtype=np.float64,
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be a one-dimensional array."
        )

    if histograms_array.ndim != 2:
        raise ValueError(
            "histograms must be a two-dimensional array."
        )

    if histograms_array.shape[0] == 0:
        raise ValueError(
            "histograms must contain at least one histogram."
        )

    if histograms_array.shape[1] != time_array.size:
        raise ValueError(
            "each histogram must contain the same number "
            "of bins as time."
        )

    feature_rows = [
        extract_features(
            time=time_array,
            counts=counts,
            config=config,
        )
        for counts in histograms_array
    ]

    return pd.concat(
        feature_rows,
        ignore_index=True,
    )


def _normalized_cumulative_counts(
    counts: np.ndarray,
) -> np.ndarray:
    total_counts = np.sum(counts)

    if total_counts <= 0:
        raise FeatureExtractionError(
            "Quantile arrival times require at least one detected photon."
        )

    cumulative_counts = np.cumsum(
        counts,
        dtype=np.float64,
    )

    return cumulative_counts / total_counts


def quantile_arrival_time(
    time: ArrayLike,
    counts: ArrayLike,
    quantile: float = 0.5,
) -> float:
    """Return the interpolated photon-arrival time for a quantile."""

    if not np.isfinite(quantile):
        raise ValueError(
            "quantile must be finite."
        )

    if quantile <= 0.0 or quantile > 1.0:
        raise ValueError(
            "quantile must satisfy 0 < quantile <= 1."
        )

    validate_histogram(
        time=time,
        counts=counts,
    )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    cumulative_fraction = _normalized_cumulative_counts(
        counts_array
    )

    index = int(
        np.searchsorted(
            cumulative_fraction,
            quantile,
            side="left",
        )
    )

    if index == 0:
        return float(time_array[0])

    if cumulative_fraction[index] == quantile:
        return float(time_array[index])

    fraction_left = cumulative_fraction[index - 1]
    fraction_right = cumulative_fraction[index]

    time_left = time_array[index - 1]
    time_right = time_array[index]

    interpolation_fraction = (
        (quantile - fraction_left)
        / (fraction_right - fraction_left)
    )

    quantile_time = (
        time_left
        + interpolation_fraction
        * (time_right - time_left)
    )

    return float(quantile_time)


def half_decay_time(
    time: ArrayLike,
    counts: ArrayLike,
) -> float:
    """Return the post-peak half-decay time."""

    validate_histogram(
        time=time,
        counts=counts,
    )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    peak_index = detect_peak(counts_array)
    peak_height = float(counts_array[peak_index])

    if peak_height <= 0.0:
        raise FeatureExtractionError(
            "Half-decay time requires a positive peak."
        )

    if peak_index == counts_array.size - 1:
        raise FeatureExtractionError(
            "Half-decay time cannot be determined because "
            "the peak occurs in the final bin."
        )

    half_height = 0.5 * peak_height

    post_peak_counts = counts_array[
        peak_index + 1:
    ]

    crossing_candidates = np.flatnonzero(
        post_peak_counts <= half_height
    )

    if crossing_candidates.size == 0:
        raise FeatureExtractionError(
            "No post-peak half-height crossing was found."
        )

    right_index = (
        peak_index
        + 1
        + int(crossing_candidates[0])
    )

    left_index = right_index - 1

    count_left = float(
        counts_array[left_index]
    )
    count_right = float(
        counts_array[right_index]
    )

    time_left = float(
        time_array[left_index]
    )
    time_right = float(
        time_array[right_index]
    )

    interpolation_fraction = (
        (half_height - count_left)
        / (count_right - count_left)
    )

    crossing_time = (
        time_left
        + interpolation_fraction
        * (time_right - time_left)
    )

    peak_time = float(
        time_array[peak_index]
    )

    return float(
        crossing_time - peak_time
    )


def tail_log_slope(
    time: ArrayLike,
    counts: ArrayLike,
    tail_start_ns: float,
    min_points: int = 3,
) -> float:
    """Return the linear slope of log-counts in the decay tail."""

    if not np.isfinite(tail_start_ns):
        raise ValueError(
            "tail_start_ns must be finite."
        )

    if type(min_points) is not int:
        raise ValueError(
            "min_points must be an integer."
        )

    if min_points < 3:
        raise ValueError(
            "min_points must be at least 3."
        )

    validate_histogram(
        time=time,
        counts=counts,
    )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    tail_mask = (
        time_array >= tail_start_ns
    )

    if not np.any(tail_mask):
        raise FeatureExtractionError(
            "Tail region contains no histogram bins."
        )

    fit_mask = (
        tail_mask
        & (counts_array > 0.0)
    )

    tail_time = time_array[
        fit_mask
    ]

    tail_counts = counts_array[
        fit_mask
    ]

    if tail_time.size < min_points:
        raise FeatureExtractionError(
            "Insufficient positive-count bins "
            "for tail-slope regression."
        )

    log_tail_counts = np.log(
        tail_counts
    )

    slope, _ = np.polyfit(
        tail_time,
        log_tail_counts,
        deg=1,
    )

    if not np.isfinite(slope):
        raise FeatureExtractionError(
            "Tail log slope is not finite."
        )

    return float(slope)


def integrated_tail_fraction(
    time: ArrayLike,
    counts: ArrayLike,
    tail_start_ns: float,
) -> float:
    """Return the fraction of total counts in the configured decay tail."""

    if not np.isfinite(tail_start_ns):
        raise ValueError(
            "tail_start_ns must be finite."
        )

    validate_histogram(
        time=time,
        counts=counts,
    )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    tail_mask = (
        time_array >= tail_start_ns
    )

    if not np.any(tail_mask):
        raise FeatureExtractionError(
            "Tail region contains no histogram bins."
        )

    total_counts = float(
        np.sum(counts_array)
    )

    if total_counts <= 0.0:
        raise FeatureExtractionError(
            "Tail fraction requires at least one detected photon."
        )

    tail_counts = float(
        np.sum(
            counts_array[tail_mask]
        )
    )

    return tail_counts / total_counts


def early_late_count_ratio(
    time: ArrayLike,
    counts: ArrayLike,
    early_stop_ns: float,
    late_start_ns: float,
) -> float:
    """Return the ratio of early-region counts to late-region counts."""

    if not np.isfinite(early_stop_ns):
        raise ValueError(
            "early_stop_ns must be finite."
        )

    if not np.isfinite(late_start_ns):
        raise ValueError(
            "late_start_ns must be finite."
        )

    if early_stop_ns >= late_start_ns:
        raise ValueError(
            "early_stop_ns must be smaller than late_start_ns."
        )

    validate_histogram(
        time=time,
        counts=counts,
    )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )

    early_mask = (
        time_array <= early_stop_ns
    )

    late_mask = (
        time_array >= late_start_ns
    )

    if not np.any(early_mask):
        raise FeatureExtractionError(
            "Early region contains no histogram bins."
        )

    if not np.any(late_mask):
        raise FeatureExtractionError(
            "Late region contains no histogram bins."
        )

    early_counts = float(
        np.sum(
            counts_array[early_mask]
        )
    )

    late_counts = float(
        np.sum(
            counts_array[late_mask]
        )
    )

    if late_counts <= 0.0:
        raise FeatureExtractionError(
            "Early/late count ratio is undefined because "
            "the late region contains zero counts."
        )

    return early_counts / late_counts
