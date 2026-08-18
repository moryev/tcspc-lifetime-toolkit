"""Feature extraction for TCSPC photon-count histograms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tcspc_toolkit.exceptions import FeatureExtractionError
from tcspc_toolkit.preprocessing import (
    detect_peak,
    validate_histogram,
)


_FEATURE_COLUMNS = (
    "total_counts",
    "peak_height",
    "peak_time_ns",
    "mean_arrival_time_ns",
    "arrival_time_variance_ns2",
    "arrival_time_skewness",
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
) -> pd.DataFrame:
    """Extract physically interpretable features from a TCSPC histogram.

    Parameters
    ----------
    time:
        One-dimensional array of uniformly spaced time-bin coordinates.
    counts:
        One-dimensional array of raw non-negative photon counts.

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

    features = {
        "total_counts": total_counts,
        "peak_height": float(counts_array[peak_index]),
        "peak_time_ns": float(time_array[peak_index]),
        "mean_arrival_time_ns": mean_arrival_time,
        "arrival_time_variance_ns2": arrival_time_variance,
        "arrival_time_skewness": arrival_time_skewness,
    }

    return pd.DataFrame(
        [features],
        columns=list(_FEATURE_COLUMNS),
    )
