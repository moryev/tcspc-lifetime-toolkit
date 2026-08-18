"""Feature extraction for TCSPC photon-count histograms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tcspc_toolkit.preprocessing import (
    detect_peak,
    validate_histogram,
)


_FEATURE_COLUMNS = (
    "total_counts",
    "peak_height",
    "peak_time_ns",
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

    Notes
    -----
    Feature extraction does not perform preprocessing automatically.
    In particular, this function does not subtract background, normalize
    counts, crop the histogram, align the time axis, or rebin the data.

    Any preprocessing required for a particular analysis should be
    performed explicitly before feature extraction so that the scientific
    meaning of the resulting features remains traceable.
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

    peak_index = detect_peak(counts_array)

    features = {
        "total_counts": float(np.sum(counts_array)),
        "peak_height": float(counts_array[peak_index]),
        "peak_time_ns": float(time_array[peak_index]),
    }

    return pd.DataFrame(
        [features],
        columns=list(_FEATURE_COLUMNS),
    )
