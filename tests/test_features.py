"""Tests for TCSPC feature extraction."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import (
    FeatureExtractionError,
    InvalidHistogramError,
)
from tcspc_toolkit.features import extract_features


@pytest.fixture
def simple_histogram() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 4.0, 9.0, 3.0, 1.0],
        dtype=np.float64,
    )

    return time, counts


def test_extract_features_returns_dataframe(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 6)


def test_extract_features_has_stable_columns(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert list(result.columns) == [
        "total_counts",
        "peak_height",
        "peak_time_ns",
        "mean_arrival_time_ns",
        "arrival_time_variance_ns2",
        "arrival_time_skewness",
    ]


def test_extract_features_calculates_total_counts(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert result.loc[0, "total_counts"] == 18.0


def test_extract_features_calculates_peak_height(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert result.loc[0, "peak_height"] == 9.0


def test_extract_features_calculates_peak_time(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert result.loc[0, "peak_time_ns"] == 0.2


def test_extract_features_does_not_modify_inputs(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = simple_histogram

    original_time = time.copy()
    original_counts = counts.copy()

    extract_features(
        time=time,
        counts=counts,
    )

    np.testing.assert_array_equal(
        time,
        original_time,
    )
    np.testing.assert_array_equal(
        counts,
        original_counts,
    )


def test_extract_features_rejects_invalid_histogram() -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, -2.0, 3.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="counts must be non-negative",
    ):
        extract_features(
            time=time,
            counts=counts,
        )


@pytest.fixture
def moment_histogram() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    time = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )
    counts = np.array(
        [3.0, 1.0, 0.0],
        dtype=np.float64,
    )

    return time, counts


def test_extract_features_calculates_mean_arrival_time(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = moment_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert result.loc[
        0,
        "mean_arrival_time_ns",
    ] == pytest.approx(0.25)


def test_extract_features_calculates_arrival_time_variance(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = moment_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    assert result.loc[
        0,
        "arrival_time_variance_ns2",
    ] == pytest.approx(0.1875)


def test_extract_features_calculates_arrival_time_skewness(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = moment_histogram

    result = extract_features(
        time=time,
        counts=counts,
    )

    expected_skewness = 2.0 / np.sqrt(3.0)

    assert result.loc[
        0,
        "arrival_time_skewness",
    ] == pytest.approx(expected_skewness)


def test_arrival_moments_are_invariant_to_count_scaling(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = moment_histogram

    original = extract_features(
        time=time,
        counts=counts,
    )

    scaled = extract_features(
        time=time,
        counts=10.0 * counts,
    )

    moment_columns = [
        "mean_arrival_time_ns",
        "arrival_time_variance_ns2",
        "arrival_time_skewness",
    ]

    np.testing.assert_allclose(
        original.loc[0, moment_columns].to_numpy(
            dtype=np.float64
        ),
        scaled.loc[0, moment_columns].to_numpy(
            dtype=np.float64
        ),
    )


def test_extract_features_rejects_all_zero_histogram() -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.zeros(
        3,
        dtype=np.float64,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="all-zero histogram",
    ):
        extract_features(
            time=time,
            counts=counts,
        )


def test_extract_features_rejects_zero_arrival_time_variance() -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [0.0, 10.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="variance is zero",
    ):
        extract_features(
            time=time,
            counts=counts,
        )


def test_extract_features_rejects_non_finite_counts() -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, np.nan, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="counts must contain only finite values",
    ):
        extract_features(
            time=time,
            counts=counts,
        )
