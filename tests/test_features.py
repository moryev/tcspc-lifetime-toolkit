"""Tests for TCSPC feature extraction."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import InvalidHistogramError
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
    assert result.shape == (1, 3)


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
