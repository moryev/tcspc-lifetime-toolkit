"""Tests for TCSPC histogram preprocessing and validation."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import InvalidHistogramError, TCSPCError
from tcspc_toolkit.preprocessing import validate_histogram


@pytest.fixture
def valid_histogram() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )
    counts = np.array(
        [0.0, 4.0, 12.0, 7.0, 2.0],
        dtype=np.float64,
    )

    return time, counts


def test_validate_histogram_accepts_valid_histogram(
    valid_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = valid_histogram

    validate_histogram(
        time=time,
        counts=counts,
    )


def test_validate_histogram_rejects_negative_counts(
    valid_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = valid_histogram
    counts = counts.copy()
    counts[2] = -1.0

    with pytest.raises(
        InvalidHistogramError,
        match="counts must be non-negative",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_nan(
    valid_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = valid_histogram
    counts = counts.copy()
    counts[1] = np.nan

    with pytest.raises(
        InvalidHistogramError,
        match="counts must contain only finite values",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_inf(
    valid_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    time, counts = valid_histogram
    time = time.copy()
    time[2] = np.inf

    with pytest.raises(
        InvalidHistogramError,
        match="time must contain only finite values",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_mismatched_lengths() -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="time and counts must have the same length",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_non_monotonic_time() -> None:
    time = np.array(
        [0.0, 0.1, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="time must be strictly increasing",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_non_uniform_time() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.31],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="time bins must be approximately uniform",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_two_dimensional_time() -> None:
    time = np.zeros((2, 3), dtype=np.float64)
    counts = np.zeros(6, dtype=np.float64)

    with pytest.raises(
        InvalidHistogramError,
        match="time must be a one-dimensional array",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_two_dimensional_counts() -> None:
    time = np.arange(6, dtype=np.float64)
    counts = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(
        InvalidHistogramError,
        match="counts must be a one-dimensional array",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_empty_histogram() -> None:
    time = np.array([], dtype=np.float64)
    counts = np.array([], dtype=np.float64)

    with pytest.raises(
        InvalidHistogramError,
        match="histogram must contain at least two bins",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_validate_histogram_rejects_fractional_raw_counts() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.array(
        [0.0, 3.0, 7.5, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="raw histogram counts must be integer-valued",
    ):
        validate_histogram(
            time=time,
            counts=counts,
        )


def test_invalid_histogram_error_is_a_value_error() -> None:
    assert issubclass(InvalidHistogramError, ValueError)


def test_invalid_histogram_error_is_a_tcspc_error() -> None:
    assert issubclass(InvalidHistogramError, TCSPCError)
