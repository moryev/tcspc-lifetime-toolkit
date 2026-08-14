"""Tests for TCSPC histogram preprocessing and validation."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import InvalidHistogramError, TCSPCError
from tcspc_toolkit.preprocessing import (
    estimate_background,
    subtract_background,
    validate_histogram,
)


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


def test_estimate_background_recovers_constant_background() -> None:
    counts = np.array(
        [5.0, 5.0, 5.0, 5.0, 20.0, 10.0],
        dtype=np.float64,
    )

    background = estimate_background(
        counts,
        start_bin=0,
        stop_bin=4,
    )

    assert background == 5.0


def test_estimate_background_recovers_zero_background() -> None:
    counts = np.array(
        [0.0, 0.0, 0.0, 10.0, 5.0],
        dtype=np.float64,
    )

    background = estimate_background(
        counts,
        start_bin=0,
        stop_bin=3,
    )

    assert background == 0.0


def test_estimate_background_respects_selected_region() -> None:
    counts = np.array(
        [100.0, 100.0, 10.0, 20.0, 30.0, 100.0],
        dtype=np.float64,
    )

    background = estimate_background(
        counts,
        start_bin=2,
        stop_bin=5,
    )

    assert background == 20.0


@pytest.mark.parametrize(
    ("start_bin", "stop_bin"),
    [
        (-1, 3),
        (3, 3),
        (4, 2),
        (0, 7),
    ],
)
def test_estimate_background_rejects_invalid_bin_interval(
    start_bin: int,
    stop_bin: int,
) -> None:
    counts = np.ones(6, dtype=np.float64)

    with pytest.raises(ValueError):
        estimate_background(
            counts,
            start_bin=start_bin,
            stop_bin=stop_bin,
        )


def test_subtract_background_returns_expected_values() -> None:
    counts = np.array(
        [2.0, 4.0, 8.0],
        dtype=np.float64,
    )

    corrected = subtract_background(
        counts,
        background=3.0,
    )

    expected = np.array(
        [-1.0, 1.0, 5.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(corrected, expected)


def test_subtract_background_does_not_modify_input() -> None:
    counts = np.array(
        [2.0, 4.0, 8.0],
        dtype=np.float64,
    )
    original = counts.copy()

    subtract_background(
        counts,
        background=3.0,
    )

    np.testing.assert_array_equal(counts, original)


def test_subtract_background_allows_negative_values() -> None:
    counts = np.array(
        [1.0, 2.0, 5.0],
        dtype=np.float64,
    )

    corrected = subtract_background(
        counts,
        background=3.0,
    )

    expected = np.array(
        [-2.0, -1.0, 2.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(corrected, expected)


def test_subtract_background_rejects_negative_background() -> None:
    counts = np.ones(5, dtype=np.float64)

    with pytest.raises(ValueError):
        subtract_background(
            counts,
            background=-1.0,
        )


def test_subtract_background_rejects_nonfinite_background() -> None:
    counts = np.ones(5, dtype=np.float64)

    with pytest.raises(ValueError):
        subtract_background(
            counts,
            background=np.nan,
        )
