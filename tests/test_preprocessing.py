"""Tests for TCSPC histogram preprocessing and validation."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import InvalidHistogramError, TCSPCError
from tcspc_toolkit.preprocessing import (
    estimate_background,
    subtract_background,
    validate_histogram,
    detect_peak,
    align_to_irf,
    crop_time_window,
    rebin_histogram,
    CountNormalization,
    normalize_counts,
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


def test_detect_peak_identifies_peak_location() -> None:
    counts = np.array(
        [1.0, 4.0, 12.0, 7.0, 2.0],
        dtype=np.float64,
    )

    peak_index = detect_peak(counts)

    assert peak_index == 2


def test_detect_peak_returns_first_maximum() -> None:
    counts = np.array(
        [1.0, 10.0, 10.0, 4.0],
        dtype=np.float64,
    )

    peak_index = detect_peak(counts)

    assert peak_index == 1


def test_detect_peak_rejects_empty_counts() -> None:
    counts = np.array([], dtype=np.float64)

    with pytest.raises(
        InvalidHistogramError,
        match="counts must not be empty",
    ):
        detect_peak(counts)


def test_align_to_irf_places_peak_at_zero() -> None:
    time = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        dtype=np.float64,
    )
    irf = np.array(
        [0.1, 0.5, 1.0, 4.0, 0.8],
        dtype=np.float64,
    )

    aligned_time = align_to_irf(
        time=time,
        irf=irf,
    )

    peak_index = int(np.argmax(irf))

    assert aligned_time[peak_index] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "peak_index",
    [1, 3],
)
def test_align_to_irf_handles_negative_and_positive_peak_times(
    peak_index: int,
) -> None:
    time = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=np.float64,
    )
    irf = np.zeros(5, dtype=np.float64)
    irf[peak_index] = 1.0

    aligned_time = align_to_irf(
        time=time,
        irf=irf,
    )

    assert aligned_time[peak_index] == pytest.approx(0.0)


def test_align_to_irf_preserves_time_spacing() -> None:
    time = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        dtype=np.float64,
    )
    irf = np.array(
        [0.1, 0.5, 1.0, 4.0, 0.8],
        dtype=np.float64,
    )

    aligned_time = align_to_irf(
        time=time,
        irf=irf,
    )

    np.testing.assert_allclose(
        np.diff(aligned_time),
        np.diff(time),
    )


def test_align_to_irf_does_not_modify_inputs() -> None:
    time = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0],
        dtype=np.float64,
    )
    irf = np.array(
        [0.1, 0.5, 1.0, 4.0, 0.8],
        dtype=np.float64,
    )

    original_time = time.copy()
    original_irf = irf.copy()

    align_to_irf(
        time=time,
        irf=irf,
    )

    np.testing.assert_array_equal(time, original_time)
    np.testing.assert_array_equal(irf, original_irf)


def test_align_to_irf_rejects_zero_irf() -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    irf = np.zeros(3, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="irf must contain at least one positive value",
    ):
        align_to_irf(
            time=time,
            irf=irf,
        )


def test_crop_time_window_selects_expected_bins() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        dtype=np.float64,
    )

    cropped_time, cropped_counts = crop_time_window(
        time=time,
        counts=counts,
        start_time=0.1,
        stop_time=0.4,
    )

    expected_time = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    expected_counts = np.array(
        [2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        cropped_time,
        expected_time,
    )
    np.testing.assert_array_equal(
        cropped_counts,
        expected_counts,
    )


def test_crop_time_window_uses_half_open_interval() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=np.float64,
    )

    cropped_time, _ = crop_time_window(
        time=time,
        counts=counts,
        start_time=0.1,
        stop_time=0.3,
    )

    expected_time = np.array(
        [0.1, 0.2],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        cropped_time,
        expected_time,
    )


def test_crop_time_window_does_not_modify_inputs() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    original_time = time.copy()
    original_counts = counts.copy()

    crop_time_window(
        time=time,
        counts=counts,
        start_time=0.1,
        stop_time=0.3,
    )

    np.testing.assert_array_equal(
        time,
        original_time,
    )
    np.testing.assert_array_equal(
        counts,
        original_counts,
    )


def test_crop_time_window_rejects_reversed_window() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.ones(4, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="start_time must be smaller than stop_time",
    ):
        crop_time_window(
            time=time,
            counts=counts,
            start_time=0.3,
            stop_time=0.1,
        )


def test_crop_time_window_rejects_empty_window() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.ones(4, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="selected time window contains no histogram bins",
    ):
        crop_time_window(
            time=time,
            counts=counts,
            start_time=1.0,
            stop_time=2.0,
        )


def test_rebin_histogram_factor_one_changes_nothing() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 4.0, 7.0, 2.0],
        dtype=np.float64,
    )

    rebinned_time, rebinned_counts = rebin_histogram(
        time=time,
        counts=counts,
        factor=1,
    )

    np.testing.assert_allclose(
        rebinned_time,
        time,
    )
    np.testing.assert_array_equal(
        rebinned_counts,
        counts,
    )


def test_rebin_histogram_factor_two_halves_number_of_bins() -> None:
    time = np.arange(
        0.0,
        0.8,
        0.1,
        dtype=np.float64,
    )
    counts = np.arange(
        1.0,
        9.0,
        dtype=np.float64,
    )

    rebinned_time, rebinned_counts = rebin_histogram(
        time=time,
        counts=counts,
        factor=2,
    )

    assert rebinned_time.size == 4
    assert rebinned_counts.size == 4


def test_rebin_histogram_sums_neighboring_bins() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 10.0, 20.0],
        dtype=np.float64,
    )

    _, rebinned_counts = rebin_histogram(
        time=time,
        counts=counts,
        factor=2,
    )

    expected = np.array(
        [3.0, 30.0],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        rebinned_counts,
        expected,
    )


def test_rebin_histogram_preserves_total_photon_count() -> None:
    time = np.arange(
        0.0,
        0.8,
        0.1,
        dtype=np.float64,
    )
    counts = np.array(
        [2.0, 5.0, 3.0, 8.0, 1.0, 4.0, 7.0, 6.0],
        dtype=np.float64,
    )

    _, rebinned_counts = rebin_histogram(
        time=time,
        counts=counts,
        factor=4,
    )

    assert np.sum(rebinned_counts) == np.sum(counts)


def test_rebin_histogram_places_new_times_at_bin_centres() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    rebinned_time, _ = rebin_histogram(
        time=time,
        counts=counts,
        factor=2,
    )

    expected_time = np.array(
        [0.05, 0.25],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        rebinned_time,
        expected_time,
    )


@pytest.mark.parametrize(
    "factor",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_rebin_histogram_rejects_invalid_factor(
    factor: object,
) -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3],
        dtype=np.float64,
    )
    counts = np.ones(4, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="factor must be a positive integer",
    ):
        rebin_histogram(
            time=time,
            counts=counts,
            factor=factor,  # type: ignore[arg-type]
        )


def test_rebin_histogram_rejects_nondivisible_number_of_bins() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )
    counts = np.ones(5, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="number of histogram bins must be divisible by factor",
    ):
        rebin_histogram(
            time=time,
            counts=counts,
            factor=2,
        )


def test_normalize_counts_total_sums_to_one() -> None:
    counts = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    normalized = normalize_counts(
        counts,
        mode=CountNormalization.TOTAL,
    )

    assert np.isclose(np.sum(normalized), 1.0)


def test_normalize_counts_peak_has_unit_maximum() -> None:
    counts = np.array(
        [1.0, 5.0, 10.0, 2.0],
        dtype=np.float64,
    )

    normalized = normalize_counts(
        counts,
        mode=CountNormalization.PEAK,
    )

    assert np.isclose(np.max(normalized), 1.0)


def test_normalize_counts_total_returns_expected_values() -> None:
    counts = np.array(
        [1.0, 2.0, 1.0],
        dtype=np.float64,
    )

    normalized = normalize_counts(
        counts,
        mode=CountNormalization.TOTAL,
    )

    expected = np.array(
        [0.25, 0.50, 0.25],
        dtype=np.float64,
    )

    np.testing.assert_allclose(normalized, expected)


@pytest.mark.parametrize(
    "mode",
    [
        CountNormalization.TOTAL,
        CountNormalization.PEAK,
    ],
)
def test_normalize_counts_rejects_zero_counts(
    mode: CountNormalization,
) -> None:
    counts = np.zeros(5, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="normalization factor must be positive",
    ):
        normalize_counts(counts, mode=mode)


def test_normalize_counts_does_not_modify_input() -> None:
    counts = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    original = counts.copy()

    normalize_counts(
        counts,
        mode=CountNormalization.TOTAL,
    )

    np.testing.assert_array_equal(counts, original)


def test_normalize_counts_accepts_background_corrected_values() -> None:
    counts = np.array(
        [-1.0, 2.0, 5.0],
        dtype=np.float64,
    )

    normalized = normalize_counts(
        counts,
        mode=CountNormalization.PEAK,
    )

    expected = np.array(
        [-0.2, 0.4, 1.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(normalized, expected)


