"""Tests for instrument response function (IRF) utilities."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
    shift_irf
)


def test_generate_gaussian_irf_preserves_time_shape(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=2.0,
        fwhm=0.4,
    )

    assert irf.shape == time_axis.shape


def test_generate_gaussian_irf_peaks_at_centre() -> None:
    time = np.linspace(0.0, 10.0, 101)
    centre = 5.0

    irf = generate_gaussian_irf(
        time=time,
        centre=centre,
        fwhm=1.0,
    )

    peak_index = np.argmax(irf)

    assert time[peak_index] == pytest.approx(centre)
    assert irf[peak_index] == pytest.approx(1.0)


def test_generate_gaussian_irf_peaks_close_to_off_grid_centre() -> None:
    time = np.linspace(0.0, 10.0, 101)
    centre = 5.03

    irf = generate_gaussian_irf(
        time=time,
        centre=centre,
        fwhm=1.0,
    )

    peak_time = time[np.argmax(irf)]
    bin_width = time[1] - time[0]

    assert abs(peak_time - centre) <= bin_width / 2.0


def test_generate_gaussian_irf_is_symmetric_about_centre() -> None:
    time = np.linspace(-5.0, 5.0, 101)

    irf = generate_gaussian_irf(
        time=time,
        centre=0.0,
        fwhm=1.0,
    )

    np.testing.assert_allclose(
        irf,
        irf[::-1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_gaussian_irf_rejects_negative_fwhm(
    time_axis: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="fwhm must be greater than zero",
    ):
        generate_gaussian_irf(
            time=time_axis,
            centre=2.0,
            fwhm=-0.5,
        )


def test_generate_gaussian_irf_rejects_zero_fwhm(
    time_axis: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="fwhm must be greater than zero",
    ):
        generate_gaussian_irf(
            time=time_axis,
            centre=2.0,
            fwhm=0.0,
        )


def test_generate_gaussian_irf_rejects_negative_amplitude(
    time_axis: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="amplitude must be non-negative",
    ):
        generate_gaussian_irf(
            time=time_axis,
            centre=2.0,
            fwhm=0.5,
            amplitude=-1.0,
        )


def test_generate_gaussian_irf_allows_zero_amplitude(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=2.0,
        fwhm=0.5,
        amplitude=0.0,
    )

    np.testing.assert_array_equal(
        irf,
        np.zeros_like(time_axis),
    )


def test_decreasing_fwhm_produces_narrower_irf() -> None:
    time = np.linspace(-5.0, 5.0, 1001)
    centre = 0.0

    narrow_irf = generate_gaussian_irf(
        time=time,
        centre=centre,
        fwhm=0.5,
    )
    wide_irf = generate_gaussian_irf(
        time=time,
        centre=centre,
        fwhm=1.0,
    )

    comparison_index = np.argmin(np.abs(time - 0.5))

    assert narrow_irf[comparison_index] < wide_irf[comparison_index]


def test_generate_gaussian_irf_rejects_multidimensional_time() -> None:
    time = np.zeros((2, 10), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="time must be a one-dimensional array",
    ):
        generate_gaussian_irf(
            time=time,
            centre=1.0,
            fwhm=0.5,
        )


def test_generate_gaussian_irf_rejects_empty_time() -> None:
    time = np.array([], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="time must not be empty",
    ):
        generate_gaussian_irf(
            time=time,
            centre=1.0,
            fwhm=0.5,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_generate_gaussian_irf_rejects_non_finite_time(
    invalid_value: float,
) -> None:
    time = np.array(
        [0.0, invalid_value, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="time must contain only finite values",
    ):
        generate_gaussian_irf(
            time=time,
            centre=0.5,
            fwhm=0.2,
        )


@pytest.mark.parametrize(
    "centre",
    [np.nan, np.inf, -np.inf],
)
def test_generate_gaussian_irf_rejects_non_finite_centre(
    centre: float,
) -> None:
    time = np.linspace(0.0, 10.0, 101)

    with pytest.raises(
        ValueError,
        match="centre must be finite",
    ):
        generate_gaussian_irf(
            time=time,
            centre=centre,
            fwhm=1.0,
        )


def test_generate_gaussian_irf_has_requested_fwhm() -> None:
    time = np.linspace(-2.0, 2.0, 4001)
    centre = 0.0
    fwhm = 0.8
    amplitude = 4.0

    irf = generate_gaussian_irf(
        time=time,
        centre=centre,
        fwhm=fwhm,
        amplitude=amplitude,
    )

    left_half_maximum_index = np.argmin(
        np.abs(time - (centre - fwhm / 2.0))
    )
    right_half_maximum_index = np.argmin(
        np.abs(time - (centre + fwhm / 2.0))
    )

    expected_half_maximum = amplitude / 2.0

    assert irf[left_half_maximum_index] == pytest.approx(
        expected_half_maximum,
        rel=1e-6,
    )
    assert irf[right_half_maximum_index] == pytest.approx(
        expected_half_maximum,
        rel=1e-6,
    )


def test_normalized_irf_has_unit_area(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=10.0,
    )

    normalized_irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    area = np.trapezoid(
        normalized_irf,
        x=time_axis,
    )

    assert np.isclose(area, 1.0)
    assert np.isclose(
        area,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_normalize_irf_preserves_shape(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=10.0,
    )

    normalized_irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    assert normalized_irf.shape == irf.shape


def test_normalization_removes_arbitrary_irf_amplitude(
    time_axis: NDArray[np.float64],
) -> None:
    first_irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=1.0,
    )
    second_irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=100.0,
    )

    first_normalized = normalize_irf(
        time=time_axis,
        irf=first_irf,
    )
    second_normalized = normalize_irf(
        time=time_axis,
        irf=second_irf,
    )

    np.testing.assert_allclose(
        first_normalized,
        second_normalized,
    )


def test_normalize_irf_rejects_zero_area(
    time_axis: NDArray[np.float64],
) -> None:
    irf = np.zeros_like(time_axis)

    with pytest.raises(
        ValueError,
        match="positive integrated area",
    ):
        normalize_irf(
            time=time_axis,
            irf=irf,
        )


def test_normalize_irf_rejects_negative_values(
    time_axis: NDArray[np.float64],
) -> None:
    # TODO: Although experimentally measured IRFs can contain slightly negative values after baseline subtraction or preprocessing,
    #       the simulation-layer IRF represents a non-negative response kernel. Negative values should therefore be rejected here.
    #       Measured-IRF preprocessing can later be implemented as a separate workflow.
    irf = np.ones_like(time_axis)
    irf[500] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        normalize_irf(
            time=time_axis,
            irf=irf,
        )

@pytest.mark.parametrize(
    ("time", "irf", "message"),
    [
        (
            np.ones((2, 3), dtype=np.float64),
            np.ones(6, dtype=np.float64),
            "time must be one-dimensional",
        ),
        (
            np.ones(6, dtype=np.float64),
            np.ones((2, 3), dtype=np.float64),
            "irf must be one-dimensional",
        ),
    ],
)
def test_normalize_irf_rejects_non_1d_inputs(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_irf(
            time=time,
            irf=irf,
        )


def test_normalize_irf_rejects_shape_mismatch() -> None:
    time = np.linspace(
        0.0,
        10.0,
        101,
        dtype=np.float64,
    )
    irf = np.ones(100, dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        normalize_irf(
            time=time,
            irf=irf,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_normalize_irf_rejects_non_finite_time(
    invalid_value: float,
) -> None:
    time = np.linspace(
        0.0,
        10.0,
        101,
        dtype=np.float64,
    )
    irf = np.ones_like(time)

    time[50] = invalid_value

    with pytest.raises(
        ValueError,
        match="time must contain only finite values",
    ):
        normalize_irf(
            time=time,
            irf=irf,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_normalize_irf_rejects_non_finite_irf(
    invalid_value: float,
) -> None:
    time = np.linspace(
        0.0,
        10.0,
        101,
        dtype=np.float64,
    )
    irf = np.ones_like(time)

    irf[50] = invalid_value

    with pytest.raises(
        ValueError,
        match="irf must contain only finite values",
    ):
        normalize_irf(
            time=time,
            irf=irf,
        )


def test_normalize_irf_rejects_repeated_time_value() -> None:
    time = np.array(
        [0.0, 0.1, 0.1, 0.2],
        dtype=np.float64,
    )
    irf = np.array(
        [0.0, 1.0, 1.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        normalize_irf(
            time=time,
            irf=irf,
        )


def test_normalize_irf_rejects_decreasing_time_axis() -> None:
    time = np.array(
        [0.0, 0.2, 0.1, 0.3],
        dtype=np.float64,
    )
    irf = np.array(
        [0.0, 1.0, 1.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        normalize_irf(
            time=time,
            irf=irf,
        )


def test_normalization_preserves_irf_peak_location(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=10.0,
    )

    normalized_irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    original_peak_index = np.argmax(irf)
    normalized_peak_index = np.argmax(normalized_irf)

    assert normalized_peak_index == original_peak_index
    assert (
        time_axis[normalized_peak_index]
        == time_axis[original_peak_index]
    )


def calculate_sampled_fwhm(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
) -> float:
    """Estimate FWHM using the first and last bins above half maximum."""
    half_maximum = 0.5 * np.max(values)
    indices = np.flatnonzero(values >= half_maximum)

    if indices.size < 2:
        raise ValueError(
            "At least two samples are required to estimate FWHM."
        )

    return float(
        time[indices[-1]]
        - time[indices[0]]
    )


def test_normalization_preserves_irf_width(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=10.0,
    )

    normalized_irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    original_fwhm = calculate_sampled_fwhm(
        time=time_axis,
        values=irf,
    )
    normalized_fwhm = calculate_sampled_fwhm(
        time=time_axis,
        values=normalized_irf,
    )

    assert normalized_fwhm == original_fwhm


def test_normalization_preserves_relative_irf_shape(
    time_axis: NDArray[np.float64],
) -> None:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=5.0,
        fwhm=0.5,
        amplitude=10.0,
    )

    normalized_irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    original_relative_shape = irf / np.max(irf)
    normalized_relative_shape = (
        normalized_irf / np.max(normalized_irf)
    )

    np.testing.assert_allclose(
        normalized_relative_shape,
        original_relative_shape,
    )


def test_shift_irf_zero_shift_returns_original_irf() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    irf = generate_gaussian_irf(
        time=time,
        centre=5.0,
        fwhm=1.0,
        amplitude=1.0,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=0.0,
    )

    np.testing.assert_allclose(
        shifted_irf,
        irf,
        rtol=0.0,
        atol=0.0,
    )


def test_shift_irf_positive_shift_moves_peak_later() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    irf = generate_gaussian_irf(
        time=time,
        centre=4.0,
        fwhm=0.5,
        amplitude=1.0,
    )

    shift = 1.0

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=shift,
    )

    original_peak_time = time[np.argmax(irf)]
    shifted_peak_time = time[np.argmax(shifted_irf)]

    assert shifted_peak_time == pytest.approx(
        original_peak_time + shift,
        abs=time[1] - time[0],
    )


def test_shift_irf_negative_shift_moves_peak_earlier() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    irf = generate_gaussian_irf(
        time=time,
        centre=6.0,
        fwhm=0.5,
        amplitude=1.0,
    )

    shift = -1.0

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=shift,
    )

    original_peak_time = time[np.argmax(irf)]
    shifted_peak_time = time[np.argmax(shifted_irf)]

    assert shifted_peak_time == pytest.approx(
        original_peak_time + shift,
        abs=time[1] - time[0],
    )


def test_shift_irf_preserves_output_shape() -> None:
    time = np.linspace(0.0, 10.0, 501)
    irf = generate_gaussian_irf(
        time=time,
        centre=5.0,
        fwhm=1.0,
        amplitude=1.0,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=0.37,
    )

    assert shifted_irf.shape == irf.shape
    assert shifted_irf.dtype == np.float64


def test_shift_irf_approximately_preserves_shape() -> None:
    time = np.linspace(0.0, 10.0, 2001)
    irf = generate_gaussian_irf(
        time=time,
        centre=4.0,
        fwhm=0.8,
        amplitude=1.0,
    )

    shift = 1.0

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=shift,
    )

    expected_shifted_irf = generate_gaussian_irf(
        time=time,
        centre=5.0,
        fwhm=0.8,
        amplitude=1.0,
    )

    np.testing.assert_allclose(
        shifted_irf,
        expected_shifted_irf,
        rtol=1e-3,
        atol=1e-5,
    )


def test_shift_irf_supports_sub_bin_shift() -> None:
    time = np.linspace(0.0, 10.0, 101)
    irf = generate_gaussian_irf(
        time=time,
        centre=4.0,
        fwhm=0.8,
        amplitude=1.0,
    )

    bin_width = time[1] - time[0]
    shift = 0.5 * bin_width

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=shift,
    )

    expected_peak_time = 4.0 + shift
    shifted_peak_time = time[np.argmax(shifted_irf)]

    assert shifted_peak_time == pytest.approx(
        expected_peak_time,
        abs=bin_width,
    )

    assert not np.array_equal(shifted_irf, irf)


def test_shift_irf_large_shift_moves_irf_outside_time_window() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    irf = generate_gaussian_irf(
        time=time,
        centre=5.0,
        fwhm=0.5,
        amplitude=1.0,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=20.0,
    )

    np.testing.assert_array_equal(
        shifted_irf,
        np.zeros_like(irf),
    )


@pytest.mark.parametrize("shift", [-20.0, 20.0])
def test_shift_irf_large_shift_returns_zeros(
    shift: float,
) -> None:
    time = np.linspace(0.0, 10.0, 1001)
    irf = generate_gaussian_irf(
        time=time,
        centre=5.0,
        fwhm=0.5,
        amplitude=1.0,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=shift,
    )

    np.testing.assert_array_equal(
        shifted_irf,
        np.zeros_like(irf),
    )


def test_shift_irf_does_not_renormalize_after_boundary_loss() -> None:
    time = np.linspace(0.0, 10.0, 1001)
    irf = generate_gaussian_irf(
        time=time,
        centre=9.5,
        fwhm=1.0,
        amplitude=1.0,
    )
    irf = normalize_irf(
        time=time,
        irf=irf,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=1.0,
    )

    original_area = np.trapezoid(irf, x=time)
    shifted_area = np.trapezoid(shifted_irf, x=time)

    assert original_area == pytest.approx(1.0)
    assert shifted_area < original_area


@pytest.mark.parametrize(
    ("time", "irf", "shift", "expected_message"),
    [
        (
            np.ones((2, 2)),
            np.ones(4),
            0.0,
            "time must be one-dimensional",
        ),
        (
            np.arange(4.0),
            np.ones((2, 2)),
            0.0,
            "irf must be one-dimensional",
        ),
        (
            np.arange(4.0),
            np.ones(3),
            0.0,
            "time and irf must have the same shape",
        ),
        (
            np.array([0.0]),
            np.array([1.0]),
            0.0,
            "at least two values",
        ),
        (
            np.array([0.0, np.nan, 2.0]),
            np.ones(3),
            0.0,
            "time must contain only finite values",
        ),
        (
            np.arange(3.0),
            np.array([1.0, np.inf, 1.0]),
            0.0,
            "irf must contain only finite values",
        ),
        (
            np.array([0.0, 2.0, 1.0]),
            np.ones(3),
            0.0,
            "time must be strictly increasing",
        ),
        (
            np.arange(3.0),
            np.array([1.0, -0.1, 1.0]),
            0.0,
            "irf values must be non-negative",
        ),
        (
            np.arange(3.0),
            np.ones(3),
            np.nan,
            "shift must be finite",
        ),
    ],
)
def test_shift_irf_rejects_invalid_input(
    time: np.ndarray,
    irf: np.ndarray,
    shift: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        shift_irf(
            time=time,
            irf=irf,
            shift=shift,
        )

