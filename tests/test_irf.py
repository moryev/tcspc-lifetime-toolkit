"""Tests for instrument response function (IRF) utilities."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.irf import generate_gaussian_irf


def test_generate_gaussian_irf_preserves_time_shape(
    time_axis = np.linspace(0.0, 10.0, 101),
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
    time_axis = np.linspace(0.0, 10.0, 101),
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
    time_axis = np.linspace(0.0, 10.0, 101),
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
    time_axis = np.linspace(0.0, 10.0, 101),
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
    time_axis = np.linspace(0.0, 10.0, 101),
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


