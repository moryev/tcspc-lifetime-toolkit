import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)
from tcspc_toolkit.models import monoexponential_decay


def test_convolution_preserves_shape(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)

    irf = generate_gaussian_irf(
        time=time_axis,
        centre=0.5,
        fwhm=0.2,
        amplitude=1.0,
    )
    irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    assert convolved.shape == decay.shape


def test_convolution_is_non_negative(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)

    irf = generate_gaussian_irf(
        time=time_axis,
        centre=0.5,
        fwhm=0.2,
        amplitude=1.0,
    )
    irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    assert np.all(convolved >= 0.0)


def test_zero_decay_produces_zero_convolution(
    time_axis: np.ndarray,
) -> None:
    decay = np.zeros_like(time_axis)

    irf = generate_gaussian_irf(
        time=time_axis,
        centre=0.5,
        fwhm=0.2,
        amplitude=1.0,
    )
    irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    np.testing.assert_allclose(
        convolved,
        np.zeros_like(decay),
    )


def test_delta_like_irf_preserves_decay(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)

    dt = time_axis[1] - time_axis[0]

    irf = np.zeros_like(time_axis)
    irf[0] = 1.0 / dt

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    np.testing.assert_allclose(
        convolved,
        decay,
        rtol=1e-12,
        atol=1e-12,
    )


def test_narrow_gaussian_irf_approaches_shifted_decay(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)

    dt = time_axis[1] - time_axis[0]

    centre_index = 10
    centre = centre_index * dt

    irf = generate_gaussian_irf(
        time=time_axis,
        centre=centre,
        fwhm=2.0 * dt,
        amplitude=1.0,
    )
    irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    np.testing.assert_allclose(
        convolved[centre_index + 5 :],
        decay[5 : -centre_index],
        rtol=0.03,
        atol=0.01,
    )


def test_wider_irf_broadens_convolved_decay(
    time_axis: np.ndarray,
) -> None:
    onset = 2.0
    lifetime = 1.0

    decay = np.where(
        time_axis >= onset,
        np.exp(-(time_axis - onset) / lifetime),
        0.0,
    )

    narrow_irf = generate_gaussian_irf(
        time=time_axis,
        centre=0.5,
        fwhm=0.05,
        amplitude=1.0,
    )
    narrow_irf = normalize_irf(
        time=time_axis,
        irf=narrow_irf,
    )

    wide_irf = generate_gaussian_irf(
        time=time_axis,
        centre=0.5,
        fwhm=0.5,
        amplitude=1.0,
    )
    wide_irf = normalize_irf(
        time=time_axis,
        irf=wide_irf,
    )

    narrow_convolution = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=narrow_irf,
    )

    wide_convolution = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=wide_irf,
    )

    narrow_area = np.trapezoid(
        narrow_convolution,
        x=time_axis,
    )
    wide_area = np.trapezoid(
        wide_convolution,
        x=time_axis,
    )

    narrow_normalized = narrow_convolution / narrow_area
    wide_normalized = wide_convolution / wide_area

    narrow_mean = np.trapezoid(
        time_axis * narrow_normalized,
        x=time_axis,
    )
    wide_mean = np.trapezoid(
        time_axis * wide_normalized,
        x=time_axis,
    )

    narrow_variance = np.trapezoid(
        (time_axis - narrow_mean) ** 2 * narrow_normalized,
        x=time_axis,
    )
    wide_variance = np.trapezoid(
        (time_axis - wide_mean) ** 2 * wide_normalized,
        x=time_axis,
    )

    assert wide_variance > narrow_variance


@pytest.mark.parametrize(
    "invalid_argument",
    ["time", "decay", "irf"],
)
def test_convolution_rejects_non_one_dimensional_arrays(
    time_axis: np.ndarray,
    invalid_argument: str,
) -> None:
    time = time_axis.copy()
    decay = np.exp(-time_axis / 2.0)
    irf = np.ones_like(time_axis)

    if invalid_argument == "time":
        time = time[:, np.newaxis]
    elif invalid_argument == "decay":
        decay = decay[:, np.newaxis]
    else:
        irf = irf[:, np.newaxis]

    with pytest.raises(
        ValueError,
        match=f"{invalid_argument} must be one-dimensional",
    ):
        convolve_decay_with_irf(
            time=time,
            decay=decay,
            irf=irf,
        )


def test_convolution_rejects_unequal_lengths(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)
    irf = np.ones(time_axis.size - 1)

    with pytest.raises(
        ValueError,
        match="time, decay, and irf must have equal lengths",
    ):
        convolve_decay_with_irf(
            time=time_axis,
            decay=decay,
            irf=irf,
        )


def test_convolution_requires_at_least_two_time_points() -> None:
    time = np.array([0.0])
    decay = np.array([1.0])
    irf = np.array([1.0])

    with pytest.raises(
        ValueError,
        match="time must contain at least two points",
    ):
        convolve_decay_with_irf(
            time=time,
            decay=decay,
            irf=irf,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_convolution_rejects_non_finite_time(
    time_axis: np.ndarray,
    invalid_value: float,
) -> None:
    time = time_axis.copy()
    time[10] = invalid_value

    decay = np.exp(-time_axis / 2.0)
    irf = np.ones_like(time_axis)

    with pytest.raises(
        ValueError,
        match="time must contain only finite values",
    ):
        convolve_decay_with_irf(
            time=time,
            decay=decay,
            irf=irf,
        )


def test_convolution_rejects_repeated_time_values(
    time_axis: np.ndarray,
) -> None:
    time = time_axis.copy()
    time[100] = time[99]

    decay = np.exp(-time_axis / 2.0)
    irf = np.ones_like(time_axis)

    with pytest.raises(
        ValueError,
        match="time must be strictly increasing",
    ):
        convolve_decay_with_irf(
            time=time,
            decay=decay,
            irf=irf,
        )


def test_convolution_rejects_decreasing_time_values(
    time_axis: np.ndarray,
) -> None:
    time = time_axis.copy()
    time[100], time[101] = time[101], time[100]

    decay = np.exp(-time_axis / 2.0)
    irf = np.ones_like(time_axis)

    with pytest.raises(
        ValueError,
        match="time must be strictly increasing",
    ):
        convolve_decay_with_irf(
            time=time,
            decay=decay,
            irf=irf,
        )


def test_convolution_rejects_non_uniform_time_axis(
    time_axis: np.ndarray,
) -> None:
    time = time_axis.copy()
    time[500:] += 0.001

    decay = np.exp(-time_axis / 2.0)
    irf = np.ones_like(time_axis)

    with pytest.raises(
        ValueError,
        match="time must be uniformly spaced",
    ):
        convolve_decay_with_irf(
            time=time,
            decay=decay,
            irf=irf,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_convolution_rejects_non_finite_decay(
    time_axis: np.ndarray,
    invalid_value: float,
) -> None:
    decay = np.exp(-time_axis / 2.0)
    decay[10] = invalid_value

    irf = np.ones_like(time_axis)

    with pytest.raises(
        ValueError,
        match="decay must contain only finite values",
    ):
        convolve_decay_with_irf(
            time=time_axis,
            decay=decay,
            irf=irf,
        )


def test_convolution_rejects_negative_decay(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)
    decay[10] = -1.0

    irf = np.ones_like(time_axis)

    with pytest.raises(
        ValueError,
        match="decay must contain only non-negative values",
    ):
        convolve_decay_with_irf(
            time=time_axis,
            decay=decay,
            irf=irf,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_convolution_rejects_non_finite_irf(
    time_axis: np.ndarray,
    invalid_value: float,
) -> None:
    decay = np.exp(-time_axis / 2.0)

    irf = np.ones_like(time_axis)
    irf[10] = invalid_value

    with pytest.raises(
        ValueError,
        match="irf must contain only finite values",
    ):
        convolve_decay_with_irf(
            time=time_axis,
            decay=decay,
            irf=irf,
        )


def test_convolution_rejects_negative_irf(
    time_axis: np.ndarray,
) -> None:
    decay = np.exp(-time_axis / 2.0)

    irf = np.ones_like(time_axis)
    irf[10] = -1.0

    with pytest.raises(
        ValueError,
        match="irf must contain only non-negative values",
    ):
        convolve_decay_with_irf(
            time=time_axis,
            decay=decay,
            irf=irf,
        )


def test_convolution_is_consistent_across_time_resolutions() -> None:
    coarse_time = np.linspace(0.0, 10.0, 1001)
    fine_time = np.linspace(0.0, 10.0, 2001)

    coarse_decay = np.exp(-coarse_time / 2.0)
    fine_decay = np.exp(-fine_time / 2.0)

    coarse_irf = generate_gaussian_irf(
        time=coarse_time,
        centre=1.0,
        fwhm=0.2,
        amplitude=1.0,
    )
    coarse_irf = normalize_irf(
        time=coarse_time,
        irf=coarse_irf,
    )

    fine_irf = generate_gaussian_irf(
        time=fine_time,
        centre=1.0,
        fwhm=0.2,
        amplitude=1.0,
    )
    fine_irf = normalize_irf(
        time=fine_time,
        irf=fine_irf,
    )

    coarse_convolution = convolve_decay_with_irf(
        time=coarse_time,
        decay=coarse_decay,
        irf=coarse_irf,
    )

    fine_convolution = convolve_decay_with_irf(
        time=fine_time,
        decay=fine_decay,
        irf=fine_irf,
    )

    fine_on_coarse_grid = fine_convolution[::2]

    np.testing.assert_allclose(
        coarse_convolution,
        fine_on_coarse_grid,
        rtol=0.02,
        atol=0.01,
    )
