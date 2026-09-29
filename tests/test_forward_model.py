import numpy as np
import pytest

from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.fitting import _reconvolution_model
from tcspc_toolkit.forward_model import (
    monoexponential_reconvolution_expected_counts,
)
from tcspc_toolkit.irf import (
    generate_emg_irf_profile,
    generate_gaussian_irf,
    generate_gaussian_irf_profile,
    normalize_irf,
    shift_irf,
)
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.simulation import build_expected_counts_from_irf


@pytest.mark.parametrize("kernel_kind", ["gaussian", "emg"])
@pytest.mark.parametrize("temporal_shift", [0.0, 0.025, -0.025])
def test_expected_counts_match_original_reconvolution_composition(
    kernel_kind: str,
    temporal_shift: float,
) -> None:
    time = np.linspace(0.0, 8.0, 161, dtype=np.float64)
    if kernel_kind == "gaussian":
        irf = normalize_irf(time, generate_gaussian_irf(time, centre=1.0, fwhm=0.4))
    else:
        profile = generate_emg_irf_profile(
            time,
            gaussian_centre_ns=1.0,
            gaussian_fwhm_ns=0.4,
            tail_time_ns=0.35,
        )
        irf = prepare_irf(profile, time).kernel

    amplitude = 125.0
    lifetime = 1.3
    background = 2.5
    decay = monoexponential_decay(time, amplitude=1.0, lifetime=lifetime, background=0.0)
    shifted_irf = shift_irf(time, irf, temporal_shift)
    original_composition = (
        amplitude * convolve_decay_with_irf(time, decay, shifted_irf) + background
    )

    actual = monoexponential_reconvolution_expected_counts(
        time, irf, amplitude, lifetime, background, temporal_shift
    )
    compatibility_result = _reconvolution_model(
        time, irf, amplitude, lifetime, background, temporal_shift
    )

    np.testing.assert_array_equal(actual, original_composition)
    np.testing.assert_array_equal(compatibility_result, original_composition)


def test_fractional_shift_retains_later_positive_shift_convention() -> None:
    time = np.linspace(0.0, 8.0, 161, dtype=np.float64)
    irf = normalize_irf(time, generate_gaussian_irf(time, centre=1.0, fwhm=0.4))
    bin_width = time[1] - time[0]
    shifts = (-0.5 * bin_width, 0.0, 0.5 * bin_width)

    signal_centres = []
    for shift in shifts:
        signal = monoexponential_reconvolution_expected_counts(
            time, irf, amplitude=1.0, lifetime=0.6, background=0.0,
            temporal_shift=shift,
        )
        signal_centres.append(float(np.sum(time * signal) / signal.sum()))

    assert signal_centres[0] < signal_centres[1] < signal_centres[2]


def test_background_is_constant_offset_after_convolution() -> None:
    time = np.linspace(0.0, 8.0, 161, dtype=np.float64)
    irf = normalize_irf(time, generate_gaussian_irf(time, centre=1.0, fwhm=0.4))
    signal = monoexponential_reconvolution_expected_counts(
        time, irf, amplitude=125.0, lifetime=1.3, background=0.0,
        temporal_shift=0.025,
    )
    with_background = monoexponential_reconvolution_expected_counts(
        time, irf, amplitude=125.0, lifetime=1.3, background=2.75,
        temporal_shift=0.025,
    )
    background_only = monoexponential_reconvolution_expected_counts(
        time, irf, amplitude=0.0, lifetime=1.3, background=2.75,
        temporal_shift=0.025,
    )

    np.testing.assert_allclose(with_background - signal, 2.75, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(background_only, 2.75, rtol=0.0, atol=1e-12)


def test_shifted_irf_is_not_renormalized_after_window_loss() -> None:
    time = np.linspace(0.0, 3.0, 121, dtype=np.float64)
    irf = normalize_irf(time, generate_gaussian_irf(time, centre=2.7, fwhm=0.35))
    shifted_irf = shift_irf(time, irf, shift=0.2)
    assert np.trapezoid(shifted_irf, x=time) < 1.0

    decay = monoexponential_decay(time, amplitude=1.0, lifetime=0.8, background=0.0)
    actual = monoexponential_reconvolution_expected_counts(
        time, irf, amplitude=10.0, lifetime=0.8, background=0.0,
        temporal_shift=0.2,
    )
    original = 10.0 * convolve_decay_with_irf(time, decay, shifted_irf)
    renormalized = 10.0 * convolve_decay_with_irf(
        time, decay, normalize_irf(time, shifted_irf)
    )

    np.testing.assert_array_equal(actual, original)
    assert actual.sum() < renormalized.sum()


def test_convolution_retains_bin_width_and_finite_window_truncation() -> None:
    time = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    irf = np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float64)
    decay = np.exp(-time / 1.0)
    expected = (
        3.5 * np.convolve(decay, irf, mode="full")[:time.size] * 0.5 + 2.0
    )

    actual = monoexponential_reconvolution_expected_counts(
        time, irf, amplitude=3.5, lifetime=1.0, background=2.0,
        temporal_shift=0.0,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_amplitude_is_scale_not_finite_window_signal_photon_budget() -> None:
    time = np.linspace(0.0, 8.0, 161, dtype=np.float64)
    prepared_irf = prepare_irf(
        generate_gaussian_irf_profile(
            time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4
        ),
        time,
    )
    lifetime = 1.3
    background = 2.0
    amplitude = 500.0
    unit_signal = monoexponential_reconvolution_expected_counts(
        time, prepared_irf.kernel, amplitude=1.0, lifetime=lifetime,
        background=0.0, temporal_shift=0.0,
    )
    reconvolution = monoexponential_reconvolution_expected_counts(
        time, prepared_irf.kernel, amplitude=amplitude, lifetime=lifetime,
        background=background, temporal_shift=0.0,
    )
    ideal_decay = monoexponential_decay(
        time, amplitude=1.0, lifetime=lifetime, background=0.0
    )
    simulation = build_expected_counts_from_irf(
        ideal_decay,
        prepared_irf,
        signal_photon_count=500,
        background_per_bin=background,
    )

    assert (reconvolution - background).sum() == pytest.approx(
        amplitude * unit_signal.sum()
    )
    assert not np.isclose((reconvolution - background).sum(), amplitude)
    assert (simulation - background).sum() == pytest.approx(500.0)
    matching_scale = 500.0 / unit_signal.sum()
    np.testing.assert_allclose(
        monoexponential_reconvolution_expected_counts(
            time, prepared_irf.kernel, matching_scale, lifetime, background, 0.0
        ),
        simulation,
    )


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("amplitude", -1.0, "amplitude must be non-negative"),
        ("lifetime", -1.0, "lifetime must be positive"),
        ("lifetime", 0.0, "lifetime must be positive"),
        ("background", -1.0, "background must be non-negative"),
    ],
)
def test_existing_parameter_validation_is_preserved(
    parameter: str,
    value: float,
    message: str,
) -> None:
    time = np.linspace(0.0, 8.0, 161, dtype=np.float64)
    irf = normalize_irf(time, generate_gaussian_irf(time, centre=1.0, fwhm=0.4))
    parameters = {
        "amplitude": 125.0,
        "lifetime": 1.3,
        "background": 2.5,
    }
    parameters[parameter] = value

    with pytest.raises(ValueError, match=message):
        monoexponential_reconvolution_expected_counts(
            time, irf, temporal_shift=0.0, **parameters
        )
