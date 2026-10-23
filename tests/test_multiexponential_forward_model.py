"""Physical contracts of the general deterministic reconvolution model."""

import numpy as np
import pytest

from tcspc_toolkit.irf.estimation import estimate_irf_from_leading_edge
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import (
    IRFProfile,
    IRFSourceKind,
    generate_emg_irf_profile,
    generate_gaussian_irf_profile,
)
from tcspc_toolkit.measurement.records import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.physics.forward_model import (
    monoexponential_reconvolution_expected_counts,
    multiexponential_reconvolution_component_counts,
    multiexponential_reconvolution_expected_counts,
)


@pytest.fixture(params=["gaussian", "emg", "sampled", "proxy"])
def prepared(request):
    time = np.linspace(0.0, 8.0, 161)
    if request.param == "gaussian":
        source = generate_gaussian_irf_profile(
            time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4
        )
    elif request.param == "emg":
        source = generate_emg_irf_profile(
            time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.3, tail_time_ns=0.35
        )
    elif request.param == "sampled":
        # A different source grid exercises explicit resampling before use.
        source_time = np.linspace(0.0, 8.0, 81)
        source = IRFProfile(
            source_time,
            np.maximum(0.0, 1.0 - np.abs(source_time - 1.0) / 0.4),
            IRFSourceKind.IMPORTED_SAMPLED,
            provenance={"origin": "synthetic sampled trace for testing"},
        )
        return prepare_irf(source, time, resampling="linear")
    else:
        measurement = TCSPCMeasurement(
            time_ns=time,
            values=np.rint(10.0 + 300.0 * np.exp(-0.5 * ((time - 2.2) / 0.3) ** 2)),
            data_kind=MeasurementDataKind.RAW_COUNTS,
        )
        proxy = estimate_irf_from_leading_edge(
            measurement,
            background_window_ns=(0.0, 1.0),
            rising_edge_window_ns=(1.0, 3.0),
            smoothing_window_bins=9,
            polynomial_order=3,
        )
        source = proxy.profile
        assert source.source_kind is IRFSourceKind.LEADING_EDGE_ESTIMATE
    return prepare_irf(source, time)


@pytest.mark.parametrize("shift", [-0.025, 0.0, 0.075])
@pytest.mark.parametrize("amplitude", [0.0, 125.0])
def test_single_component_is_exactly_existing_mono(prepared, shift, amplitude):
    time, kernel = prepared.time_ns, prepared.kernel
    actual = multiexponential_reconvolution_expected_counts(
        time.tolist(),
        kernel.tolist(),
        amplitudes=(amplitude,),
        lifetimes=[1.3],
        background=2.5,
        temporal_shift=shift,
    )
    expected = monoexponential_reconvolution_expected_counts(
        time, kernel, amplitude, 1.3, 2.5, shift
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("shift", [-0.025, 0.0, 0.075])
@pytest.mark.parametrize(
    "amplitudes,lifetimes",
    [([10.0, 40.0], [0.7, 2.9]), ([8.0, 0.0, 3.0], [3.1, 1.0, 3.1])],
)
def test_components_match_independent_direct_convolution(
    prepared, shift, amplitudes, lifetimes
):
    time, kernel = prepared.time_ns, prepared.kernel
    time_before, kernel_before = time.copy(), kernel.copy()
    source_before = prepared.source.values.copy()
    components = multiexponential_reconvolution_component_counts(
        time, kernel, amplitudes=amplitudes, lifetimes=lifetimes, temporal_shift=shift
    )
    shifted = np.interp(time - shift, time, kernel, left=0.0, right=0.0)
    direct = np.array(
        [
            amplitude
            * np.convolve(np.exp(-time / lifetime), shifted)[: time.size]
            * (time[1] - time[0])
            for amplitude, lifetime in zip(amplitudes, lifetimes, strict=True)
        ]
    )
    # FFT roundoff is absolute near zero; no physics tolerance is fitted here.
    np.testing.assert_allclose(components, direct, rtol=1e-12, atol=1e-12)
    total = multiexponential_reconvolution_expected_counts(
        time,
        kernel,
        amplitudes=amplitudes,
        lifetimes=lifetimes,
        temporal_shift=shift,
        background=2.75,
    )
    np.testing.assert_array_equal(total, components.sum(axis=0) + 2.75)
    assert components.shape == (len(amplitudes), len(time))
    np.testing.assert_array_equal(time, time_before)
    np.testing.assert_array_equal(kernel, kernel_before)
    np.testing.assert_array_equal(prepared.source.values, source_before)


def test_shift_direction_support_loss_and_amplitude_semantics():
    time = np.linspace(0.0, 3.0, 121)
    kernel = prepare_irf(
        generate_gaussian_irf_profile(
            time, gaussian_centre_ns=2.7, gaussian_fwhm_ns=0.35
        ),
        time,
    ).kernel
    curves = [
        multiexponential_reconvolution_component_counts(
            time,
            kernel,
            amplitudes=[10.0, 10.0],
            lifetimes=[0.4, 1.2],
            temporal_shift=shift,
        )
        for shift in [-0.2, 0.0, 0.2]
    ]
    centers = [np.sum(time * curve, axis=1) / curve.sum(axis=1) for curve in curves]
    assert np.all(centers[0] < centers[1])
    assert np.all(centers[1] < centers[2])
    shifted = np.interp(time - 0.2, time, kernel, left=0.0, right=0.0)
    area = np.trapezoid(shifted, x=time)
    assert area < 0.9
    renormalized = multiexponential_reconvolution_component_counts(
        time, shifted / area, amplitudes=[10.0, 10.0], lifetimes=[0.4, 1.2]
    )
    np.testing.assert_allclose(curves[2], area * renormalized, rtol=1e-12, atol=1e-12)
    photons = curves[1].sum(axis=1)
    assert not np.allclose(photons, [10.0, 10.0])
    assert photons[1] > photons[0]  # Equal amplitudes need not give equal photons.


def test_no_time_rebasing_background_convolution_or_implicit_normalization():
    time = np.array([1.0, 1.5, 2.0, 2.5])
    kernel = np.array([0.0, 4.0, 0.0, 0.0])  # Deliberately scaled supplied kernel.
    amplitudes, lifetimes = [3.5, 2.0], [1.0, 2.0]
    ideal = sum(a * np.exp(-time / tau) for a, tau in zip(amplitudes, lifetimes))
    direct = np.convolve(ideal, kernel)[: time.size] * 0.5 + 2.0
    actual = multiexponential_reconvolution_expected_counts(
        time, kernel, amplitudes=amplitudes, lifetimes=lifetimes, background=2.0
    )
    np.testing.assert_allclose(actual, direct, rtol=1e-14, atol=1e-14)
    background_only = multiexponential_reconvolution_expected_counts(
        time, kernel, amplitudes=[0.0, 0.0], lifetimes=lifetimes, background=2.0
    )
    np.testing.assert_array_equal(background_only, np.full(time.size, 2.0))


def test_permutation_and_equal_lifetime_identities(prepared):
    args = prepared.time_ns, prepared.kernel
    components = multiexponential_reconvolution_component_counts(
        *args, amplitudes=[3.0, 0.0, 1.0], lifetimes=[2.0, 0.5, 2.0]
    )
    permuted = multiexponential_reconvolution_component_counts(
        *args, amplitudes=[1.0, 3.0, 0.0], lifetimes=[2.0, 2.0, 0.5]
    )
    np.testing.assert_array_equal(permuted, components[[2, 0, 1]])
    np.testing.assert_array_equal(components[1], np.zeros(args[0].size))
    mono = monoexponential_reconvolution_expected_counts(*args, 4.0, 2.0, 0.0, 0.0)
    np.testing.assert_allclose(components.sum(axis=0), mono, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    "changes",
    [
        {"amplitudes": []},
        {"lifetimes": []},
        {"amplitudes": 1.0},
        {"lifetimes": [[1.0]]},
        {"amplitudes": [-1.0]},
        {"amplitudes": [np.nan]},
        {"amplitudes": [np.inf]},
        {"lifetimes": [0.0]},
        {"lifetimes": [-1.0]},
        {"lifetimes": [np.nan]},
        {"lifetimes": [np.inf]},
        {"temporal_shift": np.inf},
        {"temporal_shift": np.nan},
        {"background": -1.0},
        {"background": np.inf},
        {"background": np.nan},
        {"time": [0.0, 0.1, 0.3]},
        {"time": [0.0, 0.0, 1.0]},
        {"time": [0.0, np.nan, 1.0]},
        {"time": [[0.0, 1.0, 2.0]]},
        {"time": [0.0]},
        {"irf": [1.0]},
        {"irf": [0.0, -1.0, 0.0]},
        {"irf": [0.0, np.nan, 0.0]},
        {"irf": [[0.0, 1.0, 0.0]]},
    ],
)
def test_invalid_model_inputs_are_rejected(changes):
    options = dict(
        time=[0.0, 0.5, 1.0], irf=[0.0, 2.0, 0.0], amplitudes=[1.0], lifetimes=[1.0]
    )
    options.update(changes)
    with pytest.raises(ValueError):
        multiexponential_reconvolution_expected_counts(**options)
