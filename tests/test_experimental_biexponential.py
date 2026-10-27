"""Canonical raw-data/provenance integration without replacing either fitter."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation.multiexponential import BiexponentialFitSettings
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import generate_gaussian_irf_profile
from tcspc_toolkit.measurement import adapters
from tcspc_toolkit.measurement.records import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
)


@pytest.fixture
def inputs():
    time = np.linspace(0, 12, 64)
    profile = generate_gaussian_irf_profile(
        time,
        gaussian_centre_ns=1,
        gaussian_fwhm_ns=0.6,
        metadata={"calibration": ["independent"]},
        provenance={"file": "calibration.csv"},
    )
    prepared = prepare_irf(profile, time)
    sampled = SampledIRF(time, 13 * profile.values, provenance={"file": "attached.csv"})
    measurement = TCSPCMeasurement(
        time,
        np.zeros(time.size, dtype=int),
        MeasurementDataKind.RAW_COUNTS,
        sampled,
        "sample-1",
        {"instrument": ["A"]},
        {"file": "counts.csv"},
    )
    return measurement, prepared


BOUNDS = dict(lifetime_bounds_ns=(0.1, 8), temporal_shift_bounds_ns=(-0.3, 0.3))


def test_prepared_precedence_and_defensive_provenance(inputs):
    measurement, prepared = inputs
    result = adapters.fit_experimental_biexponential_reconvolution(
        measurement, prepared_irf=prepared, **BOUNDS
    )
    context = result.context
    assert context.irf_selection == "explicit_prepared_irf"
    assert not context.normalization_performed
    assert context.assumed_irf.diagnostics == prepared.diagnostics
    assert (
        context.assumed_irf.source.source_parameters
        == prepared.source.source_parameters
    )
    assert context.assumed_irf.source.provenance == prepared.source.provenance
    assert result.fit.assumptions.preparation_diagnostics == prepared.diagnostics
    assert result.fit.assumptions.source_provenance["file"] == "calibration.csv"
    assert result.fit.optimization.numerical_validation_passed
    assert result.fit.components.amplitude_fractions is None
    measurement.metadata["instrument"].append("changed")
    prepared.source.metadata["calibration"].append("changed")
    assert context.measurement.metadata["instrument"] == ["A"]
    assert context.assumed_irf.source.metadata["calibration"] == ["independent"]
    assert result.fit.assumptions.source_metadata["calibration"] == ["independent"]
    for array in (
        context.kernel,
        context.measurement.values,
        context.assumed_irf.kernel,
        context.assumed_irf.source.values,
        result.fit.expected_counts,
    ):
        assert not array.flags.writeable


def test_attached_trace_normalization_is_explicit_and_not_prepared(inputs):
    measurement, _ = inputs
    original = measurement.irf.values.copy()
    result = adapters.fit_experimental_biexponential_reconvolution(
        measurement, **BOUNDS
    )
    context = result.context
    assert context.irf_selection == "attached_sampled_irf"
    assert context.normalization_performed
    assert context.area_before_normalization == pytest.approx(
        np.trapezoid(original, x=measurement.time_ns)
    )
    assert context.area_after_normalization == pytest.approx(1)
    assert isinstance(context.assumed_irf, SampledIRF)
    assert result.fit.assumptions.preparation_diagnostics is None
    assert context.assumed_irf.provenance["file"] == "attached.csv"
    np.testing.assert_array_equal(measurement.irf.values, original)
    assert not np.shares_memory(context.assumed_irf.values, measurement.irf.values)


@pytest.mark.parametrize(
    "function",
    [
        adapters.fit_experimental_biexponential_reconvolution,
        adapters.compare_experimental_reconvolution,
    ],
)
def test_raw_guard_precedes_irf_and_fit_validation(inputs, function):
    measurement, _ = inputs
    processed = replace(
        measurement, data_kind=MeasurementDataKind.PROCESSED_INTENSITY, irf=None
    )
    with pytest.raises(InvalidMeasurementError, match="raw"):
        function(processed, **BOUNDS)


def test_missing_and_incompatible_irfs_are_rejected(inputs):
    measurement, prepared = inputs
    with pytest.raises(InvalidMeasurementError):
        adapters.fit_experimental_biexponential_reconvolution(
            replace(measurement, irf=None), **BOUNDS
        )
    other = prepare_irf(
        prepared.source, measurement.time_ns + 0.01, resampling="linear"
    )
    with pytest.raises(InvalidMeasurementError):
        adapters.fit_experimental_biexponential_reconvolution(
            measurement, prepared_irf=other, **BOUNDS
        )


def test_comparison_preserves_mono_initial_shift_restriction(inputs):
    measurement, _ = inputs
    with pytest.raises(InvalidMeasurementError, match="include 0"):
        adapters.compare_experimental_reconvolution(
            measurement,
            lifetime_bounds_ns=(0.1, 8),
            temporal_shift_bounds_ns=(0.1, 0.3),
        )
    result = adapters.fit_experimental_biexponential_reconvolution(
        measurement, lifetime_bounds_ns=(0.1, 8), temporal_shift_bounds_ns=(0.1, 0.3)
    )
    assert result.fit.parameters.temporal_shift_ns == 0.1


def test_comparison_identical_observations_and_kernel(inputs, monkeypatch):
    measurement, prepared = inputs
    t = measurement.time_ns
    expected = 2 + 1000 * np.convolve(np.exp(-t / 2), prepared.kernel)[: t.size] * (
        t[1] - t[0]
    )
    measurement = replace(measurement, values=np.rint(expected).astype(int))
    calls = []
    mono_original = adapters.fit_single_reconvolution_curve
    bi_original = adapters.fit_biexponential_reconvolution

    def mono(**kwargs):
        calls.append(
            (kwargs["time"].copy(), kwargs["counts"].copy(), kwargs["irf"].copy())
        )
        return mono_original(**kwargs)

    def bi(time, counts, irf, **kwargs):
        calls.append((time.copy(), counts.copy(), irf.kernel.copy()))
        return bi_original(time, counts, irf, **kwargs)

    monkeypatch.setattr(adapters, "fit_single_reconvolution_curve", mono)
    monkeypatch.setattr(adapters, "fit_biexponential_reconvolution", bi)
    result = adapters.compare_experimental_reconvolution(
        measurement,
        prepared_irf=prepared,
        settings=BiexponentialFitSettings(policy="economical", max_starts=3),
        **BOUNDS,
    )
    for left, right in zip(calls[0], calls[1], strict=True):
        np.testing.assert_array_equal(left, right)
    assert result.statistics.mono.n_parameters == 4
    assert result.statistics.bi.n_parameters == 6
    assert result.statistics.bi.n_bins == measurement.values.size
    assert result.bounds.mono_lifetime_bounds_ns == (1e-12, None)
    assert result.bounds.bi_background_lower == 0
    assert result.bounds.mono_background_lower == 1e-12


def test_real_seeded_adapter_matches_low_level_and_direct_convolution(inputs):
    measurement, prepared = inputs
    t = measurement.time_ns
    shifted = np.interp(t - 0.07, t, prepared.kernel, left=0, right=0)
    expected = 3 + sum(
        a * np.convolve(np.exp(-t / tau), shifted)[: t.size] * (t[1] - t[0])
        for a, tau in [(800, 0.8), (600, 3.2)]
    )
    measurement = replace(
        measurement, values=np.random.default_rng(283991).poisson(expected)
    )
    settings = BiexponentialFitSettings(policy="economical", max_starts=3)
    adapted = adapters.fit_experimental_biexponential_reconvolution(
        measurement, prepared_irf=prepared, settings=settings, **BOUNDS
    )
    direct = adapters.fit_biexponential_reconvolution(
        t, measurement.values, prepared, settings=settings, **BOUNDS
    )
    assert adapted.fit.parameters == direct.parameters
    np.testing.assert_array_equal(adapted.fit.expected_counts, direct.expected_counts)
    assert (
        np.linalg.norm(adapted.fit.expected_counts - expected)
        / np.linalg.norm(expected)
        < 0.1
    )
