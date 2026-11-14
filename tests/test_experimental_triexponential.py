"""Thin raw-measurement integration and bounded research-example checks."""

from dataclasses import replace
from pathlib import Path
import runpy

import numpy as np
import pytest

from tcspc_toolkit.estimation.triexponential import (
    MultiexponentialFitResult,
    TriexponentialFitSettings,
    TriexponentialParameters,
)
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import generate_gaussian_irf_profile
from tcspc_toolkit.measurement import triexponential as adapters
from tcspc_toolkit.measurement.records import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
)


ROOT = Path(__file__).resolve().parents[1]
BOUNDS = dict(lifetime_bounds_ns=(0.1, 10), temporal_shift_bounds_ns=(-0.3, 0.3))
SETTINGS = TriexponentialFitSettings(10, 1, 200, 30, solver="trf")


@pytest.fixture
def inputs():
    time = np.arange(32) / 4
    profile = generate_gaussian_irf_profile(
        time,
        gaussian_centre_ns=1,
        gaussian_fwhm_ns=0.5,
        provenance={"file": ["prepared-source"]},
    )
    prepared = prepare_irf(profile, time)
    sampled = SampledIRF(
        time, 7 * profile.values, provenance={"file": ["attached-source"]}
    )
    measurement = TCSPCMeasurement(
        time,
        np.arange(32),
        MeasurementDataKind.RAW_COUNTS,
        sampled,
        sample_id="test",
        provenance={"file": ["counts-source"]},
    )
    return measurement, prepared


@pytest.mark.parametrize("use_prepared", [True, False])
def test_raw_forwarding_settings_and_source_context(inputs, monkeypatch, use_prepared):
    measurement, prepared = inputs
    guesses = [TriexponentialParameters((1, 0, 2), (2, 1, 2))]
    returned = object()
    original_values = measurement.values.copy()
    original_irf = measurement.irf.values.copy()

    def fit(time, counts, irf, **kwargs):
        np.testing.assert_array_equal(time, measurement.time_ns)
        np.testing.assert_array_equal(counts, original_values)
        assert counts.dtype == np.int64
        assert kwargs == BOUNDS | dict(settings=SETTINGS, initial_guesses=guesses)
        assert kwargs["settings"] is SETTINGS and kwargs["initial_guesses"] is guesses
        if use_prepared:
            np.testing.assert_array_equal(irf.kernel, prepared.kernel)
            assert irf.source.provenance == prepared.source.provenance
        else:
            np.testing.assert_allclose(
                irf, original_irf / np.trapezoid(original_irf, measurement.time_ns)
            )
        return returned

    monkeypatch.setattr(adapters, "fit_triexponential_reconvolution", fit)
    result = adapters.fit_experimental_triexponential_reconvolution(
        measurement,
        prepared_irf=prepared if use_prepared else None,
        initial_guesses=guesses,
        settings=SETTINGS,
        **BOUNDS,
    )
    assert result.fit is returned
    context = result.context
    assert context.normalization_performed is (not use_prepared)
    assert context.area_after_normalization == pytest.approx(1)
    assert context.measurement.provenance["file"] == ["counts-source"]
    measurement.provenance["file"].append("mutated")
    assert context.measurement.provenance["file"] == ["counts-source"]
    source = context.assumed_irf.source if use_prepared else context.assumed_irf
    assert source.provenance["file"] == (
        ["prepared-source"] if use_prepared else ["attached-source"]
    )
    np.testing.assert_array_equal(measurement.values, original_values)
    np.testing.assert_array_equal(measurement.irf.values, original_irf)
    assert not context.kernel.flags.writeable


def test_processed_normalized_observations_fail_before_dispatch(inputs, monkeypatch):
    measurement, prepared = inputs
    processed = replace(
        measurement,
        values=measurement.values / measurement.values.sum(),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    monkeypatch.setattr(
        adapters,
        "fit_triexponential_reconvolution",
        lambda *a, **kw: pytest.fail("fit dispatched"),
    )
    with pytest.raises(InvalidMeasurementError, match="raw"):
        adapters.fit_experimental_triexponential_reconvolution(
            processed, prepared_irf=prepared, settings=SETTINGS, **BOUNDS
        )


@pytest.mark.parametrize("kind", ["missing", "wrong_type", "wrong_grid"])
def test_irf_resolution_rejects_incompatible_inputs(inputs, kind):
    measurement, prepared = inputs
    if kind == "missing":
        measurement, prepared = replace(measurement, irf=None), None
    elif kind == "wrong_type":
        prepared = object()
    else:
        prepared = prepare_irf(
            prepared.source, measurement.time_ns + 0.02, resampling="linear"
        )
    with pytest.raises(InvalidMeasurementError):
        adapters.fit_experimental_triexponential_reconvolution(
            measurement, prepared_irf=prepared, settings=SETTINGS, **BOUNDS
        )


def test_explicit_settings_required(inputs):
    measurement, prepared = inputs
    with pytest.raises(TypeError, match="settings"):
        adapters.fit_experimental_triexponential_reconvolution(
            measurement, prepared_irf=prepared, **BOUNDS
        )
    with pytest.raises(TypeError, match="TriexponentialFitSettings"):
        adapters.fit_experimental_triexponential_reconvolution(
            measurement, prepared_irf=prepared, settings=None, **BOUNDS
        )


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(ROOT / "examples/triexponential_reconvolution.py"))


@pytest.fixture(scope="module")
def worked(example):
    return example["run_example"]()


def test_example_inputs_and_coincident_ambiguity_are_reproducible(example):
    first, second = example["build_inputs"](), example["build_inputs"]()
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            left["measurement"].values, right["measurement"].values
        )
        np.testing.assert_array_equal(left["prepared"].kernel, right["prepared"].kernel)
        assert left["measurement"].provenance == right["measurement"].provenance
    overlap = first[1]
    assert overlap["truth"].amplitudes != overlap["alternative"].amplitudes
    np.testing.assert_allclose(
        overlap["expected"], overlap["alternative_expected"], rtol=1e-14, atol=1e-12
    )


def test_worked_example_shapes_order_and_existing_diagnostics(worked):
    for case in worked:
        result = case["result"]
        fit = result.fit
        assert isinstance(fit, MultiexponentialFitResult)
        assert fit.assumptions.settings.solver == "trf"
        assert fit.parameters is not None
        assert list(fit.parameters.lifetimes_ns) == sorted(fit.parameters.lifetimes_ns)
        assert fit.expected_counts.shape == (256,)
        assert fit.component_counts.shape == (3, 256)
        np.testing.assert_allclose(
            fit.component_counts.sum(axis=0) + fit.parameters.background,
            fit.expected_counts,
        )
        assert fit.statistics.deviance_residuals.shape == (256,)
        assert fit.statistics.n_parameters == 8
        assert fit.statistics.deviance == pytest.approx(2 * fit.centered_nll)
        assert fit.assumptions.source_provenance == case["prepared"].source.provenance
        assert fit.optimization.work.model_calls > 0
        assert fit.identifiability.candidate == fit.parameters
        # Accuracy and numerical acceptance are outputs, never CI recovery gates.


def test_rejected_finite_result_is_preserved(worked, monkeypatch):
    case = worked[0]
    original = case["result"].fit
    rejected = replace(
        original,
        numerical=replace(
            original.numerical, accepted=False, reasons=("controlled_rejection",)
        ),
    )
    monkeypatch.setattr(
        adapters, "fit_triexponential_reconvolution", lambda *a, **kw: rejected
    )
    result = adapters.fit_experimental_triexponential_reconvolution(
        case["measurement"], prepared_irf=case["prepared"], settings=SETTINGS, **BOUNDS
    )
    assert result.fit is rejected
    assert result.fit.optimization is original.optimization
    assert result.fit.identifiability is original.identifiability
    assert result.fit.expected_counts is original.expected_counts


def test_example_plot_reuses_results_and_handles_unavailable_outputs(
    example, worked, tmp_path, monkeypatch
):
    monkeypatch.setitem(
        example["run_example"].__globals__,
        "fit_experimental_triexponential_reconvolution",
        lambda *a, **kw: pytest.fail("plot refitted"),
    )
    case = worked[0]
    empty = replace(
        case["result"].fit, expected_counts=None, component_counts=None, statistics=None
    )
    cases = [case, worked[1] | dict(result=replace(case["result"], fit=empty))]
    output = tmp_path / "example.png"
    example["save_figure"](output, cases)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
