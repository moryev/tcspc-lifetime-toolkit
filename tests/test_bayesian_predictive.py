"""Stage-4 conditional prediction, not calibration or mismatch evaluation."""

import builtins
import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.bayesian import (
    BayesianModelContext,
    BayesianPriorConfig,
    BoundedUniformPrior,
    GammaPrior,
    LogNormalPrior,
    fit_bayesian_monoexponential_reconvolution,
)
from tcspc_toolkit.bayesian_predictive import (
    calculate_bayesian_posterior_predictive_diagnostics,
    sample_bayesian_posterior_predictive,
    sample_bayesian_prior_predictive,
    summarize_predictive_band,
)
from tcspc_toolkit.bayesian_sampling import (
    BayesianPosteriorSamples,
    BayesianSamplingConfig,
)
from tcspc_toolkit.evaluation import calculate_poisson_deviance_residuals
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import generate_emg_irf_profile, generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import (
    MeasurementDataKind, SampledIRF, TCSPCMeasurement,
)
from tcspc_toolkit.simulation import sample_photon_counts


def _context_and_priors(*, free_shift: bool, emg: bool = False):
    time_ns = np.linspace(0.0, 6.0, 101)
    source = (
        generate_emg_irf_profile(
            time_ns, gaussian_centre_ns=0.7, gaussian_fwhm_ns=0.25,
            tail_time_ns=0.2,
        ) if emg else generate_gaussian_irf_profile(
            time_ns, gaussian_centre_ns=0.7, gaussian_fwhm_ns=0.25,
        )
    )
    prepared = prepare_irf(source, time_ns)
    shift = 0.05 if free_shift else 0.0
    generating_counts = monoexponential_reconvolution_expected_counts(
        time_ns, prepared.kernel, amplitude=400.0, lifetime=1.2,
        background=2.0, temporal_shift=shift,
    )
    measurement = TCSPCMeasurement(
        time_ns=time_ns,
        values=sample_photon_counts(generating_counts, np.random.default_rng(314)),
        data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id="stage-4-matched-synthetic",
    )
    context = BayesianModelContext(measurement, prepared_irf=prepared)
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(shape=4.0, rate=0.01),
        lifetime_ns=LogNormalPrior(log_mean=math.log(1.2), log_std=0.4),
        background_per_bin=GammaPrior(shape=2.0, rate=1.0),
        temporal_shift_prior=BoundedUniformPrior(-0.2, 0.2) if free_shift else None,
        fixed_temporal_shift_ns=None if free_shift else 0.0,
    )
    return context, priors


def _short_sampling_config(*, free_shift: bool) -> BayesianSamplingConfig:
    # Regression-only policy: exercises the real sampler but does not establish
    # scientific convergence or interval calibration.
    return BayesianSamplingConfig(
        random_seed=122 if free_shift else 121,
        n_walkers=12, warmup_steps=35, production_steps=50,
        max_production_steps=50, extension_steps=20, n_ensembles=2,
        min_autocorrelation_multiples=2.0,
        max_autocorrelation_relative_change=5.0,
        min_effective_samples=10.0,
        max_ensemble_location_difference_sd=3.0,
        min_mean_acceptance_fraction=0.01,
        max_mean_acceptance_fraction=0.99,
    )


@pytest.fixture(scope="module")
def fixed_inference_run():
    context, priors = _context_and_priors(free_shift=False)
    return fit_bayesian_monoexponential_reconvolution(
        context.measurement, prepared_irf=context.prepared_irf,
        priors=priors, sampling_config=_short_sampling_config(free_shift=False),
    )


@pytest.fixture(scope="module")
def free_emg_inference_run():
    context, priors = _context_and_priors(free_shift=True, emg=True)
    return fit_bayesian_monoexponential_reconvolution(
        context.measurement, prepared_irf=context.prepared_irf,
        priors=priors, sampling_config=_short_sampling_config(free_shift=True),
    )


@pytest.mark.parametrize("free_shift,emg", [(False, False), (True, True)])
def test_prior_predictive_support_physics_and_reproducibility(free_shift, emg) -> None:
    context, priors = _context_and_priors(free_shift=free_shift, emg=emg)
    result = sample_bayesian_prior_predictive(
        context, priors, n_draws=35, random_seed=81, interval_level=0.9
    )
    repeated = sample_bayesian_prior_predictive(
        context, priors, n_draws=35, random_seed=81, interval_level=0.9
    )
    different = sample_bayesian_prior_predictive(
        context, priors, n_draws=35, random_seed=82, interval_level=0.9
    )
    assert result.physical_parameter_draws.shape == (35, 4)
    assert result.expected_counts.shape == (35, context.time_ns.size)
    assert result.replicated_counts.shape == (35, context.time_ns.size)
    assert np.all(result.physical_parameter_draws[:, :3] > 0.0)
    assert np.all(np.isfinite(result.expected_counts))
    assert np.all(result.expected_counts >= 0.0)
    assert np.issubdtype(result.replicated_counts.dtype, np.integer)
    assert np.all(result.replicated_counts >= 0)
    if free_shift:
        assert np.all((-0.2 <= result.physical_parameter_draws[:, 3]) & (
            result.physical_parameter_draws[:, 3] <= 0.2
        ))
    else:
        np.testing.assert_array_equal(result.physical_parameter_draws[:, 3], 0.0)
    np.testing.assert_array_equal(result.physical_parameter_draws, repeated.physical_parameter_draws)
    np.testing.assert_array_equal(result.replicated_counts, repeated.replicated_counts)
    assert not np.array_equal(result.physical_parameter_draws, different.physical_parameter_draws)
    first = result.physical_parameter_draws[0]
    np.testing.assert_array_equal(result.expected_counts[0],
        monoexponential_reconvolution_expected_counts(
            context.time_ns, context.irf_kernel,
            amplitude=first[0], lifetime=first[1],
            background=first[2], temporal_shift=first[3],
        )
    )
    assert result.model_context is context
    assert result.fixed_irf_assumption
    for array in (
        result.physical_parameter_draws, result.expected_counts,
        result.replicated_counts, result.prior_expected_count_band.median,
        result.prior_predictive_count_band.upper,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_prior_predictive_never_uses_observed_histogram_to_change_prior_draws() -> None:
    context, priors = _context_and_priors(free_shift=True, emg=True)
    other_measurement = TCSPCMeasurement(
        time_ns=context.time_ns,
        values=np.zeros_like(context.counts),
        data_kind=MeasurementDataKind.RAW_COUNTS,
    )
    other_context = BayesianModelContext(
        other_measurement, prepared_irf=context.prepared_irf
    )
    first = sample_bayesian_prior_predictive(
        context, priors, n_draws=12, random_seed=7, interval_level=0.8
    )
    second = sample_bayesian_prior_predictive(
        other_context, priors, n_draws=12, random_seed=7, interval_level=0.8
    )
    np.testing.assert_array_equal(first.physical_parameter_draws, second.physical_parameter_draws)
    np.testing.assert_array_equal(first.expected_counts, second.expected_counts)
    np.testing.assert_array_equal(first.replicated_counts, second.replicated_counts)


@pytest.mark.parametrize("fixture_name", ["fixed_inference_run", "free_emg_inference_run"])
def test_posterior_expected_curves_match_shared_forward_model(
    fixture_name, request
) -> None:
    run = request.getfixturevalue(fixture_name)
    result = sample_bayesian_posterior_predictive(
        run, n_draws=14, random_seed=20, interval_level=0.9,
    )
    assert result.selected_posterior_indices.shape == (14, 3)
    assert result.physical_parameter_draws.shape == (14, 4)
    assert result.expected_counts.shape == (14, run.context.time_ns.size)
    assert result.replicated_counts.shape == (14, run.context.time_ns.size)
    assert result.model_context is run.context
    assert result.priors is run.result.priors
    assert result.sampling_config is run.result.sampling_config
    assert result.inference_sampling_status is run.result.status
    assert result.fixed_irf_assumption
    for index in (0, 5, 13):
        ensemble, step, walker = result.selected_posterior_indices[index]
        np.testing.assert_array_equal(
            result.physical_parameter_draws[index],
            run.samples.physical[ensemble, step, walker],
        )
        amplitude, lifetime, background, shift = result.physical_parameter_draws[index]
        direct = monoexponential_reconvolution_expected_counts(
            run.context.time_ns, run.context.irf_kernel,
            amplitude=amplitude, lifetime=lifetime,
            background=background, temporal_shift=shift,
        )
        np.testing.assert_array_equal(result.expected_counts[index], direct)
    if fixture_name == "fixed_inference_run":
        np.testing.assert_array_equal(result.physical_parameter_draws[:, 3], 0.0)
    else:
        assert np.ptp(result.physical_parameter_draws[:, 3]) > 0.0


def test_posterior_selection_policy_and_reproducibility(fixed_inference_run) -> None:
    run = fixed_inference_run
    without = sample_bayesian_posterior_predictive(
        run, n_draws=40, random_seed=15, interval_level=0.9
    )
    repeated = sample_bayesian_posterior_predictive(
        run, n_draws=40, random_seed=15, interval_level=0.9
    )
    different = sample_bayesian_posterior_predictive(
        run, n_draws=40, random_seed=16, interval_level=0.9
    )
    assert not without.selection_with_replacement
    assert np.unique(without.selected_posterior_indices, axis=0).shape[0] == 40
    np.testing.assert_array_equal(without.selected_posterior_indices,
                                  repeated.selected_posterior_indices)
    np.testing.assert_array_equal(without.expected_counts, repeated.expected_counts)
    np.testing.assert_array_equal(without.replicated_counts, repeated.replicated_counts)
    assert not np.array_equal(without.selected_posterior_indices,
                              different.selected_posterior_indices)


def test_replacement_policy_and_poisson_mean_variance_with_constant_expected_curve(
    fixed_inference_run,
) -> None:
    run = fixed_inference_run
    physical = np.broadcast_to(
        np.array([400.0, 1.2, 2.0, 0.0]), (2, 1, 2, 4)
    ).copy()
    transformed = np.log(physical[..., :3])
    samples = BayesianPosteriorSamples(
        physical=physical, transformed=transformed,
        log_probability=np.zeros((2, 1, 2)),
    )
    controlled_run = replace(run, samples=samples)
    result = sample_bayesian_posterior_predictive(
        controlled_run, n_draws=600, random_seed=37, interval_level=0.9
    )
    assert result.selection_with_replacement
    assert result.retained_posterior_sample_count == 4
    assert np.unique(result.selected_posterior_indices, axis=0).shape[0] <= 4
    np.testing.assert_array_equal(result.expected_counts[0], result.expected_counts[-1])
    np.testing.assert_array_equal(
        result.posterior_expected_count_band.lower,
        result.posterior_expected_count_band.upper,
    )
    peak_bin = int(np.argmax(result.expected_counts[0]))
    assert (
        result.posterior_predictive_count_band.lower[peak_bin]
        < result.posterior_predictive_count_band.upper[peak_bin]
    )
    expected_mean = result.expected_counts[0, peak_bin]
    sampled = result.replicated_counts[:, peak_bin]
    assert np.mean(sampled) == pytest.approx(expected_mean, rel=0.10)
    assert np.var(sampled, ddof=1) == pytest.approx(expected_mean, rel=0.25)


def test_predictive_bands_are_independent_empirical_quantiles() -> None:
    draws = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]])
    band = summarize_predictive_band(draws, interval_level=0.5)
    np.testing.assert_array_equal(band.median, [1.5, 25.0])
    np.testing.assert_array_equal(band.lower, [0.75, 17.5])
    np.testing.assert_array_equal(band.upper, [2.25, 32.5])
    assert band.interval_level == 0.5


def test_predictive_arrays_and_diagnostics_are_read_only(fixed_inference_run) -> None:
    result = sample_bayesian_posterior_predictive(
        fixed_inference_run, n_draws=10, random_seed=5, interval_level=0.9,
        early_window_ns=(0.0, 1.5), tail_window_ns=(4.0, 6.1),
    )
    arrays = (
        result.selected_posterior_indices, result.physical_parameter_draws,
        result.expected_counts, result.replicated_counts,
        result.posterior_expected_count_band.median,
        result.posterior_predictive_count_band.lower,
        result.diagnostics.mean_observed_signed_deviance_residual_profile,
        result.diagnostics.total_counts.replicated,
        result.diagnostics.windows[0].total_counts.replicated,
    )
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert not hasattr(result.diagnostics, "model_valid")


def test_controlled_diagnostics_match_existing_residuals_and_half_open_windows() -> None:
    time_ns = np.arange(5.0)
    measurement = TCSPCMeasurement(
        time_ns=time_ns, values=[2, 4, 1, 0, 3],
        data_kind=MeasurementDataKind.RAW_COUNTS,
        irf=SampledIRF(time_ns=time_ns, values=[0, 1, 0, 0, 0]),
    )
    context = BayesianModelContext(measurement)
    expected = np.array([[2.0, 3.0, 1.0, 1.0, 2.0]] * 2)
    replicated = np.array([[1, 5, 0, 1, 3], [3, 3, 2, 0, 2]], dtype=np.int64)
    diagnostics = calculate_bayesian_posterior_predictive_diagnostics(
        context, expected, replicated,
        early_window_ns=(0.0, 2.0), tail_window_ns=(3.0, 5.0),
    )
    reference_residual = calculate_poisson_deviance_residuals(context.counts, expected[0])
    np.testing.assert_array_equal(
        diagnostics.mean_observed_signed_deviance_residual_profile,
        reference_residual,
    )
    np.testing.assert_array_equal(
        diagnostics.poisson_deviance.observed,
        [np.sum(reference_residual ** 2)] * 2,
    )
    reference_replicated_deviance = [
        np.sum(calculate_poisson_deviance_residuals(rep, curve) ** 2)
        for rep, curve in zip(replicated, expected)
    ]
    np.testing.assert_allclose(
        diagnostics.poisson_deviance.replicated,
        reference_replicated_deviance,
    )
    assert diagnostics.total_counts.observed == 10.0
    np.testing.assert_array_equal(diagnostics.total_counts.replicated, [10.0, 10.0])
    assert diagnostics.total_counts.posterior_predictive_tail_probability == 1.0
    assert diagnostics.peak_counts.observed == 4.0
    assert diagnostics.peak_time_ns.observed == 1.0
    assert len(diagnostics.windows) == 2
    assert diagnostics.windows[0].name == "early_window_ns"
    assert diagnostics.windows[0].total_counts.observed == 6.0
    np.testing.assert_array_equal(diagnostics.windows[0].total_counts.replicated, [6.0, 6.0])
    assert diagnostics.windows[1].total_counts.observed == 3.0
    np.testing.assert_array_equal(diagnostics.windows[1].total_counts.replicated, [4.0, 2.0])
    assert diagnostics.windows[1].total_counts.posterior_predictive_tail_probability == 0.5
    assert 0.0 <= diagnostics.poisson_deviance.posterior_predictive_tail_probability <= 1.0
    assert 0.0 <= diagnostics.rms_signed_deviance_residual.posterior_predictive_tail_probability <= 1.0
    assert 0.0 <= diagnostics.maximum_absolute_signed_deviance_residual.posterior_predictive_tail_probability <= 1.0


@pytest.mark.parametrize("window", [(2.0, 2.0), (7.0, 8.0), (math.nan, 2.0)])
def test_invalid_or_empty_caller_windows_are_rejected(window) -> None:
    context, _ = _context_and_priors(free_shift=False)
    expected = np.ones((2, context.time_ns.size))
    replicated = np.ones_like(expected, dtype=np.int64)
    with pytest.raises(ValueError, match="early_window_ns"):
        calculate_bayesian_posterior_predictive_diagnostics(
            context, expected, replicated, early_window_ns=window,
        )


@pytest.mark.parametrize("overrides", [
    {"n_draws": 0}, {"n_draws": True}, {"random_seed": -1},
    {"interval_level": 0.0}, {"interval_level": 1.0},
])
def test_invalid_predictive_requests_are_rejected(overrides) -> None:
    context, priors = _context_and_priors(free_shift=False)
    arguments = dict(n_draws=4, random_seed=1, interval_level=0.9)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        sample_bayesian_prior_predictive(context, priors, **arguments)


def test_real_inference_to_prediction_preserves_plausible_observed_totals(
    fixed_inference_run,
) -> None:
    result = sample_bayesian_posterior_predictive(
        fixed_inference_run, n_draws=40, random_seed=700,
        interval_level=0.9,
        early_window_ns=(0.0, 1.5), tail_window_ns=(4.0, 6.1),
    )
    observed_total = int(np.sum(fixed_inference_run.context.counts))
    replicated_totals = result.diagnostics.total_counts.replicated
    assert abs(observed_total - np.median(replicated_totals)) < max(
        5.0 * np.sqrt(observed_total), 0.10 * observed_total
    )
    assert result.diagnostics.windows[0].lower_ns == 0.0
    assert result.diagnostics.windows[1].upper_ns == 6.1


def test_predictive_import_and_prior_sampling_need_no_emcee() -> None:
    script = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'emcee' or name.startswith('emcee.'):
        raise ModuleNotFoundError("No module named 'emcee'", name='emcee')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import numpy as np
from tcspc_toolkit.bayesian import BayesianModelContext, BayesianPriorConfig, GammaPrior, LogNormalPrior
from tcspc_toolkit.bayesian_predictive import sample_bayesian_prior_predictive
from tcspc_toolkit.measurements import TCSPCMeasurement, SampledIRF, MeasurementDataKind
time = np.linspace(0.0, 2.0, 11)
measurement = TCSPCMeasurement(
    time_ns=time, values=np.ones(time.size, dtype=int),
    data_kind=MeasurementDataKind.RAW_COUNTS,
    irf=SampledIRF(time_ns=time, values=np.ones(time.size)),
)
context = BayesianModelContext(measurement)
priors = BayesianPriorConfig(
    GammaPrior(2.0, 1.0), LogNormalPrior(0.0, 1.0), GammaPrior(2.0, 1.0),
    temporal_shift_prior=None, fixed_temporal_shift_ns=0.0,
)
result = sample_bayesian_prior_predictive(
    context, priors, n_draws=2, random_seed=3, interval_level=0.9,
)
assert result.expected_counts.shape == (2, 11)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_posterior_prediction_from_existing_run_never_imports_emcee(
    fixed_inference_run, monkeypatch,
) -> None:
    original_import = builtins.__import__

    def block_emcee(name, *args, **kwargs):
        if name == "emcee" or name.startswith("emcee."):
            raise AssertionError("posterior prediction attempted to import emcee")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "__import__", block_emcee)
        result = sample_bayesian_posterior_predictive(
            fixed_inference_run, n_draws=5, random_seed=2, interval_level=0.8,
        )
    assert result.expected_counts.shape[0] == 5
