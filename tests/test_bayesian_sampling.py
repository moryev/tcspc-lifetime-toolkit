"""Fast real-emcee regression checks, not scientific convergence studies."""

import math
import subprocess
import sys

import numpy as np
import pytest

import tcspc_toolkit.bayesian_sampling as sampling_module
from tcspc_toolkit.bayesian import (
    BayesianPriorConfig,
    BoundedUniformPrior,
    GammaPrior,
    IRFModelRelation,
    LogNormalPrior,
    fit_bayesian_monoexponential_reconvolution,
)
from tcspc_toolkit.bayesian_sampling import (
    BayesianInitializationError,
    BayesianPosteriorSamples,
    BayesianSamplingConfig,
    BayesianSamplingStatus,
)
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import generate_emg_irf_profile, generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, SampledIRF, TCSPCMeasurement


def _fixture(*, free_shift: bool, emg: bool = False):
    time = np.linspace(0.0, 7.0, 141)
    if emg:
        source = generate_emg_irf_profile(
            time, gaussian_centre_ns=0.7, gaussian_fwhm_ns=0.25,
            tail_time_ns=0.20,
        )
    else:
        source = generate_gaussian_irf_profile(
            time, gaussian_centre_ns=0.7, gaussian_fwhm_ns=0.25,
        )
    prepared = prepare_irf(source, time)
    true = (700.0, 1.25, 2.5, 0.06 if free_shift else 0.0)
    expected = monoexponential_reconvolution_expected_counts(
        time, prepared.kernel, amplitude=true[0], lifetime=true[1],
        background=true[2], temporal_shift=true[3],
    )
    counts = np.random.default_rng(1824).poisson(expected)
    measurement = TCSPCMeasurement(
        time_ns=time, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id="matched-synthetic-regression",
        provenance={"origin": "controlled_synthetic_test"},
    )
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(shape=4.0, rate=4.0 / 700.0),
        lifetime_ns=LogNormalPrior(log_mean=math.log(1.25), log_std=0.5),
        background_per_bin=GammaPrior(shape=2.0, rate=2.0 / 2.5),
        temporal_shift_prior=BoundedUniformPrior(-0.2, 0.2) if free_shift else None,
        fixed_temporal_shift_ns=None if free_shift else 0.0,
    )
    return measurement, prepared, priors, true


def _fast_config(seed: int = 35, *, production_steps: int = 100) -> BayesianSamplingConfig:
    # Regression-only thresholds intentionally do not certify convergence.
    return BayesianSamplingConfig(
        random_seed=seed, n_walkers=16, warmup_steps=70,
        production_steps=production_steps,
        max_production_steps=production_steps,
        extension_steps=40, n_ensembles=2,
        credible_interval_level=0.90,
        min_autocorrelation_multiples=2.0,
        max_autocorrelation_relative_change=5.0,
        min_effective_samples=10.0,
        max_ensemble_location_difference_sd=3.0,
        min_mean_acceptance_fraction=0.01,
        max_mean_acceptance_fraction=0.99,
    )


@pytest.fixture(scope="module")
def fixed_run():
    measurement, prepared, priors, true = _fixture(free_shift=False)
    run = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors,
        sampling_config=_fast_config(), irf_model_relation=IRFModelRelation.MATCHED,
    )
    return run, true


@pytest.fixture(scope="module")
def free_run():
    measurement, prepared, priors, true = _fixture(free_shift=True, emg=True)
    run = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors,
        sampling_config=_fast_config(seed=36, production_steps=130),
        irf_model_relation=IRFModelRelation.MATCHED,
    )
    return run, true


def test_fixed_shift_gaussian_recovery_and_result_contract(fixed_run) -> None:
    run, true = fixed_run
    result = run.result
    assert result.status is BayesianSamplingStatus.SUCCESS
    assert result.inferred_parameter_names == (
        "amplitude", "lifetime_ns", "background_per_bin"
    )
    assert len(result.parameter_summaries) == 3
    assert run.samples.physical.shape == (2, 100, 16, 4)
    assert run.samples.transformed.shape == (2, 100, 16, 3)
    assert run.samples.log_probability.shape == (2, 100, 16)
    np.testing.assert_array_equal(run.samples.physical[..., 3], 0.0)
    assert result.correlation_matrix.shape == (3, 3)
    assert result.model_context.irf_model_relation is IRFModelRelation.MATCHED
    assert result.model_context.fixed_irf_assumption
    assert result.runtime.total_seconds > 0.0
    assert result.runtime.sampling_seconds > 0.0
    summaries = result.parameter_summaries
    assert summaries[0].median == pytest.approx(true[0], rel=0.25)
    assert summaries[1].median == pytest.approx(true[1], rel=0.20)
    assert summaries[2].median == pytest.approx(true[2], rel=0.8)
    for summary in summaries:
        assert summary.credible_lower < summary.median < summary.credible_upper


def test_free_shift_emg_recovery_and_four_parameter_labels(free_run) -> None:
    run, true = free_run
    assert run.result.status is BayesianSamplingStatus.SUCCESS
    assert run.result.inferred_parameter_names == (
        "amplitude", "lifetime_ns", "background_per_bin", "temporal_shift_ns"
    )
    assert run.samples.physical.shape == (2, 130, 16, 4)
    assert run.samples.transformed.shape == (2, 130, 16, 4)
    assert run.result.correlation_matrix.shape == (4, 4)
    median = [summary.median for summary in run.result.parameter_summaries]
    assert median[0] == pytest.approx(true[0], rel=0.25)
    assert median[1] == pytest.approx(true[1], rel=0.20)
    assert median[2] == pytest.approx(true[2], rel=0.8)
    assert median[3] == pytest.approx(true[3], abs=0.055)
    assert run.context.irf_source is run.context.prepared_irf.source


def test_sample_buffers_are_defensively_copied_and_read_only(fixed_run) -> None:
    run, _ = fixed_run
    for array in (
        run.samples.physical, run.samples.transformed, run.samples.log_probability,
        run.result.correlation_matrix, run.result.diagnostics.acceptance_fraction,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)

    physical = np.ones((2, 2, 8, 4))
    transformed = np.ones((2, 2, 8, 3))
    log_probability = np.ones((2, 2, 8))
    samples = BayesianPosteriorSamples(
        physical=physical, transformed=transformed,
        log_probability=log_probability,
        sampling_coordinate_names=(
            "log_amplitude", "log_lifetime_ns", "log_background_per_bin"
        ),
    )
    physical[0, 0, 0, 0] = 99.0
    assert samples.physical[0, 0, 0, 0] == 1.0


def test_same_seed_reproduces_chains_and_different_seed_changes_chains() -> None:
    measurement, prepared, priors, _ = _fixture(free_shift=False)
    config = _fast_config(seed=112, production_steps=45)
    first = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors, sampling_config=config
    )
    second = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors, sampling_config=config
    )
    different = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors,
        sampling_config=_fast_config(seed=113, production_steps=45),
    )
    np.testing.assert_array_equal(first.samples.transformed, second.samples.transformed)
    np.testing.assert_array_equal(first.samples.log_probability, second.samples.log_probability)
    assert not np.array_equal(first.samples.transformed, different.samples.transformed)
    assert first.result.parameter_summaries[1].median == pytest.approx(
        different.result.parameter_summaries[1].median, rel=0.3
    )


def test_public_sampler_rejects_processed_histograms_and_invalid_irfs() -> None:
    measurement, prepared, priors, _ = _fixture(free_shift=False)
    processed = TCSPCMeasurement(
        time_ns=measurement.time_ns,
        values=measurement.values.astype(float),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    with pytest.raises(InvalidMeasurementError, match="raw photon counts"):
        fit_bayesian_monoexponential_reconvolution(
            processed, prepared_irf=prepared, priors=priors,
            sampling_config=_fast_config(),
        )
    with pytest.raises(InvalidMeasurementError, match="measured IRF"):
        fit_bayesian_monoexponential_reconvolution(
            measurement, priors=priors, sampling_config=_fast_config()
        )
    mismatched = prepare_irf(
        prepared.source, measurement.time_ns + 0.01, resampling="linear"
    )
    with pytest.raises(InvalidMeasurementError, match="prepare the IRF explicitly"):
        fit_bayesian_monoexponential_reconvolution(
            measurement, prepared_irf=mismatched, priors=priors,
            sampling_config=_fast_config(),
        )


def test_public_sampler_uses_attached_sampled_irf_fallback() -> None:
    measurement, prepared, priors, _ = _fixture(free_shift=False)
    attached = SampledIRF(
        time_ns=measurement.time_ns, values=5.0 * prepared.kernel,
        provenance={"origin": "imported_sampled_test"},
    )
    with_irf = TCSPCMeasurement(
        time_ns=measurement.time_ns, values=measurement.values,
        data_kind=MeasurementDataKind.RAW_COUNTS, irf=attached,
    )
    run = fit_bayesian_monoexponential_reconvolution(
        with_irf, priors=priors, sampling_config=_fast_config(production_steps=20)
    )
    assert run.context.irf_selection == "attached_sampled"
    assert run.result.model_context.irf_selection == "attached_sampled"
    assert run.context.irf_source is attached
    np.testing.assert_allclose(run.context.irf_kernel, prepared.kernel)


@pytest.mark.parametrize("overrides", [
    {"n_walkers": 0}, {"warmup_steps": 0}, {"production_steps": -1},
    {"max_production_steps": 0}, {"extension_steps": 0},
    {"n_ensembles": 1}, {"credible_interval_level": 1.0},
    {"credible_interval_level": 0.0},
    {"production_steps": 20, "max_production_steps": 10},
    {"min_autocorrelation_multiples": 0.0},
    {"max_autocorrelation_relative_change": -1.0},
    {"min_effective_samples": 0.0},
    {"max_ensemble_location_difference_sd": 0.0},
    {"random_seed": -1},
])
def test_invalid_sampling_configuration_is_rejected(overrides) -> None:
    parameters = {"random_seed": 5, **overrides}
    with pytest.raises(ValueError):
        BayesianSamplingConfig(**parameters)


def test_walker_count_is_validated_against_model_dimension() -> None:
    measurement, prepared, priors, _ = _fixture(free_shift=True)
    with pytest.raises(ValueError, match="at least 8"):
        fit_bayesian_monoexponential_reconvolution(
            measurement, prepared_irf=prepared, priors=priors,
            sampling_config=BayesianSamplingConfig(random_seed=5, n_walkers=7),
        )


def test_maximum_step_budget_returns_insufficient_sampling_with_partial_draws() -> None:
    measurement, prepared, priors, _ = _fixture(free_shift=False)
    config = BayesianSamplingConfig(
        random_seed=99, n_walkers=8, n_ensembles=2,
        warmup_steps=10, production_steps=12,
        extension_steps=7, max_production_steps=26,
        min_autocorrelation_multiples=1_000_000.0,
    )
    run = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors, sampling_config=config
    )
    assert run.result.status is BayesianSamplingStatus.INSUFFICIENT_SAMPLING
    assert run.result.diagnostics.production_steps == 26
    assert run.result.diagnostics.extension_count == 2
    assert run.result.diagnostics.retained_samples == 2 * 26 * 8
    assert run.samples.physical.shape == (2, 26, 8, 4)
    assert len(run.result.parameter_summaries) == 3


def test_initialization_failure_has_explicit_status_and_empty_samples(monkeypatch) -> None:
    measurement, prepared, priors, _ = _fixture(free_shift=False)

    def fail_initialization(*args, **kwargs):
        raise BayesianInitializationError("no valid walker state")

    monkeypatch.setattr(sampling_module, "_initialize_walkers", fail_initialization)
    run = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors,
        sampling_config=_fast_config(production_steps=20),
    )
    assert run.result.status is BayesianSamplingStatus.INITIALIZATION_FAILED
    assert run.result.failure_reason == "no valid walker state"
    assert run.samples.physical.shape == (2, 0, 16, 4)
    assert run.result.parameter_summaries == ()
    assert not run.result.diagnostics.accepted


def test_numerical_sampler_failure_has_explicit_status(monkeypatch) -> None:
    import emcee

    measurement, prepared, priors, _ = _fixture(free_shift=False)

    class FailingSampler:
        def __init__(self, n_walkers, ndim, target, args):
            self.iteration = 0
            self.acceptance_fraction = np.full(n_walkers, np.nan)

        def run_mcmc(self, *args, **kwargs):
            raise FloatingPointError("controlled sampler numerical failure")

    monkeypatch.setattr(emcee, "EnsembleSampler", FailingSampler)
    run = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors,
        sampling_config=_fast_config(production_steps=20),
    )
    assert run.result.status is BayesianSamplingStatus.NUMERICAL_FAILURE
    assert run.result.failure_reason == "controlled sampler numerical failure"
    assert run.samples.physical.shape == (2, 0, 16, 4)


def test_optional_dependency_is_imported_only_when_sampling_is_called() -> None:
    script = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'emcee' or name.startswith('emcee.'):
        raise ModuleNotFoundError("No module named 'emcee'", name='emcee')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import numpy as np
from tcspc_toolkit.bayesian import (
    BayesianPriorConfig, GammaPrior, LogNormalPrior,
    fit_bayesian_monoexponential_reconvolution,
)
from tcspc_toolkit.bayesian_sampling import BayesianSamplingConfig
from tcspc_toolkit.measurements import TCSPCMeasurement, SampledIRF, MeasurementDataKind
time = np.linspace(0.0, 2.0, 11)
measurement = TCSPCMeasurement(
    time_ns=time, values=np.ones(time.size, dtype=int),
    data_kind=MeasurementDataKind.RAW_COUNTS,
    irf=SampledIRF(time_ns=time, values=np.ones(time.size)),
)
priors = BayesianPriorConfig(
    GammaPrior(2.0, 1.0), LogNormalPrior(0.0, 1.0), GammaPrior(2.0, 1.0),
    temporal_shift_prior=None, fixed_temporal_shift_ns=0.0,
)
try:
    fit_bayesian_monoexponential_reconvolution(
        measurement, priors=priors,
        sampling_config=BayesianSamplingConfig(random_seed=2, n_walkers=8),
    )
except ImportError as error:
    assert 'optional emcee extra' in str(error)
else:
    raise AssertionError('expected actionable missing-extra error')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
