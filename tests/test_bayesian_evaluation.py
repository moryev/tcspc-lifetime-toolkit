"""Stage-5 paired plumbing and accounting, not a scientific calibration run."""

import json
import math
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tcspc_toolkit.bayesian_evaluation as evaluation
from tcspc_toolkit.bayesian import BayesianPriorConfig, BoundedUniformPrior, GammaPrior, LogNormalPrior
from tcspc_toolkit.bayesian_evaluation import (
    BayesianClassicalCondition,
    BayesianClassicalEvaluationConfig,
    BayesianClassicalRealizationResult,
    Issue4Bayesian,
    Issue4Bootstrap,
    Issue4ClassicalFit,
    Issue4Covariance,
    Issue4RealizationSeeds,
    build_issue4_expected_counts,
    derive_issue4_seed,
    evaluate_issue4_realization,
    issue4_realization_seeds,
    sample_issue4_observation,
    summarize_issue4_condition,
)
from tcspc_toolkit.bayesian_sampling import (
    BayesianParameterSummary, BayesianRuntime, BayesianSamplingConfig, BayesianSamplingStatus,
)
from tcspc_toolkit.classical_evaluation import ReconvolutionCurveResult
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement


def _condition_and_config():
    time = np.arange(0.0, 6.0, 0.06)
    prepared = prepare_irf(
        generate_gaussian_irf_profile(time, gaussian_centre_ns=0.7, gaussian_fwhm_ns=0.25),
        time,
    )
    condition = BayesianClassicalCondition(
        condition_id="matched-medium", time_ns=time,
        generating_irf=prepared, assumed_irf=prepared,
        true_lifetime_ns=1.5, signal_photon_count=3000,
        background_per_bin=1.0, true_temporal_shift_ns=0.04, n_repeats=2,
    )
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(1.0, 0.001),
        lifetime_ns=LogNormalPrior(math.log(2.0), 0.8),
        background_per_bin=GammaPrior(1.0, 0.2),
        temporal_shift_prior=BoundedUniformPrior(-0.2, 0.2),
        fixed_temporal_shift_ns=None,
    )
    config = BayesianClassicalEvaluationConfig(
        profile="smoke", prior_policy_id="baseline", priors=priors,
        sampling_config=BayesianSamplingConfig(
            random_seed=0, n_walkers=12, warmup_steps=10,
            production_steps=10, max_production_steps=10,
            extension_steps=5, n_ensembles=2, credible_interval_level=0.9,
        ),
        n_bootstrap_resamples=3, nominal_interval_level=0.9,
        temporal_shift_bounds_ns=(-0.2, 0.2), random_seed=9405,
    )
    return condition, config


def _toy_record(index: int, *, classical_lifetime: float = 2.0,
                covariance_valid: bool = True, bootstrap_valid: bool = True,
                bayesian_status: BayesianSamplingStatus = BayesianSamplingStatus.SUCCESS,
                covariance_bounds=(1.9, 2.1), bootstrap_bounds=(1.9, 2.1),
                bayesian_bounds=(1.9, 2.1)) -> BayesianClassicalRealizationResult:
    return BayesianClassicalRealizationResult(
        condition_id="toy", prior_policy_id="baseline", profile="smoke",
        realization_index=index, true_lifetime_ns=2.0,
        signal_photon_count=1000, true_reconvolution_amplitude=25.0,
        background_per_bin=1.0, true_temporal_shift_ns=0.0,
        observed_total_counts=1100, observed_counts_sha256="a" * 64,
        seeds=Issue4RealizationSeeds(1, 2, 3, 4),
        classical_fit=Issue4ClassicalFit(
            True, True, classical_lifetime, 25.0, 1.0, 0.0, False, None, 1.0, 1.1,
        ),
        covariance=Issue4Covariance(
            covariance_valid, 0.1 if covariance_valid else math.nan,
            *covariance_bounds, 100.0, not covariance_valid,
            None if covariance_valid else "boundary_hit", 0.2,
        ),
        bootstrap=Issue4Bootstrap(
            bootstrap_valid, 2.0, 0.12 if bootstrap_valid else math.nan,
            *bootstrap_bounds, 5, 5 if bootstrap_valid else 0,
            0.0 if bootstrap_valid else 1.0,
            None if bootstrap_valid else "fewer_than_two_successful_refits", 0.4,
        ),
        bayesian=Issue4Bayesian(
            status=bayesian_status,
            diagnostics_accepted=bayesian_status is BayesianSamplingStatus.SUCCESS,
            lifetime_mean_ns=2.0, lifetime_median_ns=2.0,
            posterior_lifetime_std_ns=0.15,
            credible_lower_ns=bayesian_bounds[0],
            credible_upper_ns=bayesian_bounds[1],
            mean_acceptance_fraction=0.4, minimum_effective_samples=200.0,
            production_steps=100, retained_samples=2400, extension_count=0,
            minimum_autocorrelation_multiples=50.0,
            maximum_autocorrelation_relative_change=0.05,
            maximum_ensemble_mean_difference_sd=0.1,
            maximum_ensemble_median_difference_sd=0.1,
            lifetime_background_correlation=-0.2,
            lifetime_shift_correlation=0.3,
            diagnostic_failure_reasons=(
                () if bayesian_status is BayesianSamplingStatus.SUCCESS else ("too_short",)
            ),
            initialization_seconds=0.1, sampling_seconds=0.8,
            diagnostic_seconds=0.1, inference_total_seconds=1.0,
            call_seconds=1.1,
        ),
        orchestration_seconds=2.8,
    )


def _fake_successful_bayesian_inference():
    summaries = tuple(BayesianParameterSummary(
        name, unit, 1.5, 1.5, 0.1, 1.3, 1.7
    ) for name, unit in (
        ("amplitude", "scale"), ("lifetime_ns", "ns"),
        ("background_per_bin", "counts_per_bin"), ("temporal_shift_ns", "ns"),
    ))
    return SimpleNamespace(result=SimpleNamespace(
        status=BayesianSamplingStatus.SUCCESS, parameter_summaries=summaries,
        correlation_matrix=np.eye(4),
        diagnostics=SimpleNamespace(
            accepted=True, approximate_effective_samples=np.ones(4) * 100,
            mean_acceptance_fraction=0.4, failure_reasons=(),
            production_steps=100, retained_samples=2400, extension_count=0,
            autocorrelation_time_steps=np.ones((2, 4)) * 10,
            autocorrelation_relative_change=np.ones((2, 4)) * 0.05,
            maximum_ensemble_mean_difference_sd=0.1,
            maximum_ensemble_median_difference_sd=0.1,
        ),
        runtime=BayesianRuntime(0.1, 0.2, 0.1, 0.4),
    ))


def test_true_reconvolution_amplitude_is_not_signal_photon_budget() -> None:
    condition, _ = _condition_and_config()
    expected, true_amplitude = build_issue4_expected_counts(condition)
    assert true_amplitude != condition.signal_photon_count
    assert np.sum(expected - condition.background_per_bin) == pytest.approx(
        condition.signal_photon_count
    )
    assert expected.shape == condition.time_ns.shape
    assert np.all(expected >= 0.0)


def test_all_methods_are_paired_to_one_raw_observation(monkeypatch) -> None:
    condition, config = _condition_and_config()
    expected, amplitude = build_issue4_expected_counts(condition)
    seeds = issue4_realization_seeds(config, condition.condition_id, 0)
    measurement = sample_issue4_observation(condition, expected, seeds.observation, 0)
    observed = measurement.require_raw_counts().copy()
    seen = []

    def fake_classical(**kwargs):
        np.testing.assert_array_equal(kwargs["counts"], observed)
        assert kwargs["objective"] == "poisson"
        assert kwargs["temporal_shift_bounds"] == (-0.2, 0.2)
        seen.append("classical")
        return ReconvolutionCurveResult(
            30.0, 1.5, 1.0, 0.0, 30.0, 1.5, 1.0, 0.04,
            True, True, False, 0.0, 0.0, 10.0, None, None,
        )

    def fake_covariance(**kwargs):
        assert kwargs["fit_result"].lifetime == 1.5
        np.testing.assert_array_equal(kwargs["irf"], condition.assumed_irf.kernel)
        seen.append("covariance")
        return SimpleNamespace(
            covariance_valid=True, lifetime_std=0.1,
            condition_number=50.0, boundary_hit=False, failure_reason=None,
        )

    def fake_bootstrap(**kwargs):
        assert kwargs["fit_result"].lifetime == 1.5
        np.testing.assert_array_equal(kwargs["irf"], condition.assumed_irf.kernel)
        seen.append("bootstrap")
        return SimpleNamespace(
            bootstrap_valid=True, bootstrap_median_ns=1.5, bootstrap_std_ns=0.11,
            lower_ns=1.3, upper_ns=1.7, n_resamples=3, n_successful_fits=3,
            fit_failure_rate=0.0, failure_reason=None,
        )

    def fake_bayesian(measured, **kwargs):
        np.testing.assert_array_equal(measured.require_raw_counts(), observed)
        assert measured is measurement
        assert kwargs["prepared_irf"] is condition.assumed_irf
        assert kwargs["sampling_config"].random_seed == seeds.bayesian
        seen.append("bayesian")
        return _fake_successful_bayesian_inference()

    monkeypatch.setattr(evaluation, "fit_single_reconvolution_curve", fake_classical)
    monkeypatch.setattr(evaluation, "estimate_poisson_reconvolution_local_covariance", fake_covariance)
    monkeypatch.setattr(evaluation, "estimate_parametric_poisson_bootstrap", fake_bootstrap)
    monkeypatch.setattr(evaluation, "fit_bayesian_monoexponential_reconvolution", fake_bayesian)
    record = evaluate_issue4_realization(
        condition, config, measurement, realization_index=0,
        true_reconvolution_amplitude=amplitude, seeds=seeds,
    )
    assert seen == ["classical", "covariance", "bootstrap", "bayesian"]
    assert record.observed_total_counts == int(np.sum(observed))
    assert record.bayesian.valid_interval
    assert "true_lifetime_ns" not in measurement.metadata
    assert "true_lifetime_ns" not in measurement.provenance
    assert not hasattr(measurement, "true_lifetime_ns")


def test_processed_intensity_cannot_enter_paired_poisson_path() -> None:
    condition, config = _condition_and_config()
    measurement = TCSPCMeasurement(
        condition.time_ns, np.ones(condition.time_ns.size),
        MeasurementDataKind.PROCESSED_INTENSITY,
    )
    with pytest.raises(InvalidMeasurementError, match="raw photon counts"):
        evaluate_issue4_realization(
            condition, config, measurement, realization_index=0,
            true_reconvolution_amplitude=10.0,
            seeds=issue4_realization_seeds(config, condition.condition_id, 0),
        )


def test_invalid_classical_fit_preserves_bayesian_and_failure_accounting(monkeypatch) -> None:
    condition, config = _condition_and_config()
    expected, amplitude = build_issue4_expected_counts(condition)
    seeds = issue4_realization_seeds(config, condition.condition_id, 0)
    measurement = sample_issue4_observation(
        condition, expected, seeds.observation, 0,
    )
    observed = measurement.require_raw_counts().copy()
    called = []

    def invalid_classical(**kwargs):
        np.testing.assert_array_equal(kwargs["counts"], observed)
        called.append("classical")
        return ReconvolutionCurveResult(
            30.0, 1.5, 1.0, 0.0, 30.0, 1.5, 1.0, 0.04,
            True, False, False, -100.0, 2.0, 1.0,
            "poisson_numerical_validation_failed", None,
            numerical_validation_passed=False,
            max_coordinate_descent_nll=0.4,
            recovery_attempted=True,
        )

    def unexpected_uncertainty(**kwargs):
        pytest.fail("invalid classical fit must not enter covariance or bootstrap")

    def fake_bayesian(measured, **kwargs):
        assert measured is measurement
        np.testing.assert_array_equal(measured.require_raw_counts(), observed)
        assert kwargs["sampling_config"].random_seed == seeds.bayesian
        called.append("bayesian")
        return _fake_successful_bayesian_inference()

    monkeypatch.setattr(evaluation, "fit_single_reconvolution_curve", invalid_classical)
    monkeypatch.setattr(
        evaluation, "estimate_poisson_reconvolution_local_covariance",
        unexpected_uncertainty,
    )
    monkeypatch.setattr(
        evaluation, "estimate_parametric_poisson_bootstrap",
        unexpected_uncertainty,
    )
    monkeypatch.setattr(
        evaluation, "fit_bayesian_monoexponential_reconvolution", fake_bayesian,
    )
    record = evaluate_issue4_realization(
        condition, config, measurement, realization_index=0,
        true_reconvolution_amplitude=amplitude, seeds=seeds,
    )
    assert called == ["classical", "bayesian"]
    assert record.classical_fit.optimizer_success
    assert not record.classical_fit.valid_fit
    assert record.classical_fit.recovery_attempted
    assert not record.covariance.valid_interval
    assert record.covariance.failure_reason == "classical_fit_invalid"
    assert not record.bootstrap.valid_interval
    assert record.bootstrap.n_requested == config.n_bootstrap_resamples
    assert record.bootstrap.n_valid_refits == 0
    assert record.bootstrap.failure_reason == "classical_fit_invalid"
    assert math.isnan(record.bootstrap.refit_failure_rate)
    assert all(math.isnan(value) for value in (
        record.bootstrap.lifetime_median_ns,
        record.bootstrap.lifetime_std_ns,
        record.bootstrap.percentile_lower_ns,
        record.bootstrap.percentile_upper_ns,
    ))
    assert record.bayesian.valid_interval
    summary = summarize_issue4_condition((record,))
    assert summary.n_valid_classical_fits == 0
    assert summary.classical_fit_failure_rate == 1.0
    assert [(entry.n_attempted, entry.n_valid, entry.failure_rate)
            for entry in summary.intervals] == [
                (1, 0, 1.0), (1, 0, 1.0), (1, 1, 0.0),
            ]

    payload = asdict(record)
    serialized = json.dumps(
        payload,
        default=lambda value: value.value if isinstance(value, Enum) else value,
    )
    assert "classical_fit_invalid" in serialized


def test_coverage_failure_denominators_and_widths_are_explicit() -> None:
    records = (
        _toy_record(0, classical_lifetime=1.9),
        _toy_record(1, classical_lifetime=2.1, covariance_valid=False,
                    bootstrap_valid=False,
                    bayesian_status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING),
        _toy_record(2, classical_lifetime=2.0,
                    covariance_bounds=(2.1, 2.3), bootstrap_bounds=(2.1, 2.3),
                    bayesian_bounds=(2.1, 2.3)),
    )
    summary = summarize_issue4_condition(records)
    for interval in summary.intervals:
        assert (interval.n_attempted, interval.n_valid, interval.n_covering) == (3, 2, 1)
        assert interval.coverage_among_valid == 0.5
        assert interval.binomial_standard_error_among_valid == pytest.approx(math.sqrt(0.125))
        assert interval.wilson_95_lower < 0.5 < interval.wilson_95_upper
        assert interval.failure_rate == pytest.approx(1 / 3)
        assert interval.successful_and_covering_fraction == pytest.approx(1 / 3)
        assert interval.mean_width_ns == pytest.approx(0.2)
        assert interval.median_width_ns == pytest.approx(0.2)
    assert summary.n_valid_classical_fits == 3
    assert summary.classical_fit_failure_rate == 0.0
    assert summary.mean_bootstrap_refit_failure_rate == pytest.approx(1 / 3)
    assert summary.estimator_spreads[0].n_valid == 3
    assert summary.estimator_spreads[1].n_valid == 2
    assert summary.estimator_spreads[0].empirical_std_ns == pytest.approx(0.1)
    assert dict(summary.mean_reported_lifetime_std_ns)["covariance"] == pytest.approx(0.1)
    assert dict(summary.mean_reported_lifetime_std_ns)["bayesian"] == pytest.approx(0.15)
    assert summary.estimator_spreads[0].empirical_std_ns != dict(
        summary.mean_reported_lifetime_std_ns
    )["bayesian"]
    assert [item.empirical_estimator for item in summary.uncertainty_scale_comparisons] == [
        "classical_fit", "classical_fit", "bayesian_posterior_median"
    ]


def test_all_covered_small_sample_still_has_nontrivial_binomial_interval() -> None:
    coverage = summarize_issue4_condition((_toy_record(0), _toy_record(1))).intervals[0]
    assert coverage.coverage_among_valid == 1.0
    assert coverage.wilson_95_lower < 0.5
    assert coverage.wilson_95_upper == pytest.approx(1.0)


def test_runtime_aggregation_uses_named_comparable_scopes() -> None:
    first = _toy_record(0)
    second = replace(
        _toy_record(1),
        classical_fit=replace(first.classical_fit, call_seconds=2.1),
        bootstrap=replace(first.bootstrap, runtime_seconds=0.8),
        bayesian=replace(first.bayesian, sampling_seconds=1.6),
    )
    runtimes = {entry.metric: entry for entry in summarize_issue4_condition((first, second)).runtimes}
    assert runtimes["classical_fit_call"].median_seconds == pytest.approx(1.6)
    assert runtimes["parametric_bootstrap"].median_seconds == pytest.approx(0.6)
    assert runtimes["bayesian_sampling"].median_seconds == pytest.approx(1.2)
    assert runtimes["bayesian_sampling"].q25_seconds == pytest.approx(1.0)
    assert runtimes["bayesian_sampling"].q75_seconds == pytest.approx(1.4)


def test_seed_streams_and_observations_are_stable_independent_of_prior_policy() -> None:
    condition, config = _condition_and_config()
    changed = replace(config, prior_policy_id="broader_lifetime")
    a = issue4_realization_seeds(config, condition.condition_id, 1)
    b = issue4_realization_seeds(config, condition.condition_id, 1)
    c = issue4_realization_seeds(changed, condition.condition_id, 1)
    assert a == b
    assert a.observation == c.observation
    assert a.bootstrap == c.bootstrap
    assert a.bayesian != c.bayesian
    assert len({a.observation, a.bootstrap, a.bayesian, a.posterior_predictive}) == 4
    expected, _ = build_issue4_expected_counts(condition)
    first = sample_issue4_observation(condition, expected, a.observation, 1)
    again = sample_issue4_observation(condition, expected, c.observation, 1)
    np.testing.assert_array_equal(first.require_raw_counts(), again.require_raw_counts())
    assert derive_issue4_seed(config.random_seed, condition.condition_id, 2, "observation") != a.observation


def test_stage5_rejects_irf_mismatch_and_nonpaired_shift_policies() -> None:
    condition, config = _condition_and_config()
    time = condition.time_ns
    wrong = prepare_irf(
        generate_gaussian_irf_profile(time, gaussian_centre_ns=0.7, gaussian_fwhm_ns=0.40),
        time,
    )
    with pytest.raises(ValueError, match="identical generating and assumed"):
        replace(condition, assumed_irf=wrong)
    with pytest.raises(ValueError, match="shift bounds"):
        replace(config, temporal_shift_bounds_ns=(-0.1, 0.1))
    fixed_priors = replace(config.priors, temporal_shift_prior=None, fixed_temporal_shift_ns=0.0)
    with pytest.raises(ValueError, match="free-shift"):
        replace(config, priors=fixed_priors)


def test_manifest_profiles_priors_and_frozen_suite_independence() -> None:
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs" /
                           "issue4_bayesian_workflow.json").read_text(encoding="utf-8"))
    assert manifest["profiles"]["smoke"]["n_repeats_per_condition"] < (
        manifest["profiles"]["scientific"]["n_repeats_per_condition"]
    )
    assert manifest["profiles"]["scientific"]["condition_ids"] == "all"
    assert len(manifest["conditions"]) == 8
    assert set(manifest["prior_policies"]) == {
        "baseline", "broader_lifetime", "narrower_lifetime"
    }
    assert all(not item["id"].startswith("Test ") for item in manifest["conditions"])
    assert "generalization" not in Path(evaluation.__file__).read_text(encoding="utf-8")
