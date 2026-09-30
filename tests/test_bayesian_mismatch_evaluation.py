"""Stage-6 mismatch plumbing; the scientific profile is not a pytest workload."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tcspc_toolkit.bayesian_mismatch_evaluation as mismatch
from scripts.run_issue4_bayesian_mismatch_benchmarks import _json_safe
from scripts.correct_issue4_mismatch_references import _comparison_from_saved_row
from tcspc_toolkit.bayesian_sampling import BayesianSamplingStatus


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "issue4_bayesian_mismatch_workflow.json"


@pytest.fixture(scope="module")
def workflow():
    return mismatch.build_mismatch_workflow(json.loads(MANIFEST.read_text()), "smoke")


@pytest.fixture(scope="module")
def references(workflow):
    return {
        (condition.condition_id, assumption.assumption_id):
            mismatch.construct_pseudo_true_reference(
                condition, assumption,
                mismatch.build_mismatch_expected_counts(condition).expected_counts,
                workflow.config.temporal_shift_bounds_ns,
            )
        for condition in workflow.conditions
        for assumption in condition.assumptions
    }


@pytest.fixture(scope="module")
def emg_pair(workflow, references):
    condition = workflow.conditions[-1]
    generating = mismatch.build_mismatch_expected_counts(condition)
    seeds = mismatch.mismatch_seeds(workflow.config, condition.condition_id, 0)
    measurement = mismatch.sample_mismatch_observation(condition, generating, seeds, 0)
    records = tuple(
        mismatch.evaluate_mismatch_inference(
            condition, assumption, workflow.config, measurement, generating,
            references[condition.condition_id, assumption.assumption_id], seeds, 0,
        )
        for assumption in condition.assumptions
    )
    return condition, generating, seeds, measurement, records


@pytest.mark.parametrize("name,fraction", [
    ("weak_biexponential", 0.05), ("moderate_biexponential", 0.15),
])
def test_biexponential_fraction_is_detected_finite_window_signal(workflow, name, fraction):
    condition = next(item for item in workflow.conditions if item.condition_id == name)
    generating = mismatch.build_mismatch_expected_counts(condition)
    assert generating.primary_expected_signal.sum() == pytest.approx(100000 * (1 - fraction))
    assert generating.secondary_expected_signal.sum() == pytest.approx(100000 * fraction)
    assert generating.expected_counts.sum() == pytest.approx(100000 + 0.5 * condition.time_ns.size)
    assert condition.mono_generating_lifetime_ns is None
    assert not hasattr(condition, "true_lifetime_ns")
    seeds = mismatch.mismatch_seeds(workflow.config, name, 0)
    observation = mismatch.sample_mismatch_observation(condition, generating, seeds, 0)
    assert "lifetime_true_ns" not in observation.metadata
    assert "lifetime_true_ns" not in observation.provenance
    assert observation.sample_id.startswith("issue4-stage6-")


def test_pseudo_true_prior_free_control_and_biexponential_projection(workflow, references):
    assert references["mono_control", "gaussian_mono"].lifetime_ns == pytest.approx(2.0, abs=1e-7)
    assert references["emg_irf", "matched_emg"].lifetime_ns == pytest.approx(2.0, abs=1e-7)
    # Independent Poisson-deviance projections of the committed Stage-6 design.
    assert references["weak_biexponential", "gaussian_mono"].lifetime_ns == pytest.approx(
        2.07392193, abs=2e-6
    )
    assert references["moderate_biexponential", "gaussian_mono"].lifetime_ns == pytest.approx(
        2.22702256, abs=2e-6
    )
    assert references["emg_irf", "gaussian_assumed"].lifetime_ns == pytest.approx(
        2.02495716, abs=2e-6
    )
    for reference in references.values():
        assert reference.n_starts == 4
        assert math.isfinite(reference.poisson_nll)
        assert reference.max_start_objective_gap <= 1e-6
        assert reference.max_start_lifetime_gap_ns <= 1e-5
        assert reference.second_best_objective_gap <= 1e-6
        assert 0.0 <= reference.maximum_coordinate_descent_nll <= 1e-5
        assert reference.maximum_scaled_gradient <= 1e-3
        assert reference.independent_objective_gap <= 1e-6
        assert reference.independent_lifetime_gap_ns <= 1e-5
        assert reference.active_bounds == ()
    condition = workflow.conditions[1]
    expected = mismatch.build_mismatch_expected_counts(condition).expected_counts
    repeated = mismatch.construct_pseudo_true_reference(
        condition, condition.assumptions[0], expected,
        workflow.config.temporal_shift_bounds_ns,
    )
    assert repeated == references["weak_biexponential", "gaussian_mono"]


def test_physical_and_pseudo_true_decomposition(workflow, references):
    condition = workflow.conditions[2]
    pseudo = references[condition.condition_id, condition.assumptions[0].assumption_id].lifetime_ns
    estimate = 2.26
    assert estimate - condition.primary_lifetime_ns == pytest.approx(
        (estimate - pseudo) + (pseudo - condition.primary_lifetime_ns)
    )
    assert pseudo != condition.primary_lifetime_ns


def test_pseudo_true_fails_explicitly_when_numerical_validation_fails(workflow, monkeypatch):
    condition = workflow.conditions[1]
    generating = mismatch.build_mismatch_expected_counts(condition)
    monkeypatch.setattr(mismatch, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(success=False, message="local descent"))
    with pytest.raises(RuntimeError, match="failed numerical validation"):
        mismatch.construct_pseudo_true_reference(
            condition, condition.assumptions[0], generating.expected_counts,
            workflow.config.temporal_shift_bounds_ns,
        )


def test_reference_strict_probe_rejects_historical_biexponential_descent(workflow):
    # The old reference accepted this background-direction descent because its
    # routine-fit threshold was 0.01 NLL at a single 1e-3 scaled probe.
    condition = next(item for item in workflow.conditions if item.condition_id == "weak_biexponential")
    assumed = condition.assumptions[0]
    expected = mismatch.build_mismatch_expected_counts(condition).expected_counts
    from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
    from tcspc_toolkit.fitting import poisson_negative_log_likelihood

    historical = dict(amplitude=2396.745151866371, lifetime=2.074616201952051,
                      background=0.507794164681629, temporal_shift=0.0383308543116744)
    at_old = poisson_negative_log_likelihood(expected,
        monoexponential_reconvolution_expected_counts(
            condition.time_ns, assumed.prepared_irf.kernel, **historical))
    historical["background"] += 0.01
    at_perturbation = poisson_negative_log_likelihood(expected,
        monoexponential_reconvolution_expected_counts(
            condition.time_ns, assumed.prepared_irf.kernel, **historical))
    assert at_old - at_perturbation > 0.02
    def historical_objective(parameters):
        return poisson_negative_log_likelihood(expected,
            monoexponential_reconvolution_expected_counts(
                condition.time_ns, assumed.prepared_irf.kernel,
                amplitude=parameters[0], lifetime=parameters[1],
                background=parameters[2], temporal_shift=parameters[3]))
    old_coordinates = np.array([2396.745151866371, 2.074616201952051,
                                0.507794164681629, 0.0383308543116744])
    bounds = ((0, math.inf), (1e-12, math.inf), (1e-12, math.inf),
              workflow.config.temporal_shift_bounds_ns)
    assert mismatch._reference_coordinate_descent(
        historical_objective, old_coordinates, bounds
    ) > 0.02


def test_saved_row_reference_reanalysis_uses_saved_intervals_without_refits():
    row = {
        "classical_fit_valid_fit": "True", "classical_fit_lifetime_ns": "2.1",
        "covariance_valid_interval": "True", "covariance_lifetime_std_ns": "0.02",
        "covariance_local_gaussian_lower_ns": "2.07",
        "covariance_local_gaussian_upper_ns": "2.13",
        "bootstrap_valid_interval": "True", "bootstrap_lifetime_std_ns": "0.021",
        "bootstrap_percentile_lower_ns": "2.06", "bootstrap_percentile_upper_ns": "2.14",
        "bayesian_status": "insufficient_sampling", "bayesian_diagnostics_accepted": "False",
        "bayesian_lifetime_median_ns": "2.11", "bayesian_posterior_lifetime_std_ns": "0.02",
        "bayesian_credible_lower_ns": "2.08", "bayesian_credible_upper_ns": "2.14",
    }
    old = _comparison_from_saved_row(row, "covariance", 2.06)
    corrected = _comparison_from_saved_row(row, "covariance", 2.08)
    assert old.reference_included is False
    assert corrected.reference_included is True
    assert corrected.deviation_ns == pytest.approx(0.02)
    assert corrected.deviation_to_reported_std == pytest.approx(1.0)
    assert _comparison_from_saved_row(row, "bootstrap", 2.08).reference_included is True
    assert _comparison_from_saved_row(row, "bayesian", 2.08).reference_included is None


def test_workflow_passes_one_measurement_object_to_both_irf_assumptions(
    workflow, monkeypatch,
):
    pair_workflow = mismatch.MismatchWorkflow(
        conditions=(workflow.conditions[-1],),
        config=replace(workflow.config, n_repeats_per_condition=1),
    )
    received = []

    def capture(condition, assumption, config, measurement, generating, reference, seeds, index):
        received.append((assumption.assumption_id, measurement, seeds))
        return object()

    monkeypatch.setattr(mismatch, "evaluate_mismatch_inference", capture)
    monkeypatch.setattr(mismatch, "summarize_mismatch_records", lambda records: ())
    report = mismatch.evaluate_mismatch_workflow(pair_workflow)
    assert len(report.per_inference) == 2
    assert received[0][1] is received[1][1]
    assert received[0][2] == received[1][2]
    assert received[0][0] != received[1][0]


def test_irf_pair_same_counts_and_all_shared_settings_except_irf(emg_pair, workflow):
    condition, generating, seeds, measurement, (matched, wrong) = emg_pair
    assert matched.sample_id == wrong.sample_id == measurement.sample_id
    assert matched.observed_counts_sha256 == wrong.observed_counts_sha256
    assert matched.expected_counts_sha256 == wrong.expected_counts_sha256
    assert matched.seeds == wrong.seeds == seeds
    assert matched.time_grid_sha256 == wrong.time_grid_sha256
    assert matched.generating_irf_sha256 == wrong.generating_irf_sha256
    assert matched.assumed_irf_sha256 == matched.generating_irf_sha256
    assert wrong.assumed_irf_sha256 != wrong.generating_irf_sha256
    assert matched.assumed_irf_sha256 != wrong.assumed_irf_sha256
    assert matched.assumed_decay_model == wrong.assumed_decay_model == "monoexponential"
    assert matched.primary_lifetime_ns == wrong.primary_lifetime_ns == 2.0
    assert matched.background_per_bin == wrong.background_per_bin == 0.5
    assert matched.true_temporal_shift_ns == wrong.true_temporal_shift_ns == 0.04
    assert condition.assumptions[0].prepared_irf is condition.generating_irf


def test_irf_pair_predictive_discrepancies_and_serialization(emg_pair):
    _, _, _, _, records = emg_pair
    for record in records:
        assert record.classical_fit.valid_fit
        assert record.bayesian.status is BayesianSamplingStatus.SUCCESS
        assert record.posterior_predictive.status == "success"
        names = {name for name, *_ in record.posterior_predictive.discrepancy_summaries}
        assert names == {
            "poisson_deviance", "residual_rms", "maximum_absolute_residual",
            "total_counts", "peak_counts", "peak_time_ns",
            "early_window_counts", "tail_window_counts",
        }
        assert len(record.posterior_predictive.mean_signed_deviance_residual_profile) == 240
        assert all(0 <= probability <= 1 for _, _, _, probability in
                   record.posterior_predictive.discrepancy_summaries)
        serial = _json_safe(asdict(record))
        json.dumps(serial, allow_nan=False)
        assert "model_valid" not in serial and "mismatch_detected" not in serial


def test_invalid_classical_fit_keeps_bayesian_attempt_and_failure_denominators(
    monkeypatch, emg_pair, workflow, references,
):
    condition, generating, seeds, measurement, records = emg_pair
    fitted = SimpleNamespace(
        optimizer_success=True, valid_fit=False, fitted_lifetime_ns=math.nan,
        fitted_amplitude=math.nan, fitted_background=math.nan,
        fitted_temporal_shift_ns=math.nan, boundary_hit=False,
        failure_reason="optimizer_validation_failed", runtime_ms=1.0,
        numerical_validation_passed=False, max_coordinate_descent_nll=0.3,
        recovery_attempted=True, optimizer_status=0,
        optimizer_message="premature convergence", optimizer_nfev=5, optimizer_njev=5,
    )
    attempted = []
    monkeypatch.setattr(mismatch, "fit_single_reconvolution_curve", lambda **kwargs: fitted)
    monkeypatch.setattr(mismatch, "fit_bayesian_monoexponential_reconvolution",
                        lambda *args, **kwargs: attempted.append(args[0]) or object())
    monkeypatch.setattr(mismatch, "_compact_bayesian", lambda run, seconds: replace(
        records[0].bayesian, status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
        diagnostics_accepted=False, diagnostic_failure_reasons=("short_chain",),
    ))
    monkeypatch.setattr(mismatch, "sample_bayesian_posterior_predictive",
                        lambda *args, **kwargs: pytest.fail("PPC must not accept failed sampling"))
    failed = mismatch.evaluate_mismatch_inference(
        condition, condition.assumptions[0], workflow.config, measurement,
        generating, references[condition.condition_id, condition.assumptions[0].assumption_id],
        seeds, 1,
    )
    assert attempted == [measurement]
    assert failed.classical_fit.numerical_validation_passed is False
    assert failed.classical_fit.recovery_attempted
    assert not failed.covariance.valid_interval
    assert failed.covariance.failure_reason == "classical_fit_invalid"
    assert not failed.bootstrap.valid_interval
    assert failed.bootstrap.n_requested == workflow.config.n_bootstrap_resamples
    assert failed.bootstrap.n_valid_refits == 0
    assert failed.posterior_predictive.status == "not_available"
    assert failed.bayesian.status is BayesianSamplingStatus.INSUFFICIENT_SAMPLING
    assert all(c.reference_included is None for c in failed.comparisons)
    summaries = mismatch.summarize_mismatch_records((records[0], failed))
    assert all(item.n_attempted == 2 for item in summaries)
    assert all(item.n_valid_intervals == 1 for item in summaries)
    json.dumps(_json_safe(asdict(failed)), allow_nan=False)


def test_seed_stream_independence_pair_sharing_and_frozen_profile_separation(workflow):
    seeds = mismatch.mismatch_seeds(workflow.config, "emg_irf", 0)
    repeated = mismatch.mismatch_seeds(workflow.config, "emg_irf", 0)
    other = mismatch.mismatch_seeds(workflow.config, "emg_irf", 1)
    assert seeds == repeated
    assert len(set(asdict(seeds).values())) == 4
    assert set(asdict(seeds).values()).isdisjoint(asdict(other).values())
    stage5 = json.loads((ROOT / "configs" / "issue4_bayesian_workflow.json").read_text())
    stage6 = json.loads(MANIFEST.read_text())
    assert workflow.config.seed_namespace == "issue4:stage6:v1"
    assert stage6["randomness"]["base_seed"] != stage5["randomness"]["base_seed"]
    assert stage6["profiles"]["scientific"]["n_repeats_per_condition"] == 20
    assert stage6["profiles"]["scientific"]["n_bootstrap_resamples"] == 100
    assert stage6["profiles"]["scientific"]["n_posterior_predictive_draws"] == 200
    assert (stage6["profiles"]["scientific"]["bayesian_sampling"] ==
            stage5["profiles"]["scientific"]["bayesian_sampling"])
    assert stage6["priors"] == stage5["prior_policies"]["baseline"]
    assert not any("week8" in condition.condition_id or "test_f" in condition.condition_id
                   for condition in workflow.conditions)
