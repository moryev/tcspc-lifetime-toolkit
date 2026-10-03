"""Issue-9 Stage-5 Bayesian result and posterior-predictive persistence."""

from __future__ import annotations

import builtins
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit import persistence as store
from tcspc_toolkit.bayesian import (
    BayesianModelContext, BayesianPriorConfig, BoundedUniformPrior, GammaPrior,
    IRFModelRelation, LogNormalPrior,
)
from tcspc_toolkit.bayesian_evaluation import (
    BayesianClassicalRealizationResult, Issue4Bayesian, Issue4RealizationSeeds,
)
from tcspc_toolkit.bayesian_mismatch_evaluation import (
    MismatchInferenceRecord, MismatchPredictiveSummary, MismatchSeeds,
)
from tcspc_toolkit.bayesian_predictive import (
    BayesianPosteriorPredictiveResult,
    calculate_bayesian_posterior_predictive_diagnostics, summarize_predictive_band,
)
from tcspc_toolkit.bayesian_sampling import (
    BayesianInferenceRun, BayesianModelContextSummary, BayesianParameterSummary,
    BayesianPosteriorSamples, BayesianReconvolutionResult, BayesianRuntime,
    BayesianSamplingConfig, BayesianSamplingDiagnostics, BayesianSamplingStatus,
)
from tcspc_toolkit.irf import (
    IRFSourceKind, generate_emg_irf_profile, generate_gaussian_irf_profile,
)
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement


_SEED = 2**64 - 1


def _priors(*, shift_bounds=(-0.2, 0.2), fixed_shift=None, lifetime_log_mean=0.0):
    return BayesianPriorConfig(
        amplitude=GammaPrior(2.0, 0.01),
        lifetime_ns=LogNormalPrior(lifetime_log_mean, 0.5),
        background_per_bin=GammaPrior(2.0, 1.0),
        temporal_shift_prior=(BoundedUniformPrior(*shift_bounds)
                              if fixed_shift is None else None),
        fixed_temporal_shift_ns=fixed_shift,
    )


def _sampler(seed=_SEED):
    return BayesianSamplingConfig(
        random_seed=seed, n_walkers=10, warmup_steps=8,
        production_steps=20, max_production_steps=40, extension_steps=10,
        n_ensembles=2, credible_interval_level=0.9,
    )


@pytest.fixture
def context():
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as db:
        store.initialize_database(db)
        db.row_factory = sqlite3.Row
        run_id = store.record_run(
            db, run_key="bayes-run", run_type="evaluation", status="complete",
            recorded_at_utc="2026-10-02T10:00:00+00:00", origin="new",
            package_version="0.7.0", configuration={"stage": 5},
        )
        time = np.arange(0.0, 2.25, 0.25)
        profile = generate_gaussian_irf_profile(
            time, gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4,
        )
        prepared = prepare_irf(profile, time)
        prepared_id = store.record_prepared_irf(
            db, prepared, source_key="bayes-source", preparation_key="bayes-prepared",
        )
        condition = store.record_simulation_condition(
            db, condition_key="bayes-condition", condition_id="local-condition",
            generating_model="monoexponential", mono_lifetime_ns=1.8,
            signal_photon_count=1000, background_per_bin=0.2,
            true_temporal_shift_ns=0.0, generating_irf_id=prepared_id,
        )
        measurement = TCSPCMeasurement(
            time_ns=time, values=np.arange(1, 10, dtype=np.int64),
            data_kind=MeasurementDataKind.RAW_COUNTS, sample_id="bayes-sample",
        )
        measurement_id = store.record_measurement(
            db, measurement, measurement_key="bayes-measurement",
            source_type="synthetic", condition_pk=condition,
        )
        store.link_run_measurement(
            db, run_id=run_id, measurement_id=measurement_id, data_role="evaluation",
        )
        priors, sampler = _priors(), _sampler()
        model_id = store.record_model_version(
            db, model_key="bayes-model", estimator_name="emcee_poisson",
            family="bayesian", configuration=store.bayesian_model_configuration(priors, sampler),
            prior_policy_id="baseline",
        )
        assumption_id = store.record_model_assumption(
            db, assumption_key="bayes-assumption", source_assumption_id="gaussian_mono",
            assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
            background_convention="per_bin", context_completeness="complete",
            prepared_irf_id=prepared_id, temporal_shift_lower_ns=-0.2,
            temporal_shift_upper_ns=0.2, bayesian_priors=priors,
        )
        yield db, run_id, measurement_id, model_id, assumption_id, measurement, prepared, priors, sampler, condition


def _source(priors=None, sampler=None, *, status=BayesianSamplingStatus.SUCCESS,
            failures=(), median=1.8, shift=0.05, tau=2.0, ess=200.0,
            correlation=-0.2):
    priors = _priors() if priors is None else priors
    sampler = _sampler() if sampler is None else sampler
    names = ("amplitude", "lifetime_ns", "background_per_bin") + (
        ("temporal_shift_ns",) if priors.infer_temporal_shift else ()
    )
    units = ("scale", "ns", "counts_per_bin") + (("ns",) if priors.infer_temporal_shift else ())
    values = (50.0, median, 1.0) + ((shift,) if priors.infer_temporal_shift else ())
    summaries = tuple(BayesianParameterSummary(
        name, unit, value + 0.05, value, 0.1,
        value - (0.1 if name == "temporal_shift_ns" else 0.2),
        value + (0.1 if name == "temporal_shift_ns" else 0.2),
    ) for name, unit, value in zip(names, units, values))
    matrix = np.eye(len(names))
    matrix[1, 2] = matrix[2, 1] = correlation
    if priors.infer_temporal_shift:
        matrix[1, 3] = matrix[3, 1] = 0.1
    diagnostics = BayesianSamplingDiagnostics(
        production_steps=20, retained_samples=400, extension_count=0,
        acceptance_fraction=np.full((2, 10), 0.4),
        mean_acceptance_fraction=0.4,
        autocorrelation_time_steps=np.full((2, len(names)), tau),
        autocorrelation_relative_change=np.full((2, len(names)), 0.05),
        approximate_effective_samples=np.full(len(names), ess),
        maximum_ensemble_mean_difference_sd=0.1,
        maximum_ensemble_median_difference_sd=0.1,
        failure_reasons=tuple(failures),
    )
    return BayesianReconvolutionResult(
        status=status, parameter_summaries=summaries,
        inferred_parameter_names=names, inferred_parameter_units=units,
        correlation_matrix=matrix, diagnostics=diagnostics,
        runtime=BayesianRuntime(0.1, 0.2, 0.1, 0.4),
        priors=priors, sampling_config=sampler,
        model_context=BayesianModelContextSummary(
            sample_id="bayes-sample", irf_selection="explicit_prepared",
            irf_source_kind=IRFSourceKind.SYNTHETIC_GAUSSIAN,
            irf_model_relation=IRFModelRelation.MATCHED,
        ),
    )


def _write(context, source=None, **changes):
    db, run_id, measurement_id, model_id, assumption_id, *_ = context
    return store.record_bayesian_result(
        db, _source() if source is None else source,
        run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id, **changes,
    )


def _ppc(context):
    _, _, _, _, _, measurement, prepared, priors, sampler, _ = context
    scientific_context = BayesianModelContext(
        measurement, prepared_irf=prepared, irf_model_relation=IRFModelRelation.MATCHED,
    )
    expected = np.full((3, measurement.time_ns.size), 5.0)
    replicated = np.tile(np.arange(1, measurement.time_ns.size + 1), (3, 1))
    diagnostics = calculate_bayesian_posterior_predictive_diagnostics(
        scientific_context, expected, replicated,
        early_window_ns=(0.0, 1.0), tail_window_ns=(1.0, 2.0),
    )
    return BayesianPosteriorPredictiveResult(
        selected_posterior_indices=np.zeros((3, 3), dtype=np.int64),
        selection_with_replacement=False, retained_posterior_sample_count=400,
        physical_parameter_draws=np.tile([50.0, 1.8, 1.0, 0.05], (3, 1)),
        expected_counts=expected, replicated_counts=replicated,
        posterior_expected_count_band=summarize_predictive_band(expected, interval_level=0.9),
        posterior_predictive_count_band=summarize_predictive_band(replicated, interval_level=0.9),
        diagnostics=diagnostics, model_context=scientific_context,
        priors=priors, sampling_config=sampler,
        inference_sampling_status=BayesianSamplingStatus.SUCCESS,
        random_seed=_SEED - 1, interval_level=0.9, runtime_seconds=0.25,
    )


def _compact():
    return Issue4Bayesian(
        status=BayesianSamplingStatus.SUCCESS, diagnostics_accepted=True,
        lifetime_mean_ns=1.85, lifetime_median_ns=1.8,
        posterior_lifetime_std_ns=0.1, credible_lower_ns=1.6,
        credible_upper_ns=2.0, mean_acceptance_fraction=0.4,
        minimum_effective_samples=200.0, production_steps=20,
        retained_samples=400, extension_count=0,
        minimum_autocorrelation_multiples=10.0,
        maximum_autocorrelation_relative_change=0.05,
        maximum_ensemble_mean_difference_sd=0.1,
        maximum_ensemble_median_difference_sd=0.1,
        lifetime_background_correlation=-0.2,
        lifetime_shift_correlation=0.1,
        diagnostic_failure_reasons=(), initialization_seconds=0.1,
        sampling_seconds=0.2, diagnostic_seconds=0.1,
        inference_total_seconds=0.4, call_seconds=0.45,
    )


def _matched_record(context):
    db, _, measurement_id, _, _, _, _, _, _, _ = context
    count_hash = db.execute(
        "SELECT values_sha256 FROM measurements WHERE measurement_id = ?", (measurement_id,),
    ).fetchone()[0]
    return BayesianClassicalRealizationResult(
        condition_id="local-condition", prior_policy_id="baseline", profile="smoke",
        realization_index=0, true_lifetime_ns=1.8, signal_photon_count=1000,
        true_reconvolution_amplitude=50.0, background_per_bin=0.2,
        true_temporal_shift_ns=0.0, observed_total_counts=45,
        observed_counts_sha256=count_hash,
        seeds=Issue4RealizationSeeds(1, 2, _SEED, _SEED - 1),
        classical_fit=None, covariance=None, bootstrap=None, bayesian=_compact(),
        orchestration_seconds=0.5,
    )


def _mismatch_record(context, *, assumption_label="gaussian_mono", predictive=None):
    db, _, measurement_id, _, _, _, _, _, _, _ = context
    count_hash = db.execute(
        "SELECT values_sha256 FROM measurements WHERE measurement_id = ?", (measurement_id,),
    ).fetchone()[0]
    if predictive is None:
        predictive = MismatchPredictiveSummary(
            status="success", reason=None, n_draws=3, runtime_seconds=0.25,
            discrepancy_summaries=(("poisson_deviance", 4.0, 5.0, 0.4),),
            mean_signed_deviance_residual_profile=(0.1,) * 9,
        )
    return MismatchInferenceRecord(
        condition_id="local-condition", mechanism="irf",
        generating_model="monoexponential", assumed_decay_model="monoexponential",
        assumption_id=assumption_label, assumed_irf_id="gaussian",
        generating_irf_id="gaussian", profile="smoke", realization_index=0,
        sample_id="bayes-sample", primary_lifetime_ns=1.8,
        secondary_lifetime_ns=None, secondary_detected_fraction=None,
        mono_generating_lifetime_ns=1.8,
        pseudo_true_mono_lifetime_ns=1.75, physical_model_discrepancy_ns=-0.05,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0, observed_total_counts=45,
        observed_counts_sha256=count_hash, expected_counts_sha256="a" * 64,
        time_grid_sha256="b" * 64, generating_irf_sha256="c" * 64,
        assumed_irf_sha256="d" * 64,
        seeds=MismatchSeeds(1, 2, _SEED, _SEED - 1),
        classical_fit=None, covariance=None, bootstrap=None,
        bayesian=_compact(), posterior_predictive=predictive,
        comparisons=(), orchestration_seconds=0.5,
    )


def _pseudo_reference(context, *, key="pseudo-v2", assumption_id=None,
                      condition_pk=None, version="stage6.5-corrected"):
    db, _, _, _, default_assumption, _, _, _, _, default_condition = context
    return store.record_pseudo_true_reference(
        db, reference_key=key, reference_version=version,
        condition_pk=default_condition if condition_pk is None else condition_pk,
        assumption_id=default_assumption if assumption_id is None else assumption_id,
        lifetime_ns=1.72, projection_temporal_shift_ns=0.05,
    )


def test_bayesian_point_summary_diagnostics_and_immutable_reuse(context):
    db, run_id, measurement_id, model_id, assumption_id, *_ = context
    source = _source()
    result_id = _write(context, source)
    point = db.execute(
        "SELECT * FROM estimator_results WHERE result_id = ?", (result_id,),
    ).fetchone()
    summary = db.execute(
        "SELECT * FROM bayesian_summaries WHERE result_id = ?", (result_id,),
    ).fetchone()
    assert point["point_summary"] == "posterior_median"
    assert point["lifetime_estimate_ns"] == source.parameter_summaries[1].median
    assert point["random_seed_decimal"] == str(_SEED)
    assert point["irf_model_relation"] == "matched"
    assert point["runtime_scope"] == "bayesian_inference_total"
    assert summary["lifetime_mean_ns"] == source.parameter_summaries[1].mean
    assert (summary["sampling_status"], summary["diagnostics_accepted"]) == ("success", 1)
    assert (summary["production_steps"], summary["retained_samples"],
            summary["extension_count"]) == (20, 400, 0)
    assert summary["minimum_effective_samples"] == 200.0
    assert summary["mean_acceptance_fraction"] == 0.4
    assert summary["sampling_seconds"] == 0.2
    parameter_json = json.loads(summary["parameter_summaries_json"])
    assert parameter_json["credible_probability"] == 0.9
    assert len(parameter_json["parameters"]) == 4
    assert parameter_json["sampling_counts"] == {
        "n_walkers": 10, "n_ensembles": 2,
        "requested_warmup_steps": 8, "requested_production_steps": 20,
        "actual_warmup_steps": None, "valid_retained_samples": None,
    }
    assert parameter_json["diagnostic_arrays"][
        "autocorrelation_time_steps_by_ensemble_parameter"
    ] == [[2.0] * 4] * 2
    pairs = json.loads(summary["correlation_json"])
    assert any(pair == {
        "first": "lifetime_ns", "second": "background_per_bin", "correlation": -0.2,
    } for pair in pairs)
    assert db.execute("SELECT COUNT(*) FROM uncertainty_results").fetchone()[0] == 0
    assert _write(context, source, on_duplicate="reuse_identical") == result_id
    with pytest.raises(store.PersistenceConflictError):
        _write(context, replace(source, parameter_summaries=tuple(
            replace(item, median=item.median + 0.1) if item.name == "lifetime_ns" else item
            for item in source.parameter_summaries
        )), on_duplicate="reuse_identical")
    with pytest.raises(store.PersistenceConflictError):
        _write(context, replace(source, diagnostics=replace(
            source.diagnostics, mean_acceptance_fraction=0.41,
        )), on_duplicate="reuse_identical")
    with pytest.raises(store.PersistenceConflictError, match="Bayesian summary"):
        _write(context, replace(source, parameter_summaries=tuple(
            replace(item, mean=item.mean + 0.01) if item.name == "lifetime_ns" else item
            for item in source.parameter_summaries
        )), on_duplicate="reuse_identical")
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM bayesian_summaries").fetchone()[0] == 1

    # One observation may have other prior, assumption, or explicit analysis identity.
    other_prior = _priors(lifetime_log_mean=0.2)
    other_model = store.record_model_version(
        db, model_key="other-prior", estimator_name="emcee_poisson",
        family="bayesian", configuration=store.bayesian_model_configuration(other_prior, _sampler()),
        prior_policy_id="broader",
    )
    second = store.record_bayesian_result(
        db, _source(other_prior), run_id=run_id, measurement_id=measurement_id,
        model_id=other_model, assumption_id=assumption_id,
    )
    assert second != result_id
    another_analysis = _write(context, source, analysis_key="repeat-2")
    assert another_analysis not in (result_id, second)
    another_assumption = store.record_model_assumption(
        db, assumption_key="alternate-assumption", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=db.execute(
            "SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
            (assumption_id,),
        ).fetchone()[0], temporal_shift_lower_ns=-0.2, temporal_shift_upper_ns=0.2,
    )
    assert store.record_bayesian_result(
        db, source, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=another_assumption,
    ) not in (result_id, second, another_analysis)
    assert "random_seed" not in json.loads(db.execute(
        "SELECT configuration_json FROM model_versions WHERE model_id = ?", (model_id,),
    ).fetchone()[0])["sampler"]


@pytest.mark.parametrize("status,failures,accepted", [
    (BayesianSamplingStatus.SUCCESS, (), True),
    (BayesianSamplingStatus.SUCCESS, ("tau_unstable",), False),
    (BayesianSamplingStatus.INSUFFICIENT_SAMPLING, ("too_short",), False),
    (BayesianSamplingStatus.NUMERICAL_FAILURE, ("numerical",), False),
    (BayesianSamplingStatus.INITIALIZATION_FAILED, ("initialization",), False),
])
def test_sampling_status_and_diagnostic_acceptance_are_distinct(
    context, status, failures, accepted,
):
    source = _source(status=status, failures=failures)
    if status is BayesianSamplingStatus.INITIALIZATION_FAILED:
        source = replace(source, parameter_summaries=(), diagnostics=replace(
            source.diagnostics, production_steps=0, retained_samples=0,
            acceptance_fraction=np.full((2, 10), math.nan), mean_acceptance_fraction=math.nan,
            autocorrelation_time_steps=np.full((2, 4), math.nan),
            approximate_effective_samples=np.full(4, math.nan),
        ))
    result_id = _write(context, source)
    db = context[0]
    point = db.execute("SELECT status, is_valid, lifetime_estimate_ns FROM estimator_results "
                       "WHERE result_id = ?", (result_id,)).fetchone()
    summary = db.execute("SELECT sampling_status, diagnostics_accepted FROM bayesian_summaries "
                         "WHERE result_id = ?", (result_id,)).fetchone()
    assert summary[:] == (status.value, int(accepted))
    assert point["is_valid"] == int(accepted)
    assert point["lifetime_estimate_ns"] == (
        None if status is BayesianSamplingStatus.INITIALIZATION_FAILED else 1.8
    )
    with pytest.raises(ValueError, match="accepted diagnostics"):
        _write(context, replace(source, diagnostics=replace(
            source.diagnostics, failure_reasons=(),
        )) if status is not BayesianSamplingStatus.SUCCESS else replace(
            source, status=BayesianSamplingStatus.NUMERICAL_FAILURE,
            diagnostics=replace(source.diagnostics, failure_reasons=()),
        ), analysis_key="contradiction")


@pytest.mark.parametrize("correlation,category", [
    (math.nan, "nan"), (math.inf, "+inf"), (-math.inf, "-inf"),
])
def test_nonfinite_diagnostics_and_correlations_keep_original_categories(context, correlation, category):
    source = _source(tau=math.nan, ess=math.nan, correlation=correlation,
                     status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
                     failures=("autocorrelation estimate unavailable",))
    result_id = _write(context, source)
    row = context[0].execute(
        "SELECT * FROM bayesian_summaries WHERE result_id = ?", (result_id,),
    ).fetchone()
    assert row["minimum_effective_samples"] is None
    assert row["minimum_autocorrelation_multiples"] is None
    assert row["lifetime_background_correlation"] is None
    nonfinite = json.loads(row["nonfinite_fields_json"])
    assert nonfinite["minimum_effective_samples"] == "nan"
    assert nonfinite["correlation.lifetime_ns.background_per_bin"] == category
    assert nonfinite["autocorrelation_time_steps[0,0]"] == "nan"
    assert json.loads(row["parameter_summaries_json"])["diagnostic_arrays"][
        "approximate_effective_samples_by_parameter"
    ] == [None] * 4
    with pytest.raises(ValueError, match="correlation"):
        _write(context, _source(correlation=1.5), analysis_key="bad-correlation")


def test_bayesian_model_measurement_irf_and_shift_context_checked_before_reuse(context):
    db, run_id, measurement_id, model_id, assumption_id, *_ = context
    source = _source()
    result_id = _write(context, source)
    # A changed scientific context cannot be hidden by the existing point key.
    db.execute("UPDATE model_assumptions SET temporal_shift_lower_ns = 0.1 "
               "WHERE assumption_id = ?", (assumption_id,))
    with pytest.raises(ValueError, match="prior exceeds|prior.*bounds|posterior median"):
        _write(context, source, on_duplicate="reuse_identical")
    db.execute("UPDATE model_assumptions SET temporal_shift_lower_ns = -0.2 "
               "WHERE assumption_id = ?", (assumption_id,))
    db.execute("UPDATE measurements SET data_kind = 'processed_intensity', "
               "observed_total_counts = NULL "
               "WHERE measurement_id = ?", (measurement_id,))
    with pytest.raises(ValueError, match="raw-count"):
        _write(context, source, on_duplicate="reuse_identical")
    db.execute("UPDATE measurements SET data_kind = 'raw_counts', "
               "observed_total_counts = 45 WHERE measurement_id = ?", (measurement_id,))
    with pytest.raises(ValueError, match="prior/sampler"):
        _write(context, _source(_priors(lifetime_log_mean=0.3)), on_duplicate="reuse_identical")
    with pytest.raises(ValueError, match="membership"):
        store.record_bayesian_result(
            db, source, run_id=run_id + 99, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1
    assert result_id > 0


def test_valid_shift_policy_and_failed_out_of_policy_summary(context):
    db, run_id, measurement_id, model_id, assumption_id, *_ = context
    with pytest.raises(ValueError, match="posterior .* shift"):
        _write(context, _source(shift=0.3))
    rejected = _source(shift=0.3, status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
                       failures=("too_short",))
    assert _write(context, rejected) > 0
    fixed_priors = _priors(fixed_shift=0.0)
    fixed_sampler = _sampler(seed=17)
    fixed_model = store.record_model_version(
        db, model_key="fixed-model", estimator_name="emcee_poisson", family="bayesian",
        configuration=store.bayesian_model_configuration(fixed_priors, fixed_sampler),
    )
    prepared_id = db.execute(
        "SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
        (assumption_id,),
    ).fetchone()[0]
    fixed_assumption = store.record_model_assumption(
        db, assumption_key="fixed-assumption", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=prepared_id,
        fixed_temporal_shift_ns=0.0, bayesian_priors=fixed_priors,
    )
    fixed_source = replace(_source(fixed_priors, fixed_sampler),
                           model_context=BayesianModelContextSummary(
                               "bayes-sample", "explicit_prepared",
                               IRFSourceKind.SYNTHETIC_GAUSSIAN, IRFModelRelation.MATCHED,
                           ))
    assert store.record_bayesian_result(
        db, fixed_source, run_id=run_id, measurement_id=measurement_id,
        model_id=fixed_model, assumption_id=fixed_assumption,
    ) > 0


def test_ppc_summary_artifacts_and_duplicate_payload(context, tmp_path):
    db, run_id, *_ = context
    posterior = store.record_artifact(
        db, artifact_key="chain", path=tmp_path / "chain.nc", path_base="absolute",
        artifact_kind="posterior_chain", format="netcdf", sha256="a" * 64,
        producing_run_id=run_id,
    )
    predictive = store.record_artifact(
        db, artifact_key="predictive", path=tmp_path / "draws.npz", path_base="absolute",
        artifact_kind="posterior_predictive_samples", format="npz", sha256="b" * 64,
        producing_run_id=run_id,
    )
    ppc = _ppc(context)
    result_id = _write(
        context, posterior_predictive=ppc, posterior_artifact_id=posterior,
        predictive_artifact_id=predictive,
    )
    row = db.execute("SELECT * FROM bayesian_summaries WHERE result_id = ?",
                     (result_id,)).fetchone()
    assert row["ppc_status"] == "success"
    assert row["ppc_n_draws"] == 3
    assert row["ppc_deviance_tail_probability"] == ppc.diagnostics.poisson_deviance.posterior_predictive_tail_probability
    assert (row["posterior_artifact_id"], row["predictive_artifact_id"]) == (
        posterior, predictive,
    )
    ppc_json = json.loads(row["ppc_discrepancies_json"])
    assert ppc_json["random_seed_decimal"] == str(_SEED - 1)
    assert ppc_json["selection"]["window_bounds_ns"]["early_window_counts"] == [0.0, 1.0]
    assert "replicated_counts" not in ppc_json
    assert _write(context, posterior_predictive=ppc, posterior_artifact_id=posterior,
                  predictive_artifact_id=predictive, on_duplicate="reuse_identical") == result_id
    with pytest.raises(store.PersistenceConflictError):
        _write(context, posterior_predictive=replace(ppc, runtime_seconds=0.26),
               posterior_artifact_id=posterior, predictive_artifact_id=predictive,
               on_duplicate="reuse_identical")
    with pytest.raises(store.PersistenceConflictError):
        _write(context, posterior_predictive=ppc, posterior_artifact_id=None,
               predictive_artifact_id=predictive, on_duplicate="reuse_identical")
    wrong = store.record_artifact(
        db, artifact_key="wrong-role", path=tmp_path / "wrong.npz", path_base="absolute",
        artifact_kind="histogram", format="npz", sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="artifact role"):
        _write(context, analysis_key="wrong-role", posterior_artifact_id=wrong)
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1


def test_composed_write_rolls_back_and_existing_point_can_gain_summary(context, monkeypatch):
    db = context[0]
    original = store._record_row

    def fail_summary(*args, **kwargs):
        if kwargs.get("table") == "bayesian_summaries":
            raise RuntimeError("summary insert failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_record_row", fail_summary)
    with pytest.raises(RuntimeError, match="summary insert failed"):
        _write(context)
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0
    monkeypatch.setattr(store, "_record_row", original)
    result_id = _write(context)
    db.execute("DELETE FROM bayesian_summaries WHERE result_id = ?", (result_id,))
    assert _write(context, on_duplicate="reuse_identical") == result_id
    assert db.execute("SELECT COUNT(*) FROM bayesian_summaries").fetchone()[0] == 1


def test_full_inference_run_checks_observation_and_counts_finite_draws(context):
    db, run_id, measurement_id, model_id, assumption_id, measurement, prepared, *_ = context
    source = _source()
    physical = np.tile([50.0, 1.8, 1.0, 0.05], (2, 20, 10, 1))
    transformed = np.tile([3.9, 0.6, 0.0, 0.05], (2, 20, 10, 1))
    log_probability = np.zeros((2, 20, 10))
    log_probability[0, 0, 0] = math.nan
    samples = BayesianPosteriorSamples(
        physical=physical, transformed=transformed,
        log_probability=log_probability,
    )
    full = BayesianInferenceRun(
        result=source, samples=samples,
        context=BayesianModelContext(
            measurement, prepared_irf=prepared,
            irf_model_relation=IRFModelRelation.MATCHED,
        ),
    )
    result_id = store.record_bayesian_result(
        db, full, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    counts = json.loads(db.execute(
        "SELECT parameter_summaries_json FROM bayesian_summaries WHERE result_id = ?",
        (result_id,),
    ).fetchone()[0])["sampling_counts"]
    assert counts["valid_retained_samples"] == 399
    changed = replace(
        measurement, values=np.arange(2, 11, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="counts differ"):
        store.record_bayesian_result(
            db, replace(full, context=BayesianModelContext(
                changed, prepared_irf=prepared,
                irf_model_relation=IRFModelRelation.MATCHED,
            )), run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            on_duplicate="reuse_identical",
        )


def test_compact_matched_and_mismatch_share_posterior_median_convention(context):
    db, run_id, measurement_id, model_id, assumption_id, *_ = context
    matched = _matched_record(context)
    matched_id = store.record_issue4_bayesian_result(
        db, matched, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        irf_model_relation="matched", analysis_key="matched-evaluation",
    )
    reference = _pseudo_reference(context)
    mismatch = _mismatch_record(context)
    mismatch_id = store.record_issue4_bayesian_result(
        db, mismatch, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        pseudo_true_reference_id=reference, irf_model_relation="matched",
        analysis_key="mismatch-evaluation",
    )
    for result_id in (matched_id, mismatch_id):
        point = db.execute(
            "SELECT point_summary, lifetime_estimate_ns, random_seed_decimal "
            "FROM estimator_results WHERE result_id = ?", (result_id,),
        ).fetchone()
        assert point[:] == ("posterior_median", 1.8, str(_SEED))
        summary = db.execute(
            "SELECT lifetime_mean_ns, ppc_status, ppc_n_draws, ppc_deviance_tail_probability, "
            "ppc_discrepancies_json FROM bayesian_summaries WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        assert summary["lifetime_mean_ns"] == 1.85
        if result_id == mismatch_id:
            assert summary["ppc_status"] == "success"
            assert summary["ppc_n_draws"] == 3
            assert summary["ppc_deviance_tail_probability"] == 0.4
            assert json.loads(summary["ppc_discrepancies_json"])[
                "random_seed_decimal"
            ] == str(_SEED - 1)
        else:
            assert summary["ppc_status"] == "not_requested"
    assert db.execute("SELECT COUNT(*) FROM uncertainty_results").fetchone()[0] == 0
    assert store.record_issue4_bayesian_result(
        db, mismatch, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        pseudo_true_reference_id=reference, irf_model_relation="matched",
        analysis_key="mismatch-evaluation", on_duplicate="reuse_identical",
    ) == mismatch_id
    changed = replace(mismatch, posterior_predictive=replace(
        mismatch.posterior_predictive, discrepancy_summaries=(
            ("poisson_deviance", 4.0, 5.0, 0.5),
        ),
    ))
    with pytest.raises(store.PersistenceConflictError):
        store.record_issue4_bayesian_result(
            db, changed, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            pseudo_true_reference_id=reference, irf_model_relation="matched",
            analysis_key="mismatch-evaluation", on_duplicate="reuse_identical",
        )


def test_mismatch_reference_condition_and_assumption_validation(context):
    db, run_id, measurement_id, model_id, assumption_id, _, prepared, _, _, condition = context
    reference = _pseudo_reference(context, key="old", version="stage6-original")
    original_reference = tuple(db.execute(
        "SELECT * FROM lifetime_references WHERE reference_id = ?", (reference,),
    ).fetchone())
    corrected = store.record_pseudo_true_reference(
        db, reference_key="corrected", reference_version="stage6.5-corrected",
        condition_pk=condition, assumption_id=assumption_id,
        lifetime_ns=1.71, projection_temporal_shift_ns=0.05,
        supersedes_reference_id=reference,
    )
    mismatch = _mismatch_record(context)
    result_id = store.record_issue4_bayesian_result(
        db, mismatch, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        pseudo_true_reference_id=corrected, irf_model_relation="matched",
    )
    assert result_id > 0
    assert tuple(db.execute(
        "SELECT * FROM lifetime_references WHERE reference_id = ?", (reference,),
    ).fetchone()) == original_reference
    assert db.execute(
        "SELECT reference_version, supersedes_reference_id, condition_pk, assumption_id "
        "FROM lifetime_references WHERE reference_id = ?", (corrected,),
    ).fetchone()[:] == ("stage6.5-corrected", reference, condition, assumption_id)
    # Corrected projection is selectable without replacing physical generating truth
    # or embedding its numerical value into the Bayesian posterior summary.
    condition_row = db.execute(
        "SELECT mono_lifetime_ns FROM simulation_conditions WHERE condition_pk = ?",
        (condition,),
    ).fetchone()
    assert condition_row[0] == 1.8
    posterior_json = db.execute(
        "SELECT parameter_summaries_json FROM bayesian_summaries WHERE result_id = ?",
        (result_id,),
    ).fetchone()[0]
    assert "1.71" not in posterior_json
    assert db.execute("SELECT COUNT(*) FROM benchmark_metrics").fetchone()[0] == 0
    other_condition = store.record_simulation_condition(
        db, condition_key="unrelated-bi", condition_id="unrelated",
        generating_model="biexponential", primary_lifetime_ns=2.0,
        secondary_lifetime_ns=4.0, secondary_detected_fraction=0.2,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0,
    )
    unrelated = _pseudo_reference(
        context, key="wrong-condition", condition_pk=other_condition,
    )
    with pytest.raises(ValueError, match="pseudo-true reference conflicts"):
        store.record_issue4_bayesian_result(
            db, mismatch, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            pseudo_true_reference_id=unrelated, irf_model_relation="matched",
            on_duplicate="reuse_identical",
        )
    source_id = store.record_prepared_irf(
        db, prepared, source_key="bayes-source",
        preparation_key="bayes-prepared", on_duplicate="reuse_identical",
    )
    other_assumption = store.record_model_assumption(
        db, assumption_key="mismatch-assumption", source_assumption_id="gaussian_assumed",
        assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=source_id, temporal_shift_lower_ns=-0.2,
        temporal_shift_upper_ns=0.2,
    )
    other_reference = _pseudo_reference(
        context, key="wrong-assumption", assumption_id=other_assumption,
    )
    with pytest.raises(ValueError, match="pseudo-true reference conflicts"):
        store.record_issue4_bayesian_result(
            db, mismatch, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            pseudo_true_reference_id=other_reference, irf_model_relation="matched",
            on_duplicate="reuse_identical",
        )
    # The same measurement can be analysed under a second declared physical assumption.
    alternate = replace(mismatch, assumption_id="gaussian_assumed")
    alternate_id = store.record_issue4_bayesian_result(
        db, alternate, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=other_assumption,
        pseudo_true_reference_id=other_reference,
        irf_model_relation="deliberately_misspecified",
    )
    assert alternate_id != result_id
    assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1


def test_failed_ppc_partial_diagnostics_and_nonfinite_value(context):
    db, run_id, measurement_id, model_id, assumption_id, *_ = context
    reference = _pseudo_reference(context)
    failed = MismatchPredictiveSummary(
        status="failed", reason="predictive numerical failure", n_draws=2,
        runtime_seconds=0.2,
        discrepancy_summaries=(("poisson_deviance", math.inf, 5.0, math.nan),),
        mean_signed_deviance_residual_profile=(),
    )
    mismatch = _mismatch_record(context, predictive=failed)
    result_id = store.record_issue4_bayesian_result(
        db, mismatch, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        pseudo_true_reference_id=reference, irf_model_relation="matched",
    )
    row = db.execute("SELECT * FROM bayesian_summaries WHERE result_id = ?",
                     (result_id,)).fetchone()
    assert row["ppc_status"] == "failed"
    assert row["ppc_n_draws"] == 2
    assert row["ppc_deviance_tail_probability"] is None
    categories = json.loads(row["nonfinite_fields_json"])
    assert categories["ppc.poisson_deviance.observed_mean"] == "+inf"
    assert categories["ppc.poisson_deviance.tail_probability"] == "nan"
    assert "predictive numerical failure" in row["ppc_discrepancies_json"]


def test_sampling_seed_changes_execution_but_not_reusable_policy(context):
    db, run_id, measured, model, assumption, _, _, priors, sampler, _ = context
    first = _write(context)
    config_before = db.execute(
        "SELECT configuration_json, configuration_sha256 FROM model_versions WHERE model_id = ?",
        (model,),
    ).fetchone()[:]
    rerun_sampler = replace(sampler, random_seed=_SEED - 2)
    assert store.record_model_version(
        db, model_key="bayes-model", estimator_name="emcee_poisson", family="bayesian",
        configuration=store.bayesian_model_configuration(priors, rerun_sampler),
        prior_policy_id="baseline", on_duplicate="reuse_identical",
    ) == model
    with pytest.raises(store.PersistenceConflictError):
        _write(context, _source(priors, rerun_sampler), on_duplicate="reuse_identical")
    second = store.record_bayesian_result(
        db, _source(priors, rerun_sampler), run_id=run_id, measurement_id=measured,
        model_id=model, assumption_id=assumption, analysis_key="new-execution",
    )
    assert first != second
    assert db.execute(
        "SELECT random_seed_decimal FROM estimator_results WHERE result_id = ?", (second,),
    ).fetchone()[0] == str(_SEED - 2)
    assert db.execute(
        "SELECT configuration_json, configuration_sha256 FROM model_versions WHERE model_id = ?",
        (model,),
    ).fetchone()[:] == config_before


def test_contradictory_existing_point_median_is_not_reused(context):
    db = context[0]
    result_id = _write(context)
    db.execute("UPDATE estimator_results SET lifetime_estimate_ns = 3.0 WHERE result_id = ?",
               (result_id,))
    with pytest.raises(store.PersistenceConflictError, match="estimator_results"):
        _write(context, on_duplicate="reuse_identical")
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1


@pytest.mark.parametrize("family", ["ml", "classical", "baseline"])
def test_bayesian_family_required_before_reuse(context, family):
    db, _, _, model, *_ = context
    _write(context)
    db.execute("UPDATE model_versions SET family = ? WHERE model_id = ?", (family, model))
    with pytest.raises(ValueError, match="Bayesian Poisson reconvolution model"):
        _write(context, on_duplicate="reuse_identical")


def test_model_label_is_not_used_to_infer_bayesian_policy(context):
    db, _, _, model, *_ = context
    db.execute("UPDATE model_versions SET estimator_name = 'lab/reconvolution-v1' "
               "WHERE model_id = ?", (model,))
    assert _write(context) > 0


@pytest.mark.parametrize("column,value", [
    ("assumed_decay_model", "biexponential"),
    ("observation_model", "least_squares_reconvolution"),
])
def test_bayesian_physical_assumption_rejected_before_reuse(context, column, value):
    db, _, _, _, assumption, *_ = context
    _write(context)
    # Column names here are a fixed test parametrization, never caller SQL.
    db.execute(f"UPDATE model_assumptions SET {column} = ? WHERE assumption_id = ?",
               (value, assumption))
    with pytest.raises(ValueError, match="monoexponential Poisson assumption"):
        _write(context, on_duplicate="reuse_identical")


@pytest.mark.parametrize("column,value", [
    ("n_bins", 10), ("time_grid_sha256", "f" * 64), ("time_start_ns", 0.25),
])
def test_prepared_target_grid_conflict_precedes_reuse(context, column, value):
    db, _, _, _, assumption, *_ = context
    _write(context)
    prepared = db.execute("SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
                          (assumption,)).fetchone()[0]
    db.execute(f"UPDATE prepared_irfs SET {column} = ? WHERE prepared_irf_id = ?",
               (value, prepared))
    with pytest.raises(ValueError, match="target grid"):
        _write(context, on_duplicate="reuse_identical")


def test_compact_configuration_must_be_complete(context):
    db, run_id, measured, model, assumption, *_ = context
    config = json.loads(db.execute(
        "SELECT configuration_json FROM model_versions WHERE model_id = ?", (model,),
    ).fetchone()[0])
    del config["priors"]["amplitude"]
    db.execute("UPDATE model_versions SET configuration_json = ? WHERE model_id = ?",
               (json.dumps(config), model))
    with pytest.raises(ValueError, match="complete current prior/sampler"):
        store.record_issue4_bayesian_result(
            db, _matched_record(context), run_id=run_id, measurement_id=measured,
            model_id=model, assumption_id=assumption, irf_model_relation="matched",
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


@pytest.mark.parametrize("nonfinite_value,category", [
    (math.nan, "nan"), (math.inf, "+inf"), (-math.inf, "-inf"),
])
def test_autocorrelation_and_compact_ess_categories(context, nonfinite_value, category):
    db, run_id, measured, model, assumption, *_ = context
    result = _source(tau=nonfinite_value, failures=("autocorrelation estimate unavailable",),
                     status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING)
    result_id = _write(context, result)
    categories = json.loads(db.execute(
        "SELECT nonfinite_fields_json FROM bayesian_summaries WHERE result_id = ?", (result_id,),
    ).fetchone()[0])
    assert categories["autocorrelation_time_steps[0,0]"] == category
    compact = replace(_compact(), minimum_effective_samples=nonfinite_value,
                      status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
                      diagnostics_accepted=False, diagnostic_failure_reasons=("unavailable ESS",))
    compact_id = store.record_issue4_bayesian_result(
        db, replace(_matched_record(context), bayesian=compact),
        run_id=run_id, measurement_id=measured, model_id=model,
        assumption_id=assumption, analysis_key="compact", irf_model_relation="matched",
    )
    categories = json.loads(db.execute(
        "SELECT nonfinite_fields_json FROM bayesian_summaries WHERE result_id = ?", (compact_id,),
    ).fetchone()[0])
    assert categories["minimum_effective_samples"] == category


def test_rejected_nonfinite_posterior_median_preserves_category_in_both_rows(context):
    source = _source(median=math.nan, status=BayesianSamplingStatus.NUMERICAL_FAILURE,
                     failures=("nonfinite posterior",))
    result_id = _write(context, source)
    point = context[0].execute(
        "SELECT lifetime_estimate_ns, nonfinite_fields_json FROM estimator_results "
        "WHERE result_id = ?", (result_id,),
    ).fetchone()
    assert point[0] is None
    assert json.loads(point[1])["lifetime_estimate_ns"] == "nan"
    categories = json.loads(context[0].execute(
        "SELECT nonfinite_fields_json FROM bayesian_summaries WHERE result_id = ?", (result_id,),
    ).fetchone()[0])
    assert categories["parameter.lifetime_ns.median"] == "nan"


def test_ppc_from_different_observation_or_sampling_execution_is_rejected(context):
    db = context[0]
    ppc = _ppc(context)
    result_id = _write(context, posterior_predictive=ppc)
    wrong_measurement = replace(ppc.model_context.measurement, values=np.arange(2, 11))
    wrong_context = BayesianModelContext(
        wrong_measurement, prepared_irf=ppc.model_context.prepared_irf,
        irf_model_relation=IRFModelRelation.MATCHED,
    )
    with pytest.raises(ValueError, match="counts differ"):
        _write(context, posterior_predictive=replace(ppc, model_context=wrong_context),
               on_duplicate="reuse_identical")
    with pytest.raises(ValueError, match="posterior prediction conflicts"):
        _write(context, posterior_predictive=replace(ppc, sampling_config=replace(
            ppc.sampling_config, random_seed=41,
        )), on_duplicate="reuse_identical")
    other_irf = prepare_irf(generate_gaussian_irf_profile(
        context[5].time_ns, gaussian_centre_ns=0.9, gaussian_fwhm_ns=0.4,
    ), context[5].time_ns)
    with pytest.raises(ValueError, match="prepared IRF"):
        _write(context, posterior_predictive=replace(ppc, model_context=BayesianModelContext(
            context[5], prepared_irf=other_irf, irf_model_relation=IRFModelRelation.MATCHED,
        )), on_duplicate="reuse_identical")
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1
    assert result_id > 0


def test_diagnostic_parameter_order_and_count_consistency(context):
    source = _source()
    permutation = (2, 1, 0, 3)
    diagnostics = replace(
        source.diagnostics,
        autocorrelation_time_steps=source.diagnostics.autocorrelation_time_steps[:, permutation],
        autocorrelation_relative_change=source.diagnostics.autocorrelation_relative_change[:, permutation],
        approximate_effective_samples=source.diagnostics.approximate_effective_samples[list(permutation)],
    )
    source = replace(
        source, inferred_parameter_names=tuple(source.inferred_parameter_names[i] for i in permutation),
        inferred_parameter_units=tuple(source.inferred_parameter_units[i] for i in permutation),
        parameter_summaries=tuple(source.parameter_summaries[i] for i in permutation),
        correlation_matrix=source.correlation_matrix[np.ix_(permutation, permutation)],
        diagnostics=diagnostics,
    )
    result_id = _write(context, source)
    row = context[0].execute(
        "SELECT lifetime_background_correlation, parameter_summaries_json FROM bayesian_summaries "
        "WHERE result_id = ?", (result_id,),
    ).fetchone()
    assert row[0] == -0.2
    assert json.loads(row[1])["diagnostic_parameter_order"] == list(source.inferred_parameter_names)
    with pytest.raises(ValueError, match="retained samples conflict"):
        _write(context, replace(source, diagnostics=replace(diagnostics, retained_samples=20)),
               analysis_key="count-conflict")


def test_persistence_paths_do_not_import_emcee_or_embed_sample_buffers(context, monkeypatch):
    original_import = builtins.__import__

    def forbid_sampler(name, *args, **kwargs):
        if name == "emcee" or name.startswith("emcee."):
            raise AssertionError("persistence must not import emcee")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_sampler)
    result_id = _write(context, posterior_predictive=_ppc(context))
    row = context[0].execute("SELECT * FROM bayesian_summaries WHERE result_id = ?",
                              (result_id,)).fetchone()
    assert not any(isinstance(value, (bytes, bytearray, memoryview)) for value in row)
    summary_text = row["parameter_summaries_json"] + row["ppc_discrepancies_json"]
    for buffer_name in ("physical_parameter_draws", "expected_counts", "replicated_counts",
                        "selected_posterior_indices", "mean_signed_deviance_residual_profile"):
        assert buffer_name not in summary_text


def test_biexponential_generation_keeps_components_separate_from_bayesian_median(context):
    db, run_id, _, model, assumption, measurement, _, _, _, _ = context
    condition = store.record_simulation_condition(
        db, condition_key="bi-population", condition_id="local-condition",
        generating_model="biexponential", primary_lifetime_ns=2.0,
        secondary_lifetime_ns=4.0, secondary_detected_fraction=0.15,
        signal_photon_count=1000, background_per_bin=0.2, true_temporal_shift_ns=0.0,
    )
    measured = store.record_measurement(
        db, measurement, measurement_key="bi-observation", source_type="synthetic",
        condition_pk=condition,
    )
    store.link_run_measurement(db, run_id=run_id, measurement_id=measured, data_role="evaluation")
    bi_context = (db, run_id, measured, model, assumption, *context[5:9], condition)
    record = replace(
        _mismatch_record(bi_context), mechanism="decay", generating_model="biexponential",
        primary_lifetime_ns=2.0, secondary_lifetime_ns=4.0,
        secondary_detected_fraction=0.15, mono_generating_lifetime_ns=None,
    )
    reference = _pseudo_reference(bi_context)
    result_id = store.record_issue4_bayesian_result(
        db, record, run_id=run_id, measurement_id=measured, model_id=model,
        assumption_id=assumption, pseudo_true_reference_id=reference,
        irf_model_relation="matched",
    )
    assert db.execute("SELECT lifetime_estimate_ns FROM estimator_results WHERE result_id = ?",
                      (result_id,)).fetchone()[0] == 1.8
    assert db.execute(
        "SELECT mono_lifetime_ns, primary_lifetime_ns, secondary_lifetime_ns, "
        "secondary_detected_fraction FROM simulation_conditions WHERE condition_pk = ?",
        (condition,),
    ).fetchone()[:] == (None, 2.0, 4.0, 0.15)


def test_emg_generation_with_matched_and_gaussian_assumed_kernels(context):
    db, run_id, _, model, gaussian_assumption, measurement, gaussian_prepared, *_ = context
    emg_source = generate_emg_irf_profile(
        measurement.time_ns, gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4,
        tail_time_ns=0.15,
    )
    emg_prepared = prepare_irf(emg_source, measurement.time_ns)
    emg_id = store.record_prepared_irf(
        db, emg_prepared, source_key="emg-source", preparation_key="emg-prepared",
    )
    condition = store.record_simulation_condition(
        db, condition_key="emg-population", condition_id="local-condition",
        generating_model="monoexponential", mono_lifetime_ns=1.8,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0, generating_irf_id=emg_id,
    )
    measured = store.record_measurement(
        db, measurement, measurement_key="emg-observation", source_type="synthetic",
        condition_pk=condition,
    )
    store.link_run_measurement(db, run_id=run_id, measurement_id=measured, data_role="evaluation")
    matched_assumption = store.record_model_assumption(
        db, assumption_key="emg-matched", source_assumption_id="matched_emg",
        assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=emg_id, temporal_shift_lower_ns=-0.2, temporal_shift_upper_ns=0.2,
    )
    result_ids = []
    for assumption, label, prepared, relation in (
        (matched_assumption, "matched_emg", emg_prepared, "matched"),
        (gaussian_assumption, "gaussian_mono", gaussian_prepared, "deliberately_misspecified"),
    ):
        emg_context = (db, run_id, measured, model, assumption, measurement, prepared,
                       context[7], context[8], condition)
        reference = _pseudo_reference(emg_context, key=f"emg-reference-{label}")
        record = replace(
            _mismatch_record(emg_context, assumption_label=label),
            generating_irf_id="emg", assumed_irf_id=label,
            time_grid_sha256=store._hash_legacy_issue4_float64_array(measurement.time_ns),
            generating_irf_sha256=store._hash_legacy_issue4_float64_array(emg_prepared.kernel),
            assumed_irf_sha256=store._hash_legacy_issue4_float64_array(prepared.kernel),
        )
        result_ids.append(store.record_issue4_bayesian_result(
            db, record, run_id=run_id, measurement_id=measured, model_id=model,
            assumption_id=assumption, pseudo_true_reference_id=reference,
            irf_model_relation=relation,
        ))
        # The full source path validates the actual prepared source and PPC context,
        # while using the same posterior-median convention as the compact record.
        full_source = _source()
        full_source = replace(full_source, model_context=replace(
            full_source.model_context, irf_source_kind=prepared.source.source_kind,
            irf_model_relation=IRFModelRelation(relation),
        ))
        ppc = replace(_ppc(emg_context), model_context=BayesianModelContext(
            measurement, prepared_irf=prepared, irf_model_relation=IRFModelRelation(relation),
        ))
        full_id = store.record_bayesian_result(
            db, full_source, run_id=run_id, measurement_id=measured, model_id=model,
            assumption_id=assumption, analysis_key="full-source", posterior_predictive=ppc,
        )
        assert db.execute(
            "SELECT e.point_summary, e.lifetime_estimate_ns, a.prepared_irf_id, b.ppc_status "
            "FROM estimator_results AS e JOIN model_assumptions AS a USING (assumption_id) "
            "JOIN bayesian_summaries AS b USING (result_id) WHERE e.result_id = ?", (full_id,),
        ).fetchone()[:] == (
            "posterior_median", record.bayesian.lifetime_median_ns,
            emg_id if assumption == matched_assumption else db.execute(
                "SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
                (gaussian_assumption,),
            ).fetchone()[0], "success",
        )
    assert result_ids[0] != result_ids[1]
    rows = db.execute(
        "SELECT measurement_id, irf_model_relation FROM estimator_results "
        "WHERE result_id IN (?, ?) ORDER BY result_id", result_ids,
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (measured, "matched"), (measured, "deliberately_misspecified"),
    ]
    assert db.execute("SELECT generating_irf_id FROM simulation_conditions WHERE condition_pk = ?",
                      (condition,)).fetchone()[0] == emg_id
    assert db.execute("SELECT COUNT(*) FROM measurements WHERE condition_pk = ?",
                      (condition,)).fetchone()[0] == 1


def test_invalid_artifact_run_and_missing_artifact_fail_atomically(context, tmp_path):
    db, _, _, _, _, *_ = context
    other_run = store.record_run(
        db, run_key="another-run", run_type="evaluation", status="complete",
        recorded_at_utc="2026-10-02T11:00:00+00:00", origin="new", package_version="0.7.0",
        configuration={},
    )
    artifact = store.record_artifact(
        db, artifact_key="other-run-chain", path=tmp_path / "same-path.nc",
        path_base="absolute", artifact_kind="posterior_chain", format="netcdf",
        sha256="a" * 64, producing_run_id=other_run,
    )
    for selected in (artifact, 999):
        with pytest.raises(ValueError, match="artifact role or producing run"):
            _write(context, posterior_artifact_id=selected)
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


def test_chain_artifact_path_does_not_define_result_identity(context, tmp_path):
    db = context[0]
    first = store.record_artifact(
        db, artifact_key="first-chain", path=tmp_path / "chain.nc", path_base="absolute",
        artifact_kind="posterior_chain", format="netcdf", sha256="a" * 64,
    )
    second = store.record_artifact(
        db, artifact_key="second-chain", path=tmp_path / "chain.nc", path_base="absolute",
        artifact_kind="posterior_chain", format="netcdf", sha256="b" * 64,
    )
    _write(context, posterior_artifact_id=first)
    with pytest.raises(store.PersistenceConflictError, match="Bayesian summary"):
        _write(context, posterior_artifact_id=second, on_duplicate="reuse_identical")
    assert _write(context, posterior_artifact_id=second, analysis_key="explicit-repeat") > 0


def test_fixed_shift_prior_uses_existing_numerical_tolerance(context):
    db, run_id, measured, _, original_assumption, _, _, _, sampler, _ = context
    priors = _priors(fixed_shift=0.05)
    model = store.record_model_version(
        db, model_key="fixed-tolerance", estimator_name="emcee_poisson", family="bayesian",
        configuration=store.bayesian_model_configuration(priors, sampler),
    )
    prepared_id = db.execute("SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
                             (original_assumption,)).fetchone()[0]
    assumption = store.record_model_assumption(
        db, assumption_key="fixed-tolerance", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=prepared_id,
        fixed_temporal_shift_ns=0.05 + 1e-10,
    )
    source = _source(priors, sampler)
    assert store.record_bayesian_result(
        db, source, run_id=run_id, measurement_id=measured,
        model_id=model, assumption_id=assumption,
    ) > 0
    db.execute("UPDATE model_assumptions SET fixed_temporal_shift_ns = 0.1 WHERE assumption_id = ?",
               (assumption,))
    with pytest.raises(ValueError, match="fixed model assumption"):
        store.record_bayesian_result(
            db, source, run_id=run_id, measurement_id=measured, model_id=model,
            assumption_id=assumption, on_duplicate="reuse_identical",
        )


def test_crossed_failed_credible_interval_is_retained(context):
    source = _source(status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
                     failures=("insufficient samples",))
    source = replace(source, parameter_summaries=tuple(
        replace(item, credible_lower=2.0, credible_upper=1.0)
        if item.name == "lifetime_ns" else item for item in source.parameter_summaries
    ))
    result_id = _write(context, source)
    summary = json.loads(context[0].execute(
        "SELECT parameter_summaries_json FROM bayesian_summaries WHERE result_id = ?", (result_id,),
    ).fetchone()[0])
    lifetime = next(item for item in summary["parameters"] if item["name"] == "lifetime_ns")
    assert (lifetime["credible_lower"], lifetime["credible_upper"]) == (2.0, 1.0)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_acceptance_fraction_range_is_validated_before_writing(context, probability):
    source = _source()
    with pytest.raises(ValueError, match="between zero and one"):
        _write(context, replace(source, diagnostics=replace(
            source.diagnostics, mean_acceptance_fraction=probability,
        )))
    assert context[0].execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


@pytest.mark.parametrize("changes", [
    {"sample_id": "another-observation"},
    {"metadata": {"acquisition": "another-observation"}},
    {"provenance": {"source": "another-observation"}},
])
def test_ppc_identical_counts_do_not_override_observation_identity(context, changes):
    ppc = _ppc(context)
    _write(context, posterior_predictive=ppc)
    other_measurement = replace(context[5], **changes)
    np.testing.assert_array_equal(other_measurement.values, context[5].values)
    other_context = BayesianModelContext(
        other_measurement, prepared_irf=context[6], irf_model_relation=IRFModelRelation.MATCHED,
    )
    with pytest.raises(ValueError, match="measurement identity/provenance"):
        _write(context, posterior_predictive=replace(ppc, model_context=other_context),
               on_duplicate="reuse_identical")


def test_ppc_identical_kernel_does_not_override_irf_source_provenance(context):
    ppc = _ppc(context)
    _write(context, posterior_predictive=ppc)
    prepared = context[6]
    other_source = replace(prepared.source, provenance={"source": "another-irf"})
    other_irf = prepare_irf(other_source, context[5].time_ns)
    np.testing.assert_array_equal(other_irf.kernel, prepared.kernel)
    with pytest.raises(ValueError, match="prepared IRF"):
        _write(context, posterior_predictive=replace(ppc, model_context=BayesianModelContext(
            context[5], prepared_irf=other_irf, irf_model_relation=IRFModelRelation.MATCHED,
        )), on_duplicate="reuse_identical")
