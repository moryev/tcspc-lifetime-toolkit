"""Issue-9 Stage-4 uncertainty and aggregate benchmark persistence."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tcspc_toolkit import persistence as store
from tcspc_toolkit.classical_uncertainty import (
    CLASSICAL_UNCERTAINTY_METHODS, ParametricPoissonBootstrapResult,
    PoissonLocalCovarianceResult,
    RepeatedPoissonUncertaintyResult,
)
from tcspc_toolkit.classical_evaluation import ReconvolutionBenchmarkSummary
from tcspc_toolkit.fitting import ReconvolutionFitResult
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.ml_uncertainty import (
    BootstrapPredictionSpreadEstimator, DEFAULT_QUANTILE_NOMINAL_COVERAGE,
    ML_UNCERTAINTY_METHODS, RandomForestTreeSpreadEstimator,
    evaluate_frozen_quantile_gradient_boosting,
)
from tcspc_toolkit.generalization_evaluation import RobustnessMetrics
from tcspc_toolkit.ml_evaluation import RegressionMetrics
from tcspc_toolkit.uncertainty_evaluation import (
    IntervalEvaluationMetrics, PredictionIntervalResult,
    QuantileIntervalEvaluationMetrics, UncertaintyScoreMetrics,
    SelectivePredictionMetrics, UncertaintyScoreResult,
    evaluate_prediction_intervals, evaluate_selective_prediction,
)


def _run(db, key):
    return store.record_run(
        db, run_key=key, run_type="evaluation", status="complete",
        recorded_at_utc="2026-10-01T10:00:00+00:00", origin="new",
        package_version="0.7.0", configuration={"stage": 4},
        protocol_id="week9", profile="frozen",
    )


@pytest.fixture
def context():
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as db:
        store.initialize_database(db)
        db.row_factory = sqlite3.Row
        run = _run(db, "evaluation-1")
        calibration = _run(db, "calibration-1")
        model = store.record_model_version(
            db, model_key="ml-1", estimator_name="ml-1", family="ml",
            configuration={"estimator": "frozen"},
        )
        condition = store.record_simulation_condition(
            db, condition_key="population-condition", condition_id="r1",
            generating_model="monoexponential", mono_lifetime_ns=1.7,
            signal_photon_count=1000, background_per_bin=0.2,
            true_temporal_shift_ns=0.0,
        )
        result_ids = []
        measurement_ids = []
        for index, prediction in enumerate((1.5, 2.0)):
            measurement = TCSPCMeasurement(
                time_ns=np.arange(0.0, 2.25, 0.25),
                values=np.arange(1, 10, dtype=np.int64),
                data_kind=MeasurementDataKind.RAW_COUNTS,
                sample_id=f"sample-{index}",
            )
            measured = store.record_measurement(
                db, measurement, measurement_key=f"measurement-{index}",
                source_type="synthetic", condition_pk=condition,
            )
            store.link_run_measurement(
                db, run_id=run, measurement_id=measured, data_role="evaluation",
                dataset_key="dataset-1", test_id="A", regime_id="r1",
            )
            point = store.record_scalar_prediction(
                db, prediction, run_id=run, measurement_id=measured,
                model_id=model, source_result_type="frozen_predict",
            )
            result_ids.append(point)
            measurement_ids.append(measured)
        yield db, run, calibration, model, tuple(result_ids), tuple(measurement_ids)


def _interval(predictions=(1.5,), lower=(1.2,), upper=(1.8,),
              coverage=0.9, method="quantile_gradient_boosting"):
    return PredictionIntervalResult(
        prediction=np.asarray(predictions, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        nominal_coverage=coverage, method_id=method,
    )


def _scope(run, model, *, test="A", coverage=0.9, method="quantile_gradient_boosting",
           selection=None, **changes):
    base = store.BenchmarkMetricScope(
        run_id=run, population_key="frozen-test-population", dataset_key="dataset-1",
        test_id=test, regime_id="r1", model_id=model,
        reference_semantics="generating_mono", reference_kind="generating_mono",
        method_id=method,
        method_configuration={"b": 2, "a": 1}, nominal_coverage=coverage,
        selection=selection,
    )
    return replace(base, **changes)


def _interval_metrics(*, coverage=0.8, nominal=0.9):
    return IntervalEvaluationMetrics(
        n_samples=2, n_valid_intervals=2, empirical_coverage=coverage,
        coverage_error=coverage - nominal, mean_interval_width=0.6,
        median_interval_width=0.6, mean_interval_score=0.7,
        interval_failure_rate=0.0,
    )


def _classical_model(db, key, *, objective="poisson"):
    return store.record_model_version(
        db, model_key=key, estimator_name="reconvolution", family="classical",
        configuration={"objective": objective},
    )


def _classical_point(db, run, measurement_id):
    model = _classical_model(db, "poisson-fit")
    time = np.arange(0.0, 2.25, 0.25)
    prepared = prepare_irf(
        generate_gaussian_irf_profile(
            time, gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4,
        ), time,
    )
    prepared_id = store.record_prepared_irf(
        db, prepared, source_key="fit-irf", preparation_key="fit-prepared",
    )
    assumption = store.record_model_assumption(
        db, assumption_key="poisson-assumption",
        assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=prepared_id,
        temporal_shift_lower_ns=-0.25, temporal_shift_upper_ns=0.25,
    )
    fit = ReconvolutionFitResult(
        amplitude=100.0, lifetime=1.7, background=0.2,
        temporal_shift=0.01, fitted_curve=np.arange(1.0, 10.0),
        success=True, optimizer_reported_success=True,
        numerical_validation_passed=True, max_coordinate_descent_nll=0.002,
        recovery_attempted=False, optimizer_status=0,
        optimizer_message="converged", optimizer_nfev=25, optimizer_njev=8,
    )
    point = store.record_reconvolution_fit(
        db, fit, run_id=run, measurement_id=measurement_id,
        model_id=model, assumption_id=assumption,
    )
    return point, assumption


def test_interval_identity_calibration_nonfinite_and_crossing(context):
    db, run, calibration, _, ids, _ = context
    source = _interval()
    method = ML_UNCERTAINTY_METHODS["quantile_gradient_boosting"]
    kwargs = dict(result_ids=ids[:1], method=method,
                  method_configuration={"a": 1, "b": 2}, interval_kind="quantile")
    first = store.record_prediction_intervals(db, source, **kwargs)[0]
    with pytest.raises(store.PersistenceConflictError, match="duplicate"):
        store.record_prediction_intervals(db, source, **kwargs)
    assert store.record_prediction_intervals(
        db, source, **{**kwargs, "method_configuration": {"b": 2, "a": 1}},
        on_duplicate="reuse_identical",
    ) == (first,)
    with pytest.raises(store.PersistenceConflictError, match="conflicting"):
        store.record_prediction_intervals(
            db, _interval(lower=(1.1,)), **kwargs, on_duplicate="reuse_identical",
        )
    store.record_prediction_intervals(
        db, _interval(coverage=0.8), **kwargs,
    )
    seeded_config = store.record_prediction_intervals(
        db, source, **{**kwargs, "method_configuration": {"a": 2}},
    )[0]
    assert "nominal_coverage" in json.loads(db.execute(
        "SELECT method_config_json FROM uncertainty_results WHERE uncertainty_id=?",
        (seeded_config,),
    ).fetchone()[0])
    retained_seed = store.record_prediction_intervals(
        db, source, **{**kwargs, "method_configuration": {"random_seed": 17}},
    )[0]
    assert json.loads(db.execute(
        "SELECT method_config_json FROM uncertainty_results WHERE uncertainty_id=?",
        (retained_seed,),
    ).fetchone()[0])["random_seed"] == 17
    equal = store.record_prediction_intervals(
        db, _interval(predictions=(2.0,), lower=(2.0,), upper=(2.0,)),
        result_ids=ids[1:], method=method,
        method_configuration={"a": 1}, interval_kind="quantile",
    )[0]
    assert db.execute("SELECT is_valid FROM uncertainty_results WHERE uncertainty_id=?", (equal,)).fetchone()[0] == 1
    crossed = store.record_quantile_intervals(
        db, _interval(predictions=(2.0,), lower=(2.2,), upper=(2.1,),
                      coverage=DEFAULT_QUANTILE_NOMINAL_COVERAGE),
        result_ids=ids[1:],
    )[0]
    row = db.execute("SELECT * FROM uncertainty_results WHERE uncertainty_id=?", (crossed,)).fetchone()
    assert (row["is_valid"], row["lower_ns"], row["upper_ns"]) == (0, 2.2, 2.1)
    assert json.loads(row["diagnostics_json"])["quantile_triplet_crossed"] is True
    triplet_crossed = store.record_quantile_intervals(
        db, _interval(predictions=(1.5,), lower=(1.6,), upper=(1.8,),
                      coverage=DEFAULT_QUANTILE_NOMINAL_COVERAGE),
        result_ids=ids[:1],
    )[0]
    triplet_row = db.execute(
        "SELECT * FROM uncertainty_results WHERE uncertainty_id=?",
        (triplet_crossed,),
    ).fetchone()
    assert triplet_row["lower_ns"] < triplet_row["upper_ns"]
    assert triplet_row["is_valid"] == 1
    assert json.loads(triplet_row["diagnostics_json"])["quantile_triplet_crossed"] is True
    invalid = store.record_prediction_intervals(
        db, _interval(lower=(float("nan"),), upper=(float("inf"),)),
        **{**kwargs, "method_configuration": {"a": 3}},
    )[0]
    row = db.execute("SELECT * FROM uncertainty_results WHERE uncertainty_id=?", (invalid,)).fetchone()
    assert row["is_valid"] == 0 and row["lower_ns"] is None and row["upper_ns"] is None
    assert json.loads(row["nonfinite_fields_json"]) == {"lower_ns": "nan", "upper_ns": "+inf"}
    calibrated = store.record_conformal_intervals(
        db, _interval(method="conformalized_quantile_gradient_boosting"),
        result_ids=ids[:1], calibration_run_id=calibration,
        correction_ns=0.2, n_calibration_scores=20,
    )[0]
    assert db.execute("SELECT calibration_run_id FROM uncertainty_results WHERE uncertainty_id=?", (calibrated,)).fetchone()[0] == calibration
    another_calibration = _run(db, "calibration-2")
    another = store.record_conformal_intervals(
        db, _interval(method="conformalized_quantile_gradient_boosting"),
        result_ids=ids[:1], calibration_run_id=another_calibration,
        correction_ns=0.2, n_calibration_scores=20,
    )[0]
    assert another != calibrated
    with pytest.raises(ValueError, match="nominal_coverage"):
        _interval(coverage=1.0)
    with pytest.raises(ValueError, match="unknown estimator"):
        store.record_prediction_intervals(db, source, **{**kwargs, "result_ids": [99999]})


def test_uncertainty_batch_is_atomic_and_alignment_is_explicit(context):
    db, _, _, _, ids, _ = context
    method = ML_UNCERTAINTY_METHODS["quantile_gradient_boosting"]
    store.record_prediction_intervals(
        db, _interval(predictions=(2.0,), lower=(1.8,), upper=(2.2,)),
        result_ids=ids[1:], method=method,
        method_configuration={"batch": 1}, interval_kind="quantile",
    )
    with pytest.raises(store.PersistenceConflictError):
        store.record_prediction_intervals(
            db, _interval(predictions=(1.5, 2.0), lower=(1.2, 1.7),
                          upper=(1.8, 2.3)),
            result_ids=ids, method=method,
            method_configuration={"batch": 1}, interval_kind="quantile",
            on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM uncertainty_results").fetchone()[0] == 1
    with pytest.raises(ValueError, match="length"):
        store.record_prediction_intervals(
            db, _interval(), result_ids=ids, method=method,
            method_configuration={}, interval_kind="quantile",
        )
    with pytest.raises(ValueError, match="ordered"):
        store.record_prediction_intervals(
            db, _interval(), result_ids={ids[0]}, method=method,
            method_configuration={}, interval_kind="quantile",
        )


def test_score_spreads_are_not_intervals(context):
    db, _, _, _, ids, _ = context
    for method_id, score in (("random_forest_tree_spread", 0.0),
                             ("ml_training_bootstrap", float("nan"))):
        source = UncertaintyScoreResult(
            prediction=np.array([1.5]), uncertainty_score=np.array([score]),
            method_id=method_id,
        )
        stored = store.record_uncertainty_scores(
            db, source, result_ids=ids[:1],
            method=ML_UNCERTAINTY_METHODS[method_id],
            method_configuration={}, n_members=17,
        )[0]
        row = db.execute("SELECT * FROM uncertainty_results WHERE uncertainty_id=?", (stored,)).fetchone()
        assert row["output_kind"] == "uncertainty_score"
        assert row["nominal_coverage"] is row["lower_ns"] is row["upper_ns"] is None
        assert row["calibration_run_id"] is None
        assert row["is_valid"] == int(np.isfinite(score))
        assert json.loads(row["method_config_json"])["n_members"] == 17
        expected = ("population_std", 0) if method_id == "random_forest_tree_spread" else ("sample_std", 1)
        config = json.loads(row["method_config_json"])
        assert (config["spread"], config["ddof"]) == expected
        if not np.isfinite(score):
            assert json.loads(row["nonfinite_fields_json"])["uncertainty_score"] == "nan"
    with pytest.raises(ValueError, match="non-negative"):
        UncertaintyScoreResult(
            prediction=np.array([1.5]), uncertainty_score=np.array([-0.1]),
            method_id="random_forest_tree_spread",
        )


def test_classical_covariance_and_bootstrap_keep_failure_facts(context):
    db, run, _, _, _, measurements = context
    point, _ = _classical_point(db, run, measurements[0])
    covariance = PoissonLocalCovarianceResult(
        covariance_matrix=np.eye(4), amplitude_std=0.2, lifetime_std=0.1,
        background_std=0.01, temporal_shift_std=0.02,
        covariance_valid=True, condition_number=10.0, information_rank=4,
        boundary_hit=False, failure_reason=None,
    )
    local = store.record_classical_covariance_uncertainty(
        db, covariance, result_id=point,
        method_configuration={"relative_step": 1e-4, "method": "fisher"},
    )
    bootstrap = ParametricPoissonBootstrapResult(
        source_lifetime_ns=1.7, lifetime_samples_ns=np.array([1.6, 1.8, np.nan]),
        bootstrap_std_ns=0.1, bootstrap_median_ns=1.7,
        lower_ns=1.5, upper_ns=1.9, nominal_coverage=0.9,
        n_resamples=3, n_successful_fits=2, n_failed_fits=1,
        n_boundary_hits=0, fit_failure_rate=1/3,
        boundary_hit_rate=0.0, bootstrap_valid=True, failure_reason=None,
    )
    boot = store.record_parametric_bootstrap_uncertainty(
        db, bootstrap, result_id=point, method_configuration={"procedure": "poisson_refit"},
        bootstrap_seed=123,
    )
    rows = db.execute("SELECT * FROM uncertainty_results WHERE result_id=? ORDER BY uncertainty_id", (point,)).fetchall()
    assert len(rows) == 2 and {row["uncertainty_id"] for row in rows} == {local, boot}
    assert rows[0]["output_kind"] == "covariance_summary" and rows[0]["reported_std_ns"] == 0.1
    assert rows[1]["output_kind"] == "prediction_interval" and rows[1]["reported_std_ns"] == 0.1
    assert (rows[1]["n_requested"], rows[1]["n_valid"], rows[1]["random_seed_decimal"]) == (3, 2, "123")
    assert rows[1]["samples_artifact_id"] is None
    covariance_interval = store.record_prediction_intervals(
        db, _interval(predictions=(1.7,), lower=(1.5,), upper=(1.9,),
                      method="covariance"),
        result_ids=[point],
        method=CLASSICAL_UNCERTAINTY_METHODS["covariance"],
        method_configuration={"relative_step": 1e-4, "method": "fisher"},
        interval_kind="normal_local_covariance",
    )[0]
    assert db.execute(
        "SELECT output_kind FROM uncertainty_results WHERE uncertainty_id=?",
        (covariance_interval,),
    ).fetchone()[0] == "prediction_interval"
    failed_cov = replace(covariance, covariance_valid=False, lifetime_std=float("nan"),
                         condition_number=float("inf"), failure_reason="singular")
    failed_id = store.record_classical_covariance_uncertainty(
        db, failed_cov, result_id=point, method_configuration={"relative_step": 2e-4},
    )
    failed_row = db.execute("SELECT * FROM uncertainty_results WHERE uncertainty_id=?", (failed_id,)).fetchone()
    assert failed_row["is_valid"] == 0 and failed_row["reported_std_ns"] is None
    assert json.loads(failed_row["nonfinite_fields_json"])["reported_std_ns"] == "nan"
    diagnostic_negative = store.record_classical_covariance_uncertainty(
        db, replace(covariance, covariance_valid=False, lifetime_std=-0.1,
                    failure_reason="invalid curvature"),
        result_id=point, method_configuration={"relative_step": 3e-4},
    )
    assert tuple(db.execute(
        "SELECT is_valid, reported_std_ns FROM uncertainty_results WHERE uncertainty_id=?",
        (diagnostic_negative,),
    ).fetchone()) == (0, -0.1)
    failed_boot = replace(
        bootstrap, lifetime_samples_ns=np.full(3, np.nan), n_successful_fits=0,
        n_failed_fits=3, bootstrap_valid=False, bootstrap_std_ns=float("nan"),
        bootstrap_median_ns=float("nan"), lower_ns=float("nan"),
        upper_ns=float("inf"), fit_failure_rate=1.0,
        failure_reason="no valid refits",
    )
    failed_id = store.record_parametric_bootstrap_uncertainty(
        db, failed_boot, result_id=point,
        method_configuration={"procedure": "poisson_refit", "variant": "failed"},
    )
    failed_row = db.execute("SELECT * FROM uncertainty_results WHERE uncertainty_id=?", (failed_id,)).fetchone()
    assert failed_row["is_valid"] == 0 and failed_row["n_valid"] == 0
    assert json.loads(failed_row["nonfinite_fields_json"])["upper_ns"] == "+inf"


def test_interval_and_quantile_metric_facts_scope_and_atomicity(context):
    db, run, _, model, _, measurements = context
    scope = _scope(run, model, measurement_ids=measurements)
    ids = store.record_interval_metrics(db, _interval_metrics(), scope=scope)
    assert len(ids) == 6
    row = db.execute("SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[0],)).fetchone()
    assert (row["metric_name"], row["n_attempted"], row["n_valid"],
            row["n_contributing"], row["denominator_kind"]) == (
                "empirical_coverage", 2, 2, 2, "valid_intervals",
            )
    assert store.record_interval_metrics(
        db, _interval_metrics(), scope=scope, on_duplicate="reuse_identical",
    ) == ids
    with pytest.raises(store.PersistenceConflictError, match="duplicate"):
        store.record_interval_metrics(db, _interval_metrics(), scope=scope)
    with pytest.raises(store.PersistenceConflictError):
        store.record_interval_metrics(
            db, _interval_metrics(coverage=0.7), scope=scope,
            on_duplicate="reuse_identical",
        )
    for changed, source_metrics in (
        (_scope(run, model, test="F"), _interval_metrics()),
        (_scope(run, model, coverage=0.8), _interval_metrics(nominal=0.8)),
        (_scope(run, model, population_key="other-population"), _interval_metrics()),
        (_scope(run, model, method_configuration={"a": 2, "b": 2}), _interval_metrics()),
    ):
        store.record_interval_metrics(db, source_metrics, scope=changed)
    assert db.execute("SELECT COUNT(DISTINCT scope_sha256) FROM benchmark_metrics").fetchone()[0] == 5
    quantile = QuantileIntervalEvaluationMetrics(
        n_samples=2, n_finite_quantile_triplets=1, lower_pinball_loss=0.1,
        median_pinball_loss=0.2, upper_pinball_loss=0.3,
        mean_pinball_loss=0.2, quantile_crossing_rate=1.0,
    )
    q_ids = store.record_quantile_interval_metrics(
        db, quantile, scope=replace(scope, population_key="quantile-eval"),
    )
    q_row = db.execute("SELECT * FROM benchmark_metrics WHERE metric_id=?", (q_ids[-1],)).fetchone()
    assert (q_row["metric_name"], q_row["n_contributing"], q_row["denominator_kind"]) == (
        "quantile_crossing_rate", 1, "finite_quantile_triplets",
    )
    db.execute("DELETE FROM benchmark_metrics WHERE metric_id IN (?, ?, ?, ?, ?)", ids[:5])
    with pytest.raises(store.PersistenceConflictError):
        store.record_interval_metrics(
            db, replace(_interval_metrics(), n_valid_intervals=1,
                        interval_failure_rate=0.5),
            scope=scope, on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM benchmark_metrics WHERE scope_sha256=?", (row["scope_sha256"],)).fetchone()[0] == 1


def test_score_metric_undefined_is_not_zero_and_scope_hash_is_canonical(context):
    db, run, _, model, _, _ = context
    scope = _scope(run, model, coverage=None, method="random_forest_tree_spread",
                   selection={"tail_fraction": 0.2})
    metrics = UncertaintyScoreMetrics(
        n_samples=2, n_valid_scores=2, score_failure_rate=0.0,
        mean_absolute_error_ns=0.1, spearman_error_correlation=float("nan"),
        low_uncertainty_mae_ns=0.08, high_uncertainty_mae_ns=0.12,
        tail_fraction=0.2,
    )
    scores = UncertaintyScoreResult(
        prediction=np.array([1.5, 2.0]), uncertainty_score=np.array([0.0, 0.2]),
        method_id="random_forest_tree_spread",
    )
    ids = store.record_uncertainty_score_metrics(db, metrics, scope=scope, scores=scores)
    assert len(ids) == 7
    rows = {row["metric_name"]: row for row in db.execute("SELECT * FROM benchmark_metrics")}
    assert rows["spearman_error_correlation"]["value_status"] == "undefined"
    assert rows["spearman_error_correlation"]["metric_value"] is None
    assert json.loads(rows["spearman_error_correlation"]["nonfinite_fields_json"]) == {"metric_value": "nan"}
    assert rows["score_failure_rate"]["value_status"] == "finite"
    assert rows["score_failure_rate"]["metric_value"] == 0.0
    assert rows["mean_uncertainty_score"]["metric_value"] == 0.1
    assert rows["low_uncertainty_mae_ns"]["n_contributing"] is None
    reordered = replace(scope, method_configuration={"a": 1, "b": 2})
    assert store.record_uncertainty_score_metrics(
        db, metrics, scope=reordered, scores=scores, on_duplicate="reuse_identical",
    ) == ids
    other_run = _run(db, "evaluation-2")
    store.record_uncertainty_score_metrics(
        db, metrics, scope=replace(scope, run_id=other_run), scores=scores,
    )
    assert db.execute("SELECT COUNT(DISTINCT scope_sha256) FROM benchmark_metrics").fetchone()[0] == 2


def test_no_valid_intervals_stay_undefined(context):
    db, run, _, model, _, _ = context
    metrics = IntervalEvaluationMetrics(
        n_samples=2, n_valid_intervals=0,
        empirical_coverage=float("nan"), coverage_error=float("nan"),
        mean_interval_width=float("nan"), median_interval_width=float("nan"),
        mean_interval_score=float("nan"), interval_failure_rate=1.0,
    )
    ids = store.record_interval_metrics(
        db, metrics, scope=_scope(run, model, population_key="all-invalid"),
    )
    coverage = db.execute(
        "SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[0],),
    ).fetchone()
    assert coverage["value_status"] == "undefined"
    assert coverage["metric_value"] is None and coverage["n_valid"] == 0
    failure = db.execute(
        "SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[-1],),
    ).fetchone()
    assert failure["metric_value"] == 1.0 and failure["n_contributing"] == 2


def test_selective_prediction_metrics_preserve_retained_count(context):
    db, run, _, model, _, _ = context
    metrics = SelectivePredictionMetrics(
        rejection_fraction=0.5, retained_fraction=0.5,
        n_valid_scores=2, n_retained=1,
        mae_all_ns=0.2, mae_retained_ns=0.1,
        mae_improvement_ns=0.1,
    )
    scope = _scope(
        run, model, method="random_forest_tree_spread", coverage=None,
        selection={"rejection_fraction": 0.5},
    )
    ids = store.record_selective_prediction_metrics(
        db, metrics, scope=scope, n_attempted=2,
    )
    assert len(ids) == 4
    retained = db.execute(
        "SELECT * FROM benchmark_metrics WHERE metric_name='mae_retained_ns'"
    ).fetchone()
    assert (retained["n_attempted"], retained["n_valid"],
            retained["n_contributing"], retained["denominator_kind"]) == (
                2, 2, 1, "retained_predictions",
            )


def _repeated_result():
    covariance_intervals = _interval(
        predictions=(1.6, 1.8), lower=(1.4, 1.6), upper=(1.8, 2.0),
        method="covariance",
    )
    bootstrap_intervals = _interval(
        predictions=(1.6, 1.8), lower=(1.3, 1.5), upper=(1.9, 2.1),
        method="parametric_bootstrap",
    )
    covariance_metrics = _interval_metrics()
    bootstrap_metrics = replace(covariance_metrics, mean_interval_width=0.8)
    return RepeatedPoissonUncertaintyResult(
        true_lifetime_ns=1.7, signal_photon_count=1000,
        background_per_bin=0.2, irf_fwhm_ns=0.4, irf_shift_ns=0.0,
        lifetime_estimates_ns=np.array([1.6, 1.8]),
        empirical_bias_ns=0.0, empirical_std_ns=0.14,
        empirical_rmse_ns=0.1, fit_failure_rate=0.0, boundary_hit_rate=0.0,
        covariance_std_ns=np.array([0.1, 0.12]),
        covariance_intervals=covariance_intervals,
        covariance_metrics=covariance_metrics,
        mean_covariance_std_ns=0.11, covariance_to_empirical_std_ratio=0.79,
        bootstrap_std_ns=np.array([0.13, 0.15]),
        bootstrap_intervals=bootstrap_intervals,
        bootstrap_metrics=bootstrap_metrics,
        mean_bootstrap_std_ns=0.14, bootstrap_to_empirical_std_ratio=1.0,
        bootstrap_fit_failure_rates=np.array([0.1, 0.2]),
        mean_bootstrap_fit_failure_rate=0.15,
    )


def test_repeated_poisson_is_aggregate_not_point_uncertainty(context):
    db, run, _, _, _, _ = context
    model = _classical_model(db, "repeated-poisson-model")
    condition = store.record_simulation_condition(
        db, condition_key="repeated-condition", condition_id="r1",
        generating_model="monoexponential", mono_lifetime_ns=1.7,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0,
    )
    repeated = _repeated_result()
    scope = _scope(
        run, model, method="repeated_poisson_realizations", condition_pk=condition,
        reference_kind="generating_mono",
        reference_semantics="generating_mono_and_empirical_repeated_poisson",
        signal_photon_count=1000, background_per_bin=0.2,
    )
    ids = store.record_repeated_poisson_metrics(db, repeated, scope=scope)
    assert len(ids) == 22
    row = db.execute("SELECT * FROM benchmark_metrics WHERE metric_name='empirical_lifetime_std_ns'").fetchone()
    assert (row["metric_value"], row["n_attempted"], row["reference_kind"], row["condition_pk"]) == (0.14, 2, "generating_mono", condition)
    assert json.loads(row["scope_json"])["selection"] == {
        "irf_fwhm_ns": 0.4, "irf_shift_ns": 0.0,
    }
    boundary = db.execute("SELECT * FROM benchmark_metrics WHERE metric_name='boundary_hit_rate'").fetchone()
    assert (boundary["denominator_kind"], boundary["n_contributing"]) == ("successful_fits", 2)
    assert db.execute("SELECT COUNT(*) FROM uncertainty_results").fetchone()[0] == 0


def test_reference_scope_distinguishes_generating_component_and_pseudo_true(context):
    db, run, _, model, _, _ = context
    condition = store.record_simulation_condition(
        db, condition_key="mismatch-condition", condition_id="test-F-condition",
        generating_model="biexponential", primary_lifetime_ns=1.5,
        secondary_lifetime_ns=3.0, secondary_detected_fraction=0.2,
        signal_photon_count=500, background_per_bin=0.1,
        true_temporal_shift_ns=0.0,
    )
    assumption = store.record_model_assumption(
        db, assumption_key="mismatch-assumption",
        assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution",
        background_convention="per_bin",
        context_completeness="historical_incomplete",
    )
    reference = store.record_pseudo_true_reference(
        db, reference_key="pseudo-F", reference_version="corrected-v1",
        condition_pk=condition, assumption_id=assumption, lifetime_ns=1.8,
    )
    reference_two = store.record_pseudo_true_reference(
        db, reference_key="pseudo-F-second", reference_version="corrected-v1",
        condition_pk=condition, assumption_id=assumption, lifetime_ns=1.9,
    )
    base = _scope(
        run, model, test="F", condition_pk=condition,
        reference_kind="primary_component",
        reference_semantics="primary_component_generating_lifetime",
    )
    primary = store.record_interval_metrics(db, _interval_metrics(), scope=base)
    pseudo = store.record_interval_metrics(
        db, _interval_metrics(), scope=replace(
            base, reference_id=reference, assumption_id=assumption,
            reference_kind="pseudo_true_mono",
            reference_version="corrected-v1",
            reference_semantics="pseudo_true_mono_projection",
        ),
    )
    assert primary[0] != pseudo[0]
    reference_set = store.record_interval_metrics(
        db, _interval_metrics(), scope=replace(
            base, reference_id=None, reference_ids=(reference_two, reference),
            assumption_id=assumption, reference_kind="pseudo_true_mono",
            reference_version="corrected-v1",
            reference_semantics="pseudo_true_mono_projection",
        ),
    )
    assert reference_set[0] not in (primary[0], pseudo[0])
    assert db.execute(
        "SELECT COUNT(DISTINCT scope_sha256) FROM benchmark_metrics"
    ).fetchone()[0] == 3
    with pytest.raises(ValueError, match="kind/version"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(
                base, reference_id=reference, assumption_id=assumption,
                reference_kind="pseudo_true_mono",
                reference_version="wrong-version",
                reference_semantics="pseudo_true_mono_projection",
            ),
        )


def test_general_aggregate_source_objects_keep_count_semantics(context):
    db, run, _, model, _, _ = context
    base = _scope(run, model, coverage=None, method=None,
                  method_configuration=None)
    regression = RegressionMetrics(
        mae_ns=0.1, median_absolute_error_ns=0.1, rmse_ns=0.12,
        mean_relative_error=0.05, median_relative_error=0.05, r2=0.9,
    )
    assert len(store.record_regression_metrics(
        db, regression, scope=base, n_attempted=2,
    )) == 6
    robustness = RobustnessMetrics(
        n_samples=2, mae_ns=0.1, median_absolute_error_ns=0.1,
        rmse_ns=0.12, bias_ns=0.0,
        p90_absolute_error_ns=0.2, p95_absolute_error_ns=0.21,
    )
    assert len(store.record_robustness_metrics(
        db, robustness, scope=replace(base, population_key="robustness"),
    )) == 6
    classical = ReconvolutionBenchmarkSummary(
        n_samples=2, n_successful_fits=1, n_failed_fits=1,
        success_rate=0.5, failure_rate=0.5, mae_valid_ns=0.1,
        median_absolute_error_valid_ns=0.1, rmse_valid_ns=0.1,
        mean_runtime_ms=2.0, median_runtime_ms=2.0,
    )
    classical_model = _classical_model(db, "classical-benchmark", objective="least_squares")
    ids = store.record_reconvolution_benchmark_metrics(
        db, classical, scope=replace(base, population_key="classical", model_id=classical_model),
    )
    assert len(ids) == 7
    runtime = db.execute(
        "SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[-1],),
    ).fetchone()
    assert runtime["denominator_kind"] == "finite_runtime_records"
    assert runtime["n_contributing"] is None


def test_scope_selection_is_validated_against_stored_population(context):
    db, run, _, model, _, measurements = context
    scope = _scope(run, model, measurement_ids=measurements)
    with pytest.raises(ValueError, match="membership"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(scope, dataset_key="other-dataset"),
        )
    with pytest.raises(ValueError, match="ordered sequence"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(scope, measurement_ids={*measurements}),
        )
    with pytest.raises(ValueError, match="reference kind"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(scope, reference_kind=None),
        )


def test_ml_scorecard_rows_require_explicit_denominators_and_preserve_legacy_identity(context):
    db, run, _, model, _, _ = context
    interval_row = pd.Series({
        "method": "quantile_gradient_boosting", "test_id": "A",
        "target_reference": "monoexponential_lifetime", "mae_ns": 0.12,
        "nominal_coverage": 0.9, "empirical_coverage": 0.8,
        "coverage_gap": -0.1, "mean_width_ns": 0.4,
        "median_width_ns": 0.38, "interval_score": 0.6,
        "interval_failure_rate": 0.1,
    })
    scope = _scope(run, model)
    db.execute("SAVEPOINT scorecard_aliases")
    ids = store.record_ml_interval_scorecard_row(
        db, interval_row, scope=scope, n_attempted=10,
        n_valid_intervals=9, n_valid_predictions=10,
    )
    assert len(ids) == 7
    mae = db.execute("SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[0],)).fetchone()
    coverage = db.execute("SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[1],)).fetchone()
    assert (mae["n_valid"], mae["n_contributing"], mae["denominator_kind"]) == (
        10, 10, "attempted_observations",
    )
    assert (coverage["n_contributing"], coverage["denominator_kind"]) == (9, "valid_intervals")
    with pytest.raises(ValueError, match="method/test"):
        store.record_ml_interval_scorecard_row(
            db, interval_row, scope=replace(scope, test_id="F"),
            n_attempted=10, n_valid_intervals=9, n_valid_predictions=10,
            on_duplicate="reuse_identical",
        )
    with pytest.raises(ValueError, match="counts"):
        store.record_ml_interval_scorecard_row(
            db, interval_row, scope=scope, n_attempted=10,
            n_valid_intervals=11, n_valid_predictions=10,
        )
    score_row = pd.Series({
        "method": "random_forest_tree_spread", "test_id": "A",
        "target_reference": "monoexponential_lifetime", "mae_ns": 0.12,
        "mean_uncertainty_score": 0.2,
        "error_score_spearman": float("nan"),
        "low_uncertainty_mae_ns": 0.08,
        "high_uncertainty_mae_ns": 0.16, "score_failure_rate": 0.0,
    })
    score_scope = _scope(
        run, model, method="random_forest_tree_spread", coverage=None,
        selection={"tail_fraction": 0.2},
    )
    score_ids = store.record_ml_uncertainty_scorecard_row(
        db, score_row, scope=score_scope, n_attempted=10, n_valid_scores=10,
    )
    assert len(score_ids) == 6
    correlation = db.execute(
        "SELECT * FROM benchmark_metrics WHERE metric_id=?", (score_ids[2],),
    ).fetchone()
    assert correlation["value_status"] == "undefined"
    assert correlation["metric_value"] is None

    # Both legacy entry points must resolve to identical scientific rows and
    # stable keys, including scope hashes, methods, metrics, and denominators.
    query = "SELECT * FROM benchmark_metrics ORDER BY metric_id"
    before = [dict(row) for row in db.execute(query)]
    # Replay from the same database state through the old names, not just reuse
    # rows already inserted by the new names.
    db.execute("ROLLBACK TO scorecard_aliases")
    for legacy, row, metric_scope, counts, expected_ids in (
        (store.record_week9_interval_scorecard_row, interval_row, scope,
         {"n_valid_intervals": 9, "n_valid_predictions": 10}, ids),
        (store.record_week9_score_only_scorecard_row, score_row, score_scope,
         {"n_valid_scores": 10}, score_ids),
    ):
        assert legacy(db, row, scope=metric_scope, n_attempted=10, **counts) == expected_ids
        with pytest.raises(store.PersistenceConflictError, match="duplicate"):
            legacy(db, row, scope=metric_scope, n_attempted=10, **counts)
        assert legacy(
            db, row, scope=metric_scope, n_attempted=10, **counts,
            on_duplicate="reuse_identical",
        ) == expected_ids
    assert [dict(row) for row in db.execute(query)] == before
    db.execute("RELEASE scorecard_aliases")


def test_pseudo_true_references_match_selected_measurement_conditions(context):
    db, run, _, model, _, measurements = context
    assumption = store.record_model_assumption(
        db, assumption_key="population-projection-assumption",
        assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="historical_incomplete",
    )
    own_condition = db.execute(
        "SELECT condition_pk FROM measurements WHERE measurement_id=?", (measurements[0],),
    ).fetchone()[0]
    other_condition = store.record_simulation_condition(
        db, condition_key="population-other-mono", condition_id="r1",
        generating_model="monoexponential", mono_lifetime_ns=2.1,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0,
    )
    unrelated_condition = store.record_simulation_condition(
        db, condition_key="population-unrelated-bi", condition_id="r1",
        generating_model="biexponential", primary_lifetime_ns=1.4,
        secondary_lifetime_ns=3.0, secondary_detected_fraction=0.2,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0,
    )
    references = {}
    for label, condition in (
        ("own", own_condition), ("other", other_condition),
        ("unrelated", unrelated_condition),
    ):
        references[label] = store.record_pseudo_true_reference(
            db, reference_key=f"population-{label}-projection",
            reference_version="v1", condition_pk=condition,
            assumption_id=assumption, lifetime_ns=1.8,
        )
    base = replace(
        _scope(run, model, measurement_ids=measurements),
        reference_kind="pseudo_true_mono", reference_version="v1",
        reference_semantics="pseudo_true_mono_projection", assumption_id=assumption,
        reference_id=references["own"],
    )
    store.record_interval_metrics(db, _interval_metrics(), scope=base)
    with pytest.raises(ValueError, match="selected measurement conditions"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(base, reference_id=references["unrelated"]),
        )
    with pytest.raises(ValueError, match="selected measurement conditions"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(base, reference_id=references["unrelated"]),
            on_duplicate="reuse_identical",
        )

    second_measurement = store.record_measurement(
        db, TCSPCMeasurement(
            time_ns=np.arange(0.0, 2.25, 0.25), values=np.arange(1, 10),
            data_kind=MeasurementDataKind.RAW_COUNTS, sample_id="other-condition-sample",
        ), measurement_key="other-condition-measurement", source_type="synthetic",
        condition_pk=other_condition,
    )
    store.link_run_measurement(
        db, run_id=run, measurement_id=second_measurement, data_role="evaluation",
        dataset_key="dataset-1", test_id="A", regime_id="r1",
    )
    multi = replace(
        base, population_key="two-condition-population",
        measurement_ids=(measurements[0], second_measurement), reference_id=None,
        reference_ids=(references["other"], references["own"]),
    )
    valid = store.record_interval_metrics(db, _interval_metrics(), scope=multi)
    assert store.record_interval_metrics(
        db, _interval_metrics(),
        scope=replace(multi, reference_ids=tuple(reversed(multi.reference_ids))),
        on_duplicate="reuse_identical",
    ) == valid
    for bad_set in ((references["own"],),
                    (references["own"], references["unrelated"])):
        with pytest.raises(ValueError, match="selected measurement conditions"):
            store.record_interval_metrics(
                db, _interval_metrics(), scope=replace(multi, reference_ids=bad_set),
                on_duplicate="reuse_identical",
            )

    experimental = store.record_measurement(
        db, TCSPCMeasurement(
            time_ns=np.arange(0.0, 2.25, 0.25), values=np.arange(1, 10),
            data_kind=MeasurementDataKind.RAW_COUNTS, sample_id="experimental-reference-sample",
        ), measurement_key="experimental-reference-measurement", source_type="experimental",
    )
    trusted = store.record_trusted_lifetime_reference(
        db, reference_key="unrelated-trusted", reference_version="v1",
        measurement_id=experimental, lifetime_ns=1.8,
    )
    with pytest.raises(ValueError, match="outside metric measurement population"):
        store.record_interval_metrics(
            db, _interval_metrics(), scope=replace(
                base, reference_kind="trusted_experimental",
                reference_semantics="trusted_experimental_reference",
                reference_id=trusted, assumption_id=None,
            ),
        )
    store.link_run_measurement(
        db, run_id=run, measurement_id=experimental, data_role="evaluation",
        dataset_key="dataset-1", test_id="A", regime_id="r1",
    )
    with pytest.raises(ValueError, match="selected measurement conditions"):
        store.record_interval_metrics(
            db, replace(_interval_metrics(), n_samples=1, n_valid_intervals=1),
            scope=replace(base, measurement_ids=(experimental,), population_key="experimental"),
        )


def test_classical_aggregate_models_match_source_workflows(context):
    db, run, _, ml_model, _, _ = context
    condition = store.record_simulation_condition(
        db, condition_key="aggregate-model-condition", condition_id="r1",
        generating_model="monoexponential", mono_lifetime_ns=1.7,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.0,
    )
    poisson = _classical_model(db, "aggregate-poisson")
    least_squares = _classical_model(db, "aggregate-least-squares", objective="least_squares")
    scope = replace(
        _scope(run, poisson), population_key="classical-aggregate",
        method_id="repeated_poisson_realizations", condition_pk=condition,
        reference_semantics="generating_mono_and_empirical_repeated_poisson",
    )
    repeated = _repeated_result()
    stored = store.record_repeated_poisson_metrics(db, repeated, scope=scope)
    assert store.record_repeated_poisson_metrics(
        db, repeated, scope=scope, on_duplicate="reuse_identical",
    ) == stored
    inconsistent_covariance = replace(
        repeated, covariance_metrics=replace(
            repeated.covariance_metrics,
            coverage_error=repeated.covariance_metrics.coverage_error + 0.05,
        ),
    )
    with pytest.raises(ValueError, match="coverage/error conflicts"):
        store.record_repeated_poisson_metrics(
            db, inconsistent_covariance, scope=scope, on_duplicate="reuse_identical",
        )
    with pytest.raises(ValueError, match="objective=poisson"):
        store.record_repeated_poisson_metrics(
            db, repeated, scope=replace(scope, model_id=least_squares),
            on_duplicate="reuse_identical",
        )
    incompatible_models = {"ml": ml_model}
    for family in ("baseline", "bayesian"):
        incompatible_models[family] = store.record_model_version(
            db, model_key=f"aggregate-{family}", estimator_name="wrong-reconvolution",
            family=family, configuration={"objective": "poisson"},
        )
    for model_id in incompatible_models.values():
        with pytest.raises(ValueError, match="classical model"):
            store.record_repeated_poisson_metrics(
                db, repeated, scope=replace(scope, model_id=model_id),
                on_duplicate="reuse_identical",
            )
    for suffix, configuration in (("missing", {}), ("unknown", {"objective": "unknown"})):
        invalid = store.record_model_version(
            db, model_key=f"aggregate-objective-{suffix}",
            estimator_name="reconvolution", family="classical",
            configuration=configuration,
        )
        with pytest.raises(ValueError, match="recognized objective"):
            store.record_repeated_poisson_metrics(
                db, repeated, scope=replace(scope, model_id=invalid),
                on_duplicate="reuse_identical",
            )

    summary = ReconvolutionBenchmarkSummary(
        n_samples=2, n_successful_fits=1, n_failed_fits=1,
        success_rate=0.5, failure_rate=0.5, mae_valid_ns=0.1,
        median_absolute_error_valid_ns=0.1, rmse_valid_ns=0.1,
        mean_runtime_ms=2.0, median_runtime_ms=2.0,
    )
    ordinary = replace(scope, population_key="ordinary-classical", condition_pk=None,
                       method_id=None, method_configuration=None, nominal_coverage=None,
                       reference_semantics="generating_mono")
    for model_id in (poisson, least_squares):
        assert len(store.record_reconvolution_benchmark_metrics(
            db, summary, scope=replace(ordinary, model_id=model_id),
        )) == 7
    contradictory_assumption = store.record_model_assumption(
        db, assumption_key="aggregate-least-squares-assumption",
        assumed_decay_model="monoexponential",
        observation_model="least_squares_reconvolution",
        background_convention="per_bin", context_completeness="historical_incomplete",
    )
    with pytest.raises(ValueError, match="objective conflicts with assumption"):
        store.record_reconvolution_benchmark_metrics(
            db, summary, scope=replace(
                ordinary, model_id=poisson, assumption_id=contradictory_assumption,
            ), on_duplicate="reuse_identical",
        )
    with pytest.raises(ValueError, match="classical model"):
        store.record_reconvolution_benchmark_metrics(
            db, summary, scope=replace(ordinary, model_id=ml_model),
            on_duplicate="reuse_identical",
        )


def test_interval_coverage_scope_matches_signed_source_error(context):
    db, run, _, model, _, _ = context
    source = _interval(
        predictions=(1.5, 2.0), lower=(1.2, 1.8), upper=(1.8, 2.2), coverage=0.9,
    )
    metrics = evaluate_prediction_intervals(np.array([1.7, 1.7]), source)
    assert metrics.coverage_error == pytest.approx(metrics.empirical_coverage - 0.9)
    scope = _scope(run, model, population_key="signed-coverage")
    ids = store.record_interval_metrics(db, metrics, scope=scope)
    assert len(ids) == 6
    for wrong_scope, wrong_metrics in (
        (replace(scope, nominal_coverage=0.8), metrics),
        (scope, replace(metrics, coverage_error=metrics.coverage_error + 0.01)),
    ):
        with pytest.raises(ValueError, match="coverage/error conflicts"):
            store.record_interval_metrics(
                db, wrong_metrics, scope=wrong_scope, on_duplicate="reuse_identical",
            )
    roundoff = replace(metrics, coverage_error=metrics.coverage_error + 5e-13)
    assert len(store.record_interval_metrics(
        db, roundoff, scope=replace(scope, population_key="roundoff"),
    )) == 6
    empty = replace(
        metrics, n_valid_intervals=0, empirical_coverage=float("nan"),
        coverage_error=float("nan"), mean_interval_width=float("nan"),
        median_interval_width=float("nan"), mean_interval_score=float("nan"),
        interval_failure_rate=1.0,
    )
    assert len(store.record_interval_metrics(
        db, empty, scope=replace(scope, population_key="empty-coverage"),
    )) == 6
    with pytest.raises(ValueError, match="undefined with no valid intervals"):
        store.record_interval_metrics(
            db, replace(empty, empirical_coverage=0.0),
            scope=replace(scope, population_key="bad-empty-coverage"),
        )
    with pytest.raises(ValueError, match="coverage/error conflicts"):
        store.record_interval_metrics(
            db, replace(metrics, coverage_error=float("inf")),
            scope=replace(scope, population_key="bad-infinite-coverage"),
        )


def test_selective_score_metrics_cannot_claim_nominal_coverage(context):
    db, run, _, model, _, _ = context
    for method_id in ("random_forest_tree_spread", "ml_training_bootstrap"):
        scores = UncertaintyScoreResult(
            prediction=np.array([1.5, 2.0]), uncertainty_score=np.array([0.1, 0.2]),
            method_id=method_id,
        )
        metrics = evaluate_selective_prediction(
            np.array([1.4, 1.7]), scores, rejection_fraction=0.5,
        )
        scope = _scope(
            run, model, method=method_id, coverage=None,
            selection={"rejection_fraction": 0.5},
        )
        ids = store.record_selective_prediction_metrics(
            db, metrics, scope=scope, n_attempted=2,
        )
        assert len(ids) == 4
        assert db.execute(
            "SELECT nominal_coverage FROM benchmark_metrics WHERE metric_id=?", (ids[0],),
        ).fetchone()[0] is None
        with pytest.raises(ValueError, match="no nominal coverage"):
            store.record_selective_prediction_metrics(
                db, metrics, scope=replace(scope, nominal_coverage=0.9),
                n_attempted=2,
            )
        with pytest.raises(ValueError, match="no nominal coverage"):
            store.record_selective_prediction_metrics(
                db, metrics, scope=replace(scope, nominal_coverage=0.9),
                n_attempted=2, on_duplicate="reuse_identical",
            )
    assert len(store.record_interval_metrics(
        db, _interval_metrics(), scope=_scope(
            run, model, population_key="interval-coverage-remains-valid",
        ),
    )) == 6


def test_week9_interval_scorecard_uses_attempted_prediction_mae(context):
    db, run, _, model, _, _ = context
    scope = _scope(run, model, population_key="scorecard-source-denominators")

    class NonfiniteIntervalPredictor:
        def predict(self, features):
            assert features.shape == (2, 1)
            return _interval(
                predictions=(1.5, float("nan")), lower=(1.2, 1.7),
                upper=(1.8, 2.3),
            )

    source = evaluate_frozen_quantile_gradient_boosting(
        SimpleNamespace(
            fitted_estimator=NonfiniteIntervalPredictor(), feature_names=("feature",),
        ),
        pd.DataFrame({"feature": [0.0, 1.0]}), np.array([1.7, 1.7]),
        condition_id="A",
    )
    assert math.isnan(source.median_prediction_mae_ns)
    assert (source.interval_metrics.n_valid_intervals,
            source.interval_metrics.interval_failure_rate) == (1, 0.5)
    row = {
        "method": source.intervals.method_id, "test_id": "A",
        "target_reference": "monoexponential_lifetime",
        "mae_ns": source.median_prediction_mae_ns,
        "nominal_coverage": source.intervals.nominal_coverage,
        "empirical_coverage": source.interval_metrics.empirical_coverage,
        "coverage_gap": source.interval_metrics.coverage_error,
        "mean_width_ns": source.interval_metrics.mean_interval_width,
        "median_width_ns": source.interval_metrics.median_interval_width,
        "interval_score": source.interval_metrics.mean_interval_score,
        "interval_failure_rate": source.interval_metrics.interval_failure_rate,
    }
    ids = store.record_ml_interval_scorecard_row(
        db, row, scope=scope, n_attempted=2,
        n_valid_intervals=1, n_valid_predictions=1,
    )
    mae = db.execute("SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[0],)).fetchone()
    coverage = db.execute("SELECT * FROM benchmark_metrics WHERE metric_id=?", (ids[1],)).fetchone()
    assert (mae["value_status"], mae["metric_value"], mae["n_valid"],
            mae["n_contributing"], mae["denominator_kind"]) == (
                "undefined", None, 1, 2, "attempted_observations",
            )
    assert (coverage["n_valid"], coverage["n_contributing"], coverage["metric_value"]) == (
        1, 1, 1.0,
    )
    with pytest.raises(ValueError, match="prediction/interval counts"):
        store.record_ml_interval_scorecard_row(
            db, row, scope=scope, n_attempted=2,
            n_valid_intervals=1, n_valid_predictions=0,
            on_duplicate="reuse_identical",
        )
    finite_row = dict(row, mae_ns=0.1)
    with pytest.raises(ValueError, match="finite ML scorecard MAE"):
        store.record_ml_interval_scorecard_row(
            db, finite_row, scope=scope, n_attempted=2,
            n_valid_intervals=1, n_valid_predictions=1,
            on_duplicate="reuse_identical",
        )
    empty_row = dict(
        row, empirical_coverage=float("nan"), coverage_gap=float("nan"),
        mean_width_ns=float("nan"), median_width_ns=float("nan"),
        interval_score=float("nan"), interval_failure_rate=1.0, mae_ns=0.0,
    )
    with pytest.raises(ValueError, match="finite ML scorecard MAE"):
        store.record_ml_interval_scorecard_row(
            db, empty_row, scope=replace(scope, population_key="empty-but-finite-mae"),
            n_attempted=2, n_valid_intervals=0, n_valid_predictions=0,
        )


def test_known_score_spread_definitions_follow_source_estimators(context):
    db, run, _, _, point_ids, measurement_ids = context
    features = np.arange(12.0).reshape(-1, 1)
    targets = 1.0 + 0.1 * features[:, 0]
    forest = RandomForestTreeSpreadEstimator(random_state=13)
    forest.pipeline.named_steps["model"].set_params(n_estimators=4)
    forest.fit(features, targets)
    rf_scores = forest.predict(features[:1])
    rf_model = store.record_model_version(
        db, model_key="rf-spread-output", estimator_name="rf-spread-output",
        family="ml", configuration={"n_estimators": 4},
    )
    rf_point = store.record_scalar_prediction(
        db, rf_scores.prediction[0], run_id=run, measurement_id=measurement_ids[0],
        model_id=rf_model, source_result_type="RandomForestTreeSpreadEstimator.predict",
    )
    rf_id = store.record_uncertainty_scores(
        db, rf_scores, result_ids=[rf_point],
        method=ML_UNCERTAINTY_METHODS[rf_scores.method_id],
        method_configuration={}, n_members=4,
    )[0]
    tree_predictions = np.array([
        tree.predict(features[:1])[0]
        for tree in forest.pipeline.named_steps["model"].estimators_
    ])
    assert rf_scores.uncertainty_score[0] == pytest.approx(np.std(tree_predictions, ddof=0))

    bootstrap = BootstrapPredictionSpreadEstimator(
        DummyRegressor(strategy="mean"), n_bootstrap=4, random_state=13,
    )
    bootstrap.fit(features[:3], np.array([1.0, 1.5, 2.0]))
    bootstrap_scores = bootstrap.predict(features[:1])
    assert bootstrap_scores.prediction[0] == 1.5
    bootstrap_id = store.record_uncertainty_scores(
        db, bootstrap_scores, result_ids=point_ids[:1],
        method=ML_UNCERTAINTY_METHODS[bootstrap_scores.method_id],
        method_configuration={}, n_members=4,
    )[0]
    bootstrap_predictions = np.array([
        estimator.predict(features[:1])[0]
        for estimator in bootstrap.bootstrap_estimators_
    ])
    assert bootstrap_scores.uncertainty_score[0] == pytest.approx(
        np.std(bootstrap_predictions, ddof=1),
    )
    configs = []
    for uncertainty_id, expected in (
        (rf_id, ("population_std", 0)),
        (bootstrap_id, ("sample_std", 1)),
    ):
        row = db.execute(
            "SELECT method_config_json, method_config_sha256 FROM uncertainty_results "
            "WHERE uncertainty_id=?", (uncertainty_id,),
        ).fetchone()
        config = json.loads(row["method_config_json"])
        assert (config["spread"], config["ddof"], config["n_members"]) == (*expected, 4)
        configs.append(row["method_config_sha256"])
    assert configs[0] != configs[1]
    for scores, result_id, wrong in (
        (rf_scores, rf_point, {"spread": "sample_std"}),
        (rf_scores, rf_point, {"ddof": 1}),
        (bootstrap_scores, point_ids[0], {"spread": "population_std"}),
        (bootstrap_scores, point_ids[0], {"ddof": 0}),
    ):
        with pytest.raises(ValueError, match="source spread definition"):
            store.record_uncertainty_scores(
                db, scores, result_ids=[result_id],
                method=ML_UNCERTAINTY_METHODS[scores.method_id],
                method_configuration=wrong, n_members=4,
                on_duplicate="reuse_identical",
            )
    assert store.record_uncertainty_scores(
        db, bootstrap_scores, result_ids=point_ids[:1],
        method=ML_UNCERTAINTY_METHODS[bootstrap_scores.method_id],
        method_configuration={"ddof": 1, "spread": "sample_std"}, n_members=4,
        on_duplicate="reuse_identical",
    ) == (bootstrap_id,)
