"""Issue-9 Stage-3 point-result and classical/ML/baseline adapters."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit import persistence
from tcspc_toolkit.baselines import (
    estimate_lifetime_from_mean_arrival, predict_constant_mean_baseline,
)
from tcspc_toolkit.classical_evaluation import ReconvolutionCurveResult
from tcspc_toolkit.fitting import ReconvolutionFitResult
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.ml_evaluation import RegressionBenchmarkResult, RegressionMetrics


@pytest.fixture
def db():
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        persistence.initialize_database(connection)
        connection.row_factory = sqlite3.Row
        yield connection


def _grid():
    return np.arange(0.0, 2.25, 0.25)


def _run(db, key="stage3-run"):
    return persistence.record_run(
        db, run_key=key, run_type="evaluation", status="complete",
        recorded_at_utc="2026-10-01T10:00:00+00:00", origin="new",
        package_version="0.7.0", configuration={"stage": 3},
    )


def _measurement(db, key, *, run_id=None, source_type="synthetic",
                 time_ns=None, data_kind=MeasurementDataKind.RAW_COUNTS):
    time = _grid() if time_ns is None else time_ns
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=np.arange(1, len(time) + 1, dtype=(
            np.int64 if data_kind == MeasurementDataKind.RAW_COUNTS else np.float64
        )),
        data_kind=data_kind, sample_id="shared-local-id",
    )
    measurement_id = persistence.record_measurement(
        db, measurement, measurement_key=key, source_type=source_type,
    )
    if run_id is not None:
        persistence.link_run_measurement(
            db, run_id=run_id, measurement_id=measurement_id,
            data_role="evaluation",
        )
    return measurement_id


def _model(db, key, *, family="ml", name=None, representation_id=None,
           configuration=None):
    if configuration is None:
        configuration = {"method": key}
        if family == "classical":
            configuration["objective"] = "poisson"
    return persistence.record_model_version(
        db, model_key=key, estimator_name=name or key, family=family,
        configuration=configuration,
        representation_id=representation_id,
    )


def _assumption(db, key, *, centre=0.75, observation_model="poisson_reconvolution",
                target_grid=None, source_grid=None, fixed_shift=None,
                shift_bounds=(-0.25, 0.25)):
    target = _grid() if target_grid is None else target_grid
    source = target if source_grid is None else source_grid
    profile = generate_gaussian_irf_profile(
        source, gaussian_centre_ns=centre, gaussian_fwhm_ns=0.4,
    )
    prepared = prepare_irf(
        profile, target, resampling="linear" if source_grid is not None else "none",
    )
    prepared_id = persistence.record_prepared_irf(
        db, prepared, source_key=f"{key}/source", preparation_key=f"{key}/prepared",
    )
    return persistence.record_model_assumption(
        db, assumption_key=key, assumed_decay_model="monoexponential",
        observation_model=observation_model,
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=prepared_id, fixed_temporal_shift_ns=fixed_shift,
        temporal_shift_lower_ns=None if shift_bounds is None else shift_bounds[0],
        temporal_shift_upper_ns=None if shift_bounds is None else shift_bounds[1],
    )


def _lower_fit(**changes):
    fields = dict(
        amplitude=100.0, lifetime=1.7, background=0.2,
        temporal_shift=0.01, fitted_curve=np.arange(1.0, 10.0),
        success=True, optimizer_reported_success=True,
        numerical_validation_passed=True, max_coordinate_descent_nll=0.002,
        recovery_attempted=False, optimizer_status=0,
        optimizer_message="converged", optimizer_nfev=25, optimizer_njev=8,
    )
    fields.update(changes)
    return ReconvolutionFitResult(**fields)


def _curve_fit(**changes):
    fields = dict(
        initial_amplitude=90.0, initial_lifetime_ns=1.5,
        initial_background=0.1, initial_temporal_shift_ns=0.0,
        fitted_amplitude=100.0, fitted_lifetime_ns=1.7,
        fitted_background=0.2, fitted_temporal_shift_ns=0.01,
        optimizer_success=True, valid_fit=True, boundary_hit=False,
        poisson_nll=25.0, poisson_deviance=3.2, runtime_ms=2.5,
        failure_reason=None, exception_message=None,
        numerical_validation_passed=True, max_coordinate_descent_nll=0.002,
        recovery_attempted=False, optimizer_status=0,
        optimizer_message="converged", optimizer_nfev=25, optimizer_njev=8,
    )
    fields.update(changes)
    return ReconvolutionCurveResult(**fields)


def _regression(name, values):
    return RegressionBenchmarkResult(
        estimator_name=name, y_pred=np.asarray(values, dtype=np.float64),
        relative_errors=np.full(len(values), 999.0),
        metrics=RegressionMetrics(
            mae_ns=999.0, median_absolute_error_ns=999.0, rmse_ns=999.0,
            mean_relative_error=999.0, median_relative_error=999.0, r2=-999.0,
        ),
    )


def test_point_identity_null_safe_reuse_and_validity(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "dataset/one", run_id=run_id)
    unlinked_id = _measurement(db, "dataset/unlinked")
    first_model = _model(db, "model/one")
    second_model = _model(db, "model/two")
    first_assumption = _assumption(db, "assumption/one")
    second_assumption = _assumption(db, "assumption/two", centre=1.0)

    def record(model_id, assumption_id=None, analysis_key="default", **extra):
        return persistence.record_scalar_prediction(
            db, 1.5, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            analysis_key=analysis_key, source_result_type="fitted_ml.predict",
            **extra,
        )

    base = record(first_model)
    identities = {
        base,
        record(second_model),
        record(first_model, first_assumption),
        record(first_model, second_assumption),
        record(first_model, None, "alternative"),
    }
    assert len(identities) == 5
    assert record(first_model, on_duplicate="reuse_identical") == base
    with pytest.raises(persistence.PersistenceConflictError, match="duplicate"):
        record(first_model)
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.record_scalar_prediction(
            db, 2.0, run_id=run_id, measurement_id=measurement_id,
            model_id=first_model, source_result_type="fitted_ml.predict",
            on_duplicate="reuse_identical",
        )
    with pytest.raises(ValueError, match="membership"):
        persistence.record_scalar_prediction(
            db, 1.5, run_id=run_id, measurement_id=unlinked_id,
            model_id=first_model, source_result_type="fitted_ml.predict",
        )
    with pytest.raises(ValueError, match="available finite lifetime"):
        persistence.record_scalar_prediction(
            db, None, run_id=run_id, measurement_id=measurement_id,
            model_id=first_model, source_result_type="fitted_ml.predict",
            analysis_key="missing-valid",
        )
    invalid = persistence.record_scalar_prediction(
        db, -0.5, run_id=run_id, measurement_id=measurement_id,
        model_id=first_model, source_result_type="fitted_ml.predict",
        status="failed", is_valid=False, failure_reason="source_rejected",
        analysis_key="invalid-with-estimate",
    )
    row = db.execute(
        "SELECT status, is_valid, lifetime_estimate_ns, failure_reason "
        "FROM estimator_results WHERE result_id = ?", (invalid,),
    ).fetchone()
    assert tuple(row) == ("failed", 0, -0.5, "source_rejected")
    assert db.execute("SELECT COUNT(*) FROM lifetime_references").fetchone()[0] == 0


def test_lower_level_reconvolution_preserves_only_available_fields(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/lower", run_id=run_id)
    model_id = _model(db, "classical/reconvolution", family="classical")
    assumption_id = _assumption(db, "assumption/lower")
    result = _lower_fit()
    result_id = persistence.record_reconvolution_fit(
        db, result, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    point = db.execute(
        "SELECT * FROM estimator_results WHERE result_id = ?", (result_id,),
    ).fetchone()
    details = db.execute(
        "SELECT * FROM fit_details WHERE result_id = ?", (result_id,),
    ).fetchone()
    assert point["lifetime_estimate_ns"] == 1.7
    assert point["source_result_type"].endswith(".ReconvolutionFitResult")
    assert point["runtime_seconds"] is None and point["random_seed_decimal"] is None
    assert details["source_success"] == 1
    assert details["optimizer_reported_success"] == 1
    assert details["numerical_validation_passed"] == 1
    assert details["fitted_amplitude"] == 100.0
    assert details["optimizer_nfev"] == 25 and details["optimizer_njev"] == 8
    for name in (
        "initial_amplitude", "initial_lifetime_ns", "initial_background_per_bin",
        "initial_temporal_shift_ns", "boundary_hit", "poisson_nll",
        "poisson_deviance", "optimizer_seconds", "call_seconds",
    ):
        assert details[name] is None
    assert all(row[1] != "fitted_curve" for row in db.execute("PRAGMA table_info(fit_details)"))
    assert persistence.record_reconvolution_fit(
        db, result, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        on_duplicate="reuse_identical",
    ) == result_id


def test_failed_lower_level_fit_retains_finite_estimate_without_fabricated_metrics(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/lower-failed", run_id=run_id)
    model_id = _model(db, "classical/lower-failed", family="classical")
    assumption_id = _assumption(db, "assumption/lower-failed")
    failed = _lower_fit(
        success=False, optimizer_reported_success=True,
        numerical_validation_passed=False, max_coordinate_descent_nll=np.inf,
        recovery_attempted=True, optimizer_status=-1,
    )
    result_id = persistence.record_reconvolution_fit(
        db, failed, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    point = db.execute("SELECT * FROM estimator_results WHERE result_id = ?", (result_id,)).fetchone()
    details = db.execute("SELECT * FROM fit_details WHERE result_id = ?", (result_id,)).fetchone()
    assert (point["status"], point["is_valid"], point["lifetime_estimate_ns"]) == (
        "failed", 0, 1.7,
    )
    assert point["failure_reason"] is None
    assert (details["source_success"], details["optimizer_reported_success"],
            details["numerical_validation_passed"], details["recovery_attempted"]) == (
        0, 1, 0, 1,
    )
    assert details["optimizer_status"] == -1
    assert details["poisson_nll"] is None and details["poisson_deviance"] is None
    assert json.loads(details["nonfinite_fields_json"]) == {
        "max_coordinate_descent_nll": "+inf",
    }


def test_curve_reconvolution_diagnostics_runtime_and_failure_states(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/curve", run_id=run_id)
    model_id = _model(db, "classical/curve", family="classical")
    assumption_id = _assumption(db, "assumption/curve")
    result_id = persistence.record_reconvolution_fit(
        db, _curve_fit(), run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        irf_model_relation="matched",
    )
    point = db.execute("SELECT * FROM estimator_results WHERE result_id = ?", (result_id,)).fetchone()
    details = db.execute("SELECT * FROM fit_details WHERE result_id = ?", (result_id,)).fetchone()
    assert (point["status"], point["is_valid"], point["irf_model_relation"]) == (
        "available", 1, "matched",
    )
    assert point["runtime_seconds"] == pytest.approx(0.0025)
    assert point["runtime_scope"] == "curve_fit_phase_after_initialization"
    assert point["source_result_type"].endswith(".ReconvolutionCurveResult")
    assert details["initial_lifetime_ns"] == 1.5
    assert details["poisson_nll"] == 25.0
    assert details["poisson_deviance"] == 3.2
    assert details["optimizer_status"] == 0
    assert details["optimizer_message"] == "converged"
    assert details["optimizer_seconds"] is None and details["call_seconds"] is None

    failed = _curve_fit(
        fitted_lifetime_ns=1.9, optimizer_success=True, valid_fit=False,
        numerical_validation_passed=False, recovery_attempted=True,
        boundary_hit=True, failure_reason="poisson_numerical_validation_failed",
    )
    failed_id = persistence.record_reconvolution_fit(
        db, failed, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id, analysis_key="rejected",
    )
    failed_point = db.execute(
        "SELECT status, is_valid, lifetime_estimate_ns, failure_reason "
        "FROM estimator_results WHERE result_id = ?", (failed_id,),
    ).fetchone()
    assert tuple(failed_point) == (
        "failed", 0, 1.9, "poisson_numerical_validation_failed",
    )
    failed_details = db.execute(
        "SELECT source_success, optimizer_reported_success, "
        "numerical_validation_passed, recovery_attempted, boundary_hit "
        "FROM fit_details WHERE result_id = ?", (failed_id,),
    ).fetchone()
    assert tuple(failed_details) == (0, 1, 0, 1, 1)


def test_classical_valid_flag_cannot_hide_nonphysical_fitted_parameters(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/inconsistent", run_id=run_id)
    model_id = _model(db, "classical/inconsistent", family="classical")
    assumption_id = _assumption(db, "assumption/inconsistent")
    for result in (
        _curve_fit(fitted_amplitude=np.nan),
        _curve_fit(fitted_background=-0.1),
        _curve_fit(optimizer_success=False),
    ):
        with pytest.raises(ValueError, match="valid classical fit"):
            persistence.record_reconvolution_fit(
                db, result, run_id=run_id, measurement_id=measurement_id,
                model_id=model_id, assumption_id=assumption_id,
            )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


def test_failed_classical_nonfinite_categories_and_finite_diagnostics(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/nonfinite", run_id=run_id)
    model_id = _model(db, "classical/nonfinite", family="classical")
    assumption_id = _assumption(db, "assumption/nonfinite")
    failed = _curve_fit(
        valid_fit=False, optimizer_success=False, fitted_lifetime_ns=np.nan,
        fitted_amplitude=np.inf, fitted_background=-np.inf,
        poisson_nll=np.inf, poisson_deviance=-np.inf,
        max_coordinate_descent_nll=np.nan, runtime_ms=np.nan,
        failure_reason="fit_exception", exception_message="optimizer blew up",
        numerical_validation_passed=None,
    )
    result_id = persistence.record_reconvolution_fit(
        db, failed, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    point = db.execute("SELECT * FROM estimator_results WHERE result_id = ?", (result_id,)).fetchone()
    details = db.execute("SELECT * FROM fit_details WHERE result_id = ?", (result_id,)).fetchone()
    assert point["status"] == "failed" and point["is_valid"] == 0
    assert point["lifetime_estimate_ns"] is None
    assert point["runtime_seconds"] is None
    assert point["runtime_scope"] == "curve_fit_phase_after_initialization"
    assert json.loads(point["nonfinite_fields_json"]) == {
        "lifetime_estimate_ns": "nan", "runtime_seconds": "nan",
    }
    assert details["fitted_amplitude"] is None
    assert details["fitted_background_per_bin"] is None
    assert details["poisson_nll"] is None and details["poisson_deviance"] is None
    assert details["fitted_temporal_shift_ns"] == 0.01
    assert details["initial_lifetime_ns"] == 1.5
    assert details["exception_message"] == "optimizer blew up"
    assert json.loads(details["nonfinite_fields_json"]) == {
        "fitted_amplitude": "+inf", "fitted_background_per_bin": "-inf",
        "poisson_nll": "+inf", "poisson_deviance": "-inf",
        "max_coordinate_descent_nll": "nan",
    }


def test_classical_reuse_checks_extension_and_composed_write_rolls_back(db, monkeypatch):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/duplicate", run_id=run_id)
    model_id = _model(db, "classical/duplicate", family="classical")
    assumption_id = _assumption(db, "assumption/duplicate")
    result = _curve_fit()
    result_id = persistence.record_reconvolution_fit(
        db, result, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    with pytest.raises(persistence.PersistenceConflictError, match="fit_details"):
        persistence.record_reconvolution_fit(
            db, replace(result, optimizer_nfev=26), run_id=run_id,
            measurement_id=measurement_id, model_id=model_id,
            assumption_id=assumption_id, on_duplicate="reuse_identical",
        )
    assert db.execute(
        "SELECT optimizer_nfev FROM fit_details WHERE result_id = ?", (result_id,),
    ).fetchone()[0] == 25

    original_record_row = persistence._record_row

    def fail_fit_details(*args, **kwargs):
        if kwargs.get("table") == "fit_details":
            raise RuntimeError("injected extension failure")
        return original_record_row(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(persistence, "_record_row", fail_fit_details)
        with pytest.raises(RuntimeError, match="injected extension failure"):
            persistence.record_reconvolution_fit(
                db, result, run_id=run_id, measurement_id=measurement_id,
                model_id=model_id, assumption_id=assumption_id,
                analysis_key="atomic-failure",
            )
    assert db.execute(
        "SELECT COUNT(*) FROM estimator_results WHERE analysis_key = 'atomic-failure'"
    ).fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM fit_details").fetchone()[0] == 1

    db.execute("DELETE FROM fit_details WHERE result_id = ?", (result_id,))
    with pytest.raises(persistence.PersistenceConflictError, match="lacks fit_details"):
        persistence.record_reconvolution_fit(
            db, result, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM fit_details").fetchone()[0] == 0


def test_classical_assumptions_are_explicit_and_irf_relation_is_declared(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "fit/irf-pair", run_id=run_id)
    model_id = _model(db, "classical/pair", family="classical")
    matched = _assumption(db, "assumption/matched", centre=0.75)
    wrong = _assumption(db, "assumption/wrong", centre=1.0)
    first = persistence.record_reconvolution_fit(
        db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=matched, irf_model_relation="matched",
    )
    second = persistence.record_reconvolution_fit(
        db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=wrong,
        irf_model_relation="deliberately_misspecified",
    )
    assert first != second
    assert [tuple(row) for row in db.execute(
        "SELECT assumption_id, irf_model_relation FROM estimator_results ORDER BY result_id"
    )] == [
        (matched, "matched"), (wrong, "deliberately_misspecified"),
    ]
    with pytest.raises(ValueError, match="integer"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=None,
        )
    with pytest.raises(ValueError, match="classical"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
            model_id=_model(db, "ml/not-classical"), assumption_id=matched,
        )


def test_classical_fit_rejects_incompatible_measurement_and_assumption(db):
    run_id = _run(db)
    raw_id = _measurement(db, "fit/raw", run_id=run_id)
    processed = TCSPCMeasurement(
        time_ns=_grid(), values=np.arange(1.0, 10.0),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    processed_id = persistence.record_measurement(
        db, processed, measurement_key="fit/processed",
        source_type="experimental",
    )
    persistence.link_run_measurement(
        db, run_id=run_id, measurement_id=processed_id,
        data_role="evaluation",
    )
    model_id = _model(db, "classical/physical-guard", family="classical")
    mono_id = _assumption(db, "assumption/mono-physical-guard")
    with pytest.raises(ValueError, match="raw-count measurement"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=processed_id,
            model_id=model_id, assumption_id=mono_id,
        )

    prepared_id = db.execute(
        "SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
        (mono_id,),
    ).fetchone()[0]
    biexponential_id = persistence.record_model_assumption(
        db, assumption_key="assumption/bi-physical-guard",
        assumed_decay_model="biexponential",
        observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=prepared_id, fixed_temporal_shift_ns=0.0,
    )
    with pytest.raises(ValueError, match="monoexponential model assumption"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=raw_id,
            model_id=model_id, assumption_id=biexponential_id,
        )

    missing_irf_id = persistence.record_model_assumption(
        db, assumption_key="assumption/incomplete-physical-guard",
        assumed_decay_model="monoexponential",
        observation_model="least_squares_reconvolution",
        background_convention="per_bin", context_completeness="complete",
    )
    least_squares_model = _model(
        db, "classical/missing-irf", family="classical",
        configuration={"objective": "least_squares"},
    )
    with pytest.raises(ValueError, match="requires a prepared IRF"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=raw_id,
            model_id=least_squares_model, assumption_id=missing_irf_id,
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


@pytest.mark.parametrize(
    "objective,data_kind,observation_model,accepted",
    [
        ("poisson", MeasurementDataKind.RAW_COUNTS, "poisson_reconvolution", True),
        ("poisson", MeasurementDataKind.PROCESSED_INTENSITY, "poisson_reconvolution", False),
        ("least_squares", MeasurementDataKind.PROCESSED_INTENSITY,
         "least_squares_reconvolution", True),
        ("least_squares", MeasurementDataKind.RAW_COUNTS,
         "least_squares_reconvolution", True),
    ],
)
def test_classical_objective_controls_measurement_semantics(
    db, objective, data_kind, observation_model, accepted,
):
    run_id = _run(db)
    measurement_id = _measurement(db, "objective/measurement", run_id=run_id,
                                  data_kind=data_kind)
    model_id = _model(db, "objective/model", family="classical",
                      configuration={"objective": objective, "method": "reconvolution"})
    assumption_id = _assumption(db, "objective/assumption",
                                observation_model=observation_model)
    record = lambda: persistence.record_reconvolution_fit(
        db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    if accepted:
        result_id = record()
        model = db.execute(
            "SELECT configuration_json FROM model_versions WHERE model_id = ?", (model_id,),
        ).fetchone()
        result = db.execute(
            "SELECT execution_json FROM estimator_results WHERE result_id = ?", (result_id,),
        ).fetchone()
        assert json.loads(model[0])["objective"] == objective
        assert "objective" not in json.loads(result[0])
    else:
        with pytest.raises(ValueError, match="raw-count measurement"):
            record()
        assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


@pytest.mark.parametrize("configuration", [
    {"method": "reconvolution"}, {"objective": "unknown"},
])
def test_classical_fit_requires_declared_recognized_model_objective(db, configuration):
    run_id = _run(db)
    measurement_id = _measurement(db, "objective/missing-observation", run_id=run_id)
    model_id = _model(db, "objective/missing-model", family="classical",
                      configuration=configuration)
    assumption_id = _assumption(db, "objective/missing-assumption")
    with pytest.raises(ValueError, match="recognized objective"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            execution={"objective": "poisson"},
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


def test_classical_objective_changes_reusable_model_configuration_identity(db):
    poisson_id = _model(db, "objective/identity-poisson", family="classical",
                        configuration={"objective": "poisson"})
    least_squares_id = _model(db, "objective/identity-least-squares",
                              family="classical",
                              configuration={"objective": "least_squares"})
    poisson = db.execute(
        "SELECT configuration_json, configuration_sha256 FROM model_versions "
        "WHERE model_id = ?", (poisson_id,),
    ).fetchone()
    least_squares = db.execute(
        "SELECT configuration_json, configuration_sha256 FROM model_versions "
        "WHERE model_id = ?", (least_squares_id,),
    ).fetchone()
    assert json.loads(poisson[0])["objective"] == "poisson"
    assert json.loads(least_squares[0])["objective"] == "least_squares"
    assert poisson[1] != least_squares[1]


@pytest.mark.parametrize("objective,observation_model", [
    ("poisson", "least_squares_reconvolution"),
    ("least_squares", "poisson_reconvolution"),
])
def test_classical_fit_rejects_explicit_objective_assumption_conflict(
    db, objective, observation_model,
):
    run_id = _run(db)
    measurement_id = _measurement(db, "objective/conflict-observation", run_id=run_id)
    model_id = _model(db, "objective/conflict-model", family="classical",
                      configuration={"objective": objective})
    assumption_id = _assumption(db, "objective/conflict-assumption",
                                observation_model=observation_model)
    with pytest.raises(ValueError, match="objective conflicts"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


def test_classical_objective_conflict_is_checked_before_identical_reuse(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "objective/reuse-observation", run_id=run_id)
    model_id = _model(db, "objective/reuse-model", family="classical")
    assumption_id = _assumption(db, "objective/reuse-assumption")
    result = _lower_fit()
    result_id = persistence.record_reconvolution_fit(
        db, result, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    db.execute(
        "UPDATE model_assumptions SET observation_model = 'least_squares_reconvolution' "
        "WHERE assumption_id = ?", (assumption_id,),
    )
    with pytest.raises(ValueError, match="objective conflicts"):
        persistence.record_reconvolution_fit(
            db, result, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1
    assert db.execute("SELECT result_id FROM fit_details").fetchone()[0] == result_id


@pytest.mark.parametrize("shift,accepted", [
    (0.01, True), (0.0100000005, True), (0.011, False),
])
def test_valid_classical_fit_respects_fixed_shift(db, shift, accepted):
    run_id = _run(db)
    measurement_id = _measurement(db, "shift/fixed-observation", run_id=run_id)
    model_id = _model(db, "shift/fixed-model", family="classical")
    assumption_id = _assumption(db, "shift/fixed-assumption",
                                fixed_shift=0.01, shift_bounds=None)
    record = lambda: persistence.record_reconvolution_fit(
        db, _lower_fit(temporal_shift=shift), run_id=run_id,
        measurement_id=measurement_id, model_id=model_id, assumption_id=assumption_id,
    )
    if accepted:
        assert record() > 0
    else:
        with pytest.raises(ValueError, match="fitted shift conflicts"):
            record()


@pytest.mark.parametrize("shift,accepted", [
    (-0.25, True), (0.25, True), (0.01, True), (-0.26, False), (0.26, False),
])
def test_valid_classical_fit_respects_bounded_shift(db, shift, accepted):
    run_id = _run(db)
    measurement_id = _measurement(db, "shift/bounded-observation", run_id=run_id)
    model_id = _model(db, "shift/bounded-model", family="classical")
    assumption_id = _assumption(db, "shift/bounded-assumption")
    record = lambda: persistence.record_reconvolution_fit(
        db, _curve_fit(fitted_temporal_shift_ns=shift), run_id=run_id,
        measurement_id=measurement_id, model_id=model_id, assumption_id=assumption_id,
    )
    if accepted:
        assert record() > 0
    else:
        with pytest.raises(ValueError, match="fitted shift lies outside"):
            record()


def test_failed_classical_shifts_and_incomplete_historical_policy_are_preserved(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "shift/failed-observation", run_id=run_id)
    model_id = _model(db, "shift/failed-model", family="classical")
    fixed_id = _assumption(db, "shift/failed-fixed", fixed_shift=0.01,
                           shift_bounds=None)
    bounded_id = _assumption(db, "shift/failed-bounded")
    for assumption_id, result, key in (
        (fixed_id, _lower_fit(success=False, temporal_shift=0.5), "failed-fixed"),
        (bounded_id, _curve_fit(valid_fit=False, fitted_temporal_shift_ns=0.5),
         "failed-bounded"),
    ):
        result_id = persistence.record_reconvolution_fit(
            db, result, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id, analysis_key=key,
        )
        point = db.execute(
            "SELECT status, is_valid FROM estimator_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        details = db.execute(
            "SELECT fitted_temporal_shift_ns FROM fit_details WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        assert tuple(point) == ("failed", 0)
        assert details[0] == 0.5

    historical_id = persistence.record_model_assumption(
        db, assumption_key="shift/historical", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="historical_incomplete",
    )
    result_id = persistence.record_reconvolution_fit(
        db, _lower_fit(temporal_shift=0.5), run_id=run_id,
        measurement_id=measurement_id, model_id=model_id, assumption_id=historical_id,
    )
    assert tuple(db.execute(
        "SELECT fixed_temporal_shift_ns, temporal_shift_lower_ns, "
        "temporal_shift_upper_ns FROM model_assumptions WHERE assumption_id = ?",
        (historical_id,),
    ).fetchone()) == (None, None, None)
    assert db.execute(
        "SELECT fitted_temporal_shift_ns FROM fit_details WHERE result_id = ?",
        (result_id,),
    ).fetchone()[0] == 0.5


def test_invalid_classical_shift_is_checked_before_identical_reuse(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "shift/reuse-observation", run_id=run_id)
    model_id = _model(db, "shift/reuse-model", family="classical")
    assumption_id = _assumption(db, "shift/reuse-assumption",
                                fixed_shift=0.01, shift_bounds=None)
    result = _lower_fit()
    result_id = persistence.record_reconvolution_fit(
        db, result, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    with pytest.raises(ValueError, match="fitted shift conflicts"):
        persistence.record_reconvolution_fit(
            db, replace(result, temporal_shift=0.02), run_id=run_id,
            measurement_id=measurement_id, model_id=model_id,
            assumption_id=assumption_id, on_duplicate="reuse_identical",
        )
    assert db.execute(
        "SELECT fitted_temporal_shift_ns FROM fit_details WHERE result_id = ?",
        (result_id,),
    ).fetchone()[0] == 0.01


def _interior_changed_grid():
    grid = _grid().copy()
    grid[4] += 1e-10  # Approximate uniformity still passes; canonical hash differs.
    return grid


@pytest.mark.parametrize("source_grid", [None, np.arange(-0.5, 2.75, 0.125)])
def test_classical_prepared_irf_accepts_matching_target_grid(db, source_grid):
    run_id = _run(db)
    measurement_id = _measurement(db, "grid/matching-observation", run_id=run_id)
    model_id = _model(db, "grid/matching-model", family="classical")
    assumption_id = _assumption(db, "grid/matching-assumption",
                                source_grid=source_grid)
    result_id = persistence.record_reconvolution_fit(
        db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    assert result_id > 0
    if source_grid is not None:
        source_hash, target_hash = db.execute(
            "SELECT s.source_grid_sha256, p.time_grid_sha256 FROM model_assumptions AS a "
            "JOIN prepared_irfs AS p ON p.prepared_irf_id = a.prepared_irf_id "
            "JOIN irf_sources AS s ON s.irf_source_id = p.irf_source_id "
            "WHERE a.assumption_id = ?", (assumption_id,),
        ).fetchone()
        assert source_hash != target_hash


@pytest.mark.parametrize("target_grid", [
    np.arange(0.0, 2.0, 0.25),
    0.1 + np.arange(9) * 0.2,
    _grid() + 0.25,
    _interior_changed_grid(),
], ids=["bin-count", "shifted-and-respaced", "different-start", "hash-only"])
def test_classical_prepared_irf_rejects_different_target_grid(db, target_grid):
    run_id = _run(db)
    measurement_id = _measurement(db, "grid/mismatch-observation", run_id=run_id)
    model_id = _model(db, "grid/mismatch-model", family="classical")
    assumption_id = _assumption(db, "grid/mismatch-assumption",
                                target_grid=target_grid)
    if len(target_grid) == len(_grid()) and np.array_equal(target_grid[:2], _grid()[:2]):
        measurement_grid = db.execute(
            "SELECT n_bins, time_start_ns, time_step_ns, time_grid_sha256 "
            "FROM measurements WHERE measurement_id = ?", (measurement_id,),
        ).fetchone()
        prepared_grid = db.execute(
            "SELECT n_bins, time_start_ns, time_step_ns, time_grid_sha256 "
            "FROM prepared_irfs WHERE prepared_irf_id = "
            "(SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?)",
            (assumption_id,),
        ).fetchone()
        assert tuple(measurement_grid[:3]) == tuple(prepared_grid[:3])
        assert measurement_grid[3] != prepared_grid[3]
    with pytest.raises(ValueError, match="prepared IRF target grid"):
        persistence.record_reconvolution_fit(
            db, _lower_fit(), run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


def test_classical_prepared_irf_grid_is_checked_before_identical_reuse(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "grid/reuse-observation", run_id=run_id)
    model_id = _model(db, "grid/reuse-model", family="classical")
    assumption_id = _assumption(db, "grid/reuse-assumption")
    other_id = _assumption(db, "grid/reuse-other", target_grid=_grid() + 0.25)
    result = _lower_fit()
    result_id = persistence.record_reconvolution_fit(
        db, result, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
    )
    other_prepared_id = db.execute(
        "SELECT prepared_irf_id FROM model_assumptions WHERE assumption_id = ?",
        (other_id,),
    ).fetchone()[0]
    db.execute(
        "UPDATE model_assumptions SET prepared_irf_id = ? WHERE assumption_id = ?",
        (other_prepared_id, assumption_id),
    )
    with pytest.raises(ValueError, match="prepared IRF target grid"):
        persistence.record_reconvolution_fit(
            db, result, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, assumption_id=assumption_id,
            on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT result_id FROM fit_details").fetchone()[0] == result_id


def test_regression_batch_preserves_order_finite_values_and_model_identity(db):
    run_id = _run(db)
    ids = [_measurement(db, f"ml/obs-{index}", run_id=run_id) for index in range(3)]
    model_id = _model(
        db, "rf/trained", family="ml", name="random_forest",
        representation_id="feature-set-v2",
        configuration={"n_estimators": 100, "random_seed": 2718},
    )
    result = _regression("random_forest", [2.5, 0.0, -0.25])
    ordered = [ids[2], ids[0], ids[1]]
    result_ids = persistence.record_regression_predictions(
        db, result, run_id=run_id, measurement_ids=ordered, model_id=model_id,
    )
    assert len(result_ids) == 3
    rows = db.execute(
        "SELECT r.measurement_id, r.lifetime_estimate_ns, r.is_valid, "
        "r.assumption_id, r.runtime_seconds, r.random_seed_decimal, "
        "r.source_result_type, r.execution_json, m.representation_id "
        "FROM estimator_results AS r JOIN model_versions AS m ON m.model_id = r.model_id "
        "WHERE r.run_id = ? ORDER BY r.result_id", (run_id,),
    ).fetchall()
    assert [(row[0], row[1], row[2]) for row in rows] == [
        (ids[2], 2.5, 1), (ids[0], 0.0, 1), (ids[1], -0.25, 1),
    ]
    assert [json.loads(row[7])["prediction_index"] for row in rows] == [0, 1, 2]
    assert all(row[3] is None and row[4] is None and row[5] is None for row in rows)
    assert all(row[6].endswith(".RegressionBenchmarkResult") for row in rows)
    assert all(row[8] == "feature-set-v2" for row in rows)
    assert db.execute("SELECT COUNT(*) FROM fit_details").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM benchmark_metrics").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM lifetime_references").fetchone()[0] == 0
    assert persistence.record_regression_predictions(
        db, result, run_id=run_id, measurement_ids=ordered, model_id=model_id,
        on_duplicate="reuse_identical",
    ) == result_ids


def test_regression_batch_rejects_misalignment_and_nonfinite_predictions(db):
    run_id = _run(db)
    ids = [_measurement(db, f"ml/invalid-{index}", run_id=run_id) for index in range(2)]
    model_id = _model(db, "ml/ridge", name="ridge")
    result = _regression("ridge", [1.0, 2.0])
    with pytest.raises(ValueError, match="length"):
        persistence.record_regression_predictions(
            db, result, run_id=run_id, measurement_ids=ids[:1], model_id=model_id,
        )
    with pytest.raises(ValueError, match="must not repeat"):
        persistence.record_regression_predictions(
            db, result, run_id=run_id, measurement_ids=[ids[0], ids[0]],
            model_id=model_id,
        )
    with pytest.raises(ValueError, match="ordered sequence"):
        persistence.record_regression_predictions(
            db, result, run_id=run_id, measurement_ids=set(ids), model_id=model_id,
        )
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            persistence.record_regression_predictions(
                db, _regression("ridge", [1.0, bad]),
                run_id=run_id, measurement_ids=ids, model_id=model_id,
            )
    with pytest.raises(ValueError, match="estimator_name"):
        persistence.record_regression_predictions(
            db, _regression("different-estimator", [1.0, 2.0]),
            run_id=run_id, measurement_ids=ids, model_id=model_id,
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0


def test_regression_batch_conflict_rolls_back_all_new_rows(db):
    run_id = _run(db)
    ids = [_measurement(db, f"ml/atomic-{index}", run_id=run_id) for index in range(2)]
    model_id = _model(db, "ml/atomic", name="ridge")
    old_id = persistence.record_scalar_prediction(
        db, 9.0, run_id=run_id, measurement_id=ids[1], model_id=model_id,
        source_result_type="fitted_ml.predict",
    )
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.record_regression_predictions(
            db, _regression("ridge", [1.0, 2.0]),
            run_id=run_id, measurement_ids=ids, model_id=model_id,
            on_duplicate="reuse_identical",
        )
    rows = db.execute(
        "SELECT result_id, measurement_id, lifetime_estimate_ns "
        "FROM estimator_results"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(old_id, ids[1], 9.0)]
    with pytest.raises(persistence.PersistenceConflictError, match="duplicate"):
        persistence.record_regression_predictions(
            db, _regression("ridge", [1.0, 2.0]),
            run_id=run_id, measurement_ids=ids, model_id=model_id,
        )
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 1


def test_regression_batch_missing_membership_rolls_back_nested_savepoint(db):
    run_id = _run(db)
    linked_id = _measurement(db, "ml/nested-linked", run_id=run_id)
    unlinked_id = _measurement(db, "ml/nested-unlinked")
    model_id = _model(db, "ml/nested", name="ridge")
    with persistence.transaction(db):
        outer_id = persistence.record_scalar_prediction(
            db, 1.0, run_id=run_id, measurement_id=linked_id,
            model_id=model_id, source_result_type="fitted_ml.predict",
            analysis_key="outer",
        )
        with pytest.raises(ValueError, match="membership"):
            persistence.record_regression_predictions(
                db, _regression("ridge", [2.0, 3.0]),
                run_id=run_id, measurement_ids=[linked_id, unlinked_id],
                model_id=model_id,
            )
    assert [tuple(row) for row in db.execute(
        "SELECT result_id, analysis_key FROM estimator_results"
    )] == [(outer_id, "outer")]


def test_current_baseline_arrays_use_shared_result_table(db):
    run_id = _run(db)
    ids = [_measurement(db, f"baseline/obs-{index}", run_id=run_id) for index in range(2)]
    constant_model = _model(
        db, "baseline/constant", family="baseline", name="constant_mean",
    )
    mean_model = _model(
        db, "baseline/arrival", family="baseline", name="mean_arrival_time",
    )
    constant = predict_constant_mean_baseline(
        y_train=np.array([1.0, 2.0, 3.0]), n_predictions=2,
    )
    mean_arrival = estimate_lifetime_from_mean_arrival(
        mean_arrival_time_ns=np.array([1.0, 0.5]),
        peak_time_ns=np.array([1.0, 1.0]),
    )
    persistence.record_baseline_predictions(
        db, constant, baseline_name="constant_mean", run_id=run_id,
        measurement_ids=ids, model_id=constant_model,
    )
    persistence.record_baseline_predictions(
        db, mean_arrival, baseline_name="mean_arrival_time", run_id=run_id,
        measurement_ids=ids, model_id=mean_model,
    )
    assert [tuple(row) for row in db.execute(
        "SELECT model_id, lifetime_estimate_ns, is_valid, source_result_type "
        "FROM estimator_results ORDER BY result_id"
    )] == [
        (constant_model, 2.0, 1, "tcspc_toolkit.baselines.predict_constant_mean_baseline"),
        (constant_model, 2.0, 1, "tcspc_toolkit.baselines.predict_constant_mean_baseline"),
        (mean_model, 0.0, 1, "tcspc_toolkit.baselines.estimate_lifetime_from_mean_arrival"),
        (mean_model, -0.5, 1, "tcspc_toolkit.baselines.estimate_lifetime_from_mean_arrival"),
    ]
    assert db.execute("SELECT COUNT(*) FROM fit_details").fetchone()[0] == 0


def test_baseline_regression_result_uses_same_batch_adapter(db):
    run_id = _run(db)
    ids = [_measurement(db, f"baseline/result-{index}", run_id=run_id) for index in range(2)]
    model_id = _model(
        db, "baseline/result", family="baseline", name="constant_mean",
    )
    result_ids = persistence.record_regression_predictions(
        db, _regression("constant_mean", [1.5, 1.5]),
        run_id=run_id, measurement_ids=ids, model_id=model_id,
    )
    assert len(result_ids) == 2
    assert all(row[0] == 1.5 for row in db.execute(
        "SELECT lifetime_estimate_ns FROM estimator_results ORDER BY result_id"
    ))
    assert db.execute("SELECT COUNT(*) FROM fit_details").fetchone()[0] == 0


def test_scalar_experimental_ml_prediction_runtime_seed_and_nonfinite_failure(db):
    run_id = _run(db)
    measurement_id = _measurement(
        db, "experimental/one", run_id=run_id, source_type="experimental",
    )
    model_id = _model(db, "ml/fitted-rf", family="ml")
    malicious_key = "prediction'); DROP TABLE estimator_results; --"
    result_id = persistence.record_scalar_prediction(
        db, 0.0, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, source_result_type="fitted_rf.predict",
        analysis_key=malicious_key, runtime_seconds=0.003,
        runtime_scope="single_prediction_call", random_seed=2**64 - 1,
        execution={"instrument": "independent laboratory"},
    )
    row = db.execute(
        "SELECT analysis_key, assumption_id, lifetime_estimate_ns, "
        "random_seed_decimal, runtime_seconds, runtime_scope, execution_json "
        "FROM estimator_results WHERE result_id = ?", (result_id,),
    ).fetchone()
    assert row[0:6] == (
        malicious_key, None, 0.0, str(2**64 - 1), 0.003,
        "single_prediction_call",
    )
    assert json.loads(row[6]) == {"instrument": "independent laboratory"}
    assert db.execute("SELECT COUNT(*) FROM simulation_conditions").fetchone()[0] == 0
    failed_id = persistence.record_scalar_prediction(
        db, np.inf, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, source_result_type="fitted_rf.predict",
        analysis_key="failed", status="failed", is_valid=False,
        failure_reason="nonfinite_output",
    )
    failed = db.execute(
        "SELECT lifetime_estimate_ns, nonfinite_fields_json FROM estimator_results "
        "WHERE result_id = ?", (failed_id,),
    ).fetchone()
    assert failed[0] is None
    assert json.loads(failed[1]) == {"lifetime_estimate_ns": "+inf"}
    with pytest.raises(ValueError, match="available finite lifetime"):
        persistence.record_scalar_prediction(
            db, np.nan, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, source_result_type="fitted_rf.predict",
            analysis_key="invalid-valid",
        )


def test_prediction_relation_requires_explicit_assumption(db):
    run_id = _run(db)
    measurement_id = _measurement(db, "ml/irf-relation", run_id=run_id)
    model_id = _model(db, "ml/no-assumption")
    with pytest.raises(ValueError, match="requires a physical assumption"):
        persistence.record_scalar_prediction(
            db, 1.5, run_id=run_id, measurement_id=measurement_id,
            model_id=model_id, source_result_type="fitted_ml.predict",
            irf_model_relation="matched",
        )
    assumption_id = _assumption(db, "assumption/explicit-ml")
    result_id = persistence.record_scalar_prediction(
        db, 1.5, run_id=run_id, measurement_id=measurement_id,
        model_id=model_id, assumption_id=assumption_id,
        source_result_type="fitted_ml.predict", irf_model_relation="matched",
    )
    assert db.execute(
        "SELECT assumption_id FROM estimator_results WHERE result_id = ?", (result_id,),
    ).fetchone()[0] == assumption_id
