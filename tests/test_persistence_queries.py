"""Stage-6 retrieval of persisted facts; no fitting, training or MCMC."""

from __future__ import annotations

import builtins
import hashlib
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit import persistence as store
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, SampledIRF, TCSPCMeasurement


def _insert(db, table, **values):
    # Only this deterministic fixture calls this helper; never caller SQL.
    return db.execute(
        f"INSERT INTO {table} ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
        tuple(values.values()),
    ).lastrowid


def _dicts(result):
    assert len(result.columns) == len(set(result.columns))
    return [dict(zip(result.columns, row)) for row in result.rows]


@pytest.fixture
def populated(tmp_path):
    path = tmp_path / "query.sqlite"
    store.initialize_database(path)
    with closing(store.connect_database(path)) as db:
        runs = [store.record_run(
            db, run_key=key, run_type="evaluation", status="complete",
            recorded_at_utc="2026-10-02T10:00:00+00:00", origin="new", package_version="0.7.0",
            configuration={"query_fixture": True}, profile="deterministic", protocol_id="week9",
        ) for key in ("run-z", "run-a", "calibration")]
        artifact = store.record_artifact(
            db, artifact_key="chain", path=tmp_path / "not-retained.nc", path_base="absolute",
            artifact_kind="posterior_chain", format="netcdf", sha256="a" * 64,
            producing_run_id=runs[0],
        )
        predictive_artifact = store.record_artifact(
            db, artifact_key="predictive", path=tmp_path / "not-retained.npz", path_base="absolute",
            artifact_kind="posterior_predictive_samples", format="npz", sha256="b" * 64,
            producing_run_id=runs[0],
        )
        time = np.arange(0.0, 2.25, 0.25)
        profile = generate_gaussian_irf_profile(time, gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4)
        source_id = store.record_irf_source(db, profile, source_key="source")
        prepared_ids = [store.record_prepared_irf(
            db, prepare_irf(profile, grid), irf_source_id=source_id, preparation_key=key,
        ) for key, grid in (("prepared-z", time), ("prepared-a", time))]
        condition = store.record_simulation_condition(
            db, condition_key="condition", condition_id="local-regime",
            generating_model="monoexponential", mono_lifetime_ns=2.0,
            signal_photon_count=1000, background_per_bin=0.2, true_temporal_shift_ns=0.0,
            generating_irf_id=prepared_ids[0],
        )
        raw = TCSPCMeasurement(time, np.arange(1, 10), MeasurementDataKind.RAW_COUNTS,
                               sample_id="repeated-local-id", metadata={"instrument": "fixture"})
        measured = store.record_measurement(
            db, raw, measurement_key="synthetic-z", source_type="synthetic", condition_pk=condition,
            origin_run_id=runs[0], observation_seed=2**64 - 1,
        )
        experimental = TCSPCMeasurement(
            time, np.linspace(0.1, 0.9, 9), MeasurementDataKind.PROCESSED_INTENSITY,
            sample_id="repeated-local-id", irf=SampledIRF(time, profile.values),
            provenance={"file": "experimental.csv"},
        )
        experiment = store.record_measurement(
            db, experimental, measurement_key="experimental-a", attached_irf_source_key="attached",
        )
        unlinked = store.record_measurement(db, experimental, measurement_key="unlinked",
                                           attached_irf_source_id=db.execute(
                                               "SELECT attached_irf_source_id FROM measurements WHERE measurement_id = ?",
                                               (experiment,),
                                           ).fetchone()[0])
        for run, measurement, test, dataset in (
            (runs[0], measured, "A", "dataset-one"), (runs[1], measured, "F", "dataset-two"),
            (runs[0], experiment, "A", "dataset-one"),
        ):
            store.link_run_measurement(
                db, run_id=run, measurement_id=measurement, data_role="evaluation",
                test_id=test, regime_id="same-label", dataset_key=dataset,
            )
        models = {family: store.record_model_version(
            db, model_key=family, estimator_name=f"{family}-estimator", family=family,
            configuration={"fixture": family, **({"objective": "poisson"} if family == "classical" else {})},
        ) for family in ("classical", "ml", "baseline", "bayesian")}
        assumptions = [store.record_model_assumption(
            db, assumption_key=key, assumed_decay_model="monoexponential",
            observation_model="poisson_reconvolution", background_convention="per_bin",
            context_completeness="complete", prepared_irf_id=prepared, fixed_temporal_shift_ns=0.0,
        ) for key, prepared in zip(("matched", "alternative"), prepared_ids)]
        points = []
        # Direct stored-row fixtures exercise retrieval independently of adapter algorithms.
        for family, measurement, assumption, value, valid, relation in (
            ("classical", measured, assumptions[0], 1.8, 1, "matched"),
            ("classical", measured, assumptions[1], 1.9, 1, "deliberately_misspecified"),
            ("ml", measured, None, 2.1, 1, "unspecified"),
            ("ml", experiment, None, 2.2, 1, "unspecified"),
            ("baseline", experiment, None, 2.3, 1, "unspecified"),
            ("bayesian", measured, assumptions[0], 1.85, 0, "matched"),
        ):
            points.append(_insert(
                db, "estimator_results", run_id=runs[0], measurement_id=measurement,
                model_id=models[family], assumption_id=assumption,
                point_summary="posterior_median" if family == "bayesian" else "lifetime_ns",
                status="available", is_valid=valid, irf_model_relation=relation,
                source_result_type="deterministic_stored_fixture", lifetime_estimate_ns=value,
                random_seed_decimal=str(2**64 - 1),
            ))
        _insert(db, "fit_details", result_id=points[0], source_success=1,
                optimizer_reported_success=1, valid_fit=1, numerical_validation_passed=1,
                fitted_amplitude=100.0, fitted_background_per_bin=0.2, fitted_temporal_shift_ns=0.0,
                boundary_hit=0, poisson_nll=7.0, poisson_deviance=0.4,
                optimizer_status=0, optimizer_nfev=12, optimizer_message="stored success")
        uncertainties = []
        for result_id, method, kind, extra in (
            (points[2], "quantile_gradient_boosting", "prediction_interval",
             {"interval_kind": "quantile", "nominal_coverage": 0.9, "lower_ns": 1.5,
              "upper_ns": 2.4, "calibration_scope": "none"}),
            (points[2], "conformalized_quantile_gradient_boosting", "prediction_interval",
             {"interval_kind": "conformal", "nominal_coverage": 0.9, "lower_ns": 1.3,
              "upper_ns": 2.6, "calibration_run_id": runs[2], "calibration_scope": "held_out"}),
            (points[2], "random_forest_tree_spread", "uncertainty_score",
             {"uncertainty_score": 0.0, "n_requested": 20, "n_valid": 20}),
            (points[0], "poisson_local_covariance", "covariance_summary", {"reported_std_ns": 0.1}),
        ):
            config = json.dumps({"method": method}, sort_keys=True)
            uncertainties.append(_insert(
                db, "uncertainty_results", result_id=result_id, method_id=method, output_kind=kind,
                method_config_json=config, method_config_sha256=hashlib.sha256(config.encode()).hexdigest(),
                interpretation="stored method-conditional output", is_valid=1,
                **({"calibration_scope": "none"} | extra),
            ))
        parameter_json = store._canonical_json({
            "credible_probability": 0.9,
            "parameters": [{"name": "lifetime_ns", "median": 1.85, "mean": 1.9,
                            "credible_lower": 1.6, "credible_upper": 2.2}],
            "diagnostic_parameter_order": ["amplitude", "lifetime_ns", "background_per_bin"],
            "sampling_counts": {"n_walkers": 10, "requested_warmup_steps": 8,
                                "valid_retained_samples": None},
        })
        _insert(db, "bayesian_summaries", result_id=points[-1], sampling_status="success",
                diagnostics_accepted=0, lifetime_mean_ns=1.9, mean_acceptance_fraction=0.4,
                production_steps=20, retained_samples=400, extension_count=0,
                parameter_summaries_json=parameter_json,
                diagnostic_failure_reasons_json='["autocorrelation_unavailable"]',
                nonfinite_fields_json='{"minimum_effective_samples":"+inf"}',
                posterior_artifact_id=artifact, predictive_artifact_id=predictive_artifact,
                ppc_status="success", ppc_n_draws=3, ppc_deviance_tail_probability=0.4,
                ppc_discrepancies_json=store._canonical_json({"random_seed_decimal": str(2**64 - 2)}))
        trusted = store.record_trusted_lifetime_reference(
            db, reference_key="trusted", reference_version="calibration-v1",
            measurement_id=experiment, lifetime_ns=2.4,
        )
        old = store.record_pseudo_true_reference(
            db, reference_key="old", reference_version="stage6-original",
            condition_pk=condition, assumption_id=assumptions[1], lifetime_ns=2.2,
            projection_temporal_shift_ns=0.0,
        )
        corrected = store.record_pseudo_true_reference(
            db, reference_key="corrected", reference_version="stage6.5-corrected",
            condition_pk=condition, assumption_id=assumptions[1], lifetime_ns=2.05,
            projection_temporal_shift_ns=0.0, supersedes_reference_id=old,
        )
        metrics = []
        for test, coverage, value in (("A", 0.9, 0.75), ("F", 0.9, None), ("A", 0.8, 0.0)):
            scope = store.BenchmarkMetricScope(
                run_id=runs[0], population_key=f"population-{test}", dataset_key=f"dataset-{test}",
                test_id=test, regime_id="same-label", model_id=models["ml"],
                condition_pk=condition, reference_kind="generating_mono",
                reference_semantics="generating_mono: known generating lifetime", method_id="quantile_gradient_boosting",
                method_configuration={"coverage": coverage}, nominal_coverage=coverage,
            )
            metrics.extend(store._record_metric_facts(
                db, scope, n_attempted=4,
                facts=[("empirical_coverage", "fraction", float("nan") if value is None else value,
                        0 if value is None else 4, 0 if value is None else 4, "valid_intervals")],
                on_duplicate="raise",
            ))
    return path, dict(runs=runs, artifact=artifact, predictive_artifact=predictive_artifact,
                      condition=condition, measured=measured, experiment=experiment, unlinked=unlinked,
                      models=models, assumptions=assumptions, points=points, uncertainties=uncertainties,
                      references=(trusted, old, corrected), metrics=metrics, source=source_id,
                      prepared=prepared_ids)


def test_measurements_keep_generating_and_experimental_semantics(populated):
    path, ids = populated
    base = store.query_measurements(path)
    assert len(base.rows) == 3
    assert "generating_mono_lifetime_ns" not in base.columns
    joined = _dicts(store.query_measurements(path, include_generating=True))
    raw, experimental, unlinked = joined
    assert raw["measurement_id"] == ids["measured"]
    assert raw["generating_mono_lifetime_ns"] == 2.0
    assert raw["generating_model"] == "monoexponential"
    assert raw["generating_signal_photon_count"] == 1000
    assert raw["generating_background_per_bin"] == 0.2
    assert raw["generating_shift_ns"] == 0.0
    assert raw["observation_seed_decimal"] == str(2**64 - 1)
    assert raw["sample_id"] == experimental["sample_id"] == unlinked["sample_id"]
    assert experimental["data_kind"] == "processed_intensity"
    assert experimental["condition_pk"] is None
    assert experimental["generating_mono_lifetime_ns"] is None
    assert experimental["attached_irf_source_id"] is not None
    assert not any(name in base.columns for name in ("true_lifetime_ns", "error", "reference_id"))


def test_measurement_membership_expansion_is_explicit_and_same_row_filtered(populated):
    path, ids = populated
    base = store.query_measurements(path, filters={"run_id": ids["runs"][0]})
    assert len(base.rows) == 2
    expanded = _dicts(store.query_measurements(path, include_membership=True))
    assert len(expanded) == 4
    assert [row["membership_run_id"] for row in expanded[:2]] == ids["runs"][:2]
    assert expanded[-1]["membership_run_id"] is None
    for include_membership in (False, True):
        assert not store.query_measurements(
            path, filters={"run_id": ids["runs"][0], "test_id": "F"},
            include_membership=include_membership,
        ).rows
        assert not store.query_measurements(
            path, filters={"test_id": None}, include_membership=include_membership,
        ).rows  # The unlinked NULL placeholder is not an actual membership.
        selected = store.query_measurements(
            path, filters={"run_id": ids["runs"][1], "test_id": "F", "dataset_key": "dataset-two"},
            include_membership=include_membership,
        )
        assert len(selected.rows) == 1


def test_irf_source_and_preparation_grains_stay_separate(populated):
    path, ids = populated
    sources = _dicts(store.query_irf_sources(path, filters={"irf_source_id": ids["source"]}))
    assert len(sources) == 1
    prepared = _dicts(store.query_prepared_irfs(
        path, filters={"irf_source_id": ids["source"]}, include_source=True, decode_json=True,
    ))
    assert len(prepared) == 2
    assert [row["prepared_irf_id"] for row in prepared] == ids["prepared"]
    assert all(row["source_kind"] == "synthetic_gaussian" for row in prepared)
    assert all(row["normalization_factor"] > 0 for row in prepared)
    assert all(isinstance(row["diagnostics_json"], dict) for row in prepared)
    assert all(row["source_grid_sha256"] == sources[0]["source_grid_sha256"] for row in prepared)
    assert len(store.query_irf_sources(path, filters={"source_kind": "imported_sampled"}).rows) == 1


def test_result_grain_and_optional_fit_details_do_not_expand_uncertainty(populated):
    path, ids = populated
    base = store.query_results(path)
    joined = store.query_results(path, include_context=True, include_fit_details=True)
    assert len(base.rows) == len(joined.rows) == 6
    rows = _dicts(joined)
    assert {row["model_family"] for row in rows} == {"ml", "classical", "baseline", "bayesian"}
    assert rows[0]["fit_fitted_amplitude"] == 100.0
    assert rows[0]["fit_poisson_deviance"] == 0.4
    assert rows[0]["fit_optimizer_nfev"] == 12
    assert rows[2]["fit_result_id"] is None
    assert rows[0]["assumption_id"] != rows[1]["assumption_id"]
    assert rows[1]["irf_model_relation"] == "deliberately_misspecified"
    assert len(store.query_results(path, filters={"model_id": ids["models"]["ml"]}).rows) == 2
    for name in ("uncertainty_id", "reference_id", "true_lifetime_ns", "error_ns", "coverage"):
        assert name not in joined.columns
    selected = _dicts(store.query_results(path, filters={
        "run_id": ids["runs"][0], "measurement_id": ids["measured"],
        "model_id": ids["models"]["classical"], "assumption_id": ids["assumptions"][1],
        "test_id": "A", "regime_id": "same-label",
    }))
    assert [row["result_id"] for row in selected] == [ids["points"][1]]


def test_uncertainty_rows_preserve_taxonomy_calibration_and_owner(populated):
    path, ids = populated
    rows = _dicts(store.query_uncertainty(path, include_context=True))
    assert len(rows) == 4
    assert {row["output_kind"] for row in rows} == {
        "prediction_interval", "uncertainty_score", "covariance_summary",
    }
    same_point = store.query_uncertainty(path, filters={"result_id": ids["points"][2]})
    assert len(same_point.rows) == 3
    score = _dicts(store.query_uncertainty(path, filters={
        "method_id": "random_forest_tree_spread", "nominal_coverage": None,
    }))[0]
    assert score["uncertainty_score"] == 0.0
    assert score["lower_ns"] is score["upper_ns"] is None
    assert (score["n_requested"], score["n_valid"]) == (20, 20)
    assert score["model_id"] == ids["models"]["ml"]
    calibrated = _dicts(store.query_uncertainty(path, filters={"calibration_run_id": ids["runs"][2]}))
    assert len(calibrated) == 1
    assert calibrated[0]["nominal_coverage"] == 0.9
    assert "covered" not in same_point.columns


def test_bayesian_summary_retains_diagnostics_not_acceptance_inference(populated, monkeypatch):
    path, ids = populated
    def no_external_open(*args, **kwargs):
        raise AssertionError("query must not load chains or predictive files")
    monkeypatch.setattr(builtins, "open", no_external_open)
    rows = _dicts(store.query_bayesian(path, include_context=True, decode_json=True))
    assert len(rows) == 1
    row = rows[0]
    assert row["sampling_status"] == "success" and row["diagnostics_accepted"] == 0
    assert row["result_is_valid"] == 0
    assert row["point_summary"] == "posterior_median"
    assert row["lifetime_estimate_ns"] == 1.85 and row["lifetime_mean_ns"] == 1.9
    assert row["minimum_effective_samples"] is None
    assert row["nonfinite_fields_json"]["minimum_effective_samples"] == "+inf"
    assert row["parameter_summaries_json"]["parameters"][0]["credible_lower"] == 1.6
    assert row["production_steps"] == 20 and row["retained_samples"] == 400
    assert row["ppc_n_draws"] == 3 and row["ppc_deviance_tail_probability"] == 0.4
    assert row["posterior_artifact_id"] == ids["artifact"]
    assert row["predictive_artifact_id"] == ids["predictive_artifact"]
    assert row["result_random_seed_decimal"] == str(2**64 - 1)
    assert row["ppc_discrepancies_json"]["random_seed_decimal"] == str(2**64 - 2)
    assert len(store.query_artifacts(path).rows) == 2


def test_reference_versions_are_returned_without_implicit_choice(populated):
    path, ids = populated
    references = _dicts(store.query_references(path, filters={
        "condition_pk": ids["condition"], "assumption_id": ids["assumptions"][1],
        "reference_kind": "pseudo_true_mono",
    }))
    assert [row["reference_id"] for row in references] == list(ids["references"][1:])
    assert references[1]["supersedes_reference_id"] == references[0]["reference_id"]
    selected = _dicts(store.query_references(path, filters={"reference_version": "stage6.5-corrected"}))
    assert selected[0]["lifetime_ns"] == 2.05
    trusted = _dicts(store.query_references(path, filters={"measurement_id": ids["experiment"]}))
    assert trusted[0]["reference_kind"] == "trusted_experimental"
    assert trusted[0]["condition_pk"] is None
    assert not store.query_references(path, filters={"measurement_id": ids["measured"]}).rows


def test_metric_facts_keep_scope_denominators_and_undefined_distinct_from_zero(populated):
    path, ids = populated
    rows = _dicts(store.query_metrics(path, filters={"metric_name": "empirical_coverage"}))
    assert len(rows) == 3
    assert [row["metric_value"] for row in rows] == [0.75, None, 0.0]
    assert [row["value_status"] for row in rows] == ["finite", "undefined", "finite"]
    assert [row["nominal_coverage"] for row in rows] == [0.9, 0.9, 0.8]
    assert [(row["n_attempted"], row["n_valid"], row["n_contributing"]) for row in rows] == [
        (4, 4, 4), (4, 0, 0), (4, 4, 4),
    ]
    assert all(row["denominator_kind"] == "valid_intervals" for row in rows)
    with closing(store.connect_database(path, readonly=True)) as db:
        for row in rows:
            stored = db.execute("SELECT scope_json, scope_sha256 FROM benchmark_metrics WHERE metric_id = ?",
                                (row["metric_id"],)).fetchone()
            assert (row["scope_json"], row["scope_sha256"]) == tuple(stored)
    selected = _dicts(store.query_metrics(path, filters={
        "run_id": ids["runs"][0], "test_id": "A", "regime_id": "same-label",
        "dataset_key": "dataset-A", "population_key": "population-A",
        "model_id": ids["models"]["ml"], "method_id": "quantile_gradient_boosting",
        "nominal_coverage": 0.8, "signal_photon_count": 1000, "background_per_bin": 0.2,
        "reference_kind": "generating_mono", "reference_semantics": "generating_mono: known generating lifetime",
    }))
    assert [row["metric_id"] for row in selected] == [ids["metrics"][2]]


@pytest.mark.parametrize("value", ["' OR 1=1 --", '\"; DROP TABLE estimator_results; --', "%", "_"])
def test_hostile_filter_values_are_exact_bound_literals(populated, value):
    path, _ = populated
    with closing(store.connect_database(path)) as db:
        run = store.record_run(db, run_key=value, run_type="evaluation", status="complete",
                               recorded_at_utc="2026-10-02T10:00:00+00:00", origin="new",
                               package_version="0.7.0", configuration={})
        assert [row["run_id"] for row in _dicts(store.query_runs(db, filters={"run_key": value}))] == [run]
        for query, field in ((store.query_measurements, "measurement_key"), (store.query_results, "analysis_key"),
                             (store.query_uncertainty, "method_id"), (store.query_metrics, "dataset_key")):
            assert not query(db, filters={field: value}).rows
        assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 6


def test_unsupported_filter_sort_and_value_types_are_rejected(populated):
    path, _ = populated
    with pytest.raises(ValueError, match="unsupported query filter"):
        store.query_results(path, filters={"model_id OR 1=1 --": 1})
    with pytest.raises(ValueError, match="unsupported query order"):
        store.query_results(path, order_by="result_id; DROP TABLE estimator_results")
    with pytest.raises(TypeError, match="scalar"):
        store.query_results(path, filters=[("model_id", 1)])
    for value in ([1, 2], float("nan"), float("inf")):
        with pytest.raises(TypeError, match="filter values"):
            store.query_results(path, filters={"model_id": value})


def test_default_and_alternate_ordering_have_deterministic_tie_breaks(populated):
    path, ids = populated
    assert [row["run_id"] for row in _dicts(store.query_runs(path))] == ids["runs"]
    rows = _dicts(store.query_runs(path, order_by="run_key"))
    assert [row["run_key"] for row in rows] == ["calibration", "run-a", "run-z"]
    reverse = _dicts(store.query_runs(path, order_by="run_key", descending=True))
    assert reverse == list(reversed(rows))
    metrics = _dicts(store.query_metrics(path, order_by="metric_name"))
    assert [row["metric_id"] for row in metrics] == ids["metrics"]


def test_json_decoding_is_uniform_and_nonfinite_metadata_remains_visible(populated):
    path, _ = populated
    for query, options in ((store.query_results, {"include_context": True, "include_fit_details": True}),
                           (store.query_measurements, {"include_generating": True}),
                           (store.query_bayesian, {}), (store.query_metrics, {})):
        raw = query(path, **options)
        decoded = query(path, decode_json=True, **options)
        assert raw.columns == decoded.columns
        for original, converted in zip(raw.rows, decoded.rows):
            for name, left, right in zip(raw.columns, original, converted):
                if name.endswith("_json") and left is not None:
                    assert isinstance(left, str)
                    assert right == json.loads(left)
                else:
                    assert right == left
    undefined = _dicts(store.query_metrics(path, filters={"value_status": "undefined"}, decode_json=True))[0]
    assert undefined["metric_value"] is None
    assert undefined["nonfinite_fields_json"] == {"metric_value": "nan"}


def test_dataframe_preserves_rows_columns_nulls_json_and_large_nullable_ids(populated):
    path, _ = populated
    result = store.query_metrics(path)
    frame = store.query_to_dataframe(result)
    assert list(frame.columns) == list(result.columns)
    assert len(frame) == len(result.rows)
    assert frame["metric_value"].tolist() == [0.75, None, 0.0]
    assert frame["value_status"].tolist() == ["finite", "undefined", "finite"]
    assert frame["metric_id"].tolist() == [1, 2, 3]
    assert frame["nonfinite_fields_json"].tolist() == ["{}", '{"metric_value":"nan"}', "{}"]
    assert frame.loc[1, "metric_value"] is None and pd.isna(frame.loc[1, "metric_value"])
    exact = store.query_to_dataframe(store.QueryResult(
        ("nullable_id", "real_value", "seed"), ((2**63 - 1, 1.0, str(2**64 - 1)), (None, None, None)),
    ))
    assert exact.loc[0, "nullable_id"] == 2**63 - 1
    assert isinstance(exact.loc[0, "nullable_id"], int)
    assert isinstance(exact.loc[0, "real_value"], float)
    assert exact.loc[1, "nullable_id"] is None
    empty = store.query_results(path, filters={"result_id": -1})
    assert list(store.query_to_dataframe(empty).columns) == list(empty.columns)


def test_plain_queries_do_not_import_pandas_and_conversion_error_is_clear(populated, monkeypatch):
    path, _ = populated
    original = builtins.__import__
    def no_pandas(name, *args, **kwargs):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_pandas)
    result = store.query_results(path)
    assert len(result.rows) == 6
    with pytest.raises(ImportError, match="requires the toolkit's pandas dependency"):
        store.query_to_dataframe(result)


_ALL_QUERIES = (
    (store.query_runs, {}), (store.query_artifacts, {}),
    (store.query_measurements, {"include_generating": True, "include_membership": True}),
    (store.query_irf_sources, {}), (store.query_prepared_irfs, {"include_source": True}),
    (store.query_results, {"include_context": True, "include_fit_details": True}),
    (store.query_uncertainty, {"include_context": True}),
    (store.query_bayesian, {"include_context": True}), (store.query_references, {}),
    (store.query_metrics, {}),
)


@pytest.mark.parametrize("query,options", _ALL_QUERIES, ids=lambda obj: getattr(obj, "__name__", "options"))
def test_every_query_works_on_readonly_and_writable_connections_without_mutation(populated, query, options):
    path, _ = populated
    before = path.read_bytes()
    with closing(store.connect_database(path, readonly=True)) as readonly:
        assert readonly.execute("PRAGMA query_only").fetchone()[0] == 1
        expected = query(readonly, **options)
        assert readonly.total_changes == 0 and not readonly.in_transaction
        assert readonly.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with closing(store.connect_database(path)) as writable:
        factory = writable.row_factory
        trace = []
        writable.set_trace_callback(trace.append)
        assert query(writable, **options) == expected
        assert all(statement.lstrip().startswith("SELECT ") for statement in trace)
        assert writable.row_factory is factory
        assert writable.total_changes == 0 and not writable.in_transaction
        assert writable.execute("PRAGMA query_only").fetchone()[0] == 0
    assert path.read_bytes() == before


def test_path_queries_open_readonly_and_missing_database_is_not_created(populated, monkeypatch, tmp_path):
    path, _ = populated
    original = store.connect_database
    opened = []
    def checked(database, *, readonly=False):
        assert readonly is True
        db = original(database, readonly=readonly)
        opened.append(db)
        return db
    monkeypatch.setattr(store, "connect_database", checked)
    for query, options in _ALL_QUERIES:
        assert query(path, **options).rows
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            db.execute("SELECT 1")
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        store.query_runs(missing)
    assert not missing.exists()


def test_caller_transaction_and_row_factory_are_left_untouched(populated):
    path, _ = populated
    with closing(store.connect_database(path)) as db:
        db.row_factory = lambda cursor, row: dict(zip((column[0] for column in cursor.description), row))
        factory = db.row_factory
        db.execute("BEGIN")
        result = store.query_results(db)
        assert isinstance(result.rows[0], tuple)
        assert db.row_factory is factory and db.in_transaction
        db.rollback()


def test_point_and_uncertainty_validity_filters_are_separate(populated):
    path, ids = populated
    with closing(store.connect_database(path)) as db:
        # Failed diagnostic output is still a row; query code must not repair it.
        db.execute("UPDATE uncertainty_results SET is_valid = 0, lower_ns = 3.0, upper_ns = 1.0 "
                   "WHERE uncertainty_id = ?", (ids["uncertainties"][0],))
        rows = _dicts(store.query_uncertainty(db, filters={"is_valid": 0, "result_is_valid": 1}))
        assert len(rows) == 1
        assert (rows[0]["lower_ns"], rows[0]["upper_ns"]) == (3.0, 1.0)
        assert rows[0]["result_status"] == "available"
        assert len(store.query_results(db, filters={"assumption_id": None}).rows) == 3
        rejected = store.query_bayesian(db, filters={
            "sampling_status": "success", "diagnostics_accepted": 0, "result_is_valid": 0,
        })
        assert len(rejected.rows) == 1


def test_queries_do_not_call_scientific_or_recording_adapters(populated, monkeypatch):
    path, _ = populated
    def forbidden(*args, **kwargs):
        raise AssertionError("retrieval must not normalize, recompute or record science")
    for name in ("_metric_scope", "_metric_fact", "_bayesian_payloads", "_estimator_payload",
                 "_record_row", "initialize_database", "transaction"):
        monkeypatch.setattr(store, name, forbidden)
    for query, options in _ALL_QUERIES:
        assert query(path, **options).rows


def test_biexponential_components_are_not_selected_as_one_truth(populated):
    path, _ = populated
    with closing(store.connect_database(path)) as db:
        condition = store.record_simulation_condition(
            db, condition_key="bi", condition_id="local-regime", generating_model="biexponential",
            primary_lifetime_ns=2.0, secondary_lifetime_ns=5.0, secondary_detected_fraction=0.2,
            signal_photon_count=1000, background_per_bin=0.2, true_temporal_shift_ns=0.0,
        )
        measurement = store.record_synthetic_observation(
            db, measurement_key="bi", time_ns=np.arange(9) * 0.25, values=np.arange(1, 10),
            data_kind="raw_counts", condition_pk=condition,
        )
        row = _dicts(store.query_measurements(
            db, filters={"measurement_id": measurement}, include_generating=True,
        ))[0]
        assert row["generating_mono_lifetime_ns"] is None
        assert (row["generating_primary_lifetime_ns"], row["generating_secondary_lifetime_ns"],
                row["generating_secondary_detected_fraction"]) == (2.0, 5.0, 0.2)
        assert "true_lifetime_ns" not in row


def test_no_orphan_preparation_source_is_invented(populated):
    path, _ = populated
    with closing(store.connect_database(path)) as db:
        # Historical kernel-only rows deliberately have no original source.
        prepared = _insert(db, "prepared_irfs", preparation_key="historical-kernel",
                           preparation_kind="supplied_kernel", time_grid_sha256="a" * 64,
                           kernel_sha256="b" * 64, n_bins=9, time_start_ns=0.0, time_step_ns=0.25)
        row = _dicts(store.query_prepared_irfs(
            db, filters={"prepared_irf_id": prepared}, include_source=True,
        ))[0]
        assert row["irf_source_id"] is None and row["source_kind"] is None
        assert row["preparation_kind"] == "supplied_kernel"


def test_database_dataframe_nullable_integer_does_not_round_trip_through_float(populated):
    path, ids = populated
    with closing(store.connect_database(path)) as db:
        large = 2**63 - 1
        original = dict(db.execute("SELECT * FROM experiment_runs WHERE run_id = ?",
                                   (ids["runs"][0],)).fetchone())
        original.update(run_id=large, run_key="large-id")
        _insert(db, "experiment_runs", **original)
        db.execute("UPDATE experiment_runs SET source_run_id = ? WHERE run_id = ?",
                   (large, ids["runs"][0]))
        frame = store.query_to_dataframe(store.query_runs(db))
        assert frame.loc[0, "source_run_id"] == large
        assert isinstance(frame.loc[0, "source_run_id"], int)
        assert frame.loc[1, "source_run_id"] is None
        assert frame.iloc[-1]["run_id"] == large
