"""Stage-7 integration example: public adapters -> disk -> read-only queries."""

import builtins
import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from tcspc_toolkit import persistence as store


@pytest.fixture
def example(monkeypatch):
    original_import = builtins.__import__

    def no_emcee(name, *args, **kwargs):
        if name == "emcee" or name.startswith("emcee."):
            raise AssertionError("The persistence example must never run MCMC")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_emcee)
    path = Path(__file__).resolve().parents[1] / "examples" / "persistence_roundtrip.py"
    spec = importlib.util.spec_from_file_location("persistence_roundtrip_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roundtrip_closes_writer_then_reads_all_layers(example, tmp_path, monkeypatch):
    connections = []
    read_started = False
    original_connect = store.connect_database

    def track_connection(path, *, readonly=False):
        nonlocal read_started
        if readonly:
            assert len(connections) == 1
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connections[0].execute("SELECT 1")
            read_started = True
        db = original_connect(path, readonly=readonly)
        if readonly:
            assert db.execute("PRAGMA query_only").fetchone()[0] == 1
            # Even an empty write is forbidden; no fixture data can be deleted.
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                db.execute("DELETE FROM measurements WHERE 0")
        connections.append(db)
        return db

    monkeypatch.setattr(store, "connect_database", track_connection)

    original_aggregation = example._query_result_counts_by_family
    aggregation_calls = []

    def checked_aggregation(connection, *, run_id):
        assert read_started and connection is connections[1]
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        aggregation_calls.append(run_id)
        return original_aggregation(connection, run_id=run_id)

    monkeypatch.setattr(example, "_query_result_counts_by_family", checked_aggregation)

    def write_phase_only(function):
        def checked(*args, **kwargs):
            assert not read_started, "Scientific computation occurred after read-only reopen"
            return function(*args, **kwargs)
        return checked

    for name in (
        "fit_monoexponential_reconvolution", "estimate_poisson_reconvolution_local_covariance",
        "calculate_robustness_metrics", "_bayesian_fixture",
        "monoexponential_reconvolution_expected_counts", "sample_photon_counts",
        "calculate_bayesian_posterior_predictive_diagnostics", "summarize_predictive_band",
    ):
        monkeypatch.setattr(example, name, write_phase_only(getattr(example, name)))

    path = tmp_path / "roundtrip.sqlite"
    queries, frames = example.run_persistence_roundtrip(path)
    assert len(connections) == 2 and read_started
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert {name: len(result.rows) for name, result in queries.items()} == {
        "runs": 1, "measurements": 1, "irf_sources": 1, "prepared_irfs": 1,
        "results": 2, "uncertainty": 1, "bayesian": 1, "metrics": 6,
        "result_counts_by_family": 2,
    }
    assert aggregation_calls == [frames["runs"].iloc[0]["run_id"]]
    assert queries["result_counts_by_family"].rows == (("bayesian", 1, 1), ("classical", 1, 1))
    for name, query in queries.items():
        assert isinstance(query, store.QueryResult)
        assert tuple(frames[name].columns) == query.columns
        assert list(frames[name].itertuples(index=False, name=None)) == list(query.rows)

    measured = frames["measurements"].iloc[0]
    prepared = frames["prepared_irfs"].iloc[0]
    assert measured["source_type"] == "synthetic" and measured["data_kind"] == "raw_counts"
    assert measured["condition_pk"] is not None
    assert measured["generating_model"] == "monoexponential"
    assert measured["generating_mono_lifetime_ns"] == 2.0
    assert measured["generating_signal_photon_count"] == 5000
    assert measured["generating_background_per_bin"] == 2.0
    assert measured["observed_total_counts"] == 5111
    assert measured["time_grid_sha256"] == prepared["time_grid_sha256"]
    assert measured["generating_irf_id"] == prepared["prepared_irf_id"]
    assert prepared["irf_source_id"] == frames["irf_sources"].iloc[0]["irf_source_id"]

    points = frames["results"].set_index("model_family")
    classical, bayesian = points.loc["classical"], points.loc["bayesian"]
    assert classical["measurement_id"] == bayesian["measurement_id"] == measured["measurement_id"]
    assert classical["model_id"] != bayesian["model_id"]
    assert classical["assumption_id"] == bayesian["assumption_id"]
    assert classical["assumed_prepared_irf_id"] == prepared["prepared_irf_id"]
    assert classical["is_valid"] == 1
    assert classical["fit_result_id"] == classical["result_id"]
    assert bayesian["fit_result_id"] is None
    assert "generating_mono_lifetime_ns" not in points.columns
    assert "true_lifetime_ns" not in points.columns
    assert "absolute_error_ns" not in points.columns

    uncertainty = frames["uncertainty"].iloc[0]
    assert uncertainty["result_id"] == classical["result_id"]
    assert uncertainty["output_kind"] == "covariance_summary"
    assert uncertainty["is_valid"] == 1 and uncertainty["reported_std_ns"] > 0
    assert uncertainty["nominal_coverage"] is None  # SD alone is not an interval.
    posterior = frames["bayesian"].iloc[0]
    assert posterior["result_id"] == bayesian["result_id"]
    assert posterior["point_summary"] == "posterior_median"
    assert posterior["lifetime_estimate_ns"] == 1.9
    assert posterior["lifetime_mean_ns"] == 1.92
    assert posterior["parameter_summaries_json"]["credible_probability"] == 0.9
    assert posterior["sampling_status"] == "success" and posterior["diagnostics_accepted"] == 1
    assert posterior["ppc_status"] == "success"
    assert posterior["ppc_n_draws"] == 3
    assert posterior["result_random_seed_decimal"] == str(example.SAMPLING_SEED)
    assert posterior["result_execution_json"]["mcmc_executed"] is False
    assert posterior["posterior_artifact_id"] is None and posterior["predictive_artifact_id"] is None

    metrics = frames["metrics"]
    assert set(metrics["metric_name"]) == {
        "mae_ns", "median_absolute_error_ns", "rmse_ns", "bias_ns",
        "p90_absolute_error_ns", "p95_absolute_error_ns",
    }
    assert set(metrics["denominator_kind"]) == {"evaluated_predictions"}
    assert metrics[["n_attempted", "n_valid", "n_contributing"]].eq(1).all().all()
    assert metrics["scope_sha256"].nunique() == 1
    assert metrics["reference_id"].isna().all()
    for scope in metrics["scope_json"]:
        assert scope["measurement_ids"] == [measured["measurement_id"]]
        assert scope["dataset_key"] == example.DATASET_KEY

    # An independent subsequent reader sees the saved facts without any source objects.
    with closing(original_connect(path, readonly=True)) as db:
        assert store.query_results(db, include_context=True, include_fit_details=True) == queries["results"]
        assert not store.query_references(db).rows
        assert not store.query_artifacts(db).rows


def test_sql_aggregation_counts_stored_results_readonly(example, tmp_path):
    path = tmp_path / "aggregation.sqlite"
    queries, frames = example.run_persistence_roundtrip(path)
    run_id = frames["runs"].iloc[0]["run_id"]
    before = path.read_bytes()
    with closing(store.connect_database(path, readonly=True)) as db:
        assert db.execute("PRAGMA query_only").fetchone()[0] == 1
        statements = []
        db.set_trace_callback(statements.append)
        grouped = example._query_result_counts_by_family(db, run_id=run_id)
        assert grouped.columns == ("model_family", "n_results", "n_valid_results")
        # Alphabetical family order, not classical-then-Bayesian insertion order.
        assert grouped.rows == (("bayesian", 1, 1), ("classical", 1, 1))
        assert grouped == queries["result_counts_by_family"]
        assert not example._query_result_counts_by_family(db, run_id=run_id + 1).rows
        assert len(statements) == 2
        assert all(sql.lstrip().startswith("SELECT ") and "GROUP BY mv.family" in sql
                   for sql in statements)
        assert db.total_changes == 0 and not db.in_transaction
    assert path.read_bytes() == before


def test_example_repeats_scientific_payloads_not_wall_clock_metadata(example, tmp_path):
    _, first = example.run_persistence_roundtrip(tmp_path / "first.sqlite")
    _, second = example.run_persistence_roundtrip(tmp_path / "second.sqlite")
    for name in first:
        # Creation/recording timestamps deliberately describe each actual invocation.
        columns = [column for column in first[name] if not column.endswith("_at_utc")]
        pd.testing.assert_frame_equal(first[name][columns], second[name][columns])


@pytest.mark.parametrize("existing_database", [False, True])
def test_example_never_overwrites_an_existing_file(example, tmp_path, existing_database):
    path = tmp_path / "existing.sqlite"
    if existing_database:
        example.run_persistence_roundtrip(path)
    else:
        path.write_bytes(b"not a database; still belongs to the user")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        example.run_persistence_roundtrip(path)
    assert path.read_bytes() == before
