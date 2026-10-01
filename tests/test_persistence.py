"""Schema-v1 and connection contracts; scientific-object adapters come later."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
from contextlib import closing

import numpy as np
import pytest

from tcspc_toolkit import persistence


_DIGEST = "a" * 64
_TABLE_NAMES = {
    "schema_metadata",
    "experiment_runs",
    "artifacts",
    "irf_sources",
    "prepared_irfs",
    "simulation_conditions",
    "measurements",
    "run_measurements",
    "model_versions",
    "model_assumptions",
    "lifetime_references",
    "estimator_results",
    "fit_details",
    "uncertainty_results",
    "bayesian_summaries",
    "benchmark_metrics",
}


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "results.sqlite"
    persistence.initialize_database(path)
    return path


def _insert_run(connection, key="run-1", *, origin="new", version_state="known",
                package_version="0.7.0", code_revision="test-revision",
                unknown_reason=None, seed=None):
    return connection.execute(
        "INSERT INTO experiment_runs "
        "(run_key, run_type, status, recorded_at_utc, origin, version_state, "
        "producing_package_version, producing_code_revision, "
        "producing_version_unknown_reason, config_json, config_sha256, "
        "base_seed_decimal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (key, "schema_fixture", "complete", "2026-10-01T00:00:00+00:00",
         origin, version_state, package_version, code_revision,
         unknown_reason, "{}", _DIGEST, seed),
    ).lastrowid


def _insert_measurement(connection, key="measurement-1"):
    return connection.execute(
        "INSERT INTO measurements "
        "(measurement_key, source_type, data_kind, n_bins, time_start_ns, "
        "time_step_ns, time_grid_sha256, values_sha256, "
        "observed_total_counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (key, "synthetic", "raw_counts", 3, 0.0, 0.25, _DIGEST, _DIGEST, 6),
    ).lastrowid


def _insert_model(connection, key="model-1", family="classical"):
    return connection.execute(
        "INSERT INTO model_versions "
        "(model_key, estimator_name, family, configuration_json, "
        "configuration_sha256) VALUES (?, ?, ?, ?, ?)",
        (key, "reconvolution" if family == "classical" else family,
         family, "{}", _DIGEST),
    ).lastrowid


def _insert_result(connection, run_id, measurement_id, model_id,
                   *, analysis_key="default", status="failed", valid=0):
    return connection.execute(
        "INSERT INTO estimator_results "
        "(run_id, measurement_id, model_id, analysis_key, point_summary, "
        "status, is_valid, source_result_type, lifetime_estimate_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, measurement_id, model_id, analysis_key, "lifetime_ns",
         status, valid, "fixture_result", 2.0 if valid else None),
    ).lastrowid


def _seed_result_graph(connection):
    run_id = _insert_run(connection)
    measurement_id = _insert_measurement(connection)
    connection.execute(
        "INSERT INTO run_measurements (run_id, measurement_id, data_role) "
        "VALUES (?, ?, ?)", (run_id, measurement_id, "evaluation"),
    )
    model_id = _insert_model(connection)
    result_id = _insert_result(connection, run_id, measurement_id, model_id)
    return run_id, measurement_id, model_id, result_id


def test_new_database_has_complete_versioned_schema(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        metadata = connection.execute(
            "SELECT * FROM schema_metadata"
        ).fetchone()
        assert metadata["schema_version"] == persistence.SCHEMA_VERSION
        assert metadata["serialization_version"] == 1
        assert metadata["definition_sha256"] == persistence._definition_sha256()
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert tables == _TABLE_NAMES
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_in_memory_database_uses_caller_connection():
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        persistence.initialize_database(connection)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        connection.row_factory = sqlite3.Row
        persistence.initialize_database(connection)
        assert connection.execute("SELECT COUNT(*) FROM schema_metadata").fetchone()[0] == 1
    with pytest.raises(ValueError, match="open sqlite3.Connection"):
        persistence.initialize_database(":memory:")


def test_initialization_rejects_an_active_caller_transaction():
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        connection.execute("BEGIN")
        with pytest.raises(ValueError, match="before starting a transaction"):
            persistence.initialize_database(connection)
        connection.rollback()
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0


def test_compatible_reopen_is_idempotent(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        before = tuple(connection.execute(
            "SELECT created_at_utc, definition_sha256 FROM schema_metadata"
        ).fetchone())
    persistence.initialize_database(database_path)
    with closing(persistence.connect_database(database_path)) as connection:
        assert tuple(connection.execute(
            "SELECT created_at_utc, definition_sha256 FROM schema_metadata"
        ).fetchone()) == before


def test_unrelated_incomplete_and_unsupported_databases_are_rejected(tmp_path, database_path):
    unrelated = tmp_path / "unrelated.sqlite"
    with closing(sqlite3.connect(unrelated)) as connection:
        connection.execute("CREATE TABLE other_data (id INTEGER)")
        connection.commit()
    with pytest.raises(persistence.PersistenceSchemaError, match="missing or altered"):
        persistence.initialize_database(unrelated)
    with pytest.raises(persistence.PersistenceSchemaError):
        persistence.connect_database(unrelated)

    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("DROP INDEX idx_results_measurement")
        connection.commit()
    with pytest.raises(persistence.PersistenceSchemaError, match="index"):
        persistence.initialize_database(database_path)
    with pytest.raises(persistence.PersistenceSchemaError, match="index"):
        persistence.connect_database(database_path)

    unsupported = tmp_path / "unsupported.sqlite"
    persistence.initialize_database(unsupported)
    with closing(sqlite3.connect(unsupported)) as connection:
        connection.execute("PRAGMA user_version = 2")
    with pytest.raises(persistence.PersistenceSchemaError, match="user_version"):
        persistence.connect_database(unsupported)
    with pytest.raises(persistence.PersistenceSchemaError, match="user_version"):
        persistence.initialize_database(unsupported)

    altered = tmp_path / "altered.sqlite"
    persistence.initialize_database(altered)
    with closing(sqlite3.connect(altered)) as connection:
        connection.execute("ALTER TABLE experiment_runs ADD COLUMN user_note TEXT")
        connection.commit()
    with pytest.raises(persistence.PersistenceSchemaError, match="missing or altered table"):
        persistence.connect_database(altered)


def test_user_tables_views_indexes_and_own_triggers_remain_compatible(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("CREATE TABLE personal_notes (note TEXT NOT NULL)")
        connection.execute(
            "CREATE VIEW personal_result_status AS "
            "SELECT result_id, status FROM estimator_results"
        )
        connection.execute(
            "CREATE INDEX personal_result_status_idx ON estimator_results(status)"
        )
        connection.execute(
            "CREATE TRIGGER personal_notes_check AFTER INSERT ON personal_notes "
            "BEGIN SELECT NEW.note; END"
        )
        connection.commit()

    persistence.initialize_database(database_path)
    with closing(persistence.connect_database(database_path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM personal_result_status"
        ).fetchone()[0] == 0
        with persistence.transaction(connection):
            connection.execute(
                "INSERT INTO personal_notes (note) VALUES (?)", ("query note",)
            )
        assert connection.execute(
            "SELECT note FROM personal_notes"
        ).fetchone()[0] == "query note"


def test_trigger_on_toolkit_table_is_rejected(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            "CREATE TRIGGER personal_result_trigger "
            "AFTER INSERT ON estimator_results "
            "BEGIN SELECT NEW.result_id; END"
        )
        connection.commit()
    with pytest.raises(persistence.PersistenceSchemaError,
                       match="unexpected trigger on toolkit table"):
        persistence.connect_database(database_path)
    with pytest.raises(persistence.PersistenceSchemaError,
                       match="unexpected trigger on toolkit table"):
        persistence.initialize_database(database_path)


def test_extra_unique_index_on_toolkit_table_is_rejected(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            "CREATE UNIQUE INDEX personal_unique_result_status "
            "ON estimator_results(status)"
        )
        connection.commit()
    with pytest.raises(persistence.PersistenceSchemaError,
                       match="unexpected unique index on toolkit table"):
        persistence.connect_database(database_path)


def test_empty_database_with_unrelated_user_version_is_rejected(tmp_path):
    path = tmp_path / "version_only.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version = 42")
    with pytest.raises(persistence.PersistenceSchemaError, match="unrelated user_version"):
        persistence.initialize_database(path)


def test_schema_indexes_foreign_keys_and_checks(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        indexes = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )}
        assert {name for name, _ in persistence._INDEXES} <= indexes
        assert {"uq_results_identity", "uq_uncertainty_identity"} <= indexes
        result_fks = connection.execute(
            "PRAGMA foreign_key_list(estimator_results)"
        ).fetchall()
        assert {row[2] for row in result_fks} >= {
            "run_measurements", "model_versions", "model_assumptions"
        }
        assert sum(row[2] == "run_measurements" for row in result_fks) == 2
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                "INSERT INTO run_measurements "
                "(run_id, measurement_id, data_role) VALUES (?, ?, ?)",
                (999, 999, "evaluation"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert_run(connection, key="bad-new", version_state="unknown",
                        package_version=None, code_revision=None,
                        unknown_reason="missing")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert_run(connection, key="bad-seed", seed="+12")


def test_new_and_historical_version_evidence_and_unsigned_seed(database_path):
    maximum_unsigned_seed = str(2**64 - 1)
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            _insert_run(connection, seed=maximum_unsigned_seed)
            _insert_run(connection, key="historical", origin="historical",
                        version_state="unknown", package_version=None,
                        code_revision=None, unknown_reason="source manifest absent")
            _insert_run(connection, key="installed-release", package_version="0.7.0",
                        code_revision=None)
        rows = connection.execute(
            "SELECT run_key, version_state, base_seed_decimal FROM experiment_runs "
            "ORDER BY run_id"
        ).fetchall()
        assert rows[0][2] == maximum_unsigned_seed
        assert rows[1][1] == "unknown"
        assert rows[1][2] is None
        assert rows[2][1] == "known"
        release = connection.execute(
            "SELECT producing_package_version, producing_code_revision, "
            "producing_git_commit FROM experiment_runs WHERE run_key = ?",
            ("installed-release",),
        ).fetchone()
        assert tuple(release) == ("0.7.0", None, None)
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "UPDATE experiment_runs SET producing_git_commit = ? "
                "WHERE run_key = ?", ("unverified-commit", "historical"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert_run(connection, key="missing-version",
                        package_version=None)
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            _insert_run(connection, key="blank-code", code_revision=" ")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "UPDATE experiment_runs SET producing_git_commit = ? "
                "WHERE run_key = ?", (" ", "installed-release"),
            )


def test_unsigned_seed_round_trips_across_run_observation_result_and_uncertainty(database_path):
    unsigned_seed = str(2**64 - 1)
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            run_id, measurement_id, _, result_id = _seed_result_graph(connection)
            connection.execute(
                "UPDATE experiment_runs SET base_seed_decimal = ? WHERE run_id = ?",
                (unsigned_seed, run_id),
            )
            connection.execute(
                "UPDATE measurements SET observation_seed_decimal = ? "
                "WHERE measurement_id = ?", (unsigned_seed, measurement_id),
            )
            connection.execute(
                "UPDATE estimator_results SET random_seed_decimal = ? "
                "WHERE result_id = ?", (unsigned_seed, result_id),
            )
            uncertainty_id = connection.execute(
                "INSERT INTO uncertainty_results "
                "(result_id, method_id, output_kind, method_config_json, "
                "method_config_sha256, interpretation, calibration_scope, "
                "is_valid, uncertainty_score, random_seed_decimal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (result_id, "bootstrap_spread", "uncertainty_score", "{}",
                 _DIGEST, "heuristic score", "none", 1, 0.2, unsigned_seed),
            ).lastrowid
        for table, id_column, id_value, seed_column in (
            ("experiment_runs", "run_id", run_id, "base_seed_decimal"),
            ("measurements", "measurement_id", measurement_id,
             "observation_seed_decimal"),
            ("estimator_results", "result_id", result_id,
             "random_seed_decimal"),
            ("uncertainty_results", "uncertainty_id", uncertainty_id,
             "random_seed_decimal"),
        ):
            # Identifiers here are fixed test constants, not user input.
            observed = connection.execute(
                f"SELECT {seed_column} FROM {table} WHERE {id_column} = ?",
                (id_value,),
            ).fetchone()[0]
            assert observed == unsigned_seed


def test_generating_attached_and_assumed_irf_links_remain_distinct(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            _insert_run(connection)
            measurement_id = _insert_measurement(connection)
            source_ids = {}
            for key, kind in (
                ("generating", "synthetic_gaussian"),
                ("attached", "imported_sampled"),
                ("assumed", "leading_edge_estimate"),
            ):
                source_ids[key] = connection.execute(
                    "INSERT INTO irf_sources "
                    "(source_key, source_representation, source_kind, n_bins, "
                    "source_grid_sha256, source_values_sha256, "
                    "derived_from_measurement_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, "profile", kind, 3, _DIGEST, _DIGEST,
                     measurement_id if key == "assumed" else None),
                ).lastrowid
            prepared_ids = {}
            for key in ("generating", "assumed"):
                prepared_ids[key] = connection.execute(
                    "INSERT INTO prepared_irfs "
                    "(preparation_key, irf_source_id, preparation_kind, "
                    "time_grid_sha256, kernel_sha256, n_bins, time_start_ns, "
                    "time_step_ns, registration_offset_ns, resampling_method, "
                    "support_loss_fraction, normalization_factor) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, source_ids[key], "explicit", _DIGEST, _DIGEST,
                     3, 0.0, 0.25, 0.0, "none", 0.0, 1.0),
                ).lastrowid
            condition_id = connection.execute(
                "INSERT INTO simulation_conditions "
                "(condition_key, condition_id, generating_model, "
                "generating_irf_id, mono_lifetime_ns, signal_photon_count, "
                "background_per_bin, true_temporal_shift_ns) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("condition-1", "A-low", "monoexponential",
                 prepared_ids["generating"], 2.0, 1000, 0.2, 0.0),
            ).lastrowid
            connection.execute(
                "UPDATE measurements SET condition_pk = ?, "
                "attached_irf_source_id = ? WHERE measurement_id = ?",
                (condition_id, source_ids["attached"], measurement_id),
            )
            assumption_id = connection.execute(
                "INSERT INTO model_assumptions "
                "(assumption_key, assumed_decay_model, observation_model, "
                "background_convention, prepared_irf_id, context_completeness) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("assumption-1", "monoexponential", "poisson_reconvolution",
                 "fitted_constant_per_bin", prepared_ids["assumed"],
                 "complete"),
            ).lastrowid
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(
                    "UPDATE measurements SET source_type = 'experimental' "
                    "WHERE measurement_id = ?", (measurement_id,),
                )
        links = connection.execute(
            "SELECT gm.source_key, ai.source_key, am.source_key "
            "FROM measurements AS m "
            "JOIN simulation_conditions AS c ON c.condition_pk = m.condition_pk "
            "JOIN prepared_irfs AS gp ON gp.prepared_irf_id = c.generating_irf_id "
            "JOIN irf_sources AS gm ON gm.irf_source_id = gp.irf_source_id "
            "JOIN irf_sources AS ai ON ai.irf_source_id = m.attached_irf_source_id "
            "JOIN model_assumptions AS a ON a.assumption_id = ? "
            "JOIN prepared_irfs AS ap ON ap.prepared_irf_id = a.prepared_irf_id "
            "JOIN irf_sources AS am ON am.irf_source_id = ap.irf_source_id "
            "WHERE m.measurement_id = ?", (assumption_id, measurement_id),
        ).fetchone()
        assert tuple(links) == ("generating", "attached", "assumed")


def test_model_specs_are_reusable_and_need_no_trained_artifact(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            run_id = _insert_run(connection)
            model_ids = {}
            for family in ("classical", "bayesian", "ml"):
                model_ids[family] = _insert_model(
                    connection, key=family, family=family,
                )
            for index in (1, 2):
                measurement_id = _insert_measurement(
                    connection, key=f"measurement-{index}",
                )
                connection.execute(
                    "INSERT INTO run_measurements "
                    "(run_id, measurement_id, data_role) VALUES (?, ?, ?)",
                    (run_id, measurement_id, "evaluation"),
                )
                _insert_result(
                    connection, run_id, measurement_id, model_ids["classical"],
                )
        rows = connection.execute(
            "SELECT family, trained_model_artifact_id FROM model_versions"
        ).fetchall()
        assert len(rows) == 3
        assert all(row[1] is None for row in rows)
        assert connection.execute(
            "SELECT COUNT(DISTINCT measurement_id) FROM estimator_results "
            "WHERE model_id = ?", (model_ids["classical"],)
        ).fetchone()[0] == 2


def test_failed_results_are_records_and_nullable_assumption_is_unique(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            run_id, measurement_id, model_id, result_id = _seed_result_graph(connection)
            connection.execute(
                "INSERT INTO fit_details (result_id, valid_fit, "
                "optimizer_reported_success, numerical_validation_passed) "
                "VALUES (?, ?, ?, ?)", (result_id, 0, 1, 0),
            )
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
                _insert_result(connection, run_id, measurement_id, model_id)
            _insert_result(connection, run_id, measurement_id, model_id,
                           analysis_key="independent-repeat")
        failure = connection.execute(
            "SELECT status, is_valid, lifetime_estimate_ns FROM estimator_results "
            "WHERE result_id = ?", (result_id,)
        ).fetchone()
        assert tuple(failure) == ("failed", 0, None)
        assert connection.execute(
            "SELECT optimizer_reported_success, numerical_validation_passed "
            "FROM fit_details WHERE result_id = ?", (result_id,)
        ).fetchone()[:] == (1, 0)


def test_rejected_bayesian_sampling_is_a_storable_result(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            run_id, measurement_id, _, _ = _seed_result_graph(connection)
            bayesian_model_id = _insert_model(connection, key="bayesian", family="bayesian")
            result_id = _insert_result(
                connection, run_id, measurement_id, bayesian_model_id,
            )
            connection.execute(
                "INSERT INTO bayesian_summaries "
                "(result_id, sampling_status, diagnostics_accepted) "
                "VALUES (?, ?, ?)",
                (result_id, "insufficient_sampling", 0),
            )
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(
                    "UPDATE bayesian_summaries SET diagnostics_accepted = 1 "
                    "WHERE result_id = ?", (result_id,),
                )
        row = connection.execute(
            "SELECT sampling_status, diagnostics_accepted FROM bayesian_summaries "
            "WHERE result_id = ?", (result_id,),
        ).fetchone()
        assert tuple(row) == ("insufficient_sampling", 0)


def test_uncertainty_kinds_and_nullable_nominal_uniqueness(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            _, _, _, result_id = _seed_result_graph(connection)
            score_values = (result_id, "rf_spread", "uncertainty_score", "{}",
                            _DIGEST, "heuristic score", "none", 1, 0.5)
            sql = (
                "INSERT INTO uncertainty_results "
                "(result_id, method_id, output_kind, method_config_json, "
                "method_config_sha256, interpretation, calibration_scope, "
                "is_valid, uncertainty_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            connection.execute(sql, score_values)
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
                connection.execute(sql, score_values)
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(
                    "INSERT INTO uncertainty_results "
                    "(result_id, method_id, output_kind, method_config_json, "
                    "method_config_sha256, interpretation, calibration_scope, "
                    "is_valid, interval_kind, nominal_coverage, lower_ns, "
                    "upper_ns, uncertainty_score) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (result_id, "interval", "prediction_interval", "{}", _DIGEST,
                     "nominal interval", "held-out", 1, "prediction", 0.9,
                     1.0, 3.0, 0.5),
                )


def test_experimental_reference_and_pseudo_true_scopes_differ(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            measurement_id = _insert_measurement(connection)
            connection.execute(
                "UPDATE measurements SET source_type = 'experimental' "
                "WHERE measurement_id = ?", (measurement_id,),
            )
            connection.execute(
                "INSERT INTO lifetime_references "
                "(reference_key, reference_kind, reference_version, "
                "measurement_id, lifetime_ns) VALUES (?, ?, ?, ?, ?)",
                ("trusted", "trusted_experimental", "v1", measurement_id, 2.3),
            )
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(
                    "INSERT INTO lifetime_references "
                    "(reference_key, reference_kind, reference_version, "
                    "measurement_id, lifetime_ns) VALUES (?, ?, ?, ?, ?)",
                    ("wrong", "pseudo_true_mono", "v1", measurement_id, 2.3),
                )


def test_schema_initialization_failure_rolls_back_all_schema_state(tmp_path, monkeypatch):
    path = tmp_path / "interrupted.sqlite"
    original = persistence._execute_schema_statement
    calls = 0

    def fail_during_ddl(connection, statement):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected DDL failure")
        original(connection, statement)

    monkeypatch.setattr(persistence, "_execute_schema_statement", fail_during_ddl)
    with pytest.raises(RuntimeError, match="injected DDL failure"):
        persistence.initialize_database(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall() == []
    monkeypatch.setattr(persistence, "_execute_schema_statement", original)
    persistence.initialize_database(path)
    with closing(persistence.connect_database(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_late_initialization_failure_rolls_back_metadata_and_user_version(
    tmp_path, monkeypatch,
):
    path = tmp_path / "late_failure.sqlite"

    def fail_final_validation(_connection):
        raise RuntimeError("injected final validation failure")

    monkeypatch.setattr(persistence, "_validate_schema", fail_final_validation)
    with pytest.raises(RuntimeError, match="injected final validation failure"):
        persistence.initialize_database(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall() == []


def test_readonly_and_missing_database_behavior(database_path, tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        persistence.connect_database(missing)
    assert not missing.exists()

    with closing(persistence.connect_database(database_path, readonly=True)) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            _insert_model(connection)
    with closing(persistence.connect_database(database_path)) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 0
        with persistence.transaction(connection):
            _insert_model(connection)


def test_top_level_transaction_commit_and_rollback(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            _insert_model(connection, key="committed")
        assert not connection.in_transaction
        with pytest.raises(RuntimeError, match="abort"):
            with persistence.transaction(connection):
                _insert_model(connection, key="rolled-back")
                raise RuntimeError("abort")
        assert not connection.in_transaction
        keys = [row[0] for row in connection.execute(
            "SELECT model_key FROM model_versions ORDER BY model_key"
        )]
        assert keys == ["committed"]


def test_nested_savepoint_rollback_preserves_outer_work(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            _insert_model(connection, key="before")
            with pytest.raises(ValueError, match="inner abort"):
                with persistence.transaction(connection):
                    _insert_model(connection, key="inner")
                    raise ValueError("inner abort")
            _insert_model(connection, key="after")
        keys = [row[0] for row in connection.execute(
            "SELECT model_key FROM model_versions ORDER BY model_key"
        )]
        assert keys == ["after", "before"]


def test_nested_success_is_still_rolled_back_with_outer_transaction(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with pytest.raises(RuntimeError, match="outer abort"):
            with persistence.transaction(connection):
                _insert_model(connection, key="outer")
                with persistence.transaction(connection):
                    _insert_model(connection, key="inner")
                raise RuntimeError("outer abort")
        assert connection.execute(
            "SELECT COUNT(*) FROM model_versions"
        ).fetchone()[0] == 0


def test_benchmark_metric_status_and_scope_uniqueness(database_path):
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            run_id = _insert_run(connection)
            statement = (
                "INSERT INTO benchmark_metrics "
                "(run_id, metric_name, metric_unit, scope_sha256, scope_json, "
                "value_status, metric_value, n_attempted, denominator_kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            undefined = (run_id, "coverage", "fraction", _DIGEST, "{}",
                         "undefined", None, 0, "attempted")
            connection.execute(statement, undefined)
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
                connection.execute(statement, undefined)
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                connection.execute(statement, (
                    run_id, "mae", "ns", _DIGEST, "{}", "finite", None,
                    1, "valid",
                ))


def test_parameter_binding_does_not_execute_user_sql(database_path):
    hostile = "model'); DROP TABLE model_versions; --"
    with closing(persistence.connect_database(database_path)) as connection:
        with persistence.transaction(connection):
            _insert_model(connection, key=hostile)
        assert connection.execute(
            "SELECT COUNT(*) FROM model_versions WHERE model_key = ?", (hostile,)
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM model_versions"
        ).fetchone()[0] == 1


def test_canonical_hashes_ignore_byte_order_width_strides_and_signed_zero():
    time = np.array([0.0, 0.25, 0.5], dtype="<f8")
    assert persistence._hash_time_grid_ns(time) == persistence._hash_time_grid_ns(
        np.array([0.0, 0.25, 0.5], dtype=">f8")
    )
    assert persistence._hash_time_grid_ns(time) == persistence._hash_time_grid_ns(
        np.array([0.0, 0.25, 0.5], dtype="<f4")
    )
    assert persistence._hash_time_grid_ns(time) == persistence._hash_time_grid_ns(
        np.array([0.0, -1.0, 0.25, -1.0, 0.5])[::2]
    )
    values = np.array([0.0, -0.0, 1.5], dtype="<f8")
    positive_zero = np.array([0.0, 0.0, 1.5], dtype=">f8")
    assert persistence._hash_continuous_histogram_values(values) == (
        persistence._hash_continuous_histogram_values(positive_zero)
    )
    for helper in (persistence._hash_irf_source_values,
                   persistence._hash_prepared_kernel):
        assert helper(np.array([0.0, -0.0, 1.5], dtype="<f4")) == helper(
            positive_zero
        )
    raw = np.array([0, 3, 12], dtype="<i8")
    expected = hashlib.sha256(raw.tobytes()).hexdigest()
    assert persistence._hash_raw_counts(raw) == expected
    for equivalent in (
        np.array([0, 3, 12], dtype=">i8"),
        np.array([0, 3, 12], dtype="<i4"),
        np.array([0, 3, 12], dtype="<f8"),
        np.array([0, 99, 3, 99, 12])[::2],
    ):
        assert persistence._hash_raw_counts(equivalent) == expected


def test_canonical_hash_rejects_invalid_scientific_arrays():
    with pytest.raises(ValueError, match="strictly increasing"):
        persistence._hash_time_grid_ns([0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        persistence._hash_time_grid_ns([0.0, float("nan")])
    with pytest.raises(ValueError, match="nonnegative"):
        persistence._hash_irf_source_values([1.0, -0.1])
    with pytest.raises(ValueError, match="nonnegative"):
        persistence._hash_prepared_kernel([1.0, -0.1])
    with pytest.raises(ValueError, match="finite"):
        persistence._hash_continuous_histogram_values([1.0, float("inf")])
    with pytest.raises(ValueError, match="exact integers"):
        persistence._hash_raw_counts([1.5, 2.0])
    with pytest.raises(ValueError, match="int64 range"):
        persistence._hash_raw_counts(np.array([2**63], dtype=np.uint64))
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        persistence._hash_raw_counts(np.array([[1, 2]]))


def test_frozen_issue4_float_hash_does_not_rewrite_signed_zero():
    historical = np.array([-0.0, 1.0], dtype="<f8")
    expected = hashlib.sha256(historical.tobytes()).hexdigest()
    assert persistence._hash_legacy_issue4_float64_array(historical) == expected
    assert persistence._hash_continuous_histogram_values(historical) != expected


def test_scientific_import_does_not_import_or_initialize_persistence(tmp_path):
    code = (
        "import sqlite3, sys\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('unexpected database connection')\n"
        "sqlite3.connect = forbidden\n"
        "import tcspc_toolkit\n"
        "import tcspc_toolkit.forward_model\n"
        "assert 'tcspc_toolkit.persistence' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []
