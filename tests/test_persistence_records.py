"""Issue-9 Stage-2 scientific identity and provenance adapters."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit import persistence
from tcspc_toolkit.bayesian import (
    BayesianPriorConfig, BoundedUniformPrior, GammaPrior, LogNormalPrior,
)
from tcspc_toolkit.bayesian_evaluation import BayesianClassicalCondition
from tcspc_toolkit.bayesian_mismatch_evaluation import (
    MismatchAssumption, MismatchCondition, PseudoTrueReference,
)
from tcspc_toolkit.bayesian_sampling import BayesianSamplingConfig
from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
from tcspc_toolkit.irf import (
    IRFProfile, IRFSourceKind, generate_emg_irf_profile, generate_gaussian_irf_profile,
)
from tcspc_toolkit.irf_estimation import estimate_irf_from_leading_edge
from tcspc_toolkit.irf_preparation import irf_profile_from_sampled_irf, prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, SampledIRF, TCSPCMeasurement
from tcspc_toolkit.ml_evaluation import BenchmarkMeasurements, BenchmarkDataset


_STAMP = "2026-10-01T10:00:00+00:00"


@pytest.fixture
def db():
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        persistence.initialize_database(connection)
        connection.row_factory = sqlite3.Row
        yield connection


def _run(connection, key="run-a", **changes):
    args = dict(
        run_key=key, run_type="scientific_test", status="complete",
        recorded_at_utc=_STAMP, origin="new", package_version="0.7.0",
        configuration={"protocol": "test", "settings": {"b": 2, "a": 1}},
    )
    args.update(changes)
    return persistence.record_run(connection, **args)


def _grid():
    return np.arange(0.0, 2.25, 0.25)


def _profile():
    return generate_gaussian_irf_profile(
        _grid(), gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4,
        provenance={"source": "fixture"},
    )


def _prepared(connection, source_key="gaussian", preparation_key="prep"):
    prepared = prepare_irf(_profile(), _grid())
    prepared_id = persistence.record_prepared_irf(
        connection, prepared, preparation_key=preparation_key, source_key=source_key,
    )
    return prepared, prepared_id


def _raw_measurement(*, sample_id="shared", irf=None):
    return TCSPCMeasurement(
        time_ns=_grid(), values=np.arange(1, 10, dtype=np.int64),
        data_kind=MeasurementDataKind.RAW_COUNTS,
        irf=irf, sample_id=sample_id,
        metadata={"instrument": "demo"}, provenance={"source": "fixture"},
    )


def _condition(connection, prepared_id=None, key="condition-a", **changes):
    args = dict(
        condition_key=key, condition_id="local-condition",
        generating_model="monoexponential", mono_lifetime_ns=1.8,
        signal_photon_count=1000, background_per_bin=0.2,
        true_temporal_shift_ns=0.05, generating_irf_id=prepared_id,
        true_reconvolution_amplitude=100.0,
    )
    args.update(changes)
    return persistence.record_simulation_condition(connection, **args)


def _priors(*, fixed=False, lower=-0.25, upper=0.25):
    return BayesianPriorConfig(
        amplitude=GammaPrior(shape=2.0, rate=0.01),
        lifetime_ns=LogNormalPrior(log_mean=0.0, log_std=0.5),
        background_per_bin=GammaPrior(shape=2.0, rate=10.0),
        temporal_shift_prior=None if fixed else BoundedUniformPrior(lower, upper),
        fixed_temporal_shift_ns=0.0 if fixed else None,
    )


def test_run_provenance_configuration_seed_and_duplicate_policy(db):
    unsigned_seed = 2**64 - 1
    run_id = _run(db, base_seed=unsigned_seed, dependency_versions={"numpy": "2.x"})
    row = db.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)).fetchone()
    assert row["base_seed_decimal"] == str(unsigned_seed)
    assert row["producing_git_commit"] is None
    assert row["config_json"] == '{"protocol":"test","settings":{"a":1,"b":2}}'
    assert row["config_sha256"] == hashlib.sha256(row["config_json"].encode()).hexdigest()
    assert _run(
        db, base_seed=str(unsigned_seed), dependency_versions={"numpy": "2.x"},
        configuration={"settings": {"a": 1, "b": 2}, "protocol": "test"},
        on_duplicate="reuse_identical",
    ) == run_id
    with pytest.raises(persistence.PersistenceConflictError, match="duplicate"):
        _run(db)
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        _run(db, base_seed=unsigned_seed, configuration={"protocol": "changed"},
             dependency_versions={"numpy": "2.x"}, on_duplicate="reuse_identical")


def test_editable_and_historical_run_evidence(db):
    edited = _run(db, "edited", editable_source=True, git_commit="actual-commit")
    release = _run(db, "release", code_revision=None, git_commit=None)
    historical = _run(
        db, "historical", origin="historical", version_state="unknown",
        package_version=None, version_unknown_reason="original manifest absent",
    )
    assert db.execute(
        "SELECT producing_git_commit FROM experiment_runs WHERE run_id = ?", (edited,),
    ).fetchone()[0] == "actual-commit"
    assert db.execute(
        "SELECT producing_git_commit FROM experiment_runs WHERE run_id = ?", (release,),
    ).fetchone()[0] is None
    assert tuple(db.execute(
        "SELECT version_state, producing_package_version, producing_git_commit "
        "FROM experiment_runs WHERE run_id = ?", (historical,),
    ).fetchone()) == ("unknown", None, None)
    with pytest.raises(ValueError, match="editable source"):
        _run(db, "edited-without-revision", editable_source=True)
    with pytest.raises(ValueError, match="unknown producer"):
        _run(db, "bad-history", origin="historical", version_state="unknown",
             version_unknown_reason="lost")
    with pytest.raises(ValueError, match="leading zeros"):
        _run(db, "bad-seed", base_seed="0005")


def test_artifact_file_hash_path_and_locator(tmp_path):
    db_path = tmp_path / "results.sqlite"
    file_path = tmp_path / "curve.npy"
    file_path.write_bytes(b"external artifact bytes\n")
    persistence.initialize_database(db_path)
    with closing(persistence.connect_database(db_path)) as connection:
        artifact_id = persistence.record_artifact(
            connection, artifact_key="curve-file", path="curve.npy",
            path_base="database_directory", artifact_kind="histogram",
            format="npy", locator={"dtype": "<i8", "shape": [9]},
        )
        row = connection.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,),
        ).fetchone()
        assert row["sha256"] == hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert row["byte_size"] == file_path.stat().st_size
        assert json.loads(row["locator_json"]) == {"dtype": "<i8", "shape": [9]}
        assert row["path"] == "curve.npy"
        assert file_path.read_bytes() == b"external artifact bytes\n"
        assert persistence.record_artifact(
            connection, artifact_key="curve-file", path="curve.npy",
            path_base="database_directory", artifact_kind="histogram",
            format="npy", locator={"shape": [9], "dtype": "<i8"},
            on_duplicate="reuse_identical",
        ) == artifact_id
        with pytest.raises(ValueError, match="relative"):
            persistence.record_artifact(
                connection, artifact_key="bad", path=file_path,
                path_base="database_directory", artifact_kind="histogram",
                format="npy", sha256=row["sha256"],
            )


def test_artifact_known_hash_can_refer_to_unavailable_absolute_file(db, tmp_path):
    missing = tmp_path / "missing-chain.nc"
    artifact_id = persistence.record_artifact(
        db, artifact_key="historical-chain", path=missing,
        path_base="absolute", artifact_kind="posterior_chain", format="netcdf",
        sha256="a" * 64, byte_size=123,
    )
    assert db.execute(
        "SELECT path, byte_size FROM artifacts WHERE artifact_id = ?", (artifact_id,),
    ).fetchone()[:] == (str(missing), 123)
    assert not missing.exists()


def test_mono_bi_and_other_conditions_keep_truth_separate(db):
    _, prepared_id = _prepared(db)
    mono_id = _condition(db, prepared_id)
    bi_id = _condition(
        db, prepared_id, key="bi", generating_model="biexponential",
        mono_lifetime_ns=None, primary_lifetime_ns=1.0,
        secondary_lifetime_ns=3.0, secondary_detected_fraction=0.25,
        generating_parameters={"mixture_convention": "detected"},
    )
    other_id = _condition(
        db, prepared_id, key="other", generating_model="other",
        mono_lifetime_ns=None, generating_parameters={"components": [1, 2, 3]},
    )
    assert db.execute("SELECT mono_lifetime_ns FROM simulation_conditions WHERE condition_pk = ?",
                      (mono_id,)).fetchone()[0] == 1.8
    bi = db.execute("SELECT mono_lifetime_ns, primary_lifetime_ns, "
                    "secondary_lifetime_ns, secondary_detected_fraction, "
                    "true_reconvolution_amplitude FROM simulation_conditions "
                    "WHERE condition_pk = ?", (bi_id,)).fetchone()
    assert tuple(bi) == (None, 1.0, 3.0, 0.25, 100.0)
    assert json.loads(db.execute(
        "SELECT generating_parameters_json FROM simulation_conditions WHERE condition_pk = ?",
        (other_id,),
    ).fetchone()[0]) == {"components": [1, 2, 3]}
    assert db.execute("SELECT COUNT(*) FROM lifetime_references").fetchone()[0] == 0
    with pytest.raises(ValueError, match="bi-exponential truth"):
        _condition(db, prepared_id, key="incomplete-bi", generating_model="biexponential",
                   mono_lifetime_ns=None, primary_lifetime_ns=1.0)


def test_issue4_condition_objects_preserve_truth_and_irf_identity(db):
    prepared, prepared_id = _prepared(db)
    matched = BayesianClassicalCondition(
        condition_id="matched-local", time_ns=_grid(),
        generating_irf=prepared, assumed_irf=prepared, true_lifetime_ns=1.5,
        signal_photon_count=500, background_per_bin=0.1,
        true_temporal_shift_ns=0.0, n_repeats=2,
    )
    matched_id = persistence.record_issue4_condition(
        db, matched, condition_key="matched-global", generating_irf_id=prepared_id,
    )
    mismatch = MismatchCondition(
        condition_id="mismatch-local", mechanism="decay",
        generating_model="biexponential", time_ns=_grid(),
        generating_irf_id="local-irf", generating_irf=prepared,
        assumptions=(MismatchAssumption("local-assumption", "local-irf", prepared),),
        primary_lifetime_ns=1.0, secondary_lifetime_ns=3.0,
        secondary_detected_fraction=0.2, signal_photon_count=500,
        background_per_bin=0.1, true_temporal_shift_ns=0.0,
    )
    mismatch_id = persistence.record_issue4_condition(
        db, mismatch, condition_key="mismatch-global", generating_irf_id=prepared_id,
    )
    assert db.execute("SELECT mono_lifetime_ns FROM simulation_conditions "
                      "WHERE condition_pk = ?", (matched_id,)).fetchone()[0] == 1.5
    row = db.execute("SELECT mono_lifetime_ns, primary_lifetime_ns, "
                     "secondary_lifetime_ns, generating_parameters_json "
                     "FROM simulation_conditions WHERE condition_pk = ?",
                     (mismatch_id,)).fetchone()
    assert row[0:3] == (None, 1.0, 3.0)
    assert json.loads(row[3])["mechanism"] == "decay"
    assert "assumption_ids" not in json.loads(row[3])
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_issue4_condition(
            db, matched, condition_key="wrong-irf", generating_irf_id=999,
        )


def test_raw_and_processed_measurements_keep_distinct_semantics(db):
    raw = _raw_measurement()
    raw_id = persistence.record_measurement(db, raw, measurement_key="exp/session-1")
    raw_row = db.execute(
        "SELECT * FROM measurements WHERE measurement_id = ?", (raw_id,),
    ).fetchone()
    assert raw_row["data_kind"] == "raw_counts"
    assert raw_row["source_type"] == "experimental"
    assert raw_row["condition_pk"] is None
    assert raw_row["observed_total_counts"] == 45
    assert raw_row["values_sha256"] == hashlib.sha256(
        raw.values.astype("<i8").tobytes()
    ).hexdigest()
    assert json.loads(raw_row["provenance_json"]) == {"source": "fixture"}
    assert persistence.record_measurement(
        db, raw, measurement_key="exp/session-1", on_duplicate="reuse_identical",
    ) == raw_id

    processed = TCSPCMeasurement(
        time_ns=_grid(), values=np.linspace(-0.5, 1.5, 9),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
        sample_id="shared", metadata={"lifetime_true_ns": 2.0},
    )
    processed_id = persistence.record_measurement(
        db, processed, measurement_key="exp/processed",
    )
    processed_row = db.execute(
        "SELECT observed_total_counts, data_kind, condition_pk, metadata_json "
        "FROM measurements WHERE measurement_id = ?", (processed_id,),
    ).fetchone()
    assert tuple(processed_row[:3]) == (None, "processed_intensity", None)
    assert json.loads(processed_row[3]) == {"lifetime_true_ns": 2.0}
    with pytest.raises(ValueError, match="experimental measurements"):
        persistence.record_measurement(
            db, raw, measurement_key="bad-experimental-truth", condition_pk=1,
        )


def test_measurement_keys_not_sample_ids_and_independent_observations(db):
    condition_id = _condition(db)
    first = persistence.record_synthetic_observation(
        db, measurement_key="suite-a/shared/realization-0", time_ns=_grid(),
        values=np.arange(1, 10, dtype=np.int64), data_kind="raw_counts",
        condition_pk=condition_id, sample_id="shared", observation_seed=2**64 - 1,
        metadata={"lifetime_true_ns": 999.0},
    )
    second = persistence.record_synthetic_observation(
        db, measurement_key="suite-b/shared/realization-0", time_ns=_grid(),
        values=np.arange(2, 11, dtype=np.int64), data_kind="raw_counts",
        condition_pk=condition_id, sample_id="shared", observation_seed=42,
    )
    unknown = persistence.record_synthetic_observation(
        db, measurement_key="suite-c/shared", time_ns=_grid(),
        values=np.arange(3, 12, dtype=np.int64), data_kind="raw_counts",
        sample_id="shared", metadata={"lifetime_true_ns": 2.5},
    )
    assert len({first, second, unknown}) == 3
    rows = db.execute(
        "SELECT condition_pk, observation_seed_decimal FROM measurements "
        "ORDER BY measurement_id"
    ).fetchall()
    assert [row[0] for row in rows] == [condition_id, condition_id, None]
    assert rows[0][1] == str(2**64 - 1)
    assert db.execute("SELECT COUNT(*) FROM lifetime_references").fetchone()[0] == 0
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.record_synthetic_observation(
            db, measurement_key="suite-a/shared/realization-0", time_ns=_grid(),
            values=np.arange(2, 11, dtype=np.int64), data_kind="raw_counts",
            condition_pk=condition_id, sample_id="shared",
            on_duplicate="reuse_identical",
        )


def test_benchmark_raw_carriers_are_supported_without_inferred_truth(db):
    counts = np.arange(1, 10, dtype=np.int64)[None, :]
    bench = BenchmarkMeasurements(
        time=_grid(), X_histograms=counts, y=np.array([1.5]),
        metadata=pd.DataFrame({"lifetime_true_ns": [1.5]}),
    )
    benchmark_id = persistence.record_benchmark_observation(
        db, bench, sample_index=0, measurement_key="benchmark/0",
    )
    general = GeneralizationTestMeasurements(
        test_id="A", time=_grid(), X_histograms=counts,
        y=np.array([1.5]), metadata=pd.DataFrame({"test_id": ["A"]}),
    )
    general_id = persistence.record_benchmark_observation(
        db, general, sample_index=0, measurement_key="generalization/A/0",
    )
    assert benchmark_id != general_id
    assert db.execute("SELECT COUNT(*) FROM simulation_conditions").fetchone()[0] == 0
    dataset = BenchmarkDataset(
        X_features=pd.DataFrame({"feature": [1.0]}), X_histograms=counts,
        y=np.array([1.5]), metadata=pd.DataFrame({"test_id": ["A"]}),
    )
    with pytest.raises(TypeError, match="BenchmarkMeasurements"):
        persistence.record_benchmark_observation(
            db, dataset, sample_index=0, measurement_key="not-raw-proven",
        )


def test_run_membership_preserves_roles_and_local_identifiers(db):
    run_id = _run(db)
    measurement_id = persistence.record_measurement(
        db, _raw_measurement(), measurement_key="experiment/1",
    )
    membership = persistence.link_run_measurement(
        db, run_id=run_id, measurement_id=measurement_id,
        data_role="external_evaluation", test_id="A", regime_id="low_photons",
        pair_id="pair-1", realization_index=0, dataset_key="frozen-a",
        membership={"protocol": "untouched"},
    )
    assert membership == (run_id, measurement_id)
    row = db.execute("SELECT * FROM run_measurements").fetchone()
    assert row["data_role"] == "external_evaluation"
    assert (row["test_id"], row["regime_id"], row["pair_id"]) == (
        "A", "low_photons", "pair-1",
    )
    assert json.loads(row["membership_json"]) == {"protocol": "untouched"}
    assert persistence.link_run_measurement(
        db, run_id=run_id, measurement_id=measurement_id,
        data_role="external_evaluation", test_id="A", regime_id="low_photons",
        pair_id="pair-1", realization_index=0, dataset_key="frozen-a",
        membership={"protocol": "untouched"}, on_duplicate="reuse_identical",
    ) == membership
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.link_run_measurement(
            db, run_id=run_id, measurement_id=measurement_id,
            data_role="training", on_duplicate="reuse_identical",
        )


def test_imported_attached_irf_and_synthetic_source_kinds(db):
    sampled = SampledIRF(
        time_ns=_grid(), values=np.array([0, 0, 1, 3, 2, 1, 0, 0, 0], dtype=float),
        metadata={"detector": "reference"}, provenance={"file": "irf.csv"},
    )
    measurement = _raw_measurement(irf=sampled)
    measured_id = persistence.record_measurement(
        db, measurement, measurement_key="exp/with-irf",
        attached_irf_source_key="imported-reference",
    )
    attached_id = db.execute(
        "SELECT attached_irf_source_id FROM measurements WHERE measurement_id = ?",
        (measured_id,),
    ).fetchone()[0]
    attached = db.execute(
        "SELECT source_representation, source_kind, source_values_sha256 FROM irf_sources "
        "WHERE irf_source_id = ?", (attached_id,),
    ).fetchone()
    assert attached[:2] == ("sampled_irf", "imported_sampled")
    assert attached[2] == persistence._hash_irf_source_values(sampled.values)

    gaussian_id = persistence.record_irf_source(db, _profile(), source_key="gaussian")
    emg = generate_emg_irf_profile(
        _grid(), gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4,
        tail_time_ns=0.3,
    )
    emg_id = persistence.record_irf_source(db, emg, source_key="emg")
    assert db.execute("SELECT source_kind FROM irf_sources WHERE irf_source_id = ?",
                      (gaussian_id,)).fetchone()[0] == IRFSourceKind.SYNTHETIC_GAUSSIAN.value
    assert db.execute("SELECT source_kind FROM irf_sources WHERE irf_source_id = ?",
                      (emg_id,)).fetchone()[0] == IRFSourceKind.SYNTHETIC_EMG.value
    with pytest.raises(ValueError, match="exactly one"):
        persistence.record_measurement(
            db, measurement, measurement_key="exp/missing-irf-key",
        )


def test_leading_edge_failure_does_not_create_irf_source(db):
    time = np.arange(0.0, 5.05, 0.05)
    failed = estimate_irf_from_leading_edge(
        TCSPCMeasurement(
            time_ns=time, values=np.zeros(time.size, dtype=np.int64),
            data_kind=MeasurementDataKind.RAW_COUNTS,
        ),
        background_window_ns=(0.0, 1.0),
        rising_edge_window_ns=(1.0, 3.0),
        smoothing_window_bins=9,
    )
    assert failed.profile is None
    with pytest.raises(ValueError, match="failed leading-edge"):
        persistence.record_irf_source(
            db, failed, source_key="failed", derived_from_measurement_id=1,
        )
    assert db.execute("SELECT COUNT(*) FROM irf_sources").fetchone()[0] == 0


def test_leading_edge_proxy_is_not_mislabeled_measured(db):
    measurement_id = persistence.record_measurement(
        db, _raw_measurement(), measurement_key="measured-fluorescence",
    )
    profile = IRFProfile(
        time_ns=_grid(), values=np.array([0, 0, 1, 3, 2, 1, 0, 0, 0], dtype=float),
        source_kind=IRFSourceKind.LEADING_EDGE_ESTIMATE,
        source_parameters={"method": "derivative_proxy"},
        provenance={"caution": "approximate"},
    )
    proxy_id = persistence.record_irf_source(
        db, profile, source_key="estimated-proxy",
        derived_from_measurement_id=measurement_id,
    )
    row = db.execute(
        "SELECT source_kind, derived_from_measurement_id, provenance_json "
        "FROM irf_sources WHERE irf_source_id = ?", (proxy_id,),
    ).fetchone()
    assert row[:2] == ("leading_edge_estimate", measurement_id)
    assert json.loads(row[2]) == {"caution": "approximate"}
    with pytest.raises(ValueError, match="source measurement"):
        persistence.record_irf_source(db, profile, source_key="unlinked-proxy")


def test_successful_leading_edge_result_preserves_construction_diagnostics(db):
    time = np.arange(0.0, 5.05, 0.05)
    counts = np.rint(
        10.0 + 300.0 * np.exp(-0.5 * ((time - 2.2) / 0.3) ** 2)
    ).astype(np.int64)
    measurement = TCSPCMeasurement(
        time_ns=time, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id="proxy-input",
    )
    measurement_id = persistence.record_measurement(
        db, measurement, measurement_key="proxy/input",
    )
    result = estimate_irf_from_leading_edge(
        measurement, background_window_ns=(0.0, 1.0),
        rising_edge_window_ns=(1.0, 3.0), smoothing_window_bins=9,
    )
    assert result.profile is not None
    source_id = persistence.record_irf_source(
        db, result, source_key="proxy/success",
        derived_from_measurement_id=measurement_id,
    )
    row = db.execute(
        "SELECT source_kind, provenance_json FROM irf_sources WHERE irf_source_id = ?",
        (source_id,),
    ).fetchone()
    assert row[0] == "leading_edge_estimate"
    assert json.loads(row[1])["leading_edge_diagnostics"]["status"] == "proxy_constructed"
    assert persistence.record_irf_source(
        db, result, source_key="proxy/success",
        derived_from_measurement_id=measurement_id,
        on_duplicate="reuse_identical",
    ) == source_id
    prepared_id = persistence.record_prepared_irf(
        db, prepare_irf(result.profile, time),
        preparation_key="proxy/prepared", irf_source_id=source_id,
    )
    assert prepared_id > 0


def test_legacy_source_and_supplied_kernel_keep_incomplete_provenance(db):
    profile = _profile()
    source_id = persistence.record_legacy_irf_source(
        db, source_key="old/bare", time_ns=profile.time_ns,
        values=profile.values, provenance={"reason": "source-kind missing"},
    )
    assert db.execute(
        "SELECT source_representation, source_kind FROM irf_sources "
        "WHERE irf_source_id = ?", (source_id,),
    ).fetchone()[:] == ("bare_array", None)
    prepared = prepare_irf(profile, _grid())
    kernel_id = persistence.record_supplied_kernel(
        db, preparation_key="old/supplied", time_ns=prepared.time_ns,
        kernel=prepared.kernel, provenance={"source": "legacy-file"},
    )
    row = db.execute(
        "SELECT preparation_kind, irf_source_id, flags_json FROM prepared_irfs "
        "WHERE prepared_irf_id = ?", (kernel_id,),
    ).fetchone()
    assert row[:2] == ("supplied_kernel", None)
    assert json.loads(row[2]) == ["source_provenance_incomplete"]


def test_one_source_two_preparations_and_diagnostics(db):
    profile = _profile()
    source_id = persistence.record_irf_source(db, profile, source_key="one-source")
    fine = prepare_irf(profile, _grid())
    coarse_grid = np.arange(0.0, 2.5, 0.5)
    coarse = prepare_irf(
        profile, coarse_grid, resampling="linear", registration_offset_ns=0.1,
    )
    fine_id = persistence.record_prepared_irf(
        db, fine, preparation_key="fine", irf_source_id=source_id,
    )
    coarse_id = persistence.record_prepared_irf(
        db, coarse, preparation_key="coarse", irf_source_id=source_id,
    )
    assert fine_id != coarse_id
    rows = db.execute(
        "SELECT irf_source_id, time_grid_sha256, registration_offset_ns, "
        "resampling_method, diagnostics_json, operations_json "
        "FROM prepared_irfs ORDER BY prepared_irf_id"
    ).fetchall()
    assert rows[0][0] == rows[1][0] == source_id
    assert rows[0][1] != rows[1][1]
    assert rows[1][2] == pytest.approx(0.1)
    assert rows[1][3] == coarse.diagnostics.resampling_method
    assert json.loads(rows[1][4])["registration_offset_ns"] == pytest.approx(0.1)
    assert json.loads(rows[1][5]) == list(coarse.diagnostics.operations)
    assert persistence.record_prepared_irf(
        db, fine, preparation_key="fine", irf_source_id=source_id,
        on_duplicate="reuse_identical",
    ) == fine_id


def test_existing_source_id_requires_matching_source_identity(db):
    original = _profile()
    source_id = persistence.record_irf_source(db, original, source_key="identity/source")
    same = replace(original, provenance=dict(original.provenance))
    accepted = persistence.record_prepared_irf(
        db, prepare_irf(same, _grid()), preparation_key="identity/same",
        irf_source_id=source_id,
    )
    assert accepted > 0
    conflicting_provenance = replace(original, provenance={"source": "other-calibration"})
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_prepared_irf(
            db, prepare_irf(conflicting_provenance, _grid()),
            preparation_key="identity/wrong-provenance", irf_source_id=source_id,
        )
    conflicting_parameters = replace(
        original, source_parameters={**original.source_parameters, "calibration": "other"},
    )
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_prepared_irf(
            db, prepare_irf(conflicting_parameters, _grid()),
            preparation_key="identity/wrong-parameters", irf_source_id=source_id,
        )
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_prepared_irf(
            db, prepare_irf(replace(original, metadata={"detector": "different"}), _grid()),
            preparation_key="identity/wrong-metadata", irf_source_id=source_id,
        )
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.record_irf_source(
            db, conflicting_provenance, source_key="identity/source",
            on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM prepared_irfs").fetchone()[0] == 1


def test_imported_sampled_source_and_equivalent_profile_share_identity(db):
    sampled = SampledIRF(
        time_ns=_grid(), values=np.array([0, 0, 1, 3, 2, 1, 0, 0, 0], dtype=float),
        metadata={"detector": "reference"}, provenance={"file": "source.csv"},
    )
    source_id = persistence.record_irf_source(db, sampled, source_key="imported/source")
    profile = irf_profile_from_sampled_irf(sampled)
    assert persistence.record_irf_source(
        db, profile, source_key="imported/source", on_duplicate="reuse_identical",
    ) == source_id
    prepared_id = persistence.record_prepared_irf(
        db, prepare_irf(profile, _grid()), preparation_key="imported/prepared",
        source_key="imported/source", on_duplicate="reuse_identical",
    )
    assert db.execute(
        "SELECT irf_source_id FROM prepared_irfs WHERE prepared_irf_id = ?",
        (prepared_id,),
    ).fetchone()[0] == source_id
    assert db.execute(
        "SELECT source_representation FROM irf_sources WHERE irf_source_id = ?",
        (source_id,),
    ).fetchone()[0] == "sampled_irf"


def test_attached_source_conflict_rejects_measurement_reuse(db):
    sampled = SampledIRF(
        time_ns=_grid(), values=np.array([0, 0, 1, 3, 2, 1, 0, 0, 0], dtype=float),
        provenance={"file": "calibration-A.csv"},
    )
    measurement = _raw_measurement(irf=sampled)
    source_id = persistence.record_irf_source(db, sampled, source_key="attached/A")
    measurement_id = persistence.record_measurement(
        db, measurement, measurement_key="observed/1",
        attached_irf_source_id=source_id,
    )
    other = replace(sampled, provenance={"file": "calibration-B.csv"})
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_measurement(
            db, replace(measurement, irf=other), measurement_key="observed/1",
            attached_irf_source_id=source_id, on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1
    assert measurement_id > 0


def test_issue4_condition_checks_source_and_preparation_history(db):
    prepared, prepared_id = _prepared(db)

    def condition(irf):
        return BayesianClassicalCondition(
            condition_id="identity-condition", time_ns=_grid(),
            generating_irf=irf, assumed_irf=irf, true_lifetime_ns=1.5,
            signal_photon_count=500, background_per_bin=0.1,
            true_temporal_shift_ns=0.0, n_repeats=2,
        )

    recorded = persistence.record_issue4_condition(
        db, condition(prepared), condition_key="identity/condition",
        generating_irf_id=prepared_id,
    )
    assert persistence.record_issue4_condition(
        db, condition(prepared), condition_key="identity/condition",
        generating_irf_id=prepared_id, on_duplicate="reuse_identical",
    ) == recorded
    different_source = prepare_irf(
        replace(prepared.source, provenance={"source": "different"}), _grid(),
    )
    np.testing.assert_array_equal(different_source.kernel, prepared.kernel)
    different_source_id = persistence.record_irf_source(
        db, different_source.source, source_key="identity/different-source",
    )
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.record_prepared_irf(
            db, different_source, preparation_key="prep",
            irf_source_id=different_source_id, on_duplicate="reuse_identical",
        )
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_issue4_condition(
            db, condition(different_source), condition_key="identity/condition",
            generating_irf_id=prepared_id, on_duplicate="reuse_identical",
        )
    different_history = replace(
        prepared, diagnostics=replace(
            prepared.diagnostics, registration_offset_ns=0.125,
        ),
    )
    with pytest.raises(ValueError, match="does not identify"):
        persistence.record_issue4_condition(
            db, condition(different_history), condition_key="identity/condition",
            generating_irf_id=prepared_id, on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM simulation_conditions").fetchone()[0] == 1


def test_prepared_irf_conflicting_reuse_rolls_back_new_source(db):
    prepared, _ = _prepared(db, "source/A", "prepared/A")
    conflicting = prepare_irf(
        replace(prepared.source, provenance={"source": "different"}), _grid(),
    )
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        persistence.record_prepared_irf(
            db, conflicting, source_key="source/temporary",
            preparation_key="prepared/A", on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM irf_sources").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM prepared_irfs").fetchone()[0] == 1


def test_generating_attached_and_assumed_irfs_are_independent(db):
    _, generating = _prepared(db, "generated-source", "generated-prep")
    attached = SampledIRF(
        time_ns=_grid(), values=np.array([0, 0, 1, 4, 1, 0, 0, 0, 0], dtype=float),
    )
    measured = _raw_measurement(irf=attached)
    condition = _condition(db, generating)
    measurement_id = persistence.record_measurement(
        db, measured, measurement_key="synthetic/with-attached",
        source_type="synthetic", condition_pk=condition,
        attached_irf_source_key="attached-imported",
    )
    assumed_profile = generate_emg_irf_profile(
        _grid(), gaussian_centre_ns=0.75, gaussian_fwhm_ns=0.4,
        tail_time_ns=0.3,
    )
    assumed_id = persistence.record_prepared_irf(
        db, prepare_irf(assumed_profile, _grid()),
        source_key="deliberately-wrong-source", preparation_key="wrong-prep",
    )
    assumption = persistence.record_model_assumption(
        db, assumption_key="wrong-physical-model", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=assumed_id,
        fixed_temporal_shift_ns=0.0,
    )
    row = db.execute(
        "SELECT c.generating_irf_id, m.attached_irf_source_id, a.prepared_irf_id "
        "FROM measurements AS m JOIN simulation_conditions AS c ON c.condition_pk = m.condition_pk "
        "CROSS JOIN model_assumptions AS a WHERE m.measurement_id = ? AND a.assumption_id = ?",
        (measurement_id, assumption),
    ).fetchone()
    assert row[0] == generating and row[2] == assumed_id
    assert row[0] != row[2] and row[1] is not None


def test_reusable_model_specs_classical_ml_bayesian(db, tmp_path):
    classical_id = persistence.record_model_version(
        db, model_key="classical/reconvolution-v1", estimator_name="poisson_reconvolution",
        family="classical", configuration={"optimizer": "L-BFGS-B"},
        implementation_version="0.7.0",
    )
    assert db.execute(
        "SELECT trained_model_artifact_id FROM model_versions WHERE model_id = ?",
        (classical_id,),
    ).fetchone()[0] is None
    baseline_id = persistence.record_model_version(
        db, model_key="baseline/mean-arrival", estimator_name="mean_arrival_time",
        family="baseline", configuration={"offset_ns": 0.0},
    )
    assert baseline_id != classical_id
    train_run = _run(db, "training-run", run_type="training")
    model_artifact = persistence.record_artifact(
        db, artifact_key="trained-rf", path=tmp_path / "rf.joblib", path_base="absolute",
        artifact_kind="trained_model", format="joblib", sha256="b" * 64,
    )
    ml_id = persistence.record_model_version(
        db, model_key="rf/spec", estimator_name="random_forest", family="ml",
        configuration={"n_estimators": 100}, representation_id="features-v1",
        training_run_id=train_run, trained_model_artifact_id=model_artifact,
    )
    assert db.execute(
        "SELECT training_run_id, trained_model_artifact_id FROM model_versions "
        "WHERE model_id = ?", (ml_id,),
    ).fetchone()[:] == (train_run, model_artifact)
    priors = _priors()
    sampler = BayesianSamplingConfig(random_seed=2**64 - 1)
    config = persistence.bayesian_model_configuration(priors, sampler)
    assert config["priors"]["temporal_shift_prior"] == {
        "lower_ns": -0.25, "upper_ns": 0.25,
    }
    assert "random_seed" not in config["sampler"]
    bayes_id = persistence.record_model_version(
        db, model_key="bayes/policy-a", estimator_name="emcee_poisson",
        family="bayesian", configuration=config, prior_policy_id="policy-a",
    )
    row = db.execute(
        "SELECT configuration_json, prior_policy_id, trained_model_artifact_id "
        "FROM model_versions WHERE model_id = ?", (bayes_id,),
    ).fetchone()
    assert json.loads(row[0]) == config
    assert row[1:] == ("policy-a", None)
    assert db.execute("SELECT COUNT(*) FROM estimator_results").fetchone()[0] == 0
    with pytest.raises(ValueError, match="prior/sampler"):
        persistence.record_model_version(
            db, model_key="bayes/empty", estimator_name="emcee_poisson",
            family="bayesian", configuration={}, prior_policy_id="policy-b",
        )


def test_bayesian_recording_boundary_excludes_execution_seed_only(db):
    priors = _priors()
    sampler = BayesianSamplingConfig(random_seed=17)

    def record_bayesian(key, selected_priors, selected_sampler, **extra):
        return persistence.record_model_version(
            db, model_key=key, estimator_name="emcee_poisson", family="bayesian",
            configuration={"priors": selected_priors, "sampler": selected_sampler},
            prior_policy_id="policy-a", **extra,
        )

    first = record_bayesian("bayes/direct", priors, sampler)
    assert record_bayesian(
        "bayes/direct", priors, replace(sampler, random_seed=2**64 - 1),
        on_duplicate="reuse_identical",
    ) == first
    reused = record_bayesian(
        "bayes/another-execution", priors, replace(sampler, random_seed=99),
    )
    rows = db.execute(
        "SELECT model_id, configuration_json, configuration_sha256 FROM model_versions "
        "WHERE model_id IN (?, ?) ORDER BY model_id", (first, reused),
    ).fetchall()
    assert rows[0][1:] == rows[1][1:]
    stored = json.loads(rows[0][1])
    assert "random_seed" not in stored["sampler"]
    assert stored["sampler"]["n_walkers"] == sampler.n_walkers
    assert stored["sampler"]["production_steps"] == sampler.production_steps

    changed_sampler = record_bayesian(
        "bayes/different-policy", priors, replace(sampler, n_walkers=64),
    )
    changed_prior = record_bayesian(
        "bayes/different-prior", replace(
            priors, lifetime_ns=LogNormalPrior(log_mean=0.1, log_std=0.5),
        ), sampler,
    )
    hashes = [row[0] for row in db.execute(
        "SELECT configuration_sha256 FROM model_versions "
        "WHERE model_id IN (?, ?, ?) ORDER BY model_id",
        (first, changed_sampler, changed_prior),
    )]
    assert len(set(hashes)) == 3
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        record_bayesian(
            "bayes/direct", priors, replace(sampler, n_walkers=64),
            on_duplicate="reuse_identical",
        )

    ml_first = persistence.record_model_version(
        db, model_key="ml/seed-1", estimator_name="random_forest", family="ml",
        configuration={"n_estimators": 100, "random_seed": 1},
    )
    ml_second = persistence.record_model_version(
        db, model_key="ml/seed-2", estimator_name="random_forest", family="ml",
        configuration={"n_estimators": 100, "random_seed": 2},
    )
    ml_rows = db.execute(
        "SELECT configuration_json, configuration_sha256 FROM model_versions "
        "WHERE model_id IN (?, ?) ORDER BY model_id", (ml_first, ml_second),
    ).fetchall()
    assert [json.loads(row[0])["random_seed"] for row in ml_rows] == [1, 2]
    assert ml_rows[0][1] != ml_rows[1][1]


def test_bayesian_seed_normalization_preserves_unclassified_nested_seeds(db):
    configuration = persistence.bayesian_model_configuration(
        _priors(), BayesianSamplingConfig(random_seed=17),
    )
    configuration["sampler"]["random_seed"] = 17
    configuration["sampler"]["future_policy"] = {"random_seed": 41}
    configuration["prior_provenance"] = {"random_seed": 53}

    def record(key, content, **extra):
        return persistence.record_model_version(
            db, model_key=key, estimator_name="emcee_poisson",
            family="bayesian", configuration=content, **extra,
        )

    first = record("bayes/nested-seeds", configuration)
    changed_execution_seed = json.loads(json.dumps(configuration))
    changed_execution_seed["sampler"]["random_seed"] = 99
    assert record(
        "bayes/nested-seeds", changed_execution_seed,
        on_duplicate="reuse_identical",
    ) == first
    stored_json, base_hash = db.execute(
        "SELECT configuration_json, configuration_sha256 FROM model_versions "
        "WHERE model_id = ?", (first,),
    ).fetchone()
    stored = json.loads(stored_json)
    assert "random_seed" not in stored["sampler"]
    assert stored["sampler"]["future_policy"]["random_seed"] == 41
    assert stored["prior_provenance"]["random_seed"] == 53
    assert configuration["sampler"]["random_seed"] == 17

    changed_future_seed = json.loads(json.dumps(configuration))
    changed_future_seed["sampler"]["future_policy"]["random_seed"] = 42
    changed_prior_seed = json.loads(json.dumps(configuration))
    changed_prior_seed["prior_provenance"]["random_seed"] = 54
    for key, changed in (
        ("bayes/future-policy-seed", changed_future_seed),
        ("bayes/prior-provenance-seed", changed_prior_seed),
    ):
        changed_id = record(key, changed)
        assert db.execute(
            "SELECT configuration_sha256 FROM model_versions WHERE model_id = ?",
            (changed_id,),
        ).fetchone()[0] != base_hash


def test_complete_and_historical_physical_assumptions(db):
    _, prepared_id = _prepared(db)
    fixed = persistence.record_model_assumption(
        db, assumption_key="mono-fixed", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=prepared_id,
        fixed_temporal_shift_ns=0.0, bayesian_priors=_priors(fixed=True),
    )
    bounded = persistence.record_model_assumption(
        db, assumption_key="mono-bounded", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=prepared_id,
        temporal_shift_lower_ns=-0.25, temporal_shift_upper_ns=0.25,
        bayesian_priors=_priors(),
    )
    persistence.record_model_assumption(
        db, assumption_key="mono-bounded-narrow-prior", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=prepared_id,
        temporal_shift_lower_ns=-0.25, temporal_shift_upper_ns=0.25,
        bayesian_priors=_priors(lower=-0.1, upper=0.1),
    )
    historical = persistence.record_model_assumption(
        db, assumption_key="legacy-unknown-irf", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="unknown",
        context_completeness="historical_incomplete",
        provenance={"missing": ["IRF", "shift"]},
    )
    rows = db.execute(
        "SELECT context_completeness, prepared_irf_id, fixed_temporal_shift_ns, "
        "temporal_shift_lower_ns, temporal_shift_upper_ns FROM model_assumptions "
        "ORDER BY assumption_id"
    ).fetchall()
    assert rows[0][:3] == ("complete", prepared_id, 0.0)
    assert rows[1][3:] == (-0.25, 0.25)
    assert rows[3][:2] == ("historical_incomplete", None)
    assert len({fixed, bounded, historical}) == 3
    base = dict(
        assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=prepared_id,
    )
    for index, changes in enumerate((
        {}, {"temporal_shift_lower_ns": -0.25},
        {"temporal_shift_upper_ns": 0.25},
        {"fixed_temporal_shift_ns": 0.0, "temporal_shift_lower_ns": -0.25,
         "temporal_shift_upper_ns": 0.25},
    )):
        with pytest.raises(ValueError, match="exactly one shift mode"):
            persistence.record_model_assumption(
                db, assumption_key=f"bad-mode-{index}", **(base | changes),
            )
    with pytest.raises(ValueError, match="prepared IRF"):
        persistence.record_model_assumption(
            db, assumption_key="missing-irf", **(base | {
                "prepared_irf_id": None, "fixed_temporal_shift_ns": 0.0,
            }),
        )
    with pytest.raises(ValueError, match="conflicts with Bayesian prior"):
        persistence.record_model_assumption(
            db, assumption_key="prior-conflict", **(base | {
                "fixed_temporal_shift_ns": 0.1, "bayesian_priors": _priors(fixed=True),
            }),
        )
    with pytest.raises(ValueError, match="support exceeds"):
        persistence.record_model_assumption(
            db, assumption_key="prior-support-outside", **(base | {
                "temporal_shift_lower_ns": -0.25,
                "temporal_shift_upper_ns": 0.25,
                "bayesian_priors": _priors(lower=-0.3, upper=0.1),
            }),
        )


def test_trusted_and_corrected_pseudo_true_references(db):
    experimental_id = persistence.record_measurement(
        db, _raw_measurement(), measurement_key="experimental/no-truth",
    )
    trusted = persistence.record_trusted_lifetime_reference(
        db, reference_key="external-calibration/v1", reference_version="v1",
        measurement_id=experimental_id, lifetime_ns=2.1,
        provenance={"laboratory": "independent"},
    )
    assert db.execute(
        "SELECT reference_kind, condition_pk, assumption_id FROM lifetime_references "
        "WHERE reference_id = ?", (trusted,),
    ).fetchone()[:] == ("trusted_experimental", None, None)
    assert db.execute("SELECT COUNT(*) FROM simulation_conditions").fetchone()[0] == 0

    _, irf_id = _prepared(db)
    condition = _condition(
        db, irf_id, key="bi-for-projection", generating_model="biexponential",
        mono_lifetime_ns=None, primary_lifetime_ns=1.0,
        secondary_lifetime_ns=3.0, secondary_detected_fraction=0.3,
    )
    assumption = persistence.record_model_assumption(
        db, assumption_key="projection-model", assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
        context_completeness="complete", prepared_irf_id=irf_id,
        fixed_temporal_shift_ns=0.0,
    )
    original = persistence.record_pseudo_true_reference(
        db, reference_key="projection/v1", reference_version="v1",
        condition_pk=condition, assumption_id=assumption, lifetime_ns=1.7,
        projection_amplitude=100.0, projection_background_per_bin=0.2,
        projection_temporal_shift_ns=0.0, projection_poisson_nll=24.0,
        validation={"status": "superseded numerical tolerance"},
    )
    corrected = persistence.record_pseudo_true_reference(
        db, reference_key="projection/v2", reference_version="v2",
        condition_pk=condition, assumption_id=assumption, lifetime_ns=1.8,
        supersedes_reference_id=original,
        validation={"status": "corrected"},
    )
    assert db.execute(
        "SELECT reference_kind, measurement_id, condition_pk, assumption_id, "
        "supersedes_reference_id FROM lifetime_references WHERE reference_id = ?",
        (corrected,),
    ).fetchone()[:] == ("pseudo_true_mono", None, condition, assumption, original)
    assert db.execute(
        "SELECT lifetime_ns FROM lifetime_references WHERE reference_id = ?",
        (original,),
    ).fetchone()[0] == 1.7
    assert db.execute(
        "SELECT mono_lifetime_ns FROM simulation_conditions WHERE condition_pk = ?",
        (condition,),
    ).fetchone()[0] is None
    with pytest.raises(ValueError, match="experimental measurement"):
        persistence.record_trusted_lifetime_reference(
            db, reference_key="not-experimental", reference_version="v1",
            measurement_id=999, lifetime_ns=1.0,
        )
    with pytest.raises(ValueError, match="same role and scope"):
        persistence.record_pseudo_true_reference(
            db, reference_key="wrong-supersession", reference_version="v3",
            condition_pk=_condition(db, irf_id, key="another-condition"),
            assumption_id=assumption, lifetime_ns=2.0,
            supersedes_reference_id=original,
        )


def test_pseudo_true_projection_shift_matches_physical_assumption(db):
    _, prepared_id = _prepared(db)
    condition = _condition(
        db, prepared_id, key="shift-projection-condition",
        generating_model="biexponential", mono_lifetime_ns=None,
        primary_lifetime_ns=1.0, secondary_lifetime_ns=3.0,
        secondary_detected_fraction=0.3,
    )
    common = dict(
        assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution", background_convention="per_bin",
    )
    fixed = persistence.record_model_assumption(
        db, assumption_key="projection-fixed", context_completeness="complete",
        prepared_irf_id=prepared_id, fixed_temporal_shift_ns=0.0, **common,
    )
    bounded = persistence.record_model_assumption(
        db, assumption_key="projection-bounded", context_completeness="complete",
        prepared_irf_id=prepared_id, temporal_shift_lower_ns=-0.25,
        temporal_shift_upper_ns=0.25, **common,
    )
    incomplete = persistence.record_model_assumption(
        db, assumption_key="projection-historical",
        context_completeness="historical_incomplete", **common,
    )

    for key, assumption, shift in (
        ("fixed-exact", fixed, 0.0),
        ("fixed-roundoff", fixed, 5e-13),
        ("bounded-middle", bounded, 0.1),
        ("bounded-lower", bounded, -0.25),
        ("bounded-upper", bounded, 0.25),
        ("historical", incomplete, 0.8),
    ):
        reference_id = persistence.record_pseudo_true_reference(
            db, reference_key=key, reference_version="v1",
            condition_pk=condition, assumption_id=assumption, lifetime_ns=1.7,
            projection_temporal_shift_ns=shift,
        )
        assert db.execute(
            "SELECT projection_temporal_shift_ns FROM lifetime_references "
            "WHERE reference_id = ?", (reference_id,),
        ).fetchone()[0] == shift

    for key, assumption, shift in (
        ("fixed-conflict", fixed, 0.01),
        ("below-bounds", bounded, -0.25001),
        ("above-bounds", bounded, 0.25001),
    ):
        with pytest.raises(ValueError, match="projection shift"):
            persistence.record_pseudo_true_reference(
                db, reference_key=key, reference_version="v1",
                condition_pk=condition, assumption_id=assumption, lifetime_ns=1.7,
                projection_temporal_shift_ns=shift,
            )
    with pytest.raises(ValueError, match="projection shift"):
        persistence.record_pseudo_true_reference(
            db, reference_key="fixed-exact", reference_version="v1",
            condition_pk=condition, assumption_id=fixed, lifetime_ns=1.7,
            projection_temporal_shift_ns=0.01, on_duplicate="reuse_identical",
        )
    assert db.execute("SELECT COUNT(*) FROM lifetime_references").fetchone()[0] == 6


def test_issue4_pseudo_true_object_preserves_validation_and_scoped_ids(db):
    _, irf_id = _prepared(db)
    condition = _condition(db, irf_id, key="issue4-condition")
    assumption = persistence.record_model_assumption(
        db, assumption_key="issue4-assumption-global",
        source_assumption_id="issue4-assumption-local",
        assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=irf_id, fixed_temporal_shift_ns=0.0,
    )
    reference = PseudoTrueReference(
        condition_id="local-condition", assumption_id="issue4-assumption-local",
        amplitude=100.0, lifetime_ns=1.7, background_per_bin=0.2,
        temporal_shift_ns=0.0, poisson_nll=24.0,
        n_starts=4, max_start_objective_gap=1e-8,
        max_start_lifetime_gap_ns=1e-7,
        active_bounds=("background_lower",), optimizer_status=0,
        optimizer_message="converged", optimizer_nfev=30,
        maximum_coordinate_descent_nll=1e-9,
    )
    reference_id = persistence.record_issue4_pseudo_true_reference(
        db, reference, reference_key="issue4/projection/v1",
        reference_version="v1", condition_pk=condition, assumption_id=assumption,
    )
    row = db.execute(
        "SELECT lifetime_ns, projection_amplitude, validation_json "
        "FROM lifetime_references WHERE reference_id = ?", (reference_id,),
    ).fetchone()
    assert row[0:2] == (1.7, 100.0)
    validation = json.loads(row[2])
    assert validation["n_starts"] == 4
    assert validation["active_bounds"] == ["background_lower"]
    assert "second_best_objective_gap" in validation["nonfinite_fields"]
    assert validation["second_best_objective_gap"] is None
    with pytest.raises(ValueError, match="projection shift conflicts"):
        persistence.record_issue4_pseudo_true_reference(
            db, replace(reference, temporal_shift_ns=0.01),
            reference_key="issue4/projection/v1", reference_version="v1",
            condition_pk=condition, assumption_id=assumption,
            on_duplicate="reuse_identical",
        )
    bounded_assumption = persistence.record_model_assumption(
        db, assumption_key="issue4-bounded-assumption-global",
        source_assumption_id="issue4-bounded-assumption-local",
        assumed_decay_model="monoexponential",
        observation_model="poisson_reconvolution",
        background_convention="per_bin", context_completeness="complete",
        prepared_irf_id=irf_id, temporal_shift_lower_ns=-0.25,
        temporal_shift_upper_ns=0.25,
    )
    bounded_reference = replace(
        reference, assumption_id="issue4-bounded-assumption-local",
        temporal_shift_ns=0.25,
    )
    persistence.record_issue4_pseudo_true_reference(
        db, bounded_reference, reference_key="issue4/bounded/v1",
        reference_version="v1", condition_pk=condition,
        assumption_id=bounded_assumption,
    )
    with pytest.raises(ValueError, match="projection shift lies outside"):
        persistence.record_issue4_pseudo_true_reference(
            db, replace(bounded_reference, temporal_shift_ns=0.26),
            reference_key="issue4/bounded/v2", reference_version="v2",
            condition_pk=condition, assumption_id=bounded_assumption,
        )
    with pytest.raises(ValueError, match="source assumption"):
        persistence.record_issue4_pseudo_true_reference(
            db, reference, reference_key="issue4/wrong", reference_version="v1",
            condition_pk=condition, assumption_id=persistence.record_model_assumption(
                db, assumption_key="different-source-assumption",
                source_assumption_id="different-local",
                assumed_decay_model="monoexponential",
                observation_model="poisson_reconvolution",
                background_convention="per_bin", context_completeness="complete",
                prepared_irf_id=irf_id, fixed_temporal_shift_ns=0.0,
            ),
        )


def test_multi_record_rollback_and_nested_savepoints(db):
    with pytest.raises(TypeError, match="must be a TCSPCMeasurement"):
        with persistence.transaction(db):
            source_id = persistence.record_irf_source(db, _profile(), source_key="rolled-source")
            prepared = prepare_irf(_profile(), _grid())
            persistence.record_prepared_irf(
                db, prepared, preparation_key="rolled-prep", irf_source_id=source_id,
            )
            persistence.record_measurement(db, object(), measurement_key="bad")
    assert db.execute("SELECT COUNT(*) FROM irf_sources").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM prepared_irfs").fetchone()[0] == 0

    with persistence.transaction(db):
        _run(db, "outer-run")
        with pytest.raises(persistence.PersistenceConflictError):
            with persistence.transaction(db):
                _run(db, "nested-run")
                _run(db, "nested-run")
        _run(db, "outer-after-savepoint")
    assert [row[0] for row in db.execute(
        "SELECT run_key FROM experiment_runs ORDER BY run_id"
    )] == ["outer-run", "outer-after-savepoint"]


def test_multi_row_adapter_rollback_and_bound_user_strings(db):
    sampled = SampledIRF(time_ns=_grid(), values=np.arange(1, 10, dtype=float))
    raw = _raw_measurement(irf=sampled)
    with pytest.raises(sqlite3.IntegrityError):
        persistence.record_measurement(
            db, raw, measurement_key="would-roll-back",
            attached_irf_source_key="temporary-source", condition_pk=999,
            source_type="synthetic",
        )
    assert db.execute("SELECT COUNT(*) FROM irf_sources").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 0
    malicious = "sample'); DROP TABLE measurements; --"
    measurement_id = persistence.record_measurement(
        db, _raw_measurement(sample_id=malicious), measurement_key=malicious,
    )
    assert db.execute(
        "SELECT sample_id FROM measurements WHERE measurement_id = ?",
        (measurement_id,),
    ).fetchone()[0] == malicious
    assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1


def test_duplicate_reuse_compares_parsed_canonical_json(db):
    run_id = _run(db, "canonical-json")
    db.execute(
        "UPDATE experiment_runs SET config_json = ? WHERE run_id = ?",
        ('{ "settings": { "a": 1, "b": 2 }, "protocol": "test" }', run_id),
    )
    assert _run(
        db, "canonical-json", on_duplicate="reuse_identical",
    ) == run_id
    with pytest.raises(persistence.PersistenceConflictError, match="conflicting"):
        _run(db, "canonical-json", on_duplicate="reuse_identical",
             configuration={"protocol": "other"})
