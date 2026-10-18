"""Small real integration paths; no MCMC or full benchmark in ordinary pytest."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_issue12_integration.py"


@pytest.fixture(scope="module")
def report():
    # A fresh process prevents already-imported facades/emcee in the full suite
    # from defeating import hooks. This runs four small workflows twice.
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--block-emcee"], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == "", completed.stderr
    return json.loads(completed.stdout)


def test_cross_feature_determinism_and_facade_blocking(report):
    assert report["status"] == "PASS"
    assert report["runs"] == 2
    assert report["facades_blocked"] == 38


def test_raw_simulation_classical_covariance_and_factual_identity(report):
    result = report["workflows"]["synthetic"]
    assert result["processed_rejected"]
    assert abs(result["parameters"][1] - 2.) < .25
    assert 0. < result["covariance_std"] < .2
    point, interval, reference = result["points"][0], result["intervals"][0], result["references"][0]
    for key in ("evaluation_id", "sample_id", "method_id", "representation_id"):
        assert point[key] == interval[key] == reference[key]
    assert point["is_valid"] and interval["is_valid_interval"]
    assert interval["nominal_level"] == .9
    assert reference["reference_kind"] == "generating_mono"


def test_prepared_source_and_proxy_are_distinct(report):
    synthetic = report["workflows"]["synthetic"]
    imported = report["workflows"]["experimental"]
    assert synthetic["source_kind"] == "synthetic_gaussian"
    assert synthetic["source"] != synthetic["kernel"]
    assert imported["source_kind"] == "imported_sampled"
    assert imported["proxy_kind"] == "leading_edge_estimate"
    assert imported["preparation"]["target_area_after_normalization"] == pytest.approx(1.)
    assert imported["proxy_diagnostics"]["status"] == "proxy_constructed"


def test_generic_ml_training_population_and_unsorted_prediction_identity(report):
    result = report["workflows"]["ml"]
    assert len(result["train_ids"]) == 24
    assert len(result["test_ids"]) == 8
    assert not set(result["train_ids"]) & set(result["test_ids"])
    assert result["test_ids"] != sorted(result["test_ids"])
    assert np.isfinite(result["predictions"]).all()
    assert [row["sample_id"] for row in result["points"]] == [f"development-{i}" for i in result["test_ids"]]
    assert {row["method_id"] for row in result["points"]} == {"user-linear"}
    assert {row["representation_id"] for row in result["points"]} == {"features"}


def test_imported_measurement_has_no_invented_reference(report):
    result = report["workflows"]["experimental"]
    assert result["reference_count"] == 0
    assert 1.5 < result["lifetime_ns"] < 2.5
    assert np.isfinite(result["nll"])


def test_bayesian_execution_alone_requires_optional_dependency(report):
    result = report["workflows"]["experimental"]["bayesian"]
    assert result["status"] == "OPTIONAL_DEPENDENCY_ERROR"
    assert "optional emcee extra" in result["message"]


def test_mismatch_reference_is_primary_not_weighted_or_pseudo_true(report):
    result = report["workflows"]["mismatch"]
    reference = result["references"][0]
    assert reference["reference_kind"] == "primary_component"
    assert reference["reference_value_ns"] == 2.
    assert result["weighted_lifetime_descriptive_ns"] == pytest.approx(2.4)
    assert result["generating_parameters"]["secondary_lifetime_ns"] == 4.
    assert np.isfinite(result["poisson_deviance"])


def test_persistence_roundtrip_retains_both_families_and_reference_identity(report):
    result = report["workflows"]["persistence"]
    assert result["result_ids"] == list(range(1, 11))
    assert result["run_ids"] == [1] * 10
    assert result["model_ids"] == [1, 1] + [2] * 8
    assert result["method_families"] == ["classical"] * 2 + ["ml"] * 8
    assert len(result["generating_references_ns"]) == 9
    assert {row["reference_kind"] for row in result["reference_provenance"]} == {"generating_mono"}
    assert result["uncertainty_rows"] == 4
    assert result["read_only_bytes_unchanged"]


def test_invalid_fit_cannot_silently_gain_valid_covariance_or_disappear(report):
    result = report["workflows"]["persistence"]
    assert result["validity"] == [1, 0] + [1] * 8
    assert result["failed_covariance_reason"] == "fit_unsuccessful"
    summary = result["interval_summary"]
    assert summary["n_samples"] == 2
    assert summary["n_valid_intervals"] == 1
    assert summary["interval_failure_rate"] == .5


def test_cli_fit_matches_python_api_and_declares_semantics(report):
    result = report["workflows"]["cli"]
    assert "Time unit: ns; data kind: raw_counts" in result["output"]
    assert f"Estimated lifetime: {result['lifetime_ns']:.2f} ns" in result["output"]
    assert "Fit status: success" in result["output"]
    assert result["model"] == "unconvolved least squares"


def test_module_cli_simulation_entry_point():
    completed = subprocess.run(
        [sys.executable, "-m", "tcspc_toolkit", "simulate", "--bins", "12",
         "--lifetime", "2", "--random-seed", "120406"], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Simulation completed." in completed.stdout
    assert "Number of bins: 12" in completed.stdout


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("issue12_integration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocker_rejects_every_facade_but_not_supported_packages(harness):
    blocker = harness.BlockFlatFacades()
    assert len(blocker.names) == 38
    for name in blocker.names:
        with pytest.raises(ImportError, match="flat facade"):
            blocker.find_spec(name)
    for name in ("tcspc_toolkit", "tcspc_toolkit.persistence", "tcspc_toolkit.irf",
                 "tcspc_toolkit.simulation", "tcspc_toolkit.estimation.fitting"):
        assert blocker.find_spec(name) is None


def test_optional_blocker_preserves_missing_module_identity(harness):
    with pytest.raises(ModuleNotFoundError) as failure:
        harness.BlockEmcee().find_spec("emcee")
    assert failure.value.name == "emcee"
    assert harness.BlockEmcee().find_spec("numpy") is None
