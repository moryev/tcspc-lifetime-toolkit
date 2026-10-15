"""Lightweight harness contracts; no full scientific reruns in normal pytest."""

import ast
import importlib.util
import json
import copy
import re
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scientific_acceptance", ROOT / "scripts/run_issue12_scientific_reproducibility.py"
)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def test_blocker_matches_reviewed_retirement_manifest():
    tree = ast.parse((ROOT / "tests/test_package_structure.py").read_text(encoding="utf-8"))
    registry = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "CLOSURE_FACADES" for t in node.targets))
    packages = {"irf", "preprocessing", "simulation", "evaluation", "persistence"}
    assert acceptance.flat_facades() == {"tcspc_toolkit." + name for name in registry.keys() - packages}
    blocker = acceptance.BlockFlatFacades()
    for name in blocker.names:
        with pytest.raises(ImportError, match="flat facade"):
            blocker.find_spec(name)
    for name in packages | {"config", "benchmarks.generalization", "estimation.classical"}:
        assert blocker.find_spec("tcspc_toolkit." + name) is None


@pytest.mark.parametrize("week,cell,rows,width", [
    (7, "83354e7feeb8b834", 4, 10),
    (8, "e802c319fc9ab6ad", 6, 7),
    (8, "93c3d55a39cf7b40", 6, 6),
    (8, "f5e40edbc2b8ea2b", 1, 7),
])
def test_committed_reference_table_is_readable_without_execution(week, cell, rows, width):
    result = acceptance.reference_rows(week, cell)
    assert len(result) == rows
    assert {len(row) for row in result} == {width}


def test_html_reader_ignores_headers_and_preserves_body_labels():
    reader = acceptance.BodyRows()
    reader.feed('<table><thead><tr><th>ignore</th></tr></thead><tbody>'
                '<tr><th>method</th><td><b>0.2500</b></td></tr></tbody></table>')
    assert reader.rows == [["method", "0.2500"]]


@pytest.mark.parametrize("decimals", [3, 4, 6])
def test_comparison_uses_only_half_of_displayed_unit(decimals):
    unit = 10 ** -decimals
    report = {}
    acceptance.compare(report, "inside", 1 + .49 * unit, 1, decimals)
    acceptance.compare(report, "outside", 1 + .51 * unit, 1, decimals)
    assert [r["status"] for r in report["comparisons"]] == ["PASS", "FAIL"]
    with pytest.raises(acceptance.AcceptanceFailure, match="1 reference comparisons failed"):
        acceptance.stop_on_discrepancy(report, "test phase")
    assert report["first_discrepancy_phase"] == "test phase"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_discrepancies_are_failures_and_json_safe(value):
    report = {}
    acceptance.compare(report, "nonfinite", value, 1., 4)
    assert report["comparisons"][0]["status"] == "FAIL"
    json.dumps(report, allow_nan=False)


def test_missing_failure_reason_is_a_json_null_not_nan_dictionary_key():
    frame = pd.DataFrame({"failure_reason": [None, "poisson_numerical_validation_failed", None]})
    counts = frame.groupby("failure_reason", dropna=False).size().reset_index(name="count")
    result = acceptance.records(counts)
    json.dumps(result, allow_nan=False)
    assert {r["count"] for r in result} == {1, 2}
    assert next(r for r in result if r["count"] == 2)["failure_reason"] is None


def test_discrepancy_writes_fail_report_and_does_not_continue_workflow(tmp_path, monkeypatch):
    def fail(report):
        acceptance.compare(report, "scientific metric", 2., 1., 4)
        acceptance.stop_on_discrepancy(report, "fitting")
        pytest.fail("workflow continued after discrepancy")

    monkeypatch.setattr(acceptance, "week7", fail)
    monkeypatch.setattr(acceptance, "flat_facades", lambda: {"tcspc_toolkit.nonexistent_test_facade"})
    monkeypatch.setattr(acceptance.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(acceptance.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    assert acceptance.main(["--week7", "--collect-candidate"]) == 1
    result = json.loads((tmp_path / "acceptance.json").read_text())
    assert result["status"] == "FAIL"
    assert result["workflows"]["week7"]["first_discrepancy_phase"] == "fitting"
    assert "week8" not in result["workflows"]


def test_current_reference_schema_protocol_provenance_and_populations():
    reference = acceptance.read_current_reference()
    assert acceptance.REFERENCE_PATH == ROOT / "data/reference/issue12_week7_week8_post_correction.json"
    assert reference["reference_id"] == "issue12-week7-week8-post-correction"
    assert reference["reference_version"] == 1
    assert reference["purpose"]
    assert re.fullmatch(r"[0-9a-f]{40}", reference["source_commit"])
    for key in ("historical_week7_notebook_publication_commit", "historical_week8_notebook_publication_commit",
                "poisson_scaling_commit", "validation_recovery_commit", "issue12_fallback_fix_commit"):
        assert re.fullmatch(r"[0-9a-f]{40}", reference["historical_provenance"][key])
    assert set(reference["environment_used_for_reference"]) == {"Python", "numpy", "scipy", "pandas", "scikit-learn"}
    w7, w8 = reference["weeks"]["week7"], reference["weeks"]["week8"]
    assert (w7["n_development"], w7["n_train"], w7["n_test"], w7["n_bins"]) == (810, 648, 162, 400)
    assert [w7["protocol"][key] for key in ("simulation_seed", "split_seed", "mismatch_seed")] == [42]*3
    assert w7["protocol"]["references"] == ["generating_mono", "primary_component"]
    assert w8["protocol_id"] == "week8-day55-v2"
    assert w8["test_order"] == list("ABCDEF")
    assert (w8["development_seed"], w8["n_development"], w8["n_bins"]) == (42, 64, 400)
    assert w8["external_seeds"] == dict(zip("ABCDEF", range(50001, 50007)))
    assert w8["external_counts"] == dict.fromkeys("ABCDEF", 192)
    assert len(w8["scorecard"]) == 72
    assert w8["methods"] == list(acceptance.METHODS)
    assert w8["representations"] == list(acceptance.REPRESENTATIONS)
    assert w8["test_f"]["severity_counts"] == {"weak": 96, "moderate": 96}
    assert w8["test_f"]["reference"] == "primary_component"
    assert w8["test_f"]["descriptive_reference"] == "signal_photon_weighted"
    populations = [w7["matched"], w7["mismatch"], *w8["classical_populations"].values(),
                   w8["test_c_irf"]["correct"], w8["test_c_irf"]["misspecified"],
                   *w8["test_f"]["classical_populations"].values()]
    for population in populations:
        assert population["n_total"] == population["n_valid"] + population["n_failed"]
        assert population["n_failed"] == len(population["failures"])
        assert population["metric_population"] == "valid fits only"
    assert w8["classical_populations"]["C"]["failures"] == [
        {"sample_id": 69, "failure_reason": "optimizer_unsuccessful"}]
    for row in w8["scorecard"] + w8["test_f"]["metrics"]:
        assert row["n_total"] == row["n_valid"] + row["n_failed"]


def test_reference_tolerances_preserve_display_precision_and_exact_identity():
    ref = acceptance.read_current_reference()
    policy = ref["comparison_tolerances"]
    assert set(policy["week7"].values()) == {5e-7}
    assert policy["week8"]["mae_ns"] == 5e-5
    assert policy["week8"]["mae_ratio_over_A"] == 5e-4
    assert set(policy["week8"].values()) == {5e-5, 5e-4}
    assert not {"n_valid", "sample_id", "secondary_fraction", "seed"} & set(policy["week8"])
    for week, expected in ref["weeks"].items():
        assert acceptance.compare_current(expected, expected, policy[week]) == []
        def check_numeric_fields(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    if name in policy[week]:
                        assert isinstance(item, float), (week, name, item)
                    if name.startswith("n_") and isinstance(item, (int, float)):
                        assert type(item) is int, (week, name, item)
                    check_numeric_fields(item)
            elif isinstance(value, list):
                for item in value:
                    check_numeric_fields(item)
        check_numeric_fields(expected)


@pytest.mark.parametrize("observed,expected,passes", [
    ({"mae_ns": 1.000049}, {"mae_ns": 1.0}, True),
    ({"mae_ns": 1.000051}, {"mae_ns": 1.0}, False),
    ({"mae_ns": float("nan")}, {"mae_ns": 1.0}, False),
    ({"n_valid": 191.0}, {"n_valid": 191}, False),
    ({"n_valid": 192}, {"n_valid": 191}, False),
    ({"failures": [{"sample_id": 68}]}, {"failures": [{"sample_id": 69}]}, False),
    ({"ids": ["B", "A"]}, {"ids": ["A", "B"]}, False),
    ({"mae_ns": 1.0, "extra": 1}, {"mae_ns": 1.0}, False),
    ({"seed": 42.000001}, {"seed": 42.0}, False),
])
def test_current_comparator_is_not_a_blanket_float_tolerance(observed, expected, passes):
    assert (not acceptance.compare_current(observed, expected, {"mae_ns": 5e-5})) == passes


def test_independent_repeat_checks_fingerprints_and_not_machine_timing():
    prior = dict(versions={"numpy": "example"}, workflows={"week7": dict(
        science={"n_valid": 162}, repeatability={"classical": "exact-byte-digest"}, timing=1)})
    current = copy.deepcopy(prior)
    current["workflows"]["week7"]["timing"] = 999
    acceptance.verify_repeat(current, prior)
    assert current["deterministic_rerun"] == "PASS"
    current["workflows"]["week7"]["repeatability"]["classical"] = "different-parameters"
    with pytest.raises(acceptance.AcceptanceFailure, match="nondeterministic"):
        acceptance.verify_repeat(current, prior)


def test_population_and_paired_metrics_do_not_hide_missing_pairs():
    first = pd.DataFrame(dict(sample_id=[1, 2, 3], valid_fit=[True, False, True],
                              error_ns=[1., 100., -2.], failure_reason=[None, "optimizer_unsuccessful", None]))
    second = first.copy()
    second["valid_fit"] = [True, True, False]
    second["error_ns"] = [2., 0., 0.]
    assert acceptance.population(first)["n_failed"] == 1
    paired = acceptance.paired_error_change(first, second)
    assert paired == dict(n_complete_pairs=1, mean_absolute_error_change_ns=1.,
                          median_absolute_error_change_ns=1., n_worse=1, n_better=0, n_equal=0)


@pytest.mark.parametrize("mode,observed_count,expected_status", [
    (["--collect-candidate"], 162, "CANDIDATE"),
    (["--reference", "issue12-post-correction"], 162, "PASS"),
    (["--reference", "issue12-post-correction"], 161, "FAIL"),
])
def test_historical_differences_are_informational_and_candidate_is_not_acceptance(
    tmp_path, monkeypatch, mode, observed_count, expected_status,
):
    def run(report):
        report.update(science={"n_valid": observed_count}, historical_notebook_comparison=[{"status": "FAIL"}])

    monkeypatch.setattr(acceptance, "week7", run)
    monkeypatch.setattr(acceptance, "flat_facades", lambda: set())
    monkeypatch.setattr(acceptance.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(acceptance, "read_current_reference", lambda: dict(
        reference_id=acceptance.REFERENCE_ID, reference_version=1,
        comparison_tolerances={"week7": {}}, weeks={"week7": {"n_valid": 162}}))
    assert acceptance.main(["--week7", *mode]) == (1 if expected_status == "FAIL" else 0)
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["status"] == expected_status
    assert report["workflows"]["week7"]["historical_difference_summary"]["differences"] == 1
    if expected_status == "FAIL":
        assert report["workflows"]["week7"]["current_reference_acceptance"]["discrepancies"]


def test_reference_selection_is_explicit():
    with pytest.raises(SystemExit) as error:
        acceptance.main(["--week7"])
    assert error.value.code == 2


@pytest.mark.parametrize("args", [
    ["--week9-ml", "--reference", "issue12-post-correction"],
    ["--week8", "--reference", "issue12-week9-ml"],
])
def test_week9_reference_cannot_silently_replace_week7_week8(args):
    with pytest.raises(SystemExit) as error:
        acceptance.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("cell,index,count", [
    ("910a6d01bbf11f44", None, 5),
    ("339769e99c73f7e3", 0, 12),
    ("339769e99c73f7e3", 1, 12),
    ("7dde16d16f42f0e6", None, 3),
])
def test_week9_saved_tables_are_read_without_execution(cell, index, count):
    assert len(acceptance.reference_rows(9, cell, index)) == count


def test_week9_reference_schema_split_kinds_populations_and_provenance():
    from tcspc_toolkit.uncertainty.evaluation import make_uncertainty_development_split

    ref = acceptance.read_week9_reference()
    assert acceptance.WEEK9_REFERENCE_PATH == ROOT / "data/reference/issue12_week9_ml_uncertainty_post_correction.json"
    assert re.fullmatch(r"[0-9a-f]{40}", ref["source_commit"])
    for key, value in ref["historical_provenance"].items():
        if key.endswith("_commit"):
            assert re.fullmatch(r"[0-9a-f]{40}", value)
    assert ref["independent_rerun"]["status"] == "PASS"
    w9 = ref["weeks"]["week9_ml"]
    split = make_uncertainty_development_split(64)
    assert (w9["development_seed"], w9["split_seed"]) == (42, 57)
    assert (w9["n_development"], w9["n_training"], w9["n_calibration"]) == (64, 48, 16)
    assert w9["training_indices"] == list(split.training_indices)
    assert w9["calibration_indices"] == list(split.calibration_indices)
    assert set(w9["training_ids"]).isdisjoint(w9["calibration_ids"])
    w8 = acceptance.read_current_reference()["weeks"]["week8"]
    assert w9["protocol_id"] == w8["protocol_id"] == "week8-day55-v2"
    assert w9["test_order"] == list("ABCDEF")
    assert w9["external_counts"] == dict.fromkeys("ABCDEF", 192)
    assert w9["external_seeds"] == dict(zip("ABCDEF", range(50001, 50007)))
    assert w9["input_fingerprints"] == {"development": w8["development_sha256"], **w8["external_sha256"]}
    assert w9["conformal"]["finite_sample_rank"] == w9["conformal"]["n_calibration_scores"] == 16
    assert w9["estimator_policy"]["bootstrap_replicates"] == 200
    assert w9["estimator_policy"]["random_state"] == 42
    methods = {"quantile_gradient_boosting", "conformalized_quantile_gradient_boosting",
               "random_forest_tree_spread", "ml_training_bootstrap"}
    assert {r["method_id"] for r in w9["calibration"]} == methods
    assert len(w9["external"]) == 24
    assert len(w9["test_f_severity"]) == 8
    for group, count in (("calibration", 16), ("external", 192), ("test_f_severity", 96)):
        for row in w9[group]:
            population = row["population"]
            assert population["n_attempted"] == count == population["n_valid"] + population["n_failed"]
            assert len(population["failures"]) == population["n_failed"]
            assert row["point_metrics"]["n_attempted"] == count
            if row["output_kind"] == "score":
                assert row["nominal_level"] is None
                assert "empirical_coverage" not in row["metrics"]
            else:
                assert row["output_kind"] in {"quantile_prediction", "conformal_prediction"}
                assert row["nominal_level"] == .95 - .05
                assert row["metrics"]["n_valid_intervals"] == population["n_valid"]
    assert w9["test_f"]["severity_counts"] == {"weak": 96, "moderate": 96}
    assert w9["test_f"]["reference"] == "primary_component"
    assert len(w9["residual_summary"]) == 3
    tolerance = ref["comparison_tolerances"]["week9_ml"]
    assert set(tolerance.values()) == {5e-5}
    assert not {"nominal_level", "nominal_coverage", "tail_fraction", "random_state", "n_valid"} & tolerance.keys()
    assert acceptance.compare_current(w9, w9, tolerance) == []


@pytest.mark.parametrize("role", ["uncertainty_training", "uncertainty_calibration"])
@pytest.mark.parametrize("test_id", list("ABCDEF"))
def test_week9_external_provenance_rejects_training_and_calibration(test_id, role):
    from tcspc_toolkit.uncertainty.evaluation import UncertaintyDataProvenance, UncertaintyDataRole
    with pytest.raises(ValueError, match="only for external"):
        UncertaintyDataProvenance(test_id, UncertaintyDataRole(role))


def test_training_audit_observes_exact_training_rows_and_bootstrap_draws(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from tcspc_toolkit.uncertainty.evaluation import make_uncertainty_development_split

    development = SimpleNamespace(X_features=pd.DataFrame(np.arange(128).reshape(64, 2)), y=np.arange(64.))
    split = make_uncertainty_development_split(64)
    X = development.X_features.to_numpy(dtype=float)[list(split.training_indices)]
    y = development.y[list(split.training_indices)]
    monkeypatch.setattr(Pipeline, "fit", lambda self, *args, **kwargs: self)
    with acceptance.audit_week9_training(development, split) as calls:
        for model in [HistGradientBoostingRegressor() for _ in range(3)] + [RandomForestRegressor(), Ridge()]:
            Pipeline([("model", model)]).fit(X, y)
        rng = np.random.default_rng(42)
        for _ in range(200):
            indices = rng.integers(0, 48, 48)
            Pipeline([("model", Ridge())]).fit(X[indices], y[indices])
    assert len(calls) == 205
    with pytest.raises(AssertionError):
        with acceptance.audit_week9_training(development, split):
            Pipeline([("model", Ridge())]).fit(X + 1, y)


def test_calibration_external_guard_rejects_refitting():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    with acceptance.frozen_week9_models():
        with pytest.raises(acceptance.AcceptanceFailure, match="attempted model/representation fitting"):
            Pipeline([("model", Ridge())]).fit([[1.]], [1.])


@pytest.mark.parametrize("interval", [False, True])
def test_uncertainty_population_preserves_failed_ids(interval):
    import numpy as np
    from tcspc_toolkit.uncertainty.evaluation import PredictionIntervalResult, UncertaintyScoreResult
    if interval:
        source = PredictionIntervalResult(np.array([1., 2.]), np.array([.5, 3.]), np.array([2., 2.]), .9, "test")
    else:
        source = UncertaintyScoreResult(np.array([1., 2.]), np.array([.1, np.nan]), "test")
    population = acceptance.uncertainty_population(source, [11, 12])
    assert population["n_attempted"] == 2
    assert population["n_valid"] == population["n_failed"] == 1
    assert population["failure_rate"] == .5
    assert population["failures"] == [{"sample_id": 12, "reason": "invalid_interval" if interval else "invalid_score"}]
