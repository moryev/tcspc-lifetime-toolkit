"""Lightweight harness contracts; no full scientific reruns in normal pytest."""

import ast
import importlib.util
import json
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
    assert acceptance.main(["--week7"]) == 1
    result = json.loads((tmp_path / "acceptance.json").read_text())
    assert result["status"] == "FAIL"
    assert result["workflows"]["week7"]["first_discrepancy_phase"] == "fitting"
    assert "week8" not in result["workflows"]
