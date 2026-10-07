"""Hand-calculated expectations, not snapshots of the implementation."""

import ast
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.evaluation_core import (
    build_point_evaluation, calculate_degradation_ratio, calculate_reference_errors,
    summarize_point_errors,
)
from tcspc_toolkit.evaluation_results import EvaluationBatch, LifetimeReference, MethodDescriptor


def test_neutral_point_schema_preserves_failed_and_nonpositive_estimates_without_reference():
    result = build_point_evaluation(
        batch=EvaluationBatch("measurement/session", ["z", "b", "a", "failed"]),
        method=MethodDescriptor("classical_reconvolution_custom", "ml"),
        lifetime_estimates_ns=[0, -2, 7, np.inf], is_valid=[True, True, False, False],
        failure_reasons=[None, None, "rejected fit", "overflow"],
    )
    expected = pd.DataFrame({
        "evaluation_id": ["measurement/session"] * 4,
        "sample_id": ["z", "b", "a", "failed"],
        "method_id": ["classical_reconvolution_custom"] * 4,
        "method_family": ["ml"] * 4, "representation_id": [None] * 4,
        "lifetime_estimate_ns": [0.0, -2.0, 7.0, np.inf],
        "is_valid": [True, True, False, False],
        "failure_reason": pd.Series([None, None, "rejected fit", "overflow"], dtype=object),
    })
    pd.testing.assert_frame_equal(result.points, expected)
    assert result.reference_comparisons.empty
    assert result.reference_comparisons.columns.tolist() == [
        "evaluation_id", "sample_id", "method_id", "representation_id",
        "reference_id", "reference_kind", "reference_version", "assumption_label",
        "reference_available", "reference_value_ns", "error_ns", "absolute_error_ns",
        "relative_error", "is_valid_comparison",
    ]


def test_multiple_partial_references_do_not_duplicate_points_or_hide_invalid_errors():
    primary = LifetimeReference("component", "primary_component", [2, 4, 8], [True, False, True])
    projection = LifetimeReference(
        "projection", "pseudo_true_mono", [1, 2, 4], [True, True, True],
        assumption_label="mono/assumed-irf", reference_version="v1",
    )
    result = build_point_evaluation(
        batch=EvaluationBatch("mismatch", [30, 10, 20], references={
            "component": primary, "projection": projection,
        }),
        method=MethodDescriptor("fit", "classical"),
        lifetime_estimates_ns=[3, 6, np.nan], is_valid=[False, True, False],
        failure_reasons=["failed", None, "missing fit"],
    )
    assert len(result.points) == 3
    expected = pd.DataFrame({
        "evaluation_id": ["mismatch"] * 6, "sample_id": [30, 10, 20] * 2,
        "method_id": ["fit"] * 6, "representation_id": [None] * 6,
        "reference_id": ["component"] * 3 + ["projection"] * 3,
        "reference_kind": ["primary_component"] * 3 + ["pseudo_true_mono"] * 3,
        "reference_version": pd.Series([None] * 3 + ["v1"] * 3, dtype=object),
        "assumption_label": pd.Series([None] * 3 + ["mono/assumed-irf"] * 3, dtype=object),
        "reference_available": [True, False, True, True, True, True],
        "reference_value_ns": [2.0, np.nan, 8.0, 1.0, 2.0, 4.0],
        "error_ns": [1.0, np.nan, np.nan, 2.0, 4.0, np.nan],
        "absolute_error_ns": [1.0, np.nan, np.nan, 2.0, 4.0, np.nan],
        "relative_error": [0.5, np.nan, np.nan, 2.0, 2.0, np.nan],
        "is_valid_comparison": [False, False, False, False, True, False],
    })
    pd.testing.assert_frame_equal(result.reference_comparisons, expected)


def test_trusted_reference_without_synthetic_truth():
    reference = LifetimeReference("certificate", "trusted_experimental", [2.0], [True])
    result = build_point_evaluation(
        batch=EvaluationBatch("observation", ["sample"], references={"certificate": reference}),
        method=MethodDescriptor("external estimate", "bayesian"),
        lifetime_estimates_ns=[2.5], is_valid=[True],
    )
    assert result.reference_comparisons.error_ns.tolist() == [0.5]
    assert result.reference_comparisons.relative_error.tolist() == [0.25]
    assert "true_lifetime_ns" not in result.points


@pytest.mark.parametrize("estimates, valid, reasons", [
    ([np.nan], [True], None), ([np.inf], [True], None),
    ([1, 2], [True], None), ([1], [1], None),
    ([1], [True], ["failure"]), ([1], [False], []),
])
def test_inconsistent_point_inputs_fail(estimates, valid, reasons):
    with pytest.raises(ValueError):
        build_point_evaluation(
            batch=EvaluationBatch("x", [0]), method=MethodDescriptor("m", "ml"),
            lifetime_estimates_ns=estimates, is_valid=valid, failure_reasons=reasons,
        )


def test_error_arithmetic_and_explicit_metric_populations():
    signed, absolute, relative = calculate_reference_errors(
        [1, 6, 10, np.inf], [2, 4, 8, 1], reference_available=[True, True, False, True],
    )
    np.testing.assert_array_equal(signed, [-1, 2, np.nan, np.nan])
    np.testing.assert_array_equal(absolute, [1, 2, np.nan, np.nan])
    np.testing.assert_array_equal(relative, [0.5, 0.5, np.nan, np.nan])
    options = dict(is_valid=[True, False, True, False], reference_available=[True, True, False, True])
    finite_population = summarize_point_errors(signed, eligible=[True, True, False, False], **options)
    assert finite_population == {
        "n_attempted": 4, "n_valid": 2, "n_available": 3, "n_contributing": 2,
        "mae_ns": 1.5, "rmse_ns": np.sqrt(2.5), "bias_ns": 0.5,
        "median_absolute_error_ns": 1.5, "p90_absolute_error_ns": 1.9,
        "p95_absolute_error_ns": 1.95,
    }
    valid_population = summarize_point_errors(signed, eligible=[True, False, False, False], **options)
    assert valid_population["n_contributing"] == 1
    assert valid_population["mae_ns"] == 1
    assert valid_population["bias_ns"] == -1
    empty = summarize_point_errors(signed, eligible=[False] * 4, **options)
    assert empty["n_attempted"] == 4 and empty["n_contributing"] == 0
    assert all(np.isnan(value) for key, value in empty.items() if not key.startswith("n_"))


@pytest.mark.parametrize("errors, available, eligible", [
    ([np.nan], [True], [True]), ([1.0], [False], [True]), ([1.0], [True], [1]),
])
def test_metrics_never_silently_drop_ineligible_selected_rows(errors, available, eligible):
    with pytest.raises(ValueError):
        summarize_point_errors(errors, is_valid=[True], reference_available=available, eligible=eligible)


@pytest.mark.parametrize("reference, shifted, threshold, expected", [
    (2, 6, 0, 3), (2, 0, 0, 0), (0, 1, 0, np.nan),
    (1e-13, 1, 1e-12, np.nan), (1e-12, 1, 1e-12, np.nan),
    (1e-13, 1e-13, 0, 1), (np.nan, 1, 0, np.nan),
    (1, np.inf, 0, np.nan), (1, -1, 0, np.nan),
])
def test_degradation_has_explicit_threshold_and_undefined_policy(reference, shifted, threshold, expected):
    result = calculate_degradation_ratio(
        reference_error=reference, shifted_error=shifted, minimum_reference_error=threshold,
    )
    assert np.isnan(result) if np.isnan(expected) else result == expected


def test_degradation_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="nonnegative"):
        calculate_degradation_ratio(reference_error=1, shifted_error=2, minimum_reference_error=-1)


def test_generic_dependencies_and_fresh_imports():
    # Root initialization is existing package behavior. Check the new modules'
    # direct dependencies as well as actual imports in a fresh process.
    allowed = {"__future__", "collections", "dataclasses", "types", "typing", "numpy", "pandas"}
    for module_name in ("evaluation_results", "evaluation_core"):
        module = importlib.import_module("tcspc_toolkit." + module_name)
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = ([node.module] if isinstance(node, ast.ImportFrom)
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in imports:
                assert name.split(".")[0] in allowed or name == "tcspc_toolkit.evaluation_results"
    script = """
import importlib.abc
import sys
blocked = {'tcspc_toolkit.generalization', 'tcspc_toolkit.generalization_evaluation',
           'tcspc_toolkit.persistence', 'tcspc_toolkit.bayesian_sampling', 'tcspc_toolkit.ml_models'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import tcspc_toolkit.evaluation_results
import tcspc_toolkit.evaluation_core
assert not blocked.intersection(sys.modules)
assert 'emcee' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
