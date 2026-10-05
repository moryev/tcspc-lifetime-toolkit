"""Independent tiny pre-migration tables; no shared expected-value builders.

These lock legacy schemas/order/masks without sklearn-version numerical snapshots.
The existing estimator-equivalence tests separately protect fitted predictions.
"""

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.conditional_evaluation import build_prediction_diagnostics
from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import MethodDescriptor
from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
from tcspc_toolkit.generalization_evaluation import (
    _build_generalization_prediction_table, build_mae_degradation_table,
    _build_generalization_evaluation_batch, _project_generalization_predictions,
    build_reference_mae_degradation_table, summarize_generalization_predictions,
)


@pytest.mark.parametrize("test_ids", [("A", "B"), ("A", "C", "B")])
@pytest.mark.parametrize("factual_projection", [False, True])
def test_legacy_ab_and_multitest_tables_have_independent_expectations(test_ids, factual_projection):
    facts = {
        "A": ([2.0, 1.0], [True, True], [1.0, -2.0]),
        "B": ([3.0, 13.0], [True, False], [2.0, np.nan]),
        "C": ([0.0, 4.0], [True, True], [-1.0, 1.0]),
    }
    tables, expected_tables, summary_rows = [], [], []
    name = "classical_reconvolution_fixture"
    for test_id in test_ids:
        predictions, valid, errors = facts[test_id]
        metadata = pd.DataFrame({"sample_id": [7, 3], "test_id": [test_id] * 2}, index=[90, 80])
        test = GeneralizationTestMeasurements(
            test_id, np.array([0.0, 1.0]), np.ones((2, 2), dtype=np.int64),
            np.array([1.0, 3.0]), metadata,
        )
        if factual_projection:
            batch = _build_generalization_evaluation_batch(test=test, representations={})
            result = build_point_evaluation(
                batch=batch, method=MethodDescriptor(name, "classical", "raw_histogram"),
                lifetime_estimates_ns=predictions, is_valid=valid,
            )
            if test_id == "B":
                # Raw facts retain the finite invalid error; the legacy view masks it.
                assert result.reference_comparisons.error_ns.tolist() == [2.0, 10.0]
            tables.append(_project_generalization_predictions(
                batch=batch, result=result, reference_id="generating_mono",
            ))
        else:
            tables.append(_build_generalization_prediction_table(
                estimator_name=name, representation_name="raw_histogram", test=test,
                y_pred=predictions, valid_mask=valid,
            ))
        expected_tables.append(pd.DataFrame({
            "sample_id": [7, 3], "test_id": [test_id] * 2,
            "estimator": [name] * 2, "representation": ["raw_histogram"] * 2,
            "true_lifetime_ns": [1.0, 3.0], "predicted_lifetime_ns": predictions,
            "valid_prediction": valid, "error_ns": errors,
            "absolute_error_ns": {"A": [1.0, 2.0], "B": [2.0, np.nan], "C": [1.0, 1.0]}[test_id],
        }))
        values = {
            "A": (2, 1.5, 1.5, np.sqrt(2.5), -0.5, 1.9, 1.95, 0.0),
            "B": (1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.5),
            "C": (2, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0),
        }[test_id]
        summary_rows.append((name, "raw_histogram", test_id, 2, *values))
    points = pd.concat(tables, ignore_index=True)
    pd.testing.assert_frame_equal(points, pd.concat(expected_tables, ignore_index=True), check_exact=True)
    expected_summary = pd.DataFrame(summary_rows, columns=[
        "estimator", "representation", "test_id", "n_total_samples", "n_valid_predictions",
        "mae_ns", "median_absolute_error_ns", "rmse_ns", "bias_ns",
        "p90_absolute_error_ns", "p95_absolute_error_ns", "classical_failure_rate",
    ])
    summary = summarize_generalization_predictions(points)
    pd.testing.assert_frame_equal(summary, expected_summary, check_exact=True)
    if test_ids == ("A", "B"):
        expected = pd.DataFrame([(name, "raw_histogram", 1.5, 2.0, 4 / 3)], columns=[
            "estimator", "representation", "mae_a_ns", "mae_b_ns", "mae_degradation",
        ])
        pd.testing.assert_frame_equal(build_mae_degradation_table(summary), expected, check_exact=True)
    else:
        expected = pd.DataFrame([
            (name, "raw_histogram", "A", "C", 1.5, 1.0, 2 / 3),
            (name, "raw_histogram", "A", "B", 1.5, 2.0, 4 / 3),
        ], columns=["estimator", "representation", "reference_test_id", "ood_test_id",
                    "reference_mae_ns", "ood_mae_ns", "mae_degradation"])
        pd.testing.assert_frame_equal(
            build_reference_mae_degradation_table(summary, reference_test_id="A"),
            expected, check_exact=True,
        )


def test_legacy_conditional_builder_keeps_finite_invalid_error_visible():
    result = build_prediction_diagnostics(
        estimator_name="method", y_true=[2.0], y_pred=[5.0],
        metadata=pd.DataFrame({"sample_id": [7]}, index=[90]), valid_mask=[False],
    )
    assert result.index.tolist() == [90]
    assert result.error_ns.tolist() == [3.0]
    assert result.valid_estimate.tolist() == [False]
