"""Factual execution and legacy projection contracts, independent of reports."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tcspc_toolkit.estimator_api import EstimatorSpec, fit_regressors
from tcspc_toolkit.evaluation_core import build_point_evaluation, combine_point_evaluations
from tcspc_toolkit.evaluation_results import EvaluationBatch, LifetimeReference, MethodDescriptor
from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
from tcspc_toolkit.generalization_evaluation import (
    _build_generalization_evaluation_batch,
    _build_classical_point_evaluation,
    _evaluate_nonclassical_batch,
    _generalization_baseline_predictions,
    _project_generalization_predictions,
    _summarize_generalization_predictions,
)


@pytest.fixture
def custom_execution():
    training = pd.DataFrame({"coordinate": [1.0, 2.0, 3.0]})
    X = pd.DataFrame({"coordinate": [-2.0, 0.0, 2.0]}, index=[90, 30, 70])
    fitted = fit_regressors(
        estimator_specs=(EstimatorSpec(
            "classical_reconvolution_custom", lambda: LinearRegression(fit_intercept=False),
            ("user/coordinate",),
        ),),
        X_train_by_representation={"user/coordinate": training}, y_train=[1.0, 2.0, 3.0],
    )
    reference = LifetimeReference("standard", "trusted_experimental", [1, 2, 3], [True] * 3)
    batch = EvaluationBatch(
        "external-observations/run-2", [7, 3, 11],
        representations={"user/coordinate": X, "unselected": np.ones((3, 2))},
        metadata=pd.DataFrame({"sample_id": [7, 3, 11]}, index=[90, 30, 70]),
        references={"standard": reference},
    )
    features = pd.DataFrame({"mean_arrival_time_ns": [0, 1, 5], "peak_time_ns": [1, 1, 2]})
    baselines = _generalization_baseline_predictions(y_development=[1, 2, 3], X_features=features)
    return batch, fitted, baselines, X


def test_explicit_ml_family_custom_representation_and_baselines_share_facts(custom_execution):
    batch, fitted, baselines, X = custom_execution
    result = _evaluate_nonclassical_batch(
        batch=batch, fitted_estimators=fitted, baseline_predictions=baselines,
    )
    points = result.points
    assert points.method_id.tolist() == (
        ["constant_mean"] * 3 + ["mean_arrival_time"] * 3 + ["classical_reconvolution_custom"] * 3
    )
    assert points.method_family.tolist() == ["baseline"] * 6 + ["ml"] * 3
    assert points.representation_id.tolist() == (
        [None] * 3 + ["engineered_features"] * 3 + ["user/coordinate"] * 3
    )
    assert points.sample_id.tolist() == [7, 3, 11] * 3
    assert points.evaluation_id.eq("external-observations/run-2").all()
    assert points.is_valid.all() and points.failure_reason.isna().all()
    assert batch.representations["user/coordinate"] is X
    np.testing.assert_array_equal(points.lifetime_estimate_ns.iloc[:6], [2, 2, 2, -1, 0, 3])
    expected = fitted["classical_reconvolution_custom"]["user/coordinate"].predict(X)
    np.testing.assert_array_equal(points.lifetime_estimate_ns.iloc[6:], expected)
    assert expected[0] < 0 and expected[1] == 0  # No positivity filter or clipping.
    assert result.reference_comparisons.reference_kind.eq("trusted_experimental").all()

    # A real classical result with a distinct ID coexists without prefix inference.
    classical = _build_classical_point_evaluation(
        batch=batch, method_id="classical_reconvolution",
        diagnostics=pd.DataFrame({
            "sample_id": batch.sample_ids, "fitted_lifetime_ns": [1.0, 5.0, np.nan],
            "valid_fit": [True, False, False], "failure_reason": [None, "boundary_hit", "fit_exception"],
        }),
    )
    combined = combine_point_evaluations([result, classical])
    families = combined.points.groupby("method_id").method_family.first()
    assert families["classical_reconvolution_custom"] == "ml"
    assert families["classical_reconvolution"] == "classical"
    table = _project_generalization_predictions(batch=batch, result=combined, reference_id="standard")
    table["test_id"] = batch.evaluation_id
    summary = _summarize_generalization_predictions(table, method_families=families.to_dict())
    assert summary.loc[summary.estimator == "classical_reconvolution_custom", "classical_failure_rate"].isna().all()
    assert summary.loc[summary.estimator == "classical_reconvolution", "classical_failure_rate"].iloc[0] == 1 - 1 / 3


def test_duplicate_baseline_facts_are_rejected_not_overwritten(custom_execution):
    batch, fitted, baselines, _ = custom_execution
    with pytest.raises(ValueError, match="Duplicate point"):
        _evaluate_nonclassical_batch(
            batch=batch, fitted_estimators=fitted,
            baseline_predictions=(baselines[0], baselines[0]),
        )


def test_arbitrary_batch_can_execute_without_a_reference(custom_execution):
    original, fitted, _, _ = custom_execution
    batch = EvaluationBatch("unreferenced-observation", original.sample_ids, original.representations)
    result = _evaluate_nonclassical_batch(batch=batch, fitted_estimators=fitted)
    assert len(result.points) == 3
    assert result.reference_comparisons.empty


@pytest.mark.parametrize("test_id, kind", [("A", "generating_mono"), ("F", "primary_component")])
@pytest.mark.parametrize("sample_ids", [True, False])
def test_frozen_batch_adapter_preserves_inputs_and_explicit_target(test_id, kind, sample_ids):
    metadata = pd.DataFrame({"test_id": [test_id] * 2, "photon_weighted_lifetime_ns": [9.0, 8.0]}, index=[90, 30])
    if sample_ids:
        metadata["sample_id"] = [7, 3]
    targets = np.array([1.0, 3.0])
    test = GeneralizationTestMeasurements(test_id, np.array([0.0, 1.0]), np.ones((2, 2)), targets, metadata)
    matrix = pd.DataFrame({"feature": [2.0, 4.0]}, index=[12, 16])
    original = metadata.copy(deep=True)
    batch = _build_generalization_evaluation_batch(test=test, representations={"prepared": matrix})
    assert batch.evaluation_id == test_id
    assert batch.sample_ids == ((7, 3) if sample_ids else (0, 1))
    assert batch.representations["prepared"] is matrix
    assert tuple(batch.references) == (kind,)
    assert batch.references[kind].kind == kind
    np.testing.assert_array_equal(batch.references[kind].values_ns, [1.0, 3.0])
    pd.testing.assert_frame_equal(batch.metadata, original)
    pd.testing.assert_frame_equal(metadata, original)
    np.testing.assert_array_equal(targets, [1.0, 3.0])


def test_projection_preserves_column_positions_metadata_and_invalid_error_policy():
    # Existing output-named metadata columns retain their positions when replaced.
    metadata = pd.DataFrame({
        "predicted_lifetime_ns": [99.0, 99.0], "test_id": ["F", "F"],
        "sample_id": [7, 3], "error_ns": [99.0, 99.0], "label": ["first", "second"],
    }, index=[90, 30])
    test = GeneralizationTestMeasurements(
        "F", np.array([0.0, 1.0]), np.ones((2, 2)), np.array([1.0, 3.0]), metadata,
    )
    batch = _build_generalization_evaluation_batch(test=test, representations={})
    result = build_point_evaluation(
        batch=batch, method=MethodDescriptor("user-fit", "ml", "custom"),
        lifetime_estimates_ns=[0.0, -2.0], is_valid=[True, False],
        failure_reasons=[None, "adapter rejection"],
    )
    projected = _project_generalization_predictions(batch=batch, result=result, reference_id="primary_component")
    expected = pd.DataFrame({
        "predicted_lifetime_ns": [0.0, -2.0], "test_id": ["F", "F"], "sample_id": [7, 3],
        "error_ns": [-1.0, np.nan], "label": ["first", "second"],
        "estimator": ["user-fit"] * 2, "representation": ["custom"] * 2,
        "true_lifetime_ns": [1.0, 3.0], "valid_prediction": [True, False],
        "absolute_error_ns": [1.0, np.nan],
    })
    pd.testing.assert_frame_equal(projected, expected, check_exact=True)
    assert result.reference_comparisons.error_ns.tolist() == [-1.0, -5.0]
    assert result.points.failure_reason.tolist() == [None, "adapter rejection"]
    assert metadata.error_ns.tolist() == [99.0, 99.0]
    assert metadata.index.tolist() == [90, 30]
