"""Independent Week-7 mismatch oracles, not dependency-version snapshots."""

from dataclasses import astuple, replace

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit import mismatch_evaluation as mismatch
from tcspc_toolkit import ml_evaluation as ml
from tcspc_toolkit.classical_evaluation import ReconvolutionBenchmarkResult, ReconvolutionBenchmarkSummary
from tcspc_toolkit.evaluation_core import build_point_evaluation, combine_point_evaluations
from tcspc_toolkit.evaluation_results import EvaluationBatch, LifetimeReference, MethodDescriptor


TARGETS = np.array([1.0, 2.0, 4.0])
CONTROL = np.array([0.0, 3.0, 2.0])
SHIFTED = np.array([-1.0, 4.0, 8.0])
CONTROL_METRICS = (4 / 3, 1.0, np.sqrt(2), 2 / 3, 0.5, 1 - 6 / (14 / 3))
SHIFTED_METRICS = (8 / 3, 2.0, np.sqrt(8), 4 / 3, 1.0, 1 - 24 / (14 / 3))


@pytest.fixture
def paired_data():
    training = pd.DataFrame({"feature": [9.0, 8.0]})
    control = pd.DataFrame({"feature": [10.0, 20.0, 30.0]}, index=[90, 20, 50])
    external = pd.DataFrame({"feature": [11.0, 22.0, 33.0]}, index=[8, 4, 2])
    metadata = pd.DataFrame({"sample_id": [7, 3, 9]}, index=[90, 20, 50])
    split = ml.BenchmarkSplit(
        np.array([0, 1]), np.array([7, 3, 9]), training, control,
        np.ones((2, 2), dtype=int), np.ones((3, 2), dtype=int),
        np.array([1.5, 2.5]), TARGETS.copy(), pd.DataFrame(index=[0, 1]), metadata,
    )
    dataset = ml.BenchmarkDataset(
        external, np.ones((3, 2), dtype=int), TARGETS.copy(),
        metadata.copy().assign(primary_lifetime_ns=TARGETS, secondary_lifetime_ns=2 * TARGETS),
    )
    return split, dataset


@pytest.fixture
def controlled_estimators(monkeypatch, paired_data):
    split, dataset = paired_data
    instances = []

    class ControlledRegressor:
        def __init__(self):
            self.calls = []
            instances.append(self)

        def fit(self, X, y):
            assert X is split.X_features_train and y is split.y_train
            self.calls.append("fit")
            return self

        def predict(self, X):
            if X is split.X_features_test:
                self.calls.append("control")
                return CONTROL.copy()
            assert X is dataset.X_features
            self.calls.append("mismatch")
            return SHIFTED.copy()

    for name in ("make_ridge_pipeline", "make_random_forest_pipeline", "make_hist_gradient_boosting_pipeline"):
        monkeypatch.setattr(mismatch, name, ControlledRegressor)
    return instances


@pytest.mark.parametrize("builder", [ml._build_regression_benchmark_result, mismatch._build_regression_result])
@pytest.mark.parametrize("columns", [None, 1, 2])
def test_regression_builders_preserve_independent_numerics_and_shape(builder, columns):
    targets = np.column_stack([TARGETS] * columns) if columns else TARGETS.copy()
    estimates = np.column_stack([CONTROL] * columns) if columns else CONTROL.copy()
    result = builder(estimator_name="arbitrary", y_true=targets, y_pred=estimates)
    assert isinstance(result, ml.RegressionBenchmarkResult)
    assert result.estimator_name == "arbitrary"
    np.testing.assert_array_equal(result.y_pred, estimates)
    expected_relative = np.array([1.0, 0.5, 0.5])
    if columns:
        expected_relative = np.column_stack([expected_relative] * columns)
    np.testing.assert_array_equal(result.relative_errors, expected_relative)
    np.testing.assert_allclose(astuple(result.metrics), CONTROL_METRICS, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(targets[:, 0] if columns else targets, TARGETS)
    np.testing.assert_array_equal(estimates[:, 0] if columns else estimates, CONTROL)


@pytest.mark.parametrize("module", [ml, mismatch])
@pytest.mark.parametrize("targets,estimates,kind", [
    ([1, 2], [np.nan], "shape"), ([1, 2], [1, np.inf], "finite"),
    ([1, 2], [1, np.nan], "finite"), ([1, 0], [1, 2], "target"),
])
def test_regression_wrapper_exception_contracts(module, targets, estimates, kind):
    ordinary = module is ml
    builder = ml._build_regression_benchmark_result if ordinary else mismatch._build_regression_result
    messages = {
        "shape": "Benchmark estimator must produce exactly one prediction per test sample." if ordinary
                 else "y_true and y_pred must have identical shapes.",
        "finite": "Benchmark predictions must be finite." if ordinary
                  else "Predictions must contain only finite values.",
        "target": "True lifetimes must be strictly positive.",
    }
    error_type = RuntimeError if ordinary and kind != "target" else ValueError
    with pytest.raises(error_type) as caught:
        builder(estimator_name="name", y_true=targets, y_pred=estimates)
    assert str(caught.value) == messages[kind]


def test_paired_predictions_metrics_summary_and_fitting_order(paired_data, controlled_estimators, captured_facts):
    split, dataset = paired_data
    control_before, external_before = split.X_features_test.copy(), dataset.X_features.copy()
    result = mismatch.evaluate_ml_mismatch_benchmark(reference_split=split, mismatch_dataset=dataset)
    names = ["ridge", "random_forest", "hist_gradient_boosting"]
    assert list(result) == names
    assert len(controlled_estimators) == 3
    assert all(estimator.calls == ["fit", "control", "mismatch"] for estimator in controlled_estimators)
    for name, pair in result.items():
        assert isinstance(pair, mismatch.EstimatorMismatchResult)
        for actual, predictions, relative, metrics in [
            (pair.in_distribution, CONTROL, [1.0, 0.5, 0.5], CONTROL_METRICS),
            (pair.mismatch, SHIFTED, [2.0, 1.0, 1.0], SHIFTED_METRICS),
        ]:
            assert actual.estimator_name == name
            np.testing.assert_array_equal(actual.y_pred, predictions)
            np.testing.assert_array_equal(actual.relative_errors, relative)
            np.testing.assert_allclose(astuple(actual.metrics), metrics, rtol=1e-15, atol=1e-15)
    expected = pd.DataFrame([
        (name, 4 / 3, 8 / 3, 4 / 3, 2.0, -2 / 3, 4 / 3, 0.0, 0.0) for name in names
    ], columns=["estimator", "in_distribution_mae_ns", "mismatch_mae_ns", "mae_change_ns",
                "mae_ratio", "in_distribution_bias_ns", "mismatch_bias_ns",
                "in_distribution_failure_rate", "mismatch_failure_rate"])
    actual = mismatch.summarize_mismatch_benchmark(y_true=TARGETS, ml_results=result)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(split.X_features_test, control_before, check_exact=True)
    pd.testing.assert_frame_equal(dataset.X_features, external_before, check_exact=True)
    np.testing.assert_array_equal(split.y_test, TARGETS)
    np.testing.assert_array_equal(dataset.y, TARGETS)
    combined = combine_point_evaluations(captured_facts)
    assert list(zip(combined.points.method_id, combined.points.evaluation_id)) == [
        (name, condition) for name in names for condition in ("in_distribution", "mismatch") for _ in range(3)
    ]
    assert combined.points.method_family.tolist() == ["ml"] * 18
    assert combined.points.representation_id.tolist() == ["engineered_features"] * 18
    assert combined.points.sample_id.tolist() == [0, 1, 2] * 6
    assert combined.points.is_valid.all()
    assert not combined.points.duplicated(["evaluation_id", "sample_id", "method_id", "representation_id"]).any()
    comparisons = combined.reference_comparisons
    assert comparisons.reference_kind.tolist() == (["generating_mono"] * 3 + ["primary_component"] * 3) * 3
    assert comparisons.reference_id.tolist() == comparisons.reference_kind.tolist()
    np.testing.assert_array_equal(comparisons.reference_value_ns, np.tile(TARGETS, 6))
    np.testing.assert_array_equal(comparisons.error_ns, [-1, 1, -2, -2, 2, 4] * 3)
    np.testing.assert_array_equal(comparisons.absolute_error_ns, [1, 1, 2, 2, 2, 4] * 3)
    assert comparisons.is_valid_comparison.all()


def test_zero_reference_ratio_remains_infinite_for_ml_but_nan_for_classical():
    perfect = ml.RegressionBenchmarkResult("ml", TARGETS.copy(), np.zeros(3), ml.RegressionMetrics(0, 0, 0, 0, 0, 1))
    shifted = ml.RegressionBenchmarkResult("ml", SHIFTED.copy(), np.array([2, 1, 1]), ml.RegressionMetrics(*SHIFTED_METRICS))
    control_fit = ReconvolutionBenchmarkResult(
        pd.DataFrame({"valid_fit": [True, True], "error_ns": [0.0, 0.0]}),
        ReconvolutionBenchmarkSummary(2, 2, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    )
    mismatch_fit = ReconvolutionBenchmarkResult(
        pd.DataFrame({"valid_fit": [True, False], "error_ns": [2.0, 10.0]}),
        ReconvolutionBenchmarkSummary(2, 1, 1, 0.5, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0),
    )
    actual = mismatch.summarize_mismatch_benchmark(
        y_true=TARGETS, ml_results={"ml": mismatch.EstimatorMismatchResult(perfect, shifted)},
        classical_result=mismatch.ClassicalMismatchResult(control_fit, mismatch_fit),
    )
    expected = pd.DataFrame({
        "estimator": ["ml", "reconvolution"], "in_distribution_mae_ns": [0.0, 0.0],
        "mismatch_mae_ns": [8 / 3, 2.0], "mae_change_ns": [8 / 3, 2.0], "mae_ratio": [np.inf, np.nan],
        "in_distribution_bias_ns": [0.0, 0.0], "mismatch_bias_ns": [4 / 3, 2.0],
        "in_distribution_failure_rate": [0.0, 0.0], "mismatch_failure_rate": [0.0, 0.5],
    })
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("targets,message", [
    ([1.0, 2.0], "Mismatch targets must match reference test targets."),
    ([2.0, 1.0, 4.0], "Mismatch targets must preserve the reference primary lifetimes."),
])
def test_existing_pairing_checks_remain_explicit(paired_data, targets, message):
    split, dataset = paired_data
    with pytest.raises(ValueError) as caught:
        mismatch.evaluate_ml_mismatch_benchmark(reference_split=split, mismatch_dataset=replace(dataset, y=np.array(targets)))
    assert str(caught.value) == message


@pytest.fixture
def captured_facts(monkeypatch):
    facts = []
    def capture(**kwargs):
        result = build_point_evaluation(**kwargs)
        facts.append(result)
        return result
    monkeypatch.setattr(ml, "build_point_evaluation", capture)
    return facts


@pytest.mark.parametrize("name,family", [("classical_reconvolution_custom", "ml"), ("user_baseline", "baseline")])
def test_explicit_private_adapter_uses_descriptor_not_name(name, family, captured_facts):
    batch = EvaluationBatch("external_run", [7, 3, 9], references={
        "component": LifetimeReference("component", "primary_component", TARGETS, [True] * 3),
    })
    actual = mismatch._build_regression_result(
        estimator_name="legacy_label", y_true=TARGETS, y_pred=SHIFTED,
        method=MethodDescriptor(name, family, "custom_features"), batch=batch,
    )
    assert actual.estimator_name == name
    np.testing.assert_array_equal(actual.y_pred, SHIFTED)
    assert captured_facts[0].points.method_family.tolist() == [family] * 3
    assert captured_facts[0].points.representation_id.tolist() == ["custom_features"] * 3
    assert captured_facts[0].points.evaluation_id.tolist() == ["external_run"] * 3
    assert captured_facts[0].reference_comparisons.reference_kind.tolist() == ["primary_component"] * 3
    actual.y_pred[0] = 123.0
    actual.relative_errors[0] = 456.0
    np.testing.assert_array_equal(captured_facts[0].points.lifetime_estimate_ns, SHIFTED)
    np.testing.assert_array_equal(captured_facts[0].reference_comparisons.relative_error, [2.0, 1.0, 1.0])


def test_bare_array_builders_share_error_kernel_without_fabricating_semantics(monkeypatch):
    calls = []
    original = ml.calculate_reference_errors
    def kernel(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    def forbidden(**kwargs):
        pytest.fail("Bare arrays do not identify method family or reference kind.")
    monkeypatch.setattr(ml, "calculate_reference_errors", kernel)
    monkeypatch.setattr(ml, "build_point_evaluation", forbidden)
    for builder in (ml._build_regression_benchmark_result, mismatch._build_regression_result):
        builder(estimator_name="classical_reconvolution_custom", y_true=TARGETS, y_pred=CONTROL)
    assert len(calls) == 2


@pytest.mark.parametrize("missing", ["method", "batch"])
def test_explicit_request_cannot_silently_fall_back(missing):
    semantics = {"method": MethodDescriptor("method", "ml"),
                 "batch": EvaluationBatch("external", [0, 1, 2], references={
                     "target": LifetimeReference("target", "generating_mono", TARGETS, [True] * 3),
                 })}
    semantics.pop(missing)
    with pytest.raises(ValueError, match="method and batch must be supplied together"):
        mismatch._build_regression_result(estimator_name="label", y_true=TARGETS, y_pred=CONTROL, **semantics)


@pytest.mark.parametrize("references,message", [
    ({}, "exactly one lifetime reference"),
    ({"target": LifetimeReference("target", "generating_mono", [2.0, 1.0, 4.0], [True] * 3)}, "match y_true exactly"),
    ({"target": LifetimeReference("target", "generating_mono", TARGETS, [True, False, True])}, "fully available"),
])
def test_explicit_projection_requires_aligned_available_reference(references, message):
    with pytest.raises(ValueError, match=message):
        mismatch._build_regression_result(
            estimator_name="label", y_true=TARGETS, y_pred=CONTROL,
            method=MethodDescriptor("method", "ml"), batch=EvaluationBatch("external", [0, 1, 2], references=references),
        )


def test_pairing_tolerance_keeps_each_branch_original_target_values(paired_data, controlled_estimators, captured_facts):
    split, dataset = paired_data
    dataset.y[:] += 1e-7  # Historically accepted by the existing np.allclose pairing check.
    mismatch.evaluate_ml_mismatch_benchmark(reference_split=split, mismatch_dataset=dataset)
    for i, facts in enumerate(captured_facts):
        expected_targets = split.y_test if i % 2 == 0 else dataset.y
        np.testing.assert_array_equal(facts.reference_comparisons.reference_value_ns, expected_targets)
