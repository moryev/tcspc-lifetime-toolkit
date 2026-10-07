"""Independent conditional-table expectations and explicit factual projections."""

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit import conditional_evaluation as conditional
from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult, ReconvolutionBenchmarkSummary,
)
from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import EvaluationBatch, LifetimeReference, MethodDescriptor
from tcspc_toolkit.generalization_evaluation import _project_generalization_predictions
from tcspc_toolkit.ml_evaluation import (
    BenchmarkSplit, RegressionBenchmarkResult, RegressionMetrics,
)


@pytest.fixture
def array_case():
    index = pd.Index([90, 10, 90, 30, 20, 50], name="source_row")
    metadata = pd.DataFrame({
        "sample_id": [7, 7, 3, 9, 2, 4],
        "error_ns": [99.0] * 6,  # Existing columns keep their original position.
        "condition": ["mixed"] * 6,
    }, index=index)
    expected = pd.DataFrame({
        "estimator_name": ["legacy_label"] * 6,
        "sample_id": [7, 7, 3, 9, 2, 4],
        "error_ns": [1.0, 2.0, -2.0, -6.0, np.nan, np.inf],
        "condition": ["mixed"] * 6,
        "true_lifetime_ns": [2.0, 4.0, 2.0, 4.0, 2.0, 4.0],
        "predicted_lifetime_ns": [3.0, 6.0, 0.0, -2.0, np.nan, np.inf],
        "absolute_error_ns": [1.0, 2.0, 2.0, 6.0, np.nan, np.inf],
        "relative_error": [0.5, 0.5, 1.0, 1.5, np.nan, np.inf],
        "valid_estimate": [True, False, True, True, False, False],
    }, index=index)
    return metadata, expected


@pytest.mark.parametrize("explicit", [False, True])
def test_array_table_has_independent_expected_schema_order_values_and_index(array_case, explicit):
    metadata, expected = array_case
    original = metadata.copy(deep=True)
    actual = conditional.build_prediction_diagnostics(
        estimator_name="legacy_label", metadata=metadata,
        y_true=expected.true_lifetime_ns, y_pred=expected.predicted_lifetime_ns,
        valid_mask=expected.valid_estimate,
        **({"method": MethodDescriptor("legacy_label", "ml", "custom_features"),
            "reference": LifetimeReference("target", "generating_mono", expected.true_lifetime_ns, [True] * 6)}
           if explicit else {}),
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(metadata, original, check_exact=True)


def test_legacy_empty_input_keeps_schema_and_index():
    metadata = pd.DataFrame({"condition": pd.Series(dtype=str)}, index=pd.Index([], name="source"))
    actual = conditional.build_prediction_diagnostics(
        estimator_name="legacy", y_true=[], y_pred=[], metadata=metadata,
    )
    expected = metadata.copy()
    expected.insert(0, "estimator_name", "legacy")
    for name in ("true_lifetime_ns", "predicted_lifetime_ns", "error_ns", "absolute_error_ns", "relative_error"):
        expected[name] = np.array([], dtype=float)
    expected["valid_estimate"] = np.array([], dtype=bool)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def regression_case(metadata, y_true, y_pred, name="legacy_label"):
    # These adapters consume only the test-side fields, not model fitting/metrics.
    empty = np.empty(0)
    split = BenchmarkSplit(
        np.array([], dtype=int), np.arange(len(y_true)),
        pd.DataFrame(), pd.DataFrame(index=range(len(y_true))),
        np.empty((0, 1)), np.empty((len(y_true), 1)), empty, np.asarray(y_true),
        pd.DataFrame(), metadata,
    )
    result = RegressionBenchmarkResult(
        name, np.asarray(y_pred), empty,
        RegressionMetrics(*(np.nan for _ in range(6))),
    )
    return split, result


def test_legacy_regression_result_does_not_need_semantic_metadata(array_case, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Ambiguous legacy input must not create canonical facts.")
    monkeypatch.setattr(conditional, "build_point_evaluation", forbidden)
    metadata, expected = array_case
    split, result = regression_case(metadata, expected.true_lifetime_ns, expected.predicted_lifetime_ns)
    expected = expected.copy()
    expected["valid_estimate"] = [True, True, True, True, False, False]
    actual = conditional.build_ml_prediction_diagnostics(split=split, result=result)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.fixture
def classical_result():
    table = pd.DataFrame({
        "sample_id": [7, 3, 4, 6, 2, 8],
        "condition": ["mixed"] * 4 + ["failed"] * 2,
        "true_lifetime_ns": [2.0] * 6,
        "fitted_lifetime_ns": [3.0, 5.0, 1.0, np.nan, np.inf, 4.0],
        "error_ns": [1.0, 3.0, -1.0, np.nan, np.nan, 2.0],
        "absolute_error_ns": [1.0, 3.0, 1.0, np.nan, np.nan, 2.0],
        "relative_error": [0.5, 1.5, 0.5, np.nan, np.nan, 1.0],
        "valid_fit": [True, False, True, False, False, False],
        "failure_reason": [None, "boundary_hit", None, "fit_exception", "nonfinite_parameters", "optimizer_failed"],
        "optimizer_success": [True, True, True, False, False, False],
        "boundary_hit": [False, True, False, False, False, False],
        "initial_lifetime_ns": [1.5] * 6,
        "fitted_amplitude": [100.0] * 6,
        "fitted_background": [0.5] * 6,
        "fitted_temporal_shift_ns": [0.05] * 6,
        "poisson_nll": [4.0] * 6,
        "poisson_deviance": [2.0] * 6,
        "runtime_ms": [0.25] * 6,
    }, index=pd.Index([80, 20, 80, 40, 10, 60], name="curve"))
    return ReconvolutionBenchmarkResult(
        table, ReconvolutionBenchmarkSummary(6, 2, 4, 1 / 3, 2 / 3, 1.0, 1.0, 1.0, 0.25, 0.25),
    )


@pytest.mark.parametrize("kind", [None, "generating_mono", "primary_component"])
def test_classical_table_keeps_every_original_diagnostic_and_index(classical_result, kind, captured_facts):
    original = classical_result.per_curve.copy(deep=True)
    expected = original.copy()
    expected.insert(0, "estimator_name", "fit")
    expected["predicted_lifetime_ns"] = [3.0, 5.0, 1.0, np.nan, np.inf, 4.0]
    expected["valid_estimate"] = [True, False, True, False, False, False]
    reference = None if kind is None else LifetimeReference("target", kind, [2.0] * 6, [True] * 6)
    actual = conditional.build_classical_prediction_diagnostics(
        classical_result, estimator_name="fit", reference=reference,
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(classical_result.per_curve, original, check_exact=True)
    if kind is None:
        assert captured_facts == []
    else:
        facts = captured_facts[0]
        assert facts.points.method_family.tolist() == ["classical"] * 6
        assert facts.points.representation_id.tolist() == ["raw_histogram"] * 6
        assert facts.points.failure_reason.tolist() == [
            None, "boundary_hit", None, "fit_exception", "nonfinite_parameters", "optimizer_failed",
        ]
        assert facts.reference_comparisons.reference_kind.tolist() == [kind] * 6
        np.testing.assert_array_equal(facts.points.lifetime_estimate_ns, original.fitted_lifetime_ns)
        np.testing.assert_array_equal(facts.reference_comparisons.error_ns, original.error_ns)
        assert facts.reference_comparisons.is_valid_comparison.tolist() == original.valid_fit.tolist()


@pytest.mark.parametrize("explicit", [False, True])
def test_conditional_summary_keeps_visible_invalid_errors_out_of_metrics(classical_result, explicit):
    diagnostics = conditional.build_classical_prediction_diagnostics(
        classical_result, estimator_name="fit",
        reference=LifetimeReference("target", "generating_mono", [2.0] * 6, [True] * 6) if explicit else None,
    )
    expected = pd.DataFrame([
        ("fit", "mixed", 4, 2, 2, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0),
        ("fit", "failed", 2, 0, 2, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan),
    ], columns=["estimator_name", "condition", "n_samples", "n_valid_estimates",
                "n_failed_estimates", "failure_rate", "mae_ns", "median_absolute_error_ns",
                "bias_ns", "p90_absolute_error_ns", "p95_absolute_error_ns"])
    actual = conditional.summarize_conditional_performance(diagnostics, condition_column="condition")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_standard_regimes_keep_right_closed_boundaries_and_order():
    diagnostics = pd.DataFrame({
        "estimator_name": ["method"] * 3, "true_lifetime_ns": [1.0, 2.0, 3.0],
        "signal_photon_count_target": [1, 2, 3], "background_per_bin": [1.0, 2.0, 3.0],
        "irf_fwhm_ns": [1.0, 2.0, 3.0], "irf_shift_ns": [-1.0, 2.0, -3.0],
        "error_ns": [1.0, 2.0, 3.0], "absolute_error_ns": [1.0, 2.0, 3.0],
        "valid_estimate": [True, False, True],
    }, index=[80, 20, 80])
    edges = [0.0, 1.0, 2.0, np.inf]
    result = conditional.add_benchmark_regimes(
        diagnostics, lifetime_edges=edges, photon_count_edges=edges,
        background_edges=edges, irf_width_edges=edges, irf_misalignment_edges=edges,
    )
    pd.testing.assert_index_equal(result.index, diagnostics.index)
    summaries = conditional.summarize_standard_regimes(result)
    assert list(summaries) == ["lifetime", "photon_count", "background", "irf_width", "irf_misalignment"]
    for name, labels in zip(summaries, [
        ["short", "medium", "long"], ["low", "medium", "high"],
        ["low", "medium", "high"], ["narrow", "medium", "broad"], ["small", "medium", "large"],
    ]):
        summary = summaries[name]
        assert result[name + "_regime"].tolist() == labels
        assert summary[name + "_regime"].tolist() == labels
        assert summary.n_samples.tolist() == [1, 1, 1]
        assert summary.n_valid_estimates.tolist() == [1, 0, 1]
        assert summary.n_failed_estimates.tolist() == [0, 1, 0]
        np.testing.assert_array_equal(summary.mae_ns, [1.0, np.nan, 3.0])


@pytest.fixture
def captured_facts(monkeypatch):
    results = []
    def capture(**kwargs):
        result = build_point_evaluation(**kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(conditional, "build_point_evaluation", capture)
    return results


@pytest.mark.parametrize("name,family", [
    ("ridge-like-user-method", "ml"), ("constant_mean", "baseline"),
    ("arbitrary-baseline", "baseline"), ("classical_reconvolution_custom", "ml"),
    ("constant_mean", "ml"),  # Even a traditional baseline name can declare ML.
])
def test_explicit_regression_family_and_identity_come_only_from_descriptor(name, family, captured_facts):
    metadata = pd.DataFrame({"sample_id": [9, 2]}, index=pd.Index([80, 20], name="source"))
    split, result = regression_case(metadata, [2.0, 4.0], [3.0, 2.0], name="unrelated_legacy_name")
    actual = conditional.build_ml_prediction_diagnostics(
        split=split, result=result, method=MethodDescriptor(name, family, "user_representation"),
        reference=LifetimeReference("known_target", "generating_mono", [2.0, 4.0], [True, True]),
    )
    expected = pd.DataFrame({
        "estimator_name": [name] * 2, "sample_id": [9, 2],
        "true_lifetime_ns": [2.0, 4.0], "predicted_lifetime_ns": [3.0, 2.0],
        "error_ns": [1.0, -2.0], "absolute_error_ns": [1.0, 2.0],
        "relative_error": [0.5, 0.5], "valid_estimate": [True, True],
    }, index=metadata.index)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    facts = captured_facts[0]
    assert facts.points.method_id.tolist() == [name] * 2
    assert facts.points.method_family.tolist() == [family] * 2
    assert facts.points.representation_id.tolist() == ["user_representation"] * 2
    assert facts.reference_comparisons.reference_kind.tolist() == ["generating_mono"] * 2
    assert facts.reference_comparisons.reference_id.tolist() == ["known_target"] * 2


def test_primary_component_reference_is_not_relabelled(captured_facts):
    conditional.build_prediction_diagnostics(
        estimator_name="ignored", y_true=[2.0], y_pred=[3.0], metadata=pd.DataFrame(index=[90]),
        method=MethodDescriptor("estimate", "ml"),
        reference=LifetimeReference("component_1", "primary_component", [2.0], [True]),
    )
    comparison = captured_facts[0].reference_comparisons.iloc[0]
    assert comparison.reference_kind == "primary_component"
    assert comparison.reference_id == "component_1"
    assert comparison.reference_value_ns == 2.0 and comparison.error_ns == 1.0


@pytest.mark.parametrize("semantics", [
    {"method": MethodDescriptor("ml", "ml")},
    {"reference": LifetimeReference("target", "generating_mono", [2.0], [True])},
])
def test_partial_semantic_request_fails_instead_of_guessing(semantics):
    with pytest.raises(ValueError, match="method and reference must be supplied together"):
        conditional.build_prediction_diagnostics(
            estimator_name="legacy", y_true=[2.0], y_pred=[3.0],
            metadata=pd.DataFrame(index=[90]), **semantics,
        )


@pytest.mark.parametrize("values,available", [
    ([4.0, 2.0], [True, True]), ([2.0], [True]), ([2.0, 4.0], [True, False]),
])
def test_explicit_reference_must_match_full_positional_target(values, available):
    with pytest.raises(ValueError, match="fully available and match the target vector"):
        conditional.build_prediction_diagnostics(
            estimator_name="legacy", y_true=[2.0, 4.0], y_pred=[3.0, 3.0],
            metadata=pd.DataFrame(index=[80, 20]), method=MethodDescriptor("ml", "ml"),
            reference=LifetimeReference("target", "generating_mono", values, available),
        )


def test_empty_and_nonfinite_valid_legacy_calls_do_not_weaken_explicit_core():
    for targets, estimates, valid, message in [
        ([], [], [], "nonempty"), ([2.0], [np.inf], [True], "Valid estimates must be finite"),
    ]:
        kwargs = dict(estimator_name="legacy", y_true=targets, y_pred=estimates,
                      valid_mask=valid, metadata=pd.DataFrame(index=range(len(targets))))
        legacy = conditional.build_prediction_diagnostics(**kwargs)
        assert len(legacy) == len(targets)
        if len(legacy):
            assert legacy.valid_estimate.iloc[0] and np.isinf(legacy.error_ns.iloc[0])
        with pytest.raises(ValueError, match=message):
            conditional.build_prediction_diagnostics(
                **kwargs, method=MethodDescriptor("ml", "ml"),
                reference=LifetimeReference("target", "generating_mono", targets,
                                            np.ones(len(targets), dtype=bool)),
            )


def test_explicit_infinite_estimate_display_does_not_change_canonical_errors(captured_facts):
    result = conditional.build_prediction_diagnostics(
        estimator_name="legacy", y_true=[2.0, 2.0], y_pred=[np.inf, -np.inf],
        metadata=pd.DataFrame(index=[90, 20]), method=MethodDescriptor("ml", "ml"),
        reference=LifetimeReference("target", "generating_mono", [2.0, 2.0], [True, True]),
    )
    np.testing.assert_array_equal(result.error_ns, [np.inf, -np.inf])
    np.testing.assert_array_equal(result.relative_error, [np.inf, np.inf])
    facts = captured_facts[0]
    assert not facts.points.is_valid.any()
    assert not facts.reference_comparisons.is_valid_comparison.any()
    assert facts.reference_comparisons[["error_ns", "absolute_error_ns", "relative_error"]].isna().all().all()


def test_classical_minimal_table_does_not_gain_diagnostic_columns():
    table = pd.DataFrame({"true_lifetime_ns": [2], "fitted_lifetime_ns": [3], "valid_fit": [True]}, index=[90])
    result = ReconvolutionBenchmarkResult(
        table, ReconvolutionBenchmarkSummary(1, 1, 0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.25, 0.25),
    )
    legacy = conditional.build_classical_prediction_diagnostics(result)
    explicit = conditional.build_classical_prediction_diagnostics(
        result, reference=LifetimeReference("target", "generating_mono", [2.0], [True]),
    )
    pd.testing.assert_frame_equal(explicit, legacy, check_exact=True)
    assert "error_ns" not in explicit
    assert explicit.predicted_lifetime_ns.dtype == table.fitted_lifetime_ns.dtype


def test_all_failed_classical_population_retains_finite_errors_but_has_no_metric_contributors(classical_result):
    table = classical_result.per_curve.copy()
    table["valid_fit"] = False
    table["failure_reason"] = "controlled_failure"
    result = ReconvolutionBenchmarkResult(table, classical_result.summary)
    diagnostic = conditional.build_classical_prediction_diagnostics(
        result, reference=LifetimeReference("target", "generating_mono", [2.0] * 6, [True] * 6),
    )
    np.testing.assert_array_equal(diagnostic.error_ns, [1.0, 3.0, -1.0, np.nan, np.nan, 2.0])
    summary = conditional.summarize_conditional_performance(diagnostic, condition_column="condition")
    assert summary.n_samples.tolist() == [4, 2]
    assert summary.n_valid_estimates.tolist() == [0, 0]
    assert summary.n_failed_estimates.tolist() == [4, 2]
    assert summary.failure_rate.tolist() == [1.0, 1.0]
    assert summary[["mae_ns", "median_absolute_error_ns", "bias_ns",
                    "p90_absolute_error_ns", "p95_absolute_error_ns"]].isna().all().all()


def test_same_finite_invalid_facts_support_opposite_error_visibility_policies():
    metadata = pd.DataFrame({"sample_id": [7, 3], "condition": ["same"] * 2}, index=[90, 20])
    batch = EvaluationBatch(
        "arbitrary_evaluation", [7, 3], metadata=metadata,
        references={"target": LifetimeReference("target", "generating_mono", [2.0, 4.0], [True, True])},
    )
    facts = build_point_evaluation(
        batch=batch, method=MethodDescriptor("classical_reconvolution_custom", "ml"),
        lifetime_estimates_ns=[5.0, 4.0], is_valid=[False, True],
    )
    generalization = _project_generalization_predictions(batch=batch, result=facts, reference_id="target")
    diagnostic = conditional._project_conditional_predictions(result=facts, reference_id="target", metadata=metadata)
    np.testing.assert_array_equal(generalization.error_ns, [np.nan, 0.0])
    np.testing.assert_array_equal(generalization.absolute_error_ns, [np.nan, 0.0])
    np.testing.assert_array_equal(diagnostic.error_ns, [3.0, 0.0])
    np.testing.assert_array_equal(diagnostic.absolute_error_ns, [3.0, 0.0])
    np.testing.assert_array_equal(diagnostic.relative_error, [1.5, 0.0])
    assert facts.points.method_family.tolist() == ["ml", "ml"]
    assert facts.points.is_valid.tolist() == [False, True]
    assert facts.reference_comparisons.is_valid_comparison.tolist() == [False, True]
    assert facts.reference_comparisons.error_ns.tolist() == [3.0, 0.0]
    assert generalization.index.tolist() == [0, 1]
    pd.testing.assert_index_equal(diagnostic.index, metadata.index)
    summary = conditional.summarize_conditional_performance(diagnostic, condition_column="condition").iloc[0]
    assert (summary.n_samples, summary.n_valid_estimates, summary.n_failed_estimates) == (2, 1, 1)
    assert summary.mae_ns == 0.0 and summary.failure_rate == 0.5
