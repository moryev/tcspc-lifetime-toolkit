"""Independent small-table expectations for the frozen Week-9 reports."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import EvaluationBatch, MethodDescriptor
from tcspc_toolkit.evaluation_uncertainty import IntervalAttachment, ScoreAttachment
from tcspc_toolkit.generalization import default_generalization_suite
from tcspc_toolkit.ml_uncertainty import (
    QUANTILE_GRADIENT_BOOSTING_METHOD_ID,
    RANDOM_FOREST_TREE_SPREAD_METHOD_ID,
)
from tcspc_toolkit.uncertainty_evaluation import (
    PredictionIntervalResult,
    UncertaintyScoreResult,
    evaluate_prediction_intervals,
)
from tcspc_toolkit import uncertainty_robustness as robustness


def _classical_rows() -> pd.DataFrame:
    rows = []
    for test_id in "ABCDEF":
        for index, photon_count in enumerate((250, 50_000)):
            rows.append({
                "test_id": test_id,
                "true_lifetime_ns": float(index + 1),
                "fitted_lifetime_ns": float(index + 1) + 0.1,
                "covariance_std_ns": 0.1,
                "covariance_lower_ns": float(index + 1) - 0.2,
                "covariance_upper_ns": float(index + 1) + 0.2,
                "bootstrap_std_ns": 0.2,
                "bootstrap_lower_ns": float(index + 1) - 0.3,
                "bootstrap_upper_ns": float(index + 1) + 0.3,
                "bootstrap_fit_failure_rate": 0.25,
                "signal_photon_count_target": photon_count,
                "background_per_bin": 0.1,
                "model_mismatch_severity": ("weak", "moderate")[index],
            })
    return pd.DataFrame(rows)


def test_classical_scorecards_have_independent_frozen_order_and_values() -> None:
    per_curve = _classical_rows()
    scorecard = robustness.build_classical_uncertainty_scorecard(
        per_curve, nominal_coverage=0.9,
    )
    assert list(zip(scorecard.test_id, scorecard.method)) == [
        (test_id, method) for test_id in "ABCDEF"
        for method in ("covariance", "parametric_bootstrap")
    ]
    assert scorecard.columns.tolist() == [
        "method", "test_id", "target_reference", "n_samples",
        "n_valid_predictions", "fit_failure_rate", "mae_ns",
        "nominal_coverage", "empirical_coverage", "coverage_gap",
        "mean_width_ns", "median_width_ns", "mean_uncertainty_std_ns",
        "mean_abs_error_over_std", "median_abs_error_over_std",
        "interval_score", "interval_failure_rate",
        "mean_bootstrap_refit_failure_rate",
    ]
    assert scorecard.n_samples.eq(2).all()
    assert scorecard.n_valid_predictions.eq(2).all()
    assert scorecard.fit_failure_rate.eq(0).all()
    np.testing.assert_allclose(scorecard.mae_ns, 0.1)
    np.testing.assert_allclose(scorecard.empirical_coverage, 1.0)
    np.testing.assert_allclose(scorecard.mean_width_ns.iloc[::2], 0.4)
    np.testing.assert_allclose(scorecard.mean_width_ns.iloc[1::2], 0.6)
    np.testing.assert_allclose(
        scorecard.mean_bootstrap_refit_failure_rate.iloc[1::2], 0.25,
    )
    assert scorecard.mean_bootstrap_refit_failure_rate.iloc[::2].isna().all()
    assert scorecard.loc[scorecard.test_id.eq("F"), "target_reference"].eq(
        "dominant_component_tau_1"
    ).all()

    conditional = robustness.build_classical_conditional_uncertainty_scorecard(
        per_curve, definition=default_generalization_suite(), nominal_coverage=0.9,
    )
    assert list(zip(conditional.condition_id, conditional.method)) == [
        (condition, method)
        for condition in (
            "B_low_photon", "B_high_photon", "D_high_background",
            "F_weak_mismatch", "F_moderate_mismatch",
        )
        for method in ("covariance", "parametric_bootstrap")
    ]
    assert conditional.columns.tolist() == [
        "method", "condition_id", "test_id", "target_reference",
        "n_samples", "mae_ns", "empirical_coverage", "coverage_gap",
        "mean_width_ns", "mean_uncertainty_std_ns",
        "mean_abs_error_over_std", "median_abs_error_over_std",
        "fit_failure_rate", "interval_failure_rate",
    ]
    assert conditional.n_samples.tolist() == [1, 1, 1, 1, 2, 2, 1, 1, 1, 1]


def test_classical_report_preserves_frozen_condition_traversal_and_fields(monkeypatch) -> None:
    source = _classical_rows()
    tests = {
        test_id: SimpleNamespace(
            test_id=test_id, y=np.array([1.0, 2.0]),
            metadata=pd.DataFrame({"primary_lifetime_ns": [1.0, 2.0]}),
        )
        for test_id in "ABCDEF"
    }
    visited = []

    def evaluate_test(*, test, **kwargs):
        visited.append(test.test_id)
        return source.loc[source.test_id.eq(test.test_id)].copy()

    monkeypatch.setattr(robustness, "_evaluate_classical_uncertainty_test", evaluate_test)
    report = robustness.evaluate_classical_uncertainty_robustness(
        prepared=SimpleNamespace(tests=tests),
        definition=default_generalization_suite(),
        nominal_coverage=0.9, n_bootstrap_resamples=4,
    )
    assert visited == list("ABCDEF")
    assert report.per_curve.test_id.tolist() == [test_id for test_id in "ABCDEF" for _ in range(2)]
    pd.testing.assert_frame_equal(report.per_curve, source)
    pd.testing.assert_frame_equal(
        report.scorecard,
        robustness.build_classical_uncertainty_scorecard(source, nominal_coverage=0.9),
    )
    pd.testing.assert_frame_equal(
        report.conditional_scorecard,
        robustness.build_classical_conditional_uncertainty_scorecard(
            source, definition=default_generalization_suite(), nominal_coverage=0.9,
        ),
    )
    assert report.nominal_coverage == 0.9
    assert report.n_bootstrap_resamples == 4


def test_classical_scorecard_keeps_attempted_rows_in_all_failed_condition() -> None:
    per_curve = _classical_rows()
    per_curve.loc[per_curve.test_id.eq("A"), "fitted_lifetime_ns"] = np.nan
    per_curve.loc[per_curve.test_id.eq("A"), [
        "covariance_lower_ns", "covariance_upper_ns",
        "bootstrap_lower_ns", "bootstrap_upper_ns",
    ]] = np.nan
    scorecard = robustness.build_classical_uncertainty_scorecard(
        per_curve, nominal_coverage=0.9,
    )
    failed = scorecard.loc[scorecard.test_id.eq("A")]
    assert failed.n_samples.tolist() == [2, 2]
    assert failed.n_valid_predictions.tolist() == [0, 0]
    assert failed.fit_failure_rate.tolist() == [1.0, 1.0]
    assert failed.interval_failure_rate.tolist() == [1.0, 1.0]
    assert failed.mae_ns.isna().all()


def test_ml_report_has_independent_schema_order_and_metric_expectations(monkeypatch) -> None:
    y = np.array([1.0, 2.0])
    prediction = np.array([1.1, 1.9])
    tests = {
        test_id: SimpleNamespace(
            test_id=test_id, y=y,
            metadata=pd.DataFrame({"sample_id": ["s0", "s1"]}),
        )
        for test_id in "ABCDEF"
    }
    prepared = SimpleNamespace(
        tests=tests,
        X_features={test_id: pd.DataFrame({"feature": [1.0, 2.0]}) for test_id in tests},
    )
    raw = PredictionIntervalResult(
        prediction=prediction, lower=np.array([0.9, 1.8]),
        upper=np.array([1.2, 2.2]), nominal_coverage=0.9,
        method_id=QUANTILE_GRADIENT_BOOSTING_METHOD_ID,
    )
    conformal = PredictionIntervalResult(
        prediction=prediction, lower=np.array([0.8, 1.7]),
        upper=np.array([1.3, 2.3]), nominal_coverage=0.9,
        method_id=robustness.CONFORMALIZED_QUANTILE_METHOD_ID,
    )

    def interval_result(intervals):
        return SimpleNamespace(
            intervals=intervals,
            interval_metrics=evaluate_prediction_intervals(y, intervals),
            median_prediction_mae_ns=0.1,
        )

    monkeypatch.setattr(
        robustness, "evaluate_frozen_quantile_gradient_boosting",
        lambda *args, **kwargs: interval_result(raw),
    )
    monkeypatch.setattr(
        robustness, "evaluate_frozen_conformalized_quantile",
        lambda *args, **kwargs: interval_result(conformal),
    )
    score = UncertaintyScoreResult(
        prediction=prediction, uncertainty_score=np.array([0.2, 0.4]),
        method_id=RANDOM_FOREST_TREE_SPREAD_METHOD_ID,
    )
    score_metrics = SimpleNamespace(
        mean_absolute_error_ns=0.1, spearman_error_correlation=1.0,
        low_uncertainty_mae_ns=0.1, high_uncertainty_mae_ns=0.1,
        score_failure_rate=0.0,
    )
    monkeypatch.setattr(
        robustness, "evaluate_frozen_ml_uncertainty_score",
        lambda *args, **kwargs: SimpleNamespace(scores=score, score_metrics=score_metrics),
    )
    score_calibration = SimpleNamespace(
        calibration_scores=SimpleNamespace(method_id=RANDOM_FOREST_TREE_SPREAD_METHOD_ID),
    )
    report = robustness.build_ml_uncertainty_robustness_report(
        prepared=prepared, quantile_calibration=object(),
        conformal_calibration=SimpleNamespace(correction_ns=0.1),
        score_calibrations={"user_rf_label": score_calibration},
    )
    assert list(zip(report.interval_scorecard.test_id, report.interval_scorecard.method)) == [
        (test_id, method) for test_id in "ABCDEF"
        for method in (
            QUANTILE_GRADIENT_BOOSTING_METHOD_ID,
            robustness.CONFORMALIZED_QUANTILE_METHOD_ID,
        )
    ]
    assert report.interval_scorecard.columns.tolist() == [
        "method", "test_id", "target_reference", "mae_ns",
        "nominal_coverage", "empirical_coverage", "coverage_gap",
        "mean_width_ns", "median_width_ns", "interval_score",
        "interval_failure_rate",
    ]
    assert report.score_only_scorecard.columns.tolist() == [
        "method", "test_id", "target_reference", "mae_ns",
        "mean_uncertainty_score", "error_score_spearman",
        "low_uncertainty_mae_ns", "high_uncertainty_mae_ns",
        "score_failure_rate",
    ]
    assert report.score_only_scorecard.method.tolist() == ["user_rf_label"] * 6
    np.testing.assert_allclose(report.score_only_scorecard.mean_uncertainty_score, 0.3)
    np.testing.assert_allclose(report.interval_scorecard.mae_ns, 0.1)
    assert report.interval_scorecard.loc[
        report.interval_scorecard.test_id.eq("F"), "target_reference"
    ].eq("dominant_component_tau_1").all()
    assert report.conformal_correction_ns == 0.1


def test_historical_custom_score_keeps_table_only_path_without_identity_guessing(monkeypatch) -> None:
    tests = {
        test_id: SimpleNamespace(
            test_id=test_id, y=np.array([1.0, 2.0]),
            metadata=pd.DataFrame({"sample_id": ["duplicate", "duplicate"]}),
        )
        for test_id in "ABCDEF"
    }
    prepared = SimpleNamespace(
        tests=tests,
        X_features={test_id: pd.DataFrame({"x": [1.0, 2.0]}) for test_id in tests},
    )
    calibration = SimpleNamespace(
        calibration_scores=SimpleNamespace(method_id="opaque_user_score"),
    )
    scores = UncertaintyScoreResult(
        prediction=np.array([1.0, 2.0]),
        uncertainty_score=np.array([0.2, 0.4]), method_id="opaque_user_score",
    )
    metrics = SimpleNamespace(
        mean_absolute_error_ns=0.0, spearman_error_correlation=np.nan,
        low_uncertainty_mae_ns=0.0, high_uncertainty_mae_ns=0.0,
        score_failure_rate=0.0,
    )
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(scores=scores, score_metrics=metrics)

    monkeypatch.setattr(robustness, "evaluate_frozen_ml_uncertainty_score", evaluate)
    card = robustness.build_ml_uncertainty_scorecard(
        prepared=prepared, calibration_results={"custom_label": calibration},
    )
    assert card.method.tolist() == ["custom_label"] * 6
    assert card.test_id.tolist() == list("ABCDEF")
    assert all(call["evaluation_batch"] is None and call["point_method"] is None for call in calls)


def _report_attachments():
    batch = EvaluationBatch("A", ["first", "second"])
    method = MethodDescriptor("random_forest", "ml", "engineered_features")
    points = build_point_evaluation(
        batch=batch, method=method,
        lifetime_estimates_ns=[1.0, 2.0], is_valid=[True, False],
    )
    point_keys = {
        "evaluation_id": ["A", "A"],
        "sample_id": ["first", "second"],
        "method_id": ["random_forest", "random_forest"],
        "representation_id": ["engineered_features", "engineered_features"],
    }
    interval = IntervalAttachment(points, pd.DataFrame({
        **point_keys,
        "uncertainty_method_id": ["quantile_interval"] * 2,
        "interval_kind": ["quantile_prediction"] * 2,
        "nominal_level": [0.9, 0.9],
        "lower_ns": [0.8, 1.8],
        "upper_ns": [1.2, 2.2],
        "is_valid_interval": [True, False],
    }))
    score = ScoreAttachment(points, pd.DataFrame({
        **point_keys,
        "uncertainty_method_id": ["random_forest_tree_spread"] * 2,
        "score": [0.1, np.nan],
        "is_valid_score": [True, False],
    }))
    return batch, method, interval, score


def test_report_alignment_keeps_invalid_attachment_rows_and_distinct_semantics() -> None:
    batch, method, interval, score = _report_attachments()
    interval_rows = robustness._report_interval_rows(
        interval, batch=batch, method=method,
        uncertainty_method_id="quantile_interval",
        interval_kind="quantile_prediction", nominal_level=0.9,
    )
    score_rows = robustness._report_score_rows(
        score, batch=batch, method=method,
        uncertainty_method_id="random_forest_tree_spread",
    )
    assert interval_rows.is_valid_interval.tolist() == [True, False]
    assert score_rows.is_valid_score.tolist() == [True, False]
    assert interval_rows.sample_id.tolist() == score_rows.sample_id.tolist()
    assert "nominal_level" not in score_rows


@pytest.mark.parametrize("changed", [
    {"method": MethodDescriptor("other_point", "ml", "engineered_features")},
    {"method": MethodDescriptor("random_forest", "ml", "other_features")},
    {"batch": EvaluationBatch("B", ["first", "second"])},
    {"batch": EvaluationBatch("A", ["second", "first"])},
    {"uncertainty_method_id": "conformal_interval"},
    {"interval_kind": "conformal_prediction"},
    {"nominal_level": 0.8},
    {"uncertainty_method_id": "poisson_local_covariance"},
    {"interval_kind": "classical_parametric_bootstrap"},
])
def test_report_interval_alignment_rejects_conflicting_identity(changed) -> None:
    batch, method, interval, _ = _report_attachments()
    expected = {
        "batch": batch, "method": method,
        "uncertainty_method_id": "quantile_interval",
        "interval_kind": "quantile_prediction", "nominal_level": 0.9,
    }
    expected.update(changed)
    with pytest.raises(ValueError, match="Attachment|Interval"):
        robustness._report_interval_rows(interval, **expected)


@pytest.mark.parametrize("uncertainty_method_id", [
    "ml_training_bootstrap", "parametric_poisson_bootstrap",
])
def test_report_score_alignment_rejects_other_uncertainty_method(
    uncertainty_method_id,
) -> None:
    batch, method, _, score = _report_attachments()
    with pytest.raises(ValueError, match="Attachment identity"):
        robustness._report_score_rows(
            score, batch=batch, method=method,
            uncertainty_method_id=uncertainty_method_id,
        )
