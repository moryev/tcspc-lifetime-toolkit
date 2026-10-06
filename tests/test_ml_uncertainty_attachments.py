"""ML uncertainty source results projected onto explicit point identities."""

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import EvaluationBatch, MethodDescriptor
from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
from tcspc_toolkit.generalization_evaluation import _build_generalization_evaluation_batch
from tcspc_toolkit.ml_evaluation import BenchmarkDataset
from tcspc_toolkit.ml_uncertainty import (
    MLUncertaintyScoreCalibrationResult,
    QuantileGradientBoostingCalibrationResult,
    _QUANTILE_MEDIAN_POINT_METHOD,
    _development_calibration_batch,
    _frozen_ml_uncertainty_batch,
    evaluate_frozen_ml_uncertainty_score,
    project_ml_prediction_intervals,
    project_ml_uncertainty_scores,
)
from tcspc_toolkit.uncertainty_evaluation import (
    PredictionIntervalResult, UncertaintyDevelopmentSplit, UncertaintyScoreResult,
    evaluate_prediction_intervals, evaluate_quantile_prediction_intervals,
    evaluate_uncertainty_scores,
)
from tcspc_toolkit.uncertainty_robustness import (
    conformalize_quantile_gradient_boosting,
    evaluate_frozen_conformalized_quantile,
)


def _facts(prediction=(2.0, 3.0, 4.0), valid=(True, False, True)):
    batch = EvaluationBatch("external/custom", ["z", "a", "b"])
    method = MethodDescriptor("classical_reconvolution_custom", "ml", "custom_features")
    points = build_point_evaluation(
        batch=batch, method=method,
        lifetime_estimates_ns=prediction, is_valid=valid,
    )
    return batch, method, points


def test_quantile_projection_preserves_crossings_nonfinite_bounds_and_point_validity():
    batch, method, points = _facts()
    intervals = PredictionIntervalResult(
        prediction=np.array([2.0, 3.0, 4.0]),
        lower=np.array([1.0, 4.0, np.nan]),
        upper=np.array([3.0, 2.0, 5.0]),
        nominal_coverage=0.9, method_id="quantile_gradient_boosting",
    )
    attached = project_ml_prediction_intervals(
        batch=batch, points=points, point_method=method,
        intervals=intervals, interval_kind="quantile_prediction",
    ).intervals
    assert attached.evaluation_id.tolist() == ["external/custom"] * 3
    assert attached.sample_id.tolist() == ["z", "a", "b"]
    assert attached.method_id.tolist() == ["classical_reconvolution_custom"] * 3
    assert attached.representation_id.tolist() == ["custom_features"] * 3
    assert attached.uncertainty_method_id.tolist() == ["quantile_gradient_boosting"] * 3
    assert attached.is_valid_interval.tolist() == [True, False, False]
    assert attached.nominal_level.tolist() == [0.9] * 3
    assert attached.loc[1, "lower_ns"] == 4.0
    assert attached.loc[1, "upper_ns"] == 2.0
    assert points.points.is_valid.tolist() == [True, False, True]
    assert points.reference_comparisons.empty


def test_interval_and_point_validity_are_independent_when_bounds_are_ordered():
    batch, method, points = _facts(valid=(False, True, True))
    intervals = PredictionIntervalResult(
        prediction=np.array([2.0, 3.0, 4.0]),
        lower=np.array([1.0, 4.0, 3.0]), upper=np.array([3.0, 2.0, 5.0]),
        nominal_coverage=0.9, method_id="quantile_gradient_boosting",
    )
    attached = project_ml_prediction_intervals(
        batch=batch, points=points, point_method=method,
        intervals=intervals, interval_kind="quantile_prediction",
    ).intervals
    assert attached.is_valid_interval.tolist() == [True, False, True]
    assert points.points.is_valid.tolist() == [False, True, True]


def test_projection_rejects_different_point_predictions_or_sample_identity():
    batch, method, points = _facts()
    source = PredictionIntervalResult(
        prediction=np.array([2.0, 3.1, 4.0]),
        lower=np.array([1.0, 2.0, 3.0]), upper=np.array([3.0, 4.0, 5.0]),
        nominal_coverage=0.9, method_id="quantile_gradient_boosting",
    )
    with pytest.raises(ValueError, match="differ from canonical point"):
        project_ml_prediction_intervals(
            batch=batch, points=points, point_method=method,
            intervals=source, interval_kind="quantile_prediction",
        )
    wrong_batch = EvaluationBatch("external/custom", ["z", "a", "unknown"])
    with pytest.raises(ValueError, match="do not match"):
        project_ml_prediction_intervals(
            batch=wrong_batch, points=points, point_method=method,
            intervals=source, interval_kind="quantile_prediction",
        )


def test_raw_conformal_and_two_scores_attach_to_same_point_without_collision():
    batch, method, points = _facts(valid=(True, True, True))
    prediction = np.array([2.0, 3.0, 4.0])
    raw = PredictionIntervalResult(
        prediction=prediction, lower=prediction - 0.2, upper=prediction + 0.2,
        nominal_coverage=0.9, method_id="quantile_gradient_boosting",
    )
    conformal = PredictionIntervalResult(
        prediction=prediction, lower=raw.lower - 0.3, upper=raw.upper + 0.3,
        nominal_coverage=raw.nominal_coverage,
        method_id="conformalized_quantile_gradient_boosting",
    )
    raw_attachment = project_ml_prediction_intervals(
        batch=batch, points=points, point_method=method,
        intervals=raw, interval_kind="quantile_prediction",
    )
    conformal_attachment = project_ml_prediction_intervals(
        batch=batch, points=points, point_method=method,
        intervals=conformal, interval_kind="conformal_prediction",
    )
    np.testing.assert_array_equal(
        conformal_attachment.intervals.lower_ns, raw_attachment.intervals.lower_ns - 0.3,
    )
    assert conformal_attachment.intervals.interval_kind.unique().tolist() == ["conformal_prediction"]
    assert conformal_attachment.intervals.nominal_level.tolist() == [0.9] * 3
    rf = UncertaintyScoreResult(prediction, np.array([0.0, 0.2, 0.4]), "random_forest_tree_spread")
    bootstrap = UncertaintyScoreResult(prediction, np.array([0.1, np.nan, 0.3]), "ml_training_bootstrap")
    rf_attachment = project_ml_uncertainty_scores(
        batch=batch, points=points, point_method=method, scores=rf,
    )
    bootstrap_attachment = project_ml_uncertainty_scores(
        batch=batch, points=points, point_method=method, scores=bootstrap,
    )
    np.testing.assert_array_equal(rf_attachment.scores.score, rf.uncertainty_score)
    np.testing.assert_array_equal(bootstrap_attachment.scores.score, bootstrap.uncertainty_score)
    assert bootstrap_attachment.scores.is_valid_score.tolist() == [True, False, True]
    assert rf_attachment.scores.uncertainty_method_id.unique().tolist() != (
        bootstrap_attachment.scores.uncertainty_method_id.unique().tolist()
    )
    assert "nominal_level" not in rf_attachment.scores


def test_frozen_batch_uses_stage3_test_and_sample_identity_after_selection():
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    test = GeneralizationTestMeasurements(
        test_id="A", time=np.array([0.0, 1.0]),
        X_histograms=np.zeros((3, 2), dtype=np.int64),
        y=np.array([1.0, 2.0, 3.0]),
        metadata=pd.DataFrame({"sample_id": ["a", "b", "c"]}),
    )
    stage3 = _build_generalization_evaluation_batch(
        test=test, representations={"engineered_features": features},
    )
    selected = _frozen_ml_uncertainty_batch(
        test, features.iloc[[2, 0]], np.array([2, 0], dtype=np.int64),
    )
    assert selected.evaluation_id == stage3.evaluation_id == "A"
    assert selected.sample_ids == (stage3.sample_ids[2], stage3.sample_ids[0])
    assert selected.references == {}


def test_calibration_uses_positional_identity_when_legacy_sample_ids_repeat():
    development = BenchmarkDataset(
        X_features=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        X_histograms=np.zeros((3, 2), dtype=np.int64),
        y=np.array([1.0, 2.0, 3.0]),
        metadata=pd.DataFrame({"sample_id": ["duplicate"] * 3}),
    )
    batch = _development_calibration_batch(
        development, np.array([2, 0], dtype=np.int64),
        development.X_features.iloc[[2, 0]],
    )
    assert batch.sample_ids == (2, 0)


def test_conformal_workflow_projects_corrected_bounds_with_no_refit(monkeypatch):
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    development = BenchmarkDataset(
        X_features=features, X_histograms=np.zeros((4, 2), dtype=np.int64),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        metadata=pd.DataFrame({"sample_id": ["s0", "s1", "s2", "s3"]}),
    )
    split = UncertaintyDevelopmentSplit((0, 1), (2, 3), 57, 0.5)
    raw = PredictionIntervalResult(
        prediction=np.array([3.0, 4.0]), lower=np.array([2.0, 3.0]),
        upper=np.array([2.5, 3.5]), nominal_coverage=0.9,
        method_id="quantile_gradient_boosting",
    )

    class FrozenPredictor:
        calls = 0

        def predict(self, X):
            self.calls += 1
            return raw

    predictor = FrozenPredictor()
    calibration = QuantileGradientBoostingCalibrationResult(
        fitted_estimator=predictor, split=split, calibration_intervals=raw,
        interval_metrics=evaluate_prediction_intervals(development.y[2:], raw),
        quantile_metrics=evaluate_quantile_prediction_intervals(
            development.y[2:], raw, lower_quantile=0.05,
            median_quantile=0.5, upper_quantile=0.95,
        ),
        feature_names=("x",),
    )
    import tcspc_toolkit.uncertainty_robustness as robustness

    captured = []
    original = robustness._project_ml_interval_source

    def capture(*args):
        result = original(*args)
        captured.append(result)
        return result

    monkeypatch.setattr(robustness, "_project_ml_interval_source", capture)
    conformal = conformalize_quantile_gradient_boosting(calibration, development)
    assert conformal.correction_ns == 0.5
    assert predictor.calls == 0
    calibration_rows = captured[-1].intervals
    assert calibration_rows.sample_id.tolist() == ["s2", "s3"]
    np.testing.assert_array_equal(calibration_rows.lower_ns, raw.lower - 0.5)
    assert calibration_rows.interval_kind.unique().tolist() == ["conformal_prediction"]
    external_batch = EvaluationBatch("external/custom", ["x", "y"])
    result = evaluate_frozen_conformalized_quantile(
        conformal, features.iloc[2:], development.y[2:], test_id="external/custom",
        evaluation_batch=external_batch, point_method=_QUANTILE_MEDIAN_POINT_METHOD,
    )
    assert predictor.calls == 1
    np.testing.assert_array_equal(result.intervals.lower, raw.lower - 0.5)
    assert result.interval_metrics.empirical_coverage == 1.0
    assert captured[-1].intervals.sample_id.tolist() == ["x", "y"]
    assert captured[-1].intervals.nominal_level.tolist() == [0.9, 0.9]


def test_external_score_projection_preserves_source_metrics_and_explicit_point_link(monkeypatch):
    import tcspc_toolkit.ml_uncertainty as ml_uncertainty

    source = UncertaintyScoreResult(
        np.array([2.0, 3.0]), np.array([0.2, 0.0]),
        "random_forest_tree_spread",
    )

    class FrozenPredictor:
        calls = 0

        def predict(self, X):
            self.calls += 1
            return source

    predictor = FrozenPredictor()
    calibration = MLUncertaintyScoreCalibrationResult(
        fitted_estimator=predictor,
        split=UncertaintyDevelopmentSplit((0, 1), (2, 3), 57, 0.5),
        calibration_scores=source,
        score_metrics=evaluate_uncertainty_scores(np.array([2.0, 3.0]), source),
        feature_names=("x",),
    )
    batch = EvaluationBatch("external/custom", ["first", "second"])
    method = MethodDescriptor("random_forest", "ml", "engineered_features")
    captured = []
    original = ml_uncertainty._project_ml_score_source

    def capture(*args):
        result = original(*args)
        captured.append(result)
        return result

    monkeypatch.setattr(ml_uncertainty, "_project_ml_score_source", capture)
    result = evaluate_frozen_ml_uncertainty_score(
        calibration, pd.DataFrame({"x": [0.0, 1.0]}), np.array([2.0, 3.0]),
        condition_id="external/custom", evaluation_batch=batch, point_method=method,
    )
    assert predictor.calls == 1
    assert result.scores is source
    expected_metrics = evaluate_uncertainty_scores(np.array([2.0, 3.0]), source)
    for field, expected in vars(expected_metrics).items():
        np.testing.assert_allclose(getattr(result.score_metrics, field), expected, equal_nan=True)
    assert captured[0].scores.method_id.tolist() == ["random_forest"] * 2
    assert captured[0].scores.sample_id.tolist() == ["first", "second"]
    np.testing.assert_array_equal(captured[0].scores.score, source.uncertainty_score)
