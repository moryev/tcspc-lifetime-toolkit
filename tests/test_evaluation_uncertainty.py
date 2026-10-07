"""Small reporting attachments; no uncertainty fitting or coverage policy."""

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.classical_uncertainty import ParametricPoissonBootstrapResult
from tcspc_toolkit.bayesian_sampling import BayesianParameterSummary
from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import EvaluationBatch, MethodDescriptor
from tcspc_toolkit.evaluation_uncertainty import IntervalAttachment, ScoreAttachment
from tcspc_toolkit.uncertainty_evaluation import PredictionIntervalResult, UncertaintyScoreResult


def _points():
    return build_point_evaluation(
        batch=EvaluationBatch("experiment/without-reference", ["a", "b", "c"]),
        method=MethodDescriptor("point/estimate", "ml", "prepared/features"),
        lifetime_estimates_ns=[2.0, 3.0, np.nan],
        is_valid=[True, False, False],
        failure_reasons=[None, "method rejected", "missing prediction"],
    )


def _intervals():
    return pd.DataFrame({
        "evaluation_id": ["experiment/without-reference"] * 3,
        "sample_id": ["a", "b", "c"],
        "method_id": ["point/estimate"] * 3,
        "representation_id": ["prepared/features"] * 3,
        "uncertainty_method_id": ["quantile"] * 3,
        "interval_kind": ["quantile_prediction"] * 3,
        "nominal_level": [0.9] * 3,
        "lower_ns": [1.0, 4.0, np.nan],
        "upper_ns": [3.0, 2.0, np.nan],
        "is_valid_interval": [True, False, False],
    })


def _scores():
    return pd.DataFrame({
        "evaluation_id": ["experiment/without-reference"] * 3,
        "sample_id": ["a", "b", "c"],
        "method_id": ["point/estimate"] * 3,
        "representation_id": ["prepared/features"] * 3,
        "uncertainty_method_id": ["tree-spread"] * 3,
        "score": [0.0, -0.25, np.nan],
        "is_valid_score": [True, True, False],
    })


def test_interval_rows_preserve_raw_crossing_and_independent_validity_without_reference():
    points = _points()
    source = _intervals()
    attachment = IntervalAttachment(points, source)
    assert points.reference_comparisons.empty
    assert attachment.intervals.columns.tolist() == list(source.columns)
    assert attachment.intervals.is_valid_interval.tolist() == [True, False, False]
    assert attachment.intervals.loc[1, "lower_ns"] == 4.0
    assert attachment.intervals.loc[1, "upper_ns"] == 2.0
    assert points.points.is_valid.tolist() == [True, False, False]
    source.loc[0, "lower_ns"] = 100
    returned = attachment.intervals
    returned.loc[0, "upper_ns"] = -1
    assert attachment.intervals.loc[0, "lower_ns"] == 1.0
    assert attachment.intervals.loc[0, "upper_ns"] == 3.0


def test_interval_validity_is_independent_of_point_validity():
    table = _intervals().iloc[:2].copy()
    table.loc[0, "is_valid_interval"] = False  # valid point, rejected finite interval
    table.loc[1, "lower_ns"] = 2.0
    table.loc[1, "upper_ns"] = 4.0
    table.loc[1, "is_valid_interval"] = True  # invalid point, accepted interval
    attached = IntervalAttachment(_points(), table).intervals
    assert attached.is_valid_interval.tolist() == [False, True]
    assert _points().points.is_valid.tolist()[:2] == [True, False]


def test_intervals_allow_multiple_levels_and_methods_but_reject_duplicate_identity():
    source = _intervals().iloc[[0]].copy()
    second_level = source.copy()
    second_level["nominal_level"] = 0.95
    other_method = source.copy()
    other_method["uncertainty_method_id"] = "conformal"
    other_method["interval_kind"] = "conformal_prediction"
    attachment = IntervalAttachment(_points(), pd.concat([source, second_level, other_method]))
    assert len(attachment.intervals) == 3
    with pytest.raises(ValueError, match="Duplicate interval attachment"):
        IntervalAttachment(_points(), pd.concat([source, source]))
    conflicting_kind = source.copy()
    conflicting_kind["interval_kind"] = "bayesian_credible"
    with pytest.raises(ValueError, match="cannot change interval kind"):
        IntervalAttachment(_points(), pd.concat([source, conflicting_kind]))


def test_missing_nominal_level_is_distinct_from_finite_level_and_has_stable_identity():
    levelled = _intervals().iloc[[0]].copy()
    unlevelled = levelled.copy()
    unlevelled["nominal_level"] = None
    attached = IntervalAttachment(_points(), pd.concat([levelled, unlevelled]))
    assert attached.intervals.nominal_level.tolist() == [0.9, None]
    same_missing_level = unlevelled.copy()
    same_missing_level["nominal_level"] = np.nan
    with pytest.raises(ValueError, match="Duplicate interval attachment"):
        IntervalAttachment(_points(), pd.concat([unlevelled, same_missing_level]))


@pytest.mark.parametrize("column,value", [
    ("sample_id", "missing"), ("method_id", "other"),
    ("representation_id", None), ("evaluation_id", "other"),
])
def test_interval_attachment_rejects_missing_point_identity(column, value):
    table = _intervals().iloc[[0]].copy()
    table[column] = value
    with pytest.raises(ValueError, match="no matching point"):
        IntervalAttachment(_points(), table)


@pytest.mark.parametrize("lower,upper,valid", [
    (np.nan, 2.0, True), (1.0, np.inf, True), (3.0, 2.0, True),
    (3.0, 2.0, False), (np.inf, 2.0, False),
])
def test_interval_endpoint_policy(lower, upper, valid):
    table = _intervals().iloc[[0]].copy()
    table["lower_ns"], table["upper_ns"], table["is_valid_interval"] = lower, upper, valid
    if valid:
        with pytest.raises(ValueError, match="finite, ordered"):
            IntervalAttachment(_points(), table)
    else:
        assert IntervalAttachment(_points(), table).intervals.loc[0, "lower_ns"] == lower or np.isnan(lower)


@pytest.mark.parametrize("level", [-0.1, 0.0, 1.0, np.inf])
def test_interval_level_must_be_a_probability_when_present(level):
    table = _intervals().iloc[[0]].copy()
    table["nominal_level"] = level
    with pytest.raises(ValueError, match="nominal_level"):
        IntervalAttachment(_points(), table)
    table["nominal_level"] = None
    assert IntervalAttachment(_points(), table).intervals.loc[0, "nominal_level"] is None


def test_score_rows_preserve_zero_signed_and_invalid_scores_independently_of_points():
    points = _points()
    source = _scores()
    attachment = ScoreAttachment(points, source)
    assert points.reference_comparisons.empty
    assert attachment.scores.score.iloc[:2].tolist() == [0.0, -0.25]
    assert attachment.scores.is_valid_score.tolist() == [True, True, False]
    assert points.points.is_valid.tolist() == [True, False, False]
    edited = attachment.scores
    edited.loc[0, "score"] = np.inf
    source.loc[0, "score"] = 42.0
    assert attachment.scores.loc[0, "score"] == 0.0


def test_score_identity_and_numerical_validation():
    source = _scores().iloc[[0]].copy()
    other = source.copy()
    other["uncertainty_method_id"] = "bootstrap-spread"
    assert len(ScoreAttachment(_points(), pd.concat([source, other])).scores) == 2
    with pytest.raises(ValueError, match="Duplicate score attachment"):
        ScoreAttachment(_points(), pd.concat([source, source]))
    orphan = source.copy()
    orphan["sample_id"] = "missing"
    with pytest.raises(ValueError, match="no matching point"):
        ScoreAttachment(_points(), orphan)
    nonfinite = source.copy()
    nonfinite["score"] = np.inf
    with pytest.raises(ValueError, match="Valid scores must be finite"):
        ScoreAttachment(_points(), nonfinite)
    nonfinite["is_valid_score"] = False
    assert np.isinf(ScoreAttachment(_points(), nonfinite).scores.loc[0, "score"])
    nonfinite["is_valid_score"] = 1
    with pytest.raises(ValueError, match="boolean vector"):
        ScoreAttachment(_points(), nonfinite)


def test_interval_and_score_schemas_cannot_be_conflated():
    with pytest.raises(ValueError, match="exactly these columns"):
        IntervalAttachment(_points(), _scores())
    with pytest.raises(ValueError, match="exactly these columns"):
        ScoreAttachment(_points(), _intervals())


def test_existing_quantile_and_rf_result_projection_keeps_source_interpretation():
    quantile = PredictionIntervalResult(
        prediction=np.array([2.0, 3.0, np.nan]),
        lower=np.array([1.0, 4.0, np.nan]), upper=np.array([3.0, 2.0, np.nan]),
        nominal_coverage=0.9, method_id="quantile_gradient_boosting",
    )
    intervals = _intervals()
    intervals["uncertainty_method_id"] = quantile.method_id
    intervals["lower_ns"], intervals["upper_ns"] = quantile.lower, quantile.upper
    intervals["nominal_level"] = quantile.nominal_coverage
    intervals["is_valid_interval"] = quantile.valid_interval_mask
    projected = IntervalAttachment(_points(), intervals).intervals
    np.testing.assert_array_equal(projected.lower_ns, quantile.lower)
    assert projected.is_valid_interval.tolist() == [True, False, False]
    spread = UncertaintyScoreResult(
        prediction=np.array([2.0, 3.0, np.nan]),
        uncertainty_score=np.array([0.0, 0.25, np.nan]),
        method_id="random_forest_tree_spread",
    )
    scores = _scores()
    scores["uncertainty_method_id"] = spread.method_id
    scores["score"], scores["is_valid_score"] = spread.uncertainty_score, spread.valid_score_mask
    projected_scores = ScoreAttachment(_points(), scores).scores
    np.testing.assert_array_equal(projected_scores.score, spread.uncertainty_score)


def test_classical_bootstrap_interval_fixture_is_only_an_endpoint_projection():
    bootstrap = ParametricPoissonBootstrapResult(
        source_lifetime_ns=2.0, lifetime_samples_ns=np.array([1.8, 2.2]),
        bootstrap_std_ns=0.2, bootstrap_median_ns=2.0,
        lower_ns=1.7, upper_ns=2.3, nominal_coverage=0.9,
        n_resamples=2, n_successful_fits=2, n_failed_fits=0, n_boundary_hits=0,
        fit_failure_rate=0.0, boundary_hit_rate=0.0,
        bootstrap_valid=True, failure_reason=None,
    )
    table = _intervals().iloc[[0]].copy()
    table["uncertainty_method_id"] = "parametric_bootstrap"
    table["interval_kind"] = "parametric_bootstrap"
    table["nominal_level"] = bootstrap.nominal_coverage
    table["lower_ns"], table["upper_ns"] = bootstrap.lower_ns, bootstrap.upper_ns
    table["is_valid_interval"] = bootstrap.bootstrap_valid
    projected = IntervalAttachment(_points(), table).intervals.iloc[0]
    assert (projected.lower_ns, projected.upper_ns) == (1.7, 2.3)
    assert projected.is_valid_interval
    assert "lifetime_samples_ns" not in table


def test_bayesian_credible_fixture_keeps_its_conditional_interval_kind():
    summary = BayesianParameterSummary(
        name="lifetime_ns", unit="ns", mean=2.1, median=2.0,
        standard_deviation=0.2, credible_lower=1.7, credible_upper=2.4,
    )
    points = build_point_evaluation(
        batch=EvaluationBatch("experimental/no-reference", ["sample"]),
        method=MethodDescriptor("posterior-median/mono-prior-a-fixed-irf", "bayesian"),
        lifetime_estimates_ns=[summary.median], is_valid=[True],
    )
    table = pd.DataFrame([{
        "evaluation_id": "experimental/no-reference", "sample_id": "sample",
        "method_id": "posterior-median/mono-prior-a-fixed-irf", "representation_id": None,
        "uncertainty_method_id": "posterior-credible/mono-prior-a-fixed-irf",
        "interval_kind": "bayesian_credible", "nominal_level": 0.95,
        "lower_ns": summary.credible_lower, "upper_ns": summary.credible_upper,
        "is_valid_interval": True,
    }])
    attached = IntervalAttachment(points, table).intervals.iloc[0]
    assert attached.interval_kind == "bayesian_credible"
    assert (attached.lower_ns, attached.upper_ns) == (1.7, 2.4)
    assert points.reference_comparisons.empty
