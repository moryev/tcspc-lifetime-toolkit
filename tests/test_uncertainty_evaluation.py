import numpy as np
import pytest

from tcspc_toolkit.uncertainty_evaluation import (
    PredictionIntervalResult,
    UncertaintyDataProvenance,
    UncertaintyDataRole,
    UncertaintyScoreResult,
    evaluate_prediction_intervals,
    evaluate_selective_prediction,
    evaluate_uncertainty_scores,
    make_uncertainty_development_split,
)


def test_uncertainty_development_split_is_reproducible_and_disjoint() -> None:
    first = make_uncertainty_development_split(
        40,
        calibration_fraction=0.25,
        random_seed=57,
    )
    second = make_uncertainty_development_split(
        40,
        calibration_fraction=0.25,
        random_seed=57,
    )

    assert first == second
    assert set(first.training_indices).isdisjoint(
        first.calibration_indices
    )
    assert (
        set(first.training_indices)
        | set(first.calibration_indices)
    ) == set(range(40))
    assert len(first.calibration_indices) == 10


@pytest.mark.parametrize("test_id", ["A", "B", "C", "D", "E", "F"])
@pytest.mark.parametrize(
    "role",
    [
        UncertaintyDataRole.TRAINING,
        UncertaintyDataRole.CALIBRATION,
    ],
)
def test_frozen_generalization_tests_cannot_enter_uncertainty_development(
    test_id: str,
    role: UncertaintyDataRole,
) -> None:
    with pytest.raises(
        ValueError,
        match="external uncertainty evaluation",
    ):
        UncertaintyDataProvenance(
            source_id=test_id,
            role=role,
        )


def test_frozen_generalization_tests_are_allowed_for_external_evaluation() -> None:
    records = [
        UncertaintyDataProvenance(
            source_id=test_id,
            role=UncertaintyDataRole.EXTERNAL_EVALUATION,
        )
        for test_id in ("A", "B", "C", "D", "E", "F")
    ]

    assert len(records) == 6


def test_interval_metrics_report_coverage_width_score_and_failures() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0])
    intervals = PredictionIntervalResult(
        prediction=np.array([1.0, 2.2, 2.8, 4.1]),
        lower=np.array([0.8, 1.8, 3.1, np.nan]),
        upper=np.array([1.2, 2.4, 3.5, np.nan]),
        nominal_coverage=0.90,
        method_id="test_interval",
    )

    metrics = evaluate_prediction_intervals(
        true,
        intervals,
    )

    assert metrics.n_valid_intervals == 3
    assert metrics.interval_failure_rate == pytest.approx(0.25)
    assert metrics.empirical_coverage == pytest.approx(2 / 3)
    assert metrics.coverage_error == pytest.approx((2 / 3) - 0.90)
    assert metrics.mean_interval_width == pytest.approx(
        (0.4 + 0.6 + 0.4) / 3
    )
    assert metrics.median_interval_width == pytest.approx(0.4)
    assert metrics.mean_interval_score > metrics.mean_interval_width


def test_uncertainty_score_metrics_reward_scores_that_track_error() -> None:
    true = np.full(10, 2.0)
    absolute_error = np.linspace(0.01, 0.50, 10)
    scores = UncertaintyScoreResult(
        prediction=true + absolute_error,
        uncertainty_score=absolute_error.copy(),
        method_id="synthetic_score",
    )

    metrics = evaluate_uncertainty_scores(
        true,
        scores,
        tail_fraction=0.20,
    )

    assert metrics.spearman_error_correlation == pytest.approx(1.0)
    assert (
        metrics.high_uncertainty_mae_ns
        > metrics.low_uncertainty_mae_ns
    )


def test_selective_prediction_improves_mae_when_score_tracks_error() -> None:
    true = np.full(10, 2.0)
    absolute_error = np.linspace(0.01, 0.50, 10)
    scores = UncertaintyScoreResult(
        prediction=true + absolute_error,
        uncertainty_score=absolute_error.copy(),
        method_id="synthetic_score",
    )

    metrics = evaluate_selective_prediction(
        true,
        scores,
        rejection_fraction=0.20,
    )

    assert metrics.n_retained == 8
    assert metrics.mae_retained_ns < metrics.mae_all_ns
    assert metrics.mae_improvement_ns > 0.0
