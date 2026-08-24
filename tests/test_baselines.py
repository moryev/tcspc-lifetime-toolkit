"""Tests for TCSPC lifetime baseline estimators."""

import numpy as np
import pytest

from tcspc_toolkit.baselines import (
    estimate_lifetime_from_mean_arrival,
    predict_constant_mean_baseline,
)


def _mean_and_peak_time(
    time: np.ndarray,
    counts: np.ndarray,
) -> tuple[float, float]:
    """Calculate mean arrival time and observed peak time."""
    mean_arrival_time = float(
        np.sum(time * counts)
        / np.sum(counts)
    )

    peak_time = float(
        time[np.argmax(counts)]
    )

    return (
        mean_arrival_time,
        peak_time,
    )


def test_constant_mean_baseline_predicts_training_mean() -> None:
    y_train = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    predictions = predict_constant_mean_baseline(
        y_train=y_train,
        n_predictions=3,
    )

    expected_mean = np.mean(
        y_train
    )

    np.testing.assert_allclose(
        predictions,
        expected_mean,
    )


def test_constant_mean_baseline_has_expected_shape() -> None:
    y_train = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    predictions = predict_constant_mean_baseline(
        y_train=y_train,
        n_predictions=5,
    )

    assert predictions.shape == (
        5,
    )


def test_constant_mean_baseline_outputs_are_finite() -> None:
    y_train = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    predictions = predict_constant_mean_baseline(
        y_train=y_train,
        n_predictions=4,
    )

    assert np.all(
        np.isfinite(predictions)
    )


def test_mean_arrival_estimator_recovers_ideal_exponential_lifetime() -> None:
    lifetime_ns = 2.5

    time = np.linspace(
        0.0,
        30.0,
        30_001,
        dtype=np.float64,
    )

    counts = np.exp(
        -time / lifetime_ns
    )

    (
        mean_arrival_time_ns,
        peak_time_ns,
    ) = _mean_and_peak_time(
        time,
        counts,
    )

    prediction = (
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=np.array(
                [mean_arrival_time_ns],
                dtype=np.float64,
            ),
            peak_time_ns=np.array(
                [peak_time_ns],
                dtype=np.float64,
            ),
        )
    )

    assert prediction[0] == pytest.approx(
        lifetime_ns,
        abs=0.01,
    )


def test_mean_arrival_estimator_is_invariant_to_count_scaling() -> None:
    lifetime_ns = 2.0

    time = np.linspace(
        0.0,
        25.0,
        25_001,
        dtype=np.float64,
    )

    counts = np.exp(
        -time / lifetime_ns
    )

    scaled_counts = (
        1000.0 * counts
    )

    (
        mean_arrival_original,
        peak_time_original,
    ) = _mean_and_peak_time(
        time,
        counts,
    )

    (
        mean_arrival_scaled,
        peak_time_scaled,
    ) = _mean_and_peak_time(
        time,
        scaled_counts,
    )

    original_prediction = (
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=np.array(
                [mean_arrival_original],
                dtype=np.float64,
            ),
            peak_time_ns=np.array(
                [peak_time_original],
                dtype=np.float64,
            ),
        )
    )

    scaled_prediction = (
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=np.array(
                [mean_arrival_scaled],
                dtype=np.float64,
            ),
            peak_time_ns=np.array(
                [peak_time_scaled],
                dtype=np.float64,
            ),
        )
    )

    np.testing.assert_allclose(
        original_prediction,
        scaled_prediction,
    )


def test_mean_arrival_estimator_has_expected_shape() -> None:
    mean_arrival_time_ns = np.array(
        [2.0, 3.5, 5.0],
        dtype=np.float64,
    )

    peak_time_ns = np.array(
        [1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    predictions = (
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=mean_arrival_time_ns,
            peak_time_ns=peak_time_ns,
        )
    )

    assert predictions.shape == (
        3,
    )


def test_mean_arrival_estimator_outputs_are_finite() -> None:
    mean_arrival_time_ns = np.array(
        [2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    peak_time_ns = np.array(
        [1.0, 1.0, 1.0],
        dtype=np.float64,
    )

    predictions = (
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=mean_arrival_time_ns,
            peak_time_ns=peak_time_ns,
        )
    )

    assert np.all(
        np.isfinite(predictions)
    )


def test_mean_arrival_estimator_preserves_nonpositive_estimates() -> None:
    mean_arrival_time_ns = np.array(
        [1.0, 0.5],
        dtype=np.float64,
    )

    peak_time_ns = np.array(
        [1.0, 1.0],
        dtype=np.float64,
    )

    predictions = (
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=mean_arrival_time_ns,
            peak_time_ns=peak_time_ns,
        )
    )

    np.testing.assert_allclose(
        predictions,
        np.array(
            [0.0, -0.5],
            dtype=np.float64,
        ),
    )


def test_mean_arrival_estimator_rejects_mismatched_shapes() -> None:
    mean_arrival_time_ns = np.array(
        [2.0, 3.0],
        dtype=np.float64,
    )

    peak_time_ns = np.array(
        [1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=mean_arrival_time_ns,
            peak_time_ns=peak_time_ns,
        )


def test_mean_arrival_estimator_rejects_non_finite_inputs() -> None:
    mean_arrival_time_ns = np.array(
        [2.0, np.nan],
        dtype=np.float64,
    )

    peak_time_ns = np.array(
        [1.0, 1.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        estimate_lifetime_from_mean_arrival(
            mean_arrival_time_ns=mean_arrival_time_ns,
            peak_time_ns=peak_time_ns,
        )
