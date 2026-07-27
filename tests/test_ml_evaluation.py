import numpy as np
import pytest

from tcspc_toolkit.ml_evaluation import (
    evaluate_regression,
    normalize_histograms,
)


def test_normalize_histograms_produces_unit_sums() -> None:
    X = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )

    normalized = normalize_histograms(X)

    assert np.allclose(
        normalized.sum(axis=1),
        1.0,
    )


def test_normalize_histograms_preserves_shape() -> None:
    X = np.ones(
        (10, 256),
        dtype=np.float64,
    )

    normalized = normalize_histograms(X)

    assert normalized.shape == X.shape


def test_normalize_histograms_rejects_zero_sum_curve() -> None:
    X = np.array(
        [
            [1.0, 2.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        normalize_histograms(X)


def test_normalize_histograms_rejects_negative_counts() -> None:
    X = np.array(
        [[1.0, -1.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        normalize_histograms(X)


def test_perfect_predictions_have_zero_error() -> None:
    y_true = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    metrics = evaluate_regression(
        y_true=y_true,
        y_pred=y_true.copy(),
    )

    assert metrics.mae_ns == pytest.approx(0.0)
    assert metrics.median_absolute_error_ns == pytest.approx(0.0)
    assert metrics.mean_relative_error == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
