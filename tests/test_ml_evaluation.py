import numpy as np
import pytest

from tcspc_toolkit.ml_evaluation import (
    evaluate_regression,
)


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
