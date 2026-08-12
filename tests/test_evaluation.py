import numpy as np
import pytest

from tcspc_toolkit.evaluation import (
    calculate_poisson_deviance_residuals,
)


def test_poisson_deviance_residuals_preserve_shape() -> None:
    observed = np.array([10, 8, 4, 1], dtype=np.int64)
    expected = np.array([9.0, 7.5, 5.0, 1.5])

    residuals = calculate_poisson_deviance_residuals(
        observed=observed,
        expected=expected,
    )

    assert residuals.shape == observed.shape


def test_poisson_deviance_residuals_are_zero_for_perfect_agreement() -> None:
    observed = np.array([1, 5, 10, 50], dtype=np.int64)
    expected = observed.astype(np.float64)

    residuals = calculate_poisson_deviance_residuals(
        observed=observed,
        expected=expected,
    )

    assert np.allclose(residuals, 0.0)


def test_poisson_deviance_residuals_handle_zero_counts() -> None:
    observed = np.array([0], dtype=np.int64)
    expected = np.array([2.0])

    residuals = calculate_poisson_deviance_residuals(
        observed=observed,
        expected=expected,
    )

    assert np.allclose(residuals, [-2.0])


def test_poisson_deviance_residuals_have_correct_sign() -> None:
    observed = np.array([12, 8, 10], dtype=np.int64)
    expected = np.array([10.0, 10.0, 10.0])

    residuals = calculate_poisson_deviance_residuals(
        observed=observed,
        expected=expected,
    )

    assert residuals[0] > 0
    assert residuals[1] < 0
    assert residuals[2] == 0


def test_poisson_deviance_residuals_are_finite() -> None:
    observed = np.array(
        [0, 1, 2, 10, 100, 1000],
        dtype=np.int64,
    )

    expected = np.array(
        [0.5, 1.5, 3.0, 9.0, 110.0, 950.0],
    )

    residuals = calculate_poisson_deviance_residuals(
        observed=observed,
        expected=expected,
    )

    assert np.all(np.isfinite(residuals))


def test_poisson_deviance_residuals_reject_negative_observed_counts() -> None:
    observed = np.array([10, -1, 5], dtype=np.int64)
    expected = np.array([10.0, 2.0, 5.0])

    with pytest.raises(
        ValueError,
        match="observed counts must be non-negative",
    ):
        calculate_poisson_deviance_residuals(
            observed=observed,
            expected=expected,
        )


def test_poisson_deviance_residuals_reject_nonpositive_expected_counts() -> None:
    observed = np.array([10, 0, 5], dtype=np.int64)
    expected = np.array([10.0, 0.0, 5.0])

    with pytest.raises(
        ValueError,
        match="expected counts must be positive",
    ):
        calculate_poisson_deviance_residuals(
            observed=observed,
            expected=expected,
        )


