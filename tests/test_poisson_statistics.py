"""Independent numerical checks for the new, module-qualified Poisson primitives."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from math import factorial

import numpy as np
import pytest
from scipy.stats import poisson

from tcspc_toolkit.estimation.fitting import poisson_negative_log_likelihood
from tcspc_toolkit.evaluation.diagnostics import (
    PoissonFitStatistics,
    calculate_poisson_deviance,
    calculate_poisson_deviance_residuals,
    calculate_poisson_fit_statistics,
)


def _decimal_deviance(observed, expected):
    # Independent high-precision definition, not the production series.
    with localcontext() as context:
        context.prec = 100
        y, mu = Decimal.from_float(float(observed)), Decimal.from_float(float(expected))
        if y == 0:
            return float(2 * mu)
        if mu == 0:
            return np.inf
        return float(2 * (mu - y + y * (y / mu).ln()))


@pytest.mark.parametrize(
    "observed,expected,deviance",
    [
        ([0], [0], 0.0),
        ([0], [2], 4.0),
        ([1], [0], np.inf),
        ([0, 1, 0], [0, 0, 5], np.inf),
        ([0, 10, 30], [0, 10, 30], 0.0),
    ],
)
def test_exact_zero_limits(observed, expected, deviance):
    with np.errstate(all="raise"):
        assert calculate_poisson_deviance(observed, expected) == deviance
        stats = calculate_poisson_fit_statistics(observed, expected, n_parameters=0)
    assert stats.deviance == deviance
    assert not np.any(np.isnan(stats.deviance_residuals))
    if np.isinf(deviance):
        assert stats.log_likelihood == -np.inf
        assert stats.aic == np.inf
        assert stats.bic_bins == np.inf
        assert np.isposinf(stats.deviance_residuals[1 if len(observed) > 1 else 0])


@pytest.mark.parametrize("y", [1.0, 1e6, 1e12, 1e15])
@pytest.mark.parametrize(
    "relative_offset",
    [-0.51, -0.49, -0.0101, -0.0099, -1e-8, 0.0, 1e-8, 0.0099, 0.0101, 0.49, 0.51],
)
def test_deviance_matches_decimal_near_and_across_branch_boundaries(y, relative_offset):
    mu = y / (1.0 + relative_offset)
    expected = _decimal_deviance(y, mu)
    actual = calculate_poisson_deviance([y], [mu])
    assert actual == pytest.approx(expected, rel=2e-12, abs=0.0)


@pytest.mark.parametrize("y", [1e-100, 1.0, 1e12, 1e300])
@pytest.mark.parametrize("direction", [0.0, np.inf])
def test_adjacent_floats_have_positive_precise_deviance(y, direction):
    mu = np.nextafter(y, direction)
    expected = _decimal_deviance(y, mu)
    actual = calculate_poisson_deviance([y], [mu])
    assert actual > 0.0
    assert actual == pytest.approx(expected, rel=2e-15, abs=0.0)


@pytest.mark.parametrize(
    "y,mu",
    [
        (1e-300, 1e300),
        (1e300, 1e-300),
        (1.0, np.nextafter(0.0, 1.0)),
        (1e-300, 1e-310),
        (1e300, 1e299),
        (1e308, 1e307),
        (1e-300, 1e308),
        (1e308, 1e-300),
    ],
)
def test_extreme_finite_ratios_do_not_create_spurious_nan(y, mu):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = calculate_poisson_deviance([y], [mu])
    expected = _decimal_deviance(y, mu)
    assert actual == pytest.approx(expected, rel=2e-12, abs=0.0)


def test_fractional_targets_only_have_deterministic_deviance_semantics():
    observed, expected = [0.1, 1.25, 3.5], [0.2, 1.1, 4.0]
    reference = sum(_decimal_deviance(y, mu) for y, mu in zip(observed, expected))
    assert calculate_poisson_deviance(observed, expected) == pytest.approx(reference)
    with pytest.raises(ValueError, match="integer-valued"):
        calculate_poisson_fit_statistics(observed, expected, n_parameters=4)


@pytest.mark.parametrize(
    "parameter_type",
    [
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_seeded_full_statistics_match_independent_poisson_probabilities(parameter_type):
    expected = np.geomspace(0.01, 10000, 80)
    observed = np.random.default_rng(28001).poisson(expected)
    observed_before, expected_before = observed.copy(), expected.copy()
    stats = calculate_poisson_fit_statistics(
        observed, expected, n_parameters=parameter_type(6)
    )
    log_likelihood = float(poisson.logpmf(observed, expected).sum())
    assert isinstance(stats, PoissonFitStatistics)
    assert stats.n_bins == 80
    assert stats.n_parameters == 6
    assert type(stats.n_parameters) is int
    assert stats.log_likelihood == pytest.approx(log_likelihood, abs=1e-9)
    assert stats.negative_log_likelihood == poisson_negative_log_likelihood(
        observed, expected
    )
    assert stats.centered_negative_log_likelihood == stats.deviance / 2.0
    assert stats.deviance == calculate_poisson_deviance(observed, expected)
    assert stats.deviance == pytest.approx(
        np.sum(stats.deviance_residuals**2), rel=1e-14
    )
    np.testing.assert_array_equal(
        np.sign(stats.deviance_residuals), np.sign(observed - expected)
    )
    assert stats.aic == pytest.approx(12.0 - 2.0 * log_likelihood, abs=2e-9)
    assert stats.bic_bins == pytest.approx(
        6.0 * np.log(80) - 2.0 * log_likelihood, abs=2e-9
    )
    np.testing.assert_array_equal(observed, observed_before)
    np.testing.assert_array_equal(expected, expected_before)
    with pytest.raises(ValueError):
        stats.deviance_residuals[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        stats.n_bins = 1


@pytest.mark.parametrize("count", [0, 1, 2, 15, 16, 17, 30, 60, 1000])
def test_log_factorial_normalization_against_decimal(count):
    with localcontext() as context:
        context.prec = 70
        y = Decimal(count)
        log_likelihood = (
            float(y * y.ln() - y - Decimal(factorial(count)).ln()) if count else 0.0
        )
    stats = calculate_poisson_fit_statistics([count], [count], n_parameters=0)
    assert stats.log_likelihood == pytest.approx(log_likelihood, rel=0.0, abs=2e-14)
    assert stats.deviance == 0.0


def test_high_count_full_likelihood_avoids_subtracting_large_nll_constants():
    # High-precision value for log(P(Y=10**12 | mu=10**12)).
    stats = calculate_poisson_fit_statistics([10**12], [10**12], n_parameters=0)
    assert stats.log_likelihood == pytest.approx(
        -14.7344490911690301792216117978, abs=5e-15
    )


@pytest.mark.parametrize(
    "parameter_type",
    [
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_parameter_penalties_do_not_overflow_numpy_integers(parameter_type):
    with np.errstate(over="raise"):
        stats = calculate_poisson_fit_statistics(
            np.zeros(200, dtype=np.int64),
            np.zeros(200),
            n_parameters=parameter_type(64),
        )
    assert stats.n_parameters == 64
    assert type(stats.n_parameters) is int
    assert stats.log_likelihood == 0.0
    assert stats.aic == 128.0
    assert stats.bic_bins == pytest.approx(64.0 * np.log(200))


def test_parameter_penalties_use_all_bins_and_nominal_parameter_count():
    observed, expected = [0, 0, 4, 8, 2], [0, 0, 3, 7, 1]
    mono = calculate_poisson_fit_statistics(observed, expected, n_parameters=4)
    bi = calculate_poisson_fit_statistics(observed, expected, n_parameters=6)
    assert bi.aic - mono.aic == pytest.approx(4.0)
    assert bi.bic_bins - mono.bic_bins == pytest.approx(2.0 * np.log(5))
    assert mono.log_likelihood == bi.log_likelihood
    # Descriptive statistics remain defined for n_bins <= n_parameters;
    # this does not imply that a six-parameter fit is identifiable here.
    assert bi.n_parameters > bi.n_bins


def test_historical_likelihood_and_residual_contracts_remain_distinct():
    observed = np.array([0, 1, 3, 15])
    expected = np.array([2.0, 3.0, 2.0, 12.0])
    historical = calculate_poisson_deviance_residuals(observed, expected)
    stats = calculate_poisson_fit_statistics(observed, expected, n_parameters=4)
    np.testing.assert_allclose(stats.deviance_residuals, historical, rtol=1e-14)
    saturated_nll = poisson_negative_log_likelihood(observed, observed)
    assert stats.centered_negative_log_likelihood == pytest.approx(
        stats.negative_log_likelihood - saturated_nll
    )
    # The old residual helper still rejects zero expectations; the new
    # statistics implement the mathematical zero limits without altering it.
    with pytest.raises(ValueError, match="positive"):
        calculate_poisson_deviance_residuals(np.array([0]), np.array([0.0]))
    assert poisson_negative_log_likelihood([1.5], [2.5]) == pytest.approx(
        2.5 - 1.5 * np.log(2.5)
    )


@pytest.mark.parametrize(
    "observed,expected",
    [
        ([], []),
        ([[1]], [[1]]),
        ([1], [1, 2]),
        ([np.nan], [1]),
        ([1], [np.inf]),
        ([np.inf], [1]),
        ([1], [np.nan]),
        ([-1], [1]),
        ([1], [-1]),
    ],
)
def test_invalid_pairs_are_rejected(observed, expected):
    with pytest.raises(ValueError):
        calculate_poisson_deviance(observed, expected)
    with pytest.raises(ValueError):
        calculate_poisson_fit_statistics(observed, expected, n_parameters=4)


@pytest.mark.parametrize(
    "n_parameters",
    [-1, np.int8(-1), np.int64(-1), 2.5, 4.0, True, np.bool_(False), None, np.nan],
)
def test_parameter_count_must_be_nonnegative_integer(n_parameters):
    with pytest.raises(ValueError, match="non-negative integer"):
        calculate_poisson_fit_statistics([1], [1], n_parameters=n_parameters)


@pytest.mark.parametrize("counts", [[1.0000000001], [2**53], [2**53 + 1]])
def test_fractional_or_inexact_float64_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        calculate_poisson_fit_statistics(counts, [1.0], n_parameters=0)
