"""Independent exact-likelihood and singular-boundary comparison checks."""

import math

import numpy as np
import pytest

from tcspc_toolkit.evaluation.reconvolution import compare_reconvolution_statistics


def test_full_likelihood_criteria_and_signs_independently():
    y = np.array([0, 1, 3, 8, 0, 2])
    m = np.array([0.2, 1.2, 2, 6, 0.5, 3])
    b = np.array([0, 1, 3.2, 8.1, 0, 2])
    result = compare_reconvolution_statistics(y, m, b)
    likelihoods = []
    for expected, parameters, record in [(m, 4, result.mono), (b, 6, result.bi)]:
        ll = sum(
            (count * math.log(mu) if count else 0) - mu - math.lgamma(count + 1)
            for count, mu in zip(y, expected, strict=True)
        )
        likelihoods.append(ll)
        assert record.log_likelihood == pytest.approx(ll)
        assert record.aic == pytest.approx(2 * parameters - 2 * ll)
        assert record.bic_bins == pytest.approx(parameters * math.log(len(y)) - 2 * ll)
    assert result.delta_log_likelihood == pytest.approx(likelihoods[1] - likelihoods[0])
    assert result.delta_aic == pytest.approx(4 - 2 * result.delta_log_likelihood)
    assert result.delta_bic_bins == pytest.approx(
        2 * math.log(len(y)) - 2 * result.delta_log_likelihood
    )
    assert result.delta_deviance == pytest.approx(-2 * result.delta_log_likelihood)


def test_zero_signal_keeps_nominal_parameter_counts():
    result = compare_reconvolution_statistics([0] * 10, [0] * 10, [0] * 10)
    assert result.delta_aic == 4
    assert result.delta_bic_bins == pytest.approx(2 * math.log(10))
    assert result.mono.deviance == result.bi.deviance == 0


@pytest.mark.parametrize("mono,bi", [(None, [1, 2]), ([1, 2], None), (None, None)])
def test_unavailable_curves_keep_reason(mono, bi):
    result = compare_reconvolution_statistics([1, 2], mono, bi)
    assert result.delta_aic is None
    assert result.unavailable_reason


def test_impossible_expectations_remain_infinite_without_floor():
    result = compare_reconvolution_statistics([1, 0], [0, 0], [1, 0])
    assert math.isinf(result.mono.deviance)
    assert result.delta_deviance is None
    assert result.unavailable_reason


@pytest.mark.parametrize(
    "y,m,b",
    [
        ([0.5, 1], None, None),
        ([1, 2], [1], None),
        ([1, 2], [1, -1], [1, 2]),
        ([1, 2], [1, np.nan], None),
    ],
)
def test_invalid_arrays_rejected(y, m, b):
    with pytest.raises(ValueError):
        compare_reconvolution_statistics(y, m, b)
