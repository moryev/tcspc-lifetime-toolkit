"""Deterministic checks for Stage-3 operational sampling diagnostics."""

import math

import numpy as np
import pytest

from tcspc_toolkit.bayesian_sampling import (
    BayesianSamplingConfig,
    _assess_sampling_diagnostics,
    _posterior_correlation,
)


def _config(**overrides) -> BayesianSamplingConfig:
    values = dict(
        random_seed=9, n_walkers=8, warmup_steps=10, production_steps=100,
        max_production_steps=100, extension_steps=10, n_ensembles=2,
        min_autocorrelation_multiples=10.0,
        max_autocorrelation_relative_change=0.1,
        min_effective_samples=100.0,
        max_ensemble_location_difference_sd=0.5,
    )
    values.update(overrides)
    return BayesianSamplingConfig(**values)


def _chains(n_steps: int = 100):
    transformed = np.random.default_rng(128).normal(
        size=(2, n_steps, 8, 3)
    )
    physical = np.empty((2, n_steps, 8, 4))
    physical[..., :3] = np.exp(transformed)
    physical[..., 3] = 0.0
    acceptance = np.full((2, 8), 0.45)
    return transformed, physical, acceptance


def test_healthy_fake_chains_pass_explicit_operational_policy() -> None:
    transformed, physical, acceptance = _chains()
    calls = 0

    def stable_tau(chain):
        nonlocal calls
        calls += 1
        return np.full(3, 5.0 if chain.shape[0] == 100 else 4.8)

    diagnostics = _assess_sampling_diagnostics(
        transformed, physical, acceptance, _config(), extension_count=1,
        autocorrelation_estimator=stable_tau,
    )
    assert diagnostics.accepted
    assert diagnostics.failure_reasons == ()
    assert diagnostics.production_steps == 100
    assert diagnostics.retained_samples == 1600
    assert diagnostics.extension_count == 1
    assert diagnostics.mean_acceptance_fraction == pytest.approx(0.45)
    np.testing.assert_allclose(diagnostics.autocorrelation_time_steps, 5.0)
    np.testing.assert_allclose(diagnostics.autocorrelation_relative_change, 0.04)
    np.testing.assert_allclose(diagnostics.approximate_effective_samples, 320.0)
    assert calls == 4


def test_too_short_chains_fail_length_check_even_with_tau_estimate() -> None:
    transformed, physical, acceptance = _chains(n_steps=12)
    diagnostics = _assess_sampling_diagnostics(
        transformed, physical, acceptance, _config(), extension_count=0,
        autocorrelation_estimator=lambda chain: np.full(3, 5.0),
    )
    assert not diagnostics.accepted
    assert "production shorter than configured autocorrelation multiple" in (
        diagnostics.failure_reasons
    )
    assert np.all(np.isfinite(diagnostics.autocorrelation_time_steps))


def test_unavailable_autocorrelation_is_reported_not_accepted() -> None:
    transformed, physical, acceptance = _chains()
    diagnostics = _assess_sampling_diagnostics(
        transformed, physical, acceptance, _config(), extension_count=0,
        autocorrelation_estimator=lambda chain: np.full(3, np.nan),
    )
    assert not diagnostics.accepted
    assert "autocorrelation estimate unavailable" in diagnostics.failure_reasons
    assert np.all(np.isnan(diagnostics.approximate_effective_samples))


def test_unstable_tau_and_ensemble_disagreement_fail_separately() -> None:
    transformed, physical, acceptance = _chains()
    physical[1, ..., 0] += 10.0
    diagnostics = _assess_sampling_diagnostics(
        transformed, physical, acceptance, _config(), extension_count=0,
        autocorrelation_estimator=lambda chain: np.full(
            3, 5.0 if chain.shape[0] == 100 else 2.0
        ),
    )
    assert "autocorrelation estimate unstable across chain halves" in (
        diagnostics.failure_reasons
    )
    assert "independent ensemble locations disagree" in diagnostics.failure_reasons
    assert diagnostics.maximum_ensemble_mean_difference_sd > 0.5


def test_bad_acceptance_and_empty_production_are_explicit() -> None:
    transformed, physical, _ = _chains(n_steps=0)
    acceptance = np.full((2, 8), np.nan)
    diagnostics = _assess_sampling_diagnostics(
        transformed, physical, acceptance, _config(), extension_count=0,
        autocorrelation_estimator=lambda chain: np.full(3, 5.0),
    )
    assert "no production samples" in diagnostics.failure_reasons
    assert "mean acceptance fraction outside configured bounds" in (
        diagnostics.failure_reasons
    )
    assert diagnostics.retained_samples == 0
    assert math.isinf(diagnostics.maximum_ensemble_mean_difference_sd)


def test_degenerate_correlation_rows_are_nan_not_false_identifiability() -> None:
    draws = np.ones((2, 10, 8, 3))
    draws[..., 0] = np.random.default_rng(5).normal(size=(2, 10, 8))
    correlation = _posterior_correlation(draws)
    assert correlation[0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(correlation[1:, :]))
    assert np.all(np.isnan(correlation[:, 1:]))
