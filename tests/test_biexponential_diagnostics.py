"""Descriptive bi-exponential degeneracy and conditioning, not certification."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import multiexponential as fitting
from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitSettings,
    BiexponentialParameters,
)
from test_biexponential_reconvolution import BOUNDS, independent_case


def test_zero_observation_analytical_solution_and_undefined_quantities():
    t, k, _, _, _ = independent_case(bins=32)
    result = fitting.fit_biexponential_reconvolution(
        t, np.zeros(32, dtype=int), k, **BOUNDS
    )
    assert result.parameters.amplitudes == (0, 0)
    assert result.parameters.background == 0
    assert np.count_nonzero(result.expected_counts) == 0
    assert result.optimization.numerical_validation_passed
    assert result.optimization.optimizer_reported_success is None
    assert result.optimization.stopping_reason == "analytical_zero_counts"
    assert result.components.amplitude_fractions is None
    assert result.components.detected_signal_fractions is None
    assert result.components.ideal_photon_fractions is None
    assert result.components.amplitude_weighted_lifetime_ns is None
    assert result.statistics.log_likelihood == 0
    assert result.statistics.aic == 12
    assert result.statistics.bic_bins == pytest.approx(6 * np.log(32))
    assert result.identifiability.weighted_jacobian_rank is None
    assert (
        "no_detected_signal_fractions_undefined"
        in result.identifiability.structural_warnings
    )


@pytest.mark.parametrize(
    "lifetimes,fractions,expected",
    [
        ((1.5, 1.5), (0.4, 0.6), "equal_lifetimes_amplitude_split_unconstrained"),
        ((0.8, 3.2), (1, 0), "zero_component_lifetime_unconstrained"),
    ],
)
def test_exact_degeneracies_preserved(lifetimes, fractions, expected):
    t, k, mu, p, _ = independent_case(
        bins=128, lifetimes=lifetimes, fractions=fractions
    )
    settings = BiexponentialFitSettings(policy="economical", max_starts=1)
    # Exact structural warnings require an exact optimum of the evaluated
    # model. Independent direct-convolution targets differ from the shared FFT
    # by roundoff and can favor a nonzero 1e-14 amplitude / unequal-ulp lifetimes.
    problem = fitting._Problem(t, mu, k, (0.1, 8), (-0.3, 0.3), settings)
    mu, _ = problem.model(problem.physical(problem.coordinates(p)), "diagnostics")
    result = fitting._fit_core(
        t, mu, k, **BOUNDS, initial_guesses=[p], settings=settings, raw_counts=False
    )
    assert expected in result.identifiability.structural_warnings
    assert result.optimization.numerical_validation_passed
    assert result.identifiability.weighted_jacobian_rank < 6
    assert np.isinf(result.identifiability.weighted_jacobian_condition)


def test_distinct_fraction_definitions_and_averages():
    t, k, mu, p, _ = independent_case(bins=128, window=6)
    result = fitting._fit_core(
        t,
        mu,
        k,
        **BOUNDS,
        initial_guesses=[p],
        settings=BiexponentialFitSettings(policy="economical", max_starts=1),
        raw_counts=False,
    )
    summary = result.components
    a, tau = (
        np.array(result.parameters.amplitudes),
        np.array(result.parameters.lifetimes_ns),
    )
    np.testing.assert_allclose(summary.amplitude_fractions, a / a.sum())
    np.testing.assert_allclose(summary.ideal_photon_fractions, a * tau / (a @ tau))
    np.testing.assert_allclose(summary.detected_signal_fractions, (0.4, 0.6), atol=1e-6)
    assert summary.amplitude_weighted_lifetime_ns == pytest.approx(a @ tau / a.sum())
    assert summary.ideal_photon_weighted_lifetime_ns == pytest.approx(
        a @ (tau**2) / (a @ tau)
    )
    assert summary.detected_signal_weighted_lifetime_ns == pytest.approx(
        np.dot(summary.detected_signal_fractions, tau)
    )
    assert not np.allclose(
        summary.detected_signal_fractions, summary.ideal_photon_fractions
    )


@pytest.mark.parametrize(
    "lifetimes,fractions,photons",
    [
        ((1.5, 1.6), (0.4, 0.6), 10000),
        ((0.8, 3.2), (0.995, 0.005), 100000),
        ((0.8, 3.2), (0.4, 0.6), 100),
    ],
)
def test_seeded_difficult_cases_report_without_recovery_certificate(
    lifetimes, fractions, photons
):
    t, k, mu, _, _ = independent_case(
        bins=128, lifetimes=lifetimes, fractions=fractions, photons=photons
    )
    result = fitting.fit_biexponential_reconvolution(
        t,
        np.random.default_rng(2802101).poisson(mu),
        k,
        **BOUNDS,
        settings=BiexponentialFitSettings(policy="economical"),
    )
    assert result.parameters is not None
    assert np.all(np.isfinite(result.expected_counts))
    assert min(result.parameters.amplitudes) >= 0
    assert result.parameters.lifetimes_ns[0] <= result.parameters.lifetimes_ns[1]
    assert not hasattr(result.identifiability, "identifiable")
    assert (
        result.optimization.objective_evaluations
        <= result.assumptions.settings.max_objective_evaluations
    )


def test_warning_thresholds_are_descriptive_not_acceptance_criteria():
    t, k, mu, p, _ = independent_case(bins=64)
    settings = BiexponentialFitSettings(
        policy="economical",
        max_starts=1,
        condition_limit=1,
        weak_fraction=0.49,
        similar_lifetime_ratio=10,
    )
    result = fitting._fit_core(
        t, mu, k, **BOUNDS, initial_guesses=[p], settings=settings, raw_counts=False
    )
    assert result.optimization.numerical_validation_passed
    assert {
        "ill_conditioned_weighted_jacobian",
        "weak_detected_component",
        "similar_lifetimes",
    } <= set(result.identifiability.conditioning_warnings)


def test_equal_lifetime_ties_use_amplitude_secondary_order():
    canonical, order = fitting._canonical(BiexponentialParameters((1, 3), (2, 2)))
    assert canonical.amplitudes == (3, 1)
    assert tuple(order) == (1, 0)


def test_shift_knot_does_not_get_a_spurious_smooth_jacobian():
    t, k, mu, p, _ = independent_case(bins=64, shift=0.0)
    result = fitting._fit_core(
        t,
        mu,
        k,
        **BOUNDS,
        initial_guesses=[p],
        settings=BiexponentialFitSettings(policy="economical", max_starts=1),
        raw_counts=False,
    )
    assert (
        result.identifiability.jacobian_unavailable_reason == "shift_interpolation_knot"
    )
    assert result.identifiability.weighted_jacobian_rank is None


def test_near_optimal_disagreement_and_diagnostic_budget():
    t, k, mu, p, _ = independent_case(bins=64, fractions=(1, 0))
    settings = BiexponentialFitSettings(max_diagnostic_evaluations=27)
    problem = fitting._Problem(t, mu, k, (0.1, 8), (-0.3, 0.3), settings)
    problem.best_value = 0
    starts = []
    for i, tau in enumerate((2.0, 7.0)):
        point = replace(p, lifetimes_ns=(0.8, tau))
        starts.append(
            fitting.BiexponentialStartDiagnostics(
                i, "test", point, point, 0, 0, True, 0, "test", 1, 0
            )
        )
    _, components = problem.model(p, "initialization")
    summary = fitting._component_summary(p, components)
    diagnostics = fitting._identifiability(problem, p, summary, starts)
    assert "near_optimal_lifetime_disagreement" in diagnostics.multistart_warnings
    assert (
        "near_optimal_contributions_budget_exhausted" in diagnostics.multistart_warnings
    )
    assert diagnostics.near_optimal_detected_fraction_ranges is None
    assert problem.counts["diagnostics"] == 27
