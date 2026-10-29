"""Issue-33 numerical mechanisms; no new scientific qualification samples."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize

from tcspc_toolkit.estimation import multiexponential as fitting
from test_biexponential_reconvolution import BOUNDS, independent_case


def problem(**settings):
    t, k, mu, p, _ = independent_case(bins=64, photons=1000, background=0.1)
    instance = fitting._Problem(
        t,
        np.rint(mu),
        k,
        (0.1, 8),
        (-0.3, 0.3),
        fitting.BiexponentialFitSettings(**settings),
    )
    return instance, instance.coordinates(p)


@pytest.mark.parametrize("placeholder", [0.0, np.nan])
def test_installed_scipy_invalid_pair_is_not_safe_backtracking(placeholder):
    calls = []

    def unsafe(x):
        if x[0] <= 0:
            calls.append(x.copy())
            return np.inf, np.array([placeholder])
        return 1000 * x[0] - np.log(x[0]), np.array([1000 - 1 / x[0]])

    result = minimize(
        unsafe,
        [0.1],
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, 2)],
        options={"ftol": 1e-12, "gtol": 1e-6, "maxls": 40},
    )
    assert calls
    # This documents the hazard, not a required SciPy failure on every version.
    # If a future SciPy recovers, it must reach the actual optimum.
    assert result.x[0] == pytest.approx(0.1) or result.x[0] == pytest.approx(
        0.001, abs=1e-7
    )


def test_guarded_real_scipy_recovers_with_each_invocation_counted():
    p, x = problem()
    x[0] = 0.1

    def evaluate(point, phase="optimization"):
        if p.counts[phase] >= p.limits[phase]:
            p.exhausted.add(phase)
            raise fitting._BudgetExceeded(phase)
        p.counts[phase] += 1
        value = np.inf if point[0] <= 0 else 1000 * point[0] - np.log(point[0])
        if value < p.local_value:
            p.local_value, p.local_x = value, point.copy()
        return value

    def derivative(point):
        value = evaluate(point)
        if not np.isfinite(value):
            raise fitting._UnusableEvaluation("nonfinite_objective")
        gradient = np.zeros(6)
        gradient[0] = 1000 - 1 / point[0]
        return value, gradient

    p.evaluate, p.value_and_gradient = evaluate, derivative
    starts = []
    while len(starts) < 46:
        starts.append(fitting._refine(p, x, len(starts), "control"))
        if p.recovery_x is None:
            break
        assert starts[-1].optimizer_reported_success is None
        x = p.recovery_x
    assert len(starts) > 1
    assert p.local_x[0] == pytest.approx(0.001, abs=1e-7)
    assert sum(s.objective_evaluations for s in starts) == p.counts["optimization"]


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(6, np.nan)])
def test_nonfinite_cost_cannot_supply_stationary_gradient(bad):
    p, _ = problem()
    with pytest.raises(fitting._UnusableEvaluation, match="nonfinite_objective"):
        p.value_and_gradient(bad)
    assert p.best_x is None


def test_failed_stencil_raises_and_retains_finite_best(monkeypatch):
    p, x = problem()
    original = p.evaluate

    def only_center(point, phase="optimization"):
        value = original(point, phase)
        return value if np.array_equal(point[2:4], x[2:4]) else np.inf

    monkeypatch.setattr(p, "evaluate", only_center)
    with pytest.raises(fitting._UnusableEvaluation, match="finite_difference_failed"):
        p.value_and_gradient(x)
    assert np.isfinite(p.best_value)
    assert p.best_x is not None
    assert p.counts["optimization"] == 17  # center plus eight pairs


def test_one_sided_finite_stencil_when_opposite_side_impossible(monkeypatch):
    p, x = problem()
    original = p.evaluate

    def right_only(point, phase="optimization"):
        value = original(point, phase)
        return np.inf if point[2] < x[2] else value

    monkeypatch.setattr(p, "evaluate", right_only)
    value, derivative = p.value_and_gradient(x)
    assert np.isfinite(value)
    assert np.all(np.isfinite(derivative))
    assert p.counts["optimization"] == 8


def test_invalid_success_endpoint_is_rejected_and_best_survives(monkeypatch):
    p, x = problem()

    def false_success(fun, x0, **kwargs):
        fun(x0)
        return OptimizeResult(x=np.zeros(6), success=True, status=0, message="false")

    monkeypatch.setattr(fitting, "minimize", false_success)
    result = fitting._refine(p, x, 0, "seed")
    assert result.optimizer_reported_success is True  # preserve solver claim
    assert np.isinf(result.endpoint_centered_nll)
    assert "invalid_solver_endpoint" in result.message
    assert result.best_parameters is not None
    assert np.isfinite(result.best_centered_nll)
    assert fitting._local_validation(p, np.zeros(6))[0] is False


@pytest.mark.parametrize("budget", [0, 1, 3, 6])
def test_exhausted_derivative_budget_is_exact_and_honest(budget):
    p, x = problem(max_objective_evaluations=budget)
    result = fitting._refine(p, x, 0, "seed")
    assert result.objective_evaluations == budget
    assert result.optimizer_reported_success is None
    assert "optimization" in p.exhausted
    assert (result.best_parameters is not None) == (budget > 0)


def test_zero_background_zero_counts_and_small_expectations_have_exact_score():
    p, x = problem()
    x[4] = 0
    mu, _ = p.model(p.physical(x), "diagnostics")
    p.targets = np.zeros_like(mu)
    value, derivative = p.value_and_gradient(x)
    assert value == pytest.approx(mu.sum())
    assert derivative[4] == p.b_scale * mu.size
    x[4] = 1e-16 / p.b_scale
    p.targets[0] = 1
    value, derivative = p.value_and_gradient(x)
    curve, _ = p.model(p.physical(x), "diagnostics")
    assert np.isfinite(value)
    assert derivative[4] == pytest.approx(
        p.b_scale * (mu.size - 1 / curve[0]), rel=1e-14
    )
    # Adding exactly zero components and retaining equal lifetimes is admissible.
    x[:2] = 0
    x[2:4] = 0
    value, derivative = p.value_and_gradient(x)
    assert np.isfinite(value)
    assert np.all(np.isfinite(derivative))


def test_start_cap_reports_unavailable_recovery_even_with_complete_bank(monkeypatch):
    t, k, mu, truth, _ = independent_case(bins=64)
    wrong = replace(truth, lifetimes_ns=(0.5, 2), background=10)
    monkeypatch.setattr(fitting, "_initial_points", lambda p, _: [p.coordinates(wrong)])

    def stopped(fun, x, **kwargs):
        fun(x)
        return OptimizeResult(x=x, success=True, status=0, message="premature")

    monkeypatch.setattr(fitting, "minimize", stopped)
    result = fitting.fit_biexponential_reconvolution(
        t,
        np.rint(mu),
        k,
        **BOUNDS,
        settings=fitting.BiexponentialFitSettings(max_starts=1),
    )
    opt = result.optimization
    assert opt.numerical_validation_passed is False
    assert opt.max_detectable_descent_nll > 0.01
    assert opt.recovery_unavailable_reason == "start_budget_exhausted"
    assert opt.stopping_reason == "recovery_start_budget_exhausted"
    assert "starts" in opt.exhausted_budgets
    assert len(opt.starts) == 1
    assert opt.objective_evaluations == sum(s.objective_evaluations for s in opt.starts)


def test_safeguard_exhaustion_keeps_best_and_does_not_retry(monkeypatch):
    p, x = problem(max_objective_evaluations=8)

    def reject_trial(fun, x0, **kwargs):
        fun(x0)
        fun(np.zeros(6))
        pytest.fail("An invalid callback must unwind the solver")

    monkeypatch.setattr(fitting, "minimize", reject_trial)
    result = fitting._refine(p, x, 0, "seed")
    assert result.objective_evaluations == 8
    assert "optimization_budget_exhausted_during_safeguard" in result.message
    assert result.best_parameters is not None
    assert p.recovery_x is None


def test_incomplete_validation_reports_unavailable_recovery(monkeypatch):
    t, k, mu, truth, _ = independent_case(bins=64)
    monkeypatch.setattr(fitting, "_initial_points", lambda p, _: [p.coordinates(truth)])
    result = fitting.fit_biexponential_reconvolution(
        t,
        np.rint(mu),
        k,
        **BOUNDS,
        settings=fitting.BiexponentialFitSettings(
            max_starts=2, max_validation_evaluations=1
        ),
    )
    assert result.optimization.validation_evaluations == 1
    assert result.optimization.numerical_validation_passed is None
    assert (
        result.optimization.recovery_unavailable_reason == "validation_budget_exhausted"
    )


@pytest.mark.parametrize("policy", ["economical", "adaptive"])
def test_continuations_share_global_start_and_forward_caps(monkeypatch, policy):
    t, k, mu, truth, _ = independent_case(bins=64)
    wrong = replace(
        truth, amplitudes=tuple(a * 1.5 for a in truth.amplitudes), background=10
    )
    monkeypatch.setattr(
        fitting, "_initial_points", lambda p, _: [p.coordinates(wrong)] * 3
    )
    invocations = []

    def reject_trial(fun, x, **kwargs):
        invocations.append(x.copy())
        fun(x)
        fun(np.zeros(6))
        pytest.fail("invalid pair reached optimizer")

    monkeypatch.setattr(fitting, "minimize", reject_trial)
    result = fitting.fit_biexponential_reconvolution(
        t,
        np.rint(mu),
        k,
        **BOUNDS,
        settings=fitting.BiexponentialFitSettings(policy=policy, max_starts=3),
    )
    opt = result.optimization
    assert len(invocations) == len(opt.starts) == 3
    assert [s.kind for s in opt.starts] == [
        "seed",
        "safeguarded_recovery",
        "safeguarded_recovery",
    ]
    assert [s.start_index for s in opt.starts] == list(range(3))
    assert opt.objective_evaluations == sum(s.objective_evaluations for s in opt.starts)
    assert all(s.optimizer_reported_success is None for s in opt.starts)
    assert opt.recovery_unavailable_reason == "start_budget_exhausted"


@pytest.mark.parametrize("shift", [-1.0, 0.0, 1.0])
def test_shape_stencils_at_bounds_and_shift_knot(shift):
    p, x = problem()
    x[2] = p.lower[2]
    x[3] = p.upper[3]
    x[5] = shift
    value, gradient = p.value_and_gradient(x)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert p.counts["optimization"] == 7


def test_linear_derivative_overflow_is_failure_not_stationarity():
    p, x = problem()
    x[:2] = 0
    x[4] = 1e-310 / p.b_scale
    p.targets[:] = 1
    assert np.isfinite(p.evaluate(x))
    with pytest.raises(
        fitting._UnusableEvaluation, match="nonfinite_linear_derivative"
    ):
        p.value_and_gradient(x)


def test_unrepresentable_shape_step_cannot_fabricate_zero_derivative():
    p, x = problem(finite_difference_step=np.nextafter(0.0, 1.0))
    result = fitting._refine(p, x, 0, "seed")
    assert result.optimizer_reported_success is None
    assert "finite_difference_step_unrepresentable" in result.message
    assert result.best_parameters is not None
    assert result.objective_evaluations == 1
