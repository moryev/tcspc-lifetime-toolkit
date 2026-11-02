"""Bounded mathematical and solver-contract tests, no development observations."""

from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tcspc_toolkit.estimation._poisson_residuals import _poisson_residuals
from tcspc_toolkit.benchmarks._triexponential_audit import audit_terms
from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from test_fixed_k_foundations import make_problem
from test_triexponential import settings, fit


@pytest.mark.parametrize("y", [0.0, 1e-200, 1e-15, 1.0, 1e8, 1e100, 1e250])
@pytest.mark.parametrize("ratio", [0.00001, 0.2, 0.9, 1.0, 1 + 1e-12, 1.1, 1.8, 1e5])
def test_residual_squared_matches_independent_decimal_objective(y, ratio):
    mu = y * ratio if y else ratio
    with localcontext() as ctx:
        ctx.prec = 90
        a, b = Decimal(y), Decimal(mu)
        expected = float(b - a + a * (a.ln() - b.ln())) if y else mu
    r = _poisson_residuals([y], [mu])
    value = (r.values[0] / np.sqrt(2)) ** 2
    assert value == pytest.approx(expected, rel=5e-13, abs=1e-300)
    assert audit_terms([y], [mu])[0] == pytest.approx(expected, rel=5e-13, abs=1e-300)


@pytest.mark.parametrize("y", [1e-100, 1.0, 1e12, 1e200])
@pytest.mark.parametrize("delta", [-1e-5, -1e-12, 0, 1e-12, 1e-5])
def test_smooth_residual_derivative_near_equality(y, delta):
    mu = y * (1 + delta)
    r = _poisson_residuals([y], [mu])
    step = y * 1e-6
    finite_difference = (
        _poisson_residuals([y], [mu + step]).values
        - _poisson_residuals([y], [mu - step]).values
    ) / (2 * step)
    np.testing.assert_allclose(r.derivatives, finite_difference, rtol=2e-9)
    assert r.derivatives[0] == pytest.approx(-1 / np.sqrt(y), rel=1e-4)


def test_zero_and_tiny_expectation_semantics():
    result = _poisson_residuals([0, 0, 2, 2], [0, 3, 0, np.nextafter(0.0, 1.0)])
    assert result.values[0] == 0 and not result.derivative_available[0]
    assert result.values[1] == pytest.approx(-np.sqrt(6))
    assert result.derivatives[1] == pytest.approx(-1 / np.sqrt(6))
    assert np.isinf(result.values[2]) and not result.derivative_available[2]
    assert np.isfinite(result.values[3]) and not result.derivative_available[3]


@pytest.mark.parametrize(
    "y,mu",
    [
        (1e308, 1e308),
        (1e308, np.nextafter(1e308, np.inf)),
        (1e308, 1.1e308),
        (1.7e308, 9e307),
        (9e307, 1.7e308),
        (1e100, 1e-250),
    ],
)
def test_finite_extreme_objective_and_derivative_avoid_intermediate_overflow(y, mu):
    with localcontext() as ctx:
        ctx.prec = 90
        a, b = Decimal(y), Decimal(mu)
        c = b - a + a * (a.ln() - b.ln())
        residual = (2 * c).sqrt() * (1 if y > mu else -1)
        derivative = (b - a) / b / residual if y != mu else -1 / a.sqrt()
    with np.errstate(over="raise", invalid="raise"):
        r = _poisson_residuals([y], [mu])
        independent = audit_terms([y], [mu])[0]
    assert (r.values[0] / np.sqrt(2)) ** 2 == pytest.approx(float(c), rel=3e-12)
    assert independent == pytest.approx(float(c), rel=3e-12)
    assert r.derivative_available[0]
    assert r.derivatives[0] == pytest.approx(float(derivative), rel=3e-12, abs=0)


def test_nonrepresentable_residual_cannot_claim_an_available_derivative():
    r = _poisson_residuals([1e308], [1e-300])
    assert np.isinf(r.values[0])
    assert not r.derivative_available[0]


@pytest.mark.parametrize(
    "y,mu", [([-1], [1]), ([1], [-1]), ([np.nan], [1]), ([1, 2], [1])]
)
def test_invalid_residual_inputs(y, mu):
    with pytest.raises(ValueError):
        _poisson_residuals(y, mu)


def invoke(problem, base, **kw):
    return solvers._run_trf(
        problem,
        base,
        settings(),
        kind="seed",
        seed_id="controlled",
        calls=kw.get("calls", 120),
        seconds=30,
        fix_zero_face=kw.get("fix_zero_face", False),
    )


def test_trf_inward_start_is_observed_and_exact_zero_is_retained():
    problem, x = make_problem(targets=np.full(65, 3.0))
    x = x.copy()
    x[0] = 0
    base = problem.evaluate(x, phase="initialization", origin="boundary")
    result = invoke(problem, base)
    assert result.inward_start_coordinates is not None
    assert result.inward_start_coordinates[0] > 0
    assert result.initial.parameters.amplitudes[0] == 0
    assert result.residual_evaluations > 0 and result.jacobian_evaluations > 0
    assert result.calls <= 120 and result.best.finite


def test_trf_refuses_unavailable_zero_residual_derivative(monkeypatch):
    problem, x = make_problem(targets=np.zeros(65))
    x = x.copy()
    x[[0, 1, 2, 6]] = 0
    base = problem.evaluate(x, phase="initialization", origin="zero")

    def controlled(fun, x, jac, **kw):
        assert np.all(fun(x) == 0)
        jac(x)
        pytest.fail("unavailable derivative must abort")

    monkeypatch.setattr(solvers, "least_squares", controlled)
    result = invoke(problem, base)
    assert result.termination == "residual_derivative_unavailable"
    assert result.best is base and result.reported_success is None


@pytest.mark.parametrize("approximate", [False, True])
def test_trf_conditional_shift_and_exact_face(approximate, monkeypatch):
    time = np.arange(7.0)
    if approximate:
        time[3:] += 2e-8
    problem, x = make_problem(time=time, kernel=np.ones(7), targets=np.full(7, 3.0))
    x = x.copy()
    x[0] = 0
    x[7] = 0
    base = problem.evaluate(x, phase="initialization", origin="face")

    def controlled(fun, x, jac, **kw):
        assert jac(x).shape == (7, len(x))
        return SimpleNamespace(x=x, success=True, status=1, message="controlled")

    monkeypatch.setattr(solvers, "least_squares", controlled)
    result = invoke(problem, base, fix_zero_face=True)
    assert result.fixed == (0, 3, 7)
    assert result.best.parameters.amplitudes[0] == 0


@pytest.mark.parametrize("calls", [0, 1, 2, 10])
def test_trf_lease_limits_and_finite_retention(calls):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    result = invoke(problem, base, calls=calls)
    assert result.calls <= calls and result.best.finite
    assert result.reported_success is None


def test_solver_switching_keeps_seed_first_reservations(monkeypatch):
    def false(fun, x, **kw):
        return SimpleNamespace(x=x, success=True, status=1, message="false")

    monkeypatch.setattr(solvers, "least_squares", false)
    monkeypatch.setattr(solvers, "minimize", false)
    result = fit(
        np.arange(33) % 4 + 1,
        config=replace(settings(), solver="trf", recovery_solver="lbfgsb"),
    )
    starts = result.optimization.starts
    assert all(s.solver == "trf" and s.kind == "seed" for s in starts[:10])
    assert all(
        s.solver == "lbfgsb" and s.kind == "required_recovery" for s in starts[10:]
    )
    assert not result.numerical.accepted
    assert (
        sum(s.model_calls for s in starts)
        == result.optimization.work.phases["optimization"]["model_calls"]
    )


def test_zero_observations_use_analytic_path_for_trf():
    result = fit(config=replace(settings(), solver="trf"))
    assert result.numerical.accepted and result.centered_nll == 0
    assert result.optimization.starts == ()


def test_trf_unusable_stencil_does_not_fake_a_jacobian(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    original = solvers._model_derivatives

    def broken(*args, **kw):
        return replace(original(*args, **kw), columns=())

    monkeypatch.setattr(solvers, "_model_derivatives", broken)
    result = invoke(problem, base)
    assert result.termination == "model_jacobian_unavailable"
    assert result.best.finite


@pytest.mark.parametrize("failure", ["infeasible", "model"])
def test_trf_failure_retains_improved_probe_and_charges_actual_work(
    monkeypatch, failure
):
    from tcspc_toolkit.estimation import _fixed_k_objective as objective

    problem, x = make_problem(targets=np.zeros(65))
    base = problem.evaluate(x, phase="initialization", origin="base")

    def controlled(fun, x, **kw):
        trial = x.copy()
        trial[:3] *= 0.5
        fun(trial)
        if failure == "infeasible":
            trial[0] = np.nan
        else:

            def failed(*a, **kw):
                raise FloatingPointError("controlled model failure")

            monkeypatch.setattr(
                objective, "multiexponential_reconvolution_component_counts", failed
            )
        fun(trial)
        pytest.fail("failed objective must abort the invocation")

    monkeypatch.setattr(solvers, "least_squares", controlled)
    result = invoke(problem, base)
    assert result.best.centered_nll < base.centered_nll
    assert result.reported_success is None
    assert result.residual_evaluations == 2
    assert result.calls == (1 if failure == "infeasible" else 2)
    assert (
        problem.work.phases["optimization"]["component_convolutions"]
        == 3 * result.calls
    )


def test_trf_numerical_solver_exception_retains_finite_evidence(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")

    def broken(fun, x, **kw):
        fun(x)
        raise np.linalg.LinAlgError("controlled factorization failure")

    monkeypatch.setattr(solvers, "least_squares", broken)
    result = invoke(problem, base)
    assert result.best.finite and result.calls == 1
    assert result.reported_success is None
    assert result.termination.startswith("solver_numerical_failure:LinAlgError")
