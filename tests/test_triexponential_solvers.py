"""One invocation, counted safeguards, partial derivatives and active faces."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from tcspc_toolkit.estimation import _fixed_k_objective as objective
from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from test_fixed_k_foundations import make_problem, policy
from test_triexponential import settings, stationary_solver


def invoke(problem, base, **kwargs):
    values = dict(kind="seed", seed_id="test", calls=80, seconds=10)
    values.update(kwargs)
    return solvers._run_lbfgsb(problem, base, settings(), **values)


@pytest.mark.parametrize(
    "region,endpoint",
    [("continuous_nondifferentiable_knot", 0), ("support_discontinuity", 1)],
)
@pytest.mark.parametrize("approximate", [False, True])
def test_conditional_shift_is_fixed_without_fabricating_full_gradient(
    region, endpoint, approximate, monkeypatch
):
    t = np.arange(7.0)
    if approximate:
        t[3:] += 2e-8
    kernel = np.array([endpoint, 1, 2, 3, 2, 1, endpoint], dtype=float)
    problem, _ = make_problem(time=t, kernel=kernel)
    p = _FixedKParameters((2, 3, 4), (0.5, 1, 3), 1, 0)
    base = problem.evaluate(
        problem.coordinates.encode(p), phase="initialization", origin="base"
    )
    assert objective._shift_geometry(t, problem.inputs.kernel, 0).region == region
    monkeypatch.setattr(solvers, "minimize", stationary_solver)
    result = invoke(problem, base)
    assert result.fixed == (7,)
    assert result.derivative_statuses[-1] == "fixed_coordinate"
    assert result.best.parameters.temporal_shift_ns == 0
    assert result.reported_success
    full = objective._objective_derivatives(
        problem, base, policy(), phase="diagnostics"
    )
    assert not full.complete and full.gradient is None


def test_exact_zero_active_face_is_explicit_and_charged(monkeypatch):
    p = _FixedKParameters((0, 2, 3), (1, 1, 3), 0, 0)
    problem, x = make_problem(parameters=p, targets=np.zeros(65))
    base = problem.evaluate(x, phase="initialization", origin="zero_face:0,6")
    monkeypatch.setattr(solvers, "minimize", stationary_solver)
    result = invoke(problem, base, fix_zero_face=True)
    assert result.fixed == (0, 3, 6, 7)
    assert (
        result.best.parameters.amplitudes[0] == result.best.parameters.background == 0
    )
    assert result.calls > 0


@pytest.mark.parametrize("calls", [0, 1, 3, 10])
def test_nested_derivatives_obey_invocation_call_lease(calls):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    result = invoke(problem, base, calls=calls)
    assert result.calls <= calls
    assert problem.work.phases["optimization"]["model_calls"] == result.calls
    assert problem.work.reservation_guard is None
    assert any(reason == "invocation_model_limit" for _, reason in problem.work.blocked)
    assert result.reported_success is None


def test_nonfinite_objective_and_failed_derivative_are_distinct(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    original = objective._model_derivatives

    def unavailable(*args, **kwargs):
        result = original(*args, **kwargs)
        broken = replace(
            result.objective.derivatives[3], value=None, status="unrepresentable"
        )
        derivatives = list(result.objective.derivatives)
        derivatives[3] = broken
        return replace(
            result,
            objective=replace(
                result.objective, derivatives=tuple(derivatives), complete=False
            ),
        )

    monkeypatch.setattr(objective, "_model_derivatives", unavailable)
    result = invoke(problem, base)
    assert result.termination == "derivative_unavailable"
    assert result.best.finite
    zero = x.copy()
    zero[[0, 1, 2, 6]] = 0
    bad = problem.evaluate(zero, phase="initialization", origin="bad")
    result = invoke(problem, bad)
    assert result.termination == "infinite_objective" and result.best is None


def test_armijo_probe_after_failed_trial_is_counted(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")

    def failed_trial(fun, x, **kwargs):
        fun(x)
        bad = x.copy()
        bad[0] = np.finfo(float).max
        fun(bad)
        raise AssertionError("unreachable")

    monkeypatch.setattr(solvers, "minimize", failed_trial)
    result = invoke(problem, base)
    assert result.termination == "unrepresentable_parameters"
    assert result.safeguard_trials > 0 and result.calls <= 80
    assert result.best.centered_nll <= base.centered_nll


def test_endpoint_is_recomputed_instead_of_trusting_solver_fun(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    monkeypatch.setattr(
        solvers,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(
            x=x, fun=-np.inf, success=True, status=0, message="claimed"
        ),
    )
    result = invoke(problem, base)
    assert result.endpoint.centered_nll == base.centered_nll
    assert result.calls == 1


def test_nested_time_leases_and_observers_restore_on_error():
    clock = [0.0]
    problem, x = make_problem(clock=lambda: clock[0])
    with solvers._lease(problem, "optimization", 10, 1):
        with solvers._lease(problem, "optimization", 5, 2):
            clock[0] = 1.1
            result = problem.evaluate(x, phase="optimization", origin="late")
            assert result.status == "invocation_time_limit"
    assert problem.work.reservation_guard is None


def test_small_actual_lbfgsb_invocation_retains_finite_descent():
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    result = invoke(problem, base, calls=180)
    assert result.best is not None
    assert result.best.centered_nll < base.centered_nll
    assert result.calls <= 180


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_partial_derivatives_never_claim_a_full_gradient(k):
    problem, x = make_problem(k)
    base = problem.evaluate(x, phase="initialization", origin="base")
    full = objective._objective_derivatives(
        problem, base, policy(), phase="diagnostics"
    )
    partial = objective._objective_derivatives(
        problem,
        base,
        policy(),
        phase="diagnostics",
        active_indices=tuple(range(2 * k + 1)),
    )
    assert full.complete
    assert not partial.complete and partial.gradient is None
    assert partial.derivatives[-1].status == "fixed_coordinate"
    np.testing.assert_allclose(
        [d.value for d in partial.derivatives[:-1]], full.gradient[:-1]
    )


@pytest.mark.parametrize("active", [(), (-1,), (8,), (np.int64(1),)])
def test_invalid_active_derivative_coordinates_are_rejected(active):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    with pytest.raises(ValueError, match="active indices"):
        objective._objective_derivatives(
            problem, base, policy(), phase="diagnostics", active_indices=active
        )


def test_solver_programming_error_restores_accounting_hooks(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")

    def observer(evaluation):
        pass

    problem.evaluation_observer = observer

    def broken(*args, **kwargs):
        raise RuntimeError("deliberate programming failure")

    monkeypatch.setattr(solvers, "minimize", broken)
    with pytest.raises(RuntimeError, match="programming"):
        invoke(problem, base)
    assert problem.evaluation_observer is observer
    assert problem.work.reservation_guard is None


def test_fully_fixed_zero_face_does_not_invoke_a_solver(monkeypatch):
    p = _FixedKParameters((0, 0, 0), (1, 1, 1), 0, 0)
    problem, x = make_problem(parameters=p, targets=np.zeros(65))
    base = problem.evaluate(x, phase="initialization", origin="base")
    monkeypatch.setattr(
        solvers, "minimize", lambda *args, **kwargs: pytest.fail("no free coordinates")
    )
    result = invoke(problem, base, fix_zero_face=True)
    assert result.termination == "no_free_coordinates"
    assert result.calls == 0 and result.reported_success is None
