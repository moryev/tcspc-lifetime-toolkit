"""New PR 2C-1 mathematical fixtures; no pilot inputs or optimizer replay."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_objective as objective
from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from test_fixed_k_foundations import make_problem, policy
from test_fixed_k_derivatives import independent_gradient
from test_triexponential import settings, stationary_solver


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("distance", [0, 1, 4, "inward"])
@pytest.mark.parametrize("index", [3, 7])
def test_exact_and_near_bounds_match_independent_smooth_oracle(upper, distance, index):
    problem, x = make_problem()
    x = x.copy()
    p = problem.coordinates
    bound = p.upper[index] if upper else p.lower[index]
    sign = -1 if upper else 1
    x[index] = bound
    if distance == "inward":
        x[index] += sign * 1e-10
    else:
        for _ in range(distance):
            x[index] = np.nextafter(x[index], sign * np.inf)
    base = problem.evaluate(x, phase="optimization", origin="mathematical_bound")
    before = base.coordinates.copy(), base.parameters
    d = objective._model_derivatives(problem, base, policy(), phase="diagnostics")
    assert d.objective.complete
    np.testing.assert_allclose(
        d.objective.gradient, independent_gradient(problem, x), rtol=4e-6, atol=3e-6
    )
    item = d.objective.derivatives[index]
    assert item.status == (
        "one_sided_bound" if distance == 0 else "feasible_side_near_bound"
    )
    near = item.right_stencil if upper else item.left_stencil
    far = item.left_stencil if upper else item.right_stencil
    assert near.status == ("bound" if distance == 0 else "insufficient_bound_room")
    assert (near.feasible_room == 0) == (distance == 0)
    assert far.status == "available" and far.refinements >= 1
    assert len({point.physical_value for point in far.points}) >= 3
    assert all(sign * point.physical_step > 0 for point in far.points)
    assert all(point.effective_step != 0 for point in far.points)
    np.testing.assert_array_equal(base.coordinates, before[0])
    assert base.parameters == before[1]
    assert problem.work.charged_component_convolutions == 3 * problem.work.model_calls


@pytest.mark.parametrize("upper", [False, True])
def test_one_ulp_physical_lifetime_is_not_silently_snapped(upper):
    tau = np.nextafter(10.0 if upper else 0.1, 1.0)
    problem, x = make_problem(
        parameters=_FixedKParameters((9, 2, 1), (tau, 1.3, 2.7), 1, 0.037)
    )
    base = problem.evaluate(x, phase="optimization", origin="physical_ulp")
    decoded = base.parameters
    d = objective._model_derivatives(problem, base, policy(), phase="diagnostics")
    assert d.columns[3] is not None
    assert base.parameters == decoded
    # The existing coordinate decoder remains authoritative, including its own
    # endpoint roundoff. Derivative construction does not change the candidate.
    assert decoded == problem.coordinates.decode(x)


@pytest.mark.parametrize("amplitude", [0, 1e-10, 7])
def test_lifetime_column_against_direct_analytic_convolution(amplitude):
    problem, x = make_problem(
        1, parameters=_FixedKParameters((amplitude,), (0.73,), 0.9, 0.037)
    )
    base = problem.evaluate(x, phase="optimization", origin="analytic_oracle")
    d = objective._model_derivatives(problem, base, policy(), phase="diagnostics")
    t, k = problem.inputs.time_ns, problem.inputs.kernel
    shifted = np.interp(t - 0.037, t, k, left=0, right=0)
    expected = np.convolve(amplitude * np.exp(-t / 0.73) * t / 0.73, shifted)[
        : len(t)
    ] * (t[1] - t[0])
    np.testing.assert_allclose(d.columns[1], expected, rtol=3e-6, atol=2e-7)
    if amplitude == 0:
        assert d.objective.derivatives[1].status == "zero_component"


def test_inconsistent_sides_are_not_replaced_by_one_smooth_side(monkeypatch):
    problem, x = make_problem(1)
    base = problem.evaluate(x, phase="optimization", origin="inconsistent")
    original = objective._shape_side

    def inconsistent(*args):
        col, err, reason, h, trace = original(*args)
        if args[2] == 1 and args[3] == -1:
            col = col + 1
        return col, err, reason, h, trace

    monkeypatch.setattr(objective, "_shape_side", inconsistent)
    d = objective._model_derivatives(problem, base, policy(), phase="diagnostics")
    assert d.columns[1] is None
    assert d.objective.derivatives[1].status == "unequal_one_sided_derivatives"


def test_bound_room_does_not_override_a_support_discontinuity():
    problem, _ = make_problem(1, time=np.arange(7.0), kernel=np.ones(7))
    problem.coordinates = replace(
        problem.coordinates, temporal_shift_bounds_ns=(-1e-10, 1)
    )
    x = problem.coordinates.encode(_FixedKParameters((3,), (1,), 1, 0))
    base = problem.evaluate(x, phase="optimization", origin="jump")
    d = objective._model_derivatives(problem, base, policy(), phase="diagnostics")
    assert d.columns[-1] is None and d.objective.shift_region == "support_discontinuity"


def test_distinct_coordinates_with_unrepresentable_physical_shift_are_recorded():
    problem, _ = make_problem(1)
    problem.coordinates = replace(
        problem.coordinates, temporal_shift_bounds_ns=(1e16, 1e16 + 4)
    )
    x = problem.coordinates.encode(_FixedKParameters((3,), (1,), 2, 1e16 + 2))
    base = problem.evaluate(x, phase="optimization", origin="physical_roundoff")
    d = objective._model_derivatives(problem, base, policy(), phase="diagnostics")
    assert d.columns[-1] is None
    trace = d.objective.derivatives[-1].left_stencil
    assert trace.points[0].coordinate_step != 0
    assert trace.points[0].physical_step == 0
    assert trace.status == "unrepresentable_stencil"
    failure = problem.recent_failures[-1]
    assert failure.candidate_id == base.evaluation_id
    assert failure.parameters == base.parameters
    assert failure.coordinates == tuple(x)
    assert failure.physical_bounds[-1] == (1e16, 1e16 + 4)


@pytest.mark.parametrize("solver", ["lbfgsb", "trf"])
def test_controlled_solver_uses_inward_stencil_without_hidden_work(solver, monkeypatch):
    problem, x = make_problem()
    x = x.copy()
    x[3] = problem.coordinates.lower[3] + 1e-10
    x[7] = 1 - 1e-10
    base = problem.evaluate(x, phase="initialization", origin="new_inward_fixture")
    if solver == "trf":

        def stationary(fun, x, jac, **kw):
            fun(x)
            matrix = jac(x)
            assert matrix.shape == (65, 8) and np.all(np.isfinite(matrix))
            return stationary_solver(lambda _: None, x)

        monkeypatch.setattr(solvers, "least_squares", stationary)
    else:
        monkeypatch.setattr(solvers, "minimize", stationary_solver)
    runner = solvers._run_trf if solver == "trf" else solvers._run_lbfgsb
    result = runner(
        problem,
        base,
        settings(),
        kind="seed",
        seed_id="new-fixture",
        calls=80,
        seconds=10,
        invocation_id=3,
    )
    assert result.reported_success
    assert result.calls == problem.work.phases["optimization"]["model_calls"] <= 80
    assert result.initial.parameters == base.parameters


def test_failure_instrumentation_is_bounded_and_captures_invocation(monkeypatch):
    problem, x = make_problem()
    x = x.copy()
    x[-1] = 0
    base = problem.evaluate(x, phase="initialization", origin="knot")

    def repeated(fun, x, **kw):
        for _ in range(20):
            bad = base.coordinates.copy()
            bad[0] = -1
            problem.evaluate(bad, phase="optimization", origin="rejected_probe")
        return stationary_solver(fun, x)

    monkeypatch.setattr(solvers, "minimize", repeated)
    result = solvers._run_lbfgsb(
        problem,
        base,
        settings(),
        kind="seed",
        seed_id="trace",
        calls=80,
        seconds=10,
        invocation_id=7,
    )
    assert result.failure_events == 20 and len(result.failure_diagnostics) == 8
    for failure in result.failure_diagnostics:
        assert failure.invocation == (7, "seed", "trace", "lbfgsb")
        assert failure.phase == "optimization" and failure.coordinates[0] == -1
        assert failure.reason == "infeasible_coordinates"
        assert not hasattr(failure, "expected_counts")


def test_failure_instrumentation_preserves_foreign_anchor_rejection():
    first, x = make_problem()
    second, _ = make_problem(time=np.arange(7.0))
    base = first.evaluate(x, phase="optimization", origin="foreign_grid")
    d = objective._model_derivatives(second, base, policy(), phase="diagnostics")
    assert d.jacobian is None and not d.objective.complete
    failure = second.recent_failures[-1]
    assert failure.reason == "invalid_derivative_anchor"
    assert failure.shift_geometry is None
    assert failure.positive_target_zero_expectation_bins is None
    assert second.work.model_calls == 0


def test_failure_instrumentation_preserves_malformed_coordinate_rejection():
    problem, x = make_problem()
    result = problem.evaluate(x.reshape(2, 4), phase="optimization", origin="bad_shape")
    assert result.status == "infeasible_coordinates"
    failure = problem.recent_failures[-1]
    assert failure.coordinate_shape == (2, 4)
    assert failure.coordinates == tuple(x)
    assert problem.work.model_calls == 0
