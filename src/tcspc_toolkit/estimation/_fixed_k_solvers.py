"""One counted L-BFGS-B invocation. No retries or continuation scheduling."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, least_squares

from tcspc_toolkit.estimation._fixed_k_objective import (
    _objective_derivatives,
    _shift_geometry,
    _model_derivatives,
)
from tcspc_toolkit.estimation._poisson_residuals import _poisson_residuals


class _UnusablePair(Exception):
    pass


@contextmanager
def _lease(problem, phase, calls, seconds):
    """Guard every physical call, including nested derivative/basis calls."""
    start_calls = problem.work.phases[phase]["model_calls"]
    started = problem.work.clock()
    previous = problem.work.reservation_guard

    def guard(actual_phase):
        if previous is not None:
            reason = previous(actual_phase)
            if reason:
                return reason
        if actual_phase == phase:
            if problem.work.phases[phase]["model_calls"] - start_calls >= calls:
                return "invocation_model_limit"
            if problem.work.clock() - started >= seconds:
                return "invocation_time_limit"
        return None

    problem.work.reservation_guard = guard
    try:
        yield
    finally:
        problem.work.reservation_guard = previous


@dataclass
class _Invocation:
    kind: str
    seed_id: str | None
    initial: object
    best: object
    endpoint: object
    reported_success: bool | None
    solver_status: int | None
    termination: str
    fixed: tuple[int, ...]
    call_limit: int
    calls: int
    seconds: float
    derivative_statuses: tuple[str, ...]
    safeguard_trials: int
    solver: str = "lbfgsb"
    residual_evaluations: int = 0
    jacobian_evaluations: int = 0
    inward_start_coordinates: tuple | None = None
    failure_events: int = 0
    failure_diagnostics: tuple = ()


def _run_lbfgsb(
    problem,
    candidate,
    settings,
    *,
    kind,
    seed_id,
    calls,
    seconds,
    fix_zero_face=False,
    invocation_id=None,
):
    started = problem.work.clock()
    before = problem.work.phases["optimization"]["model_calls"]
    x0 = candidate.coordinates.copy()
    geometry = _shift_geometry(
        problem.inputs.time_ns,
        problem.inputs.kernel,
        candidate.parameters.temporal_shift_ns,
    )
    fixed = set()
    if geometry.region in (
        "support_discontinuity",
        "continuous_nondifferentiable_knot",
    ):
        fixed.add(7)
    if fix_zero_face:
        fixed.update(i for i in (0, 1, 2, 6) if x0[i] == 0)
        fixed.update(i + 3 for i in range(3) if i in fixed)
    free = tuple(i for i in range(8) if i not in fixed)
    best = endpoint = None
    anchor = None
    statuses = ()
    trials = 0
    success = status = None
    termination = "not_invoked"
    previous_observer = problem.evaluation_observer
    previous_identity = problem.invocation_identity
    problem.invocation_identity = (invocation_id, kind, seed_id, "lbfgsb")
    failure_before = problem.failure_events
    last_evaluation = candidate

    def observe(evaluation):
        nonlocal best
        if evaluation.finite and (
            best is None or evaluation.centered_nll < best.centered_nll
        ):
            best = evaluation
        if previous_observer is not None:
            previous_observer(evaluation)

    def expand(values):
        x = x0.copy()
        x[list(free)] = values
        return x

    def fun(values):
        nonlocal anchor, statuses, last_evaluation
        x = expand(values)
        base = problem.evaluate(x, phase="optimization", origin=kind)
        last_evaluation = base
        if not base.finite:
            raise _UnusablePair(base.status)
        derivative = _objective_derivatives(
            problem,
            base,
            settings.derivative_policy,
            phase="optimization",
            active_indices=free,
        )
        statuses = tuple(d.status for d in derivative.derivatives)
        values = [derivative.derivatives[i].value for i in free]
        if any(v is None for v in values) or problem.work.deadline_reached:
            raise _UnusablePair("derivative_unavailable")
        gradient = np.asarray(values)
        if not np.all(np.isfinite(gradient)):
            raise _UnusablePair("nonfinite_gradient")
        if anchor is None or base.centered_nll < anchor[0].centered_nll:
            anchor = (base, gradient.copy())
        return base.centered_nll, gradient

    problem.evaluation_observer = observe
    try:
        with _lease(problem, "optimization", calls, seconds):
            try:
                if not free:
                    # A completely fixed face occupies this scheduled slot but
                    # does not fabricate a solver call or stationary success.
                    raise _UnusablePair("no_free_coordinates")
                result = minimize(
                    fun,
                    x0[list(free)],
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[
                        (problem.coordinates.lower[i], problem.coordinates.upper[i])
                        for i in free
                    ],
                    options=dict(
                        maxiter=settings.max_iterations,
                        maxls=settings.max_line_search,
                        ftol=settings.ftol,
                        gtol=settings.gtol,
                        maxcor=10,
                    ),
                )
                success, status = bool(result.success), int(result.status)
                termination = str(result.message)
                endpoint = problem.evaluate(
                    expand(result.x), phase="optimization", origin="solver_endpoint"
                )
                if not endpoint.finite:
                    termination += "; endpoint:" + endpoint.status
            except _UnusablePair as error:
                termination = str(error)
                # A finite Armijo probe can survive an unsuccessful invocation.
                # It never causes a second solver invocation here.
                if anchor is not None:
                    base, gradient = anchor
                    direction = np.zeros(8)
                    indices = list(free)
                    direction[indices] = (
                        np.clip(
                            base.coordinates[indices] - gradient,
                            problem.coordinates.lower[indices],
                            problem.coordinates.upper[indices],
                        )
                        - base.coordinates[indices]
                    )
                    slope = float(gradient @ direction[indices])
                    if np.isfinite(slope) and slope < 0:
                        for exponent in range(settings.max_line_search):
                            point = base.coordinates + 0.5**exponent * direction
                            if np.array_equal(point, base.coordinates):
                                break
                            probe = problem.evaluate(
                                point, phase="optimization", origin="armijo_probe"
                            )
                            trials += 1
                            if "limit" in probe.status:
                                break
                            if (
                                probe.finite
                                and probe.centered_nll < base.centered_nll
                                and probe.centered_nll
                                <= base.centered_nll + 1e-4 * 0.5**exponent * slope
                            ):
                                break
    finally:
        if success is not True:
            problem.record_failure(last_evaluation, termination, active_indices=free)
        problem.evaluation_observer = previous_observer
        problem.invocation_identity = previous_identity
    return _Invocation(
        kind,
        seed_id,
        candidate,
        best,
        endpoint,
        success,
        status,
        termination,
        tuple(sorted(fixed)),
        calls,
        problem.work.phases["optimization"]["model_calls"] - before,
        problem.work.clock() - started,
        statuses,
        trials,
        failure_events=problem.failure_events - failure_before,
        failure_diagnostics=tuple(
            f for f in problem.recent_failures if f.event > failure_before
        ),
    )


def _run_trf(
    problem,
    candidate,
    settings,
    *,
    kind,
    seed_id,
    calls,
    seconds,
    fix_zero_face=False,
    invocation_id=None,
):
    """One restricted or joint TRF call; every physical evaluation is charged.

    SciPy's feasible inward adjustment is observed, not reproduced or used as
    a hidden lower bound. The exact submitted boundary candidate is retained.
    """
    started = problem.work.clock()
    before = problem.work.phases["optimization"]["model_calls"]
    x0 = candidate.coordinates.copy()
    fixed = set()
    region = _shift_geometry(
        problem.inputs.time_ns,
        problem.inputs.kernel,
        candidate.parameters.temporal_shift_ns,
    ).region
    if region in ("support_discontinuity", "continuous_nondifferentiable_knot"):
        fixed.add(7)
    if fix_zero_face:
        fixed.update(i for i in (0, 1, 2, 6) if x0[i] == 0)
        fixed.update(i + 3 for i in range(3) if i in fixed)
    free = tuple(i for i in range(8) if i not in fixed)
    best, endpoint = candidate, None
    previous = problem.evaluation_observer
    previous_identity = problem.invocation_identity
    problem.invocation_identity = (invocation_id, kind, seed_id, "trf")
    failure_before = problem.failure_events
    last_evaluation = candidate
    residual_calls = jacobian_calls = 0
    inward = None
    statuses = ()
    success = status = None
    termination = "not_invoked"

    def observe(e):
        nonlocal best
        if e.finite and e.centered_nll < best.centered_nll:
            best = e
        if previous is not None:
            previous(e)

    def expand(values):
        x = x0.copy()
        x[list(free)] = values
        return x

    def evaluate(values, origin):
        nonlocal last_evaluation
        e = problem.evaluate(expand(values), phase="optimization", origin=origin)
        last_evaluation = e
        if not e.finite:
            raise _UnusablePair(e.status)
        return e

    def fun(values):
        nonlocal residual_calls, inward
        if residual_calls == 0 and not np.array_equal(expand(values), x0):
            inward = tuple(map(float, expand(values)))
        residual_calls += 1
        e = evaluate(values, "trf_residual")
        r = _poisson_residuals(problem.inputs.targets, e.expected_counts).values
        if not np.all(np.isfinite(r)):
            raise _UnusablePair("residual_unavailable")
        return r

    def jac(values):
        nonlocal jacobian_calls, statuses
        jacobian_calls += 1
        e = evaluate(values, "trf_jacobian_anchor")
        r = _poisson_residuals(problem.inputs.targets, e.expected_counts)
        if not np.all(r.derivative_available):
            # Conservative: a zero model slope alone does not prove that sqrt(mu)
            # has zero derivative (mu may grow quadratically on a free face).
            raise _UnusablePair("residual_derivative_unavailable")
        model = _model_derivatives(
            problem,
            e,
            settings.derivative_policy,
            phase="optimization",
            active_indices=free,
        )
        statuses = tuple(d.status for d in model.objective.derivatives)
        if not model.columns or any(model.columns[i] is None for i in free):
            raise _UnusablePair("model_jacobian_unavailable")
        with np.errstate(over="ignore", invalid="ignore"):
            result = (
                np.column_stack([model.columns[i] for i in free])
                * r.derivatives[:, None]
            )
        if not np.all(np.isfinite(result)) or problem.work.deadline_reached:
            raise _UnusablePair("residual_jacobian_unavailable")
        return result

    problem.evaluation_observer = observe
    try:
        with _lease(problem, "optimization", calls, seconds):
            try:
                if not free:
                    raise _UnusablePair("no_free_coordinates")
                result = least_squares(
                    fun,
                    x0[list(free)],
                    jac=jac,
                    method="trf",
                    bounds=(
                        problem.coordinates.lower[list(free)],
                        problem.coordinates.upper[list(free)],
                    ),
                    ftol=settings.ftol,
                    xtol=settings.ftol,
                    gtol=settings.gtol,
                    x_scale=1.0,
                    max_nfev=settings.max_iterations,
                )
                success, status = bool(result.success), int(result.status)
                termination = str(result.message)
                endpoint = problem.evaluate(
                    expand(result.x), phase="optimization", origin="trf_endpoint"
                )
                if not endpoint.finite:
                    termination += "; endpoint:" + endpoint.status
            except _UnusablePair as error:
                termination = str(error)
            except (
                ValueError,
                FloatingPointError,
                OverflowError,
                np.linalg.LinAlgError,
            ) as error:
                termination = f"solver_numerical_failure:{type(error).__name__}:{error}"
    finally:
        if success is not True:
            problem.record_failure(last_evaluation, termination, active_indices=free)
        problem.evaluation_observer = previous
        problem.invocation_identity = previous_identity
    return _Invocation(
        kind,
        seed_id,
        candidate,
        best,
        endpoint,
        success,
        status,
        termination,
        tuple(sorted(fixed)),
        calls,
        problem.work.phases["optimization"]["model_calls"] - before,
        problem.work.clock() - started,
        statuses,
        0,
        "trf",
        residual_calls,
        jacobian_calls,
        inward,
        problem.failure_events - failure_before,
        tuple(f for f in problem.recent_failures if f.event > failure_before),
    )
