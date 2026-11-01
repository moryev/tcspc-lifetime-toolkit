"""One counted L-BFGS-B invocation. No retries or continuation scheduling."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from tcspc_toolkit.estimation._fixed_k_objective import (
    _objective_derivatives,
    _shift_geometry,
)


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


def _run_lbfgsb(
    problem, candidate, settings, *, kind, seed_id, calls, seconds, fix_zero_face=False
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
        nonlocal anchor, statuses
        x = expand(values)
        base = problem.evaluate(x, phase="optimization", origin=kind)
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
        problem.evaluation_observer = previous_observer
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
    )
