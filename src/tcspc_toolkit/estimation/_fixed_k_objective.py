"""Counted fixed-K model/objective and derivatives, independent of any solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from time import monotonic
from types import MappingProxyType

import numpy as np

from tcspc_toolkit.estimation._fixed_k import (
    _CoordinateDerivative,
    _CoordinateMap,
    _DerivativeEvaluation,
    _DerivativePolicy,
    _Evaluation,
    _FixedKInputs,
    _FixedKParameters,
    _positive_integer,
    _readonly,
    _scalar,
    _StencilPoint,
    _SideStencil,
    _NumericalFailure,
)
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
)

_PHASES = ("initialization", "optimization", "validation", "assessment", "diagnostics")


@dataclass(frozen=True)
class _WorkLimits:
    """Explicit ceilings. Phase limits count model calls, including derivative work."""

    phase_model_calls: dict[str, int]
    model_calls: int
    objective_calls: int
    component_convolutions: int
    wall_seconds: float

    def __post_init__(self):
        if set(self.phase_model_calls) != set(_PHASES):
            raise ValueError("each numerical phase needs an explicit model-call limit")
        limits = {}
        for name, count in self.phase_model_calls.items():
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count < 0
            ):
                raise ValueError("phase limits must be nonnegative integers")
            limits[name] = int(count)
        object.__setattr__(self, "phase_model_calls", MappingProxyType(limits))
        for name in ("model_calls", "objective_calls", "component_convolutions"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        value = _scalar(self.wall_seconds, "wall_seconds")
        if value <= 0:
            raise ValueError("wall_seconds must be positive")
        object.__setattr__(self, "wall_seconds", value)


@dataclass
class _WorkLedger:
    limits: _WorkLimits
    clock: object = monotonic
    reservation_guard: object = None
    requests: int = 0
    model_calls: int = 0
    objective_calls: int = 0
    objective_attempts: int = 0
    charged_component_convolutions: int = 0
    completed_component_convolutions: int = 0
    model_failures_with_unknown_partial_work: int = 0
    blocked: list[tuple[str, str]] = field(default_factory=list)
    phases: dict = field(
        default_factory=lambda: {
            p: {
                "requests": 0,
                "model_calls": 0,
                "objective_calls": 0,
                "objective_attempts": 0,
                "component_convolutions": 0,
            }
            for p in _PHASES
        }
    )

    def __post_init__(self):
        self.started = self.clock()

    @property
    def elapsed_seconds(self):
        return max(0.0, self.clock() - self.started)

    @property
    def deadline_reached(self):
        return self.elapsed_seconds >= self.limits.wall_seconds

    def request(self, phase):
        if phase not in _PHASES:
            raise ValueError("unknown numerical phase")
        self.requests += 1
        self.phases[phase]["requests"] += 1

    def reserve(self, phase, k, *, objective):
        limits = self.limits
        reason = None
        if self.deadline_reached:
            reason = "wall_time_limit"
        elif self.reservation_guard is not None:
            reason = self.reservation_guard(phase)
        if (
            reason is None
            and self.phases[phase]["model_calls"] >= limits.phase_model_calls[phase]
        ):
            reason = "phase_model_limit"
        elif reason is None and self.model_calls >= limits.model_calls:
            reason = "model_limit"
        elif (
            reason is None
            and objective
            and self.objective_attempts >= limits.objective_calls
        ):
            reason = "objective_limit"
        elif (
            reason is None
            and self.charged_component_convolutions + k > limits.component_convolutions
        ):
            reason = "component_work_limit"
        if reason:
            self.blocked.append((phase, reason))
            return reason
        self.model_calls += 1
        self.objective_attempts += int(objective)
        self.charged_component_convolutions += k
        self.phases[phase]["model_calls"] += 1
        self.phases[phase]["objective_attempts"] += int(objective)
        self.phases[phase]["component_convolutions"] += k
        return None


class _FixedKProblem:
    """Every physical call passes through this ledger; there is no cache/retry."""

    def __init__(
        self,
        inputs: _FixedKInputs,
        coordinates: _CoordinateMap,
        limits: _WorkLimits,
        *,
        clock=monotonic,
    ):
        self.inputs = inputs
        self.coordinates = coordinates
        self.work = _WorkLedger(limits, clock)
        self.best_finite = None
        # Optional consumer retains evidence; it must explicitly select/promote it.
        self.evaluation_observer = None
        self._evaluation_id = 0
        self._identity = object()
        self.invocation_identity = None
        self.failure_events = 0
        self.recent_failures = deque(maxlen=8)

    def record_failure(self, evaluation, reason, *, derivatives=(), active_indices=()):
        self.failure_events += 1
        p = self.coordinates
        same_problem = evaluation.problem_id is self._identity
        geometry = (
            _shift_geometry(
                self.inputs.time_ns,
                self.inputs.kernel,
                evaluation.parameters.temporal_shift_ns,
            )
            if same_problem and evaluation.parameters is not None
            else None
        )
        self.recent_failures.append(
            _NumericalFailure(
                self.failure_events,
                self.invocation_identity,
                evaluation.evaluation_id,
                evaluation.phase,
                evaluation.origin,
                reason,
                tuple(map(float, evaluation.coordinates.ravel()[: p.dimension])),
                evaluation.parameters,
                tuple(zip(map(float, p.lower), map(float, p.upper))),
                ((0.0, np.inf),) * p.n_components
                + (p.lifetime_bounds_ns,) * p.n_components
                + ((0.0, np.inf), p.temporal_shift_bounds_ns),
                tuple(active_indices),
                geometry,
                tuple(derivatives),
                int(np.sum(evaluation.expected_counts == 0))
                if same_problem
                and evaluation.expected_counts is not None
                and evaluation.expected_counts.shape == self.inputs.targets.shape
                else None,
                int(
                    np.sum(
                        (evaluation.expected_counts == 0) & (self.inputs.targets > 0)
                    )
                )
                if same_problem
                and evaluation.expected_counts is not None
                and evaluation.expected_counts.shape == self.inputs.targets.shape
                else None,
                tuple(evaluation.coordinates.shape),
            )
        )

    def _model(self, parameters, phase, *, objective):
        reason = self.work.reserve(phase, parameters.n_components, objective=objective)
        if reason:
            return None, reason
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="ignore"
            ):
                components = multiexponential_reconvolution_component_counts(
                    self.inputs.time_ns,
                    self.inputs.kernel,
                    amplitudes=parameters.amplitudes,
                    lifetimes=parameters.lifetimes_ns,
                    temporal_shift=parameters.temporal_shift_ns,
                )
        except (FloatingPointError, OverflowError):
            self.work.model_failures_with_unknown_partial_work += 1
            return None, "model_numerical_failure"
        self.work.completed_component_convolutions += parameters.n_components
        if (
            components.shape != (parameters.n_components, self.inputs.time_ns.size)
            or not np.all(np.isfinite(components))
            or np.any(components < 0)
        ):
            return None, "invalid_model_output"
        return components, None

    def evaluate(self, coordinates, *, phase, origin):
        self.work.request(phase)
        self._evaluation_id += 1
        x = _readonly(coordinates)
        parameters = components = expected = value = None
        status = "infeasible_coordinates"
        if self.coordinates.feasible(x):
            try:
                parameters = self.coordinates.decode(x)
            except (ValueError, FloatingPointError, OverflowError):
                status = "unrepresentable_parameters"
            else:
                components, status = self._model(parameters, phase, objective=True)
                if components is not None:
                    with np.errstate(over="ignore", invalid="ignore"):
                        expected = components.sum(axis=0) + parameters.background
                    if not np.all(np.isfinite(expected)):
                        status = "nonfinite_expectation"
                    else:
                        # The existing stable centered objective preserves exact zeros.
                        self.work.objective_calls += 1
                        self.work.phases[phase]["objective_calls"] += 1
                        value = 0.5 * calculate_poisson_deviance(
                            self.inputs.targets, expected
                        )
                        if np.isfinite(value):
                            status = "finite"
                        elif np.any((self.inputs.targets > 0) & (expected == 0)):
                            status = "infinite_objective"
                        else:
                            status = "objective_overflow"
        result = _Evaluation(
            self._evaluation_id,
            x,
            parameters,
            expected,
            components,
            value,
            status,
            phase,
            origin,
            self._identity,
        )
        if not result.finite:
            self.record_failure(result, result.status)
        # Validation/assessment/diagnostics never silently select a candidate.
        if (
            result.finite
            and phase in ("initialization", "optimization")
            and (self.best_finite is None or value < self.best_finite.centered_nll)
        ):
            self.best_finite = result
        if self.evaluation_observer is not None:
            self.evaluation_observer(result)
        return result


@dataclass(frozen=True)
class _ShiftGeometry:
    region: str
    distance_to_next_knot_ns: float
    discontinuous_left: bool
    discontinuous_right: bool


def _shift_geometry(time, kernel, shift):
    """Use actual interpolation abscissae, never round shift / nominal dt."""
    q = time - shift
    left_index = np.searchsorted(time, q, side="left")
    right_index = np.searchsorted(time, q, side="right")
    at = left_index != right_index
    left_jump = bool(np.any((q == time[-1]) & (kernel[-1] != 0)))
    right_jump = bool(np.any((q == time[0]) & (kernel[0] != 0)))
    slopes = np.diff(kernel) / np.diff(time)
    left_slopes = np.zeros(q.size)
    right_slopes = np.zeros(q.size)
    inside_left = (left_index > 0) & (left_index < time.size)
    inside_right = (right_index > 0) & (right_index < time.size)
    left_slopes[inside_left] = slopes[left_index[inside_left] - 1]
    right_slopes[inside_right] = slopes[right_index[inside_right] - 1]
    if left_jump or right_jump:
        region = "support_discontinuity"
    elif np.any(at):
        region = (
            "continuous_knot_agreeing_slopes"
            if np.array_equal(left_slopes, right_slopes)
            else "continuous_nondifferentiable_knot"
        )
    else:
        region = "smooth"
    distances = []
    previous = left_index - 1
    following = right_index
    mask = previous >= 0
    distances.extend((q[mask] - time[previous[mask]]).tolist())
    mask = following < time.size
    distances.extend((time[following[mask]] - q[mask]).tolist())
    positive = [d for d in distances if d > 0]
    return _ShiftGeometry(region, min(positive, default=np.inf), left_jump, right_jump)


@dataclass(frozen=True, eq=False)
class _ModelDerivatives:
    objective: _DerivativeEvaluation
    jacobian: object
    column_errors: object
    columns: tuple = ()
    errors: tuple = ()


def _shape_side(problem, base, index, sign, step, policy, phase):
    """Three-point differences using actual decoded displacements, not nominal h.

    Shift displacements are divided by the declared scale; lifetime displacements
    are measured in log coordinates. No physical endpoint is moved or snapped.
    Only the latest twelve scalar probe records are retained per side.
    """
    x = base.coordinates
    previous = {}
    trace = deque(maxlen=12)
    attempted = 0
    reason = "unstable_stencil"
    p = problem.coordinates
    shift = index == p.dimension - 1
    physical_base = (
        base.parameters.temporal_shift_ns
        if shift
        else base.parameters.lifetimes_ns[index - p.n_components]
    )
    room = x[index] - p.lower[index] if sign < 0 else p.upper[index] - x[index]

    def finish(column, error, status, h, refinements):
        return (
            column,
            error,
            status,
            h,
            _SideStencil(
                sign, float(room), step, status, refinements, tuple(trace), attempted
            ),
        )

    for reduction in range(policy.max_reductions):
        h = step / 2**reduction
        values = []
        displacements = []
        for multiple in (0.5, 1.0, 2.0):
            offset = sign * multiple * h
            trial = x.copy()
            trial[index] += offset
            attempted += 1
            actual = float(trial[index] - x[index])
            physical = delta = effective = None
            result = None
            status = "unrepresentable_stencil"
            if trial[index] == x[index] or not problem.coordinates.feasible(trial):
                trace.append(
                    _StencilPoint(
                        reduction,
                        offset,
                        float(trial[index]),
                        actual,
                        None,
                        None,
                        None,
                        None,
                        status,
                    )
                )
                return finish(None, None, status, h, reduction + 1)
            key = float(trial[index])
            if key not in previous:
                result = problem.evaluate(
                    trial, phase=phase, origin="derivative_stencil"
                )
                previous[key] = result
            result = previous[key]
            if result.parameters is not None:
                physical = (
                    result.parameters.temporal_shift_ns
                    if shift
                    else result.parameters.lifetimes_ns[index - p.n_components]
                )
                delta = physical - physical_base
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    effective = (
                        delta / p.shift_scale
                        if shift
                        else float(np.log1p(delta / physical_base))
                    )
            if not result.finite:
                status = result.status
            elif (
                effective is not None
                and np.isfinite(effective)
                and sign * effective > 0
                and np.isclose(effective, actual, rtol=1e-6, atol=0)
                and (not displacements or abs(effective) > abs(displacements[-1]))
            ):
                status = "finite"
            trace.append(
                _StencilPoint(
                    reduction,
                    offset,
                    float(trial[index]),
                    actual,
                    physical,
                    delta,
                    effective,
                    result.evaluation_id,
                    status,
                )
            )
            if status != "finite":
                return finish(None, None, status, h, reduction + 1)
            values.append(result.expected_counts)
            displacements.append(effective)
        half, full, twice = values
        dh, df, dt = displacements
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coarse = (
                (dt / df) * (full - base.expected_counts)
                - (df / dt) * (twice - base.expected_counts)
            ) / (dt - df)
            fine = (
                (df / dh) * (half - base.expected_counts)
                - (dh / df) * (full - base.expected_counts)
            ) / (df - dh)
            error = np.abs(fine - coarse)
        if (
            np.all(np.isfinite(fine))
            and np.all(np.isfinite(error))
            and np.all(
                error
                <= policy.absolute_tolerance
                + policy.relative_tolerance * np.maximum(np.abs(fine), np.abs(coarse))
            )
        ):
            return finish(fine, error, "available", h, reduction + 1)
    return finish(None, None, reason, h, policy.max_reductions)


def _model_derivatives(
    problem: _FixedKProblem,
    base: _Evaluation,
    policy: _DerivativePolicy,
    *,
    phase,
    active_indices=None,
):
    """Analytic linear columns; safeguarded nonlinear columns and sided scores."""
    p = problem.coordinates
    active = set(range(p.dimension)) if active_indices is None else set(active_indices)
    if not active or any(
        type(i) is not int or not 0 <= i < p.dimension for i in active
    ):
        raise ValueError("active indices must be nonempty valid coordinate indices")
    unavailable = _CoordinateDerivative(None, None, None, "invalid_base", None)
    if (
        base.problem_id is not problem._identity
        or not base.finite
        or not p.feasible(base.coordinates)
        or p.decode(base.coordinates) != base.parameters
    ):
        problem.record_failure(
            base, "invalid_derivative_anchor", active_indices=sorted(active)
        )
        return _ModelDerivatives(
            _DerivativeEvaluation(
                base, (unavailable,) * p.dimension, "unavailable", False
            ),
            None,
            None,
        )
    y, mu = problem.inputs.targets, base.expected_counts
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        score = np.ones_like(mu)
        positive = y > 0
        score[positive] -= y[positive] / mu[positive]
    k = p.n_components
    columns = [None] * p.dimension
    errors = [None] * p.dimension
    descriptions = [unavailable] * p.dimension
    geometry = _shift_geometry(
        problem.inputs.time_ns, problem.inputs.kernel, base.parameters.temporal_shift_ns
    )
    # A charged unit-amplitude basis avoids division by zero/tiny fitted A.
    unit = _FixedKParameters(
        (p.amplitude_scale,) * k,
        base.parameters.lifetimes_ns,
        0,
        base.parameters.temporal_shift_ns,
    )
    problem.work.request(phase)
    basis, failure = problem._model(unit, phase, objective=False)
    for index in (*range(k), p.dimension - 2):
        if index not in active:
            descriptions[index] = _CoordinateDerivative(
                None, None, None, "fixed_coordinate", None
            )
            continue
        column = (
            np.full(mu.size, p.background_scale)
            if index == p.dimension - 2
            else (basis[index] if basis is not None else None)
        )
        if column is None:
            descriptions[index] = _CoordinateDerivative(None, None, None, failure, None)
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(score @ column)
        columns[index], errors[index] = column, np.zeros_like(mu)
        descriptions[index] = _CoordinateDerivative(
            value if np.isfinite(value) else None,
            None,
            None,
            "analytic" if np.isfinite(value) else "nonfinite_score",
            None,
        )
    for index in (*range(k, 2 * k), p.dimension - 1):
        if index not in active:
            descriptions[index] = _CoordinateDerivative(
                None, None, None, "fixed_coordinate", None
            )
            continue
        shift = index == p.dimension - 1
        if shift and not any(base.parameters.amplitudes):
            if geometry.region == "support_discontinuity":
                # The shift partial is zero at A=0, but joint A/shift paths can
                # cross a jump with an O(A) change: no joint derivative exists.
                descriptions[index] = _CoordinateDerivative(
                    None, 0, 0, "zero_signal_at_support_discontinuity", None
                )
                continue
            columns[index] = errors[index] = np.zeros_like(mu)
            descriptions[index] = _CoordinateDerivative(0, 0, 0, "zero_signal", None)
            continue
        if not shift and base.parameters.amplitudes[index - k] == 0:
            columns[index] = errors[index] = np.zeros_like(mu)
            descriptions[index] = _CoordinateDerivative(0, 0, 0, "zero_component", None)
            continue
        h = policy.relative_step * max(1.0, abs(base.coordinates[index]))
        if shift:
            h = min(h, geometry.distance_to_next_knot_ns / (4 * p.shift_scale))
        sides = []
        for sign in (-1, 1):
            room = (
                (base.coordinates[index] - p.lower[index])
                if sign < 0
                else (p.upper[index] - base.coordinates[index])
            )
            blocked_jump = shift and (
                geometry.discontinuous_left
                if sign < 0
                else geometry.discontinuous_right
            )
            if room == 0 or blocked_jump:
                why = "bound" if room == 0 else "discontinuous_side"
                sides.append(
                    (None, None, why, None, _SideStencil(sign, float(room), h, why))
                )
            elif room < 4 * h:
                # A near bound is feasible, not an exact active face. Do not
                # collapse the stencil into roundoff merely to use both sides.
                # The other side must independently pass its two-scale check.
                why = "insufficient_bound_room"
                sides.append(
                    (None, None, why, None, _SideStencil(sign, float(room), h, why))
                )
            else:
                sides.append(
                    _shape_side(
                        problem, base, index, sign, min(h, room / 4), policy, phase
                    )
                )
        left, right = sides
        directional = []
        for side in sides:
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(score @ side[0]) if side[0] is not None else None
            directional.append(
                value if value is not None and np.isfinite(value) else None
            )
        available = [s for s in sides if s[0] is not None]
        status = "unavailable_stencil"
        column = error = None
        if shift and geometry.region == "support_discontinuity":
            status = "support_discontinuity"
        elif shift and geometry.region == "continuous_nondifferentiable_knot":
            status = "unequal_one_sided_derivatives"
        elif len(available) == 2:
            if np.all(
                np.abs(left[0] - right[0])
                <= policy.absolute_tolerance
                + policy.relative_tolerance
                * np.maximum(np.abs(left[0]), np.abs(right[0]))
            ):
                column = 0.5 * left[0] + 0.5 * right[0]
                error = np.maximum(left[1], right[1]) + 0.5 * np.abs(left[0] - right[0])
                status = "safeguarded"
            else:
                status = "unequal_one_sided_derivatives"
        elif len(available) == 1 and any(s[2] == "bound" for s in sides):
            column, error = available[0][:2]
            status = "one_sided_bound"
        elif (
            len(available) == 1
            and any(s[2] == "insufficient_bound_room" for s in sides)
            and (not shift or geometry.region == "smooth")
        ):
            column, error = available[0][:2]
            status = "feasible_side_near_bound"
        elif not available:
            status = ";".join(s[2] for s in sides)
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(score @ column) if column is not None else None
        if value is not None and not np.isfinite(value):
            value, status = None, "nonfinite_score"
        columns[index], errors[index] = column, error
        descriptions[index] = _CoordinateDerivative(
            value,
            *directional,
            status,
            h,
            left[2],
            right[2],
            left[3],
            right[3],
            left[4],
            right[4],
        )
    complete = (
        all(d.value is not None for d in descriptions)
        and not problem.work.deadline_reached
    )
    jacobian = (
        _readonly(np.column_stack(columns))
        if all(c is not None for c in columns)
        else None
    )
    error_array = _readonly(np.column_stack(errors)) if jacobian is not None else None
    if any(columns[i] is None or descriptions[i].value is None for i in active):
        problem.record_failure(
            base,
            "derivative_unavailable",
            derivatives=descriptions,
            active_indices=sorted(active),
        )
    return _ModelDerivatives(
        _DerivativeEvaluation(base, tuple(descriptions), geometry.region, complete),
        jacobian,
        error_array,
        tuple(_readonly(c) if c is not None else None for c in columns),
        tuple(_readonly(e) if e is not None else None for e in errors),
    )


def _objective_derivatives(problem, base, policy, *, phase, active_indices=None):
    return _model_derivatives(
        problem, base, policy, phase=phase, active_indices=active_indices
    ).objective
