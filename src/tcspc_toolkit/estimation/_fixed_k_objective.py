"""Counted fixed-K model/objective and derivatives, independent of any solver."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        elif self.phases[phase]["model_calls"] >= limits.phase_model_calls[phase]:
            reason = "phase_model_limit"
        elif self.model_calls >= limits.model_calls:
            reason = "model_limit"
        elif objective and self.objective_attempts >= limits.objective_calls:
            reason = "objective_limit"
        elif self.charged_component_convolutions + k > limits.component_convolutions:
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
        self._evaluation_id = 0
        self._identity = object()

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
        # Validation/assessment/diagnostics never silently select a candidate.
        if (
            result.finite
            and phase in ("initialization", "optimization")
            and (self.best_finite is None or value < self.best_finite.centered_nll)
        ):
            self.best_finite = result
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


def _shape_side(problem, base, index, sign, step, policy, phase):
    """Three-point one-sided differences at h and h/2, bounded refinements."""
    x = base.coordinates
    previous = {}
    reason = "unstable_stencil"
    for reduction in range(policy.max_reductions):
        h = step / 2**reduction
        values = []
        for multiple in (0.5, 1.0, 2.0):
            offset = sign * multiple * h
            trial = x.copy()
            trial[index] += offset
            if trial[index] == x[index] or not problem.coordinates.feasible(trial):
                return None, None, "unrepresentable_stencil", h
            if not np.isclose(trial[index] - x[index], offset, rtol=1e-6, atol=0):
                return None, None, "unrepresentable_stencil", h
            key = float(trial[index])
            if key not in previous:
                result = problem.evaluate(
                    trial, phase=phase, origin="derivative_stencil"
                )
                previous[key] = result
            result = previous[key]
            if not result.finite:
                return None, None, result.status, h
            # Distinct coordinates can decode to the same floating-point physics.
            if result.parameters == base.parameters:
                return None, None, "unrepresentable_stencil", h
            if index == problem.coordinates.dimension - 1 and not np.isclose(
                result.parameters.temporal_shift_ns - base.parameters.temporal_shift_ns,
                offset * problem.coordinates.shift_scale,
                rtol=1e-6,
                atol=0,
            ):
                return None, None, "unrepresentable_stencil", h
            values.append(result.expected_counts)
        half, full, twice = values
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coarse = (
                sign
                * (4 * (full - base.expected_counts) - (twice - base.expected_counts))
                / (2 * h)
            )
            fine = (
                sign
                * (4 * (half - base.expected_counts) - (full - base.expected_counts))
                / h
            )
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
            return fine, error, "available", h
    return None, None, reason, step / 2 ** (policy.max_reductions - 1)


def _model_derivatives(
    problem: _FixedKProblem, base: _Evaluation, policy: _DerivativePolicy, *, phase
):
    """Analytic linear columns; safeguarded nonlinear columns and sided scores."""
    p = problem.coordinates
    unavailable = _CoordinateDerivative(None, None, None, "invalid_base", None)
    if (
        base.problem_id is not problem._identity
        or not base.finite
        or not p.feasible(base.coordinates)
        or p.decode(base.coordinates) != base.parameters
    ):
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
                sides.append(
                    (None, None, "bound" if room == 0 else "discontinuous_side", None)
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
        elif not available:
            status = ";".join(s[2] for s in sides)
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(score @ column) if column is not None else None
        if value is not None and not np.isfinite(value):
            value, status = None, "nonfinite_score"
        columns[index], errors[index] = column, error
        descriptions[index] = _CoordinateDerivative(
            value, *directional, status, h, left[2], right[2], left[3], right[3]
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
    return _ModelDerivatives(
        _DerivativeEvaluation(base, tuple(descriptions), geometry.region, complete),
        jacobian,
        error_array,
    )


def _objective_derivatives(problem, base, policy, *, phase):
    return _model_derivatives(problem, base, policy, phase=phase).objective
