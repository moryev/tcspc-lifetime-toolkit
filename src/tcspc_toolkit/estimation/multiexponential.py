"""Constrained bi-exponential Poisson reconvolution with conditional diagnostics.

This module deliberately adds no root or package-facade exports. Numerical
acceptance, local conditioning, structural degeneracy and physical validity
are different questions. None of the returned diagnostics certifies the latter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, nnls

from tcspc_toolkit.evaluation.diagnostics import (
    PoissonFitStatistics,
    calculate_poisson_deviance,
    calculate_poisson_fit_statistics,
)
from tcspc_toolkit.irf.preparation import IRFPreparationDiagnostics, PreparedIRF
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
)
from tcspc_toolkit.preprocessing.histograms import (
    time_axes_compatible,
    validate_time_axis,
)


FloatArray = NDArray[np.float64]
InitializationPolicy = Literal["economical", "adaptive", "exhaustive"]
PARAMETER_NAMES = (
    "amplitude_1",
    "amplitude_2",
    "log_lifetime_1",
    "log_lifetime_2",
    "background",
    "shift",
)


def _readonly(values: ArrayLike) -> FloatArray:
    if np.iscomplexobj(values):
        raise ValueError("values must be real")
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BiexponentialParameters:
    """Pre-exponential amplitudes, ns lifetimes/shift, and background per bin."""

    amplitudes: tuple[float, float]
    lifetimes_ns: tuple[float, float]
    background: float = 0.0
    temporal_shift_ns: float = 0.0

    def __post_init__(self) -> None:
        if any(
            np.iscomplexobj(value)
            for value in (
                self.amplitudes,
                self.lifetimes_ns,
                self.background,
                self.temporal_shift_ns,
            )
        ):
            raise ValueError("parameters must be real")
        a = np.asarray(self.amplitudes, dtype=float)
        t = np.asarray(self.lifetimes_ns, dtype=float)
        if a.shape != (2,) or t.shape != (2,):
            raise ValueError("exactly two amplitudes and lifetimes are required")
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError("amplitudes must be finite and non-negative")
        if not np.all(np.isfinite(t)) or np.any(t <= 0):
            raise ValueError("lifetimes must be finite and positive")
        if not np.isfinite(self.background) or self.background < 0:
            raise ValueError("background must be finite and non-negative")
        if not np.isfinite(self.temporal_shift_ns):
            raise ValueError("temporal shift must be finite")
        object.__setattr__(self, "amplitudes", tuple(map(float, a)))
        object.__setattr__(self, "lifetimes_ns", tuple(map(float, t)))
        object.__setattr__(self, "background", float(self.background))
        object.__setattr__(self, "temporal_shift_ns", float(self.temporal_shift_ns))


@dataclass(frozen=True)
class BiexponentialFitSettings:
    """Explicit deterministic work limits and descriptive warning thresholds.

    Evaluation limits count actual shared forward-model calls, including finite
    differences. Initialization, optimization, validation and diagnostics each
    have their own hard limit. Defaults were selected on the declared development
    observations only; see the Issue-28 numerical development record.
    """

    policy: InitializationPolicy = "exhaustive"
    max_starts: int = 46
    max_objective_evaluations: int = 56000
    max_initialization_evaluations: int = 128
    max_validation_evaluations: int = 100
    max_diagnostic_evaluations: int = 100
    max_iterations: int = 1000
    ftol: float = 1e-12
    gtol: float = 1e-6
    finite_difference_step: float = 1e-5
    descent_tolerance: float = 0.01
    near_optimal_nll: float = 1.0
    near_optimal_lifetime_ratio: float = 1.1
    near_optimal_fraction_span: float = 0.1
    weak_fraction: float = 0.01
    similar_lifetime_ratio: float = 1.1
    boundary_tolerance: float = 1e-3
    condition_limit: float = 1e8
    derivative_relative_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.policy not in ("economical", "adaptive", "exhaustive"):
            raise ValueError("unknown initialization policy")
        for name in (
            "max_starts",
            "max_objective_evaluations",
            "max_initialization_evaluations",
            "max_validation_evaluations",
            "max_diagnostic_evaluations",
            "max_iterations",
        ):
            value = getattr(self, name)
            minimum = 1 if name in ("max_starts", "max_iterations") else 0
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))
        for name in (
            "ftol",
            "gtol",
            "finite_difference_step",
            "descent_tolerance",
            "near_optimal_nll",
            "near_optimal_lifetime_ratio",
            "near_optimal_fraction_span",
            "weak_fraction",
            "similar_lifetime_ratio",
            "boundary_tolerance",
            "condition_limit",
            "derivative_relative_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.weak_fraction >= 0.5
            or self.similar_lifetime_ratio < 1
            or self.condition_limit < 1
            or self.boundary_tolerance >= 0.5
            or self.near_optimal_lifetime_ratio < 1
            or self.near_optimal_fraction_span > 1
        ):
            raise ValueError("invalid diagnostic threshold range")


@dataclass(frozen=True)
class BiexponentialFitAssumptions:
    lifetime_bounds_ns: tuple[float, float]
    temporal_shift_bounds_ns: tuple[float, float]
    settings: BiexponentialFitSettings
    parameter_scales: tuple[float, ...]
    irf_input_kind: str
    irf_area: float
    preparation_diagnostics: IRFPreparationDiagnostics | None
    source_kind: str | None
    source_metadata: Mapping[str, Any]
    source_provenance: Mapping[str, Any]
    target_kind: str


@dataclass(frozen=True)
class BiexponentialStartDiagnostics:
    start_index: int
    kind: str
    initial_parameters: BiexponentialParameters
    best_parameters: BiexponentialParameters | None
    best_centered_nll: float
    endpoint_centered_nll: float | None
    optimizer_reported_success: bool | None
    status: int | None
    message: str
    objective_evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class BiexponentialOptimizationDiagnostics:
    starts: tuple[BiexponentialStartDiagnostics, ...]
    starts_screened: int
    selected_start_index: int | None
    selected_candidate_source: str | None
    optimizer_reported_success: bool | None
    numerical_validation_passed: bool | None
    max_detectable_descent_nll: float | None
    validation_anchor_parameters: BiexponentialParameters | None
    stopping_reason: str
    exhausted_budgets: tuple[str, ...]
    initialization_evaluations: int
    objective_evaluations: int
    validation_evaluations: int
    diagnostic_evaluations: int
    initialization_seconds: float
    optimization_seconds: float
    validation_seconds: float
    diagnostic_seconds: float
    total_seconds: float
    initialization_rejected_candidates: int
    initialization_scaling_failed: bool


@dataclass(frozen=True)
class BiexponentialComponentSummary:
    amplitude_fractions: tuple[float, float] | None
    detected_signal_photons: tuple[float, float]
    detected_signal_fractions: tuple[float, float] | None
    ideal_photon_fractions: tuple[float, float] | None
    amplitude_weighted_lifetime_ns: float | None
    detected_signal_weighted_lifetime_ns: float | None
    ideal_photon_weighted_lifetime_ns: float | None


@dataclass(frozen=True)
class BiexponentialIdentifiabilityDiagnostics:
    """Conditional descriptions, never a physical-identifiability certificate."""

    structural_warnings: tuple[str, ...]
    conditioning_warnings: tuple[str, ...]
    boundary_warnings: tuple[str, ...]
    multistart_warnings: tuple[str, ...]
    lifetime_difference_ns: float | None
    lifetime_ratio: float | None
    boundary_distances: tuple[float, ...] | None
    weighted_jacobian_singular_values: tuple[float, ...] | None
    weighted_jacobian_rank: int | None
    weighted_jacobian_condition: float | None
    jacobian_unavailable_reason: str | None
    jacobian_relative_step: float
    jacobian_rank_tolerance: float | None
    near_optimal_solution_count: int
    near_optimal_lifetime_ranges_ns: tuple[tuple[float, float], ...] | None
    near_optimal_detected_fraction_ranges: tuple[tuple[float, float], ...] | None


@dataclass(frozen=True)
class BiexponentialFitResult:
    parameters: BiexponentialParameters | None
    expected_counts: FloatArray | None
    component_counts: FloatArray | None
    components: BiexponentialComponentSummary | None
    statistics: PoissonFitStatistics | None
    optimization: BiexponentialOptimizationDiagnostics
    assumptions: BiexponentialFitAssumptions
    identifiability: BiexponentialIdentifiabilityDiagnostics


def _bounds(
    values: tuple[float, float], name: str, *, positive: bool = False
) -> tuple[float, float]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (2,)
        or not np.all(np.isfinite(array))
        or array[0] >= array[1]
        or (positive and array[0] <= 0)
    ):
        raise ValueError(
            f"{name} must be a finite increasing pair"
            + (" with positive lower bound" if positive else "")
        )
    if positive and np.log(array[0]) >= np.log(array[1]):
        raise ValueError(f"{name} must be distinguishable in log coordinates")
    return float(array[0]), float(array[1])


def _canonical(
    parameters: BiexponentialParameters,
) -> tuple[BiexponentialParameters, NDArray[np.int64]]:
    order = np.lexsort((-np.asarray(parameters.amplitudes), parameters.lifetimes_ns))
    return BiexponentialParameters(
        tuple(parameters.amplitudes[i] for i in order),
        tuple(parameters.lifetimes_ns[i] for i in order),
        parameters.background,
        parameters.temporal_shift_ns,
    ), order


class _BudgetExceeded(Exception):
    pass


class _Problem:
    """One kernel/objective path; explicit accounting for every forward call."""

    def __init__(self, time, targets, kernel, lifetime_bounds, shift_bounds, settings):
        self.time, self.targets, self.kernel = time, targets, kernel
        self.lifetime_bounds, self.shift_bounds, self.settings = (
            lifetime_bounds,
            shift_bounds,
            settings,
        )
        self.counts = dict(
            initialization=0, optimization=0, validation=0, diagnostics=0
        )
        self.seconds = dict.fromkeys(self.counts, 0.0)
        self.limits = dict(
            initialization=settings.max_initialization_evaluations,
            optimization=settings.max_objective_evaluations,
            validation=settings.max_validation_evaluations,
            diagnostics=settings.max_diagnostic_evaluations,
        )
        self.exhausted: set[str] = set()
        self.log_mid = 0.5 * (np.log(lifetime_bounds[0]) + np.log(lifetime_bounds[1]))
        self.shift_mid = 0.5 * shift_bounds[0] + 0.5 * shift_bounds[1]
        self.shift_scale = 0.5 * shift_bounds[1] - 0.5 * shift_bounds[0]
        if not np.isfinite(self.shift_scale) or self.shift_scale <= 0:
            raise ValueError("shift bounds cannot be represented as a finite interval")
        self.a_scale = max(float(np.sum(targets)), 1.0)
        self.b_scale = max(float(np.mean(targets)), 1.0)
        self.lower = np.array(
            [
                0,
                0,
                np.log(lifetime_bounds[0]) - self.log_mid,
                np.log(lifetime_bounds[0]) - self.log_mid,
                0,
                -1.0,
            ]
        )
        self.upper = np.array(
            [
                np.inf,
                np.inf,
                np.log(lifetime_bounds[1]) - self.log_mid,
                np.log(lifetime_bounds[1]) - self.log_mid,
                np.inf,
                1.0,
            ]
        )
        self.best_value = np.inf
        self.best_x = None
        self.best_curve = None
        self.best_components = None
        self.best_start = None
        self.best_source = None
        self.active_start = None
        self.local_value = np.inf
        self.local_x = None
        self.starts_screened = 0
        self.initialization_rejected_candidates = 0
        self.initialization_scaling_failed = False
        self.validation_anchor = None

    def physical(self, x) -> BiexponentialParameters:
        tau = np.clip(np.exp(self.log_mid + np.asarray(x)[2:4]), *self.lifetime_bounds)
        shift = np.clip(self.shift_mid + self.shift_scale * x[5], *self.shift_bounds)
        return BiexponentialParameters(
            tuple(self.a_scale * np.asarray(x)[:2]),
            tuple(tau),
            self.b_scale * x[4],
            shift,
        )

    def coordinates(self, p: BiexponentialParameters) -> FloatArray:
        return np.array(
            [
                p.amplitudes[0] / self.a_scale,
                p.amplitudes[1] / self.a_scale,
                np.log(p.lifetimes_ns[0]) - self.log_mid,
                np.log(p.lifetimes_ns[1]) - self.log_mid,
                p.background / self.b_scale,
                (p.temporal_shift_ns - self.shift_mid) / self.shift_scale,
            ]
        )

    def feasible(self, x) -> bool:
        return bool(
            np.shape(x) == (6,)
            and np.all(np.isfinite(x))
            and np.all(x >= self.lower)
            and np.all(x <= self.upper)
        )

    def model(self, p: BiexponentialParameters, phase: str):
        if self.counts[phase] >= self.limits[phase]:
            self.exhausted.add(phase)
            raise _BudgetExceeded(phase)
        self.counts[phase] += 1
        components = multiexponential_reconvolution_component_counts(
            self.time,
            self.kernel,
            amplitudes=p.amplitudes,
            lifetimes=p.lifetimes_ns,
            temporal_shift=p.temporal_shift_ns,
        )
        return components.sum(axis=0) + p.background, components

    def evaluate(self, x, phase="optimization") -> float:
        if not self.feasible(x):
            return np.inf
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                curve, components = self.model(self.physical(x), phase)
                value = 0.5 * calculate_poisson_deviance(self.targets, curve)
        except (ValueError, FloatingPointError):
            return np.inf
        if phase in ("initialization", "optimization", "validation") and np.isfinite(
            value
        ):
            if phase != "validation" and value < self.local_value:
                self.local_value, self.local_x = value, np.array(x, copy=True)
            if value < self.best_value:
                self.best_value, self.best_x = value, np.array(x, copy=True)
                self.best_curve, self.best_components = curve.copy(), components.copy()
                self.best_start = self.active_start if phase == "optimization" else None
                self.best_source = phase
        return value

    def value_and_gradient(self, x):
        """Bound-aware differences without differencing infinite objectives.

        A line-search trial with an impossible expectation has infinite cost.
        Its placeholder gradient is never accepted as a stationarity check;
        the infinite cost forces backtracking. At finite points, use actual
        finite objective differences, reducing the step if the domain requires
        it. All probes are charged to the optimization budget.
        """
        value = self.evaluate(x)
        gradient = np.zeros(6)
        if not np.isfinite(value):
            return value, gradient
        for index in range(6):
            h = self.settings.finite_difference_step * max(1.0, abs(x[index]))
            for _ in range(8):
                left, right = x[index] - self.lower[index], self.upper[index] - x[index]
                if min(left, right) >= h:
                    plus, minus = x.copy(), x.copy()
                    plus[index] += h
                    minus[index] -= h
                    high, low = self.evaluate(plus), self.evaluate(minus)
                    derivative = (high - low) / (2 * h)
                else:
                    sign = 1 if right >= left else -1
                    step = min(h, (right if sign == 1 else left) / 2)
                    first, second = x.copy(), x.copy()
                    first[index] += sign * step
                    second[index] += sign * 2 * step
                    f1, f2 = self.evaluate(first), self.evaluate(second)
                    derivative = sign * (-3 * value + 4 * f1 - f2) / (2 * step)
                if np.isfinite(derivative):
                    gradient[index] = derivative
                    break
                h *= 0.1
            else:
                # This is a failed derivative, not evidence of convergence.
                # Keep the best candidate and let independent validation fail.
                return value, np.full(6, np.nan)
        return value, gradient


def _initial_points(problem: _Problem, guesses) -> list[FloatArray]:
    started = perf_counter()
    candidates: list[tuple[float, FloatArray]] = []
    seen = set()

    def model(p):
        # Raise at the numerical operation, before downstream array validators
        # turn overflow into ValueError. Unrelated validation/programming errors
        # must propagate. The normal model path charges every attempted call.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            curve, components = problem.model(p, "initialization")
        if not np.all(np.isfinite(curve)) or not np.all(np.isfinite(components)):
            raise FloatingPointError("nonfinite initialization prediction")
        return curve, components

    def add(p):
        p, _ = _canonical(p)
        x = problem.coordinates(p)
        # The logarithm/affine round trip may put a bound a few ulps outside.
        x = np.clip(x, problem.lower, problem.upper)
        key = tuple(x)
        if key not in seen:
            seen.add(key)
            value = problem.evaluate(x, "initialization")
            candidates.append((value, x))
            problem.starts_screened += 1
            if not np.isfinite(value):
                problem.initialization_rejected_candidates += 1

    try:
        reference = BiexponentialParameters(
            (1, 0),
            (np.exp(problem.log_mid), np.exp(problem.log_mid)),
            0,
            problem.shift_mid,
        )
        try:
            ref, _ = model(reference)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                area = float(np.sum(ref))
            if np.isfinite(area) and area > 0:
                problem.a_scale /= area
        except FloatingPointError:
            # Scaling is optional; retain the original finite count scale.
            # This is a charged reference evaluation, not a screened candidate.
            problem.initialization_scaling_failed = True
        for guess in guesses:
            add(guess)
        explicit_count = len(candidates)
        tau = np.exp(
            np.linspace(
                np.log(problem.lifetime_bounds[0]),
                np.log(problem.lifetime_bounds[1]),
                5,
            )
        )
        tau[[0, -1]] = problem.lifetime_bounds
        for shift in (
            problem.shift_bounds[0],
            problem.shift_mid,
            problem.shift_bounds[1],
        ):
            for i in range(5):
                for j in range(i, 5):
                    p = BiexponentialParameters((1, 1), (tau[i], tau[j]), 0, shift)
                    try:
                        _, components = model(p)
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            design = np.column_stack(
                                (components.T, np.ones(problem.time.size))
                            )
                            scales = np.linalg.norm(design, axis=0)
                            scales[scales == 0] = 1
                            y_scale = max(float(np.max(problem.targets)), 1)
                            try:
                                coefficients, _ = nnls(
                                    design / scales,
                                    problem.targets / y_scale,
                                    maxiter=100,
                                )
                                coefficients = coefficients / scales * y_scale
                            except (RuntimeError, np.linalg.LinAlgError):
                                coefficients = np.array(
                                    [0, 0, np.mean(problem.targets)]
                                )
                            mu = design @ coefficients
                        if not np.all(np.isfinite(coefficients)) or not np.all(
                            np.isfinite(mu)
                        ):
                            raise FloatingPointError(
                                "nonfinite initialization projection"
                            )
                    except FloatingPointError:
                        problem.starts_screened += 1
                        problem.initialization_rejected_candidates += 1
                        continue
                    # This is a finite starting point, not a constraint or an
                    # expectation floor. Exact zero-background seeds survive
                    # whenever their Poisson objective is finite.
                    if np.any((problem.targets > 0) & (mu == 0)):
                        coefficients[2] = max(coefficients[2], 0.01 * problem.b_scale)
                    add(
                        BiexponentialParameters(
                            tuple(coefficients[:2]),
                            (tau[i], tau[j]),
                            coefficients[2],
                            shift,
                        )
                    )
        add(
            BiexponentialParameters(
                (0, 0),
                (np.exp(problem.log_mid), np.exp(problem.log_mid)),
                float(np.mean(problem.targets)),
                problem.shift_mid,
            )
        )
    except _BudgetExceeded:
        explicit_count = min(len(guesses), len(candidates))
    finally:
        problem.seconds["initialization"] += perf_counter() - started
    ranked = sorted(
        candidates[explicit_count:], key=lambda item: (item[0], tuple(item[1]))
    )
    ordered = candidates[:explicit_count]
    # Give the economical prefix shift coverage, then append the other seeds
    # by actual Poisson objective. No generating parameter is consulted.
    for shift in (-1.0, 0.0, 1.0):
        match = next((item for item in ranked if item[1][5] == shift), None)
        if match is not None:
            ordered.append(match)
            ranked = [item for item in ranked if item is not match]
    ordered.extend(ranked)
    return [x for value, x in ordered if np.isfinite(value)]


def _local_validation(problem: _Problem, x):
    """Independent feasible objective probes, including coupled directions."""
    started = perf_counter()
    problem.validation_anchor = _canonical(problem.physical(x))[0]
    best_x = np.array(x, copy=True)
    try:
        baseline = problem.evaluate(x, "validation")
        if not np.isfinite(baseline):
            return False, None, best_x
        best = baseline
        directions = list(np.eye(6))
        directions += [np.array([1, -1, 0, 0, 0, 0]), np.array([0, 0, 1, -1, 0, 0])]
        for step in (1e-4, 1e-3, 1e-2):
            for direction in directions:
                for sign in (-1, 1):
                    trial = np.clip(
                        x + sign * step * direction, problem.lower, problem.upper
                    )
                    if np.array_equal(trial, x):
                        continue
                    value = problem.evaluate(trial, "validation")
                    if value < best:
                        best, best_x = value, trial
        descent = max(0.0, baseline - best)
        return descent <= problem.settings.descent_tolerance, float(descent), best_x
    except _BudgetExceeded:
        return None, None, best_x
    finally:
        problem.seconds["validation"] += perf_counter() - started


def _refine(
    problem: _Problem, x, index: int, kind: str
) -> BiexponentialStartDiagnostics:
    started, initial_count = perf_counter(), problem.counts["optimization"]
    initial_parameters = problem.physical(x)
    problem.active_start = index
    problem.local_value, problem.local_x = np.inf, None
    success, status, message, endpoint = (
        None,
        None,
        "optimization_budget_exhausted",
        None,
    )
    try:
        result = minimize(
            problem.value_and_gradient,
            x,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(problem.lower, problem.upper, strict=True)),
            options=dict(
                ftol=problem.settings.ftol,
                gtol=problem.settings.gtol,
                maxiter=problem.settings.max_iterations,
                maxls=40,
            ),
        )
        success, status, message = (
            bool(result.success),
            int(result.status),
            str(result.message),
        )
        endpoint = problem.evaluate(result.x)
    except _BudgetExceeded:
        pass
    elapsed = perf_counter() - started
    problem.seconds["optimization"] += elapsed
    return BiexponentialStartDiagnostics(
        index,
        kind,
        initial_parameters,
        None
        if problem.local_x is None
        else _canonical(problem.physical(problem.local_x))[0],
        float(problem.local_value),
        endpoint,
        success,
        status,
        message,
        problem.counts["optimization"] - initial_count,
        elapsed,
    )


def _fractions(values) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    largest = float(np.max(values))
    if largest == 0 or not np.isfinite(largest):
        return None
    scaled = values / largest
    return tuple(map(float, scaled / np.sum(scaled)))


def _component_summary(p, components) -> BiexponentialComponentSummary:
    amplitude_fractions = _fractions(p.amplitudes)
    photons = tuple(map(float, components.sum(axis=1)))
    detected_fractions = _fractions(photons)
    # Scale before multiplying to avoid unnecessary A*tau overflow.
    a = np.asarray(p.amplitudes)
    ideal = (
        None
        if not np.any(a)
        else _fractions(
            (a / np.max(a)) * (np.asarray(p.lifetimes_ns) / max(p.lifetimes_ns))
        )
    )

    def average(fractions):
        return None if fractions is None else float(np.dot(fractions, p.lifetimes_ns))

    return BiexponentialComponentSummary(
        amplitude_fractions,
        photons,
        detected_fractions,
        ideal,
        average(amplitude_fractions),
        average(detected_fractions),
        average(ideal),
    )


def _jacobian(problem, x, step):
    base, _ = problem.model(problem.physical(x), "diagnostics")
    columns = []
    for index in range(6):
        h = step * max(1.0, abs(x[index]))
        plus, minus = x.copy(), x.copy()
        plus[index] = min(problem.upper[index], x[index] + h)
        minus[index] = max(problem.lower[index], x[index] - h)
        fplus, _ = problem.model(problem.physical(plus), "diagnostics")
        fminus, _ = problem.model(problem.physical(minus), "diagnostics")
        columns.append((fplus - fminus) / (plus[index] - minus[index]))
    return base, np.column_stack(columns)


def _identifiability(problem, p, component_summary, starts):
    started = perf_counter()
    structural, conditioning, boundary, multistart = [], [], [], []
    singular, rank, condition, reason, rank_tol = None, None, None, None, None
    separation = ratio = distances = ranges = fraction_ranges = None
    near = []
    try:
        if p is None:
            reason = "no_finite_candidate"
        else:
            x = problem.coordinates(p)
            x = np.clip(x, problem.lower, problem.upper)
            separation = p.lifetimes_ns[1] - p.lifetimes_ns[0]
            ratio = p.lifetimes_ns[1] / p.lifetimes_ns[0]
            if min(p.amplitudes) == 0:
                structural.append("zero_component_lifetime_unconstrained")
            if separation == 0:
                structural.append("equal_lifetimes_amplitude_split_unconstrained")
            elif ratio < problem.settings.similar_lifetime_ratio:
                conditioning.append("similar_lifetimes")
            if component_summary.detected_signal_fractions is None:
                structural.append("no_detected_signal_fractions_undefined")
            elif (
                min(component_summary.detected_signal_fractions)
                < problem.settings.weak_fraction
            ):
                conditioning.append("weak_detected_component")
            normalized = x.copy()
            for j in (2, 3, 5):
                normalized[j] = min(
                    x[j] - problem.lower[j], problem.upper[j] - x[j]
                ) / (problem.upper[j] - problem.lower[j])
            distances = tuple(map(float, normalized))
            boundary = [
                name
                for name, distance in zip(PARAMETER_NAMES, distances, strict=True)
                if distance <= problem.settings.boundary_tolerance
            ]
            try:
                mu, jac = _jacobian(problem, x, problem.settings.finite_difference_step)
                _, half_jac = _jacobian(
                    problem, x, 0.5 * problem.settings.finite_difference_step
                )
                norm = np.linalg.norm(half_jac, axis=0)
                difference = np.linalg.norm(jac - half_jac, axis=0)
                dt = problem.time[1] - problem.time[0]
                shift_step = (
                    problem.shift_scale
                    * problem.settings.finite_difference_step
                    * max(1.0, abs(x[5]))
                )
                at_shift_knot = (
                    abs(p.temporal_shift_ns - np.rint(p.temporal_shift_ns / dt) * dt)
                    <= shift_step
                )
                if np.any(mu == 0):
                    reason = "zero_expectation_nonregular_information"
                elif at_shift_knot:
                    # Interpolation is piecewise linear. A central secant at
                    # a knot need not be a derivative even if h and h/2 agree.
                    reason = "shift_interpolation_knot"
                elif np.any(
                    difference
                    > problem.settings.derivative_relative_tolerance
                    * np.maximum(norm, np.finfo(float).eps)
                ):
                    reason = "step_sensitive_jacobian"
                else:
                    weighted = half_jac / np.sqrt(mu[:, None])
                    if not np.all(np.isfinite(weighted)):
                        reason = "nonfinite_weighted_jacobian"
                    else:
                        values = np.linalg.svd(weighted, compute_uv=False)
                        singular = tuple(map(float, values))
                        # Finite differences do not have machine-precision
                        # accuracy. Include the measured step discrepancy so
                        # numerical noise cannot give an exactly degenerate
                        # model an apparently full-rank Jacobian.
                        derivative_error = np.linalg.norm(
                            (jac - half_jac) / np.sqrt(mu[:, None]), ord=2
                        )
                        rank_tol = float(
                            max(
                                max(weighted.shape) * np.finfo(float).eps * values[0],
                                derivative_error,
                            )
                        )
                        rank = int(np.sum(values > rank_tol))
                        condition = (
                            float(values[0] / values[-1]) if rank == 6 else np.inf
                        )
                        if rank < 6:
                            conditioning.append("rank_deficient_weighted_jacobian")
                        elif condition > problem.settings.condition_limit:
                            conditioning.append("ill_conditioned_weighted_jacobian")
            except _BudgetExceeded:
                reason = "diagnostic_budget_exhausted"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                reason = "jacobian_evaluation_failed"
            near = [
                s.best_parameters
                for s in starts
                if s.best_parameters is not None
                and s.best_centered_nll
                <= problem.best_value + problem.settings.near_optimal_nll
            ]
            if near:
                tau = np.array([v.lifetimes_ns for v in near])
                ranges = tuple(
                    (float(lo), float(hi))
                    for lo, hi in zip(tau.min(axis=0), tau.max(axis=0), strict=True)
                )
                if np.any(
                    tau.max(axis=0)
                    > problem.settings.near_optimal_lifetime_ratio * tau.min(axis=0)
                ):
                    multistart.append("near_optimal_lifetime_disagreement")
                fractions = []
                try:
                    for candidate in near:
                        _, components = problem.model(candidate, "diagnostics")
                        value = _fractions(components.sum(axis=1))
                        if value is not None:
                            fractions.append(value)
                except _BudgetExceeded:
                    multistart.append("near_optimal_contributions_budget_exhausted")
                    fractions = []
                if fractions:
                    values = np.asarray(fractions)
                    fraction_ranges = tuple(
                        (float(lo), float(hi))
                        for lo, hi in zip(
                            values.min(axis=0), values.max(axis=0), strict=True
                        )
                    )
                    if np.any(
                        np.ptp(values, axis=0)
                        > problem.settings.near_optimal_fraction_span
                    ):
                        multistart.append("near_optimal_contribution_disagreement")
            if len(near) < 2:
                multistart.append("insufficient_starts_for_agreement_assessment")
        return BiexponentialIdentifiabilityDiagnostics(
            tuple(structural),
            tuple(conditioning),
            tuple(boundary),
            tuple(multistart),
            separation,
            ratio,
            distances,
            singular,
            rank,
            condition,
            reason,
            problem.settings.finite_difference_step,
            rank_tol,
            len(near),
            ranges,
            fraction_ranges,
        )
    finally:
        problem.seconds["diagnostics"] += perf_counter() - started


def _fit_core(
    time_ns,
    targets,
    irf,
    *,
    lifetime_bounds_ns,
    temporal_shift_bounds_ns,
    initial_guesses=None,
    settings=None,
    raw_counts: bool,
):
    started = perf_counter()
    settings = BiexponentialFitSettings() if settings is None else settings
    if not isinstance(settings, BiexponentialFitSettings):
        raise TypeError("settings must be BiexponentialFitSettings")
    if np.iscomplexobj(time_ns):
        raise ValueError("time_ns must be real")
    time = _readonly(validate_time_axis(time_ns))
    y = _readonly(targets)
    if y.shape != time.shape or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError(
            "counts/targets must be finite non-negative values matching time"
        )
    if raw_counts and (np.any(y >= 2**53) or np.any(y != np.rint(y))):
        raise ValueError("raw counts must be exactly integer-valued and below 2**53")
    lifetime_bounds = _bounds(lifetime_bounds_ns, "lifetime_bounds_ns", positive=True)
    shift_bounds = _bounds(temporal_shift_bounds_ns, "temporal_shift_bounds_ns")
    prepared = isinstance(irf, PreparedIRF)
    if prepared and not time_axes_compatible(time, irf.time_ns):
        raise ValueError("PreparedIRF grid does not match time_ns")
    kernel = _readonly(irf.kernel if prepared else irf)
    if (
        kernel.shape != time.shape
        or not np.all(np.isfinite(kernel))
        or np.any(kernel < 0)
    ):
        raise ValueError("IRF must be finite, non-negative, and match time_ns")
    area = float(np.trapezoid(kernel, x=time))
    if not np.isfinite(area) or not np.isclose(area, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("IRF must already have unit trapezoidal area")
    preparation_snapshot = deepcopy(irf.diagnostics) if prepared else None
    source_kind = irf.source.source_kind.value if prepared else None
    metadata_snapshot = MappingProxyType(
        deepcopy(dict(irf.source.metadata)) if prepared else {}
    )
    provenance_snapshot = MappingProxyType(
        deepcopy(dict(irf.source.provenance)) if prepared else {}
    )
    guesses = () if initial_guesses is None else tuple(initial_guesses)
    for guess in guesses:
        if not isinstance(guess, BiexponentialParameters):
            raise TypeError("initial guesses must be BiexponentialParameters")
        if (
            min(guess.lifetimes_ns) < lifetime_bounds[0]
            or max(guess.lifetimes_ns) > lifetime_bounds[1]
            or not shift_bounds[0] <= guess.temporal_shift_ns <= shift_bounds[1]
        ):
            raise ValueError("initial guess lies outside supplied bounds")
    problem = _Problem(time, y, kernel, lifetime_bounds, shift_bounds, settings)
    starts, points = [], []
    accepted, descent, stopping = None, None, "no_finite_candidate"
    if not np.any(y):
        p = BiexponentialParameters(
            (0, 0),
            (np.exp(problem.log_mid), np.exp(problem.log_mid)),
            0,
            float(np.clip(0, *shift_bounds)),
        )
        problem.best_x, problem.best_value = problem.coordinates(p), 0.0
        problem.best_source = "analytical_zero_counts"
        problem.best_curve, problem.best_components = (
            np.zeros_like(y),
            np.zeros((2, y.size)),
        )
        accepted, descent, stopping = True, 0.0, "analytical_zero_counts"
    else:
        points = _initial_points(problem, guesses)
        tiers = (
            (4,)
            if settings.policy == "economical"
            else (
                (4, 8, 16, len(points))
                if settings.policy == "adaptive"
                else (len(points),)
            )
        )
        next_point, previous_value = 0, np.inf
        for tier in tiers:
            while (
                next_point < min(tier, len(points))
                and len(starts) < settings.max_starts
            ):
                starts.append(_refine(problem, points[next_point], len(starts), "seed"))
                next_point += 1
                if "optimization" in problem.exhausted:
                    break
            if problem.best_x is not None:
                accepted, descent, improved = _local_validation(problem, problem.best_x)
                # At most two bounded continuations per tier, charged to the
                # same start/evaluation budgets. Retain the best trial point.
                for _ in range(2):
                    if (
                        accepted is not False
                        or len(starts) >= settings.max_starts
                        or "optimization" in problem.exhausted
                    ):
                        break
                    starts.append(
                        _refine(problem, improved, len(starts), "validation_recovery")
                    )
                    accepted, descent, improved = _local_validation(
                        problem, problem.best_x
                    )
            if "optimization" in problem.exhausted:
                stopping = "optimization_budget_exhausted"
                break
            if "validation" in problem.exhausted:
                stopping = "validation_budget_exhausted"
                break
            endpoints = [
                s.best_centered_nll for s in starts if np.isfinite(s.best_centered_nll)
            ]
            disagreement = bool(
                endpoints
                and max(endpoints) - min(endpoints) > settings.descent_tolerance
            )
            stable = (
                np.isfinite(previous_value)
                and previous_value - problem.best_value <= settings.descent_tolerance
            )
            if (
                settings.policy == "adaptive"
                and accepted
                and stable
                and not disagreement
            ):
                stopping = "adaptive_stable_tiers"
                break
            previous_value = problem.best_value
            if next_point == len(points):
                stopping = "candidate_bank_completed"
                break
            if len(starts) >= settings.max_starts:
                problem.exhausted.add("starts")
                stopping = "start_budget_exhausted"
                break
            stopping = (
                "economical_policy_completed"
                if settings.policy == "economical"
                else "adaptive_tier_completed"
            )
    parameters = summary = stats = curve = components = None
    if problem.best_x is not None:
        parameters, order = _canonical(problem.physical(problem.best_x))
        curve, components = (
            _readonly(problem.best_curve),
            _readonly(problem.best_components[order]),
        )
        summary = _component_summary(parameters, components)
        if raw_counts:
            stats = calculate_poisson_fit_statistics(y, curve, n_parameters=6)
    diagnostics = _identifiability(problem, parameters, summary, starts)
    selected = next((s for s in starts if s.start_index == problem.best_start), None)
    if problem.best_x is None:
        stopping = "no_finite_candidate"
    optimization = BiexponentialOptimizationDiagnostics(
        tuple(starts),
        problem.starts_screened,
        problem.best_start,
        problem.best_source,
        None if selected is None else selected.optimizer_reported_success,
        accepted,
        descent,
        problem.validation_anchor,
        stopping,
        tuple(sorted(problem.exhausted)),
        problem.counts["initialization"],
        problem.counts["optimization"],
        problem.counts["validation"],
        problem.counts["diagnostics"],
        problem.seconds["initialization"],
        problem.seconds["optimization"],
        problem.seconds["validation"],
        problem.seconds["diagnostics"],
        perf_counter() - started,
        problem.initialization_rejected_candidates,
        problem.initialization_scaling_failed,
    )
    assumptions = BiexponentialFitAssumptions(
        lifetime_bounds,
        shift_bounds,
        settings,
        (problem.a_scale, problem.a_scale, 1, 1, problem.b_scale, problem.shift_scale),
        "prepared_irf" if prepared else "caller_declared_array",
        area,
        preparation_snapshot,
        source_kind,
        metadata_snapshot,
        provenance_snapshot,
        "raw_counts_caller_asserted" if raw_counts else "deterministic_expected_counts",
    )
    return BiexponentialFitResult(
        parameters,
        curve,
        components,
        summary,
        stats,
        optimization,
        assumptions,
        diagnostics,
    )


def fit_biexponential_reconvolution(
    time_ns: ArrayLike,
    counts: ArrayLike,
    irf: PreparedIRF | ArrayLike,
    *,
    lifetime_bounds_ns: tuple[float, float],
    temporal_shift_bounds_ns: tuple[float, float],
    initial_guesses: Sequence[BiexponentialParameters] | None = None,
    settings: BiexponentialFitSettings | None = None,
) -> BiexponentialFitResult:
    """Fit six constrained parameters to caller-asserted raw photon counts.

    Both IRF input forms must already be prepared and area-normalized on the
    supplied ns grid. No resampling, registration, normalization, time-origin
    change, count transformation or expectation floor is applied here. Positive
    shift moves the shared IRF later without renormalizing its remaining support.
    Components are sorted only in the returned result. Exact zero amplitudes,
    zero background and equal lifetimes remain admissible.

    Numerical failure returns the best finite evaluated candidate with explicit
    diagnostics; invalid inputs raise. No result establishes global optimality,
    component identifiability or physical correctness of the assumed IRF/model.
    Canonical measurement adapters must first enforce ``require_raw_counts()``.
    """
    return _fit_core(
        time_ns,
        counts,
        irf,
        lifetime_bounds_ns=lifetime_bounds_ns,
        temporal_shift_bounds_ns=temporal_shift_bounds_ns,
        initial_guesses=initial_guesses,
        settings=settings,
        raw_counts=True,
    )
