"""Independent, bounded direct-convolution numerical audit. No production imports.

Search results are immutable tuples, sealed before any cross-implementation
scoring. Agreement is finite-search evidence, never global optimality.
"""

from dataclasses import dataclass
from collections import Counter, deque
from math import fsum
from time import monotonic

import numpy as np
from scipy.optimize import least_squares, minimize


@dataclass(frozen=True)
class AuditSettings:
    trf_starts: int
    powell_starts: int
    model_calls: int
    wall_seconds: float
    validation_calls: int = 256
    assessment_calls: int = 2
    max_iterations: int = 1000
    descent_tolerance: float = 0.001

    def __post_init__(self):
        for name in (
            "trf_starts",
            "powell_starts",
            "model_calls",
            "validation_calls",
            "assessment_calls",
            "max_iterations",
        ):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError("nonnegative integer budgets required")
        if self.trf_starts + self.powell_starts == 0 or self.max_iterations == 0:
            raise ValueError("at least one start and iteration required")
        if self.validation_calls + self.assessment_calls > self.model_calls:
            raise ValueError("audit phase reservations exceed total")
        if (
            not np.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0
            or not np.isfinite(self.descent_tolerance)
            or self.descent_tolerance < 0
        ):
            raise ValueError("invalid time/tolerance")

    @classmethod
    def development_candidate(cls):
        return cls(16, 8, 200000, 900)


@dataclass(frozen=True)
class AuditStart:
    solver: str
    initial: tuple
    attempted: bool
    termination: str
    solver_success: bool | None
    calls: int
    call_limit: int
    elapsed_seconds: float
    fixed_shift: bool
    work: tuple = ()
    best_parameters: tuple | None = None
    best_centered_nll: float | None = None
    endpoint_parameters: tuple | None = None
    endpoint_centered_nll: float | None = None
    solver_status: int | None = None
    solver_message: str | None = None
    inward_start: tuple | None = None
    trial_diagnostics: tuple = ()
    trial_records_dropped: int = 0
    derivative_diagnostics: tuple = ()
    derivative_records_dropped: int = 0
    initial_shift_geometry: str | None = None
    free_coordinates: tuple = ()
    solver_centered_nll: float | None = None
    endpoint_objective_consistent: bool | None = None


@dataclass(frozen=True)
class AuditResult:
    parameters: tuple | None
    expected_counts: tuple | None
    centered_nll: float | None
    starts: tuple[AuditStart, ...]
    all_starts_attempted: bool
    search_complete: bool
    assessment_complete: bool
    consistent: bool
    validation_complete: bool
    validation_passed: bool | None
    best_probe_improvement: float | None
    validation_probes: tuple
    complete: bool
    reasons: tuple[str, ...]
    phase_calls: tuple
    component_convolutions: int
    elapsed_seconds: float
    settings: AuditSettings
    sealed: bool = True
    failed_model_calls: int = 0
    objective_evaluations: int = 0
    residual_evaluations: int = 0
    jacobian_evaluations: int = 0
    work: tuple = ()
    model_calls_by_origin: tuple = ()
    completed_component_convolutions: int = 0
    component_work_unknown_calls: int = 0
    assessment_candidate: tuple | None = None
    validation_candidate: tuple | None = None
    validation_details: tuple = ()
    final_trial_diagnostics: tuple = ()
    final_trial_records_dropped: int = 0
    search_best_centered_nll: float | None = None


@dataclass(frozen=True)
class AuditTrial:
    phase: str
    origin: str
    coordinates: tuple
    parameters: tuple
    lower_distances: tuple
    upper_distances: tuple
    status: str
    centered_nll: float | None
    model_calls: int
    charged_components: int


@dataclass(frozen=True)
class AuditDifference:
    coordinate: int
    anchor: tuple
    requested_step: float
    side: int
    at_bound: bool
    geometry: str
    points: tuple
    status: str
    slope_disagreement: float | None = None
    left_room: float | None = None
    right_room: float | None = None
    side_failures: tuple = ()


@dataclass(frozen=True)
class AuditValidationProbe:
    coordinate: int
    requested_step: float
    optimizer_step: float | None
    physical_step: float | None
    parameters: tuple | None
    status: str
    centered_nll: float | None


def direct_expected(time, kernel, parameters):
    """Physical vector [A0..2, tau0..2, B, shift]; no source normalization."""
    t, k, p = (
        np.asarray(time, float),
        np.asarray(kernel, float),
        np.asarray(parameters, float),
    )
    if (
        p.shape != (8,)
        or not np.all(np.isfinite(p))
        or np.any(p[:3] < 0)
        or np.any(p[3:6] <= 0)
        or p[6] < 0
    ):
        raise ValueError("invalid audit parameters")
    shifted = np.interp(t - p[7], t, k, left=0, right=0)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        rows = [
            p[j] * np.convolve(np.exp(-t / p[j + 3]), shifted)[: len(t)] * (t[1] - t[0])
            for j in range(3)
        ]
        return np.sum(rows, axis=0) + p[6]


def audit_terms(y, mu):
    """Independent centered-NLL bins, with a symmetric-ratio series near equality."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    if y.shape != mu.shape or not np.all(
        np.isfinite(y) & np.isfinite(mu) & (y >= 0) & (mu >= 0)
    ):
        raise ValueError("invalid audit objective inputs")
    c = mu.copy()
    positive = (y > 0) & (mu > 0)
    a, b = y[positive], mu[positive]
    # Divide first to avoid overflow in y+mu and in y/mu for extreme ratios.
    scale = np.maximum(a, b)
    rho = (a / scale - b / scale) / (a / scale + b / scale)
    close = np.abs(rho) < 0.2
    z = rho[close]
    tail = np.zeros_like(z)
    for n in range(31, 2, -2):
        tail = tail * z * z + 1 / n
    out = np.empty_like(a)
    # Use the original difference to retain small, representable offsets.
    z = (a[close] - b[close]) / a[close] / (1 + b[close] / a[close])
    out[close] = (a[close] * z) * (
        z * (1 + b[close] / a[close]) * (1 + (1 + z) * z * tail)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        # Factor the larger expectation/count before combining terms so that
        # finite centered objectives survive even near the floating-point limit.
        left, right = a[~close], b[~close]
        magnitude = np.maximum(left, right)
        out[~close] = magnitude * (
            (left / magnitude) * (np.log(left) - np.log(right))
            + (right - left) / magnitude
        )
    c[positive] = out
    c[(y > 0) & (mu == 0)] = np.inf
    return c


def _residual(y, mu):
    return np.sign(y - mu) * np.sqrt(2.0) * np.sqrt(audit_terms(y, mu))


def _seeds(y, bounds, shift_bounds, number):
    """Independent low-discrepancy physical-lifetime bank, data-only count scales."""
    mean = float(np.mean(y))
    for i in range(number):
        fractions = np.mod(
            (i + 1) * np.array([0.61803398875, 0.41421356237, 0.73205080757]), 1
        )
        tau = bounds[0] + (bounds[1] - bounds[0]) * fractions
        shift = shift_bounds[0] + (shift_bounds[1] - shift_bounds[0]) * (
            (i + 0.5) / number
        )
        yield np.r_[np.full(3, mean / 3), tau, mean / 10, shift]


class _Exhausted(Exception):
    pass


class _TrialFailure(Exception):
    """A numerical or representability failure, independent of work exhaustion."""


def _shift_region(time, kernel, shift):
    """Classify physical interpolation events before rounded query subtraction.

    Compensated summation preserves differences between supplied float64 values,
    including on platforms whose longdouble has no additional precision.
    This only constructs metadata. The physical model still uses np.interp.
    Rooms are distances to the next event on each side of the physical shift.
    """
    t = time
    q = t - shift
    places = np.searchsorted(t, q)
    left_room = right_room = np.inf
    region = "smooth"
    for offset in (-1, 0, 1):
        indices = places + offset
        valid = (indices >= 0) & (indices < len(t))
        distances = np.array(
            [
                fsum((float(a), -float(b), -float(shift)))
                for a, b in zip(t[valid], t[indices[valid]], strict=True)
            ]
        )
        negative, positive = distances[distances < 0], distances[distances > 0]
        if len(negative):
            left_room = min(left_room, float(np.min(-negative)))
        if len(positive):
            right_room = min(right_room, float(np.min(positive)))
        for j in indices[valid][distances == 0]:
            if j in (0, len(t) - 1) and kernel[j] != 0:
                region = "support_discontinuity"
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                before = (kernel[j] - kernel[j - 1]) / (t[j] - t[j - 1]) if j else 0
                after = (
                    (kernel[j + 1] - kernel[j]) / (t[j + 1] - t[j])
                    if j + 1 < len(t)
                    else 0
                )
            if region != "support_discontinuity":
                if not np.all(np.isfinite([before, after])):
                    region = "unrepresentable_shift_slopes"
                elif region == "unrepresentable_shift_slopes":
                    continue
                elif before != after:
                    region = "continuous_nondifferentiable_knot"
                elif region == "smooth":
                    region = "continuous_smooth_knot"
    return region, left_room, right_room


def _distinct_points(point, j, side, step, room, scales, *, time=None):
    """Construct two feasible physical points using bounded metadata work only."""
    physical = point[j] * scales[j]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinate_ulp = abs(np.nextafter(point[j], side * np.inf) - point[j])
        physical_ulp = abs(np.nextafter(physical, side * np.inf) - physical) / scales[j]
        minimum = 4 * max(coordinate_ulp, physical_ulp)
    h = min(step, room / 2)
    for _ in range(8):
        points = []
        for requested in (h, h / 2):
            probe = point.copy()
            probe[j] += side * requested
            delta = probe[j] - point[j]
            physical_delta = probe[j] * scales[j] - physical
            if (
                not np.isfinite(probe[j])
                or not np.isfinite(physical_delta)
                or side * delta <= 0
                or side * physical_delta <= 0
                or abs(delta) > room
                or (
                    time is not None
                    and np.array_equal(time - probe[j], time - point[j])
                )
            ):
                break
            points.append((probe, delta, physical_delta))
        if (
            len(points) == 2
            and points[0][2] != points[1][2]
            and (
                time is None
                or not np.array_equal(time - points[0][0][j], time - points[1][0][j])
            )
        ):
            return points
        if h >= room / 2:
            break
        h = min(max(2 * h, minimum), room / 2)
    raise _TrialFailure("unrepresentable_jacobian")


def _residual_jacobian(
    point, mu, y, t, k, scales, lower, upper, indices, evaluate, record
):
    """Independent two-scale, feasible-sided residual differences."""
    if np.any(mu == 0):
        record(
            AuditDifference(
                -1,
                tuple(point * scales),
                0,
                0,
                False,
                "zero_expectation",
                (),
                "residual_derivative_unavailable",
            )
        )
        raise _TrialFailure("residual_derivative_unavailable")
    base = _residual(y, mu)
    columns = []
    for j in indices:
        requested = 1e-5 * max(1.0, abs(point[j]))
        left, right = point[j] - lower[j], upper[j] - point[j]
        at_bound = left == 0 or right == 0
        geometry = "parameter_bound" if at_bound else "interior"
        side, details, discrepancy = 0, [], None
        side_failures = []
        try:
            if j == 7:
                geometry, left_shift, right_shift = _shift_region(t, k, point[j])
                if geometry == "unrepresentable_shift_slopes":
                    raise _TrialFailure("shift_geometry_unavailable")
                if geometry in (
                    "support_discontinuity",
                    "continuous_nondifferentiable_knot",
                ):
                    raise _TrialFailure("shift_knot_jacobian_unavailable")
                left, right = min(left, left_shift), min(right, right_shift)
            side = 1 if right >= left else -1
            room = right if side > 0 else left
            try:
                points = _distinct_points(
                    point, j, side, requested, room, scales, time=t if j == 7 else None
                )
            except _TrialFailure as error:
                # Metadata-only fallback to the other feasible side. No evaluation
                # has occurred and no numerical stencil is silently retried.
                side_failures.append((side, str(error)))
                side = -side
                room = right if side > 0 else left
                try:
                    points = _distinct_points(
                        point,
                        j,
                        side,
                        requested,
                        room,
                        scales,
                        time=t if j == 7 else None,
                    )
                except _TrialFailure as error:
                    side_failures.append((side, str(error)))
                    raise
            slopes = []
            for probe, delta, physical_delta in points:
                # Physical difference divided first, then chain rule to solver coordinates.
                details.append(
                    (tuple(probe * scales), float(delta), float(physical_delta))
                )
                value, prediction = evaluate(probe, "search", origin="jacobian_probe")
                if not np.isfinite(value):
                    raise _TrialFailure("infinite_residual_probe")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    slopes.append(
                        ((_residual(y, prediction) - base) / physical_delta) * scales[j]
                    )
            if not np.all(np.isfinite(slopes)):
                raise _TrialFailure("nonfinite_jacobian")
            discrepancy = float(np.max(np.abs(slopes[0] - slopes[1])))
            if not np.allclose(slopes[0], slopes[1], rtol=2e-3, atol=1e-5):
                raise _TrialFailure("unstable_jacobian")
        except (_TrialFailure, _Exhausted) as error:
            record(
                AuditDifference(
                    int(j),
                    tuple(point * scales),
                    requested,
                    side,
                    at_bound,
                    geometry,
                    tuple(details),
                    str(error),
                    discrepancy,
                    float(left),
                    float(right),
                    tuple(side_failures),
                )
            )
            raise
        record(
            AuditDifference(
                int(j),
                tuple(point * scales),
                requested,
                side,
                at_bound,
                geometry,
                tuple(details),
                "available",
                discrepancy,
                float(left),
                float(right),
                tuple(side_failures),
            )
        )
        columns.append(slopes[1])
    return np.column_stack(columns)


def _validation_point(x, j, step, scales, lower, upper):
    """Distinguish a bound-blocked direction from an unrepresentable probe."""
    side = 1 if step > 0 else -1
    room = upper[j] - x[j] if side > 0 else x[j] - lower[j]
    if room <= 0:
        return None, "infeasible"
    h = min(abs(step), room)
    for _ in range(8):
        q = x.copy()
        q[j] += side * h
        if (
            np.isfinite(q[j])
            and lower[j] <= q[j] <= upper[j]
            and side * (q[j] * scales[j] - x[j] * scales[j]) > 0
        ):
            return q, "available"
        if h >= room:
            break
        neighbor = np.nextafter(x[j], side * np.inf)
        physical_neighbor = np.nextafter(x[j] * scales[j], side * np.inf)
        minimum = max(
            abs(neighbor - x[j]), abs(physical_neighbor - x[j] * scales[j]) / scales[j]
        )
        h = min(max(2 * h, minimum), room)
    return None, "unrepresentable_validation"


def run_audit(
    time, observed, kernel, *, lifetime_bounds, shift_bounds, settings, clock=monotonic
):
    """Run explicitly configured audit only. No truth/production seed arguments."""
    if not isinstance(settings, AuditSettings):
        raise TypeError("explicit AuditSettings required")
    t, y, k = (np.array(a, dtype=float, copy=True) for a in (time, observed, kernel))
    if (
        t.ndim != 1
        or len(t) < 2
        or t.shape != y.shape
        or t.shape != k.shape
        or not np.all(np.isfinite(t))
        or not np.all(np.isfinite(y) & (y >= 0) & np.isfinite(k) & (k >= 0))
    ):
        raise ValueError("invalid audit arrays")
    if (
        np.any(np.diff(t) <= 0)
        or not np.allclose(np.diff(t), t[1] - t[0])
        or not np.isclose(np.trapezoid(k, t), 1, rtol=1e-7, atol=1e-12)
    ):
        raise ValueError(
            "prepared normalized kernel and approximately uniform grid required"
        )
    lo_tau, hi_tau = lifetime_bounds
    lo_shift, hi_shift = shift_bounds
    if (
        not np.all(np.isfinite([lo_tau, hi_tau, lo_shift, hi_shift]))
        or not 0 < lo_tau < hi_tau
        or not lo_shift < hi_shift
    ):
        raise ValueError("invalid audit bounds")
    scale = max(float(y.mean()), 1.0)
    scales = np.r_[np.full(3, scale), np.ones(3), scale, 1.0]
    lower = np.r_[np.zeros(3), np.full(3, lo_tau), 0, lo_shift] / scales
    upper = np.r_[np.full(3, np.inf), np.full(3, hi_tau), np.inf, hi_shift] / scales
    phases = dict(search=0, validation=0, assessment=0)
    started = clock()
    search_limit = (
        settings.model_calls - settings.validation_calls - settings.assessment_calls
    )
    caps = dict(
        search=search_limit,
        validation=settings.validation_calls,
        assessment=settings.assessment_calls,
    )
    best = None
    lease_end, lease_calls = np.inf, np.inf
    work, origins = Counter(), Counter()
    trials, derivatives = deque(maxlen=32), deque(maxlen=32)
    trial_events = derivative_events = 0
    start_best = None

    def record_derivative(detail):
        nonlocal derivative_events
        derivative_events += 1
        derivatives.append(detail)

    def record_trial(x, phase, origin, status, value, charged):
        nonlocal trial_events
        work[status] += 1
        trial_events += 1
        with np.errstate(over="ignore", invalid="ignore"):
            p = x * scales if x.shape == (8,) else x
            below = tuple((x - lower) * scales) if x.shape == (8,) else ()
            above = tuple((upper - x) * scales) if x.shape == (8,) else ()
        trials.append(
            AuditTrial(
                phase,
                origin,
                tuple(x.ravel()),
                tuple(p.ravel()),
                below,
                above,
                status,
                value,
                charged,
                3 * charged,
            )
        )

    def evaluate(x, phase, retain=True, *, origin="initial"):
        nonlocal best, start_best
        x = np.asarray(x, dtype=float)
        work["evaluation_requests"] += 1
        work[origin + "_requests"] += 1
        # Rejected callbacks still obey deadlines, even when no model call occurs.
        reason = None
        if clock() - started >= settings.wall_seconds:
            reason = "deadline"
        elif sum(phases.values()) >= settings.model_calls:
            reason = "global_model_limit"
        elif phase == "search" and clock() >= lease_end:
            reason = "invocation_time_limit"
        elif phase == "search" and phases[phase] >= lease_calls:
            reason = "invocation_model_limit"
        elif phases[phase] >= caps[phase]:
            reason = phase + "_limit"
        if reason:
            record_trial(x, phase, origin, reason, None, 0)
            raise _Exhausted(reason)
        if x.shape != (8,) or not np.all(np.isfinite(x)):
            record_trial(x, phase, origin, "invalid_trial", None, 0)
            raise _TrialFailure("invalid_trial")
        if np.any(x < lower) or np.any(x > upper):
            record_trial(x, phase, origin, "infeasible_trial", None, 0)
            raise _TrialFailure("infeasible_trial")
        with np.errstate(over="ignore", invalid="ignore"):
            physical = x * scales
        if not np.all(np.isfinite(physical)):
            record_trial(x, phase, origin, "unrepresentable_parameters", None, 0)
            raise _TrialFailure("unrepresentable_parameters")
        phases[phase] += 1
        work["model_calls"] += 1
        work[phase + "_model_calls"] += 1
        origins[origin] += 1
        try:
            with np.errstate(over="raise", invalid="raise"):
                mu = direct_expected(t, k, physical)
        except (ValueError, FloatingPointError, OverflowError):
            work["failed_model_calls"] += 1
            record_trial(x, phase, origin, "nonfinite_model", None, 1)
            raise _TrialFailure("nonfinite_model") from None
        work["completed_model_calls"] += 1
        work["objective_evaluations"] += 1
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                terms = audit_terms(y, mu)
                c = float(np.sum(terms))
        except (ValueError, FloatingPointError, OverflowError):
            record_trial(x, phase, origin, "invalid_objective", None, 1)
            raise _TrialFailure("invalid_objective") from None
        if not np.isfinite(c):
            impossible = (y > 0) & (mu == 0)
            if (
                np.isposinf(c)
                and np.any(impossible)
                and np.all(np.isfinite(terms[~impossible]))
            ):
                record_trial(x, phase, origin, "infinite_objective", c, 1)
                return c, mu
            record_trial(x, phase, origin, "objective_numerical_failure", c, 1)
            raise _TrialFailure("objective_numerical_failure")
        if c < 0:
            record_trial(x, phase, origin, "invalid_objective", c, 1)
            raise _TrialFailure("invalid_objective")
        work["finite_evaluations"] += 1
        if retain and (start_best is None or c < start_best[0]):
            start_best = (c, x.copy())
        if retain and (best is None or c < best[0]):
            best = (c, x.copy(), mu.copy())
            record_trial(x, phase, origin, "finite_best_discovery", c, 1)
        return c, mu

    count = settings.trf_starts + settings.powell_starts
    starts = []
    for index, p in enumerate(_seeds(y, lifetime_bounds, shift_bounds, count)):
        solver = "trf" if index < settings.trf_starts else "powell"
        allowance = search_limit // count + int(index < search_limit % count)
        begin, calls_before = clock(), phases["search"]
        work_before = work.copy()
        start_best = None
        trials.clear()
        derivatives.clear()
        trial_events = derivative_events = 0
        lease_end = min(
            started + settings.wall_seconds,
            begin + 0.85 * settings.wall_seconds / count,
        )
        lease_calls = calls_before + allowance
        initial_geometry = None
        fixed = False
        indices = np.arange(8)
        x = p / scales
        attempted, success, reason = False, None, "unattempted_budget"
        endpoint = endpoint_value = inward = solver_status = solver_message = None
        solver_value = endpoint_consistent = None
        first_residual = True

        def expand(v):
            result = x.copy()
            result[indices] = v
            return result

        def residual(v):
            nonlocal inward, first_residual
            work["residual_evaluations"] += 1
            work["solver_callback_requests"] += 1
            point = expand(v)
            if first_residual:
                if not np.array_equal(point, x):
                    inward = tuple(point * scales)
                first_residual = False
            value, mu = evaluate(point, "search", origin="residual")
            if not np.isfinite(value):
                raise _TrialFailure("infinite_residual")
            return _residual(y, mu)

        def jacobian(v):
            work["jacobian_evaluations"] += 1
            work["solver_callback_requests"] += 1
            point = expand(v)
            value, mu = evaluate(point, "search", origin="jacobian_anchor")
            if not np.isfinite(value):
                raise _TrialFailure("infinite_residual")
            return _residual_jacobian(
                point,
                mu,
                y,
                t,
                k,
                scales,
                lower,
                upper,
                indices,
                evaluate,
                record_derivative,
            )

        def scalar(v):
            work["scalar_evaluations"] += 1
            work["solver_callback_requests"] += 1
            try:
                return evaluate(expand(v), "search", origin="scalar")[0]
            except _TrialFailure as error:
                if str(error) == "infeasible_trial":
                    # Extended-real objective outside the physical domain.
                    return np.inf
                raise

        try:
            if allowance and begin - started < settings.wall_seconds:
                attempted = True
                initial_geometry = _shift_region(t, k, p[7])[0]
                fixed = solver == "trf" and initial_geometry != "smooth"
                indices = np.arange(7 if fixed else 8)
                evaluate(x, "search")
                if solver == "trf":
                    result = least_squares(
                        residual,
                        x[indices],
                        jac=jacobian,
                        method="trf",
                        bounds=(lower[indices], upper[indices]),
                        max_nfev=settings.max_iterations,
                        x_scale=1.0,
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-8,
                    )
                else:
                    result = minimize(
                        scalar,
                        x[indices],
                        method="Powell",
                        bounds=list(zip(lower[indices], upper[indices])),
                        options=dict(
                            maxiter=settings.max_iterations,
                            maxfev=allowance,
                            ftol=1e-10,
                            xtol=1e-7,
                        ),
                    )
                success = bool(result.success)
                solver_status = getattr(result, "status", None)
                solver_message = str(getattr(result, "message", ""))
                reason = "terminated" if success else "solver_unsuccessful"
                endpoint = tuple(expand(result.x) * scales)
                endpoint_value, _ = evaluate(
                    expand(result.x), "search", origin="endpoint"
                )
                if not np.isfinite(endpoint_value):
                    raise _TrialFailure("nonfinite_endpoint")
                endpoint_consistent = False
                reported = np.asarray(getattr(result, "fun", None), dtype=float)
                expected_shape = y.shape if solver == "trf" else ()
                if reported.shape != expected_shape or not np.all(
                    np.isfinite(reported)
                ):
                    raise _TrialFailure("solver_objective_unavailable")
                with np.errstate(over="ignore", invalid="ignore"):
                    solver_value = (
                        float(np.sum((0.5 * reported) * reported))
                        if solver == "trf"
                        else float(reported)
                    )
                endpoint_consistent = bool(
                    np.isfinite(solver_value)
                    and abs(endpoint_value - solver_value)
                    <= 1e-9 + 1e-12 * abs(endpoint_value)
                )
                if not endpoint_consistent:
                    raise _TrialFailure("endpoint_objective_mismatch")
                if clock() >= lease_end:
                    reason = "invocation_time_limit"
        except (_Exhausted, _TrialFailure) as error:
            reason = str(error)
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            reason = f"solver_numerical_failure:{type(error).__name__}:{error}"
        starts.append(
            AuditStart(
                solver,
                tuple(p),
                attempted,
                reason,
                success,
                phases["search"] - calls_before,
                allowance,
                clock() - begin,
                fixed,
                work=tuple(
                    (name, value - work_before[name])
                    for name, value in sorted(work.items())
                    if value != work_before[name]
                ),
                best_parameters=tuple(start_best[1] * scales) if start_best else None,
                best_centered_nll=start_best[0] if start_best else None,
                endpoint_parameters=endpoint,
                endpoint_centered_nll=endpoint_value,
                solver_status=solver_status,
                solver_message=solver_message,
                inward_start=inward,
                trial_diagnostics=tuple(trials),
                trial_records_dropped=max(0, trial_events - len(trials)),
                derivative_diagnostics=tuple(derivatives),
                derivative_records_dropped=max(0, derivative_events - len(derivatives)),
                initial_shift_geometry=initial_geometry,
                free_coordinates=tuple(int(i) for i in indices),
                solver_centered_nll=solver_value,
                endpoint_objective_consistent=endpoint_consistent,
            )
        )
    probes, details, improvement = [], [], None
    trials.clear()
    trial_events = 0
    assessment_candidate = validation_candidate = None
    search_best_value = best[0] if best else None
    validation_complete = assessment_complete = consistent = False
    passed = None
    if best is not None:
        c, original_x, mu = best
        x = original_x.copy()
        order = np.lexsort((-x[:3], x[3:6]))
        x[:3], x[3:6] = x[:3][order], x[3:6][order]
        assessment_candidate = tuple(x * scales)
        try:
            fresh, prediction = evaluate(
                x, "assessment", retain=False, origin="assessment"
            )
            if not np.isfinite(fresh):
                raise _TrialFailure("nonfinite_assessment")
            assessment_complete = True
            consistent = bool(
                np.allclose(prediction, mu, rtol=1e-11, atol=1e-10)
                and abs(fresh - c) <= 1e-9 + 1e-12 * abs(c)
            )
            c, mu = fresh, prediction
            validation_candidate = tuple(x * scales)
            improvement = 0.0
            for j in range(8):
                for step in (-0.01, -0.0001, 0.0001, 0.01):
                    try:
                        q, status = _validation_point(x, j, step, scales, lower, upper)
                    except (ValueError, FloatingPointError, OverflowError) as error:
                        status = f"validation_numerical_failure:{type(error).__name__}"
                        details.append(
                            AuditValidationProbe(
                                j, step, None, None, None, status, None
                            )
                        )
                        raise _TrialFailure(status) from error
                    if status == "infeasible":
                        probes.append((j, step, "infeasible", None))
                        details.append(
                            AuditValidationProbe(
                                j, step, None, None, None, status, None
                            )
                        )
                        continue
                    if q is None:
                        details.append(
                            AuditValidationProbe(
                                j, step, None, None, None, status, None
                            )
                        )
                        raise _TrialFailure(status)
                    try:
                        value, _ = evaluate(
                            q, "validation", retain=False, origin="validation"
                        )
                    except (_TrialFailure, _Exhausted) as error:
                        details.append(
                            AuditValidationProbe(
                                j,
                                step,
                                float(q[j] - x[j]),
                                float(q[j] * scales[j] - x[j] * scales[j]),
                                tuple(q * scales),
                                str(error),
                                None,
                            )
                        )
                        raise
                    status = "finite" if np.isfinite(value) else "infinite_objective"
                    probes.append((j, step, status, value))
                    details.append(
                        AuditValidationProbe(
                            j,
                            step,
                            float(q[j] - x[j]),
                            float(q[j] * scales[j] - x[j] * scales[j]),
                            tuple(q * scales),
                            status,
                            value,
                        )
                    )
                    improvement = max(improvement, c - value)
            validation_complete = True
            passed = improvement <= settings.descent_tolerance
        except (_Exhausted, _TrialFailure) as error:
            probes.append((-1, 0, str(error), None))
        best = (c, x, mu)
    all_attempted = all(s.attempted for s in starts)
    search_complete = all(s.termination == "terminated" for s in starts)
    reasons = []
    for valid, reason in (
        (all_attempted, "unattempted_starts"),
        (search_complete, "search_incomplete"),
        (best is not None, "no_finite_candidate"),
        (assessment_complete and consistent, "assessment_unavailable_or_inconsistent"),
        (validation_complete, "validation_incomplete"),
        (passed is True, "local_descent_or_unavailable"),
        (clock() - started < settings.wall_seconds, "deadline"),
    ):
        if not valid:
            reasons.append(reason)
    return AuditResult(
        tuple(best[1] * scales) if best else None,
        tuple(best[2]) if best else None,
        best[0] if best else None,
        tuple(starts),
        all_attempted,
        search_complete,
        assessment_complete,
        consistent,
        validation_complete,
        passed,
        improvement,
        tuple(probes),
        not reasons,
        tuple(reasons),
        tuple(phases.items()),
        3 * sum(phases.values()),
        clock() - started,
        settings,
        failed_model_calls=work["failed_model_calls"],
        objective_evaluations=work["objective_evaluations"],
        residual_evaluations=work["residual_evaluations"],
        jacobian_evaluations=work["jacobian_evaluations"],
        work=tuple(sorted(work.items())),
        model_calls_by_origin=tuple(sorted(origins.items())),
        completed_component_convolutions=3 * work["completed_model_calls"],
        component_work_unknown_calls=work["failed_model_calls"],
        assessment_candidate=assessment_candidate,
        validation_candidate=validation_candidate,
        validation_details=tuple(details),
        final_trial_diagnostics=tuple(trials),
        final_trial_records_dropped=max(0, trial_events - len(trials)),
        search_best_centered_nll=search_best_value,
    )
