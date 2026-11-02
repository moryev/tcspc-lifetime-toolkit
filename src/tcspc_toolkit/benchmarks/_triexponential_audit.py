"""Independent, bounded direct-convolution numerical audit. No production imports.

Search results are immutable tuples, sealed before any cross-implementation
scoring. Agreement is finite-search evidence, never global optimality.
"""

from dataclasses import dataclass
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


def _at_knot(t, shift):
    q = t - shift
    indices = np.searchsorted(t, q)
    mask = indices < len(t)
    return bool(np.any(t[indices[mask]] == q[mask]))


class _Exhausted(Exception):
    pass


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
    failed_models = objective_calls = residual_calls = jacobian_calls = 0

    def evaluate(x, phase, retain=True):
        nonlocal best, failed_models, objective_calls
        if (
            x.shape != (8,)
            or not np.all(np.isfinite(x))
            or np.any(x < lower)
            or np.any(x > upper)
        ):
            raise _Exhausted("infeasible_trial")
        if clock() - started >= settings.wall_seconds:
            raise _Exhausted("deadline")
        if phase == "search" and (clock() >= lease_end or phases[phase] >= lease_calls):
            raise _Exhausted("invocation_limit")
        if phases[phase] >= caps[phase]:
            raise _Exhausted(phase + "_limit")
        phases[phase] += 1
        try:
            mu = direct_expected(t, k, x * scales)
            objective_calls += 1
            c = float(np.sum(audit_terms(y, mu)))
        except (ValueError, FloatingPointError, OverflowError):
            failed_models += 1
            raise _Exhausted("nonfinite_model") from None
        if not np.isfinite(c):
            raise _Exhausted("nonfinite_objective")
        if retain and (best is None or c < best[0]):
            best = (c, x.copy(), mu.copy())
        return c, mu

    count = settings.trf_starts + settings.powell_starts
    starts = []
    for index, p in enumerate(_seeds(y, lifetime_bounds, shift_bounds, count)):
        solver = "trf" if index < settings.trf_starts else "powell"
        allowance = search_limit // count + int(index < search_limit % count)
        begin, calls_before = clock(), phases["search"]
        lease_end = min(
            started + settings.wall_seconds,
            begin + 0.85 * settings.wall_seconds / count,
        )
        lease_calls = calls_before + allowance
        fixed = solver == "trf" and _at_knot(t, p[7])
        indices = np.arange(7 if fixed else 8)
        x = p / scales
        attempted, success, reason = False, None, "unattempted_budget"

        def expand(v):
            result = x.copy()
            result[indices] = v
            return result

        def residual(v):
            nonlocal residual_calls
            residual_calls += 1
            _, mu = evaluate(expand(v), "search")
            return _residual(y, mu)

        def jacobian(v):
            nonlocal jacobian_calls
            jacobian_calls += 1
            point = expand(v)
            _, mu = evaluate(point, "search")
            if np.any(mu == 0):
                raise _Exhausted("residual_derivative_unavailable")
            if not fixed and _at_knot(t, point[7]):
                raise _Exhausted("shift_knot_jacobian_unavailable")
            base = _residual(y, mu)
            columns = []
            for j in indices:
                h = 1e-5 * max(1.0, abs(point[j]))
                if j == 7:
                    # Never span any actual interpolation knot or support edge.
                    q = t - point[7]
                    places = np.searchsorted(t, q)
                    distances = []
                    for offset in (0, -1):
                        loc = places + offset
                        valid = (loc >= 0) & (loc < len(t))
                        distances.extend(np.abs(q[valid] - t[loc[valid]]))
                    h = min(h, min(distances, default=h) / 4)
                direction = 1 if upper[j] - point[j] >= point[j] - lower[j] else -1
                room = upper[j] - point[j] if direction > 0 else point[j] - lower[j]
                h = min(h, room / 2)
                slopes = []
                for step in (h, h / 2):
                    probe = point.copy()
                    probe[j] += direction * step
                    if step <= 0 or probe[j] == point[j]:
                        raise _Exhausted("unrepresentable_jacobian")
                    _, prediction = evaluate(probe, "search")
                    slopes.append(
                        (_residual(y, prediction) - base) / (direction * step)
                    )
                if not np.all(np.isfinite(slopes)) or not np.allclose(
                    slopes[0], slopes[1], rtol=2e-3, atol=1e-5
                ):
                    raise _Exhausted("unstable_jacobian")
                columns.append(slopes[1])
            return np.column_stack(columns)

        try:
            if allowance and begin - started < settings.wall_seconds:
                attempted = True
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
                        lambda v: evaluate(expand(v), "search")[0],
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
                reason = "terminated" if success else "solver_unsuccessful"
                evaluate(expand(result.x), "search")
                if clock() >= lease_end:
                    reason = "invocation_limit"
        except _Exhausted as error:
            reason = str(error)
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
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
            )
        )
    probes, improvement = [], None
    validation_complete = assessment_complete = consistent = False
    passed = None
    if best is not None:
        c, x, mu = best
        order = np.lexsort((-x[:3], x[3:6]))
        x[:3], x[3:6] = x[:3][order], x[3:6][order]
        try:
            fresh, prediction = evaluate(x, "assessment", retain=False)
            assessment_complete = True
            consistent = bool(
                np.allclose(prediction, mu, rtol=1e-11, atol=1e-10)
                and abs(fresh - c) <= 1e-9 + 1e-12 * abs(c)
            )
            c, mu = fresh, prediction
            improvement = 0.0
            for j in range(8):
                for step in (-0.01, -0.0001, 0.0001, 0.01):
                    q = x.copy()
                    q[j] += step
                    if q[j] < lower[j] or q[j] > upper[j]:
                        probes.append((j, step, "infeasible", None))
                        continue
                    if q[j] == x[j] or np.array_equal(q * scales, x * scales):
                        raise _Exhausted("unrepresentable_validation")
                    value, _ = evaluate(q, "validation", retain=False)
                    probes.append((j, step, "finite", value))
                    improvement = max(improvement, c - value)
            validation_complete = True
            passed = improvement <= settings.descent_tolerance
        except _Exhausted as error:
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
        failed_model_calls=failed_models,
        objective_evaluations=objective_calls,
        residual_evaluations=residual_calls,
        jacobian_evaluations=jacobian_calls,
    )
