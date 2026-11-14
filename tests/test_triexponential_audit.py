"""Independent audit controls and honest completion/cross-score contracts."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal, localcontext
import warnings

import numpy as np
import pytest

from tcspc_toolkit.benchmarks import _triexponential_audit as audit
from tcspc_toolkit.benchmarks import triexponential as development
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_expected_counts,
)


def control():
    t = np.arange(25.0) / 4
    k = np.exp(-(((t - 0.7) / 0.2) ** 2))
    k /= np.trapezoid(k, t)
    return t, np.full(25, 3.0), k


@pytest.mark.parametrize("shift", [-0.3, 0, 0.031, 0.3])
@pytest.mark.parametrize("amplitudes", [(1, 2, 3), (0, 2, 3), (0, 0, 0)])
def test_direct_model_matches_independent_physical_controls(shift, amplitudes):
    t, y, k = control()
    t[8:] += 1e-8
    p = (*amplitudes, 0.5, 0.5, 3, 0, shift)
    expected = multiexponential_reconvolution_expected_counts(
        t,
        k,
        amplitudes=amplitudes,
        lifetimes=p[3:6],
        background=0,
        temporal_shift=shift,
    )
    np.testing.assert_allclose(
        audit.direct_expected(t, k, p), expected, rtol=1e-12, atol=1e-14
    )


def run(settings, y=None):
    t, observed, k = control()
    return audit.run_audit(
        t,
        observed if y is None else y,
        k,
        lifetime_bounds=(0.1, 5),
        shift_bounds=(-0.3, 0.3),
        settings=settings,
    )


@pytest.mark.parametrize("budget", [0, 2, 20, 70])
def test_audit_small_limits_preserve_incompleteness(budget):
    result = run(
        audit.AuditSettings(1, 1, budget, 20, validation_calls=0, assessment_calls=0)
    )
    assert sum(v for _, v in result.phase_calls) <= budget
    assert result.component_convolutions == 3 * sum(v for _, v in result.phase_calls)
    assert not result.complete and result.sealed
    assert all(s.calls <= s.call_limit for s in result.starts)
    if budget:
        assert result.parameters is not None


def test_zero_audit_can_complete_with_real_solvers():
    result = run(audit.AuditSettings(1, 1, 800, 30, max_iterations=10), np.zeros(25))
    assert result.parameters is not None
    # TRF's interior derivative at zero can fail or run out of work: no claim
    # of completion merely because the exact zero seed has objective zero.
    assert result.centered_nll == 0
    assert result.assessment_complete and result.consistent
    assert result.validation_complete and result.validation_passed


def completed_control(monkeypatch):
    monkeypatch.setattr(
        audit, "_seeds", lambda *args: iter([np.array([0, 0, 0, 1, 2, 3, 3, 0.031])])
    )
    monkeypatch.setattr(
        audit,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(x=x, success=True, fun=0.0),
    )
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=80))
    assert result.complete and result.centered_nll == 0
    assert result.all_starts_attempted and result.search_complete
    assert result.assessment_complete and result.validation_complete
    assert result.parameters[:3] == (0, 0, 0)
    return result


def test_completed_audit_requires_each_stage(monkeypatch):
    completed_control(monkeypatch)


def test_independent_seed_and_dependency_isolation():
    source = Path(audit.__file__).read_text()
    imports = [
        n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Import)
        for alias in n.names
    ]
    assert not any(n and n.startswith("tcspc_toolkit") for n in imports)
    a = list(audit._seeds(np.ones(25), (0.1, 5), (-0.3, 0.3), 24))
    b = list(audit._seeds(np.ones(25), (0.1, 5), (-0.3, 0.3), 24))
    np.testing.assert_array_equal(a, b)
    assert len({tuple(p) for p in a}) == 24


def fake_fit(p, time, kernel, y):
    mu = development._production_prediction(time, kernel, p)
    return SimpleNamespace(
        parameters=SimpleNamespace(
            amplitudes=p[:3],
            lifetimes_ns=p[3:6],
            background=p[6],
            temporal_shift_ns=p[7],
        ),
        expected_counts=mu,
        centered_nll=float(sum(audit.audit_terms(y, mu))),
        numerical=SimpleNamespace(accepted=True),
    )


@pytest.mark.parametrize(
    "production_b, audit_b, expected",
    [
        (3, 3, "agreement"),
        (4, 3, "production_worse"),
        (3, 4, "production_better_audit_limitation"),
    ],
)
def test_both_signed_gaps_and_sealed_search(
    production_b, audit_b, expected, monkeypatch
):
    result = completed_control(monkeypatch)
    t, y, k = control()
    p = (0, 0, 0, 1, 2, 3, production_b, 0.031)
    q = (0, 0, 0, 1, 2, 3, audit_b, 0.031)
    mu = audit.direct_expected(t, k, q)
    result = replace(
        result,
        parameters=q,
        expected_counts=tuple(mu),
        centered_nll=float(sum(audit.audit_terms(y, mu))),
    )
    before = repr(result)
    compared = development.cross_score(t, y, k, fake_fit(p, t, k, y), result)
    assert compared["status"] == expected
    assert compared["component_convolutions"] == 12
    assert repr(result) == before
    incomplete = development.cross_score(
        t, y, k, fake_fit(p, t, k, y), replace(result, complete=False)
    )
    assert incomplete["status"] == "audit_incomplete"
    bad = development.cross_score(
        t,
        y,
        k,
        fake_fit(p, t, k, y),
        replace(result, centered_nll=result.centered_nll + 1),
    )
    assert bad["status"] == "cross_implementation_disagreement"
    with pytest.raises(ValueError, match="sealed"):
        development.cross_score(
            t, y, k, fake_fit(p, t, k, y), replace(result, sealed=False)
        )


def test_audit_deadline_keeps_attempt_denominator():
    tick = iter(np.arange(1000.0))
    t, y, k = control()
    result = audit.run_audit(
        t,
        y,
        k,
        lifetime_bounds=(0.1, 5),
        shift_bounds=(-0.3, 0.3),
        settings=audit.AuditSettings(2, 2, 100, 1, validation_calls=0),
        clock=lambda: next(tick),
    )
    assert len(result.starts) == 4 and not result.complete
    assert not result.all_starts_attempted
    assert "deadline" in result.reasons


def test_audit_solver_exceptions_preserve_starts_and_finite_candidate(monkeypatch):
    def broken(fun, x, **kw):
        fun(x)
        raise np.linalg.LinAlgError("controlled factorization failure")

    monkeypatch.setattr(audit, "least_squares", broken)
    monkeypatch.setattr(audit, "minimize", broken)
    result = run(audit.AuditSettings(1, 1, 100, 30, validation_calls=50))
    assert len(result.starts) == 2 and result.all_starts_attempted
    assert all(
        s.termination.startswith("solver_numerical_failure") for s in result.starts
    )
    assert all(s.calls == 2 for s in result.starts)
    assert result.parameters is not None and result.assessment_complete
    assert not result.complete and not result.search_complete


@pytest.mark.parametrize("shift", [-0.25, 0.0, 0.25])
@pytest.mark.parametrize("amplitudes", [(1, 2, 3), (0, 0, 0)])
def test_direct_convolution_has_explicit_sum_oracle(shift, amplitudes):
    time = np.arange(6) / 4 + 0.125
    kernel = np.array([0, 2, 2, 0, 0, 0.0])
    lifetimes = [0.5, 1, 1]
    expected = []
    shifted = np.interp(time - shift, time, kernel, left=0, right=0)
    for i in range(len(time)):
        total = 0.3
        for a, tau in zip(amplitudes, lifetimes, strict=True):
            total += 0.25 * sum(
                a * np.exp(-time[j] / tau) * shifted[i - j] for j in range(i + 1)
            )
        expected.append(total)
    np.testing.assert_allclose(
        audit.direct_expected(time, kernel, (*amplitudes, *lifetimes, 0.3, shift)),
        expected,
        rtol=2e-15,
    )


@pytest.mark.parametrize(
    "y,mu",
    [
        (0.0, 0.0),
        (0.0, 2.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (2.0, 2.0 + 1e-10),
        (1e200, np.nextafter(1e200, np.inf)),
        (1e308, 1e308),
        (1e308, 8e307),
        (1.0, 1e-300),
        (1e-300, 1e-300),
    ],
)
def test_audit_objective_decimal_reference(y, mu):
    with localcontext() as context:
        context.prec = 90
        a, b = Decimal(y), Decimal(mu)
        expected = (
            b if a == 0 else Decimal("Infinity") if b == 0 else b - a + a * (a / b).ln()
        )
    actual = audit.audit_terms([y], [mu])[0]
    if expected.is_infinite():
        assert np.isposinf(actual)
    else:
        assert actual == pytest.approx(float(expected), rel=3e-14, abs=1e-310)


def background_seed(monkeypatch, count=1, shift=0.031):
    point = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, shift])
    monkeypatch.setattr(
        audit, "_seeds", lambda *args: iter([point.copy() for _ in range(count)])
    )
    return point


def test_powell_callback_rejects_then_continues_and_counts(monkeypatch):
    background_seed(monkeypatch)

    def solver(fun, x, **kwargs):
        assert fun(x) == 0
        impossible = x.copy()
        impossible[6] = 0
        assert np.isposinf(fun(impossible))
        infeasible = x.copy()
        infeasible[0] = -np.nextafter(0.0, 1.0)
        assert np.isposinf(fun(infeasible))
        assert fun(x) == 0
        return SimpleNamespace(x=x, success=True, fun=0.0)

    monkeypatch.setattr(audit, "minimize", solver)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.complete
    start = result.starts[0]
    work = dict(start.work)
    assert start.calls == 5  # initial, three feasible callbacks, endpoint
    assert work["scalar_evaluations"] == 4 and work["solver_callback_requests"] == 4
    assert work["infeasible_trial"] == 1 and work["infinite_objective"] == 1
    bad = next(d for d in start.trial_diagnostics if d.status == "infeasible_trial")
    assert bad.model_calls == bad.charged_components == 0 and bad.lower_distances[0] < 0
    infinite = next(
        d for d in start.trial_diagnostics if d.status == "infinite_objective"
    )
    assert infinite.model_calls == 1 and infinite.charged_components == 3
    assert np.isposinf(infinite.centered_nll)
    assert result.component_convolutions == result.completed_component_convolutions
    assert sum(dict(result.model_calls_by_origin).values()) == sum(
        dict(result.phase_calls).values()
    )


def test_installed_powell_accepts_extended_real_bounded_objective():
    values = []

    def objective(x):
        value = np.inf if 1.4 < x[0] < 1.6 else (x[0] - 2) ** 2
        values.append(value)
        return value

    # SciPy may emit intermediate inf-inf warnings while recovering a finite line.
    with warnings.catch_warnings(record=True):
        result = audit.minimize(
            objective,
            [3.0],
            method="Powell",
            bounds=[(0, 4)],
            options=dict(maxfev=100, maxiter=10),
        )
    assert result.success and result.fun < 1e-12
    assert any(np.isposinf(v) for v in values)
    first = next(i for i, v in enumerate(values) if np.isposinf(v))
    assert np.isfinite(values[0]) and any(np.isfinite(v) for v in values[first + 1 :])
    assert len(values) <= 100


@pytest.mark.parametrize("damage", ["infinite", "infeasible", "nan"])
def test_false_powell_success_never_completes_bad_endpoint(monkeypatch, damage):
    background_seed(monkeypatch)

    def solver(fun, x, **kw):
        q = x.copy()
        if damage == "infinite":
            q[6] = 0
        elif damage == "infeasible":
            q[0] = -1e-16
        else:
            q[6] = np.nan
        return SimpleNamespace(x=q, success=True)

    monkeypatch.setattr(audit, "minimize", solver)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.starts[0].solver_success
    assert result.starts[0].termination != "terminated"
    assert result.centered_nll == 0 and result.assessment_complete
    assert not result.complete and not result.search_complete


@pytest.mark.parametrize("failure", ["model", "nan", "overflow"])
def test_numerical_failures_are_fatal_and_keep_finite_best(monkeypatch, failure):
    background_seed(monkeypatch)
    real = audit.direct_expected

    def model(t, k, p):
        if p[6] != 3:
            if failure == "model":
                raise FloatingPointError("controlled")
            return np.full_like(t, np.nan if failure == "nan" else 1e308)
        return real(t, k, p)

    monkeypatch.setattr(audit, "direct_expected", model)

    def solver(fun, x, **kw):
        q = x.copy()
        q[6] = 2
        fun(q)
        pytest.fail("numerical failure concealed")

    monkeypatch.setattr(audit, "minimize", solver)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.parameters is not None and result.centered_nll == 0
    assert not result.search_complete
    assert result.starts[0].calls == 2
    if failure == "model":
        assert result.component_work_unknown_calls > 0
        assert (
            result.component_convolutions - result.completed_component_convolutions
            == 3 * result.failed_model_calls
        )
    else:
        assert result.failed_model_calls == 0


@pytest.mark.parametrize(
    "shift,expected",
    [
        (0.0, "continuous_nondifferentiable_knot"),
        (np.nextafter(0.0, 1.0), "smooth"),
        (np.nextafter(0.0, -1.0), "smooth"),
        (0.031, "smooth"),
    ],
)
def test_shift_geometry_preserves_near_knot_distance(shift, expected):
    t = np.arange(5) / 4
    k = np.array([0, 1, 2, 1, 0.0])
    region, left, right = audit._shift_region(t, k, shift)
    assert region == expected
    if shift > 0 and shift < 1e-10:
        assert left == shift and right > 1e-3
    if shift < 0:
        assert right == -shift and left > 1e-3


def test_shift_geometry_smooth_knots_jumps_and_actual_grid():
    t = np.arange(5) / 4
    assert (
        audit._shift_region(t, np.array([0, 0, 0, 0, 2.0]), 0.5)[0]
        == "continuous_smooth_knot"
    )
    assert audit._shift_region(t, np.ones(5), 0)[0] == "support_discontinuity"
    t[3:] += 1e-9
    assert audit._shift_region(t, np.ones(5), 0.031)[0] == "smooth"
    shift = np.nextafter(t[2] - t[1], np.inf)
    assert audit._shift_region(t, np.ones(5), shift)[0] == "smooth"


@pytest.mark.parametrize(
    "where", ["lower", "near_lower", "upper", "near_upper", "interior"]
)
def test_independent_residual_jacobian_uses_actual_steps(where):
    x = np.ones(8)
    lower = np.zeros(8)
    upper = np.full(8, 2.0)
    x[3] = {
        "lower": 0.0,
        "near_lower": np.nextafter(0.0, 1.0),
        "upper": 2.0,
        "near_upper": np.nextafter(2.0, 0.0),
        "interior": 1.0,
    }[where]
    scale = np.ones(8)
    scale[3] = 3.7
    y = np.array([2.0, 3.0])
    mu = y.copy()
    records = []
    points = []

    def evaluate(q, *args, **kw):
        points.append(q.copy())
        return 0.0, mu + (q[3] * scale[3] - x[3] * scale[3])

    t = np.array([0.0, 1.0])
    k = np.ones(2)
    jac = audit._residual_jacobian(
        x, mu, y, t, k, scale, lower, upper, [3], evaluate, records.append
    )
    np.testing.assert_allclose(jac[:, 0], -scale[3] / np.sqrt(y), rtol=3e-5)
    assert len(points) == 2 and len(records) == 1 and records[0].status == "available"
    assert records[0].at_bound == (where in ("lower", "upper"))
    for q, record in zip(points, records[0].points, strict=True):
        assert record[1] == q[3] - x[3]
        assert record[2] == q[3] * scale[3] - x[3] * scale[3]


@pytest.mark.parametrize(
    "shift", [np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0), 0.031]
)
def test_near_shift_knot_derivative_has_analytic_linear_oracle(shift):
    t = np.arange(5) / 4
    k = np.array([0, 1, 2, 1, 0.0])
    x = np.ones(8)
    x[7] = shift
    lower = np.zeros(8)
    lower[7] = -0.3
    upper = np.full(8, 2.0)
    upper[7] = 0.3
    y = mu = np.full(5, 2.0)
    records = []

    def evaluate(q, *a, **kw):
        return 0.0, mu + (q[7] - x[7])

    jac = audit._residual_jacobian(
        x, mu, y, t, k, np.ones(8), lower, upper, [7], evaluate, records.append
    )
    np.testing.assert_allclose(jac[:, 0], -1 / np.sqrt(2), rtol=1e-5)
    assert records[0].geometry == "smooth"


@pytest.mark.parametrize("kind", ["kink", "jump", "zero", "narrow", "unstable"])
def test_jacobian_unavailability_is_explicit(kind):
    t = np.arange(5) / 4
    k = np.array([0, 1, 2, 1, 0.0])
    x = np.ones(8)
    x[7] = 0.0
    lower = np.zeros(8)
    lower[7] = -0.3
    upper = np.full(8, 2.0)
    upper[7] = 0.3
    y = mu = np.full(5, 2.0)
    j = 7
    records = []
    calls = []
    if kind == "jump":
        k = np.ones(5)
    if kind == "zero":
        mu = np.zeros(5)
    if kind == "narrow":
        j = 3
        lower[j] = x[j]
        upper[j] = np.nextafter(x[j], np.inf)
    if kind == "unstable":
        j = 3

    def evaluate(q, *a, **kw):
        calls.append(q.copy())
        return 0.0, mu + (1.0 if kind == "unstable" else q[j] - x[j])

    with pytest.raises(audit._TrialFailure):
        audit._residual_jacobian(
            x, mu, y, t, k, np.ones(8), lower, upper, [j], evaluate, records.append
        )
    assert records and records[-1].status != "available"
    if kind != "unstable":
        assert not calls


def test_trf_inward_start_and_nested_accounting(monkeypatch):
    background_seed(monkeypatch)

    def solver(fun, x, jac, **kw):
        q = x.copy()
        q[0] = 1e-10
        residuals = fun(q)
        jac(q)
        return SimpleNamespace(x=q, success=True, fun=residuals)

    monkeypatch.setattr(audit, "least_squares", solver)
    result = run(audit.AuditSettings(1, 0, 150, 30, validation_calls=50))
    s = result.starts[0]
    w = dict(s.work)
    assert s.inward_start is not None and s.inward_start[0] == pytest.approx(3e-10)
    assert s.calls == 20  # initial, residual, anchor, 16 stencil calls, endpoint
    assert w["solver_callback_requests"] == 2 and w["jacobian_evaluations"] == 1
    assert len(s.derivative_diagnostics) == 8


@pytest.mark.parametrize(
    "budget,expected", [(1, "global_model_limit"), (5, "invocation_model_limit")]
)
def test_limits_interrupt_jacobian_and_retain_accounting(monkeypatch, budget, expected):
    background_seed(monkeypatch, count=1 if budget == 1 else 2)

    def solver(fun, x, jac, **kw):
        jac(x)
        return SimpleNamespace(x=x, success=True, fun=0.0)

    monkeypatch.setattr(audit, "least_squares", solver)
    settings = audit.AuditSettings(
        1 if budget == 1 else 2, 0, budget, 30, validation_calls=0, assessment_calls=0
    )
    result = run(settings)
    assert result.starts[0].termination == expected
    assert sum(dict(result.phase_calls).values()) <= budget
    assert all(s.calls <= s.call_limit for s in result.starts)
    assert result.parameters is not None and not result.complete


def test_rejected_trial_diagnostics_are_bounded(monkeypatch):
    background_seed(monkeypatch)

    def solver(fun, x, **kw):
        for _ in range(40):
            q = x.copy()
            q[0] = -1
            assert np.isinf(fun(q))
        return SimpleNamespace(x=x, success=True, fun=0.0)

    monkeypatch.setattr(audit, "minimize", solver)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    s = result.starts[0]
    assert (
        s.calls == 2 and len(s.trial_diagnostics) == 32 and s.trial_records_dropped == 9
    )
    assert dict(s.work)["infeasible_trial"] == 40


def test_final_validation_infinite_probes_keep_candidate_identity(monkeypatch):
    point = background_seed(monkeypatch)
    point[6] = 0.03
    monkeypatch.setattr(audit, "_seeds", lambda *a: iter([point]))
    monkeypatch.setattr(
        audit,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(
            x=x, success=True, fun=25 * (0.03 - 3 + 3 * np.log(100))
        ),
    )
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.validation_complete
    assert any(p.status == "infinite_objective" for p in result.validation_details)
    assert (
        result.parameters == result.assessment_candidate == result.validation_candidate
    )
    assert result.parameters[6] == 0.03
    assert not result.validation_passed and not result.complete


@pytest.mark.parametrize(
    "point,step,status",
    [
        (0.0, -0.01, "infeasible"),
        (np.nextafter(0.0, 1.0), -0.01, "available"),
        (1.0, 0.01, "infeasible"),
        (np.nextafter(1.0, 0.0), 0.01, "available"),
    ],
)
def test_validation_exact_and_near_bounds(point, step, status):
    x = np.ones(8) * point
    lo = np.zeros(8)
    hi = np.ones(8)
    q, actual = audit._validation_point(x, 0, step, np.ones(8), lo, hi)
    assert actual == status
    if q is not None:
        assert q[0] != x[0] and lo[0] <= q[0] <= hi[0]


def test_pilot_allocation_is_fixed_with_mock_solvers(monkeypatch):
    def solver(fun, x, **kw):
        value = fun(x)
        return SimpleNamespace(x=x, success=True, fun=value)

    monkeypatch.setattr(audit, "least_squares", solver)
    monkeypatch.setattr(audit, "minimize", solver)
    # Only tiny software-control arrays. No pilot problem is constructed.
    result = run(audit.AuditSettings(16, 8, 40000, 120))
    assert len(result.starts) == 24 and result.all_starts_attempted
    assert [s.call_limit for s in result.starts] == [1656] * 22 + [1655] * 2
    assert sum(s.call_limit for s in result.starts) == 39742
    assert all(s.calls == 3 for s in result.starts)
    assert result.settings.validation_calls == 256
    assert result.settings.assessment_calls == 2
    assert dict(result.phase_calls)["search"] == 72


@pytest.mark.parametrize(
    "elapsed,reason", [(4.3, "invocation_time_limit"), (121, "deadline")]
)
def test_time_limits_are_distinct_and_charge_no_blocked_model(
    monkeypatch, elapsed, reason
):
    now = [0.0]

    def solver(fun, x, **kw):
        now[0] += elapsed
        fun(x)
        return SimpleNamespace(x=x, success=True, fun=0.0)

    monkeypatch.setattr(audit, "minimize", solver)
    t, y, k = control()
    result = audit.run_audit(
        t,
        y,
        k,
        lifetime_bounds=(0.1, 5),
        shift_bounds=(-0.3, 0.3),
        settings=audit.AuditSettings(0, 24, 40000, 120),
        clock=lambda: now[0],
    )
    assert result.starts[0].termination == reason
    assert result.starts[0].calls == 1
    assert dict(result.starts[0].work)[reason] == 1
    assert not result.search_complete


def test_zero_expectation_jacobian_failure_preserves_scalar_candidate(monkeypatch):
    p = background_seed(monkeypatch)
    p[6] = 0
    monkeypatch.setattr(audit, "_seeds", lambda *a: iter([p]))

    def solver(fun, x, jac, **kw):
        np.testing.assert_array_equal(fun(x), np.zeros(25))
        jac(x)
        pytest.fail("zero-expectation Jacobian manufactured")

    monkeypatch.setattr(audit, "least_squares", solver)
    result = run(audit.AuditSettings(1, 0, 150, 30, validation_calls=50), np.zeros(25))
    assert result.centered_nll == 0
    assert result.starts[0].termination == "residual_derivative_unavailable"
    assert result.assessment_complete and not result.search_complete


def test_canonical_return_assessment_and_validation_are_same_point(monkeypatch):
    p = np.array([1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 0.031])
    t, y, k = control()
    reference = float(sum(audit.audit_terms(y, audit.direct_expected(t, k, p))))
    monkeypatch.setattr(audit, "_seeds", lambda *a: iter([p]))
    monkeypatch.setattr(
        audit,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(x=x, success=True, fun=reference),
    )
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.parameters[3:6] == (1.0, 2.0, 3.0)
    assert result.parameters[:3] == (2.0, 3.0, 1.0)
    assert (
        result.parameters == result.assessment_candidate == result.validation_candidate
    )
    t, y, k = control()
    np.testing.assert_array_equal(
        result.expected_counts, audit.direct_expected(t, k, result.parameters)
    )
    np.testing.assert_allclose(
        result.centered_nll,
        sum(audit.audit_terms(y, result.expected_counts)),
        rtol=1e-13,
        atol=1e-13,
    )
    assert result.validation_complete and not result.validation_passed
    assert result.best_probe_improvement > 0 and not result.complete


def test_fresh_assessment_failure_keeps_search_best(monkeypatch):
    background_seed(monkeypatch)
    real = audit.direct_expected
    calls = []

    def model(*args):
        calls.append(args)
        if len(calls) == 3:
            raise FloatingPointError("fresh assessment failed")
        return real(*args)

    monkeypatch.setattr(audit, "direct_expected", model)
    monkeypatch.setattr(
        audit,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(x=x, success=True, fun=0.0),
    )
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.centered_nll == result.search_best_centered_nll == 0
    assert result.parameters is not None
    assert not result.assessment_complete and not result.complete
    assert result.validation_candidate is None


def test_derivative_log_truncation_is_explicit(monkeypatch):
    background_seed(monkeypatch)

    def solver(fun, x, jac, **kw):
        for _ in range(5):
            jac(x)
        return SimpleNamespace(x=x, success=True, fun=np.zeros(25))

    monkeypatch.setattr(audit, "least_squares", solver)
    result = run(audit.AuditSettings(1, 0, 200, 30, validation_calls=50))
    s = result.starts[0]
    assert len(s.derivative_diagnostics) == 32 and s.derivative_records_dropped == 8
    assert dict(s.work)["jacobian_evaluations"] == 5
    assert s.calls == 87


def test_metadata_step_growth_does_not_evaluate_a_collapsed_point():
    x = np.ones(8)
    x[0] = 1e20
    points = audit._distinct_points(x, 0, 1, 1e-20, 1e10, np.ones(8))
    assert len(points) == 2
    assert all(p[1] > 0 and p[2] > 0 for p in points)
    assert points[0][1] != points[1][1]


def test_unrepresentable_validation_has_distinct_status():
    x = np.ones(8)
    scales = np.ones(8)
    scales[0] = np.nextafter(0.0, 1.0)
    lower = np.zeros(8)
    upper = np.full(8, 2.0)
    upper[0] = np.nextafter(1.0, 2.0)
    point, status = audit._validation_point(x, 0, 0.01, scales, lower, upper)
    assert point is None and status == "unrepresentable_validation"


def test_coordinate_overflow_is_rejected_before_charging_a_model(monkeypatch):
    background_seed(monkeypatch)

    def solver(fun, x, **kw):
        q = x.copy()
        q[0] = np.finfo(float).max
        fun(q)
        pytest.fail("unrepresentable trial concealed")

    monkeypatch.setattr(audit, "minimize", solver)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    s = result.starts[0]
    assert s.termination == "unrepresentable_parameters" and s.calls == 1
    assert dict(s.work)["unrepresentable_parameters"] == 1
    assert s.trial_diagnostics[-1].model_calls == 0
    assert result.centered_nll == 0


@pytest.mark.parametrize("kind", ["smooth", "kink", "jump"])
def test_initial_knots_preserve_restricted_shift_contract(monkeypatch, kind):
    t, y, k = control()
    shift = 0.0
    if kind == "smooth":
        k = np.r_[np.zeros(24), 8.0]
        shift = t[-2]
    elif kind == "jump":
        k = np.ones(25) / 6
    else:
        k[[0, -1]] = 0
        k /= np.trapezoid(k, t)
    background_seed(monkeypatch, shift=shift)

    def solver(fun, x, jac, **kw):
        assert len(x) == 7
        assert jac(x).shape == (25, 7)
        return SimpleNamespace(x=x, success=True, fun=np.zeros(25))

    monkeypatch.setattr(audit, "least_squares", solver)
    result = audit.run_audit(
        t,
        y,
        k,
        lifetime_bounds=(0.1, 5),
        shift_bounds=(-7, 7),
        settings=audit.AuditSettings(1, 0, 150, 30, validation_calls=50),
    )
    s = result.starts[0]
    assert s.fixed_shift and s.free_coordinates == tuple(range(7))
    assert (
        s.initial_shift_geometry
        == {
            "smooth": "continuous_smooth_knot",
            "kink": "continuous_nondifferentiable_knot",
            "jump": "support_discontinuity",
        }[kind]
    )


def test_shift_stencil_reports_agreeing_smooth_knot():
    t = np.arange(5) / 4
    k = np.array([0, 0, 0, 0, 2.0])
    x = np.ones(8)
    x[7] = 0.5
    lo = np.zeros(8)
    hi = np.full(8, 2.0)
    y = mu = np.full(5, 2.0)
    details = []

    def evaluate(q, *a, **kw):
        return 0.0, mu.copy()

    jac = audit._residual_jacobian(
        x, mu, y, t, k, np.ones(8), lo, hi, [7], evaluate, details.append
    )
    np.testing.assert_array_equal(jac, np.zeros((5, 1)))
    assert (
        details[0].geometry == "continuous_smooth_knot"
        and details[0].status == "available"
    )


def test_powell_all_infinite_callback_region_cannot_override_finite_seed(monkeypatch):
    background_seed(monkeypatch)

    def solver(fun, x, **kw):
        q = x.copy()
        q[6] = 0
        for _ in range(4):
            assert np.isposinf(fun(q))
        return SimpleNamespace(x=q, success=True)

    monkeypatch.setattr(audit, "minimize", solver)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert result.centered_nll == 0 and not result.search_complete
    assert dict(result.starts[0].work)["infinite_objective"] == 5


def test_geometry_exception_retains_previous_start_candidate(monkeypatch):
    background_seed(monkeypatch, count=2)
    real = audit._shift_region
    calls = []

    def geometry(*args):
        calls.append(args)
        if len(calls) == 2:
            raise OverflowError("controlled geometry")
        return real(*args)

    monkeypatch.setattr(audit, "_shift_region", geometry)
    monkeypatch.setattr(
        audit,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(x=x, success=True, fun=0.0),
    )
    result = run(audit.AuditSettings(0, 2, 100, 30, validation_calls=50))
    assert result.parameters is not None and result.centered_nll == 0
    assert (
        result.starts[1].calls == 0
        and "solver_numerical_failure" in result.starts[1].termination
    )
    assert not result.search_complete


@pytest.mark.parametrize("error", [OverflowError, FloatingPointError, ValueError])
def test_validation_construction_failure_keeps_exact_finite_candidate(
    monkeypatch, error
):
    background_seed(monkeypatch)
    monkeypatch.setattr(
        audit,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(x=x, success=True, fun=0.0),
    )

    def failed(*args):
        raise error("controlled metadata failure")

    monkeypatch.setattr(audit, "_validation_point", failed)
    result = run(audit.AuditSettings(0, 1, 100, 30, validation_calls=50))
    assert (
        result.parameters == result.assessment_candidate == result.validation_candidate
    )
    assert result.centered_nll == result.search_best_centered_nll == 0
    assert result.assessment_complete and result.consistent
    assert not result.validation_complete and not result.complete
    assert (
        result.validation_details[0].status
        == "validation_numerical_failure:" + error.__name__
    )
    assert result.validation_details[0].coordinate == 0
    assert dict(result.phase_calls)["validation"] == 0


def test_overflowing_knot_slopes_cannot_claim_a_smooth_jacobian():
    t = np.arange(4) * 1e-308
    k = np.array([0.0, 5e307, 5e307, 0.0])
    assert np.isclose(np.trapezoid(k, t), 1)
    assert audit._shift_region(t, k, 0)[0] == "unrepresentable_shift_slopes"
    point = np.ones(8)
    point[7] = 0
    lower = np.zeros(8)
    lower[7] = -1
    upper = np.full(8, 2.0)
    records = []

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable geometry must stop before stencil evaluation")

    with pytest.raises(audit._TrialFailure, match="shift_geometry_unavailable"):
        audit._residual_jacobian(
            point,
            np.ones(4),
            np.ones(4),
            t,
            k,
            np.ones(8),
            lower,
            upper,
            [7],
            forbidden,
            records.append,
        )
    assert records[-1].geometry == "unrepresentable_shift_slopes"


@pytest.mark.parametrize("solver", ["powell", "trf"])
@pytest.mark.parametrize(
    "reported", ["consistent", "mismatch", "missing", "nan", "shape"]
)
def test_fresh_endpoint_checks_solver_reported_objective(monkeypatch, solver, reported):
    background_seed(monkeypatch)

    def controlled(fun, x, **kwargs):
        result = SimpleNamespace(x=x, success=True)
        if reported != "missing":
            value = (
                1.0 if reported == "mismatch" else np.nan if reported == "nan" else 0.0
            )
            result.fun = np.full(25, value) if solver == "trf" else value
            if reported == "shape":
                result.fun = 0.0 if solver == "trf" else [0.0]
        return result

    monkeypatch.setattr(
        audit, "least_squares" if solver == "trf" else "minimize", controlled
    )
    result = run(
        audit.AuditSettings(
            int(solver == "trf"), int(solver == "powell"), 100, 30, validation_calls=50
        )
    )
    start = result.starts[0]
    assert start.calls == 2
    assert start.endpoint_centered_nll == result.centered_nll == 0.0
    assert result.assessment_complete and result.consistent and result.validation_passed
    assert start.endpoint_objective_consistent == (reported == "consistent")
    assert result.search_complete == result.complete == (reported == "consistent")
    if reported == "mismatch":
        assert start.termination == "endpoint_objective_mismatch"
    elif reported != "consistent":
        assert start.termination == "solver_objective_unavailable"
