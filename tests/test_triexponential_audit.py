"""Independent audit controls and honest completion/cross-score contracts."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
        audit, "minimize", lambda fun, x, **kw: SimpleNamespace(x=x, success=True)
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
