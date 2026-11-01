"""109 templates, protected diversity, scaled NNLS and explicit failures."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_initialization as initialization
from tcspc_toolkit.estimation._fixed_k_solvers import _lease
from test_fixed_k_foundations import make_problem
from test_triexponential import settings


def test_templates_and_nested_deterministic_selection():
    selections = []
    for capacity in (16, 32, 109):
        problem, _ = make_problem()
        templates = list(initialization._templates(problem.coordinates))
        assert len(templates) == 108
        assert sum(not t[-1] for t in templates) == 105
        bank = initialization._build_seed_bank(
            problem, settings(max_seed_starts=capacity), []
        )
        assert bank.generated == 109
        assert bank.projections == 105
        assert bank.complete and not bank.missing
        assert len(bank.selected) <= capacity
        assert bank.screened == 109 + bank.repairs
        assert problem.work.model_calls == 218 + bank.repairs
        assert problem.work.objective_calls == 109 + bank.repairs
        for bucket in bank.buckets.values():
            assert {s[0] for s in bank.selected}.intersection(s[0] for s in bucket)
        selections.append([s[0] for s in bank.selected])
    assert selections[0] == selections[1][:16] == selections[2][:16]
    assert selections[1] == selections[2][:32]


def test_projection_is_scaled_and_matches_independent_nonnegative_solution():
    problem, _ = make_problem(targets=np.full(65, 4.0))
    bank = initialization._build_seed_bank(problem, settings(), [])
    background = next(e for n, e in bank.candidates if n == "background")
    assert background.parameters.amplitudes == (0, 0, 0)
    assert background.centered_nll == 0
    assert all(
        e.parameters.background >= 0 and min(e.parameters.amplitudes) >= 0
        for _, e in bank.candidates
    )


def test_nnls_failure_is_counted_without_retry(monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("controlled NNLS failure")

    monkeypatch.setattr(initialization, "nnls", broken)
    problem, _ = make_problem()
    bank = initialization._build_seed_bank(problem, settings(), [])
    assert bank.projection_failures == bank.projections == len(calls) == 105
    assert all(c == {"maxiter": 100} for c in calls)
    assert bank.complete
    assert all(e.finite for _, e in bank.selected)


def test_canonical_duplicates_do_not_spend_multiple_seed_slots():
    problem, x = make_problem()
    p = problem.coordinates.decode(x)
    q = replace(p, amplitudes=p.amplitudes[::-1], lifetimes_ns=p.lifetimes_ns[::-1])
    bank = initialization._build_seed_bank(
        problem, settings(max_seed_starts=11), [p, q]
    )
    assert bank.duplicates >= 1
    assert sum(name.startswith("supplied") for name, _ in bank.selected) == 1


def test_initializer_lease_and_scaling_fallback_are_explicit():
    problem, _ = make_problem()
    with _lease(problem, "initialization", 0, 10):
        bank = initialization._build_seed_bank(problem, settings(), [])
    assert bank.generated == 109 and bank.scaling_fallback
    assert not bank.complete and len(bank.missing) == 10
    assert not bank.selected and problem.work.model_calls == 0


def test_capacity_overflow_does_not_drop_supplied_guesses():
    problem, x = make_problem()
    p = problem.coordinates.decode(x)
    with pytest.raises(ValueError, match="protected coverage"):
        initialization._build_seed_bank(problem, settings(), [p])
