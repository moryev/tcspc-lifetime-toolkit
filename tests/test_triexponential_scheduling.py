"""Seed-first capacity, exact candidate assessment and independent work counts."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_objective as objective
from tcspc_toolkit.estimation import _fixed_k_search as search
from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from tcspc_toolkit.estimation._fixed_k_diagnostics import _assess_candidate
from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from test_fixed_k_foundations import make_problem
from test_triexponential import fit, settings


def false_success(fun, x, **kwargs):
    return SimpleNamespace(
        x=x, success=True, status=0, message="controlled false success"
    )


def test_seed_coverage_precedes_recovery_and_unused_calls_are_not_loaned(monkeypatch):
    monkeypatch.setattr(solvers, "minimize", false_success)
    validate = search._validate_full

    def needs_recovery(*args):
        # Schedule testing must not depend on which finite seed happens to win.
        return replace(validate(*args), complete=True, passed=False, best_probe=None)

    monkeypatch.setattr(search, "_validate_full", needs_recovery)
    result = fit(np.arange(33) % 4 + 1)
    o = result.optimization
    assert [s.kind for s in o.starts] == ["seed"] * 10 + ["required_recovery"] * 2
    assert tuple(s.seed_id for s in o.starts[:10]) == o.selected_seed_ids
    assert not o.unattempted_seed_ids
    assert all(s.model_call_limit == 32 for s in o.starts[:10])
    assert all(s.model_call_limit == 40 for s in o.starts[10:])
    assert o.recovery_calls_reserved == 80 and o.recovery_calls_used == 2
    assert o.recovery_starts_used == 2
    assert o.recovery_unavailable_reason == "required_recovery_start_limit"
    assert not result.numerical.accepted


@pytest.mark.parametrize("budget", [1, 4, 10, 99])
def test_tiny_global_optimization_budgets_preserve_finite_evidence(budget):
    result = fit(
        np.arange(33) % 4 + 1,
        config=settings(max_optimization_calls=budget, max_diagnostic_calls=0),
    )
    o = result.optimization
    assert result.parameters is not None
    assert o.work.phases["optimization"]["model_calls"] <= budget
    assert (
        sum(s.model_calls for s in o.starts)
        == o.work.phases["optimization"]["model_calls"]
    )
    assert all(s.model_calls <= s.model_call_limit for s in o.starts)
    if budget < 10:
        assert o.unattempted_seed_ids
        assert not result.numerical.accepted


def test_independently_instrumented_work_includes_derivatives_validation_and_assessment(
    monkeypatch,
):
    calls = {"model": 0, "objective": 0}
    physical = objective.multiexponential_reconvolution_component_counts
    poisson = objective.calculate_poisson_deviance

    def counted_model(*args, **kwargs):
        calls["model"] += 1
        return physical(*args, **kwargs)

    def counted_objective(*args, **kwargs):
        calls["objective"] += 1
        return poisson(*args, **kwargs)

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", counted_model
    )
    monkeypatch.setattr(objective, "calculate_poisson_deviance", counted_objective)
    result = fit(np.arange(33) % 4 + 1)
    w = result.optimization.work
    assert w.model_calls == calls["model"]
    assert w.objective_calls == calls["objective"]
    assert (
        w.charged_component_convolutions
        == w.completed_component_convolutions
        == 3 * calls["model"]
    )
    assert sum(p["model_calls"] for p in w.phases.values()) == w.model_calls
    assert sum(p["objective_calls"] for p in w.phases.values()) == w.objective_calls
    assert w.statistics_calculations == 1
    assert sum(w.phase_seconds.values()) <= w.total_seconds
    assert all(v >= 0 for v in w.phase_seconds.values())


def test_validation_anchor_cannot_accept_promoted_probe():
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    check = search._validate_full(problem, base, settings())
    assert check.candidate is base and check.submitted_objective == base.centered_nll
    assert check.complete and not check.passed and check.best_improvement > 0
    best = check.best_probe
    assessment = _assess_candidate(
        problem, best, validation=check, absolute_tolerance=1e-9, relative_tolerance=0
    )
    assert not assessment.consistent
    assert "validation_for_different_candidate" in assessment.reasons
    assert assessment.fresh.parameters == best.parameters
    directions = {p.direction for p in check.probes}
    assert "zero_face:0,1,2,6" in directions
    assert "shift_interval" in directions
    public = search._validation_record(check)
    assert len(public.probes[0].coordinate_direction) == 8
    assert public.best_probe != public.candidate


def test_explicit_promotion_exhaustion_is_rejected_and_reassessed(monkeypatch):
    monkeypatch.setattr(solvers, "minimize", false_success)
    result = fit(np.arange(33) % 4 + 1, config=settings(reserved_recovery_starts=0))
    assert "validation_probe" in result.optimization.promotions
    assert result.numerical.candidate == result.parameters
    assert result.numerical.validation.candidate == result.parameters
    assert "validation_for_different_candidate" not in result.numerical.reasons
    assert result.numerical.recomputed_centered_nll == result.centered_nll
    assert result.identifiability.candidate == result.parameters
    assert (
        result.identifiability.conditioning_status != "candidate_changed_or_unavailable"
    )
    assert result.numerical.fresh_prediction_consistent
    assert result.numerical.validation_matches_candidate
    assert not result.numerical.required_recovery_complete
    assert (
        result.optimization.recovery_unavailable_reason
        == "required_recovery_start_limit"
    )
    assert not result.numerical.accepted


@pytest.mark.parametrize("approximate", [False, True])
def test_neighbor_intervals_use_actual_coordinates(approximate):
    t = np.arange(7.0)
    if approximate:
        t[3:] += 2e-8
    points, unavailable = search._neighbor_shifts(t, 0, (-0.4, 0.4))
    assert not unavailable
    assert 0 in points
    assert any(p < 0 for p in points) and any(p > 0 for p in points)
    assert all(-0.4 <= p <= 0.4 for p in points)
    assert len(points) <= 9
    assert len(points) == len(set(points))


def test_canonical_finite_candidate_survives_assessment_exhaustion():
    problem, x = make_problem()
    x = x.copy()
    x[:3] = x[:3][::-1]
    x[3:6] = x[3:6][::-1]
    base = problem.evaluate(x, phase="initialization", origin="permuted")
    with solvers._lease(problem, "assessment", 0, 10):
        candidate = search._canonical_candidate(problem, base)
        checked = _assess_candidate(
            problem, candidate, absolute_tolerance=0, relative_tolerance=0
        )
    assert candidate.finite and candidate.origin == "canonical_unchecked"
    assert candidate.parameters.lifetimes_ns == tuple(
        sorted(base.parameters.lifetimes_ns)
    )
    assert not checked.consistent


def test_unrepresentable_shift_probe_makes_validation_incomplete(monkeypatch):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="base")
    monkeypatch.setattr(
        search,
        "_neighbor_shifts",
        lambda *args: ((), (-1, 1)),
    )
    check = search._validate_full(problem, base, settings())
    assert not check.complete and check.passed is None
    record = search._validation_record(check)
    assert any(p.status == "unrepresentable_shift_probe" for p in record.probes)


def test_overflowing_feasible_probe_cannot_be_classified_as_out_of_bounds():
    problem, _ = make_problem(targets=np.full(65, 3.0))
    problem.coordinates = replace(
        problem.coordinates, amplitude_scale=1e-308, background_scale=1e-308
    )
    p = _FixedKParameters((1, 0, 0), (1, 2, 3), 1, 0.031)
    candidate = problem.evaluate(
        problem.coordinates.encode(p), phase="initialization", origin="extreme_scale"
    )
    assert candidate.finite
    check = search._validate_full(
        problem, candidate, settings(validation_steps=(1e308,), descent_tolerance=1e308)
    )
    assert not check.complete and check.passed is None
    probes = search._validation_record(check).probes
    assert any(p.feasible and p.status == "unrepresentable_probe" for p in probes)
    # The negative step reaches the zero face before the positive overflow.
    assert probes[0].parameters.amplitudes[0] == 0
    assert probes[0].feasible


def test_coordinate_step_lost_in_physical_shift_is_enlarged_before_evaluation():
    problem, _ = make_problem(targets=np.full(65, 3.0))
    problem.coordinates = replace(
        problem.coordinates, temporal_shift_bounds_ns=(1e16, 1e16 + 4)
    )
    p = _FixedKParameters((0, 0, 0), (1, 2, 3), 3, 1e16 + 2)
    candidate = problem.evaluate(
        problem.coordinates.encode(p), phase="initialization", origin="rounded_shift"
    )
    assert candidate.finite and candidate.centered_nll == 0
    check = search._validate_full(problem, candidate, settings())
    assert check.complete and check.passed
    probes = [
        p for p in search._validation_record(check).probes if p.direction == "shift"
    ]
    assert probes
    assert {p.parameters.temporal_shift_ns for p in probes} == {1e16, 1e16 + 4}
    assert all(p.status == "finite" for p in probes)


def test_global_deadline_retains_last_completed_candidate_and_blocks_fresh_acceptance(
    monkeypatch,
):
    clock = [0.0]
    calls = [0]
    physical = objective.multiexponential_reconvolution_component_counts

    def slow(*args, **kwargs):
        result = physical(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 3:
            clock[0] = 2
        return result

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", slow
    )
    problem, _ = make_problem()
    result = search._fit_triexponential(
        problem.inputs.time_ns,
        problem.inputs.targets,
        problem.inputs.kernel,
        lifetime_bounds_ns=(0.1, 10),
        temporal_shift_bounds_ns=(-0.4, 0.4),
        settings=settings(wall_seconds=1),
        target_kind="deterministic_expected_counts",
        clock=lambda: clock[0],
    )
    assert result.parameters is not None
    assert not result.numerical.accepted
    assert "wall_time_limit" in result.numerical.reasons
    assert result.optimization.work.model_calls == 3
    assert not result.optimization.initialization_complete
