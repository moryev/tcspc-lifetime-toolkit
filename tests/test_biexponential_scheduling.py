"""Deterministic scheduling controls, independent of qualification observations."""

from dataclasses import replace
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tcspc_toolkit.benchmarks.multiexponential import assess_biexponential_fit
from tcspc_toolkit.estimation import multiexponential as fitting
from test_biexponential_reconvolution import BOUNDS, independent_case


ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "configs/issue28_pr3_qualification.json").read_text())[
    "acceptance"
]


def stationary(fun, x, **kwargs):
    fun(x)
    return OptimizeResult(x=x, success=True, status=0, message="control")


def invalid_trial(fun, x, **kwargs):
    fun(x)
    fun(np.zeros(6))
    pytest.fail("An unusable trial must unwind the solver")


def controlled_fit(monkeypatch, backgrounds, solver=stationary, **settings):
    time, kernel, *_ = independent_case(bins=32)
    counts = np.full(time.size, 2)

    def bank(problem, guesses):
        problem.starts_screened = len(backgrounds)
        return [
            problem.coordinates(fitting.BiexponentialParameters((0, 0), (1, 3), b, 0))
            for b in backgrounds
        ]

    monkeypatch.setattr(fitting, "_initial_points", bank)
    monkeypatch.setattr(fitting, "minimize", solver)
    result = fitting.fit_biexponential_reconvolution(
        time,
        counts,
        kernel,
        **BOUNDS,
        settings=fitting.BiexponentialFitSettings(**settings),
    )
    case = dict(time=time, counts=counts, irf=kernel)
    return result, case


def assess(result, case):
    return assess_biexponential_fit(
        result, case["time"], case["counts"], case["irf"], RULES
    )


def test_exhaustive_attempts_each_seed_and_retains_failed_step_best(monkeypatch):
    result, case = controlled_fit(monkeypatch, [4, 6, 8], invalid_trial, max_starts=3)
    opt = result.optimization
    assert [s.initial_parameters.background for s in opt.starts] == [4, 6, 8]
    assert [s.kind for s in opt.starts] == ["seed"] * 3
    assert all(s.optimizer_reported_success is None for s in opt.starts)
    assert opt.optional_continuations_omitted == (0, 1, 2)
    assert opt.finite_seed_candidates == 3
    assert opt.unattempted_seed_candidates == 0
    # The first failed invocation's Armijo probe reaches the exact optimum.
    assert result.parameters.background == 2
    assert result.statistics.centered_negative_log_likelihood == 0
    assert opt.selected_start_index == 0
    assert opt.numerical_validation_passed is True
    assert opt.stopping_reason == "candidate_bank_completed"
    assert opt.recovery_unavailable_reason is None
    assert not opt.exhausted_budgets
    assert opt.objective_evaluations == sum(s.objective_evaluations for s in opt.starts)
    assert assess(result, case).accepted


def test_exactly_46_complete_starts_with_optional_requests_are_not_exhaustion(
    monkeypatch,
):
    result, case = controlled_fit(monkeypatch, [4] * 46, invalid_trial)
    opt = result.optimization
    assert len(opt.starts) == 46
    assert len(opt.optional_continuations_omitted) == 46
    assert opt.unattempted_seed_candidates == 0
    assert opt.stopping_reason == "candidate_bank_completed"
    assert not opt.exhausted_budgets
    assert assess(result, case).accepted


def test_incomplete_bank_fails_even_when_best_candidate_validates(monkeypatch):
    result, case = controlled_fit(monkeypatch, [2] * 4, max_starts=3)
    opt = result.optimization
    assert opt.numerical_validation_passed is True
    assert opt.finite_seed_candidates == 4
    assert opt.unattempted_seed_candidates == 1
    assert opt.stopping_reason == "start_budget_exhausted"
    assert opt.exhausted_budgets == ("starts",)
    assessment = assess(result, case)
    assert not assessment.accepted
    assert set(assessment.reasons) == {
        "required_stage_budget_exhausted",
        "required_fitting_stages_incomplete",
    }


def test_required_recovery_uses_spare_slot_only_after_bank(monkeypatch):
    calls = 0

    def solver(fun, x, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            x = x.copy()
            x[4] = 1  # background scale is 2; exact target is background 2.
        return stationary(fun, x, **kwargs)

    result, case = controlled_fit(monkeypatch, [6, 4], solver, max_starts=3)
    opt = result.optimization
    assert [s.kind for s in opt.starts] == ["seed", "seed", "validation_recovery"]
    # Two zero amplitudes skip six clipped, unchanged probes per validation.
    assert opt.validation_evaluations == 86
    assert opt.numerical_validation_passed is True
    assert not opt.exhausted_budgets
    assert assess(result, case).accepted


def test_failed_validation_at_start_cap_is_required_work(monkeypatch):
    result, case = controlled_fit(monkeypatch, [4, 6], max_starts=2)
    opt = result.optimization
    assert opt.unattempted_seed_candidates == 0
    assert opt.numerical_validation_passed is False
    assert opt.recovery_unavailable_reason == "start_budget_exhausted"
    assert opt.stopping_reason == "recovery_start_budget_exhausted"
    assert opt.exhausted_budgets == ("starts",)
    assert not assess(result, case).accepted


def test_recovery_limit_is_still_two_per_tier(monkeypatch):
    result, case = controlled_fit(
        monkeypatch, [4], max_starts=10, max_validation_evaluations=200
    )
    opt = result.optimization
    assert [s.kind for s in opt.starts] == ["seed"] + ["validation_recovery"] * 2
    assert opt.recovery_unavailable_reason == "validation_recovery_limit_reached"
    assert opt.numerical_validation_passed is False
    assert opt.validation_evaluations == 129
    assert not assess(result, case).accepted


@pytest.mark.parametrize(
    "settings,phase,field,limit",
    [
        ({"max_objective_evaluations": 1}, "optimization", "objective_evaluations", 1),
        ({"max_validation_evaluations": 1}, "validation", "validation_evaluations", 1),
        ({"max_diagnostic_evaluations": 1}, "diagnostics", "diagnostic_evaluations", 1),
    ],
)
def test_phase_exhaustion_is_not_hidden_by_seed_completion(
    monkeypatch, settings, phase, field, limit
):
    result, case = controlled_fit(monkeypatch, [2], **settings)
    opt = result.optimization
    assert getattr(opt, field) == limit
    assert phase in opt.exhausted_budgets
    assert assess(result, case).accepted is (phase == "diagnostics")
    assert opt.objective_evaluations == sum(s.objective_evaluations for s in opt.starts)


@pytest.mark.parametrize("policy", ["economical", "adaptive", "exhaustive"])
def test_incomplete_initialization_remains_required_budget_failure(policy):
    time, kernel, *_ = independent_case(bins=32)
    counts = np.full(time.size, 2)
    result = fitting.fit_biexponential_reconvolution(
        time,
        counts,
        kernel,
        **BOUNDS,
        settings=fitting.BiexponentialFitSettings(
            policy=policy, max_initialization_evaluations=3, max_starts=1
        ),
    )
    assert result.optimization.initialization_evaluations == 3
    assert "initialization" in result.optimization.exhausted_budgets
    checked = assess(result, dict(time=time, counts=counts, irf=kernel))
    assert not checked.accepted
    assert "required_stage_budget_exhausted" in checked.reasons


@pytest.mark.parametrize(
    "policy,seeds,stop",
    [
        ("economical", 4, "economical_policy_completed"),
        ("adaptive", 8, "adaptive_stable_tiers"),
        ("exhaustive", 12, "candidate_bank_completed"),
    ],
)
def test_policy_seed_tiers_and_early_stops_remain_explicit(
    monkeypatch, policy, seeds, stop
):
    result, case = controlled_fit(monkeypatch, [2] * 12, policy=policy)
    opt = result.optimization
    assert len(opt.starts) == seeds
    assert opt.finite_seed_candidates == 12
    assert opt.unattempted_seed_candidates == 12 - seeds
    assert opt.stopping_reason == stop
    assert not opt.exhausted_budgets
    assert not opt.optional_continuations_omitted
    assert opt.validation_evaluations == (86 if policy == "adaptive" else 43)
    assert assess(result, case).accepted is (policy == "exhaustive")


def test_regression_reporting_uses_full_gate_and_separate_external_call(monkeypatch):
    report = runpy.run_path(str(ROOT / "scripts/regress_issue33_optimizer.py"))[
        "assess_result"
    ]
    result, case = controlled_fit(monkeypatch, [2])
    original = result.optimization
    good = report(result, case, RULES)
    assert good["numerical_acceptance"] is True
    assert good["local_numerical_validation_passed"] is True
    assert good["numerical_assessment"]["external_forward_evaluations"] == 1
    assert good["total_forward_evaluations"] == good["fitter_forward_evaluations"] + 1
    assert good["total_seconds_including_assessment"] >= original.total_seconds
    assert result.optimization is original
    incomplete = replace(
        result,
        optimization=replace(
            original,
            stopping_reason="start_budget_exhausted",
            exhausted_budgets=("starts",),
        ),
    )
    bad = report(incomplete, case, RULES)
    assert bad["local_numerical_validation_passed"] is True
    assert bad["numerical_acceptance"] is False
    invalid = replace(result, parameters=replace(result.parameters, background=0))
    checked = report(invalid, case, RULES)
    assert checked["numerical_acceptance"] is False
    assert "objective_recheck_failed" in checked["numerical_assessment"]["reasons"]
    assert checked["numerical_assessment"]["external_forward_evaluations"] == 1


def test_regression_source_snapshots_preserve_same_named_modules(tmp_path, monkeypatch):
    snapshot = runpy.run_path(str(ROOT / "scripts/regress_issue33_optimizer.py"))[
        "save_source_snapshots"
    ]
    monkeypatch.setitem(snapshot.__globals__, "ROOT", tmp_path)
    sources = [tmp_path / group / "multiexponential.py" for group in ("fit", "assess")]
    for index, source in enumerate(sources):
        source.parent.mkdir()
        source.write_text(str(index))
    output = tmp_path / "record"
    snapshot(sources, output)
    for source in sources:
        assert (
            output / "sources" / source.relative_to(tmp_path)
        ).read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        snapshot(sources, output)
