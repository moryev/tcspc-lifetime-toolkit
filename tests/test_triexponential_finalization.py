"""PR 2C-1 finalization contracts, using new controlled mathematical fixtures."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_search as search
from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from tcspc_toolkit.estimation._fixed_k_diagnostics import _conditioning_diagnostic
from test_fixed_k_foundations import make_problem, policy, limits
from test_triexponential import fit, settings
from test_triexponential_scheduling import false_success


def test_conditioning_runs_after_assessment_on_exact_promoted_return(monkeypatch):
    monkeypatch.setattr(solvers, "minimize", false_success)
    events = []
    assess, condition = search._assess_candidate, search._conditioning_diagnostic

    def assessed(problem, candidate, **kw):
        events.append(("assessment", candidate))
        return assess(problem, candidate, **kw)

    def conditioned(problem, candidate, policy):
        events.append(("conditioning", candidate))
        return condition(problem, candidate, policy)

    monkeypatch.setattr(search, "_assess_candidate", assessed)
    monkeypatch.setattr(search, "_conditioning_diagnostic", conditioned)
    r = fit(np.arange(33) % 4 + 1, config=settings(reserved_recovery_starts=0))
    assert r.optimization.promotions == ("validation_probe",)
    assert [e[0] for e in events] == ["assessment", "conditioning"]
    assert events[0][1] is events[1][1]
    assert r.identifiability.candidate == r.parameters == r.numerical.candidate
    assert r.identifiability.candidate_evaluation_id == events[1][1].evaluation_id
    assert r.numerical.fresh_prediction_consistent
    assert r.numerical.fresh_prediction_reasons == ()
    assert not r.numerical.validation_matches_candidate
    assert not r.numerical.required_recovery_complete
    assert not r.numerical.accepted and not r.numerical.consistent


@pytest.mark.parametrize("improvement", [0.0005, 0.002])
def test_diagnostic_evidence_cannot_promote_or_inherit_acceptance(
    monkeypatch, improvement
):
    monkeypatch.setattr(solvers, "minimize", false_success)
    original = search._conditioning_diagnostic

    def diagnostic(problem, candidate, policy):
        result = original(problem, candidate, policy)
        # Controlled observer evidence tests finalization, not objective accuracy.
        point = candidate.coordinates.copy()
        point[-2] += 0.1
        probe = problem.evaluate(point, phase="diagnostics", origin="controlled_probe")
        probe = replace(probe, centered_nll=candidate.centered_nll - improvement)
        problem.evaluation_observer(probe)
        return result

    monkeypatch.setattr(search, "_conditioning_diagnostic", diagnostic)
    r = fit()
    assert r.parameters.amplitudes == (0, 0, 0) and r.parameters.background == 0
    assert r.centered_nll == 0 and not r.optimization.promotions
    assert r.numerical.diagnostic_best_centered_nll == -improvement
    assert r.numerical.diagnostic_best_parameters != r.parameters
    assert r.numerical.diagnostic_improvement == improvement
    assert r.numerical.accepted == (improvement <= settings().descent_tolerance)
    assert ("diagnostic_descent_detected" in r.numerical.reasons) == (
        improvement > 0.001
    )
    assert r.numerical.fresh_prediction_consistent
    assert r.numerical.validation_matches_candidate


def test_stale_conditioning_is_not_attached(monkeypatch):
    original = search._conditioning_diagnostic

    def stale(problem, candidate, policy):
        other = problem.evaluate(
            candidate.coordinates, phase="diagnostics", origin="different_anchor"
        )
        return original(problem, other, policy)

    monkeypatch.setattr(search, "_conditioning_diagnostic", stale)
    r = fit()
    assert r.identifiability.candidate is None
    assert r.identifiability.candidate_evaluation_id is None
    assert r.identifiability.conditioning_status == "candidate_changed_or_unavailable"
    assert r.numerical.objective_gradient_available is None


def test_valid_zero_objective_and_gradient_do_not_claim_residual_jacobian():
    r = fit()
    assert r.numerical.accepted and r.centered_nll == 0
    assert r.numerical.fresh_prediction_consistent
    assert r.numerical.objective_gradient_available
    assert not r.numerical.residual_jacobian_available
    assert r.identifiability.conditioning_status == "nonregular_zero_expectation"


@pytest.mark.parametrize("endpoint", [0, 1])
def test_restricted_diagnostic_labels_fixed_shift_and_preserves_full_unavailability(
    endpoint,
):
    p = _FixedKParameters((4, 0, 6), (0.8, 1.2, 0.8), 1, 0)
    problem, x = make_problem(
        time=np.arange(7.0), kernel=[endpoint, 1, 2, 3, 2, 1, endpoint], parameters=p
    )
    base = problem.evaluate(x, phase="optimization", origin="new_restricted_fixture")
    c = _conditioning_diagnostic(problem, base, policy())
    assert c.status == "derivative_unavailable" and c.rank is None
    assert c.candidate_id == base.evaluation_id
    assert c.restricted.fixed_coordinate_names == ("shift",)
    assert len(c.restricted.coordinate_names) == 7
    assert c.restricted.rank < 7
    assert np.isinf(c.restricted.condition_number)


def test_diagnostic_exhaustion_is_explicit_and_does_not_invent_rank():
    problem, x = make_problem()
    phase = dict(limits().phase_model_calls)
    phase["diagnostics"] = 0
    problem.work.limits = limits(phase_model_calls=phase)
    base = problem.evaluate(x, phase="optimization", origin="budget_fixture")
    c = _conditioning_diagnostic(problem, base, policy())
    assert c.status == "diagnostic_budget_exhausted" and c.rank is None
    assert problem.work.phases["diagnostics"]["model_calls"] == 0
    assert problem.recent_failures[-1].candidate_id == base.evaluation_id
    assert all(d.value is None for d in c.derivative.objective.derivatives[:3])


def test_full_result_diagnostic_budget_preserves_assessment_and_unavailability():
    r = fit(config=settings(max_diagnostic_calls=0))
    assert r.numerical.fresh_prediction_consistent
    assert r.numerical.validation_matches_candidate
    assert r.identifiability.conditioning_status == "diagnostic_budget_exhausted"
    assert r.identifiability.candidate == r.parameters
    assert r.optimization.work.phases["diagnostics"]["model_calls"] == 0
    assert r.identifiability.weighted_jacobian_rank is None
