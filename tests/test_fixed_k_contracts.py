"""Work ceilings, exact Poisson limits, provenance and compatibility boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_objective as objective
from tcspc_toolkit.estimation._fixed_k import _FixedKInputs, _FixedKParameters
from tcspc_toolkit.estimation._fixed_k_diagnostics import _assess_candidate
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import IRFProfile, IRFSourceKind
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_expected_counts,
)
from test_fixed_k_foundations import limits, make_problem, policy


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_existing_forward_model_exact_compatibility_and_permutation(k):
    problem, x = make_problem(k)
    base = problem.evaluate(x, phase="optimization", origin="test")
    p = base.parameters
    expected = multiexponential_reconvolution_expected_counts(
        problem.inputs.time_ns,
        problem.inputs.kernel,
        amplitudes=p.amplitudes,
        lifetimes=p.lifetimes_ns,
        background=p.background,
        temporal_shift=p.temporal_shift_ns,
    )
    np.testing.assert_array_equal(base.expected_counts, expected)
    permuted = replace(
        p, amplitudes=p.amplitudes[::-1], lifetimes_ns=p.lifetimes_ns[::-1]
    )
    other = problem.evaluate(
        problem.coordinates.encode(permuted), phase="optimization", origin="permuted"
    )
    np.testing.assert_allclose(base.expected_counts, other.expected_counts, rtol=5e-16)
    assert other.centered_nll == pytest.approx(base.centered_nll, rel=1e-13)


@pytest.mark.parametrize(
    "targets,expected_status,value",
    [(np.zeros(65), "finite", 0), (np.ones(65), "infinite_objective", np.inf)],
)
def test_exact_zero_expectation_limits(targets, expected_status, value):
    problem, x = make_problem(
        parameters=_FixedKParameters((0, 0, 0), (1, 2, 3), 0, 0.031), targets=targets
    )
    base = problem.evaluate(x, phase="optimization", origin="zero")
    assert base.status == expected_status
    assert base.centered_nll == value
    np.testing.assert_array_equal(base.expected_counts, np.zeros(65))


def test_centered_objective_near_equality_has_independent_high_precision_oracle():
    from decimal import Decimal, localcontext

    problem, x = make_problem(
        1,
        parameters=_FixedKParameters((0,), (1,), 2, 0.031),
        targets=np.full(65, 2 + 1e-9),
    )
    base = problem.evaluate(x, phase="optimization", origin="close")
    with localcontext() as context:
        context.prec = 60
        y, mu = Decimal(float(problem.inputs.targets[0])), Decimal(2)
        reference = float(65 * (mu - y + y * (y / mu).ln()))
    assert base.centered_nll == pytest.approx(reference, rel=3e-15)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"model_calls": 1}, "model_limit"),
        ({"objective_calls": 1}, "objective_limit"),
        ({"component_convolutions": 3}, "component_work_limit"),
        ({"phase_model_calls": {p: 1 for p in objective._PHASES}}, "phase_model_limit"),
    ],
)
def test_hard_work_limits_and_best_finite_retention(changes, reason):
    problem, x = make_problem(work_limits=limits(**changes))
    first = problem.evaluate(x, phase="optimization", origin="first")
    second = problem.evaluate(x, phase="optimization", origin="blocked")
    assert first.finite and second.status == reason
    assert problem.best_finite is first
    assert problem.work.requests == 2
    assert problem.work.model_calls == problem.work.objective_calls == 1
    assert problem.work.charged_component_convolutions == 3
    assert problem.work.completed_component_convolutions == 3
    assert problem.work.blocked == [("optimization", reason)]


def test_infeasible_request_costs_no_model_work():
    problem, x = make_problem()
    x = x.copy()
    x[0] = -1
    result = problem.evaluate(x, phase="optimization", origin="infeasible")
    assert result.status == "infeasible_coordinates"
    assert problem.work.requests == 1 and problem.work.model_calls == 0
    with pytest.raises(ValueError, match="phase"):
        problem.evaluate(x, phase="hidden", origin="bad")


def test_deadline_is_cooperative_and_retains_completed_finite_work(monkeypatch):
    clock = [0.0]
    problem, x = make_problem(
        clock=lambda: clock[0], work_limits=limits(wall_seconds=1)
    )
    original = objective.multiexponential_reconvolution_component_counts

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 2
        return result

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", slow
    )
    first = problem.evaluate(x, phase="optimization", origin="slow")
    assert first.finite and problem.best_finite is first
    assert problem.work.deadline_reached and problem.work.elapsed_seconds == 2
    assert (
        problem.evaluate(x, phase="optimization", origin="blocked").status
        == "wall_time_limit"
    )
    assessment = _assess_candidate(
        problem, first, absolute_tolerance=0, relative_tolerance=0
    )
    assert not assessment.complete and not assessment.consistent


@pytest.mark.parametrize("failure", ["raise", "nan", "negative", "shape"])
def test_failed_model_work_is_charged_and_never_gives_false_objective(
    failure, monkeypatch
):
    problem, x = make_problem()

    def failed(*args, **kwargs):
        if failure == "raise":
            raise FloatingPointError("deliberate")
        return np.full(
            (3, 65 if failure != "shape" else 64),
            np.nan if failure == "nan" else -1 if failure == "negative" else 0,
        )

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", failed
    )
    result = problem.evaluate(x, phase="optimization", origin="failure")
    assert not result.finite and result.centered_nll is None
    assert problem.work.model_calls == 1
    assert problem.work.charged_component_convolutions == 3
    assert problem.work.objective_attempts == 1 and problem.work.objective_calls == 0
    assert problem.best_finite is None


def test_programming_errors_are_not_hidden(monkeypatch):
    problem, x = make_problem()

    def fail(*args, **kwargs):
        raise RuntimeError("programming error")

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", fail
    )
    with pytest.raises(RuntimeError, match="programming"):
        problem.evaluate(x, phase="optimization", origin="failure")


def test_all_derivative_work_is_phase_charged():
    problem, x = make_problem(2)
    base = problem.evaluate(x, phase="initialization", origin="base")
    derivative = objective._objective_derivatives(
        problem, base, policy(), phase="diagnostics"
    )
    assert derivative.complete
    assert problem.work.phases["optimization"]["model_calls"] == 0
    assert problem.work.phases["initialization"]["model_calls"] == 1
    for key in ("requests", "model_calls", "objective_calls", "objective_attempts"):
        assert getattr(problem.work, key) == sum(
            v[key] for v in problem.work.phases.values()
        )
    assert problem.work.charged_component_convolutions == sum(
        v["component_convolutions"] for v in problem.work.phases.values()
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model_calls": 0},
        {"objective_calls": True},
        {"wall_seconds": 0},
        {"phase_model_calls": {}},
        {"phase_model_calls": dict.fromkeys(objective._PHASES, -1)},
    ],
)
def test_work_limits_are_explicit_validated_and_snapshotted(changes):
    with pytest.raises(ValueError):
        limits(**changes)


def test_prepared_irf_provenance_is_snapshotted_without_repreparing():
    t = np.arange(5.0)
    source = IRFProfile(
        t,
        [0, 2, 0, 0, 0],
        IRFSourceKind.IMPORTED_SAMPLED,
        metadata={"label": "measured"},
        provenance={"path": "example.csv"},
    )
    prepared = prepare_irf(source, t)
    inputs = _FixedKInputs(t, np.ones(5), prepared, "raw_counts")
    assert inputs.irf is not prepared
    np.testing.assert_array_equal(inputs.kernel, prepared.kernel)
    assert inputs.irf.diagnostics == prepared.diagnostics
    assert dict(inputs.irf.source.provenance) == {"path": "example.csv"}
    assert inputs.irf.source.source_kind is IRFSourceKind.IMPORTED_SAMPLED
    with pytest.raises(ValueError, match="grid"):
        _FixedKInputs(t + 0.1, np.ones(5), prepared, "raw_counts")


def test_foreign_problem_candidate_cannot_inherit_diagnostics():
    first, x = make_problem()
    second, _ = make_problem(targets=np.ones(65))
    base = first.evaluate(x, phase="optimization", origin="first")
    derivative = objective._objective_derivatives(
        second, base, policy(), phase="diagnostics"
    )
    assert derivative.gradient is None
    result = _assess_candidate(second, base, absolute_tolerance=0, relative_tolerance=0)
    assert not result.consistent and "different_problem" in result.reasons


def test_mathematical_infinity_and_unrepresentable_finite_objective_are_distinct():
    problem, x = make_problem(
        1, parameters=_FixedKParameters((0,), (1,), 1e308, 0.031), targets=np.zeros(65)
    )
    result = problem.evaluate(x, phase="optimization", origin="overflow")
    assert result.status == "objective_overflow" and result.centered_nll == np.inf
    assert np.all(np.isfinite(result.expected_counts))


def test_nonfinite_expectation_and_parameter_overflow_are_explicit():
    problem, x = make_problem(1)
    x = x.copy()
    x[0] = np.finfo(float).max
    result = problem.evaluate(x, phase="optimization", origin="overflow")
    assert result.status == "unrepresentable_parameters"
    assert problem.work.model_calls == 0
    problem, x = make_problem(3)
    problem.coordinates = replace(problem.coordinates, amplitude_scale=1)
    x = x.copy()
    x[:3] = 1e308
    result = problem.evaluate(x, phase="optimization", origin="overflow")
    assert result.status == "nonfinite_expectation"
    assert result.centered_nll is None


def test_ledger_matches_independently_instrumented_physics_and_objective(monkeypatch):
    from tcspc_toolkit.estimation._fixed_k_diagnostics import (
        _validate_candidate,
        _conditioning_diagnostic,
    )

    calls = {"model": 0, "poisson": 0}
    model = objective.multiexponential_reconvolution_component_counts
    poisson = objective.calculate_poisson_deviance

    def counted_model(*args, **kwargs):
        calls["model"] += 1
        return model(*args, **kwargs)

    def counted_poisson(*args, **kwargs):
        calls["poisson"] += 1
        return poisson(*args, **kwargs)

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", counted_model
    )
    monkeypatch.setattr(objective, "calculate_poisson_deviance", counted_poisson)
    problem, x = make_problem(3)
    base = problem.evaluate(x, phase="initialization", origin="test")
    objective._objective_derivatives(problem, base, policy(), phase="optimization")
    checked = _validate_candidate(
        problem, base, steps=(0.01,), absolute_tolerance=0, relative_tolerance=0
    )
    _assess_candidate(
        problem, base, validation=checked, absolute_tolerance=0, relative_tolerance=0
    )
    _conditioning_diagnostic(problem, base, policy())
    assert problem.work.model_calls == calls["model"]
    assert problem.work.objective_calls == calls["poisson"]
    assert problem.work.charged_component_convolutions == 3 * calls["model"]
    assert all(row["model_calls"] > 0 for row in problem.work.phases.values())
