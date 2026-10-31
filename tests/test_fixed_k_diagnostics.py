"""Candidate identity, local probes and descriptive structural sensitivity."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from tcspc_toolkit.estimation._fixed_k_diagnostics import (
    _assess_candidate,
    _component_structure,
    _conditioning_diagnostic,
    _validate_candidate,
    _validation_directions,
)
from test_fixed_k_foundations import direct_components, limits, make_problem, policy


def validation(problem, base, **changes):
    kwargs = dict(steps=(0.01,), absolute_tolerance=1e-9, relative_tolerance=1e-9)
    kwargs.update(changes)
    return _validate_candidate(problem, base, **kwargs)


def assessment(problem, base, **changes):
    return _assess_candidate(
        problem, base, absolute_tolerance=1e-9, relative_tolerance=1e-9, **changes
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_validation_direction_dimension_and_all_pair_directions(k):
    problem, x = make_problem(k)
    directions = _validation_directions(problem.coordinates)
    assert len(directions) == 2 * k + 2 + k * (k - 1)
    assert all(v.shape == x.shape for _, v in directions)
    base = problem.evaluate(x, phase="optimization", origin="test")
    result = validation(problem, base)
    assert result.complete
    assert result.candidate is base
    assert result.submitted_objective == base.centered_nll
    assert len(result.probes) == 2 * len(directions)
    assert problem.work.phases["validation"]["model_calls"] == len(result.probes)
    assert result.best_improvement > 0
    assert result.passed is False
    assert result.best_probe.centered_nll < base.centered_nll
    assert problem.best_finite is base  # probes are not silently selected


def test_truth_expected_count_validation_and_fresh_assessment():
    problem, x = make_problem()
    p = problem.coordinates.decode(x)
    targets = (
        direct_components(problem.inputs.time_ns, problem.inputs.kernel, p).sum(axis=0)
        + p.background
    )
    problem, x = make_problem(targets=targets, parameters=p)
    base = problem.evaluate(x, phase="optimization", origin="truth")
    checked = validation(problem, base)
    assert checked.complete and checked.passed
    result = assessment(problem, base, validation=checked)
    assert result.complete and result.consistent and result.validation_matches
    assert result.validation_passed
    assert result.signed_objective_gap == 0
    assert result.fresh is not base


def test_better_probe_must_be_validated_in_its_own_right():
    problem, x = make_problem()
    base = problem.evaluate(x, phase="optimization", origin="test")
    checked = validation(problem, base)
    probe = checked.best_probe
    result = assessment(problem, probe, validation=checked)
    assert result.fresh.coordinates.tolist() == probe.coordinates.tolist()
    assert not result.validation_matches and result.validation_passed is None
    assert not result.consistent
    assert "validation_for_different_candidate" in result.reasons
    own = validation(problem, probe)
    assert own.candidate is probe


@pytest.mark.parametrize(
    "field", ["centered_nll", "expected_counts", "component_counts", "parameters"]
)
def test_fresh_assessment_detects_corrupt_returned_records(field):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="optimization", origin="test")
    value = getattr(base, field)
    changed = (
        replace(value, background=value.background + 1)
        if field == "parameters"
        else value + 1
    )
    corrupt = replace(base, **{field: changed})
    result = assessment(problem, corrupt)
    assert result.complete and not result.consistent
    if field == "centered_nll":
        assert result.signed_objective_gap == -1


def test_incomplete_validation_is_not_a_pass_and_records_evaluated_work():
    phases = limits().phase_model_calls.copy()
    phases["validation"] = 2
    problem, x = make_problem(work_limits=limits(phase_model_calls=phases))
    base = problem.evaluate(x, phase="optimization", origin="test")
    checked = validation(problem, base)
    assert not checked.complete and checked.passed is None
    assert len(checked.probes) == 3
    assert checked.probes[-1].evaluation.status == "phase_model_limit"
    assert checked.best_probe is not None
    assert problem.work.phases["validation"]["model_calls"] == 2


def test_validation_records_infeasible_directions_at_zero_boundaries():
    problem, x = make_problem(
        1, parameters=_FixedKParameters((0,), (1,), 0, 0.031), targets=np.zeros(65)
    )
    base = problem.evaluate(x, phase="optimization", origin="test")
    checked = validation(problem, base)
    assert checked.complete and checked.passed
    skipped = [probe for probe in checked.probes if not probe.feasible]
    assert len(skipped) == 2
    assert all(probe.evaluation is None for probe in skipped)


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": ()},
        {"steps": (0,)},
        {"absolute_tolerance": -1},
        {"relative_tolerance": np.inf},
    ],
)
def test_validation_rejects_undefined_policy(changes):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="optimization", origin="test")
    with pytest.raises(ValueError):
        validation(problem, base, **changes)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_structural_summaries_keep_nominal_dimension_and_distinct_fractions(k):
    problem, x = make_problem(k)
    base = problem.evaluate(x, phase="optimization", origin="test")
    structure = _component_structure(base)
    assert structure.nominal_parameters == 2 * k + 2
    assert len(structure.lifetime_ratios) == k * (k - 1) // 2
    for fractions in (
        structure.amplitude_fractions,
        structure.infinite_window_area_fractions,
        structure.finite_window_signal_fractions,
    ):
        assert sum(fractions) == pytest.approx(1)
    if k > 1:
        assert structure.amplitude_fractions != structure.infinite_window_area_fractions
    diagnostic = _conditioning_diagnostic(problem, base, policy())
    assert diagnostic.status == "available"
    assert 0 < diagnostic.rank <= min(65, 2 * k + 2)
    assert diagnostic.rank_tolerance > 0
    assert len(diagnostic.singular_values) == 2 * k + 2


def test_equal_and_vanishing_components_reduce_rank_without_reducing_nominal_count():
    p = _FixedKParameters((10, 0, 20), (1, 2, 1), 1, 0.031)
    problem, x = make_problem(parameters=p)
    base = problem.evaluate(x, phase="optimization", origin="test")
    structure = _component_structure(base)
    assert structure.zero_amplitude_components == (1,)
    assert structure.equal_lifetime_pairs == ((0, 2),)
    diagnostic = _conditioning_diagnostic(problem, base, policy())
    assert diagnostic.rank < 8 and diagnostic.n_parameters == 8
    assert np.isinf(diagnostic.condition_number)


def test_zero_signal_fractions_and_nonregular_zero_expectation():
    problem, x = make_problem(
        parameters=_FixedKParameters((0, 0, 0), (1, 1, 1), 0, 0.031),
        targets=np.zeros(65),
    )
    base = problem.evaluate(x, phase="optimization", origin="test")
    structure = _component_structure(base)
    assert structure.amplitude_fractions is None
    assert structure.infinite_window_area_fractions is None
    assert structure.finite_window_signal_fractions is None
    diagnostic = _conditioning_diagnostic(problem, base, policy())
    assert diagnostic.status == "nonregular_zero_expectation"
    assert diagnostic.rank is None


def test_fewer_bins_than_parameters_and_knot_unavailability():
    problem, x = make_problem(4, time=np.arange(5.0), kernel=[0, 1, 0, 0, 0])
    base = problem.evaluate(x, phase="optimization", origin="test")
    diagnostic = _conditioning_diagnostic(problem, base, policy())
    assert diagnostic.rank <= 5
    assert diagnostic.n_parameters == 10
    assert np.isinf(diagnostic.condition_number)
    x = x.copy()
    x[-1] = 0
    knot = problem.evaluate(x, phase="optimization", origin="knot")
    diagnostic = _conditioning_diagnostic(problem, knot, policy())
    assert diagnostic.status == "derivative_unavailable"
    assert diagnostic.rank is None


def test_svd_failure_is_reported(monkeypatch):
    problem, x = make_problem(1)
    base = problem.evaluate(x, phase="optimization", origin="test")

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate")

    monkeypatch.setattr(np.linalg, "svd", fail)
    assert _conditioning_diagnostic(problem, base, policy()).status == "svd_failure"


def test_diagnostic_deadline_after_svd_is_not_reported_as_available(monkeypatch):
    clock = [0.0]
    problem, x = make_problem(
        1, clock=lambda: clock[0], work_limits=limits(wall_seconds=1)
    )
    base = problem.evaluate(x, phase="optimization", origin="test")
    original = np.linalg.svd

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 2
        return result

    monkeypatch.setattr(np.linalg, "svd", slow)
    diagnostic = _conditioning_diagnostic(problem, base, policy())
    assert diagnostic.status == "wall_time_limit"
    assert diagnostic.rank is not None  # Completed facts remain inspectable.
