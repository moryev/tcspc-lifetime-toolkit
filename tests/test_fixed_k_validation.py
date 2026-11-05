"""Scalar coverage and terminal promotion on small, newly defined fixtures."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation import _fixed_k_search as search
from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from tcspc_toolkit.estimation._fixed_k import _FixedKParameters, _readonly
from tcspc_toolkit.estimation._fixed_k_diagnostics import (
    _LocalValidation,
    _ValidationProbe,
    _conditioning_diagnostic,
    _validate_candidate,
    _validation_trial,
)
from tcspc_toolkit.estimation._fixed_k_initialization import _SeedBank
from test_fixed_k_foundations import make_problem, policy, limits
from test_triexponential import fit, settings
from test_triexponential_scheduling import false_success


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("ulps", [0, 1, 5])
@pytest.mark.parametrize("coordinate", ["lifetime", "shift"])
def test_exact_and_ulp_bounds_have_distinct_inward_scalar_probes(
    k, side, ulps, coordinate
):
    problem, x = make_problem(k)
    mapping = problem.coordinates
    index = k if coordinate == "lifetime" else mapping.dimension - 1
    bound = mapping.lower[index] if side < 0 else mapping.upper[index]
    x = x.copy()
    for _ in range(ulps):
        bound = np.nextafter(bound, -side * np.inf)
    x[index] = bound
    base = problem.evaluate(x, phase="initialization", origin="bound_fixture")
    direction = np.eye(mapping.dimension)[index]
    trial, feasible = _validation_trial(mapping, base, direction, -side * 0.01)
    assert feasible and mapping.feasible(trial)
    decoded = mapping.decode(trial)
    before = (
        base.parameters.lifetimes_ns[0]
        if coordinate == "lifetime"
        else base.parameters.temporal_shift_ns
    )
    after = (
        decoded.lifetimes_ns[0]
        if coordinate == "lifetime"
        else decoded.temporal_shift_ns
    )
    assert (after > before) if side < 0 else (after < before)
    assert problem.work.phases["validation"]["model_calls"] == 0
    outward, outward_feasible = _validation_trial(mapping, base, direction, side * 0.01)
    physical_bound = (
        mapping.lifetime_bounds_ns[int(side > 0)]
        if coordinate == "lifetime"
        else mapping.temporal_shift_bounds_ns[int(side > 0)]
    )
    if before == physical_bound:
        assert not outward_feasible and outward is None
    else:
        assert outward_feasible
        if outward is not None:
            assert mapping.decode(outward) != base.parameters


def test_rounded_endpoint_alias_is_redundant_without_a_same_point_model_call(
    monkeypatch,
):
    problem, x = make_problem()
    x = x.copy()
    x[5] = np.nextafter(problem.coordinates.upper[5], -np.inf)
    base = problem.evaluate(x, phase="initialization", origin="endpoint_alias")
    assert base.parameters.lifetimes_ns[2] == 10.0
    original = problem.evaluate
    evaluated = []

    def distinct(point, **kwargs):
        assert problem.coordinates.decode(point) != base.parameters
        evaluated.append(tuple(point))
        return original(point, **kwargs)

    monkeypatch.setattr(problem, "evaluate", distinct)
    result = search._validate_full(problem, base, settings())
    assert result.complete
    endpoint = [
        p
        for p in search._validation_record(result).probes
        if p.direction == "bound:log_lifetime_3"
    ]
    assert any(p.status == "redundant_physical_candidate" for p in endpoint)
    assert problem.work.phases["validation"]["model_calls"] == len(evaluated)


@pytest.mark.parametrize("step", [np.nextafter(0.0, 1.0), 1e-20])
def test_rounding_lost_scalar_steps_enlarge_and_check_every_transfer_component(step):
    problem, x = make_problem()
    base = problem.evaluate(x, phase="initialization", origin="tiny_step")
    direction = np.zeros(8)
    direction[3:5] = (1, -1)
    trial, feasible = _validation_trial(problem.coordinates, base, direction, step)
    assert feasible
    decoded = problem.coordinates.decode(trial)
    assert decoded.lifetimes_ns[0] > base.parameters.lifetimes_ns[0]
    assert decoded.lifetimes_ns[1] < base.parameters.lifetimes_ns[1]
    assert decoded.lifetimes_ns[2] == base.parameters.lifetimes_ns[2]
    assert decoded.amplitudes == base.parameters.amplitudes


@pytest.mark.parametrize("shift", [0.017, -0.019, 0.039])
def test_neighbors_use_actual_knot_differences_without_round_trip_cancellation(shift):
    time = np.arange(17) * 0.125
    knots = search._adjacent_shift_knots(time, shift)
    assert knots == ((-0.125, 0.0) if shift < 0 else (0.0, 0.125))
    points, unavailable = search._neighbor_shifts(time, shift, (-0.3, 0.3))
    assert not unavailable
    assert set(knots).issubset(points)
    assert all(-0.3 <= p <= 0.3 for p in points)
    assert len(points) <= 9


def test_real_approximate_grid_knots_remain_distinct():
    time = np.arange(9) * 0.25
    time[4:] += 2e-9
    true_knots = np.unique(time[:, None] - time[None, :])
    near = search._adjacent_shift_knots(time, 0.25)
    assert len(near) == 3 and near[0] < near[1] < near[2]
    assert all(k in true_knots for k in near)
    points, unavailable = search._neighbor_shifts(time, 0.25, (-0.4, 0.4))
    # Actual adjacent-float knots can lack an interior. They are never merged.
    assert all(k in points for k in near)
    assert unavailable or len(points) > len(near)


def test_interval_without_representable_interior_is_unavailable():
    time = np.arange(7.0)
    bound = np.nextafter(0.0, 1.0)
    points, unavailable = search._neighbor_shifts(time, 0.0, (-bound, bound))
    assert points == (0.0,)
    assert unavailable == (-1, 1)


@pytest.mark.parametrize("endpoint", [0.0, 0.25])
def test_scalar_validation_crosses_real_knots_and_jumps_without_derivative(endpoint):
    parameters = _FixedKParameters((3, 4, 2), (0.6, 1.5, 3), 1, 0)
    problem, x = make_problem(
        time=np.arange(9.0),
        kernel=[endpoint, 1, 2, 1, 0, 0, 0, 0, endpoint],
        parameters=parameters,
    )
    base = problem.evaluate(x, phase="initialization", origin="knot_fixture")
    condition = _conditioning_diagnostic(problem, base, policy())
    assert condition.status == "derivative_unavailable"
    result = search._validate_full(problem, base, settings())
    assert result.complete
    probes = [
        p for p in result.probes if p.direction == "shift_interval" and p.evaluation
    ]
    assert any(p.evaluation.parameters.temporal_shift_ns < 0 for p in probes)
    assert any(p.evaluation.parameters.temporal_shift_ns > 0 for p in probes)
    assert all(p.evaluation.status in ("finite", "infinite_objective") for p in probes)


def test_scalar_probes_at_zero_faces_have_exact_objective_or_infinity():
    problem, x = make_problem(parameters=_FixedKParameters((0, 0, 0), (1, 2, 3), 3, 0))
    problem = type(problem)(
        replace(problem.inputs, targets=np.full(65, 3.0)), problem.coordinates, limits()
    )
    base = problem.evaluate(x, phase="initialization", origin="constant_counts")
    result = search._validate_full(problem, base, settings())
    assert result.complete and result.passed
    assert base.centered_nll == 0
    assert any(
        p.evaluation and p.evaluation.status == "infinite_objective"
        for p in result.probes
    )
    for probe in result.probes:
        if probe.direction == "background" and probe.evaluation:
            background = probe.evaluation.parameters.background
            delta = (background - 3) / 3
            oracle = 65 * 3 * (delta - np.log1p(delta))
            assert probe.evaluation.centered_nll == pytest.approx(oracle, abs=1e-12)


def test_exhausted_scalar_validation_retains_descent_without_passing():
    phase = dict(limits().phase_model_calls)
    phase["validation"] = 1
    problem, x = make_problem(work_limits=limits(phase_model_calls=phase))
    base = problem.evaluate(x, phase="initialization", origin="limited")
    checked = _validate_candidate(
        problem, base, steps=(0.01,), absolute_tolerance=0.001, relative_tolerance=0
    )
    assert not checked.complete and checked.passed is None
    assert problem.work.phases["validation"]["model_calls"] == 1
    assert checked.probes[-1].evaluation.status == "phase_model_limit"
    assert all(
        p.evaluation.parameters != base.parameters
        for p in checked.probes
        if p.evaluation and p.evaluation.finite
    )


def controlled_promotion(monkeypatch, *, complete):
    """A constant-count problem with a counted known better background probe."""
    seen = []
    original = search._validate_full

    def bank(problem, config, guesses):
        parameters = _FixedKParameters((0, 0, 0), (1, 2, 3), 8, 0.017)
        base = problem.evaluate(
            problem.coordinates.encode(parameters),
            phase="initialization",
            origin="controlled_seed",
        )
        return _SeedBank(selected=[(f"control-{i}", base) for i in range(16)])

    def validate(problem, candidate, config):
        seen.append(candidate)
        if len(seen) > 1:
            return original(problem, candidate, config)
        point = candidate.coordinates.copy()
        point[6] = 4 / problem.coordinates.background_scale
        probe = problem.evaluate(
            point, phase="validation", origin="known_background_descent"
        )
        assert candidate.centered_nll - probe.centered_nll > 0.001
        return _LocalValidation(
            candidate,
            candidate.centered_nll,
            0.001,
            (
                _ValidationProbe(
                    "background",
                    _readonly(point - candidate.coordinates),
                    1.0,
                    True,
                    probe,
                ),
            ),
            probe,
            candidate.centered_nll - probe.centered_nll,
            complete,
            False if complete else None,
        )

    monkeypatch.setattr(search, "_build_seed_bank", bank)
    monkeypatch.setattr(solvers, "minimize", false_success)
    monkeypatch.setattr(search, "_validate_full", validate)
    return seen


def test_successful_promoted_candidate_validation_follows_counted_required_recovery(
    monkeypatch,
):
    seen = controlled_promotion(monkeypatch, complete=True)
    result = fit(
        np.full(33, 4),
        config=settings(
            max_seed_starts=16, reserved_recovery_starts=4, max_optimization_calls=20000
        ),
    )
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert result.parameters.background == 4
    assert result.numerical.accepted and result.numerical.validation_matches_candidate
    assert result.numerical.fresh_prediction_consistent
    assert (
        result.identifiability.candidate
        == result.parameters
        == result.numerical.candidate
    )
    assert result.optimization.promotions == ("validation_probe",)
    assert [s.kind for s in result.optimization.starts] == ["seed"] * 16 + [
        "required_recovery"
    ]
    assert result.optimization.recovery_starts_used == 1
    assert result.optimization.recovery_calls_reserved == 4000
    assert result.optimization.starts[-1].model_call_limit == 1000
    assert (
        result.optimization.recovery_calls_used
        == result.optimization.starts[-1].model_calls
    )


@pytest.mark.parametrize("validation_budget", [1, 1024])
def test_incomplete_validation_preserves_recovery_gate_and_terminal_identity(
    monkeypatch, validation_budget
):
    seen = controlled_promotion(monkeypatch, complete=False)
    result = fit(
        np.full(33, 4),
        config=settings(
            max_seed_starts=16,
            reserved_recovery_starts=4,
            max_optimization_calls=20000,
            max_validation_calls=validation_budget,
        ),
    )
    assert len(seen) == 2
    assert not result.numerical.accepted
    assert result.numerical.validation_matches_candidate
    assert result.numerical.validation.complete == (validation_budget > 1)
    assert result.numerical.validation.passed == (
        True if validation_budget > 1 else None
    )
    assert result.numerical.fresh_prediction_consistent
    assert result.identifiability.candidate == result.parameters
    assert (
        result.optimization.recovery_starts_used
        == result.optimization.recovery_calls_used
        == 0
    )
    assert result.optimization.recovery_starts_reserved == 4
    assert result.optimization.recovery_calls_reserved == 4000
    assert result.optimization.recovery_unavailable_reason == "validation_incomplete"
    assert not result.numerical.required_recovery_complete
    assert (
        result.optimization.work.phases["validation"]["model_calls"]
        <= validation_budget
    )


def test_terminal_validation_success_does_not_waive_missing_required_recovery(
    monkeypatch,
):
    seen = controlled_promotion(monkeypatch, complete=True)
    result = fit(
        np.full(33, 4), config=settings(max_seed_starts=16, reserved_recovery_starts=0)
    )
    assert len(seen) == 2 and result.numerical.validation.passed
    assert result.numerical.validation_matches_candidate
    assert not result.numerical.accepted
    assert (
        result.optimization.recovery_unavailable_reason
        == "required_recovery_start_limit"
    )
    assert result.optimization.best_observed_centered_nll == 0
