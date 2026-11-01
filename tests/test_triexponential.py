"""Dedicated development API, exact boundaries and independent curve checks."""

from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from tcspc_toolkit.estimation import triexponential as api
from tcspc_toolkit.estimation import _fixed_k_search as search
from tcspc_toolkit.estimation import _fixed_k_solvers as solvers
from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import IRFProfile, IRFSourceKind
from test_fixed_k_foundations import direct_components, make_problem


def settings(**changes):
    values = dict(
        max_seed_starts=10,
        reserved_recovery_starts=2,
        max_optimization_calls=400,
        wall_seconds=60,
    )
    values.update(changes)
    return api.TriexponentialFitSettings(**values)


def fit(y=None, *, config=None, **kwargs):
    t = np.arange(33) / 4
    kernel = np.zeros(33)
    kernel[2:5] = (1, 2, 1)
    return api.fit_triexponential_reconvolution(
        t,
        np.zeros(33) if y is None else y,
        kernel,
        lifetime_bounds_ns=(0.1, 10),
        temporal_shift_bounds_ns=(-0.4, 0.4),
        settings=config or settings(),
        **kwargs,
    )


def stationary_solver(fun, x, **kwargs):
    fun(x)
    return SimpleNamespace(
        x=x, success=True, status=0, message="controlled stationary endpoint"
    )


def test_explicit_settings_and_no_root_export():
    import tcspc_toolkit

    with pytest.raises(TypeError, match="explicit"):
        api.fit_triexponential_reconvolution(
            [], [], [], lifetime_bounds_ns=(1, 2), temporal_shift_bounds_ns=(-1, 1)
        )
    assert len(tcspc_toolkit.__all__) == 66
    assert not hasattr(tcspc_toolkit, "fit_triexponential_reconvolution")


@pytest.mark.parametrize(
    "name,seeds,recovery,calls",
    [
        ("compact", 16, 4, 60000),
        ("balanced", 32, 8, 120000),
        ("exhaustive", 109, 8, 300000),
    ],
)
def test_explicit_candidate_policies(name, seeds, recovery, calls):
    s = api.TriexponentialFitSettings.development(name)
    assert (
        s.max_seed_starts,
        s.reserved_recovery_starts,
        s.max_optimization_calls,
    ) == (seeds, recovery, calls)
    assert s.total_start_limit == seeds + recovery
    assert s.recovery_calls == calls // 5


@pytest.mark.parametrize(
    "change",
    [
        {"max_seed_starts": 9},
        {"max_seed_starts": 110},
        {"max_seed_starts": True},
        {"reserved_recovery_starts": -1},
        {"max_validation_calls": 1.2},
        {"wall_seconds": 0},
        {"wall_seconds": np.inf},
        {"policy": "production"},
        {"ftol": 0},
        {"descent_tolerance": -1},
        {"validation_steps": ()},
        {"validation_steps": (0,)},
        {"derivative_reductions": 0},
    ],
)
def test_invalid_settings(change):
    with pytest.raises(ValueError):
        settings(**change)


@pytest.mark.parametrize(
    "amplitudes,lifetimes",
    [((1, 2), (1, 2)), ((-1, 0, 1), (1, 2, 3)), ((1, 2, 3), (0, 2, 3))],
)
def test_invalid_parameters(amplitudes, lifetimes):
    with pytest.raises(ValueError):
        api.TriexponentialParameters(amplitudes, lifetimes)


def test_analytic_zero_counts_preserves_exact_zero_and_equal_lifetimes():
    result = fit()
    assert result.numerical.accepted, result.numerical
    assert result.parameters.amplitudes == (0, 0, 0)
    assert result.parameters.background == 0
    assert len(set(result.parameters.lifetimes_ns)) == 1
    assert result.centered_nll == 0
    assert result.statistics.centered_negative_log_likelihood == 0
    assert result.components.amplitude_fractions is None
    assert result.identifiability.zero_amplitude_components == (0, 1, 2)
    assert result.optimization.starts == ()
    assert result.numerical.validation.candidate == result.parameters
    assert result.numerical.signed_objective_gap == 0
    assert not result.expected_counts.flags.writeable
    with pytest.raises(TypeError):
        result.optimization.work.phases["validation"]["model_calls"] = 0


def test_raw_count_guard_and_guess_limits():
    with pytest.raises(ValueError):
        fit(np.full(33, 1.5))
    with pytest.raises(TypeError, match="initial guesses"):
        fit(initial_guesses=[(1, 2, 3)])
    p = api.TriexponentialParameters((0, 1, 2), (1, 1, 3))
    with pytest.raises(ValueError, match="ten protected"):
        fit(initial_guesses=[p])
    with pytest.raises(ValueError):
        fit(
            config=settings(max_seed_starts=11),
            initial_guesses=[replace(p, temporal_shift_ns=2)],
        )


def test_prepared_irf_snapshot_and_unavailable_diagnostics():
    t = np.arange(33) / 4
    kernel = np.zeros(33)
    kernel[2:5] = (1, 2, 1)
    source = IRFProfile(
        t,
        kernel,
        IRFSourceKind.IMPORTED_SAMPLED,
        metadata={"sample": "control"},
        provenance={"path": "fixture"},
    )
    prepared = prepare_irf(source, t)
    result = api.fit_triexponential_reconvolution(
        t,
        np.zeros(33),
        prepared,
        lifetime_bounds_ns=(0.1, 10),
        temporal_shift_bounds_ns=(-0.4, 0.4),
        settings=settings(max_diagnostic_calls=0),
    )
    assert result.numerical.accepted
    assert result.identifiability.weighted_jacobian_rank is None
    assert result.assumptions.source_provenance["path"] == "fixture"
    assert result.assumptions.preparation_diagnostics == prepared.diagnostics
    assert result.assumptions.target_kind == "raw_counts"


def test_supplied_permutations_are_deduplicated_and_exact_curve_is_assessed(
    monkeypatch,
):
    monkeypatch.setattr(solvers, "minimize", stationary_solver)
    p = _FixedKParameters((30, 0, 10), (3, 1, 1), 0.4, 0.031)
    problem, _ = make_problem(parameters=p)
    y = (
        direct_components(problem.inputs.time_ns, problem.inputs.kernel, p).sum(axis=0)
        + p.background
    )
    guesses = [
        api.TriexponentialParameters(
            tuple(p.amplitudes[i] for i in order),
            tuple(p.lifetimes_ns[i] for i in order),
            p.background,
            p.temporal_shift_ns,
        )
        for order in permutations(range(3))
    ]
    result = search._fit_triexponential(
        problem.inputs.time_ns,
        y,
        problem.inputs.kernel,
        lifetime_bounds_ns=(0.1, 10),
        temporal_shift_bounds_ns=(-0.4, 0.4),
        settings=settings(max_seed_starts=11),
        initial_guesses=guesses,
        target_kind="deterministic_expected_counts",
    )
    assert result.numerical.accepted, result.numerical
    assert result.parameters.lifetimes_ns == pytest.approx((1, 1, 3))
    assert result.parameters.amplitudes == pytest.approx((10, 0, 30))
    assert result.parameters.amplitudes[1] == 0
    assert result.statistics is None
    assert result.centered_nll < 1e-23
    np.testing.assert_allclose(result.expected_counts, y, rtol=2e-14)
    ordered_components = direct_components(
        problem.inputs.time_ns, problem.inputs.kernel, result.parameters
    )
    np.testing.assert_allclose(
        result.component_counts, ordered_components, rtol=2e-14, atol=1e-14
    )
    photons = ordered_components.sum(axis=1)
    np.testing.assert_allclose(result.components.detected_signal_photons, photons)
    np.testing.assert_allclose(
        result.components.detected_signal_fractions, photons / photons.sum()
    )
    assert result.components.amplitude_fractions == pytest.approx((0.25, 0, 0.75))
    assert result.components.ideal_photon_fractions == pytest.approx((0.1, 0, 0.9))
    assert result.identifiability.zero_amplitude_components == (1,)
    assert result.identifiability.equal_lifetime_pairs == ((0, 1),)
    assert result.numerical.candidate == result.parameters
    assert result.numerical.validation.candidate == result.parameters
    assert len(result.optimization.starts) == 11


def test_finite_seed_does_not_inherit_false_solver_success(monkeypatch):
    monkeypatch.setattr(
        solvers,
        "minimize",
        lambda fun, x, **kw: SimpleNamespace(
            x=x, success=True, status=0, message="false success"
        ),
    )
    result = fit(np.arange(33) % 4 + 1, config=settings(reserved_recovery_starts=0))
    assert result.parameters is not None
    assert not result.numerical.accepted
    assert result.optimization.unattempted_seed_ids == ()
    assert len(result.optimization.starts) == 10
    assert all(s.optimizer_reported_success for s in result.optimization.starts)
    assert result.numerical.validation.best_improvement > settings().descent_tolerance


@pytest.mark.parametrize(
    "changes",
    [
        {"max_initialization_calls": 0},
        {"max_validation_calls": 0},
        {"max_assessment_calls": 0},
    ],
)
def test_missing_required_work_prevents_acceptance(changes):
    result = fit(config=settings(**changes))
    assert not result.numerical.accepted
    assert not result.numerical.required_stages_complete
