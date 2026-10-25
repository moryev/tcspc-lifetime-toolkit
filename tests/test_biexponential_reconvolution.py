"""Physical/numerical contracts for the module-qualified bi-exponential fitter."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import tcspc_toolkit.estimation.multiexponential as fitting
from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitSettings,
    BiexponentialParameters,
    fit_biexponential_reconvolution,
)
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import IRFProfile, IRFSourceKind


def independent_case(
    *,
    bins=256,
    window=20.0,
    lifetimes=(0.8, 3.2),
    fractions=(0.4, 0.6),
    photons=1e6,
    background=2.0,
    shift=0.075,
    origin=0.0,
):
    time = np.linspace(origin, window + origin, bins)
    sigma = 0.3 / np.sqrt(8 * np.log(2))
    kernel = np.exp(-0.5 * ((time - (1 + origin)) / sigma) ** 2)
    kernel /= np.trapezoid(kernel, x=time)
    shifted = np.interp(time - shift, time, kernel, left=0, right=0)
    unit = np.array(
        [
            np.convolve(np.exp(-time / tau), shifted)[:bins] * (time[1] - time[0])
            for tau in lifetimes
        ]
    )
    amplitudes = photons * np.asarray(fractions) / unit.sum(axis=1)
    components = amplitudes[:, None] * unit
    parameters = BiexponentialParameters(
        tuple(amplitudes), lifetimes, background, shift
    )
    return time, kernel, components.sum(axis=0) + background, parameters, components


BOUNDS = dict(lifetime_bounds_ns=(0.1, 8.0), temporal_shift_bounds_ns=(-0.3, 0.3))
ECONOMICAL = BiexponentialFitSettings(
    policy="economical",
    max_objective_evaluations=100000,
    max_validation_evaluations=2000,
)


def test_noise_free_issue28_baseline_predeclared_targets():
    t, k, y, truth, _ = independent_case(bins=512)
    result = fitting._fit_core(t, y, k, **BOUNDS, raw_counts=False)
    assert result.optimization.numerical_validation_passed is True
    np.testing.assert_allclose(
        result.parameters.amplitudes, truth.amplitudes, rtol=0.001
    )
    np.testing.assert_allclose(
        result.parameters.lifetimes_ns, truth.lifetimes_ns, rtol=0.001
    )
    assert abs(result.parameters.background / truth.background - 1) <= 0.001
    assert abs(result.parameters.temporal_shift_ns - truth.temporal_shift_ns) <= 1e-4
    assert np.linalg.norm(result.expected_counts - y) / np.linalg.norm(y) < 1e-6
    assert result.statistics is None
    assert result.assumptions.target_kind == "deterministic_expected_counts"


@pytest.mark.parametrize("shift,background", [(-0.13, 0.0), (0.13, 20.0), (0.0, 2.0)])
def test_seeded_separated_recovery(shift, background):
    t, k, mu, truth, _ = independent_case(shift=shift, background=background)
    y = np.random.default_rng(2802001).poisson(mu)
    result = fit_biexponential_reconvolution(t, y, k, **BOUNDS, settings=ECONOMICAL)
    assert result.optimization.numerical_validation_passed
    np.testing.assert_allclose(
        result.parameters.lifetimes_ns, truth.lifetimes_ns, rtol=0.05
    )
    assert abs(result.parameters.temporal_shift_ns - shift) < 0.005
    assert result.parameters.background >= 0
    assert result.statistics.n_parameters == 6
    np.testing.assert_allclose(
        result.expected_counts,
        result.component_counts.sum(axis=0) + result.parameters.background,
        rtol=0,
        atol=0,
    )


def test_swapped_guesses_canonicalize_parameters_and_curves():
    t, k, mu, p, _ = independent_case(bins=128)
    y = np.random.default_rng(2802002).poisson(mu)
    reverse = BiexponentialParameters(
        p.amplitudes[::-1], p.lifetimes_ns[::-1], p.background, p.temporal_shift_ns
    )
    settings = replace(ECONOMICAL, max_starts=1)
    first = fit_biexponential_reconvolution(
        t, y, k, **BOUNDS, initial_guesses=[p], settings=settings
    )
    second = fit_biexponential_reconvolution(
        t, y, k, **BOUNDS, initial_guesses=[reverse], settings=settings
    )
    assert first.parameters == second.parameters
    np.testing.assert_array_equal(first.component_counts, second.component_counts)
    assert first.parameters.lifetimes_ns[0] <= first.parameters.lifetimes_ns[1]


def test_prepared_array_equivalence_provenance_and_mutation_safety():
    t, k, mu, p, _ = independent_case(bins=128, origin=0.2)
    source = IRFProfile(
        t,
        k,
        IRFSourceKind.IMPORTED_SAMPLED,
        metadata={"nested": {"value": 1}},
        provenance={"source": "synthetic_test"},
    )
    prepared = prepare_irf(source, t)
    y = np.random.default_rng(2802003).poisson(mu)
    original = (t.copy(), y.copy(), prepared.kernel.copy(), source.values.copy())
    settings = replace(ECONOMICAL, max_starts=1)
    a = fit_biexponential_reconvolution(
        t, y, prepared, **BOUNDS, initial_guesses=[p], settings=settings
    )
    b = fit_biexponential_reconvolution(
        t, y, prepared.kernel, **BOUNDS, initial_guesses=[p], settings=settings
    )
    assert a.parameters == b.parameters
    np.testing.assert_array_equal(a.expected_counts, b.expected_counts)
    assert a.assumptions.preparation_diagnostics == prepared.diagnostics
    assert a.assumptions.source_provenance["source"] == "synthetic_test"
    assert b.assumptions.preparation_diagnostics is None
    assert b.assumptions.irf_input_kind == "caller_declared_array"
    source.metadata["nested"]["value"] = 99
    assert a.assumptions.source_metadata["nested"]["value"] == 1
    for values, saved in zip(
        (t, y, prepared.kernel, source.values), original, strict=True
    ):
        np.testing.assert_array_equal(values, saved)
    for values in (
        a.expected_counts,
        a.component_counts,
        a.statistics.deviance_residuals,
    ):
        assert not values.flags.writeable


@pytest.mark.parametrize(
    "counts",
    [
        [0, 1.5, 2],
        [-1, 0, 1],
        [0, np.nan, 1],
        [0, np.inf, 1],
        [0, 2**53, 1],
        [[0, 1, 2]],
        [1, 2],
    ],
)
def test_invalid_observations_rejected(counts):
    with pytest.raises(ValueError):
        fit_biexponential_reconvolution([0, 1, 2], counts, [0, 1, 0], **BOUNDS)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lifetime_bounds_ns", (0, 8)),
        ("lifetime_bounds_ns", (-1, 8)),
        ("lifetime_bounds_ns", (1, 1)),
        ("lifetime_bounds_ns", (2, 1)),
        ("lifetime_bounds_ns", (1, np.inf)),
        ("lifetime_bounds_ns", (np.nan, 8)),
        ("temporal_shift_bounds_ns", (1, 1)),
        ("temporal_shift_bounds_ns", (1, -1)),
        ("temporal_shift_bounds_ns", (-np.inf, 1)),
        ("temporal_shift_bounds_ns", (0, np.nan)),
    ],
)
def test_invalid_bounds_rejected(key, value):
    bounds = {**BOUNDS, key: value}
    with pytest.raises(ValueError):
        fit_biexponential_reconvolution([0, 1, 2], [1, 1, 1], [0, 1, 0], **bounds)


@pytest.mark.parametrize(
    "kernel",
    [
        [0, 2, 0],
        [0, 0, 0],
        [0, -1, 0],
        [0, np.nan, 0],
        [0, np.inf, 0],
        [0, 1],
        [[0, 1, 0]],
    ],
)
def test_invalid_irf_rejected_without_normalization(kernel):
    with pytest.raises(ValueError):
        fit_biexponential_reconvolution([0, 1, 2], [1, 1, 1], kernel, **BOUNDS)


@pytest.mark.parametrize("time", [[0, 1, 3], [0, 0, 1], [2, 1, 0], [0, np.nan, 2], [0]])
def test_invalid_time_rejected(time):
    with pytest.raises(ValueError):
        fit_biexponential_reconvolution(time, [1, 1, 1], [0, 1, 0], **BOUNDS)


def test_prepared_grid_mismatch_rejected():
    source = IRFProfile(
        np.arange(3.0), np.array([0, 1, 0]), IRFSourceKind.IMPORTED_SAMPLED
    )
    prepared = prepare_irf(source, source.time_ns)
    with pytest.raises(ValueError, match="grid"):
        fit_biexponential_reconvolution(
            np.arange(3.0) + 0.1, [1, 1, 1], prepared, **BOUNDS
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_starts", 0),
        ("max_starts", True),
        ("max_iterations", 1.2),
        ("max_objective_evaluations", -1),
        ("ftol", 0),
        ("gtol", np.nan),
        ("weak_fraction", 0.5),
        ("condition_limit", 0.5),
        ("boundary_tolerance", 1),
        ("policy", "random"),
    ],
)
def test_invalid_settings_rejected(field, value):
    with pytest.raises(ValueError):
        BiexponentialFitSettings(**{field: value})


@pytest.mark.parametrize(
    "amplitudes,lifetimes,background,shift",
    [
        ([1], [1, 2], 0, 0),
        ([1, -1], [1, 2], 0, 0),
        ([1, np.inf], [1, 2], 0, 0),
        ([1, 1], [0, 2], 0, 0),
        ([1, 1], [1, np.nan], 0, 0),
        ([1, 1], [1, 2], -1, 0),
        ([1, 1], [1, 2], 0, np.inf),
    ],
)
def test_invalid_parameters_rejected(amplitudes, lifetimes, background, shift):
    with pytest.raises(ValueError):
        BiexponentialParameters(amplitudes, lifetimes, background, shift)


def test_impossible_expectations_keep_exact_poisson_limit():
    problem = fitting._Problem(
        np.arange(3.0),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        (0.1, 8),
        (-0.3, 0.3),
        ECONOMICAL,
    )
    x = problem.coordinates(BiexponentialParameters((0, 0), (1, 2), 0, 0))
    assert np.isinf(problem.evaluate(x))
    assert problem.best_x is None
    value, gradient = problem.value_and_gradient(x)
    assert np.isinf(value)
    assert np.all(np.isfinite(gradient))  # Invalid line-search trial, not accepted.


def test_false_optimizer_success_does_not_pass_local_validation(monkeypatch):
    t, k, mu, p, _ = independent_case(bins=128)
    wrong = replace(p, lifetimes_ns=(0.5, 2.0), background=10)
    monkeypatch.setattr(
        fitting,
        "_initial_points",
        lambda problem, guesses: [problem.coordinates(wrong)],
    )

    def false_success(fun, x0, **kwargs):
        fun(x0)
        return OptimizeResult(
            x=x0, fun=-1e99, success=True, status=0, message="false success"
        )

    monkeypatch.setattr(fitting, "minimize", false_success)
    result = fit_biexponential_reconvolution(
        t, np.rint(mu), k, **BOUNDS, settings=replace(ECONOMICAL, max_starts=1)
    )
    assert any(s.optimizer_reported_success for s in result.optimization.starts)
    assert result.optimization.selected_candidate_source == "validation"
    assert result.optimization.numerical_validation_passed is False
    assert result.optimization.max_detectable_descent_nll > 0.01
    assert result.statistics.centered_negative_log_likelihood > 0


def test_best_failed_candidate_not_replaced_by_worse_success(monkeypatch):
    t, k, mu, p, _ = independent_case(bins=128)
    wrong = replace(p, lifetimes_ns=(0.5, 2.0))
    monkeypatch.setattr(
        fitting,
        "_initial_points",
        lambda problem, guesses: [problem.coordinates(p), problem.coordinates(wrong)],
    )
    calls = []

    def controlled(fun, x0, **kwargs):
        fun(x0)
        calls.append(1)
        return OptimizeResult(
            x=x0,
            fun=-1e99 if len(calls) == 2 else 1e99,
            success=len(calls) == 2,
            status=2,
            message="controlled",
        )

    monkeypatch.setattr(fitting, "minimize", controlled)
    result = fitting._fit_core(
        t, mu, k, **BOUNDS, settings=replace(ECONOMICAL, max_starts=2), raw_counts=False
    )
    assert result.parameters.lifetimes_ns == pytest.approx(p.lifetimes_ns)
    assert result.optimization.optimizer_reported_success is False
    assert result.optimization.numerical_validation_passed is True


@pytest.mark.parametrize(
    "field,phase",
    [
        ("max_objective_evaluations", "optimization"),
        ("max_validation_evaluations", "validation"),
        ("max_diagnostic_evaluations", "diagnostics"),
        ("max_initialization_evaluations", "initialization"),
    ],
)
def test_explicit_budget_exhaustion(field, phase):
    t, k, mu, _, _ = independent_case(bins=64)
    result = fit_biexponential_reconvolution(
        t,
        np.rint(mu),
        k,
        **BOUNDS,
        settings=replace(ECONOMICAL, max_starts=1, **{field: 0}),
    )
    assert phase in result.optimization.exhausted_budgets
    counter = (
        "objective_evaluations"
        if phase == "optimization"
        else (
            "diagnostic_evaluations"
            if phase == "diagnostics"
            else phase + "_evaluations"
        )
    )
    assert getattr(result.optimization, counter) == 0
    if phase == "validation":
        assert result.optimization.numerical_validation_passed is None
    if phase == "initialization":
        assert result.parameters is None


def test_initial_guesses_checked_before_optimization():
    with pytest.raises(ValueError, match="outside"):
        fit_biexponential_reconvolution(
            [0, 1, 2],
            [1, 1, 1],
            [0, 1, 0],
            **BOUNDS,
            initial_guesses=[BiexponentialParameters((1, 1), (1, 20))],
        )
    with pytest.raises(TypeError):
        fit_biexponential_reconvolution(
            [0, 1, 2], [1, 1, 1], [0, 1, 0], **BOUNDS, initial_guesses=[(1, 2)]
        )


def negative_origin_case():
    time = np.linspace(-1, 5, 64)
    kernel = np.exp(-0.5 * (time / 0.2) ** 2)
    kernel /= np.trapezoid(kernel, x=time)
    return time, np.ones(time.size, dtype=int), kernel


@pytest.mark.parametrize("lower_bound", [0.001, 1e-9])
def test_overflowing_generated_initialization_retains_supplied_candidate(lower_bound):
    time, counts, kernel = negative_origin_case()
    guess = BiexponentialParameters((0, 0), (1, 2), 1, 0)
    result = fit_biexponential_reconvolution(
        time,
        counts,
        kernel,
        lifetime_bounds_ns=(lower_bound, 8),
        temporal_shift_bounds_ns=(-0.2, 0.2),
        initial_guesses=[guess],
        settings=BiexponentialFitSettings(policy="economical", max_starts=1),
    )
    np.testing.assert_array_equal(result.expected_counts, counts)
    assert result.parameters == guess
    assert result.statistics.centered_negative_log_likelihood == 0
    opt = result.optimization
    assert opt.numerical_validation_passed is True
    assert len(opt.starts) == 1
    assert opt.starts[0].initial_parameters == guess
    assert opt.initialization_rejected_candidates > 0
    assert opt.initialization_scaling_failed is (lower_bound == 1e-9)
    assert opt.initialization_evaluations <= 128
    assert opt.objective_evaluations <= 56000


@pytest.mark.parametrize("budget", [5, 128])
def test_all_generated_initialization_trials_overflow_report_failure(budget):
    time, counts, kernel = negative_origin_case()
    result = fit_biexponential_reconvolution(
        time,
        counts,
        kernel,
        lifetime_bounds_ns=(0.0001, 0.001),
        temporal_shift_bounds_ns=(-0.2, 0.2),
        settings=BiexponentialFitSettings(max_initialization_evaluations=budget),
    )
    assert result.parameters is None
    assert result.expected_counts is None
    assert result.component_counts is None
    assert result.statistics is None
    opt = result.optimization
    assert opt.stopping_reason == "no_finite_candidate"
    assert opt.numerical_validation_passed is None
    assert opt.optimizer_reported_success is None
    assert opt.starts == ()
    # One scale reference, 45 component templates, and one background candidate.
    assert opt.initialization_evaluations == min(budget, 47)
    assert opt.starts_screened == min(budget - 1, 46)
    assert opt.initialization_rejected_candidates == opt.starts_screened
    assert opt.initialization_scaling_failed is True
    assert ("initialization" in opt.exhausted_budgets) is (budget < 47)
    assert opt.objective_evaluations == 0
    assert opt.validation_evaluations == 0
    assert opt.diagnostic_evaluations == 0


def test_nonfinite_generated_initialization_predictions_are_rejected(monkeypatch):
    time, counts, kernel = negative_origin_case()
    monkeypatch.setattr(
        fitting,
        "multiexponential_reconvolution_component_counts",
        lambda *args, **kwargs: np.full((2, time.size), np.nan),
    )
    result = fit_biexponential_reconvolution(time, counts, kernel, **BOUNDS)
    assert result.parameters is None
    assert result.optimization.stopping_reason == "no_finite_candidate"
    assert result.optimization.initialization_rejected_candidates == 46
    assert result.optimization.initialization_evaluations == 47
    assert result.optimization.objective_evaluations == 0


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError])
def test_unrelated_generated_initialization_errors_propagate(monkeypatch, error):
    def broken_model(*args, **kwargs):
        raise error("unrelated model defect")

    monkeypatch.setattr(
        fitting, "multiexponential_reconvolution_component_counts", broken_model
    )
    time, counts, kernel = negative_origin_case()
    with pytest.raises(error, match="unrelated model defect"):
        fit_biexponential_reconvolution(time, counts, kernel, **BOUNDS)


@pytest.mark.parametrize("budget", [2, 3])
def test_rejected_initialization_trials_cannot_bypass_work_limits(budget):
    time, counts, kernel = negative_origin_case()
    result = fit_biexponential_reconvolution(
        time,
        counts,
        kernel,
        lifetime_bounds_ns=(0.001, 8),
        temporal_shift_bounds_ns=(-0.2, 0.2),
        initial_guesses=[BiexponentialParameters((0, 0), (1, 2), 1, 0)],
        settings=BiexponentialFitSettings(
            max_initialization_evaluations=budget,
            max_objective_evaluations=0,
            max_validation_evaluations=0,
            max_diagnostic_evaluations=0,
        ),
    )
    np.testing.assert_array_equal(result.expected_counts, counts)
    opt = result.optimization
    assert opt.initialization_evaluations == budget
    assert opt.starts_screened == budget - 1
    assert opt.initialization_rejected_candidates == budget - 2
    assert opt.objective_evaluations == 0
    assert opt.validation_evaluations == 0
    assert opt.diagnostic_evaluations == 0
    assert "initialization" in opt.exhausted_budgets
    assert "optimization" in opt.exhausted_budgets
    assert opt.numerical_validation_passed is None


@pytest.mark.parametrize("step", [5e-6, 2e-5])
def test_step_sensitivity_preserves_noise_free_recovery(step):
    t, k, y, p, _ = independent_case(bins=256)
    result = fitting._fit_core(
        t,
        y,
        k,
        **BOUNDS,
        settings=replace(ECONOMICAL, finite_difference_step=step),
        raw_counts=False,
    )
    assert result.optimization.numerical_validation_passed
    np.testing.assert_allclose(
        result.parameters.lifetimes_ns, p.lifetimes_ns, rtol=0.001
    )
    assert np.linalg.norm(result.expected_counts - y) / np.linalg.norm(y) < 1e-6


def test_adaptive_escalates_with_disagreeing_candidates(monkeypatch):
    t, k, mu, p, _ = independent_case(bins=64)
    guesses = [
        replace(p, lifetimes_ns=(0.5 + 0.05 * i, 2 + 0.1 * i)) for i in range(10)
    ]
    monkeypatch.setattr(
        fitting,
        "_initial_points",
        lambda problem, supplied: [problem.coordinates(g) for g in guesses],
    )

    def stationary(fun, x0, **kwargs):
        fun(x0)
        return OptimizeResult(x=x0, fun=-1, success=True, status=0, message="stub")

    monkeypatch.setattr(fitting, "minimize", stationary)
    result = fitting._fit_core(
        t,
        mu,
        k,
        **BOUNDS,
        settings=BiexponentialFitSettings(
            policy="adaptive", max_starts=10, max_validation_evaluations=2000
        ),
        raw_counts=False,
    )
    assert len(result.optimization.starts) > 4
    assert "starts" in result.optimization.exhausted_budgets
    assert result.optimization.numerical_validation_passed is False


def test_positive_integer_float_counts_and_two_bin_rank_limit():
    result = fit_biexponential_reconvolution(
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 2.0],
        **BOUNDS,
        settings=replace(ECONOMICAL, max_starts=1),
    )
    assert result.statistics.n_bins == 2
    assert (
        result.identifiability.weighted_jacobian_rank is None
        or result.identifiability.weighted_jacobian_rank <= 2
    )


@pytest.mark.parametrize("field", ["time", "counts", "kernel"])
def test_complex_arrays_are_rejected_without_discarding_imaginary_parts(field):
    values = dict(
        time=np.array([0, 1, 2]), counts=np.array([1, 1, 1]), kernel=np.array([0, 1, 0])
    )
    values[field] = values[field].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        fit_biexponential_reconvolution(
            values["time"], values["counts"], values["kernel"], **BOUNDS
        )


def test_lifetime_bounds_must_be_distinguishable_in_log_coordinates():
    lower = 1e100
    upper = np.nextafter(lower, np.inf)
    assert np.log(lower) == np.log(upper)
    with pytest.raises(ValueError, match="log"):
        fit_biexponential_reconvolution(
            [0, 1, 2],
            [1, 1, 1],
            [0, 1, 0],
            lifetime_bounds_ns=(lower, upper),
            temporal_shift_bounds_ns=(-0.3, 0.3),
        )
