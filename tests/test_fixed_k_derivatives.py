"""Deterministic derivative oracles, actual-grid knots and unavailable stencils."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation._fixed_k import _FixedKParameters
from tcspc_toolkit.estimation import _fixed_k_objective as objective
from test_fixed_k_foundations import direct_components, limits, make_problem, policy


def evaluate(problem, x):
    return problem.evaluate(x, phase="optimization", origin="test")


def independent_gradient(problem, x):
    """Complex-step direct-convolution oracle away from interpolation knots.

    Shift column uses the exact piecewise-linear IRF slope, not a differenced NLL.
    """
    p, t, kernel = problem.coordinates, problem.inputs.time_ns, problem.inputs.kernel
    k = p.n_components
    h = 1e-25

    def curve(z):
        a = z[:k] * p.amplitude_scale
        tau = np.exp(z[k : 2 * k] + p.log_mid)
        shift = p.shift_mid + p.shift_scale * z[-1].real
        q = t - shift
        shifted = np.interp(q, t, kernel, left=0, right=0).astype(complex)
        index = np.searchsorted(t, q) - 1
        mask = (index >= 0) & (index < t.size - 1)
        slope = np.zeros_like(t)
        slope[mask] = np.diff(kernel)[index[mask]] / np.diff(t)[index[mask]]
        shifted -= 1j * z[-1].imag * p.shift_scale * slope
        return (
            sum(
                ai * np.convolve(np.exp(-t / ti), shifted)[: t.size] * (t[1] - t[0])
                for ai, ti in zip(a, tau, strict=True)
            )
            + p.background_scale * z[-2]
        )

    mu = curve(x.astype(complex)).real
    score = 1 - problem.inputs.targets / mu
    columns = []
    for j in range(p.dimension):
        z = x.astype(complex)
        z[j] += 1j * h
        columns.append(curve(z).imag / h)
    return score @ np.column_stack(columns)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_gradient_against_independent_complex_step_and_shift_slope(k):
    problem, x = make_problem(k)
    base = evaluate(problem, x)
    derivative = objective._objective_derivatives(
        problem, base, policy(), phase="optimization"
    )
    assert derivative.complete, derivative
    assert derivative.shift_region == "smooth"
    np.testing.assert_allclose(
        derivative.gradient, independent_gradient(problem, x), rtol=3e-6, atol=2e-6
    )
    assert problem.work.charged_component_convolutions == k * problem.work.model_calls
    assert problem.work.objective_calls == problem.work.model_calls - 1


@pytest.mark.parametrize("approximate", [False, True])
def test_actual_coordinates_determine_knots(approximate):
    t = np.arange(7.0)
    if approximate:
        t[3:] += 2e-8
    kernel = np.array([0, 1, 2, 3, 2, 1, 0.0])
    g = objective._shift_geometry(t, kernel, 1.0)
    assert g.region == "continuous_nondifferentiable_knot"
    if approximate:
        assert g.distance_to_next_knot_ns == pytest.approx(2e-8, abs=1e-15)
    # A shift which is only nominally one dt must not be snapped to a knot.
    g = objective._shift_geometry(t, kernel, 1.0 + 1e-9)
    assert g.region == "smooth"


def test_agreeing_knot_slopes_are_distinguished():
    t = np.arange(5.0)
    # At shift=4 only the zero endpoint is sampled; adjacent flat segment agrees.
    g = objective._shift_geometry(t, np.array([0, 0, 1, 0, 0.0]), 4)
    assert g.region == "continuous_knot_agreeing_slopes"
    problem, x = make_problem(1, time=t, kernel=[0, 0, 1, 0, 0])
    # A broader explicit shift map admits this exact knot.
    problem.coordinates = replace(problem.coordinates, temporal_shift_bounds_ns=(-5, 5))
    x = problem.coordinates.encode(_FixedKParameters((2,), (1,), 1, 4))
    derivative = objective._objective_derivatives(
        problem, evaluate(problem, x), policy(), phase="optimization"
    )
    assert derivative.shift_region == g.region
    assert derivative.derivatives[-1].value == pytest.approx(0, abs=1e-8)


def test_continuous_kink_keeps_one_sided_information_without_gradient():
    problem, x = make_problem(
        1,
        time=np.arange(5.0),
        kernel=[0, 1, 0, 0, 0],
        parameters=_FixedKParameters((3,), (1,), 1, 0),
    )
    base = evaluate(problem, x)
    d = objective._objective_derivatives(problem, base, policy(), phase="optimization")
    assert d.shift_region == "continuous_nondifferentiable_knot"
    assert d.gradient is None and d.derivatives[-1].value is None
    assert d.derivatives[-1].left is not None
    assert d.derivatives[-1].right is not None
    assert d.derivatives[-1].left != pytest.approx(d.derivatives[-1].right)
    assert base.finite


@pytest.mark.parametrize(
    "kernel,usable",
    [([1, 1, 0, 0, 0], "left"), ([0, 0, 0, 1, 1], "right"), ([1, 0, 0, 0, 1], None)],
)
def test_zero_extrapolation_discontinuities_do_not_become_derivatives(kernel, usable):
    problem, x = make_problem(
        1,
        time=np.arange(5.0),
        kernel=kernel,
        parameters=_FixedKParameters((3,), (1,), 1, 0),
    )
    base = evaluate(problem, x)
    d = objective._objective_derivatives(problem, base, policy(), phase="optimization")
    shift = d.derivatives[-1]
    assert base.finite and d.gradient is None
    assert d.shift_region == shift.status == "support_discontinuity"
    assert shift.value is None
    assert (shift.left is not None) == (usable == "left")
    assert (shift.right is not None) == (usable == "right")


def test_zero_signal_keeps_directional_slopes_but_not_joint_gradient_at_jump():
    problem, x = make_problem(
        3,
        time=np.arange(5.0),
        kernel=[1, 1, 1, 1, 1],
        parameters=_FixedKParameters((0, 0, 0), (1, 1, 1), 0, 0),
        targets=np.zeros(5),
    )
    d = objective._objective_derivatives(
        problem, evaluate(problem, x), policy(), phase="optimization"
    )
    assert not d.complete and d.gradient is None
    assert all(item.value == 0 for item in d.derivatives[3:6])
    assert d.derivatives[-1].left == d.derivatives[-1].right == 0
    assert d.derivatives[-1].value is None
    assert d.derivatives[-2].value == 10  # Includes mu=y=0.
    assert all(item.value >= 0 for item in d.derivatives[:3])
    # Direct convolution verifies why zero shift partials are insufficient.
    trial = x.copy()
    trial[0] += 1e-6
    trial[-1] += 1e-6
    p = problem.coordinates.decode(trial)
    joint = (
        direct_components(problem.inputs.time_ns, problem.inputs.kernel, p).sum() / 1e-6
    )
    assert joint != pytest.approx(d.derivatives[0].value, rel=0.01)


@pytest.mark.parametrize("boundary", [0, 1])
def test_feasible_one_sided_lifetime_and_shift_bounds(boundary):
    problem, _ = make_problem(1)
    p = problem.coordinates
    x = p.encode(
        _FixedKParameters(
            (2,),
            (p.lifetime_bounds_ns[boundary],),
            1,
            p.temporal_shift_bounds_ns[boundary],
        )
    )
    d = objective._objective_derivatives(
        problem, evaluate(problem, x), policy(), phase="optimization"
    )
    assert d.derivatives[1].status == "one_sided_bound"
    assert d.derivatives[-1].status == "one_sided_bound"


def test_failed_derivative_preserves_finite_candidate_and_charges_failure(monkeypatch):
    problem, x = make_problem()
    base = evaluate(problem, x)

    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate")

    monkeypatch.setattr(
        objective, "multiexponential_reconvolution_component_counts", fail
    )
    d = objective._objective_derivatives(problem, base, policy(), phase="optimization")
    assert not d.complete and d.gradient is None
    assert problem.best_finite is base
    assert problem.work.model_failures_with_unknown_partial_work > 0
    assert problem.work.completed_component_convolutions == 3
    assert problem.work.charged_component_convolutions > 3


def test_unrepresentable_and_unstable_stencils_are_explicit():
    problem, x = make_problem(1)
    base = evaluate(problem, x)
    d = objective._objective_derivatives(
        problem,
        base,
        policy(relative_step=np.nextafter(0.0, 1.0)),
        phase="optimization",
    )
    assert d.gradient is None
    assert "unrepresentable" in d.derivatives[1].status
    d = objective._objective_derivatives(
        problem,
        base,
        policy(
            relative_step=0.01,
            relative_tolerance=0,
            absolute_tolerance=0,
            max_reductions=1,
        ),
        phase="optimization",
    )
    assert "unstable" in d.derivatives[1].status


def test_infinite_base_and_budget_failure_have_no_dummy_gradient():
    problem, x = make_problem(1, work_limits=limits(model_calls=1))
    base = evaluate(problem, x)
    d = objective._objective_derivatives(problem, base, policy(), phase="optimization")
    assert d.gradient is None and problem.work.model_calls == 1
    invalid = replace(base, status="infinite_objective", centered_nll=np.inf)
    d = objective._objective_derivatives(
        problem, invalid, policy(), phase="optimization"
    )
    assert all(item.status == "invalid_base" for item in d.derivatives)


def test_one_sided_score_matches_direct_directional_objective():
    problem, x = make_problem(
        1,
        time=np.arange(5.0),
        kernel=[0, 1, 0, 0, 0],
        parameters=_FixedKParameters((3,), (1,), 1, 0),
    )
    base = evaluate(problem, x)
    d = objective._objective_derivatives(problem, base, policy(), phase="optimization")
    y = problem.inputs.targets

    def cost(point):
        p = problem.coordinates.decode(point)
        mu = (
            direct_components(problem.inputs.time_ns, problem.inputs.kernel, p).sum(
                axis=0
            )
            + p.background
        )
        return np.sum(mu - y + y * np.log(y / mu))

    for sign, value in ((-1, d.derivatives[-1].left), (1, d.derivatives[-1].right)):
        trial = x.copy()
        trial[-1] += sign * 1e-6
        assert value == pytest.approx(
            (cost(trial) - cost(x)) / (sign * 1e-6), rel=2e-5, abs=2e-5
        )


def test_derivative_uses_approximately_uniform_abscissae_in_forward_model():
    time = np.linspace(0, 8, 65)
    time[25:] += 2e-9
    problem, x = make_problem(3, time=time)
    derivative = objective._objective_derivatives(
        problem, evaluate(problem, x), policy(), phase="optimization"
    )
    assert derivative.complete
    np.testing.assert_allclose(
        derivative.gradient, independent_gradient(problem, x), rtol=3e-6, atol=2e-6
    )


def test_nearby_approximate_grid_knots_can_make_shift_stencil_unavailable():
    time = np.arange(7.0)
    time[3:] += 2e-8
    problem, _ = make_problem(1, time=time, kernel=[0, 1, 2, 3, 2, 1, 0])
    problem.coordinates = replace(problem.coordinates, temporal_shift_bounds_ns=(-2, 2))
    x = problem.coordinates.encode(_FixedKParameters((3,), (1,), 1, 1 + 1e-15))
    derivative = objective._objective_derivatives(
        problem, evaluate(problem, x), policy(), phase="optimization"
    )
    assert derivative.gradient is None
    assert derivative.derivatives[-1].value is None
    assert "unrepresentable" in derivative.derivatives[-1].status
