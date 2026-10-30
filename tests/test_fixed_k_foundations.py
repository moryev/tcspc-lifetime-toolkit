"""Private foundations: dimensions, physical boundaries and independent curves."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from tcspc_toolkit.estimation._fixed_k import (
    _CoordinateMap,
    _DerivativePolicy,
    _FixedKInputs,
    _FixedKParameters,
    _canonicalize_components,
)
from tcspc_toolkit.estimation._fixed_k_objective import (
    _FixedKProblem,
    _WorkLimits,
    _PHASES,
)


def limits(**changes):
    values = dict(
        phase_model_calls=dict.fromkeys(_PHASES, 1000),
        model_calls=5000,
        objective_calls=5000,
        component_convolutions=20000,
        wall_seconds=30,
    )
    values.update(changes)
    return _WorkLimits(**values)


def policy(**changes):
    values = dict(
        relative_step=1e-4,
        relative_tolerance=2e-4,
        absolute_tolerance=1e-6,
        max_reductions=3,
    )
    values.update(changes)
    return _DerivativePolicy(**values)


def direct_components(time, kernel, parameters):
    """Direct convolution, independent of the package FFT forward implementation."""
    shifted = np.interp(
        time - parameters.temporal_shift_ns, time, kernel, left=0, right=0
    )
    return np.array(
        [
            a
            * np.convolve(np.exp(-time / tau), shifted)[: time.size]
            * (time[1] - time[0])
            for a, tau in zip(
                parameters.amplitudes, parameters.lifetimes_ns, strict=True
            )
        ]
    )


def make_problem(
    k=3,
    *,
    time=None,
    kernel=None,
    parameters=None,
    targets=None,
    work_limits=None,
    clock=None,
):
    time = np.linspace(0, 8, 65) if time is None else np.asarray(time)
    if kernel is None:
        kernel = np.exp(-0.5 * ((time - 0.8) / 0.22) ** 2)
        kernel[[0, -1]] = 0
    kernel = np.asarray(kernel) / np.trapezoid(kernel, x=time)
    parameters = parameters or _FixedKParameters(
        tuple(np.linspace(12, 30, k)), tuple(np.linspace(0.5, 3, k)), 0.4, 0.031
    )
    if targets is None:
        targets = direct_components(time, kernel, parameters).sum(axis=0) * 1.13 + 0.7
    inputs = _FixedKInputs(time, targets, kernel, "deterministic_expected_counts")
    coordinates = _CoordinateMap(k, (0.1, 10), (-0.4, 0.4), 20, 2)
    kwargs = {} if clock is None else {"clock": clock}
    problem = _FixedKProblem(inputs, coordinates, work_limits or limits(), **kwargs)
    return problem, coordinates.encode(parameters)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_dimension_roundtrip_and_direct_convolution(k):
    problem, x = make_problem(k)
    p = problem.coordinates
    assert p.dimension == 2 * k + 2 == len(p.names)
    result = problem.evaluate(x, phase="optimization", origin="test")
    assert result.finite
    np.testing.assert_allclose(p.encode(result.parameters), x, rtol=0, atol=5e-16)
    np.testing.assert_allclose(
        result.component_counts,
        direct_components(
            problem.inputs.time_ns, problem.inputs.kernel, result.parameters
        ),
        rtol=2e-13,
        atol=2e-14,
    )
    assert not result.coordinates.flags.writeable
    assert not result.component_counts.flags.writeable


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("edge", [0, 1])
def test_exact_zeros_equal_lifetimes_and_coordinate_bounds(k, edge):
    p = _CoordinateMap(k, (0.1, 10), (-0.3, 0.7), 8, 4)
    original = _FixedKParameters(
        (0,) * k, (p.lifetime_bounds_ns[edge],) * k, 0, p.temporal_shift_bounds_ns[edge]
    )
    assert p.decode(p.encode(original)) == original
    assert p.decode(p.encode(original)).background == 0


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_canonicalization_all_permutations_and_equal_lifetime_tie(k):
    amplitudes = (0, 4, 2, 1)[:k]
    lifetimes = (1, 1, 3, 0.5)[:k]
    reference, _ = _canonicalize_components(_FixedKParameters(amplitudes, lifetimes))
    for order in permutations(range(k)):
        original = _FixedKParameters(
            tuple(amplitudes[i] for i in order), tuple(lifetimes[i] for i in order)
        )
        canonical, mapping = _canonicalize_components(original)
        assert canonical == reference
        assert tuple(original.amplitudes[i] for i in mapping) == canonical.amplitudes
    assert _canonicalize_components(reference)[0] == reference


@pytest.mark.parametrize(
    "changes",
    [
        {"amplitudes": ()},
        {"amplitudes": (-1,)},
        {"amplitudes": (np.inf,)},
        {"lifetimes_ns": (0,)},
        {"lifetimes_ns": (np.nan,)},
        {"background": -1},
        {"temporal_shift_ns": np.inf},
        {"amplitudes": (1, 2)},
        {"background": 1j},
    ],
)
def test_invalid_physical_parameters(changes):
    values = dict(amplitudes=(1,), lifetimes_ns=(1,))
    values.update(changes)
    with pytest.raises(ValueError):
        _FixedKParameters(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"n_components": True},
        {"n_components": 0},
        {"n_components": 1.2},
        {"lifetime_bounds_ns": (0, 2)},
        {"lifetime_bounds_ns": (2, 1)},
        {"temporal_shift_bounds_ns": (0, np.inf)},
        {"amplitude_scale": 0},
        {"background_scale": np.nan},
    ],
)
def test_invalid_coordinate_contract(changes):
    values = dict(
        n_components=3,
        lifetime_bounds_ns=(0.1, 10),
        temporal_shift_bounds_ns=(-1, 1),
        amplitude_scale=2,
        background_scale=1,
    )
    values.update(changes)
    with pytest.raises(ValueError):
        _CoordinateMap(**values)


def test_coordinate_rejection_and_no_implicit_sorting():
    p = _CoordinateMap(2, (0.1, 10), (-1, 1), 2, 1)
    original = _FixedKParameters((2, 3), (5, 1))
    assert p.decode(p.encode(original)).lifetimes_ns == pytest.approx((5, 1))
    for bad in (np.zeros(5), [0, 0, 0, 0, 0, 2], [np.nan] * 6):
        assert not p.feasible(bad)
        with pytest.raises(ValueError):
            p.decode(bad)
    with pytest.raises(ValueError, match="count"):
        p.encode(_FixedKParameters((1,), (1,)))
    with pytest.raises(ValueError, match="bounds"):
        p.encode(replace(original, temporal_shift_ns=2))


@pytest.mark.parametrize("kind", ["raw_counts", "deterministic_expected_counts"])
def test_input_snapshot_and_semantics(kind):
    t = np.arange(5.0)
    y = np.ones(5)
    kernel = np.array([0, 1, 0, 0, 0.0])
    inputs = _FixedKInputs(t, y, kernel, kind)
    y[:] = 8
    kernel[:] = 0
    assert inputs.targets.tolist() == [1] * 5
    assert inputs.kernel[1] == 1
    with pytest.raises(ValueError):
        inputs.time_ns[0] = 9


@pytest.mark.parametrize(
    "targets,kind",
    [
        ([0.5] * 5, "raw_counts"),
        ([2**53] * 5, "raw_counts"),
        ([-1] * 5, "raw_counts"),
        ([np.nan] * 5, "raw_counts"),
        ([1] * 4, "raw_counts"),
        ([1] * 5, "processed"),
    ],
)
def test_input_rejections(targets, kind):
    with pytest.raises(ValueError):
        _FixedKInputs(np.arange(5.0), targets, [0, 1, 0, 0, 0], kind)


def test_kernel_requires_explicit_normalization_and_registration():
    with pytest.raises(ValueError, match="unit"):
        _FixedKInputs(np.arange(5.0), np.ones(5), np.ones(5), "raw_counts")
    with pytest.raises(ValueError, match="kernel"):
        _FixedKInputs(np.arange(5.0), np.ones(5), [-1, 2, 0, 0, 0], "raw_counts")


@pytest.mark.parametrize(
    "changes",
    [
        {"relative_step": 0},
        {"max_reductions": 0},
        {"relative_tolerance": -1},
        {"absolute_tolerance": np.nan},
    ],
)
def test_derivative_policy_requires_explicit_finite_controls(changes):
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("background", [False, True])
def test_positive_linear_underflow_is_not_silently_zero(background):
    tiny = np.nextafter(0.0, 1.0)
    p = _CoordinateMap(1, (0.1, 10), (-1, 1), 2, 2)
    parameters = _FixedKParameters(
        (1 if background else tiny,), (1,), tiny if background else 1
    )
    with pytest.raises(ValueError, match="underflow"):
        p.encode(parameters)
    p = replace(p, amplitude_scale=0.5, background_scale=0.5)
    x = np.ones(4)
    x[-1] = 0
    x[-2 if background else 0] = tiny
    with pytest.raises(ValueError, match="underflow"):
        p.decode(x)


def test_unrepresentable_log_bounds_and_amplitude_coordinates():
    with pytest.raises(ValueError, match="log"):
        _CoordinateMap(1, (1e100, np.nextafter(1e100, np.inf)), (-1, 1), 1, 1)
    p = _CoordinateMap(1, (0.1, 10), (-1, 1), 1e-300, 1)
    with pytest.raises(ValueError, match="unrepresentable"):
        p.encode(_FixedKParameters((1e300,), (1,)))
