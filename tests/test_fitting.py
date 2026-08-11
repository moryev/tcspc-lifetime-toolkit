import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.fitting import (
    LifetimeFitResult,
    ReconvolutionFitResult,
    fit_monoexponential_decay,
    _reconvolution_model,
    fit_monoexponential_reconvolution,
    poisson_negative_log_likelihood
)
from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.simulation import (
    sample_photon_counts,
    simulate_monoexponential_decay,
)
from tcspc_toolkit.convolution import convolve_decay_with_irf


def test_fit_returns_lifetime_fit_result() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    _, measured_counts = simulate_monoexponential_decay(
        time=time,
        amplitude=100_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    result = fit_monoexponential_decay(
        time=time,
        counts=measured_counts,
        initial_guess=(
            90_000.0,
            2.0,
            5.0,
        ),
    )

    assert isinstance(
        result,
        LifetimeFitResult,
    )


def test_fit_recovers_lifetime_for_high_count_decay() -> None:
    true_lifetime = 2.5

    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    _, measured_counts = simulate_monoexponential_decay(
        time=time,
        amplitude=1_000_000.0,
        lifetime=true_lifetime,
        background=5.0,
        random_seed=42,
    )

    result = fit_monoexponential_decay(
        time=time,
        counts=measured_counts,
        initial_guess=(
            900_000.0,
            2.0,
            5.0,
        ),
    )

    assert abs(
        result.lifetime - true_lifetime
    ) < 0.1


def test_fitted_parameters_are_positive() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    _, measured_counts = simulate_monoexponential_decay(
        time=time,
        amplitude=100_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    result = fit_monoexponential_decay(
        time=time,
        counts=measured_counts,
        initial_guess=(
            90_000.0,
            2.0,
            4.0,
        ),
    )

    assert result.amplitude >= 0.0
    assert result.lifetime > 0.0
    assert result.background >= 0.0


def test_fit_rejects_mismatched_shapes() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    counts = np.ones(
        500,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="time and counts must have the same shape",
    ):
        fit_monoexponential_decay(
            time=time,
            counts=counts,
            initial_guess=(
                1000.0,
                2.0,
                5.0,
            ),
        )


def test_reconvolution_model_preserves_shape(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    result = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=2.0,
        background=5.0,
        temporal_shift=0.0,
    )

    assert result.shape == time_axis.shape


def test_reconvolution_model_zero_shift_matches_convolution(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    amplitude = 1000.0
    lifetime = 2.0
    background = 5.0

    decay = monoexponential_decay(
        time=time_axis,
        amplitude=1.0,
        lifetime=lifetime,
        background=0.0,
    )

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    expected = amplitude * convolved + background

    actual = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
        temporal_shift=0.0,
    )

    np.testing.assert_allclose(
        actual,
        expected,
    )


def test_reconvolution_model_longer_lifetime_has_slower_tail(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    short_lifetime = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=0.5,
        background=0.0,
        temporal_shift=0.0,
    )

    long_lifetime = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=3.0,
        background=0.0,
        temporal_shift=0.0,
    )

    tail_time = 5.0
    tail_index = np.searchsorted(
        time_axis,
        tail_time,
    )

    assert (
        long_lifetime[tail_index]
        > short_lifetime[tail_index]
    )


def _leading_edge_index(
    signal: NDArray[np.float64],
) -> int:
    threshold = 0.5 * np.max(signal)

    return int(
        np.flatnonzero(
            signal >= threshold
        )[0]
    )


def test_reconvolution_model_positive_shift_moves_leading_edge_later(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    unshifted = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=2.0,
        background=0.0,
        temporal_shift=0.0,
    )

    shifted = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=2.0,
        background=0.0,
        temporal_shift=0.3,
    )

    unshifted_leading_edge = _leading_edge_index(
        unshifted
    )

    shifted_leading_edge = _leading_edge_index(
        shifted
    )

    assert (
        shifted_leading_edge
        > unshifted_leading_edge
    )


def test_reconvolution_model_returns_finite_values(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    result = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=2.0,
        background=5.0,
        temporal_shift=0.1,
    )

    assert np.all(
        np.isfinite(result)
    )


def test_reconvolution_model_returns_non_negative_values(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    result = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=2.0,
        background=5.0,
        temporal_shift=0.1,
    )

    assert np.all(result >= 0.0)


def test_reconvolution_model_rejects_negative_amplitude(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="amplitude must be non-negative",
    ):
        _reconvolution_model(
            time=time_axis,
            irf=irf,
            amplitude=-1.0,
            lifetime=2.0,
            background=0.0,
            temporal_shift=0.0,
        )


def test_reconvolution_model_rejects_negative_lifetime(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="lifetime must be positive",
    ):
        _reconvolution_model(
            time=time_axis,
            irf=irf,
            amplitude=1000.0,
            lifetime=-1.0,
            background=0.0,
            temporal_shift=0.0,
        )


def test_reconvolution_model_rejects_zero_lifetime(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="lifetime must be positive",
    ):
        _reconvolution_model(
            time=time_axis,
            irf=irf,
            amplitude=1000.0,
            lifetime=0.0,
            background=0.0,
            temporal_shift=0.0,
        )


def test_reconvolution_model_rejects_negative_background(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    with pytest.raises(
        ValueError,
        match="background must be non-negative",
    ):
        _reconvolution_model(
            time=time_axis,
            irf=irf,
            amplitude=1000.0,
            lifetime=2.0,
            background=-1.0,
            temporal_shift=0.0,
        )


def test_reconvolution_model_accepts_negative_temporal_shift(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    result = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=1000.0,
        lifetime=2.0,
        background=0.0,
        temporal_shift=-0.2,
    )

    assert result.shape == time_axis.shape


def test_reconvolution_fit_recovers_lifetime_and_shift_for_high_counts(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    true_amplitude = 1_000_000.0
    true_lifetime = 0.3
    true_background = 20.0
    true_temporal_shift = 0.18

    expected_counts = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=true_amplitude,
        lifetime=true_lifetime,
        background=true_background,
        temporal_shift=true_temporal_shift,
    )

    rng = np.random.default_rng(42)

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    result = fit_monoexponential_reconvolution(
        time=time_axis,
        counts=measured_counts,
        irf=irf,
        initial_guess=(
            900_000.0,
            0.5,
            10.0,
            0.0,
        ),
        temporal_shift_bounds=(
            -0.5,
            0.5,
        ),
    )

    assert result.success

    assert abs(
        result.lifetime - true_lifetime
    ) < 0.05

    assert abs(
        result.temporal_shift - true_temporal_shift
    ) < 0.05


def test_reconvolution_fit_returns_result_with_fitted_curve(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    expected_counts = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=100_000.0,
        lifetime=0.5,
        background=5.0,
        temporal_shift=0.1,
    )

    rng = np.random.default_rng(42)

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    result = fit_monoexponential_reconvolution(
        time=time_axis,
        counts=measured_counts,
        irf=irf,
        initial_guess=(
            90_000.0,
            0.7,
            5.0,
            0.0,
        ),
    )

    assert isinstance(
        result,
        ReconvolutionFitResult,
    )

    assert result.fitted_curve.shape == time_axis.shape

    assert np.all(
        np.isfinite(result.fitted_curve)
    )

    assert result.success


def test_reconvolution_fit_returns_physical_parameters(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    expected_counts = _reconvolution_model(
        time=time_axis,
        irf=irf,
        amplitude=100_000.0,
        lifetime=0.5,
        background=5.0,
        temporal_shift=0.1,
    )

    rng = np.random.default_rng(42)

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    result = fit_monoexponential_reconvolution(
        time=time_axis,
        counts=measured_counts,
        irf=irf,
        initial_guess=(
            90_000.0,
            0.7,
            4.0,
            0.0,
        ),
        temporal_shift_bounds=(
            -0.5,
            0.5,
        ),
    )

    assert result.amplitude >= 0.0
    assert result.lifetime > 0.0
    assert result.background >= 0.0

    assert (
        -0.5
        <= result.temporal_shift
        <= 0.5
    )


def test_reconvolution_fit_rejects_invalid_temporal_shift_bounds(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    counts = np.ones_like(
        time_axis,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="lower temporal shift bound",
    ):
        fit_monoexponential_reconvolution(
            time=time_axis,
            counts=counts,
            irf=irf,
            initial_guess=(
                1000.0,
                1.0,
                5.0,
                0.0,
            ),
            temporal_shift_bounds=(
                0.5,
                -0.5,
            ),
        )


def test_poisson_negative_log_likelihood_is_finite_for_valid_counts() -> None:
    observed = np.array([10, 20, 15, 5])
    expected = np.array([9.0, 21.0, 14.0, 6.0])

    nll = poisson_negative_log_likelihood(
        observed=observed,
        expected=expected,
    )

    assert np.isfinite(nll)


def test_poisson_negative_log_likelihood_handles_zero_observed_counts() -> None:
    observed = np.array([0, 0, 5, 10])
    expected = np.array([0.5, 1.0, 4.0, 11.0])

    nll = poisson_negative_log_likelihood(
        observed=observed,
        expected=expected,
    )

    assert np.isfinite(nll)


def test_poisson_negative_log_likelihood_rejects_negative_expected_counts() -> None:
    observed = np.array([5, 10, 15])
    expected = np.array([5.0, -1.0, 15.0])

    with pytest.raises(
        ValueError,
        match="expected counts must be non-negative",
    ):
        poisson_negative_log_likelihood(
            observed=observed,
            expected=expected,
        )


def test_poisson_negative_log_likelihood_allows_zero_expected_for_zero_observed() -> None:
    observed = np.array([0, 5, 10])
    expected = np.array([0.0, 5.0, 10.0])

    nll = poisson_negative_log_likelihood(
        observed=observed,
        expected=expected,
    )

    assert np.isfinite(nll)


def test_poisson_negative_log_likelihood_is_infinite_for_positive_observed_and_zero_expected() -> None:
    observed = np.array([1, 5, 10])
    expected = np.array([0.0, 5.0, 10.0])

    nll = poisson_negative_log_likelihood(
        observed=observed,
        expected=expected,
    )

    assert np.isinf(nll)


def test_poisson_negative_log_likelihood_is_lower_for_better_model() -> None:
    observed = np.array([10, 20, 30, 20, 10])

    expected_close = np.array([11.0, 19.0, 29.0, 21.0, 9.0])
    expected_far = np.array([3.0, 8.0, 12.0, 8.0, 3.0])

    nll_close = poisson_negative_log_likelihood(
        observed=observed,
        expected=expected_close,
    )

    nll_far = poisson_negative_log_likelihood(
        observed=observed,
        expected=expected_far,
    )

    assert nll_close < nll_far


def test_poisson_negative_log_likelihood_rejects_shape_mismatch() -> None:
    observed = np.array([5, 10, 15])
    expected = np.array([5.0, 10.0])

    with pytest.raises(
        ValueError,
        match="observed and expected must have the same shape",
    ):
        poisson_negative_log_likelihood(
            observed=observed,
            expected=expected,
        )


