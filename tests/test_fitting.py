import numpy as np
import pytest

from tcspc_toolkit.fitting import (
    LifetimeFitResult,
    fit_monoexponential_decay,
)
from tcspc_toolkit.simulation import simulate_monoexponential_decay


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


