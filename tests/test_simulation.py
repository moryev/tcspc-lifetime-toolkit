import numpy as np
import pytest

from tcspc_toolkit.simulation import (
    sample_photon_counts,
    simulate_monoexponential_decay,
)


def test_simulate_ideal_decay_returns_correct_shapes() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    expected_counts, measured_counts = simulate_monoexponential_decay(
        time=time,
        amplitude=10_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    assert expected_counts.shape == time.shape
    assert measured_counts.shape == time.shape


def test_simulate_ideal_decay_returns_expected_dtypes() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    expected_counts, measured_counts = simulate_monoexponential_decay(
        time=time,
        amplitude=10_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    assert np.issubdtype(
        expected_counts.dtype,
        np.floating,
    )

    assert np.issubdtype(
        measured_counts.dtype,
        np.integer,
    )


def test_simulation_is_reproducible_with_fixed_seed() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    expected_1, measured_1 = simulate_monoexponential_decay(
        time=time,
        amplitude=10_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    expected_2, measured_2 = simulate_monoexponential_decay(
        time=time,
        amplitude=10_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    assert np.array_equal(
        expected_1,
        expected_2,
    )

    assert np.array_equal(
        measured_1,
        measured_2,
    )


def test_different_random_seeds_produce_different_measurements() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    expected_1, measured_1 = simulate_monoexponential_decay(
        time=time,
        amplitude=10_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=42,
    )

    expected_2, measured_2 = simulate_monoexponential_decay(
        time=time,
        amplitude=10_000.0,
        lifetime=2.5,
        background=5.0,
        random_seed=43,
    )

    assert np.array_equal(
        expected_1,
        expected_2,
    )

    assert not np.array_equal(
        measured_1,
        measured_2,
    )


def test_sampled_photon_counts_are_non_negative() -> None:
    expected_counts = np.array(
        [0.0, 1.0, 10.0, 100.0],
        dtype=np.float64,
    )

    rng = np.random.default_rng(42)

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    assert np.all(measured_counts >= 0)

    assert np.issubdtype(
        measured_counts.dtype,
        np.integer,
    )


def test_negative_expected_counts_raise_error() -> None:
    expected_counts = np.array(
        [10.0, 5.0, -1.0],
        dtype=np.float64,
    )

    rng = np.random.default_rng(42)

    with pytest.raises(
        ValueError,
        match="expected counts must be non-negative",
    ):
        sample_photon_counts(
            expected_counts=expected_counts,
            rng=rng,
        )

