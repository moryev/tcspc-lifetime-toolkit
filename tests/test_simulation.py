import numpy as np
import pytest

from tcspc_toolkit.simulation import (
    sample_photon_counts,
    simulate_biexponential_decay,
    simulate_monoexponential_decay,
    simulate_multiexponential_decay,
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

def test_simulate_biexponential_decay_returns_correct_shapes() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    expected_counts, measured_counts = (
        simulate_biexponential_decay(
            time=time,
            amplitude_1=8_000.0,
            lifetime_1=0.8,
            amplitude_2=2_000.0,
            lifetime_2=3.5,
            background=5.0,
            random_seed=42,
        )
    )

    assert expected_counts.shape == time.shape
    assert measured_counts.shape == time.shape

    assert np.issubdtype(
        expected_counts.dtype,
        np.floating,
    )
    assert np.issubdtype(
        measured_counts.dtype,
        np.integer,
    )


def test_simulate_multiexponential_decay_is_reproducible() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    amplitudes = np.array(
        [8_000.0, 2_000.0, 500.0],
        dtype=np.float64,
    )
    lifetimes = np.array(
        [0.5, 2.0, 6.0],
        dtype=np.float64,
    )

    expected_1, measured_1 = (
        simulate_multiexponential_decay(
            time=time,
            amplitudes=amplitudes,
            lifetimes=lifetimes,
            background=5.0,
            random_seed=42,
        )
    )

    expected_2, measured_2 = (
        simulate_multiexponential_decay(
            time=time,
            amplitudes=amplitudes,
            lifetimes=lifetimes,
            background=5.0,
            random_seed=42,
        )
    )

    np.testing.assert_array_equal(
        expected_1,
        expected_2,
    )
    np.testing.assert_array_equal(
        measured_1,
        measured_2,
    )


def test_biexponential_and_multiexponential_simulations_match() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    expected_bi, measured_bi = (
        simulate_biexponential_decay(
            time=time,
            amplitude_1=8_000.0,
            lifetime_1=0.8,
            amplitude_2=2_000.0,
            lifetime_2=3.5,
            background=5.0,
            random_seed=42,
        )
    )

    expected_multi, measured_multi = (
        simulate_multiexponential_decay(
            time=time,
            amplitudes=np.array(
                [8_000.0, 2_000.0],
                dtype=np.float64,
            ),
            lifetimes=np.array(
                [0.8, 3.5],
                dtype=np.float64,
            ),
            background=5.0,
            random_seed=42,
        )
    )

    np.testing.assert_allclose(
        expected_bi,
        expected_multi,
    )
    np.testing.assert_array_equal(
        measured_bi,
        measured_multi,
    )


def test_simulate_multiexponential_decay_rejects_mismatched_shapes() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="amplitudes and lifetimes must have the same shape",
    ):
        simulate_multiexponential_decay(
            time=time,
            amplitudes=np.array(
                [8_000.0, 2_000.0],
                dtype=np.float64,
            ),
            lifetimes=np.array(
                [0.8, 3.5, 6.0],
                dtype=np.float64,
            ),
            background=5.0,
            random_seed=42,
        )
