import numpy as np
import pytest

from tcspc_toolkit.simulation import (
    sample_photon_counts,
    simulate_biexponential_decay,
    simulate_irf_convolved_biexponential_histogram,
    simulate_irf_convolved_histogram,
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


def test_irf_convolved_simulation_preserves_expected_count_budget() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    signal_photon_count = 5_000
    background_per_bin = 0.5

    rng = np.random.default_rng(42)

    measured_counts, metadata = (
        simulate_irf_convolved_histogram(
            time=time,
            lifetime_ns=2.5,
            signal_photon_count=signal_photon_count,
            background_per_bin=background_per_bin,
            irf_centre_ns=1.0,
            irf_fwhm_ns=0.3,
            irf_shift_ns=0.1,
            rng=rng,
        )
    )

    expected_background_counts = (
        background_per_bin
        * time.size
    )

    expected_total_counts = (
        signal_photon_count
        + expected_background_counts
    )

    assert measured_counts.shape == time.shape

    assert np.issubdtype(
        measured_counts.dtype,
        np.integer,
    )

    assert metadata[
        "expected_signal_counts"
    ] == pytest.approx(
        signal_photon_count
    )

    assert metadata[
        "expected_background_counts"
    ] == pytest.approx(
        expected_background_counts
    )

    assert metadata[
        "expected_total_counts"
    ] == pytest.approx(
        expected_total_counts
    )

    assert metadata[
        "measured_total_counts"
    ] == pytest.approx(
        measured_counts.sum()
    )


def test_irf_convolved_simulation_is_reproducible() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    measured_a, metadata_a = (
        simulate_irf_convolved_histogram(
            time=time,
            lifetime_ns=2.5,
            signal_photon_count=5_000,
            background_per_bin=0.5,
            irf_centre_ns=1.0,
            irf_fwhm_ns=0.3,
            irf_shift_ns=0.1,
            rng=rng_a,
        )
    )

    measured_b, metadata_b = (
        simulate_irf_convolved_histogram(
            time=time,
            lifetime_ns=2.5,
            signal_photon_count=5_000,
            background_per_bin=0.5,
            irf_centre_ns=1.0,
            irf_fwhm_ns=0.3,
            irf_shift_ns=0.1,
            rng=rng_b,
        )
    )

    np.testing.assert_array_equal(
        measured_a,
        measured_b,
    )

    assert metadata_a == metadata_b


def test_irf_convolved_biexponential_histogram_returns_expected_outputs() -> None:
    time = np.linspace(
        0.0,
        20.0,
        512,
        dtype=np.float64,
    )

    counts, metadata = (
        simulate_irf_convolved_biexponential_histogram(
            time=time,
            primary_lifetime_ns=2.0,
            secondary_lifetime_ns=4.0,
            secondary_fraction=0.10,
            signal_photon_count=10_000,
            background_per_bin=1.0,
            irf_centre_ns=2.0,
            irf_fwhm_ns=0.3,
            irf_shift_ns=0.05,
            rng=np.random.default_rng(42),
        )
    )

    assert counts.shape == time.shape
    assert np.issubdtype(
        counts.dtype,
        np.integer,
    )

    assert metadata[
        "primary_lifetime_ns"
    ] == pytest.approx(2.0)

    assert metadata[
        "secondary_lifetime_ns"
    ] == pytest.approx(4.0)

    assert metadata[
        "secondary_fraction"
    ] == pytest.approx(0.10)

    assert metadata[
        "expected_signal_counts"
    ] == pytest.approx(10_000.0)


def test_irf_convolved_biexponential_histogram_is_reproducible() -> None:
    time = np.linspace(
        0.0,
        20.0,
        512,
        dtype=np.float64,
    )

    kwargs = {
        "time": time,
        "primary_lifetime_ns": 2.0,
        "secondary_lifetime_ns": 4.0,
        "secondary_fraction": 0.10,
        "signal_photon_count": 10_000,
        "background_per_bin": 1.0,
        "irf_centre_ns": 2.0,
        "irf_fwhm_ns": 0.3,
        "irf_shift_ns": 0.05,
    }

    counts_1, _ = (
        simulate_irf_convolved_biexponential_histogram(
            **kwargs,
            rng=np.random.default_rng(42),
        )
    )

    counts_2, _ = (
        simulate_irf_convolved_biexponential_histogram(
            **kwargs,
            rng=np.random.default_rng(42),
        )
    )

    np.testing.assert_array_equal(
        counts_1,
        counts_2,
    )


@pytest.mark.parametrize(
    "secondary_fraction",
    [-0.1, 1.0, 1.1],
)
def test_irf_convolved_biexponential_histogram_rejects_invalid_fraction(
    secondary_fraction: float,
) -> None:
    time = np.linspace(
        0.0,
        20.0,
        128,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="secondary_fraction",
    ):
        simulate_irf_convolved_biexponential_histogram(
            time=time,
            primary_lifetime_ns=2.0,
            secondary_lifetime_ns=4.0,
            secondary_fraction=secondary_fraction,
            signal_photon_count=1_000,
            background_per_bin=0.0,
            irf_centre_ns=2.0,
            irf_fwhm_ns=0.3,
            irf_shift_ns=0.0,
            rng=np.random.default_rng(42),
        )


