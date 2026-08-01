import numpy as np
import pytest

from tcspc_toolkit.models import (
    biexponential_decay,
    monoexponential_decay,
    multiexponential_decay,
)


def test_monoexponential_decay_preserves_time_shape() -> None:
    time = np.linspace(
        start=0.0,
        stop=10.0,
        num=100,
        dtype=np.float64,
    )

    signal = monoexponential_decay(
        time=time,
        amplitude=1000.0,
        lifetime=2.5,
        background=5.0,
    )

    assert signal.shape == time.shape


def test_negative_lifetime_raises_error() -> None:
    time = np.linspace(
        start=0.0,
        stop=10.0,
        num=100,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="lifetime must be positive",
    ):
        monoexponential_decay(
            time=time,
            amplitude=1000.0,
            lifetime=-2.5,
            background=5.0,
        )


def test_zero_lifetime_raises_error() -> None:
    time = np.linspace(
        start=0.0,
        stop=10.0,
        num=100,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="lifetime must be positive",
    ):
        monoexponential_decay(
            time=time,
            amplitude=1000.0,
            lifetime=0.0,
            background=5.0,
        )


def test_negative_amplitude_raises_error() -> None:
    time = np.linspace(
        start=0.0,
        stop=10.0,
        num=100,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="amplitude must be non-negative",
    ):
        monoexponential_decay(
            time=time,
            amplitude=-1000.0,
            lifetime=2.5,
            background=5.0,
        )


def test_zero_background_gives_amplitude_at_time_zero() -> None:
    time = np.array(
        [0.0],
        dtype=np.float64,
    )

    amplitude = 1000.0

    signal = monoexponential_decay(
        time=time,
        amplitude=amplitude,
        lifetime=2.5,
        background=0.0,
    )

    assert np.isclose(
        signal[0],
        amplitude,
    )


def test_signal_decreases_for_positive_amplitude_and_lifetime() -> None:
    time = np.array(
        [0.0, 1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    signal = monoexponential_decay(
        time=time,
        amplitude=1000.0,
        lifetime=2.5,
        background=0.0,
    )

    signal_differences = np.diff(signal)

    assert np.all(signal_differences < 0.0)


def test_signal_approaches_background_at_long_times() -> None:
    background = 5.0

    time = np.array(
        [100.0],
        dtype=np.float64,
    )

    signal = monoexponential_decay(
        time=time,
        amplitude=1000.0,
        lifetime=2.5,
        background=background,
    )

    assert np.isclose(
        signal[0],
        background,
        atol=1e-10,
    )


def test_zero_background_signal_remains_non_negative() -> None:
    time = np.linspace(
        start=0.0,
        stop=20.0,
        num=512,
        dtype=np.float64,
    )

    signal = monoexponential_decay(
        time=time,
        amplitude=1000.0,
        lifetime=2.5,
        background=0.0,
    )

    assert np.all(signal >= 0.0)


def test_biexponential_decay_matches_manual_calculation() -> None:
    time = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    amplitude_1 = 800.0
    lifetime_1 = 1.5
    amplitude_2 = 300.0
    lifetime_2 = 4.0
    background = 5.0

    signal = biexponential_decay(
        time=time,
        amplitude_1=amplitude_1,
        lifetime_1=lifetime_1,
        amplitude_2=amplitude_2,
        lifetime_2=lifetime_2,
        background=background,
    )

    expected = (
        amplitude_1 * np.exp(-time / lifetime_1)
        + amplitude_2 * np.exp(-time / lifetime_2)
        + background
    )

    np.testing.assert_allclose(
        signal,
        expected,
    )


def test_multiexponential_decay_matches_manual_calculation() -> None:
    time = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )

    amplitudes = np.array(
        [800.0, 300.0, 100.0],
        dtype=np.float64,
    )
    lifetimes = np.array(
        [0.8, 2.5, 6.0],
        dtype=np.float64,
    )
    background = 5.0

    signal = multiexponential_decay(
        time=time,
        amplitudes=amplitudes,
        lifetimes=lifetimes,
        background=background,
    )

    expected = (
        amplitudes[0] * np.exp(-time / lifetimes[0])
        + amplitudes[1] * np.exp(-time / lifetimes[1])
        + amplitudes[2] * np.exp(-time / lifetimes[2])
        + background
    )

    np.testing.assert_allclose(
        signal,
        expected,
    )


def test_biexponential_decay_matches_multiexponential_decay() -> None:
    time = np.linspace(
        start=0.0,
        stop=10.0,
        num=100,
        dtype=np.float64,
    )

    biexponential_signal = biexponential_decay(
        time=time,
        amplitude_1=800.0,
        lifetime_1=1.5,
        amplitude_2=300.0,
        lifetime_2=4.0,
        background=5.0,
    )

    multiexponential_signal = multiexponential_decay(
        time=time,
        amplitudes=np.array(
            [800.0, 300.0],
            dtype=np.float64,
        ),
        lifetimes=np.array(
            [1.5, 4.0],
            dtype=np.float64,
        ),
        background=5.0,
    )

    np.testing.assert_allclose(
        biexponential_signal,
        multiexponential_signal,
    )


def test_multiexponential_decay_rejects_mismatched_shapes() -> None:
    time = np.linspace(
        start=0.0,
        stop=10.0,
        num=100,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="amplitudes and lifetimes must have the same shape",
    ):
        multiexponential_decay(
            time=time,
            amplitudes=np.array(
                [800.0, 300.0],
                dtype=np.float64,
            ),
            lifetimes=np.array(
                [1.5, 4.0, 6.0],
                dtype=np.float64,
            ),
            background=5.0,
        )
