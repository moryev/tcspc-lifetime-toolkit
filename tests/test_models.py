import numpy as np
import pytest

from tcspc_toolkit.models import monoexponential_decay


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
