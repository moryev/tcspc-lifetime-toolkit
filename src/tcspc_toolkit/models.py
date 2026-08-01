import numpy as np
from numpy.typing import NDArray


def monoexponential_decay(
    time: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float = 0.0,
) -> NDArray[np.float64]:
    """Evaluate a mono-exponential decay model."""
    if lifetime <= 0:
        raise ValueError("lifetime must be positive")

    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    return amplitude * np.exp(-time / lifetime) + background


def biexponential_decay(
    time: NDArray[np.float64],
    amplitude_1: float,
    lifetime_1: float,
    amplitude_2: float,
    lifetime_2: float,
    background: float = 0.0,
) -> NDArray[np.float64]:
    """
    Evaluate a bi-exponential decay model.

    The model is

        I(t) = A1 * exp(-t / tau1)
             + A2 * exp(-t / tau2)
             + background

    Parameters
    ----------
    time:
        Time axis.
    amplitude_1:
        Amplitude of the first exponential component.
    lifetime_1:
        Lifetime of the first exponential component.
    amplitude_2:
        Amplitude of the second exponential component.
    lifetime_2:
        Lifetime of the second exponential component.
    background:
        Constant background level.

    Returns
    -------
    numpy.ndarray
        Expected decay values evaluated at each time point.
    """
    amplitudes = np.array(
        [amplitude_1, amplitude_2],
        dtype=np.float64,
    )
    lifetimes = np.array(
        [lifetime_1, lifetime_2],
        dtype=np.float64,
    )

    return multiexponential_decay(
        time=time,
        amplitudes=amplitudes,
        lifetimes=lifetimes,
        background=background,
    )


def multiexponential_decay(
    time: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    lifetimes: NDArray[np.float64],
    background: float = 0.0,
) -> NDArray[np.float64]:
    """
    Evaluate a multi-exponential decay model.

    The model is

        I(t) = sum_k A_k * exp(-t / tau_k) + background

    Parameters
    ----------
    time:
        One-dimensional time axis with shape ``(n_time_bins,)``.
    amplitudes:
        One-dimensional array of component amplitudes with shape
        ``(n_components,)``.
    lifetimes:
        One-dimensional array of component lifetimes with shape
        ``(n_components,)``.
    background:
        Constant background level.

    Returns
    -------
    numpy.ndarray
        Expected decay values with shape ``(n_time_bins,)``.

    Raises
    ------
    ValueError
        If the input arrays are not one-dimensional, contain no
        components, have different shapes, or contain invalid values.
    """
    time = np.asarray(
        time,
        dtype=np.float64,
    )
    amplitudes = np.asarray(
        amplitudes,
        dtype=np.float64,
    )
    lifetimes = np.asarray(
        lifetimes,
        dtype=np.float64,
    )

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")

    if amplitudes.ndim != 1:
        raise ValueError("amplitudes must be one-dimensional")

    if lifetimes.ndim != 1:
        raise ValueError("lifetimes must be one-dimensional")

    if amplitudes.size == 0:
        raise ValueError(
            "at least one exponential component is required"
        )

    if amplitudes.shape != lifetimes.shape:
        raise ValueError(
            "amplitudes and lifetimes must have the same shape"
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values")

    if not np.all(np.isfinite(amplitudes)):
        raise ValueError(
            "amplitudes must contain only finite values"
        )

    if not np.all(np.isfinite(lifetimes)):
        raise ValueError(
            "lifetimes must contain only finite values"
        )

    if not np.isfinite(background):
        raise ValueError("background must be finite")

    if np.any(amplitudes < 0):
        raise ValueError(
            "amplitudes must be non-negative"
        )

    if np.any(lifetimes <= 0):
        raise ValueError(
            "lifetimes must be positive"
        )

    if background < 0:
        raise ValueError(
            "background must be non-negative"
        )

    component_decays = amplitudes[:, np.newaxis] * np.exp(
        -time[np.newaxis, :]
        / lifetimes[:, np.newaxis]
    )

    return component_decays.sum(axis=0) + background
