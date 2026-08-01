import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.models import (
    biexponential_decay,
    monoexponential_decay,
    multiexponential_decay,
)


def sample_photon_counts(
    expected_counts: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample photon counts using Poisson statistics."""
    if np.any(expected_counts < 0):
        raise ValueError("expected counts must be non-negative")

    return rng.poisson(expected_counts)


def simulate_monoexponential_decay(
    time: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate expected and Poisson-sampled mono-exponential decay curves."""
    rng = np.random.default_rng(random_seed)

    expected_counts = monoexponential_decay(
        time=time,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    return expected_counts, measured_counts


def simulate_biexponential_decay(
    time: NDArray[np.float64],
    amplitude_1: float,
    lifetime_1: float,
    amplitude_2: float,
    lifetime_2: float,
    background: float = 0.0,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate expected and Poisson-sampled bi-exponential decay curves.

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
    random_seed:
        Seed used for reproducible Poisson sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Expected photon counts and Poisson-sampled measured counts.
    """
    rng = np.random.default_rng(random_seed)

    expected_counts = biexponential_decay(
        time=time,
        amplitude_1=amplitude_1,
        lifetime_1=lifetime_1,
        amplitude_2=amplitude_2,
        lifetime_2=lifetime_2,
        background=background,
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    return expected_counts, measured_counts


def simulate_multiexponential_decay(
    time: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    lifetimes: NDArray[np.float64],
    background: float = 0.0,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate expected and Poisson-sampled multi-exponential decay curves.

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
    random_seed:
        Seed used for reproducible Poisson sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Expected photon counts and Poisson-sampled measured counts.
    """
    rng = np.random.default_rng(random_seed)

    expected_counts = multiexponential_decay(
        time=time,
        amplitudes=amplitudes,
        lifetimes=lifetimes,
        background=background,
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    return expected_counts, measured_counts