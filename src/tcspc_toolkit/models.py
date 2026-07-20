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


def sample_photon_counts(
    expected_counts: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample photon counts using Poisson statistics."""
    if np.any(expected_counts < 0):
        raise ValueError("expected counts must be non-negative")

    return rng.poisson(expected_counts)


def simulate_ideal_decay(
    time: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float,
    random_seed: int | None = None,
) -> NDArray[np.int64]:
    rng = np.random.default_rng(random_seed)

    expected = monoexponential_decay(
        time=time,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
    )

    return sample_photon_counts(expected, rng)
