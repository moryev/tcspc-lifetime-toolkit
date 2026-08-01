import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.models import monoexponential_decay


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
