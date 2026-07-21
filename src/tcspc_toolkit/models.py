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
