"""Instrument response function (IRF) generation and manipulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_gaussian_irf(
    time: NDArray[np.float64],
    centre: float,
    fwhm: float,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a Gaussian instrument response function.

    Parameters
    ----------
    time
        One-dimensional array containing the time-bin coordinates.
    centre
        Temporal centre of the Gaussian IRF.
    fwhm
        Full width at half maximum of the Gaussian IRF.
        Must be finite and strictly positive.
    amplitude
        Peak amplitude of the Gaussian IRF. Must be finite and
        non-negative.

    Returns
    -------
    numpy.ndarray
        Gaussian IRF evaluated at the supplied time coordinates.
        The returned array has the same shape as ``time``.

    Raises
    ------
    ValueError
        If ``time`` is not one-dimensional, is empty, or contains
        non-finite values.
        If ``centre`` is not finite.
        If ``fwhm`` is not finite or is not strictly positive.
        If ``amplitude`` is not finite or is negative.

    Notes
    -----
    This function does not normalize the IRF. The ``amplitude`` parameter
    specifies its peak height, not its discrete sum or continuous area.
    """

    time_array = np.asarray(time, dtype=np.float64)

    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")

    if time_array.size == 0:
        raise ValueError("time must not be empty.")

    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values.")

    if not np.isfinite(centre):
        raise ValueError("centre must be finite.")

    if not np.isfinite(fwhm):
        raise ValueError("fwhm must be finite.")

    if fwhm <= 0.0:
        raise ValueError("fwhm must be greater than zero.")

    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite.")

    if amplitude < 0.0:
        raise ValueError("amplitude must be non-negative.")

    # Compute standard deviation for the given Gaussian IRF
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    exponent = -0.5 * ((time_array - centre) / sigma) ** 2

    return amplitude * np.exp(exponent)
