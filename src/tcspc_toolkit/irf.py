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


def normalize_irf(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize an instrument response function to unit integrated area.

    The IRF is normalized according to

        integral IRF(t) dt = 1,

    where the integral is approximated numerically using the trapezoidal
    rule over the supplied time axis.

    Parameters
    ----------
    time
        One-dimensional, strictly increasing time axis.
    irf
        One-dimensional array containing non-negative IRF values.

    Returns
    -------
    NDArray[np.float64]
        A new IRF array with unit integrated area.

    Raises
    ------
    ValueError
        If either input is not one-dimensional, if their lengths differ,
        if they contain non-finite values, if the time axis is not strictly
        increasing, if the IRF contains negative values, or if the
        integrated IRF area is not positive.
    """
    time_array = np.asarray(time, dtype=np.float64)
    irf_array = np.asarray(irf, dtype=np.float64)

    if time_array.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if irf_array.ndim != 1:
        raise ValueError("irf must be one-dimensional.")

    if time_array.shape != irf_array.shape:
        raise ValueError("time and irf must have the same shape.")

    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(irf_array)):
        raise ValueError("irf must contain only finite values.")

    if np.any(irf_array < 0.0):
        raise ValueError("irf values must be non-negative.")

    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing.")

    area = np.trapezoid(irf_array, x=time_array)

    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("irf must have a positive integrated area.")

    return irf_array / area


from numpy.typing import NDArray
import numpy as np


def shift_irf(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    shift: float,
) -> NDArray[np.float64]:
    """Shift an instrument response function along its time axis.

    The shifted IRF is defined as

        shifted_irf(t) = irf(t - shift)

    so that a positive shift moves the IRF toward later times and a
    negative shift moves it toward earlier times.

    Linear interpolation is used, which permits shifts that are not
    integer multiples of the time-bin width. Values outside the supplied
    time interval are replaced with zero.

    The shifted IRF is not renormalized. If part of the IRF moves outside
    the observed time window, its integrated area will therefore decrease.

    Parameters
    ----------
    time
        One-dimensional, strictly increasing time axis.
    irf
        One-dimensional instrument response function evaluated at `time`.
    shift
        Temporal shift in the same units as `time`.

    Returns
    -------
    NDArray[np.float64]
        Shifted IRF with the same shape as the input IRF.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional, have different lengths,
        contain non-finite values, the time axis is not strictly increasing,
        the IRF contains negative values, or the shift is not finite.
    """
    time = np.asarray(time, dtype=np.float64)
    irf = np.asarray(irf, dtype=np.float64)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if irf.ndim != 1:
        raise ValueError("irf must be one-dimensional.")

    if time.shape != irf.shape:
        raise ValueError("time and irf must have the same shape.")

    if time.size < 2:
        raise ValueError("time and irf must contain at least two values.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(irf)):
        raise ValueError("irf must contain only finite values.")

    if not np.isfinite(shift):
        raise ValueError("shift must be finite.")

    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be strictly increasing.")

    if np.any(irf < 0.0):
        raise ValueError("irf values must be non-negative.")

    return np.interp(
        time - shift,
        time,
        irf,
        left=0.0,
        right=0.0,
    )
