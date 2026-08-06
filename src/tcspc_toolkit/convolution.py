"""Numerical convolution utilities for TCSPC forward modelling.

This module combines ideal fluorescence-decay curves with instrument-response
functions. It handles discrete convolution, time-bin scaling, output alignment,
and truncation to the original measurement window.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve


def convolve_decay_with_irf(
    time: NDArray[np.float64],
    decay: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convolve an ideal decay curve with an instrument-response function.

    The input arrays must be one-dimensional and have equal lengths. The time
    axis must be strictly increasing and uniformly spaced.

    The IRF is assumed to be normalized by continuous area. Therefore, the
    discrete convolution is multiplied by the time-bin width. NB! A user should explicitly use
    normalize_irf(time, irf) function before convolution. Keeping normalization separate makes
    the workflow transparent and avoids silently changing user inputs.

    The full convolution is truncated to the original measurement window. The
    output is aligned to the beginning of the supplied time grid, and no
    automatic correction for the IRF centre is applied.

    Parameters
    ----------
    time
        Uniformly spaced, strictly increasing time axis.
    decay
        Non-negative ideal decay values.
    irf
        Non-negative instrument-response values.
    # TODO: Although experimentally measured IRFs can contain slightly negative values after baseline subtraction or preprocessing,
    #       the simulation-layer IRF represents a non-negative response kernel. Negative values should therefore be rejected here.
    #       Measured-IRF preprocessing can later be implemented as a separate workflow.

    Returns
    -------
    numpy.ndarray
        Instrument-broadened decay with the same shape as the input arrays.

    Raises
    ------
    ValueError
        If an input array is not one-dimensional, the arrays have unequal
        lengths, the time axis is invalid, or the decay or IRF contains
        non-finite or negative values.
    """

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")

    if decay.ndim != 1:
        raise ValueError("decay must be one-dimensional")

    if irf.ndim != 1:
        raise ValueError("irf must be one-dimensional")

    if not (time.size == decay.size == irf.size):
        raise ValueError(
            "time, decay, and irf must have equal lengths"
        )
    # TODO: Even though mathematical convolution can combine arrays of different lengths,
    #       the first toolkit implementation deliberately uses one common measurement grid.

    if time.size < 2:
        raise ValueError(
            "time must contain at least two points"
        )

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values")

    time_differences = np.diff(time)
    if np.any(time_differences <= 0.0):
        raise ValueError("time must be strictly increasing")

    if not np.allclose(
        time_differences,
        time_differences[0],
    ):
        raise ValueError("time must be uniformly spaced")

    if not np.all(np.isfinite(decay)):
        raise ValueError("decay must contain only finite values")

    if np.any(decay < 0.0):
        raise ValueError("decay must contain only non-negative values")

    if not np.all(np.isfinite(irf)):
        raise ValueError("irf must contain only finite values")

    if np.any(irf < 0.0):
        raise ValueError("irf must contain only non-negative values")

    dt = time_differences[0]

    full_convolution = fftconvolve(
        decay,
        irf,
        mode="full",
    )

    convolved = full_convolution[: decay.size] * dt

    convolved = np.maximum(convolved, 0.0)

    return convolved