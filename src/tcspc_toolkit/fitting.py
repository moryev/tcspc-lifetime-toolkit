from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from tcspc_toolkit.models import monoexponential_decay


@dataclass(frozen=True)
class LifetimeFitResult:
    """Results of a mono-exponential lifetime fit."""

    amplitude: float
    lifetime: float
    background: float

    amplitude_std: float
    lifetime_std: float
    background_std: float


def fit_monoexponential_decay(
    time: NDArray[np.float64],
    counts: NDArray[np.float64] | NDArray[np.int64],
    initial_guess: tuple[float, float, float],
) -> LifetimeFitResult:
    """Fit a mono-exponential decay to measured count data.

    Parameters
    ----------
    time:
        One-dimensional array containing time-bin positions.
    counts:
        One-dimensional array containing measured counts.
    initial_guess:
        Initial guesses for amplitude, lifetime, and background.

    Returns
    -------
    LifetimeFitResult
        Fitted parameters and their estimated standard deviations.
    """
    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array")

    if counts.ndim != 1:
        raise ValueError("counts must be a one-dimensional array")

    if time.shape != counts.shape:
        raise ValueError("time and counts must have the same shape")

    if len(time) < 3:
        raise ValueError("at least three data points are required")

    # TODO: For photon-counting data, ordinary unweighted least squares is not ultimately the best statistical method
    #       because the variance changes with the expected count. Later, add Poisson-aware fitting.
    #       For now, curve_fit is a useful baseline.
    optimal_parameters, covariance_matrix = curve_fit(
        f=monoexponential_decay,
        xdata=time,
        ydata=counts,
        p0=initial_guess,
        bounds=(
            (0.0, 1e-12, 0.0),
            (np.inf, np.inf, np.inf),
        ),
    )

    amplitude, lifetime, background = optimal_parameters

    parameter_variances = np.diag(covariance_matrix)
    parameter_standard_deviations = np.sqrt(parameter_variances)

    amplitude_std, lifetime_std, background_std = (
        parameter_standard_deviations
    )

    return LifetimeFitResult(
        amplitude=float(amplitude),
        lifetime=float(lifetime),
        background=float(background),
        amplitude_std=float(amplitude_std),
        lifetime_std=float(lifetime_std),
        background_std=float(background_std),
    )
