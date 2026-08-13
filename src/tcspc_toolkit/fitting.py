from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit, least_squares, minimize

from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.irf import shift_irf
from tcspc_toolkit.convolution import convolve_decay_with_irf


@dataclass(frozen=True)
class LifetimeFitResult:
    """Results of a mono-exponential lifetime fit."""

    amplitude: float
    lifetime: float
    background: float

    amplitude_std: float
    lifetime_std: float
    background_std: float


@dataclass(frozen=True)
class ReconvolutionFitResult:
    """Results of a mono-exponential reconvolution fit."""

    amplitude: float
    lifetime: float
    background: float
    temporal_shift: float

    fitted_curve: NDArray[np.float64]
    success: bool


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


# Reconvolution fitting block
def _reconvolution_model(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float,
    temporal_shift: float,
) -> NDArray[np.float64]:
    """
     time:
        One-dimensional array containing time-bin positions.
    irf:
        Instrument response function evaluated on the same time grid.
        The IRF should already be normalized.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    if lifetime <= 0:
        raise ValueError("lifetime must be positive")

    if background < 0:
        raise ValueError("background must be non-negative")

    decay = monoexponential_decay(
        time=time,
        amplitude=1.0,
        lifetime=lifetime,
        background=0.0,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=temporal_shift,
    )

    convolved = convolve_decay_with_irf(
        time=time,
        decay=decay,
        irf=shifted_irf,
    )

    expected_counts = amplitude * convolved + background

    return expected_counts


# Poisson (reduced) NLL
def poisson_negative_log_likelihood(
    observed: np.ndarray,
    expected: np.ndarray,
) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape.")

    if not np.all(np.isfinite(observed)):
        raise ValueError("observed must contain only finite values.")

    if not np.all(np.isfinite(expected)):
        raise ValueError("expected must contain only finite values.")

    if np.any(observed < 0):
        raise ValueError("observed counts must be non-negative.")

    if np.any(expected < 0):
        raise ValueError("expected counts must be non-negative.")

    zero_expected_with_counts = (expected == 0.0) & (observed > 0.0)

    if np.any(zero_expected_with_counts):
        return float("inf")

    positive_expected = expected > 0.0

    terms = np.zeros_like(expected, dtype=float)

    terms[positive_expected] = (
        expected[positive_expected]
        - observed[positive_expected]
        * np.log(expected[positive_expected])
    )

    return float(np.sum(terms))


def fit_monoexponential_reconvolution(
    time: NDArray[np.float64],
    counts: NDArray[np.float64] | NDArray[np.int64],
    irf: NDArray[np.float64],
    initial_guess: tuple[float, float, float, float],
    temporal_shift_bounds: tuple[float, float] | None = None,
    objective: str = "least_squares",
) -> ReconvolutionFitResult:
    """Fit a mono-exponential decay using IRF reconvolution.

    Parameters
    ----------
    time:
        One-dimensional array containing time-bin positions.
    counts:
        One-dimensional array containing measured photon counts.
    irf:
        Instrument response function evaluated on the same time grid.
        The IRF should already be normalized.
    initial_guess:
        Initial guesses for amplitude, lifetime, background,
        and temporal shift.
    temporal_shift_bounds:
        Lower and upper bounds for the temporal shift. If omitted,
        the shift is limited to 10% of the measurement time span
        in either direction.
    objective:
        objective function for the fitter - 'least_squares' or 'poisson'

    Returns
    -------
    ReconvolutionFitResult
        Fitted physical parameters, fitted curve, and optimizer
        success status.
    """
    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array")

    if counts.ndim != 1:
        raise ValueError("counts must be a one-dimensional array")

    if irf.ndim != 1:
        raise ValueError("irf must be a one-dimensional array")

    if not (
        time.shape == counts.shape == irf.shape
    ):
        raise ValueError(
            "time, counts, and irf must have the same shape"
        )

    if time.size < 5:
        raise ValueError(
            "at least five data points are required"
        )

    if not np.all(np.isfinite(time)):
        raise ValueError(
            "time must contain only finite values"
        )

    if not np.all(np.isfinite(counts)):
        raise ValueError(
            "counts must contain only finite values"
        )

    if np.any(counts < 0.0):
        raise ValueError(
            "counts must contain only non-negative values"
        )

    time_differences = np.diff(time)

    if np.any(time_differences <= 0.0):
        raise ValueError(
            "time must be strictly increasing"
        )

    if not np.allclose(
        time_differences,
        time_differences[0],
    ):
        raise ValueError(
            "time must be uniformly spaced"
        )

    if not np.all(np.isfinite(irf)):
        raise ValueError(
            "irf must contain only finite values"
        )

    if np.any(irf < 0.0):
        raise ValueError(
            "irf must contain only non-negative values"
        )

    initial_parameters = np.asarray(
        initial_guess,
        dtype=np.float64,
    )

    if initial_parameters.shape != (4,):
        raise ValueError(
            "initial_guess must contain four values"
        )

    if not np.all(np.isfinite(initial_parameters)):
        raise ValueError(
            "initial_guess must contain only finite values"
        )

    (
        initial_amplitude,
        initial_lifetime,
        initial_background,
        initial_temporal_shift,
    ) = initial_parameters

    if initial_amplitude < 0.0:
        raise ValueError(
            "initial amplitude must be non-negative"
        )

    if initial_lifetime <= 0.0:
        raise ValueError(
            "initial lifetime must be positive"
        )

    if initial_background < 0.0:
        raise ValueError(
            "initial background must be non-negative"
        )

    if temporal_shift_bounds is None:
        time_span = time[-1] - time[0]
        shift_limit = 0.1 * time_span

        temporal_shift_bounds = (
            -shift_limit,
            shift_limit,
        )

    shift_lower, shift_upper = temporal_shift_bounds

    if not (
        np.isfinite(shift_lower)
        and np.isfinite(shift_upper)
    ):
        raise ValueError(
            "temporal shift bounds must be finite"
        )

    if shift_lower >= shift_upper:
        raise ValueError(
            "lower temporal shift bound must be smaller "
            "than upper temporal shift bound"
        )

    if not (
        shift_lower
        <= initial_temporal_shift
        <= shift_upper
    ):
        raise ValueError(
            "initial temporal shift must lie within "
            "temporal shift bounds"
        )

    if objective not in {
        "least_squares",
        "poisson",
    }:
        raise ValueError(
            "objective must be either "
            "'least_squares' or 'poisson'"
        )

    if objective == "poisson":
        background_lower_bound = 1e-12
    else:
        background_lower_bound = 0.0

    lower_bounds = np.array(
        [
            0.0,
            1e-12,
            background_lower_bound,
            shift_lower,
        ],
        dtype=np.float64,
    )

    upper_bounds = np.array(
        [
            np.inf,
            np.inf,
            np.inf,
            shift_upper,
        ],
        dtype=np.float64,
    )

    counts_float = counts.astype(
        np.float64,
        copy=False,
    )

    def residual_function(
        parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        (
            amplitude,
            lifetime,
            background,
            temporal_shift,
        ) = parameters

        fitted = _reconvolution_model(
            time=time,
            irf=irf,
            amplitude=amplitude,
            lifetime=lifetime,
            background=background,
            temporal_shift=temporal_shift,
        )

        return counts_float - fitted

    def poisson_objective(
            parameters: NDArray[np.float64],
    ) -> float:
        (
            amplitude,
            lifetime,
            background,
            temporal_shift,
        ) = parameters

        expected_counts = _reconvolution_model(
            time=time,
            irf=irf,
            amplitude=amplitude,
            lifetime=lifetime,
            background=background,
            temporal_shift=temporal_shift,
        )

        return poisson_negative_log_likelihood(
            observed=counts_float,
            expected=expected_counts,
        )

    if objective == "least_squares":
        optimization_result = least_squares(
            fun=residual_function,
            x0=initial_parameters,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
        )

    elif objective == "poisson":
        bounds = list(
            zip(
                lower_bounds,
                upper_bounds,
            )
        )

        optimization_result = minimize(
            fun=poisson_objective,
            x0=initial_parameters,
            method="L-BFGS-B",
            bounds=bounds,
        )
        # TODO: L-BFGS-B supports parameter bounds directly.
        #       We do not introduce log-parameter transformations or custom gradients yet.
        #       Those may become worthwhile later if optimization conditioning becomes a problem.
    (
        amplitude,
        lifetime,
        background,
        temporal_shift,
    ) = optimization_result.x

    fitted_curve = _reconvolution_model(
        time=time,
        irf=irf,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
        temporal_shift=temporal_shift,
    )

    return ReconvolutionFitResult(
        amplitude=float(amplitude),
        lifetime=float(lifetime),
        background=float(background),
        temporal_shift=float(temporal_shift),
        fitted_curve=fitted_curve,
        success=bool(optimization_result.success),
    )
