from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit, least_squares, minimize

from tcspc_toolkit.forward_model import (
    monoexponential_reconvolution_expected_counts,
)
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


@dataclass(frozen=True)
class ReconvolutionFitResult:
    """Results of a mono-exponential reconvolution fit."""

    amplitude: float
    lifetime: float
    background: float
    temporal_shift: float

    fitted_curve: NDArray[np.float64]
    success: bool
    optimizer_reported_success: bool | None = None
    numerical_validation_passed: bool | None = None
    max_coordinate_descent_nll: float = np.nan
    recovery_attempted: bool = False
    optimizer_status: int | None = None
    optimizer_message: str | None = None
    optimizer_nfev: int | None = None
    optimizer_njev: int | None = None


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
    """Compatibility wrapper for the shared deterministic reconvolution model."""
    return monoexponential_reconvolution_expected_counts(
        time=time,
        irf=irf,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
        temporal_shift=temporal_shift,
    )


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


def _poisson_coordinate_descent_check(
    objective: Callable[[NDArray[np.float64]], float],
    scaled_parameters: NDArray[np.float64],
    scaled_bounds: list[tuple[float, float]],
) -> tuple[bool, float]:
    """Reject grossly improvable Poisson fits without using generating truth.

    The fixed 1e-3 dimensionless coordinate probes are deliberately a modest
    local check, not a proof of global optimality. A reduction above 0.01 NLL
    units (0.02 deviance units) is larger than the numerical tolerance used
    here; probes outside the physical bounds are skipped.
    """
    if not np.all(np.isfinite(scaled_parameters)):
        return False, np.nan
    try:
        objective_at_fit = float(objective(scaled_parameters))
    except ValueError:
        return False, np.nan
    if not np.isfinite(objective_at_fit):
        return False, np.nan

    maximum_descent = 0.0
    for index, (lower, upper) in enumerate(scaled_bounds):
        if not lower <= scaled_parameters[index] <= upper:
            return False, np.nan
        for direction in (-1.0, 1.0):
            trial = scaled_parameters.copy()
            trial[index] += direction * 1e-3
            if not lower <= trial[index] <= upper:
                continue
            try:
                trial_objective = float(objective(trial))
            except ValueError:
                continue
            if np.isfinite(trial_objective):
                maximum_descent = max(
                    maximum_descent, objective_at_fit - trial_objective
                )
    return maximum_descent <= 0.01, maximum_descent


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
        Fitted physical parameters, fitted curve, and validated success status.
        For Poisson fits, optimizer termination alone is insufficient: a
        feasible local coordinate probe must also pass.
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

        optimal_parameters = (
            optimization_result.x
        )
        numerical_validation_passed = None
        maximum_descent = np.nan
        recovery_attempted = False
        total_nfev = getattr(optimization_result, "nfev", None)
        total_njev = getattr(optimization_result, "njev", None)

    elif objective == "poisson":
        # L-BFGS-B is sensitive to strongly different parameter
        # magnitudes. In reconvolution fitting, amplitude may be
        # O(10^3-10^6), while lifetime, background, and temporal
        # shift are typically O(10^-2-10^1).
        #
        # Optimize dimensionless scaled parameters internally while
        # preserving the physical parameterization and public API.

        bin_width = float(
            time_differences[0]
        )

        shift_scale = max(
            abs(
                initial_temporal_shift
            ),
            0.1
            * (
                    shift_upper
                    - shift_lower
            ),
            bin_width,
        )

        parameter_scales = np.asarray(
            [
                max(
                    abs(initial_amplitude),
                    1.0,
                ),
                max(
                    abs(initial_lifetime),
                    bin_width,
                ),
                max(
                    abs(initial_background),
                    1.0,
                ),
                shift_scale,
            ],
            dtype=np.float64,
        )

        poisson_initial_parameters = (
            initial_parameters.copy()
        )

        poisson_initial_parameters[2] = max(
            poisson_initial_parameters[2],
            background_lower_bound,
        )

        scaled_initial_parameters = (
                poisson_initial_parameters
                / parameter_scales
        )

        scaled_lower_bounds = (
                lower_bounds
                / parameter_scales
        )

        scaled_upper_bounds = (
                upper_bounds
                / parameter_scales
        )

        def scaled_poisson_objective(
                scaled_parameters: NDArray[np.float64],
        ) -> float:
            physical_parameters = (
                    scaled_parameters
                    * parameter_scales
            )

            return poisson_objective(
                physical_parameters
            )

        scaled_bounds = list(
            zip(
                scaled_lower_bounds,
                scaled_upper_bounds,
            )
        )

        optimization_result = minimize(
            fun=scaled_poisson_objective,
            x0=scaled_initial_parameters,
            method="L-BFGS-B",
            bounds=scaled_bounds,
        )

        numerical_validation_passed, maximum_descent = (
            _poisson_coordinate_descent_check(
                scaled_poisson_objective,
                optimization_result.x,
                scaled_bounds,
            )
        )
        recovery_attempted = False
        total_nfev = getattr(optimization_result, "nfev", None)
        total_njev = getattr(optimization_result, "njev", None)

        # A relative-objective termination can occur while the local
        # Poisson objective is still plainly descending. Continue only
        # those suspicious, otherwise successful fits from their own
        # endpoint, with a tighter relative tolerance and a finite budget.
        if (
            optimization_result.success
            and not numerical_validation_passed
            and np.all(np.isfinite(optimization_result.x))
            and np.isfinite(optimization_result.fun)
        ):
            recovery_attempted = True
            continuation = minimize(
                fun=scaled_poisson_objective,
                x0=optimization_result.x.copy(),
                method="L-BFGS-B",
                bounds=scaled_bounds,
                options={"ftol": 1e-12, "gtol": 1e-6, "maxiter": 1000},
            )
            if total_nfev is not None:
                total_nfev += getattr(continuation, "nfev", 0)
            if total_njev is not None:
                total_njev += getattr(continuation, "njev", 0)
            if (
                np.isfinite(continuation.fun)
                and np.isfinite(optimization_result.fun)
                and continuation.fun <= optimization_result.fun
            ):
                optimization_result = continuation
                numerical_validation_passed, maximum_descent = (
                    _poisson_coordinate_descent_check(
                        scaled_poisson_objective,
                        optimization_result.x,
                        scaled_bounds,
                    )
                )

        optimal_parameters = (
                optimization_result.x
                * parameter_scales
        )

    (
        amplitude,
        lifetime,
        background,
        temporal_shift,
    ) = optimal_parameters

    if np.all(np.isfinite(optimal_parameters)):
        fitted_curve = _reconvolution_model(
            time=time,
            irf=irf,
            amplitude=amplitude,
            lifetime=lifetime,
            background=background,
            temporal_shift=temporal_shift,
        )
    else:
        fitted_curve = np.full_like(time, np.nan, dtype=np.float64)

    optimizer_reported_success = bool(optimization_result.success)
    success = optimizer_reported_success and (
        numerical_validation_passed is not False
    )

    return ReconvolutionFitResult(
        amplitude=float(amplitude),
        lifetime=float(lifetime),
        background=float(background),
        temporal_shift=float(temporal_shift),
        fitted_curve=fitted_curve,
        success=success,
        optimizer_reported_success=optimizer_reported_success,
        numerical_validation_passed=numerical_validation_passed,
        max_coordinate_descent_nll=float(maximum_descent),
        recovery_attempted=recovery_attempted,
        optimizer_status=int(optimization_result.status),
        optimizer_message=str(optimization_result.message),
        optimizer_nfev=total_nfev,
        optimizer_njev=total_njev,
    )
