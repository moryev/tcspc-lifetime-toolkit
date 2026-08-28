import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.fitting import LifetimeFitResult


def generate_fitted_signal(
    time: NDArray[np.float64],
    fit_result: LifetimeFitResult,
) -> NDArray[np.float64]:
    """Generate an unconvolved mono-exponential signal from fitted parameters.

    Parameters
    ----------
    time:
        One-dimensional array containing time-bin positions.
    fit_result:
        Result returned by the mono-exponential fitting function.

    Returns
    -------
    NDArray[np.float64]
        Model-predicted counts for every time bin.
    """
    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array")

    return monoexponential_decay(
        time=time,
        amplitude=fit_result.amplitude,
        lifetime=fit_result.lifetime,
        background=fit_result.background,
    )


def calculate_residuals(
    observed: NDArray[np.float64] | NDArray[np.int64],
    fitted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate raw residuals as observed minus fitted values.

    Parameters
    ----------
    observed:
        Measured counts.
    fitted:
        Counts predicted by the fitted model.

    Returns
    -------
    NDArray[np.float64]
        Residual for every time bin.
    """
    if observed.ndim != 1:
        raise ValueError("observed must be a one-dimensional array")

    if fitted.ndim != 1:
        raise ValueError("fitted must be a one-dimensional array")

    if observed.shape != fitted.shape:
        raise ValueError("observed and fitted must have the same shape")

    return observed.astype(np.float64) - fitted


def calculate_reduced_residuals(
    observed: NDArray[np.float64] | NDArray[np.int64],
    fitted: NDArray[np.float64],
    minimum_expected_count: float = 1.0,
    # TODO: rename it to calculate_pearson_residuals() during next APIs refactoring?
) -> NDArray[np.float64]:
    """Calculate Poisson-scaled Pearson residuals.

    The residuals are calculated as:

        (observed - fitted) / sqrt(fitted)

    A minimum fitted count is used to avoid division by zero.

    Parameters
    ----------
    observed:
        Measured photon counts.
    fitted:
        Counts predicted by the fitted model.
    minimum_expected_count:
        Lower limit applied to fitted counts in the denominator.

    Returns
    -------
    NDArray[np.float64]
        Dimensionless residuals scaled by approximate Poisson uncertainty.
    """
    if minimum_expected_count <= 0:
        raise ValueError("minimum_expected_count must be positive")

    raw_residuals = calculate_residuals(
        observed=observed,
        fitted=fitted,
    )

    safe_fitted = np.maximum(
        fitted,
        minimum_expected_count,
    )

    poisson_standard_deviation = np.sqrt(safe_fitted)

    return raw_residuals / poisson_standard_deviation


def calculate_absolute_lifetime_error(
    true_lifetime: float,
    estimated_lifetime: float,
) -> float:
    """Calculate the absolute lifetime estimation error."""
    if true_lifetime <= 0:
        raise ValueError("true_lifetime must be positive")

    if estimated_lifetime <= 0:
        raise ValueError("estimated_lifetime must be positive")

    return abs(estimated_lifetime - true_lifetime)


def calculate_relative_lifetime_error(
    true_lifetime: float,
    estimated_lifetime: float,
) -> float:
    """Calculate the relative lifetime error as a fraction."""
    absolute_error = calculate_absolute_lifetime_error(
        true_lifetime=true_lifetime,
        estimated_lifetime=estimated_lifetime,
    )

    return absolute_error / true_lifetime


def calculate_poisson_deviance_residuals(
    observed: NDArray[np.float64] | NDArray[np.int64],
    expected: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate signed Poisson deviance residuals.

    Parameters
    ----------
    observed:
        Measured photon counts.
    expected:
        Expected photon counts predicted by the fitted model.

    Returns
    -------
    NDArray[np.float64]
        Signed Poisson deviance residual for every time bin.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, have different shapes,
        contain negative observed counts, or contain non-positive expected
        counts.
    """
    if observed.ndim != 1:
        raise ValueError("observed must be a one-dimensional array")

    if expected.ndim != 1:
        raise ValueError("expected must be a one-dimensional array")

    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape")

    if np.any(observed < 0):
        raise ValueError("observed counts must be non-negative")

    if np.any(expected <= 0):
        raise ValueError("expected counts must be positive")

    observed_float = observed.astype(np.float64)

    deviance = np.empty_like(expected, dtype=np.float64)

    positive_counts = observed_float > 0

    deviance[positive_counts] = 2.0 * (
        observed_float[positive_counts]
        * np.log(
            observed_float[positive_counts]
            / expected[positive_counts]
        )
        - (
            observed_float[positive_counts]
            - expected[positive_counts]
        )
    )

    deviance[~positive_counts] = (
        2.0 * expected[~positive_counts]
    )

    deviance = np.maximum(deviance, 0.0)

    residual_sign = np.sign(
        observed_float - expected
    )

    return residual_sign * np.sqrt(deviance)


def calculate_lifetime_errors(
    true_lifetimes: NDArray[np.float64],
    estimated_lifetimes: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Calculate signed, absolute, and relative lifetime errors."""

    if true_lifetimes.ndim != 1:
        raise ValueError(
            "true_lifetimes must be one-dimensional"
        )

    if estimated_lifetimes.ndim != 1:
        raise ValueError(
            "estimated_lifetimes must be one-dimensional"
        )

    if true_lifetimes.shape != estimated_lifetimes.shape:
        raise ValueError(
            "true_lifetimes and estimated_lifetimes "
            "must have the same shape"
        )

    if not np.all(
        np.isfinite(true_lifetimes)
    ):
        raise ValueError(
            "true_lifetimes must contain only finite values"
        )

    if np.any(
        true_lifetimes <= 0.0
    ):
        raise ValueError(
            "true_lifetimes must be positive"
        )

    signed_error = (
        estimated_lifetimes
        - true_lifetimes
    )

    absolute_error = np.abs(
        signed_error
    )

    relative_error = (
        absolute_error
        / true_lifetimes
    )

    return (
        signed_error,
        absolute_error,
        relative_error,
    )
