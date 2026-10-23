from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from tcspc_toolkit.physics.models import monoexponential_decay
from tcspc_toolkit.estimation.fitting import (
    LifetimeFitResult,
    poisson_negative_log_likelihood,
)


@dataclass(frozen=True)
class PoissonFitStatistics:
    """Descriptive statistics conditional on independent raw Poisson counts.

    ``negative_log_likelihood`` omits log-factorials, matching the historical
    fitting objective. ``centered_negative_log_likelihood`` subtracts its
    saturated-model value and equals half the deviance. ``log_likelihood``
    includes log-factorials. The returned residual array is read-only.

    ``n_parameters`` is the caller-declared nominal number of fitted parameters
    (normally 4 for mono and 6 for bi reconvolution). It is not an estimate of
    identifiable rank. ``bic_bins`` uses the number of observed bins, including
    zero-count bins, not the number of photons. These criteria require the same
    observations/window/likelihood to compare fits and do not establish model
    validity, especially at singular or boundary solutions.
    """

    n_bins: int
    n_parameters: int
    negative_log_likelihood: float
    centered_negative_log_likelihood: float
    log_likelihood: float
    deviance: float
    deviance_residuals: NDArray[np.float64]
    aic: float
    bic_bins: float


def _validate_poisson_pair(
    observed: ArrayLike, expected: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if observed.ndim != 1 or expected.ndim != 1:
        raise ValueError("observed and expected must be one-dimensional")
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape")
    if observed.size == 0:
        raise ValueError("observed and expected must not be empty")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected)):
        raise ValueError("observed and expected must contain only finite values")
    if np.any(observed < 0.0) or np.any(expected < 0.0):
        raise ValueError("observed and expected must be non-negative")
    return observed, expected


def _poisson_centered_terms(
    observed: NDArray[np.float64], expected: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate mu - y + y*log(y/mu), including its exact zero limits."""
    terms = expected.copy()  # y = 0: term = mu, including (0, 0).
    impossible = (observed > 0.0) & (expected == 0.0)
    terms[impossible] = np.inf
    positive = (observed > 0.0) & (expected > 0.0)
    y, mu = observed[positive], expected[positive]
    difference = y - mu
    close = np.abs(difference) <= 0.01 * mu
    values = np.empty_like(y)

    # For u = (y-mu)/mu, (1+u)*log1p(u)-u has leading term u**2/2.
    # The degree-10 series avoids cancellation; at |u| <= .01 the first
    # omitted term is less than 2e-20 relative to that leading term.
    u = difference[close] / mu[close]
    polynomial = np.full_like(u, 1.0 / 90.0)
    for power in range(9, 1, -1):
        polynomial = (-1.0) ** power / (power * (power - 1)) + u * polynomial
    values[close] = difference[close] * u * polynomial

    # log1p is accurate for moderate ratios. For extreme ratios, subtract
    # logarithms instead of forming y/mu, which can overflow or underflow.
    moderate = (~close) & (np.abs(difference) <= 0.5 * mu)
    u = difference[moderate] / mu[moderate]
    values[moderate] = y[moderate] * np.log1p(u) - difference[moderate]
    smaller = y < 0.5 * mu
    far = ~(close | moderate | smaller)
    log_ratio = np.log(y[far]) - np.log(mu[far])
    # Factoring y prevents an overflowing y*log_ratio intermediate when the
    # final centered term is still representable. Genuine overflow is inf.
    with np.errstate(over="ignore"):
        values[far] = y[far] * (log_ratio - difference[far] / y[far])
    # For y << mu, difference/y can overflow. Each term below is safe.
    values[smaller] = (
        mu[smaller]
        - y[smaller]
        + y[smaller] * (np.log(y[smaller]) - np.log(mu[smaller]))
    )
    terms[positive] = np.maximum(values, 0.0)
    return terms


def calculate_poisson_deviance(observed: ArrayLike, expected: ArrayLike) -> float:
    """Return stable Poisson deviance, twice the centered negative log likelihood.

    Inputs must be finite, non-negative, nonempty matching one-dimensional
    arrays. No expectation floor is used: (y, mu) = (0, 0) contributes zero,
    (0, mu) contributes 2*mu, and y > 0 with mu = 0 gives infinity.

    Non-integer targets are allowed for deterministic expected-count
    projections. This mathematical extension does not make processed,
    normalized, or background-subtracted observations Poisson counts.
    """
    observed, expected = _validate_poisson_pair(observed, expected)
    with np.errstate(over="ignore"):
        return float(2.0 * np.sum(_poisson_centered_terms(observed, expected)))


def _poisson_saturated_log_likelihood(observed: NDArray[np.float64]) -> float:
    """Include the log-factorial without high-count cancellation."""
    y = observed[observed > 0.0]
    small = y < 16.0
    values = np.empty_like(y)
    values[small] = y[small] * np.log(y[small]) - y[small] - gammaln(y[small] + 1.0)
    # Stirling's log-factorial remainder through y**-9. For y >= 16 the
    # omitted term is below 1.1e-16 in absolute log-likelihood units.
    inverse = 1.0 / y[~small]
    square = inverse * inverse
    correction = inverse * (
        1.0 / 12.0
        + square
        * (
            -1.0 / 360.0
            + square * (1.0 / 1260.0 + square * (-1.0 / 1680.0 + square / 1188.0))
        )
    )
    values[~small] = -0.5 * (np.log(2.0 * np.pi) + np.log(y[~small])) - correction
    return float(np.sum(values))


def calculate_poisson_fit_statistics(
    observed: ArrayLike, expected: ArrayLike, *, n_parameters: int
) -> PoissonFitStatistics:
    """Summarize a fitted expectation for independent, raw Poisson counts.

    The caller asserts raw-count semantics: values alone cannot detect prior
    processing. Canonical measurements must pass ``require_raw_counts()``
    before supplying their counts. Fractional observations are rejected here;
    use :func:`calculate_poisson_deviance` for deterministic projections.
    Counts must be below 2**53 so conversion to float64 preserves integers.

    All observed bins, including zeros, count toward ``bic_bins``. Parameter
    count is a non-negative integer supplied by the caller; it is not reduced
    when a fitted component vanishes. No goodness-of-fit p-value, effective
    degrees of freedom, or automatic model choice is returned.
    """
    observed, expected = _validate_poisson_pair(observed, expected)
    if np.any(observed >= 2**53):
        raise ValueError("raw counts must be below 2**53 for float64 statistics")
    if np.any(observed != np.rint(observed)):
        raise ValueError("raw observed counts must be integer-valued")
    if (
        isinstance(n_parameters, (bool, np.bool_))
        or not isinstance(n_parameters, (int, np.integer))
        or n_parameters < 0
    ):
        raise ValueError("n_parameters must be a non-negative integer")
    n_parameters = int(n_parameters)

    terms = _poisson_centered_terms(observed, expected)
    with np.errstate(over="ignore"):
        centered_nll = float(np.sum(terms))
        deviance = 2.0 * centered_nll
        residuals = np.sign(observed - expected) * (np.sqrt(terms) * np.sqrt(2.0))
    residuals.setflags(write=False)
    log_likelihood = _poisson_saturated_log_likelihood(observed) - centered_nll
    return PoissonFitStatistics(
        n_bins=int(observed.size),
        n_parameters=n_parameters,
        negative_log_likelihood=poisson_negative_log_likelihood(observed, expected),
        centered_negative_log_likelihood=centered_nll,
        log_likelihood=log_likelihood,
        deviance=deviance,
        deviance_residuals=residuals,
        aic=float(2 * n_parameters - 2.0 * log_likelihood),
        bic_bins=float(n_parameters * np.log(observed.size) - 2.0 * log_likelihood),
    )


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
        * np.log(observed_float[positive_counts] / expected[positive_counts])
        - (observed_float[positive_counts] - expected[positive_counts])
    )

    deviance[~positive_counts] = 2.0 * expected[~positive_counts]

    deviance = np.maximum(deviance, 0.0)

    residual_sign = np.sign(observed_float - expected)

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
        raise ValueError("true_lifetimes must be one-dimensional")

    if estimated_lifetimes.ndim != 1:
        raise ValueError("estimated_lifetimes must be one-dimensional")

    if true_lifetimes.shape != estimated_lifetimes.shape:
        raise ValueError(
            "true_lifetimes and estimated_lifetimes must have the same shape"
        )

    if not np.all(np.isfinite(true_lifetimes)):
        raise ValueError("true_lifetimes must contain only finite values")

    if np.any(true_lifetimes <= 0.0):
        raise ValueError("true_lifetimes must be positive")

    signed_error = estimated_lifetimes - true_lifetimes

    absolute_error = np.abs(signed_error)

    relative_error = absolute_error / true_lifetimes

    return (
        signed_error,
        absolute_error,
        relative_error,
    )
