"""Historical diagnostics API; canonical implementation: evaluation.diagnostics.

These direct compatibility exports remain supported through at least Issue #12.
This facade deliberately does not aggregate factual evaluation or reporting APIs.
"""

from .diagnostics import (
    generate_fitted_signal,
    calculate_residuals,
    calculate_reduced_residuals,
    calculate_absolute_lifetime_error,
    calculate_relative_lifetime_error,
    calculate_poisson_deviance_residuals,
    calculate_lifetime_errors,
)

__all__ = [
    "generate_fitted_signal",
    "calculate_residuals",
    "calculate_reduced_residuals",
    "calculate_absolute_lifetime_error",
    "calculate_relative_lifetime_error",
    "calculate_poisson_deviance_residuals",
    "calculate_lifetime_errors",
]
