"""Compatibility exports for canonical classical fitting."""

from tcspc_toolkit.estimation.fitting import (
    LifetimeFitResult,
    ReconvolutionFitResult,
    fit_monoexponential_decay,
    poisson_negative_log_likelihood,
    fit_monoexponential_reconvolution,
)

__all__ = [
    "LifetimeFitResult",
    "ReconvolutionFitResult",
    "fit_monoexponential_decay",
    "poisson_negative_log_likelihood",
    "fit_monoexponential_reconvolution",
]
