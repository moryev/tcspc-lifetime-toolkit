"""Compatibility exports for :mod:`tcspc_toolkit.estimation.bayesian.sampling`.

Historical flat imports remain supported through at least Issue #12.
"""

from tcspc_toolkit.estimation.bayesian.sampling import (
    PHYSICAL_PARAMETER_NAMES,
    SAMPLING_COORDINATE_NAMES,
    PARAMETER_UNITS,
    BayesianSamplingConfig,
    BayesianSamplingStatus,
    BayesianPosteriorSamples,
    BayesianParameterSummary,
    BayesianSamplingDiagnostics,
    BayesianRuntime,
    BayesianModelContextSummary,
    BayesianReconvolutionResult,
    BayesianInferenceRun,
    BayesianInitializationError,
    fit_bayesian_monoexponential_reconvolution,
)

__all__ = [
    "PHYSICAL_PARAMETER_NAMES",
    "SAMPLING_COORDINATE_NAMES",
    "PARAMETER_UNITS",
    "BayesianSamplingConfig",
    "BayesianSamplingStatus",
    "BayesianPosteriorSamples",
    "BayesianParameterSummary",
    "BayesianSamplingDiagnostics",
    "BayesianRuntime",
    "BayesianModelContextSummary",
    "BayesianReconvolutionResult",
    "BayesianInferenceRun",
    "BayesianInitializationError",
    "fit_bayesian_monoexponential_reconvolution",
]
