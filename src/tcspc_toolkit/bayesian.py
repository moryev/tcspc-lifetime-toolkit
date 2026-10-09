"""Compatibility exports for :mod:`tcspc_toolkit.estimation.bayesian.posterior`.

Historical flat imports remain supported through at least Issue #12.
"""

from tcspc_toolkit.estimation.bayesian.posterior import (
    GammaPrior,
    LogNormalPrior,
    BoundedUniformPrior,
    BayesianPriorConfig,
    BayesianPhysicalParameters,
    IRFModelRelation,
    BayesianModelContext,
    sampling_to_physical_parameters,
    physical_to_sampling_coordinates,
    log_transformation_jacobian,
    log_prior,
    log_likelihood,
    log_posterior,
    fit_bayesian_monoexponential_reconvolution,
)

__all__ = [
    "GammaPrior",
    "LogNormalPrior",
    "BoundedUniformPrior",
    "BayesianPriorConfig",
    "BayesianPhysicalParameters",
    "IRFModelRelation",
    "BayesianModelContext",
    "sampling_to_physical_parameters",
    "physical_to_sampling_coordinates",
    "log_transformation_jacobian",
    "log_prior",
    "log_likelihood",
    "log_posterior",
    "fit_bayesian_monoexponential_reconvolution",
]
