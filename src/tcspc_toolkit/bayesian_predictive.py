"""Compatibility exports for :mod:`tcspc_toolkit.estimation.bayesian.predictive`.

Historical flat imports remain supported through at least Issue #12.
"""

from tcspc_toolkit.estimation.bayesian.predictive import (
    PHYSICAL_PARAMETER_ORDER,
    BayesianPredictiveBand,
    summarize_predictive_band,
    BayesianPredictiveDiscrepancy,
    BayesianPredictiveWindowDiscrepancy,
    BayesianPosteriorPredictiveDiagnostics,
    calculate_bayesian_posterior_predictive_diagnostics,
    BayesianPriorPredictiveResult,
    BayesianPosteriorPredictiveResult,
    sample_bayesian_prior_predictive,
    sample_bayesian_posterior_predictive,
)

__all__ = [
    "PHYSICAL_PARAMETER_ORDER",
    "BayesianPredictiveBand",
    "summarize_predictive_band",
    "BayesianPredictiveDiscrepancy",
    "BayesianPredictiveWindowDiscrepancy",
    "BayesianPosteriorPredictiveDiagnostics",
    "calculate_bayesian_posterior_predictive_diagnostics",
    "BayesianPriorPredictiveResult",
    "BayesianPosteriorPredictiveResult",
    "sample_bayesian_prior_predictive",
    "sample_bayesian_posterior_predictive",
]
