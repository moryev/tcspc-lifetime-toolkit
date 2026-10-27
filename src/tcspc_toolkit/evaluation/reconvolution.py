"""Descriptive paired Poisson statistics on an identical observation window.

Nominal parameter counts stay four and six at singular boundaries. Differences
are always bi minus mono; they are neither a model selector nor an LRT p-value.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from tcspc_toolkit.evaluation.diagnostics import (
    PoissonFitStatistics,
    calculate_poisson_fit_statistics,
)


@dataclass(frozen=True)
class ReconvolutionStatisticsComparison:
    mono: PoissonFitStatistics | None
    bi: PoissonFitStatistics | None
    delta_log_likelihood: float | None
    delta_deviance: float | None
    delta_aic: float | None
    delta_bic_bins: float | None
    unavailable_reason: str | None


def compare_reconvolution_statistics(
    observations: ArrayLike,
    mono_expected: ArrayLike | None,
    bi_expected: ArrayLike | None,
) -> ReconvolutionStatisticsComparison:
    """Recompute full likelihood and bin-based BIC using the PR-1 primitives.

    ``None`` explicitly denotes an unavailable fitted curve. Invalid supplied
    arrays raise; finite curves from numerically rejected fits remain usable for
    descriptive statistics, but callers must retain the fit's validity flags.
    """
    # Validate raw counts even if neither fit has an available curve.
    observed = np.asarray(observations)
    calculate_poisson_fit_statistics(observed, np.zeros(observed.shape), n_parameters=0)
    mono = (
        None
        if mono_expected is None
        else calculate_poisson_fit_statistics(observed, mono_expected, n_parameters=4)
    )
    bi = (
        None
        if bi_expected is None
        else calculate_poisson_fit_statistics(observed, bi_expected, n_parameters=6)
    )
    reason = None
    differences = (None,) * 4
    if mono is None or bi is None:
        reason = "one or both fitted expectations are unavailable"
    else:
        left = np.array([mono.log_likelihood, mono.deviance, mono.aic, mono.bic_bins])
        right = np.array([bi.log_likelihood, bi.deviance, bi.aic, bi.bic_bins])
        if np.all(np.isfinite(left)) and np.all(np.isfinite(right)):
            differences = tuple(map(float, right - left))
        else:
            reason = "one or both likelihoods are non-finite"
    return ReconvolutionStatisticsComparison(mono, bi, *differences, reason)
