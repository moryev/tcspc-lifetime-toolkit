"""Compatibility imports for :mod:`tcspc_toolkit.estimation.baselines`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.estimation.baselines import (
    FloatArray,
    estimate_lifetime_from_mean_arrival,
    predict_constant_mean_baseline,
)

__all__ = [
    "FloatArray",
    "estimate_lifetime_from_mean_arrival",
    "predict_constant_mean_baseline",
]
