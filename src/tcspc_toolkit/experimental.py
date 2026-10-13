"""Compatibility imports for :mod:`tcspc_toolkit.measurement.adapters`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.measurement.adapters import (
    ReferenceLifetimeEvaluation,
    fit_experimental_reconvolution,
    measurement_to_feature_table,
    measurement_to_histogram_batch,
    evaluate_estimate_against_reference,
)

__all__ = [
    "ReferenceLifetimeEvaluation",
    "fit_experimental_reconvolution",
    "measurement_to_feature_table",
    "measurement_to_histogram_batch",
    "evaluate_estimate_against_reference",
]
