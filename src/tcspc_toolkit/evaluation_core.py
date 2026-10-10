"""Compatibility imports for :mod:`tcspc_toolkit.evaluation.core`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.evaluation.core import (
    __all__,
    build_point_evaluation,
    combine_point_evaluations,
    calculate_reference_errors,
    summarize_point_errors,
    calculate_degradation_ratio,
)
