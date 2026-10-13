"""Compatibility imports for :mod:`tcspc_toolkit.evaluation.conditional`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.evaluation.conditional import (
    build_prediction_diagnostics,
    build_ml_prediction_diagnostics,
    build_classical_prediction_diagnostics,
    summarize_conditional_performance,
    assign_numeric_regimes,
    add_benchmark_regimes,
    summarize_standard_regimes,
)

__all__ = [
    "build_prediction_diagnostics",
    "build_ml_prediction_diagnostics",
    "build_classical_prediction_diagnostics",
    "summarize_conditional_performance",
    "assign_numeric_regimes",
    "add_benchmark_regimes",
    "summarize_standard_regimes",
]
