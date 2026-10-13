"""Compatibility imports for :mod:`tcspc_toolkit.evaluation.plots`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.evaluation.plots import (
    plot_true_vs_predicted,
    plot_error_histogram,
    plot_error_vs_condition,
    plot_paired_absolute_error_comparison,
)

__all__ = [
    "plot_true_vs_predicted",
    "plot_error_histogram",
    "plot_error_vs_condition",
    "plot_paired_absolute_error_comparison",
]
