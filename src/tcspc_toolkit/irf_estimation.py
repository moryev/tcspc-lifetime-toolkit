"""Compatibility imports for :mod:`tcspc_toolkit.irf.estimation`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.irf.estimation import (
    TimeWindow,
    BinRange,
    LeadingEdgeIRFDiagnostics,
    LeadingEdgeIRFResult,
    estimate_irf_from_leading_edge,
)

__all__ = [
    "TimeWindow",
    "BinRange",
    "LeadingEdgeIRFDiagnostics",
    "LeadingEdgeIRFResult",
    "estimate_irf_from_leading_edge",
]
