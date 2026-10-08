"""Compatibility imports for :mod:`tcspc_toolkit.irf.preparation`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.irf.preparation import (
    FloatArray,
    ResamplingMethod,
    IRFPreparationDiagnostics,
    PreparedIRF,
    irf_profile_from_sampled_irf,
    prepare_irf,
)

__all__ = [
    "FloatArray",
    "ResamplingMethod",
    "IRFPreparationDiagnostics",
    "PreparedIRF",
    "irf_profile_from_sampled_irf",
    "prepare_irf",
]
