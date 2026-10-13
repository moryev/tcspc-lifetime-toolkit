"""Compatibility imports for :mod:`tcspc_toolkit.measurement.io`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.measurement.io import (
    load_sampled_irf_csv,
    load_tcspc_measurement_csv,
)

__all__ = [
    "load_sampled_irf_csv",
    "load_tcspc_measurement_csv",
]
