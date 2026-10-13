"""Compatibility imports for :mod:`tcspc_toolkit.measurement.records`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.measurement.records import (
    FloatArray,
    IntArray,
    TimeUnit,
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
    convert_time_to_ns,
)

__all__ = [
    "FloatArray",
    "IntArray",
    "TimeUnit",
    "MeasurementDataKind",
    "SampledIRF",
    "TCSPCMeasurement",
    "convert_time_to_ns",
]
