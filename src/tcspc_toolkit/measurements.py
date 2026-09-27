"""Canonical single-curve experimental TCSPC measurements.

The arrays describe imported observations. Analysis transformations must keep
this source measurement available rather than changing its count semantics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.irf import normalize_irf
from tcspc_toolkit.preprocessing import (
    time_axes_compatible,
    validate_histogram,
    validate_time_axis,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class TimeUnit(str, Enum):
    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"
    PICOSECOND = "ps"

    @property
    def nanoseconds_per_unit(self) -> float:
        return {
            TimeUnit.SECOND: 1e9,
            TimeUnit.MILLISECOND: 1e6,
            TimeUnit.MICROSECOND: 1e3,
            TimeUnit.NANOSECOND: 1.0,
            TimeUnit.PICOSECOND: 1e-3,
        }[self]


class MeasurementDataKind(str, Enum):
    RAW_COUNTS = "raw_counts"
    PROCESSED_INTENSITY = "processed_intensity"


def convert_time_to_ns(values: ArrayLike, unit: TimeUnit | str) -> FloatArray:
    """Convert explicitly labelled time coordinates to nanoseconds."""
    try:
        selected_unit = TimeUnit(unit)
    except ValueError as exc:
        raise InvalidMeasurementError(f"unsupported time unit: {unit!r}") from exc
    try:
        converted = np.asarray(values, dtype=np.float64) * selected_unit.nanoseconds_per_unit
    except (TypeError, ValueError) as exc:
        raise InvalidMeasurementError("time must contain numeric values") from exc
    if not np.all(np.isfinite(converted)):
        raise InvalidMeasurementError("converted time must contain only finite values")
    return np.array(converted, dtype=np.float64, copy=True)


def _freeze_mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Defend the top-level mapping; nested mutable values remain mutable."""
    if not isinstance(values, Mapping):
        raise InvalidMeasurementError(f"{name} must be a mapping")
    copied = deepcopy(dict(values))
    if any(not isinstance(key, str) for key in copied):
        raise InvalidMeasurementError(f"{name} keys must be strings")
    return MappingProxyType(copied)


def _readonly_copy(values: ArrayLike, dtype: np.dtype) -> NDArray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class SampledIRF:
    """Imported IRF trace on an explicit ns grid, retaining its source scale.

    Array buffers are copied and read-only. Metadata and provenance are copied
    and protected at the top level, but nested objects are not recursively frozen.
    """

    time_ns: FloatArray
    values: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            time = validate_time_axis(self.time_ns)
            values = np.asarray(self.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidMeasurementError(f"invalid sampled IRF: {exc}") from exc
        if values.ndim != 1 or values.shape != time.shape:
            raise InvalidMeasurementError("IRF values must match its one-dimensional time grid")
        try:
            normalize_irf(time=time, irf=values)
        except ValueError as exc:
            raise InvalidMeasurementError(f"invalid sampled IRF: {exc}") from exc
        object.__setattr__(self, "time_ns", _readonly_copy(time, np.dtype(np.float64)))
        object.__setattr__(self, "values", _readonly_copy(values, np.dtype(np.float64)))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "IRF metadata"))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, "IRF provenance"))


@dataclass(frozen=True)
class TCSPCMeasurement:
    """One experimental histogram, with explicit raw or processed semantics.

    Array buffers are copied and read-only. Metadata and provenance are copied
    and protected at the top level, but nested objects are not recursively frozen.
    """

    time_ns: FloatArray
    values: IntArray | FloatArray
    data_kind: MeasurementDataKind
    irf: SampledIRF | None = None
    sample_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data_kind, MeasurementDataKind):
            raise InvalidMeasurementError("data_kind must be a MeasurementDataKind")
        try:
            time = validate_time_axis(self.time_ns)
        except ValueError as exc:
            raise InvalidMeasurementError(f"invalid measurement time: {exc}") from exc

        if self.data_kind is MeasurementDataKind.RAW_COUNTS:
            try:
                validate_histogram(time=time, counts=self.values)
                original = np.asarray(self.values)
                if original.dtype.kind in "iu":
                    if np.any(original > np.iinfo(np.int64).max):
                        raise InvalidMeasurementError("raw counts exceed int64 range")
                    values = np.array(original, dtype=np.int64, copy=True)
                else:
                    as_float = np.asarray(self.values, dtype=np.float64)
                    if np.any(as_float >= 2**53):
                        raise InvalidMeasurementError(
                            "floating raw counts at or above 2**53 are ambiguous; "
                            "supply integer values"
                        )
                    if np.any(as_float != np.rint(as_float)):
                        raise InvalidMeasurementError(
                            "raw counts must be exactly integer-valued; no rounding is applied"
                        )
                    values = as_float.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise InvalidMeasurementError(f"invalid raw counts: {exc}") from exc
        else:
            try:
                values = np.asarray(self.values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise InvalidMeasurementError("processed values must be numeric") from exc
            if values.ndim != 1 or values.shape != time.shape:
                raise InvalidMeasurementError("processed values must match the time grid")
            if not np.all(np.isfinite(values)):
                raise InvalidMeasurementError("processed values must contain only finite values")

        if self.irf is not None:
            if not isinstance(self.irf, SampledIRF):
                raise InvalidMeasurementError("irf must be a SampledIRF")
            if not time_axes_compatible(time, self.irf.time_ns):
                raise InvalidMeasurementError(
                    "IRF and measurement grids differ after conversion to ns; "
                    "Issue #8 will provide resampling"
                )
        if self.sample_id is not None and (
            not isinstance(self.sample_id, str) or not self.sample_id.strip()
        ):
            raise InvalidMeasurementError("sample_id must be a nonempty string")

        object.__setattr__(self, "time_ns", _readonly_copy(time, np.dtype(np.float64)))
        object.__setattr__(self, "values", _readonly_copy(values, values.dtype))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, "provenance"))

    def require_raw_counts(self) -> IntArray:
        """Return raw photon counts or reject invalid Poisson semantics."""
        if self.data_kind is not MeasurementDataKind.RAW_COUNTS:
            raise InvalidMeasurementError(
                "raw photon counts are required for this analysis; "
                "processed intensity cannot use Poisson-count methods"
            )
        return self.values
