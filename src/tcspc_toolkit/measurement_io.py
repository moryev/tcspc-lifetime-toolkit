"""Strict CSV import for single-curve experimental TCSPC measurements."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.measurements import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
    TimeUnit,
    convert_time_to_ns,
)


def _selected_unit(unit: TimeUnit | str) -> TimeUnit:
    try:
        return TimeUnit(unit)
    except ValueError as exc:
        raise InvalidMeasurementError(f"unsupported time unit: {unit!r}") from exc


def _parse_raw_count_token(token: str, row_number: int, column_name: str) -> int:
    """Validate a CSV count exactly, before any binary-float conversion."""
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise InvalidMeasurementError(
            f"CSV row {row_number}, column {column_name!r} is not numeric"
        ) from exc
    if not value.is_finite():
        raise InvalidMeasurementError(
            f"CSV row {row_number}, column {column_name!r} must be finite"
        )
    if value < 0 or value != value.to_integral_value():
        raise InvalidMeasurementError(
            f"invalid raw counts in CSV row {row_number}, column {column_name!r}: "
            "counts must be non-negative integers"
        )
    if value > np.iinfo(np.int64).max:
        raise InvalidMeasurementError(
            f"invalid raw counts in CSV row {row_number}, column {column_name!r}: "
            "counts exceed int64 range"
        )
    return int(value)


def _read_csv_columns(
    path: str | Path,
    column_names: tuple[str, ...],
    *,
    raw_count_column: str | None = None,
) -> dict[str, np.ndarray]:
    input_path = Path(path)
    if len(set(column_names)) != len(column_names):
        raise InvalidMeasurementError("selected CSV columns must be distinct")
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise InvalidMeasurementError("CSV file has no header")
            if len(set(header)) != len(header) or any(not name.strip() for name in header):
                raise InvalidMeasurementError("CSV header contains duplicate or empty columns")
            missing = [name for name in column_names if name not in header]
            if missing:
                raise InvalidMeasurementError(
                    "missing CSV column(s): " + ", ".join(missing)
                )
            indices = {name: header.index(name) for name in column_names}
            columns: dict[str, list[int | float]] = {name: [] for name in column_names}
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise InvalidMeasurementError(
                        f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
                    )
                for name, index in indices.items():
                    if name == raw_count_column:
                        columns[name].append(_parse_raw_count_token(row[index], row_number, name))
                        continue
                    try:
                        value = float(row[index])
                    except ValueError as exc:
                        raise InvalidMeasurementError(
                            f"CSV row {row_number}, column {name!r} is not numeric"
                        ) from exc
                    if not np.isfinite(value):
                        raise InvalidMeasurementError(
                            f"CSV row {row_number}, column {name!r} must be finite"
                        )
                    columns[name].append(value)
    except OSError as exc:
        raise InvalidMeasurementError(f"cannot read CSV file {input_path}: {exc}") from exc
    except csv.Error as exc:
        raise InvalidMeasurementError(f"malformed CSV file {input_path}: {exc}") from exc
    if not next(iter(columns.values())):
        raise InvalidMeasurementError("CSV file contains no data rows")
    return {
        name: np.asarray(
            values,
            dtype=np.int64 if name == raw_count_column else np.float64,
        )
        for name, values in columns.items()
    }


def _source_provenance(
    path: str | Path, *, time_unit: TimeUnit, time_column: str, value_column: str
) -> dict[str, str]:
    return {
        "source_path": str(Path(path).resolve()),
        "source_format": "csv",
        "original_time_unit": time_unit.value,
        "time_column": time_column,
        "value_column": value_column,
    }


def load_sampled_irf_csv(
    path: str | Path,
    *,
    time_unit: TimeUnit | str,
    time_column: str = "time",
    value_column: str = "irf",
    metadata: Mapping[str, Any] | None = None,
) -> SampledIRF:
    """Load one sampled IRF without altering its imported amplitudes."""
    unit = _selected_unit(time_unit)
    columns = _read_csv_columns(path, (time_column, value_column))
    return SampledIRF(
        time_ns=convert_time_to_ns(columns[time_column], unit),
        values=columns[value_column],
        metadata={} if metadata is None else metadata,
        provenance=_source_provenance(
            path, time_unit=unit, time_column=time_column, value_column=value_column
        ),
    )


def load_tcspc_measurement_csv(
    path: str | Path,
    *,
    time_unit: TimeUnit | str,
    data_kind: MeasurementDataKind | str,
    time_column: str = "time",
    value_column: str = "counts",
    irf: SampledIRF | None = None,
    irf_column: str | None = None,
    sample_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TCSPCMeasurement:
    """Import a single CSV histogram with declared units and data semantics."""
    unit = _selected_unit(time_unit)
    try:
        selected_kind = MeasurementDataKind(data_kind)
    except ValueError as exc:
        raise InvalidMeasurementError(f"unsupported data kind: {data_kind!r}") from exc
    if irf is not None and irf_column is not None:
        raise InvalidMeasurementError("provide either irf or irf_column, not both")
    selected_columns = (time_column, value_column)
    if irf_column is not None:
        selected_columns += (irf_column,)
    columns = _read_csv_columns(
        path,
        selected_columns,
        raw_count_column=value_column if selected_kind is MeasurementDataKind.RAW_COUNTS else None,
    )
    time_ns = convert_time_to_ns(columns[time_column], unit)
    if irf_column is not None:
        irf = SampledIRF(
            time_ns=time_ns,
            values=columns[irf_column],
            provenance=_source_provenance(
                path, time_unit=unit, time_column=time_column, value_column=irf_column
            ),
        )
    provenance = _source_provenance(
        path, time_unit=unit, time_column=time_column, value_column=value_column
    )
    provenance["data_kind"] = selected_kind.value
    return TCSPCMeasurement(
        time_ns=time_ns,
        values=columns[value_column],
        data_kind=selected_kind,
        irf=irf,
        sample_id=sample_id,
        metadata={} if metadata is None else metadata,
        provenance=provenance,
    )
