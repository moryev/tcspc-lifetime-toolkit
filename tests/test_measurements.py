"""Canonical experimental measurement and sampled IRF semantics."""

import numpy as np
import pytest

from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.irf import normalize_irf
from tcspc_toolkit.measurements import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
    TimeUnit,
    convert_time_to_ns,
)


@pytest.mark.parametrize(
    ("unit", "input_value"),
    [
        ("s", 2e-9),
        ("ms", 2e-6),
        ("us", 2e-3),
        ("ns", 2.0),
        ("ps", 2000.0),
    ],
)
def test_time_units_convert_to_ns(unit: str, input_value: float) -> None:
    result = convert_time_to_ns([0.0, input_value], TimeUnit(unit))
    np.testing.assert_allclose(result, [0.0, 2.0])


def test_unsupported_time_unit_is_rejected() -> None:
    with pytest.raises(InvalidMeasurementError, match="unsupported time unit"):
        convert_time_to_ns([0.0, 1.0], "minutes")


def test_raw_counts_are_integer_and_source_inputs_are_preserved() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.3])
    counts = np.array([0.0, 2.0, 5.0, 1.0])
    metadata = {"instrument": {"channel": 2}}
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=counts,
        data_kind=MeasurementDataKind.RAW_COUNTS,
        metadata=metadata,
    )
    time[0] = 10.0
    counts[0] = 10.0
    metadata["instrument"]["channel"] = 3
    assert measurement.values.dtype == np.dtype(np.int64)
    assert measurement.values[0] == 0
    assert measurement.time_ns[0] == 0.0
    assert measurement.metadata["instrument"]["channel"] == 2
    with pytest.raises(ValueError):
        measurement.values[0] = 99


@pytest.mark.parametrize("counts", [[1, 2.5, 3, 4], [1, -2, 3, 4]])
def test_invalid_raw_counts_are_rejected(counts: list[float]) -> None:
    with pytest.raises(InvalidMeasurementError, match="invalid raw counts"):
        TCSPCMeasurement(
            time_ns=[0.0, 0.1, 0.2, 0.3],
            values=counts,
            data_kind=MeasurementDataKind.RAW_COUNTS,
        )


def test_raw_count_overflow_is_rejected() -> None:
    with pytest.raises(InvalidMeasurementError, match="int64 range"):
        TCSPCMeasurement(
            time_ns=[0.0, 0.1, 0.2, 0.3],
            values=np.array([0, 1, 2**63, 3], dtype=np.uint64),
            data_kind=MeasurementDataKind.RAW_COUNTS,
        )


def test_floating_raw_counts_are_not_silently_rounded() -> None:
    with pytest.raises(InvalidMeasurementError, match="no rounding"):
        TCSPCMeasurement(
            time_ns=[0.0, 0.1, 0.2, 0.3],
            values=[0.0, 1.0000000001, 2.0, 3.0],
            data_kind=MeasurementDataKind.RAW_COUNTS,
        )
    with pytest.raises(InvalidMeasurementError, match="ambiguous"):
        TCSPCMeasurement(
            time_ns=[0.0, 0.1, 0.2, 0.3],
            values=[0.0, 1.0, float(2**53), 3.0],
            data_kind=MeasurementDataKind.RAW_COUNTS,
        )


def test_processed_values_allow_fractional_and_negative_but_not_poisson() -> None:
    measurement = TCSPCMeasurement(
        time_ns=[0.0, 0.1, 0.2, 0.3],
        values=[-0.25, 2.5, 1.0, -0.1],
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    assert measurement.values.dtype == np.dtype(np.float64)
    with pytest.raises(InvalidMeasurementError, match="raw photon counts are required"):
        measurement.require_raw_counts()


@pytest.mark.parametrize("time", [[0.0, 0.1, 0.1, 0.3], [0.0, 0.1, 0.25, 0.3]])
def test_invalid_processed_grid_is_rejected(time: list[float]) -> None:
    with pytest.raises(InvalidMeasurementError, match="invalid measurement time"):
        TCSPCMeasurement(
            time_ns=time,
            values=[1.0, 2.0, 3.0, 4.0],
            data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
        )


def test_sampled_irf_retains_scale_and_accepts_grid_roundoff() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    irf_values = np.array([0.0, 2.0, 10.0, 2.0, 0.0])
    irf = SampledIRF(time_ns=time + 5e-13, values=irf_values)
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=[1, 2, 5, 2, 1],
        data_kind=MeasurementDataKind.RAW_COUNTS,
        irf=irf,
    )
    assert measurement.irf is irf
    np.testing.assert_array_equal(irf.values, irf_values)
    np.testing.assert_allclose(np.trapezoid(normalize_irf(time, irf.values), x=time), 1.0)


@pytest.mark.parametrize("values", [[0, 0, 0, 0], [0, -1, 2, 0], [0, np.nan, 2, 0]])
def test_invalid_sampled_irf_is_rejected(values: list[float]) -> None:
    with pytest.raises(InvalidMeasurementError, match="invalid sampled IRF"):
        SampledIRF(time_ns=[0.0, 0.1, 0.2, 0.3], values=values)


def test_incompatible_irf_grid_is_rejected() -> None:
    irf = SampledIRF(time_ns=[0.0, 0.1, 0.2, 0.3], values=[0, 1, 2, 0])
    with pytest.raises(InvalidMeasurementError, match="grids differ"):
        TCSPCMeasurement(
            time_ns=[0.01, 0.11, 0.21, 0.31],
            values=[0, 2, 5, 1],
            data_kind=MeasurementDataKind.RAW_COUNTS,
            irf=irf,
        )


def test_large_time_origin_does_not_hide_bin_misalignment() -> None:
    time = np.array([1e6, 1e6 + 0.1, 1e6 + 0.2, 1e6 + 0.3])
    irf = SampledIRF(time_ns=time + 0.01, values=[0, 1, 2, 0])
    with pytest.raises(InvalidMeasurementError, match="grids differ"):
        TCSPCMeasurement(
            time_ns=time,
            values=[0, 2, 5, 1],
            data_kind=MeasurementDataKind.RAW_COUNTS,
            irf=irf,
        )
