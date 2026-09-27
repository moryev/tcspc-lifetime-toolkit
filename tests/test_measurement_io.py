"""Strict experimental CSV import tests."""

from pathlib import Path

import numpy as np
import pytest

from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.measurement_io import (
    load_sampled_irf_csv,
    load_tcspc_measurement_csv,
)
from tcspc_toolkit.measurements import MeasurementDataKind


def _csv(tmp_path: Path, content: str, name: str = "measurement.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_import_raw_ps_csv_with_same_file_irf_and_provenance(tmp_path: Path) -> None:
    path = _csv(
        tmp_path,
        "delay_ps,photons,irf,ignored\n"
        "0,0,0,a\n100,2,2,b\n200,5,10,c\n300,1,2,d\n400,0,0,e\n",
    )
    metadata = {"instrument": {"channel": 2}}
    measurement = load_tcspc_measurement_csv(
        path,
        time_unit="ps",
        data_kind="raw_counts",
        time_column="delay_ps",
        value_column="photons",
        irf_column="irf",
        sample_id="sample-1",
        metadata=metadata,
    )
    metadata["instrument"]["channel"] = 3
    np.testing.assert_allclose(measurement.time_ns, [0, 0.1, 0.2, 0.3, 0.4])
    assert measurement.values.dtype == np.dtype(np.int64)
    assert measurement.sample_id == "sample-1"
    assert measurement.metadata["instrument"]["channel"] == 2
    assert measurement.provenance["source_path"] == str(path.resolve())
    assert measurement.provenance["original_time_unit"] == "ps"
    assert measurement.provenance["data_kind"] == "raw_counts"
    assert measurement.irf is not None
    assert measurement.irf.provenance["value_column"] == "irf"
    np.testing.assert_array_equal(measurement.irf.values, [0, 2, 10, 2, 0])


def test_separate_irf_csv_converts_grid_and_retains_own_source(tmp_path: Path) -> None:
    decay_path = _csv(
        tmp_path, "time,counts\n0,0\n0.1,2\n0.2,5\n0.3,1\n", "decay.csv"
    )
    irf_path = _csv(
        tmp_path, "time,irf\n0,0\n100,1\n200,4\n300,0\n", "irf.csv"
    )
    irf = load_sampled_irf_csv(irf_path, time_unit="ps")
    measurement = load_tcspc_measurement_csv(
        decay_path, time_unit="ns", data_kind="raw_counts", irf=irf
    )
    assert measurement.irf is irf
    assert irf.provenance["source_path"] == str(irf_path.resolve())


def test_separate_irf_grid_accepts_roundoff_after_unit_conversion(tmp_path: Path) -> None:
    decay = _csv(tmp_path, "time,counts\n0,1\n0.1,2\n0.2,5\n0.3,1\n", "decay.csv")
    irf_path = _csv(
        tmp_path,
        "time,irf\n0.0000000004,0\n100.0000000004,1\n"
        "200.0000000004,4\n300.0000000004,0\n",
        "irf.csv",
    )
    irf = load_sampled_irf_csv(irf_path, time_unit="ps")
    measurement = load_tcspc_measurement_csv(
        decay, time_unit="ns", data_kind="raw_counts", irf=irf
    )
    assert measurement.irf is irf


def test_processed_csv_keeps_floating_negative_values(tmp_path: Path) -> None:
    path = _csv(tmp_path, "time,intensity\n0,-0.2\n1,1.25\n2,0.5\n")
    measurement = load_tcspc_measurement_csv(
        path, time_unit="ns", data_kind="processed_intensity", value_column="intensity"
    )
    assert measurement.data_kind is MeasurementDataKind.PROCESSED_INTENSITY
    np.testing.assert_allclose(measurement.values, [-0.2, 1.25, 0.5])


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("time,other\n0,1\n1,2\n", "missing CSV column"),
        ("time,counts,counts\n0,1,2\n1,2,3\n", "duplicate or empty"),
        ("time,counts\n0,1\n1\n", "has 1 fields"),
        ("time,counts\n", "no data rows"),
        ("time,counts\n0,1\n1,nope\n", "not numeric"),
        ("time,counts\n0,1\n1,nan\n", "must be finite"),
    ],
)
def test_malformed_csv_is_rejected(tmp_path: Path, content: str, error: str) -> None:
    path = _csv(tmp_path, content)
    with pytest.raises(InvalidMeasurementError, match=error):
        load_tcspc_measurement_csv(path, time_unit="ns", data_kind="raw_counts")


@pytest.mark.parametrize("value", ["2.5", "-1"])
def test_raw_csv_rejects_noncount_values(tmp_path: Path, value: str) -> None:
    path = _csv(tmp_path, f"time,counts\n0,1\n1,{value}\n2,3\n")
    with pytest.raises(InvalidMeasurementError, match="invalid raw counts"):
        load_tcspc_measurement_csv(path, time_unit="ns", data_kind="raw_counts")


@pytest.mark.parametrize(
    "token", ["9007199254740993", "9223372036854775807"]
)
def test_raw_csv_preserves_large_exact_int64_counts(tmp_path: Path, token: str) -> None:
    path = _csv(tmp_path, f"time,counts\n0,1\n1,{token}\n2,3\n")
    measurement = load_tcspc_measurement_csv(
        path, time_unit="ns", data_kind="raw_counts"
    )
    assert measurement.values.dtype == np.dtype(np.int64)
    assert measurement.values[1] == int(token)


@pytest.mark.parametrize(
    ("token", "error"),
    [
        ("9223372036854775808", "int64 range"),
        ("18446744073709551616", "int64 range"),
        ("9007199254740993.5", "non-negative integers"),
        ("1.0000000001", "non-negative integers"),
    ],
)
def test_raw_csv_rejects_inexact_or_out_of_range_tokens(
    tmp_path: Path, token: str, error: str
) -> None:
    path = _csv(tmp_path, f"time,counts\n0,1\n1,{token}\n2,3\n")
    with pytest.raises(InvalidMeasurementError, match=error):
        load_tcspc_measurement_csv(path, time_unit="ns", data_kind="raw_counts")


def test_missing_unit_and_ambiguous_irf_are_rejected(tmp_path: Path) -> None:
    path = _csv(tmp_path, "time,counts,irf\n0,1,0\n1,2,1\n2,3,0\n")
    with pytest.raises(TypeError, match="time_unit"):
        load_tcspc_measurement_csv(path, data_kind="raw_counts")
    irf = load_sampled_irf_csv(path, time_unit="ns")
    with pytest.raises(InvalidMeasurementError, match="either irf or irf_column"):
        load_tcspc_measurement_csv(
            path, time_unit="ns", data_kind="raw_counts", irf=irf, irf_column="irf"
        )


def test_separate_incompatible_irf_grid_is_rejected(tmp_path: Path) -> None:
    decay = _csv(tmp_path, "time,counts\n0,1\n1,2\n2,3\n", "decay.csv")
    irf_path = _csv(tmp_path, "time,irf\n0.1,0\n1.1,1\n2.1,0\n", "irf.csv")
    irf = load_sampled_irf_csv(irf_path, time_unit="ns")
    with pytest.raises(InvalidMeasurementError, match="grids differ"):
        load_tcspc_measurement_csv(
            decay, time_unit="ns", data_kind="raw_counts", irf=irf
        )
