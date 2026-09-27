"""Compatibility checks for the CSV baseline CLI."""

import sys
from pathlib import Path

import pytest

from tcspc_toolkit.cli import main


def _run_fit(monkeypatch: pytest.MonkeyPatch, path: Path, *options: str) -> None:
    monkeypatch.setattr(sys, "argv", ["tcspc", "fit", "--input", str(path), *options])
    main()


def test_cli_fit_uses_canonical_raw_csv_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "decay.csv"
    path.write_text(
        "time_ns,measured_counts\n"
        "0,100\n1,62\n2,40\n3,26\n4,18\n5,13\n6,10\n",
        encoding="utf-8",
    )
    _run_fit(monkeypatch, path)
    output = capsys.readouterr().out
    assert "Time unit: ns; data kind: raw_counts" in output
    assert "Fit status: success" in output


def test_cli_requires_explicit_processed_kind_for_fractional_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "processed.csv"
    path.write_text(
        "time_ns,measured_counts\n"
        "0,100.5\n1,62.2\n2,40.1\n3,26.4\n4,18.3\n5,13.2\n6,10.1\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="invalid raw counts"):
        _run_fit(monkeypatch, path)
    _run_fit(monkeypatch, path, "--data-kind", "processed_intensity")
    assert "data kind: processed_intensity" in capsys.readouterr().out
