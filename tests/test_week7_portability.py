"""Separate Week-7 realization integrity from portable simulator behavior."""

from __future__ import annotations

import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy

from tcspc_toolkit.ml.evaluation import generate_benchmark_measurements
import tcspc_toolkit.simulation.histograms as histograms


REFERENCE = Path(__file__).parent / "fixtures/week7_sample_770_windows_reference.json"
REFERENCE_SHA256 = "035f408aa945ef4df47b43ab8c7cada90e05877f5f029323804d1c8a3e1e95ca"
LIFETIMES_NS = (0.5, 1.0, 2.0, 3.0, 4.5, 6.0)
SIGNAL_PHOTONS = (1000, 10000, 100000)
BACKGROUNDS = (0.0, 0.5, 2.0)
IRF_FWHM_NS = (0.15, 0.30, 0.60)
IRF_SHIFTS_NS = (-0.20, -0.05, 0.0, 0.05, 0.20)


def _reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _generate_week7():
    """Use the original frozen parameter grid and single seeded RNG stream."""
    return generate_benchmark_measurements(
        time=np.arange(0.0, 20.0, 0.05),
        lifetimes_ns=LIFETIMES_NS,
        signal_photon_counts=SIGNAL_PHOTONS,
        background_levels=BACKGROUNDS,
        irf_centre_ns=1.0,
        irf_fwhm_values_ns=IRF_FWHM_NS,
        irf_shift_values_ns=IRF_SHIFTS_NS,
        random_seed=42,
    )


def test_week7_sample_770_original_fixture_integrity() -> None:
    """The original Windows count realization, not a Linux hash, is fixed."""
    reference = _reference()
    counts = np.asarray(reference["counts"], dtype="<i8")
    assert reference["sample_id"] == 770
    assert counts.shape == (400,)
    assert np.all(counts >= 0)
    assert hashlib.sha256(counts.tobytes()).hexdigest() == REFERENCE_SHA256


def test_week7_simulator_forward_model_is_portable(monkeypatch) -> None:
    """Sub-ULP platform differences must not alter the physical count model."""
    expected_by_sample = []
    original_sample = histograms.sample_photon_counts

    def record_expected(*, expected_counts, rng):
        expected_by_sample.append(np.array(expected_counts, copy=True))
        return original_sample(expected_counts=expected_counts, rng=rng)

    monkeypatch.setattr(histograms, "sample_photon_counts", record_expected)
    measured = _generate_week7()
    reference = _reference()

    assert measured.X_histograms.shape == (810, 400)
    assert measured.X_histograms.dtype == np.int64
    assert np.all(measured.X_histograms >= 0)
    assert len(expected_by_sample) == 810
    assert np.array_equal(measured.time, np.arange(0.0, 20.0, 0.05))

    for sample_id, parameters in enumerate(
        product(LIFETIMES_NS, SIGNAL_PHOTONS, BACKGROUNDS, IRF_FWHM_NS, IRF_SHIFTS_NS)
    ):
        lifetime, photons, background, fwhm, shift = parameters
        row = measured.metadata.iloc[sample_id]
        assert row["sample_id"] == sample_id
        assert measured.y[sample_id] == lifetime
        assert row["signal_photon_count_target"] == photons
        assert row["background_per_bin"] == background
        assert row["irf_fwhm_ns"] == fwhm
        assert row["irf_shift_ns"] == shift
        assert row["expected_signal_counts"] == pytest.approx(photons, rel=0, abs=1e-9)
        assert row["expected_background_counts"] == pytest.approx(
            400 * background, rel=0, abs=1e-12
        )
        assert row["expected_total_counts"] == pytest.approx(
            photons + 400 * background, rel=0, abs=1e-9
        )
        assert row["measured_total_counts"] == int(
            measured.X_histograms[sample_id].sum()
        )
        assert np.all(np.isfinite(expected_by_sample[sample_id]))
        assert np.all(expected_by_sample[sample_id] >= 0)

    # These profiles span the near-zero tail, first direct Poisson divergence,
    # and frozen fitter sample. 1e-11 absolute counts is >5x the largest
    # Windows/Linux deviation observed over all 810 traces (1.82e-12 counts),
    # yet negligible beside even one detected photon.
    for sample_id, expected_reference in reference["expected_counts_by_sample"].items():
        np.testing.assert_allclose(
            expected_by_sample[int(sample_id)],
            np.asarray(expected_reference, dtype=np.float64),
            rtol=1e-13,
            atol=1e-11,
        )


@pytest.mark.skipif(
    os.environ.get("TCSPC_VERIFY_WEEK7_REFERENCE") != "1",
    reason="Exact stochastic regeneration requires the qualified Windows reference environment",
)
def test_week7_sample_770_regenerates_in_reference_environment() -> None:
    """Run explicitly in the pinned Windows environment, never as a Linux gate."""
    assert sys.platform == "win32"
    assert sys.version_info[:3] == (3, 12, 0)
    assert np.__version__ == "2.5.1"
    assert scipy.__version__ == "1.18.0"
    measured = _generate_week7()
    counts = np.asarray(measured.X_histograms[770], dtype="<i8")
    assert measured.metadata.iloc[770]["sample_id"] == 770
    assert hashlib.sha256(counts.tobytes()).hexdigest() == REFERENCE_SHA256
    np.testing.assert_array_equal(
        counts, np.asarray(_reference()["counts"], dtype="<i8")
    )
