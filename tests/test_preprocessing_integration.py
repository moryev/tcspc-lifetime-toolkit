"""Integration tests for analysis-specific TCSPC preprocessing workflows."""

import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.config import CountNormalization
from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.fitting import fit_monoexponential_reconvolution
from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.preprocessing import (
    align_to_irf,
    crop_time_window,
    estimate_background,
    normalize_counts,
    rebin_histogram,
    validate_histogram,
)
from tcspc_toolkit.simulation import sample_photon_counts


def test_aligned_histogram_can_be_cropped_relative_to_irf_peak() -> None:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 4.0, 10.0, 20.0, 8.0, 2.0],
        dtype=np.float64,
    )
    irf = np.array(
        [0.0, 0.1, 0.5, 1.0, 0.4, 0.1],
        dtype=np.float64,
    )

    aligned_time = align_to_irf(
        time=time,
        irf=irf,
    )

    cropped_time, cropped_counts = crop_time_window(
        time=aligned_time,
        counts=counts,
        start_time=0.0,
        stop_time=0.3,
    )

    expected_time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    expected_counts = np.array(
        [20.0, 8.0, 2.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        cropped_time,
        expected_time,
    )
    np.testing.assert_array_equal(
        cropped_counts,
        expected_counts,
    )


def test_ml_preprocessing_sequence_produces_normalized_representation() -> None:
    time = (
        np.arange(16, dtype=np.float64)
        * 0.1
    )
    raw_counts = np.array(
        [
            1.0,
            2.0,
            4.0,
            8.0,
            15.0,
            25.0,
            40.0,
            30.0,
            20.0,
            12.0,
            8.0,
            5.0,
            3.0,
            2.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )

    original_time = time.copy()
    original_counts = raw_counts.copy()

    validate_histogram(
        time=time,
        counts=raw_counts,
    )

    cropped_time, cropped_counts = crop_time_window(
        time=time,
        counts=raw_counts,
        start_time=time[2],
        stop_time=time[14],
    )

    rebinned_time, rebinned_counts = rebin_histogram(
        time=cropped_time,
        counts=cropped_counts,
        factor=3,
    )

    normalized_counts = normalize_counts(
        rebinned_counts,
        mode=CountNormalization.TOTAL,
    )

    assert cropped_counts.size == 12
    assert rebinned_counts.size == 4

    assert np.sum(rebinned_counts) == np.sum(cropped_counts)

    assert np.sum(normalized_counts) == pytest.approx(1.0)

    assert normalized_counts.shape == rebinned_counts.shape

    np.testing.assert_array_equal(
        time,
        original_time,
    )
    np.testing.assert_array_equal(
        raw_counts,
        original_counts,
    )


def test_background_estimate_can_initialize_raw_count_poisson_fit(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    true_amplitude = 100_000.0
    true_lifetime = 0.5
    true_background = 5.0

    decay = monoexponential_decay(
        time=time_axis,
        amplitude=1.0,
        lifetime=true_lifetime,
        background=0.0,
    )

    convolved = convolve_decay_with_irf(
        time=time_axis,
        decay=decay,
        irf=irf,
    )

    expected_counts = (
        true_amplitude * convolved
        + true_background
    )

    rng = np.random.default_rng(42)

    raw_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    original_counts = raw_counts.copy()

    validate_histogram(
        time=time_axis,
        counts=raw_counts,
    )

    background_guess = estimate_background(
        counts=raw_counts,
        start_bin=0,
        stop_bin=50,
    )

    result = fit_monoexponential_reconvolution(
        time=time_axis,
        counts=raw_counts,
        irf=irf,
        initial_guess=(
            90_000.0,
            0.7,
            max(background_guess, 1e-6),
            0.0,
        ),
        temporal_shift_bounds=(
            -0.5,
            0.5,
        ),
        objective="poisson",
    )

    assert result.success

    assert result.lifetime > 0.0
    assert result.background > 0.0

    np.testing.assert_array_equal(
        raw_counts,
        original_counts,
    )
