import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.classical_evaluation import (
    estimate_reconvolution_initial_guess,
    evaluate_reconvolution_benchmark,
    fit_single_reconvolution_curve,
)
from tcspc_toolkit.simulation import (
    simulate_irf_convolved_histogram,
)


def _make_clean_measurement(
    time_axis: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    float,
]:
    true_lifetime_ns = 0.5

    counts, _ = (
        simulate_irf_convolved_histogram(
            time=time_axis,
            lifetime_ns=true_lifetime_ns,
            signal_photon_count=1_000_000,
            background_per_bin=5.0,
            irf_centre_ns=1.0,
            irf_fwhm_ns=0.2,
            irf_shift_ns=0.0,
            rng=np.random.default_rng(42),
        )
    )

    return (
        counts,
        true_lifetime_ns,
    )


def test_reconvolution_initial_guess_is_physical(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    counts, _ = _make_clean_measurement(
        time_axis
    )

    guess = (
        estimate_reconvolution_initial_guess(
            time=time_axis,
            counts=counts,
            irf=irf,
        )
    )

    assert np.isfinite(
        guess.amplitude
    )

    assert np.isfinite(
        guess.lifetime_ns
    )

    assert np.isfinite(
        guess.background
    )

    assert guess.amplitude > 0.0
    assert guess.lifetime_ns > 0.0
    assert guess.background > 0.0

    assert (
        guess.temporal_shift_ns
        == pytest.approx(0.0)
    )


def test_reconvolution_initial_guess_estimates_background(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    counts, _ = _make_clean_measurement(
        time_axis
    )

    guess = (
        estimate_reconvolution_initial_guess(
            time=time_axis,
            counts=counts,
            irf=irf,
        )
    )

    assert guess.background == pytest.approx(
        5.0,
        abs=2.0,
    )


def test_single_reconvolution_curve_recovers_clean_lifetime(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    counts, true_lifetime_ns = (
        _make_clean_measurement(
            time_axis
        )
    )

    result = (
        fit_single_reconvolution_curve(
            time=time_axis,
            counts=counts,
            irf=irf,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
            objective="poisson",
        )
    )

    assert result.optimizer_success
    assert result.valid_fit

    assert np.isfinite(
        result.fitted_lifetime_ns
    )

    assert abs(
        result.fitted_lifetime_ns
        - true_lifetime_ns
    ) < 0.10

    assert result.runtime_ms > 0.0

    assert np.isfinite(
        result.poisson_nll
    )

    assert np.isfinite(
        result.poisson_deviance
    )

    assert result.failure_reason is None
    assert result.exception_message is None


def test_single_reconvolution_curve_records_initialization_failure(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    counts = np.zeros(
        time_axis.size,
        dtype=np.int64,
    )

    result = (
        fit_single_reconvolution_curve(
            time=time_axis,
            counts=counts,
            irf=irf,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
            objective="poisson",
        )
    )

    assert not result.optimizer_success
    assert not result.valid_fit

    assert (
        result.failure_reason
        == "initialization_error"
    )

    assert (
        result.exception_message
        is not None
    )

    assert np.isnan(
        result.fitted_lifetime_ns
    )


def test_reconvolution_benchmark_counts_failures(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    X_histograms = np.zeros(
        (
            2,
            time_axis.size,
        ),
        dtype=np.int64,
    )

    y_true = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": [10, 11],
        }
    )

    result = (
        evaluate_reconvolution_benchmark(
            time=time_axis,
            X_histograms=X_histograms,
            y_true=y_true,
            metadata=metadata,
            irf=irf,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
            objective="poisson",
        )
    )

    assert result.summary.n_samples == 2

    assert (
        result.summary.n_successful_fits
        == 0
    )

    assert (
        result.summary.n_failed_fits
        == 2
    )

    assert (
        result.summary.success_rate
        == pytest.approx(0.0)
    )

    assert (
        result.summary.failure_rate
        == pytest.approx(1.0)
    )

    assert np.isnan(
        result.summary.mae_valid_ns
    )

    assert np.isnan(
        result.summary.rmse_valid_ns
    )


def test_reconvolution_benchmark_preserves_sample_alignment(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    X_histograms = np.zeros(
        (
            2,
            time_axis.size,
        ),
        dtype=np.int64,
    )

    y_true = np.array(
        [1.25, 3.75],
        dtype=np.float64,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": [
                101,
                202,
            ],
            "background_per_bin": [
                1.0,
                5.0,
            ],
        }
    )

    result = (
        evaluate_reconvolution_benchmark(
            time=time_axis,
            X_histograms=X_histograms,
            y_true=y_true,
            metadata=metadata,
            irf=irf,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )
    )

    np.testing.assert_array_equal(
        result.per_curve[
            "sample_id"
        ].to_numpy(),
        np.array(
            [101, 202]
        ),
    )

    np.testing.assert_allclose(
        result.per_curve[
            "true_lifetime_ns"
        ].to_numpy(),
        y_true,
    )


def test_reconvolution_benchmark_rejects_target_length_mismatch(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    X_histograms = np.zeros(
        (
            2,
            time_axis.size,
        ),
        dtype=np.int64,
    )

    y_true = np.array(
        [1.0],
        dtype=np.float64,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": [
                0,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="one lifetime per histogram",
    ):
        evaluate_reconvolution_benchmark(
            time=time_axis,
            X_histograms=X_histograms,
            y_true=y_true,
            metadata=metadata,
            irf=irf,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )


