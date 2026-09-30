import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

import tcspc_toolkit.fitting as fitting

from tcspc_toolkit.classical_evaluation import (
    estimate_reconvolution_initial_guess,
    evaluate_reconvolution_benchmark,
    fit_single_reconvolution_curve,
)
from tcspc_toolkit.bayesian_evaluation import (
    BayesianClassicalCondition,
    build_issue4_expected_counts,
    derive_issue4_seed,
    sample_issue4_observation,
)
from tcspc_toolkit.fitting import poisson_negative_log_likelihood
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    generate_gaussian_irf_profile,
    normalize_irf,
)
from tcspc_toolkit.irf_preparation import prepare_irf
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


def test_single_poisson_reconvolution_curve_recovers_high_count_lifetime_from_histogram_guess(
) -> None:
    """Regression test for Poisson optimizer parameter scaling."""

    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    true_lifetime_ns = 2.0

    irf = generate_gaussian_irf(
        time=time,
        centre=1.0,
        fwhm=0.25,
    )

    irf = normalize_irf(
        time=time,
        irf=irf,
    )

    counts, _ = (
        simulate_irf_convolved_histogram(
            time=time,
            lifetime_ns=(
                true_lifetime_ns
            ),
            signal_photon_count=(
                100_000
            ),
            background_per_bin=0.5,
            irf_centre_ns=1.0,
            irf_fwhm_ns=0.25,
            irf_shift_ns=0.05,
            rng=np.random.default_rng(
                61_999
            ),
        )
    )

    initial_guess = (
        estimate_reconvolution_initial_guess(
            time=time,
            counts=counts,
            irf=irf,
        )
    )

    # This realization deliberately produces a noticeably
    # imperfect histogram-derived lifetime guess.
    assert (
        abs(
            initial_guess.lifetime_ns
            - true_lifetime_ns
        )
        > 0.10
    )

    result = (
        fit_single_reconvolution_curve(
            time=time,
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

    assert (
        abs(
            result.fitted_lifetime_ns
            - true_lifetime_ns
        )
        < 0.02
    )

    assert (
        abs(
            result.fitted_temporal_shift_ns
            - 0.05
        )
        < 0.02
    )

    assert np.isfinite(
        result.poisson_nll
    )


def test_stage5_tau4_realization_22_is_not_falsely_accepted() -> None:
    """Reproduce the frozen observation, without reading generated results."""
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" /
         "issue4_bayesian_workflow.json").read_text(encoding="utf-8")
    )
    grid = manifest["time_grid_ns"]
    time = np.arange(grid["start"], grid["stop"], grid["step"])
    irf_spec = manifest["irf"]
    prepared = prepare_irf(
        generate_gaussian_irf_profile(
            time, gaussian_centre_ns=irf_spec["centre_ns"],
            gaussian_fwhm_ns=irf_spec["fwhm_ns"],
            provenance={"workflow": "issue4_stage5_matched"},
        ), time,
    )
    spec = next(item for item in manifest["conditions"]
                if item["id"] == "tau4_n10000_b0p5")
    model = manifest["common_model"]
    condition = BayesianClassicalCondition(
        condition_id=spec["id"], time_ns=time,
        generating_irf=prepared, assumed_irf=prepared,
        true_lifetime_ns=spec["lifetime_ns"],
        signal_photon_count=spec["signal_photon_count"],
        background_per_bin=spec["background_per_bin"],
        true_temporal_shift_ns=model["true_temporal_shift_ns"],
        n_repeats=manifest["profiles"]["scientific"]["n_repeats_per_condition"],
    )
    seed = derive_issue4_seed(
        manifest["randomness"]["base_seed"], condition.condition_id, 22,
        "observation",
    )
    assert seed == 17339757615716598031
    expected, _ = build_issue4_expected_counts(condition)
    counts = sample_issue4_observation(
        condition, expected, seed, 22,
    ).require_raw_counts()
    assert hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest() == (
        "af659370fe88e8d9080edc1454af70f7699698fd6379d286fddb290af206088d"
    )

    def nll(parameters: np.ndarray) -> float:
        return poisson_negative_log_likelihood(
            counts,
            monoexponential_reconvolution_expected_counts(
                time, prepared.kernel, *parameters,
            ),
        )

    historical = np.array([
        125.02722273171824, 4.139127277582876,
        3.3232667423982516, 0.007020469422371944,
    ])
    lower_background = historical.copy()
    lower_background[2] -= 0.01
    assert nll(historical) - nll(lower_background) > 0.3

    fitted = fit_single_reconvolution_curve(
        time=time, counts=counts, irf=prepared.kernel,
        temporal_shift_bounds=tuple(model["temporal_shift_bounds_ns"]),
        objective="poisson",
        background_fraction=model["classical_background_fraction"],
    )
    assert fitted.optimizer_success
    assert fitted.valid_fit
    assert fitted.numerical_validation_passed is True
    assert fitted.max_coordinate_descent_nll <= 0.01
    parameters = np.array([
        fitted.fitted_amplitude, fitted.fitted_lifetime_ns,
        fitted.fitted_background, fitted.fitted_temporal_shift_ns,
    ])
    assert nll(historical) - nll(parameters) > 1.0

    # Independently probe the accepted fit in the same dimensionless
    # coordinate system as the optimizer's numerical check.
    scales = np.array([
        max(abs(fitted.initial_amplitude), 1.0),
        max(abs(fitted.initial_lifetime_ns), time[1] - time[0]),
        max(abs(fitted.initial_background), 1.0),
        max(abs(fitted.initial_temporal_shift_ns), 0.04, time[1] - time[0]),
    ])
    bounds = [(0.0, np.inf), (1e-12, np.inf),
              (1e-12, np.inf), tuple(model["temporal_shift_bounds_ns"])]
    for index in range(4):
        for direction in (-1.0, 1.0):
            trial = parameters.copy()
            trial[index] += direction * 1e-3 * scales[index]
            if bounds[index][0] <= trial[index] <= bounds[index][1]:
                assert nll(parameters) - nll(trial) <= 0.01


def test_nonfinite_optimizer_output_is_recorded_as_invalid(monkeypatch) -> None:
    time = np.arange(0.0, 6.0, 0.05)
    irf = normalize_irf(time, generate_gaussian_irf(time, centre=1.0, fwhm=0.3))
    counts = np.random.default_rng(42).poisson(
        monoexponential_reconvolution_expected_counts(
            time, irf, 1000.0, 1.5, 1.0, 0.0,
        )
    )

    def invalid_minimize(*, x0, **kwargs):
        return OptimizeResult(
            x=np.full_like(x0, np.nan), fun=np.nan, success=True, status=0,
            message="spurious success", nfev=1, njev=1,
        )

    monkeypatch.setattr(fitting, "minimize", invalid_minimize)
    result = fit_single_reconvolution_curve(
        time=time, counts=counts, irf=irf,
        temporal_shift_bounds=(-0.2, 0.2), objective="poisson",
    )
    assert result.optimizer_success
    assert not result.valid_fit
    assert result.failure_reason == "non_finite_parameters"
    assert result.numerical_validation_passed is False
    assert not result.recovery_attempted
    assert np.isnan(result.poisson_nll)
