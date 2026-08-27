"""Classical reconvolution benchmarking for TCSPC lifetime estimation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

import numpy as np
from numpy.typing import ArrayLike

from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.evaluation import (
    calculate_poisson_deviance_residuals,
)
from tcspc_toolkit.fitting import (
    fit_monoexponential_reconvolution,
    poisson_negative_log_likelihood,
)
from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.preprocessing import (
    detect_peak,
    estimate_background,
    validate_histogram,
)


@dataclass(frozen=True)
class ReconvolutionInitialGuess:
    """Histogram-derived initial parameters for reconvolution fitting."""

    amplitude: float
    lifetime_ns: float
    background: float
    temporal_shift_ns: float


@dataclass(frozen=True)
class ReconvolutionCurveResult:
    """Reconvolution result and diagnostics for one TCSPC histogram."""

    initial_amplitude: float
    initial_lifetime_ns: float
    initial_background: float
    initial_temporal_shift_ns: float

    fitted_amplitude: float
    fitted_lifetime_ns: float
    fitted_background: float
    fitted_temporal_shift_ns: float

    optimizer_success: bool
    valid_fit: bool
    boundary_hit: bool

    poisson_nll: float
    poisson_deviance: float
    runtime_ms: float

    failure_reason: str | None
    exception_message: str | None


@dataclass(frozen=True)
class ReconvolutionBenchmarkSummary:
    """Aggregate results of a classical reconvolution benchmark."""

    n_samples: int
    n_successful_fits: int
    n_failed_fits: int

    success_rate: float
    failure_rate: float

    mae_valid_ns: float
    median_absolute_error_valid_ns: float
    rmse_valid_ns: float

    mean_runtime_ms: float
    median_runtime_ms: float


@dataclass(frozen=True)
class ReconvolutionBenchmarkResult:
    """Per-curve diagnostics and aggregate reconvolution results."""

    per_curve: pd.DataFrame
    summary: ReconvolutionBenchmarkSummary


def estimate_reconvolution_initial_guess(
    *,
    time: ArrayLike,
    counts: ArrayLike,
    irf: ArrayLike,
    background_fraction: float = 0.10,
) -> ReconvolutionInitialGuess:
    """Estimate reconvolution starting parameters from one histogram."""

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )
    irf_array = np.asarray(
        irf,
        dtype=np.float64,
    )

    validate_histogram(
        time=time_array,
        counts=counts_array,
    )

    if irf_array.ndim != 1:
        raise ValueError(
            "irf must be one-dimensional."
        )

    if irf_array.shape != time_array.shape:
        raise ValueError(
            "irf and time must have the same shape."
        )

    if not np.all(np.isfinite(irf_array)):
        raise ValueError(
            "irf must contain only finite values."
        )

    if np.any(irf_array < 0.0):
        raise ValueError(
            "irf must contain only non-negative values."
        )

    if not (
        0.0
        < background_fraction
        < 1.0
    ):
        raise ValueError(
            "background_fraction must lie between 0 and 1."
        )

    n_bins = counts_array.size

    n_background_bins = max(
        1,
        int(
            np.ceil(
                background_fraction
                * n_bins
            )
        ),
    )

    background_start_bin = (
        n_bins
        - n_background_bins
    )

    background_guess = estimate_background(
        counts=counts_array,
        start_bin=background_start_bin,
        stop_bin=n_bins,
    )

    signal_weights = np.clip(
        counts_array - background_guess,
        a_min=0.0,
        a_max=None,
    )

    estimated_signal_counts = float(
        np.sum(signal_weights)
    )

    if estimated_signal_counts <= 0.0:
        raise ValueError(
            "Could not estimate a positive signal "
            "above background."
        )

    peak_index = detect_peak(
        counts_array
    )

    peak_time_ns = float(
        time_array[peak_index]
    )

    mean_arrival_time_ns = float(
        np.sum(
            time_array
            * signal_weights
        )
        / estimated_signal_counts
    )

    bin_width_ns = float(
        time_array[1]
        - time_array[0]
    )

    lifetime_guess_ns = (
        mean_arrival_time_ns
        - peak_time_ns
    )

    lifetime_guess_ns = max(
        lifetime_guess_ns,
        bin_width_ns,
    )

    unit_decay = monoexponential_decay(
        time=time_array,
        amplitude=1.0,
        lifetime=lifetime_guess_ns,
        background=0.0,
    )

    unit_convolved = convolve_decay_with_irf(
        time=time_array,
        decay=unit_decay,
        irf=irf_array,
    )

    unit_signal_sum = float(
        np.sum(unit_convolved)
    )

    if (
        not np.isfinite(unit_signal_sum)
        or unit_signal_sum <= 0.0
    ):
        raise ValueError(
            "Initial reconvolved model has "
            "non-positive signal."
        )

    amplitude_guess = (
        estimated_signal_counts
        / unit_signal_sum
    )

    background_guess = max(
        float(background_guess),
        1e-12,
    )

    return ReconvolutionInitialGuess(
        amplitude=float(amplitude_guess),
        lifetime_ns=float(lifetime_guess_ns),
        background=background_guess,
        temporal_shift_ns=0.0,
    )


def _fit_hits_boundary(
    *,
    amplitude: float,
    lifetime_ns: float,
    background: float,
    temporal_shift_ns: float,
    temporal_shift_bounds: tuple[float, float],
    bin_width_ns: float,
) -> bool:
    """Return whether fitted parameters lie effectively on a bound."""

    shift_lower, shift_upper = (
        temporal_shift_bounds
    )

    shift_tolerance = (
        0.5 * bin_width_ns
    )

    amplitude_at_lower_bound = np.isclose(
        amplitude,
        0.0,
        rtol=0.0,
        atol=1e-10,
    )

    lifetime_at_lower_bound = np.isclose(
        lifetime_ns,
        1e-12,
        rtol=0.0,
        atol=1e-10,
    )

    background_at_lower_bound = (
        background <= 1e-10
    )

    shift_at_lower_bound = np.isclose(
        temporal_shift_ns,
        shift_lower,
        rtol=0.0,
        atol=shift_tolerance,
    )

    shift_at_upper_bound = np.isclose(
        temporal_shift_ns,
        shift_upper,
        rtol=0.0,
        atol=shift_tolerance,
    )

    return bool(
        amplitude_at_lower_bound
        or lifetime_at_lower_bound
        or background_at_lower_bound
        or shift_at_lower_bound
        or shift_at_upper_bound
    )


def fit_single_reconvolution_curve(
    *,
    time: ArrayLike,
    counts: ArrayLike,
    irf: ArrayLike,
    temporal_shift_bounds: tuple[float, float],
    objective: str = "poisson",
    background_fraction: float = 0.10,
) -> ReconvolutionCurveResult:
    """Fit one histogram and retain optimizer and fit diagnostics."""

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )
    counts_array = np.asarray(
        counts,
        dtype=np.float64,
    )
    irf_array = np.asarray(
        irf,
        dtype=np.float64,
    )

    try:
        initial_guess = (
            estimate_reconvolution_initial_guess(
                time=time_array,
                counts=counts_array,
                irf=irf_array,
                background_fraction=background_fraction,
            )
        )

    except Exception as exc:
        return ReconvolutionCurveResult(
            initial_amplitude=np.nan,
            initial_lifetime_ns=np.nan,
            initial_background=np.nan,
            initial_temporal_shift_ns=np.nan,
            fitted_amplitude=np.nan,
            fitted_lifetime_ns=np.nan,
            fitted_background=np.nan,
            fitted_temporal_shift_ns=np.nan,
            optimizer_success=False,
            valid_fit=False,
            boundary_hit=False,
            poisson_nll=np.nan,
            poisson_deviance=np.nan,
            runtime_ms=np.nan,
            failure_reason="initialization_error",
            exception_message=str(exc),
        )

    fit_start = perf_counter()

    try:
        fit_result = (
            fit_monoexponential_reconvolution(
                time=time_array,
                counts=counts_array,
                irf=irf_array,
                initial_guess=(
                    initial_guess.amplitude,
                    initial_guess.lifetime_ns,
                    initial_guess.background,
                    initial_guess.temporal_shift_ns,
                ),
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective=objective,
            )
        )

    except Exception as exc:
        runtime_ms = (
            perf_counter()
            - fit_start
        ) * 1000.0

        return ReconvolutionCurveResult(
            initial_amplitude=initial_guess.amplitude,
            initial_lifetime_ns=initial_guess.lifetime_ns,
            initial_background=initial_guess.background,
            initial_temporal_shift_ns=(
                initial_guess.temporal_shift_ns
            ),
            fitted_amplitude=np.nan,
            fitted_lifetime_ns=np.nan,
            fitted_background=np.nan,
            fitted_temporal_shift_ns=np.nan,
            optimizer_success=False,
            valid_fit=False,
            boundary_hit=False,
            poisson_nll=np.nan,
            poisson_deviance=np.nan,
            runtime_ms=float(runtime_ms),
            failure_reason="fit_exception",
            exception_message=str(exc),
        )

    runtime_ms = (
        perf_counter()
        - fit_start
    ) * 1000.0

    fitted_parameters = np.array(
        [
            fit_result.amplitude,
            fit_result.lifetime,
            fit_result.background,
            fit_result.temporal_shift,
        ],
        dtype=np.float64,
    )

    parameters_are_finite = bool(
        np.all(
            np.isfinite(
                fitted_parameters
            )
        )
    )

    parameters_are_physical = bool(
        fit_result.amplitude >= 0.0
        and fit_result.lifetime > 0.0
        and fit_result.background >= 0.0
    )

    valid_fit = bool(
        fit_result.success
        and parameters_are_finite
        and parameters_are_physical
    )

    bin_width_ns = float(
        time_array[1]
        - time_array[0]
    )

    boundary_hit = _fit_hits_boundary(
        amplitude=fit_result.amplitude,
        lifetime_ns=fit_result.lifetime,
        background=fit_result.background,
        temporal_shift_ns=fit_result.temporal_shift,
        temporal_shift_bounds=temporal_shift_bounds,
        bin_width_ns=bin_width_ns,
    )

    poisson_nll = (
        poisson_negative_log_likelihood(
            observed=counts_array,
            expected=fit_result.fitted_curve,
        )
    )

    try:
        deviance_residuals = (
            calculate_poisson_deviance_residuals(
                observed=counts_array,
                expected=fit_result.fitted_curve,
            )
        )

        poisson_deviance = float(
            np.sum(
                deviance_residuals**2
            )
        )

    except ValueError:
        poisson_deviance = np.nan

    failure_reason: str | None = None

    if not fit_result.success:
        failure_reason = (
            "optimizer_unsuccessful"
        )

    elif not parameters_are_finite:
        failure_reason = (
            "non_finite_parameters"
        )

    elif not parameters_are_physical:
        failure_reason = (
            "non_physical_parameters"
        )

    return ReconvolutionCurveResult(
        initial_amplitude=initial_guess.amplitude,
        initial_lifetime_ns=initial_guess.lifetime_ns,
        initial_background=initial_guess.background,
        initial_temporal_shift_ns=(
            initial_guess.temporal_shift_ns
        ),
        fitted_amplitude=fit_result.amplitude,
        fitted_lifetime_ns=fit_result.lifetime,
        fitted_background=fit_result.background,
        fitted_temporal_shift_ns=(
            fit_result.temporal_shift
        ),
        optimizer_success=fit_result.success,
        valid_fit=valid_fit,
        boundary_hit=boundary_hit,
        poisson_nll=float(poisson_nll),
        poisson_deviance=poisson_deviance,
        runtime_ms=float(runtime_ms),
        failure_reason=failure_reason,
        exception_message=None,
    )


def evaluate_reconvolution_benchmark(
    *,
    time: ArrayLike,
    X_histograms: ArrayLike,
    y_true: ArrayLike,
    metadata: pd.DataFrame,
    irf: ArrayLike,
    temporal_shift_bounds: tuple[float, float],
    objective: str = "poisson",
    background_fraction: float = 0.10,
) -> ReconvolutionBenchmarkResult:
    """Evaluate reconvolution fitting on a batch of TCSPC histograms."""

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    X_histograms_array = np.asarray(
        X_histograms,
    )

    y_true_array = np.asarray(
        y_true,
        dtype=np.float64,
    )

    irf_array = np.asarray(
        irf,
        dtype=np.float64,
    )

    if X_histograms_array.ndim != 2:
        raise ValueError(
            "X_histograms must be two-dimensional."
        )

    n_samples, n_time_bins = (
        X_histograms_array.shape
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    if time_array.size != n_time_bins:
        raise ValueError(
            "Each histogram must match the time axis."
        )

    if y_true_array.shape != (n_samples,):
        raise ValueError(
            "y_true must contain one lifetime per histogram."
        )

    if metadata.shape[0] != n_samples:
        raise ValueError(
            "metadata must contain one row per histogram."
        )

    if irf_array.shape != time_array.shape:
        raise ValueError(
            "irf must have the same shape as time."
        )

    if not np.all(
        np.isfinite(y_true_array)
    ):
        raise ValueError(
            "y_true must contain only finite values."
        )

    if np.any(
        y_true_array <= 0.0
    ):
        raise ValueError(
            "True lifetimes must be positive."
        )

    per_curve_rows: list[
        dict[str, object]
    ] = []

    for sample_index in range(
        n_samples
    ):
        counts = X_histograms_array[
            sample_index
        ]

        curve_result = (
            fit_single_reconvolution_curve(
                time=time_array,
                counts=counts,
                irf=irf_array,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective=objective,
                background_fraction=(
                    background_fraction
                ),
            )
        )

        true_lifetime_ns = float(
            y_true_array[
                sample_index
            ]
        )

        fitted_lifetime_ns = (
            curve_result.fitted_lifetime_ns
        )

        if np.isfinite(
            fitted_lifetime_ns
        ):
            error_ns = (
                fitted_lifetime_ns
                - true_lifetime_ns
            )

            absolute_error_ns = abs(
                error_ns
            )

            relative_error = (
                absolute_error_ns
                / true_lifetime_ns
            )

        else:
            error_ns = np.nan
            absolute_error_ns = np.nan
            relative_error = np.nan

        metadata_row = (
            metadata.iloc[
                sample_index
            ]
            .to_dict()
        )

        row = {
            **metadata_row,
            "true_lifetime_ns": (
                true_lifetime_ns
            ),
            "fitted_lifetime_ns": (
                fitted_lifetime_ns
            ),
            "error_ns": float(
                error_ns
            ),
            "absolute_error_ns": float(
                absolute_error_ns
            ),
            "relative_error": float(
                relative_error
            ),
            "initial_amplitude": (
                curve_result.initial_amplitude
            ),
            "initial_lifetime_ns": (
                curve_result.initial_lifetime_ns
            ),
            "initial_background": (
                curve_result.initial_background
            ),
            "initial_temporal_shift_ns": (
                curve_result.initial_temporal_shift_ns
            ),
            "fitted_amplitude": (
                curve_result.fitted_amplitude
            ),
            "fitted_background": (
                curve_result.fitted_background
            ),
            "fitted_temporal_shift_ns": (
                curve_result.fitted_temporal_shift_ns
            ),
            "optimizer_success": (
                curve_result.optimizer_success
            ),
            "valid_fit": (
                curve_result.valid_fit
            ),
            "boundary_hit": (
                curve_result.boundary_hit
            ),
            "poisson_nll": (
                curve_result.poisson_nll
            ),
            "poisson_deviance": (
                curve_result.poisson_deviance
            ),
            "runtime_ms": (
                curve_result.runtime_ms
            ),
            "failure_reason": (
                curve_result.failure_reason
            ),
            "exception_message": (
                curve_result.exception_message
            ),
        }

        per_curve_rows.append(
            row
        )

    per_curve = pd.DataFrame(
        per_curve_rows
    )

    valid_mask = (
        per_curve[
            "valid_fit"
        ].to_numpy(
            dtype=bool
        )
    )

    n_successful_fits = int(
        np.sum(
            valid_mask
        )
    )

    n_failed_fits = (
        n_samples
        - n_successful_fits
    )

    success_rate = (
        n_successful_fits
        / n_samples
    )

    failure_rate = (
        n_failed_fits
        / n_samples
    )

    if n_successful_fits > 0:
        valid_errors = (
            per_curve.loc[
                valid_mask,
                "error_ns",
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        valid_absolute_errors = np.abs(
            valid_errors
        )

        mae_valid_ns = float(
            np.mean(
                valid_absolute_errors
            )
        )

        median_absolute_error_valid_ns = float(
            np.median(
                valid_absolute_errors
            )
        )

        rmse_valid_ns = float(
            np.sqrt(
                np.mean(
                    valid_errors**2
                )
            )
        )

    else:
        mae_valid_ns = np.nan
        median_absolute_error_valid_ns = np.nan
        rmse_valid_ns = np.nan

    runtime_values = (
        per_curve[
            "runtime_ms"
        ].to_numpy(
            dtype=np.float64
        )
    )

    finite_runtime_values = (
        runtime_values[
            np.isfinite(
                runtime_values
            )
        ]
    )

    if finite_runtime_values.size > 0:
        mean_runtime_ms = float(
            np.mean(
                finite_runtime_values
            )
        )

        median_runtime_ms = float(
            np.median(
                finite_runtime_values
            )
        )

    else:
        mean_runtime_ms = np.nan
        median_runtime_ms = np.nan

    summary = ReconvolutionBenchmarkSummary(
        n_samples=n_samples,
        n_successful_fits=n_successful_fits,
        n_failed_fits=n_failed_fits,
        success_rate=float(
            success_rate
        ),
        failure_rate=float(
            failure_rate
        ),
        mae_valid_ns=mae_valid_ns,
        median_absolute_error_valid_ns=(
            median_absolute_error_valid_ns
        ),
        rmse_valid_ns=rmse_valid_ns,
        mean_runtime_ms=mean_runtime_ms,
        median_runtime_ms=(
            median_runtime_ms
        ),
    )

    return ReconvolutionBenchmarkResult(
        per_curve=per_curve,
        summary=summary,
    )
