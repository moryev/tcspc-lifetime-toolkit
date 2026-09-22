"""Classical uncertainty methods for TCSPC lifetime estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from statistics import NormalDist

from tcspc_toolkit.classical_evaluation import (
    ReconvolutionCurveResult,
    fit_single_reconvolution_curve,
)
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)
from tcspc_toolkit.simulation import (
    sample_photon_counts,
    simulate_irf_convolved_histogram,
)
from tcspc_toolkit.fitting import (
    ReconvolutionFitResult,
    _reconvolution_model,
)
from tcspc_toolkit.uncertainty_evaluation import (
    IntervalEvaluationMetrics,
    PredictionIntervalResult,
    UncertaintyMethodDefinition,
    UncertaintyOutputKind,
    evaluate_prediction_intervals,
)


POISSON_LOCAL_COVARIANCE_METHOD_ID = "covariance"

DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP = 1e-4
DEFAULT_INFORMATION_CONDITION_LIMIT = 1e12


PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID = (
    "parametric_bootstrap"
)

REPEATED_POISSON_REFERENCE_METHOD_ID = (
    "repeated_poisson_realizations"
)

DEFAULT_CLASSICAL_BOOTSTRAP_REPLICATES = 200
DEFAULT_CLASSICAL_NOMINAL_COVERAGE = 0.90


CLASSICAL_UNCERTAINTY_METHODS = {
    POISSON_LOCAL_COVARIANCE_METHOD_ID: (
        UncertaintyMethodDefinition(
            method_id=(
                POISSON_LOCAL_COVARIANCE_METHOD_ID
            ),
            output_kind=(
                UncertaintyOutputKind.PREDICTION_INTERVAL
            ),
            interpretation=(
                "Local fitted-parameter uncertainty derived from "
                "the expected Poisson Fisher information of the "
                "reconvolution model."
            ),
            calibration_scope=(
                "Meaningful only when the fitted physical model is "
                "adequate, the optimum is away from parameter bounds, "
                "and the local information matrix is sufficiently "
                "well conditioned."
            ),
        )
    ),
    PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID: (
        UncertaintyMethodDefinition(
            method_id=(
                PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID
            ),
            output_kind=(
                UncertaintyOutputKind.PREDICTION_INTERVAL
            ),
            interpretation=(
                "Sampling variability obtained by resimulating "
                "measurements from the fitted model and refitting them."
            ),
            calibration_scope=(
                "Conditional on the fitted physical model; it does not "
                "capture model misspecification such as Test F."
            ),
        )
    ),
    REPEATED_POISSON_REFERENCE_METHOD_ID: (
        UncertaintyMethodDefinition(
            method_id=(
                REPEATED_POISSON_REFERENCE_METHOD_ID
            ),
            output_kind=(
                UncertaintyOutputKind.EMPIRICAL_REFERENCE
            ),
            interpretation=(
                "Empirical estimator variability over repeated Poisson "
                "measurements generated from known physics."
            ),
            calibration_scope=(
                "Reference benchmark for repeated-measurement "
                "variability, not a per-curve uncertainty estimate "
                "available in deployment."
            ),
        )
    ),
}


@dataclass(frozen=True)
class PoissonLocalCovarianceResult:
    """Local covariance diagnostics for one Poisson reconvolution fit."""

    covariance_matrix: NDArray[np.float64]

    amplitude_std: float
    lifetime_std: float
    background_std: float
    temporal_shift_std: float

    covariance_valid: bool
    condition_number: float
    information_rank: int
    boundary_hit: bool

    failure_reason: str | None


@dataclass(frozen=True)
class ParametricPoissonBootstrapResult:
    """Parametric Poisson-bootstrap uncertainty for one fitted histogram."""

    source_lifetime_ns: float

    lifetime_samples_ns: NDArray[np.float64]

    bootstrap_std_ns: float
    bootstrap_median_ns: float

    lower_ns: float
    upper_ns: float
    nominal_coverage: float

    n_resamples: int
    n_successful_fits: int
    n_failed_fits: int
    n_boundary_hits: int

    fit_failure_rate: float
    boundary_hit_rate: float

    bootstrap_valid: bool
    failure_reason: str | None


def estimate_parametric_poisson_bootstrap(
    *,
    time: ArrayLike,
    irf: ArrayLike,
    fit_result: ReconvolutionFitResult,
    temporal_shift_bounds: tuple[float, float],
    rng: np.random.Generator,
    n_resamples: int = (
        DEFAULT_CLASSICAL_BOOTSTRAP_REPLICATES
    ),
    nominal_coverage: float = (
        DEFAULT_CLASSICAL_NOMINAL_COVERAGE
    ),
    background_fraction: float = 0.10,
) -> ParametricPoissonBootstrapResult:
    """Estimate lifetime uncertainty by parametric Poisson bootstrap.

    Each bootstrap histogram is generated independently from the
    fitted expected-count curve,

        k_i* ~ Poisson(mu_hat_i),

    and is then refitted with the same Poisson reconvolution pipeline
    used by the classical benchmark.

    Time bins themselves are never resampled.
    """

    if (
        isinstance(
            n_resamples,
            (bool, np.bool_),
        )
        or not isinstance(
            n_resamples,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "n_resamples must be an integer."
        )

    if n_resamples < 2:
        raise ValueError(
            "n_resamples must be at least 2."
        )

    if not (
        0.0
        < nominal_coverage
        < 1.0
    ):
        raise ValueError(
            "nominal_coverage must lie strictly "
            "between 0 and 1."
        )

    if not fit_result.success:
        raise ValueError(
            "fit_result must represent a successful fit."
        )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    irf_array = np.asarray(
        irf,
        dtype=np.float64,
    )

    expected_counts = np.asarray(
        fit_result.fitted_curve,
        dtype=np.float64,
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    if irf_array.ndim != 1:
        raise ValueError(
            "irf must be one-dimensional."
        )

    if expected_counts.ndim != 1:
        raise ValueError(
            "fit_result.fitted_curve must be "
            "one-dimensional."
        )

    if not (
        time_array.shape
        == irf_array.shape
        == expected_counts.shape
    ):
        raise ValueError(
            "time, irf, and fitted_curve must "
            "have the same shape."
        )

    fitted_parameters = np.asarray(
        [
            fit_result.amplitude,
            fit_result.lifetime,
            fit_result.background,
            fit_result.temporal_shift,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            fitted_parameters
        )
    ):
        raise ValueError(
            "fit_result parameters must be finite."
        )

    if (
        fit_result.amplitude < 0.0
        or fit_result.lifetime <= 0.0
        or fit_result.background < 0.0
    ):
        raise ValueError(
            "fit_result parameters must be physical."
        )

    if (
        not np.all(
            np.isfinite(
                expected_counts
            )
        )
        or np.any(
            expected_counts < 0.0
        )
    ):
        raise ValueError(
            "fitted expected counts must be finite "
            "and non-negative."
        )

    lifetime_samples = np.full(
        n_resamples,
        np.nan,
        dtype=np.float64,
    )

    boundary_hits = np.zeros(
        n_resamples,
        dtype=bool,
    )

    for bootstrap_index in range(
        n_resamples
    ):
        bootstrap_counts = (
            sample_photon_counts(
                expected_counts=expected_counts,
                rng=rng,
            )
        )

        bootstrap_fit = (
            fit_single_reconvolution_curve(
                time=time_array,
                counts=bootstrap_counts,
                irf=irf_array,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective="poisson",
                background_fraction=(
                    background_fraction
                ),
            )
        )

        if not bootstrap_fit.valid_fit:
            continue

        lifetime_samples[
            bootstrap_index
        ] = (
            bootstrap_fit.fitted_lifetime_ns
        )

        boundary_hits[
            bootstrap_index
        ] = (
            bootstrap_fit.boundary_hit
        )

    successful_mask = np.isfinite(
        lifetime_samples
    )

    n_successful = int(
        np.count_nonzero(
            successful_mask
        )
    )

    n_failed = (
        n_resamples
        - n_successful
    )

    n_boundary_hits = int(
        np.count_nonzero(
            boundary_hits
            & successful_mask
        )
    )

    fit_failure_rate = (
        n_failed
        / n_resamples
    )

    if n_successful > 0:
        boundary_hit_rate = (
            n_boundary_hits
            / n_successful
        )
    else:
        boundary_hit_rate = np.nan

    if n_successful < 2:
        return (
            ParametricPoissonBootstrapResult(
                source_lifetime_ns=float(
                    fit_result.lifetime
                ),
                lifetime_samples_ns=(
                    lifetime_samples
                ),
                bootstrap_std_ns=np.nan,
                bootstrap_median_ns=np.nan,
                lower_ns=np.nan,
                upper_ns=np.nan,
                nominal_coverage=float(
                    nominal_coverage
                ),
                n_resamples=n_resamples,
                n_successful_fits=(
                    n_successful
                ),
                n_failed_fits=n_failed,
                n_boundary_hits=(
                    n_boundary_hits
                ),
                fit_failure_rate=float(
                    fit_failure_rate
                ),
                boundary_hit_rate=float(
                    boundary_hit_rate
                ),
                bootstrap_valid=False,
                failure_reason=(
                    "fewer_than_two_successful_refits"
                ),
            )
        )

    successful_lifetimes = (
        lifetime_samples[
            successful_mask
        ]
    )

    alpha = (
        1.0
        - nominal_coverage
    )

    lower_quantile = (
        alpha / 2.0
    )

    upper_quantile = (
        1.0
        - alpha / 2.0
    )

    (
        lower_ns,
        bootstrap_median_ns,
        upper_ns,
    ) = np.quantile(
        successful_lifetimes,
        [
            lower_quantile,
            0.5,
            upper_quantile,
        ],
    )

    bootstrap_std_ns = float(
        np.std(
            successful_lifetimes,
            ddof=1,
        )
    )

    return ParametricPoissonBootstrapResult(
        source_lifetime_ns=float(
            fit_result.lifetime
        ),
        lifetime_samples_ns=(
            lifetime_samples
        ),
        bootstrap_std_ns=(
            bootstrap_std_ns
        ),
        bootstrap_median_ns=float(
            bootstrap_median_ns
        ),
        lower_ns=float(
            lower_ns
        ),
        upper_ns=float(
            upper_ns
        ),
        nominal_coverage=float(
            nominal_coverage
        ),
        n_resamples=n_resamples,
        n_successful_fits=(
            n_successful
        ),
        n_failed_fits=n_failed,
        n_boundary_hits=(
            n_boundary_hits
        ),
        fit_failure_rate=float(
            fit_failure_rate
        ),
        boundary_hit_rate=float(
            boundary_hit_rate
        ),
        bootstrap_valid=True,
        failure_reason=None,
    )


@dataclass(frozen=True)
class RepeatedPoissonUncertaintyResult:
    """Classical uncertainty versus repeated-measurement reference."""

    true_lifetime_ns: float
    signal_photon_count: int
    background_per_bin: float
    irf_fwhm_ns: float
    irf_shift_ns: float

    lifetime_estimates_ns: NDArray[np.float64]

    empirical_bias_ns: float
    empirical_std_ns: float
    empirical_rmse_ns: float

    fit_failure_rate: float
    boundary_hit_rate: float

    covariance_std_ns: NDArray[np.float64]
    covariance_intervals: PredictionIntervalResult
    covariance_metrics: IntervalEvaluationMetrics

    mean_covariance_std_ns: float
    covariance_to_empirical_std_ratio: float

    bootstrap_std_ns: NDArray[np.float64]
    bootstrap_intervals: PredictionIntervalResult
    bootstrap_metrics: IntervalEvaluationMetrics

    mean_bootstrap_std_ns: float
    bootstrap_to_empirical_std_ratio: float

    bootstrap_fit_failure_rates: NDArray[np.float64]
    mean_bootstrap_fit_failure_rate: float


def _reconstruct_reconvolution_fit_result(
    *,
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    curve_result: ReconvolutionCurveResult,
) -> ReconvolutionFitResult:
    """Reconstruct the physical fit result from benchmark diagnostics."""

    if not curve_result.valid_fit:
        raise ValueError(
            "curve_result must represent a valid fit."
        )

    fitted_curve = _reconvolution_model(
        time=time,
        irf=irf,
        amplitude=(
            curve_result.fitted_amplitude
        ),
        lifetime=(
            curve_result.fitted_lifetime_ns
        ),
        background=(
            curve_result.fitted_background
        ),
        temporal_shift=(
            curve_result.fitted_temporal_shift_ns
        ),
    )

    return ReconvolutionFitResult(
        amplitude=(
            curve_result.fitted_amplitude
        ),
        lifetime=(
            curve_result.fitted_lifetime_ns
        ),
        background=(
            curve_result.fitted_background
        ),
        temporal_shift=(
            curve_result.fitted_temporal_shift_ns
        ),
        fitted_curve=fitted_curve,
        success=(
            curve_result.optimizer_success
        ),
    )


def evaluate_repeated_poisson_uncertainty(
    *,
    time: ArrayLike,
    true_lifetime_ns: float,
    signal_photon_count: int,
    background_per_bin: float,
    irf_centre_ns: float,
    irf_fwhm_ns: float,
    irf_shift_ns: float,
    temporal_shift_bounds: tuple[float, float],
    n_repeats: int,
    n_bootstrap_resamples: int,
    rng: np.random.Generator,
    nominal_coverage: float = (
        DEFAULT_CLASSICAL_NOMINAL_COVERAGE
    ),
    background_fraction: float = 0.10,
) -> RepeatedPoissonUncertaintyResult:
    """Compare classical uncertainty with repeated Poisson measurements.

    The physical condition is held fixed while independent Poisson
    realizations are generated. For every successful reconvolution fit,
    the function evaluates:

    - local Poisson/Fisher covariance uncertainty;
    - parametric Poisson-bootstrap uncertainty;
    - empirical repeated-measurement estimator error.
    """

    if (
        isinstance(
            n_repeats,
            (bool, np.bool_),
        )
        or not isinstance(
            n_repeats,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "n_repeats must be an integer."
        )

    if n_repeats < 2:
        raise ValueError(
            "n_repeats must be at least 2."
        )

    if n_bootstrap_resamples < 2:
        raise ValueError(
            "n_bootstrap_resamples must be "
            "at least 2."
        )

    if not (
        0.0
        < nominal_coverage
        < 1.0
    ):
        raise ValueError(
            "nominal_coverage must lie strictly "
            "between 0 and 1."
        )

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    base_irf = generate_gaussian_irf(
        time=time_array,
        centre=irf_centre_ns,
        fwhm=irf_fwhm_ns,
    )

    base_irf = normalize_irf(
        time=time_array,
        irf=base_irf,
    )

    lifetime_estimates = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    fit_boundary_hits = np.zeros(
        n_repeats,
        dtype=bool,
    )

    covariance_std = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    covariance_lower = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    covariance_upper = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    bootstrap_std = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    bootstrap_lower = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    bootstrap_upper = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    bootstrap_failure_rates = np.full(
        n_repeats,
        np.nan,
        dtype=np.float64,
    )

    normal_quantile = (
        NormalDist().inv_cdf(
            0.5
            + nominal_coverage / 2.0
        )
    )

    for repeat_index in range(
        n_repeats
    ):
        measured_counts, _ = (
            simulate_irf_convolved_histogram(
                time=time_array,
                lifetime_ns=(
                    true_lifetime_ns
                ),
                signal_photon_count=(
                    signal_photon_count
                ),
                background_per_bin=(
                    background_per_bin
                ),
                irf_centre_ns=(
                    irf_centre_ns
                ),
                irf_fwhm_ns=(
                    irf_fwhm_ns
                ),
                irf_shift_ns=(
                    irf_shift_ns
                ),
                rng=rng,
            )
        )

        curve_result = (
            fit_single_reconvolution_curve(
                time=time_array,
                counts=measured_counts,
                irf=base_irf,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective="poisson",
                background_fraction=(
                    background_fraction
                ),
            )
        )

        if not curve_result.valid_fit:
            continue

        fitted_lifetime = (
            curve_result.fitted_lifetime_ns
        )

        lifetime_estimates[
            repeat_index
        ] = fitted_lifetime

        fit_boundary_hits[
            repeat_index
        ] = curve_result.boundary_hit

        fit_result = (
            _reconstruct_reconvolution_fit_result(
                time=time_array,
                irf=base_irf,
                curve_result=curve_result,
            )
        )

        covariance_result = (
            estimate_poisson_reconvolution_local_covariance(
                time=time_array,
                irf=base_irf,
                fit_result=fit_result,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
            )
        )

        if covariance_result.covariance_valid:
            lifetime_std = (
                covariance_result.lifetime_std
            )

            covariance_std[
                repeat_index
            ] = lifetime_std

            covariance_lower[
                repeat_index
            ] = (
                fitted_lifetime
                - normal_quantile
                * lifetime_std
            )

            covariance_upper[
                repeat_index
            ] = (
                fitted_lifetime
                + normal_quantile
                * lifetime_std
            )

        bootstrap_result = (
            estimate_parametric_poisson_bootstrap(
                time=time_array,
                irf=base_irf,
                fit_result=fit_result,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                rng=rng,
                n_resamples=(
                    n_bootstrap_resamples
                ),
                nominal_coverage=(
                    nominal_coverage
                ),
                background_fraction=(
                    background_fraction
                ),
            )
        )

        bootstrap_failure_rates[
            repeat_index
        ] = (
            bootstrap_result.fit_failure_rate
        )

        if bootstrap_result.bootstrap_valid:
            bootstrap_std[
                repeat_index
            ] = (
                bootstrap_result.bootstrap_std_ns
            )

            bootstrap_lower[
                repeat_index
            ] = (
                bootstrap_result.lower_ns
            )

            bootstrap_upper[
                repeat_index
            ] = (
                bootstrap_result.upper_ns
            )

    valid_fit_mask = np.isfinite(
        lifetime_estimates
    )

    n_valid_fits = int(
        np.count_nonzero(
            valid_fit_mask
        )
    )

    fit_failure_rate = (
        1.0
        - n_valid_fits / n_repeats
    )

    if n_valid_fits > 0:
        boundary_hit_rate = float(
            np.mean(
                fit_boundary_hits[
                    valid_fit_mask
                ]
            )
        )
    else:
        boundary_hit_rate = np.nan

    if n_valid_fits >= 2:
        valid_estimates = (
            lifetime_estimates[
                valid_fit_mask
            ]
        )

        errors = (
            valid_estimates
            - true_lifetime_ns
        )

        empirical_bias_ns = float(
            np.mean(
                errors
            )
        )

        empirical_std_ns = float(
            np.std(
                valid_estimates,
                ddof=1,
            )
        )

        empirical_rmse_ns = float(
            np.sqrt(
                np.mean(
                    errors**2
                )
            )
        )

    else:
        empirical_bias_ns = np.nan
        empirical_std_ns = np.nan
        empirical_rmse_ns = np.nan

    true_lifetimes = np.full(
        n_repeats,
        true_lifetime_ns,
        dtype=np.float64,
    )

    covariance_intervals = (
        PredictionIntervalResult(
            prediction=(
                lifetime_estimates.copy()
            ),
            lower=covariance_lower,
            upper=covariance_upper,
            nominal_coverage=(
                nominal_coverage
            ),
            method_id=(
                POISSON_LOCAL_COVARIANCE_METHOD_ID
            ),
        )
    )

    covariance_metrics = (
        evaluate_prediction_intervals(
            true_lifetimes,
            covariance_intervals,
        )
    )

    bootstrap_intervals = (
        PredictionIntervalResult(
            prediction=(
                lifetime_estimates.copy()
            ),
            lower=bootstrap_lower,
            upper=bootstrap_upper,
            nominal_coverage=(
                nominal_coverage
            ),
            method_id=(
                PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID
            ),
        )
    )

    bootstrap_metrics = (
        evaluate_prediction_intervals(
            true_lifetimes,
            bootstrap_intervals,
        )
    )

    finite_covariance_std = (
        covariance_std[
            np.isfinite(
                covariance_std
            )
        ]
    )

    if finite_covariance_std.size > 0:
        mean_covariance_std_ns = float(
            np.mean(
                finite_covariance_std
            )
        )
    else:
        mean_covariance_std_ns = np.nan

    finite_bootstrap_std = (
        bootstrap_std[
            np.isfinite(
                bootstrap_std
            )
        ]
    )

    if finite_bootstrap_std.size > 0:
        mean_bootstrap_std_ns = float(
            np.mean(
                finite_bootstrap_std
            )
        )
    else:
        mean_bootstrap_std_ns = np.nan

    if (
        np.isfinite(
            empirical_std_ns
        )
        and empirical_std_ns > 0.0
    ):
        covariance_ratio = (
            mean_covariance_std_ns
            / empirical_std_ns
        )

        bootstrap_ratio = (
            mean_bootstrap_std_ns
            / empirical_std_ns
        )

    else:
        covariance_ratio = np.nan
        bootstrap_ratio = np.nan

    finite_bootstrap_failure_rates = (
        bootstrap_failure_rates[
            np.isfinite(
                bootstrap_failure_rates
            )
        ]
    )

    if (
        finite_bootstrap_failure_rates.size
        > 0
    ):
        mean_bootstrap_fit_failure_rate = (
            float(
                np.mean(
                    finite_bootstrap_failure_rates
                )
            )
        )
    else:
        mean_bootstrap_fit_failure_rate = (
            np.nan
        )

    return RepeatedPoissonUncertaintyResult(
        true_lifetime_ns=float(
            true_lifetime_ns
        ),
        signal_photon_count=int(
            signal_photon_count
        ),
        background_per_bin=float(
            background_per_bin
        ),
        irf_fwhm_ns=float(
            irf_fwhm_ns
        ),
        irf_shift_ns=float(
            irf_shift_ns
        ),
        lifetime_estimates_ns=(
            lifetime_estimates
        ),
        empirical_bias_ns=(
            empirical_bias_ns
        ),
        empirical_std_ns=(
            empirical_std_ns
        ),
        empirical_rmse_ns=(
            empirical_rmse_ns
        ),
        fit_failure_rate=float(
            fit_failure_rate
        ),
        boundary_hit_rate=float(
            boundary_hit_rate
        ),
        covariance_std_ns=(
            covariance_std
        ),
        covariance_intervals=(
            covariance_intervals
        ),
        covariance_metrics=(
            covariance_metrics
        ),
        mean_covariance_std_ns=(
            mean_covariance_std_ns
        ),
        covariance_to_empirical_std_ratio=float(
            covariance_ratio
        ),
        bootstrap_std_ns=(
            bootstrap_std
        ),
        bootstrap_intervals=(
            bootstrap_intervals
        ),
        bootstrap_metrics=(
            bootstrap_metrics
        ),
        mean_bootstrap_std_ns=(
            mean_bootstrap_std_ns
        ),
        bootstrap_to_empirical_std_ratio=float(
            bootstrap_ratio
        ),
        bootstrap_fit_failure_rates=(
            bootstrap_failure_rates
        ),
        mean_bootstrap_fit_failure_rate=(
            mean_bootstrap_fit_failure_rate
        ),
    )


def _invalid_local_covariance_result(
    *,
    failure_reason: str,
    condition_number: float = np.nan,
    information_rank: int = 0,
    boundary_hit: bool = False,
) -> PoissonLocalCovarianceResult:
    """Construct a failure-aware covariance result."""

    return PoissonLocalCovarianceResult(
        covariance_matrix=np.full(
            (4, 4),
            np.nan,
            dtype=np.float64,
        ),
        amplitude_std=np.nan,
        lifetime_std=np.nan,
        background_std=np.nan,
        temporal_shift_std=np.nan,
        covariance_valid=False,
        condition_number=float(
            condition_number
        ),
        information_rank=int(
            information_rank
        ),
        boundary_hit=boundary_hit,
        failure_reason=failure_reason,
    )


def _poisson_fit_hits_boundary(
    *,
    parameters: NDArray[np.float64],
    temporal_shift_bounds: tuple[float, float],
    bin_width: float,
) -> bool:
    """Return whether the fitted solution lies effectively on a bound."""

    (
        amplitude,
        lifetime,
        background,
        temporal_shift,
    ) = parameters

    shift_lower, shift_upper = (
        temporal_shift_bounds
    )

    shift_tolerance = (
        0.5 * bin_width
    )

    amplitude_at_lower_bound = np.isclose(
        amplitude,
        0.0,
        rtol=0.0,
        atol=1e-10,
    )

    lifetime_at_lower_bound = np.isclose(
        lifetime,
        1e-12,
        rtol=0.0,
        atol=1e-10,
    )

    background_at_lower_bound = (
        background <= 1e-10
    )

    shift_at_lower_bound = np.isclose(
        temporal_shift,
        shift_lower,
        rtol=0.0,
        atol=shift_tolerance,
    )

    shift_at_upper_bound = np.isclose(
        temporal_shift,
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


def _parameter_scales(
    *,
    parameters: NDArray[np.float64],
    bin_width: float,
) -> NDArray[np.float64]:
    """Return physical scales used for dimensionless conditioning."""

    (
        amplitude,
        lifetime,
        background,
        temporal_shift,
    ) = parameters

    return np.asarray(
        [
            max(
                abs(amplitude),
                1.0,
            ),
            max(
                abs(lifetime),
                bin_width,
            ),
            max(
                abs(background),
                1.0,
            ),
            max(
                abs(temporal_shift),
                bin_width,
            ),
        ],
        dtype=np.float64,
    )


def _expected_count_jacobian(
    *,
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    parameters: NDArray[np.float64],
    parameter_scales: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    relative_step: float,
) -> NDArray[np.float64]:
    """Numerically differentiate expected counts with respect to parameters."""

    n_bins = time.size
    n_parameters = parameters.size

    jacobian = np.empty(
        (
            n_bins,
            n_parameters,
        ),
        dtype=np.float64,
    )

    def evaluate(
        candidate_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        (
            amplitude,
            lifetime,
            background,
            temporal_shift,
        ) = candidate_parameters

        return _reconvolution_model(
            time=time,
            irf=irf,
            amplitude=float(
                amplitude
            ),
            lifetime=float(
                lifetime
            ),
            background=float(
                background
            ),
            temporal_shift=float(
                temporal_shift
            ),
        )

    base_expected = evaluate(
        parameters
    )

    for parameter_index in range(
        n_parameters
    ):
        step = (
            relative_step
            * parameter_scales[
                parameter_index
            ]
        )

        minimum_step = (
            np.sqrt(
                np.finfo(
                    np.float64
                ).eps
            )
            * parameter_scales[
                parameter_index
            ]
        )

        step = max(
            step,
            minimum_step,
        )

        plus_parameters = (
            parameters.copy()
        )

        minus_parameters = (
            parameters.copy()
        )

        plus_parameters[
            parameter_index
        ] += step

        minus_parameters[
            parameter_index
        ] -= step

        can_step_plus = (
            plus_parameters[
                parameter_index
            ]
            < upper_bounds[
                parameter_index
            ]
        )

        can_step_minus = (
            minus_parameters[
                parameter_index
            ]
            > lower_bounds[
                parameter_index
            ]
        )

        if (
            can_step_plus
            and can_step_minus
        ):
            plus_expected = evaluate(
                plus_parameters
            )

            minus_expected = evaluate(
                minus_parameters
            )

            derivative = (
                plus_expected
                - minus_expected
            ) / (
                2.0 * step
            )

        elif can_step_plus:
            plus_expected = evaluate(
                plus_parameters
            )

            derivative = (
                plus_expected
                - base_expected
            ) / step

        elif can_step_minus:
            minus_expected = evaluate(
                minus_parameters
            )

            derivative = (
                base_expected
                - minus_expected
            ) / step

        else:
            derivative = np.full(
                n_bins,
                np.nan,
                dtype=np.float64,
            )

        jacobian[
            :,
            parameter_index,
        ] = derivative

    return jacobian


def estimate_poisson_reconvolution_local_covariance(
    *,
    time: ArrayLike,
    irf: ArrayLike,
    fit_result: ReconvolutionFitResult,
    temporal_shift_bounds: tuple[float, float],
    relative_step: float = (
        DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP
    ),
    condition_number_limit: float = (
        DEFAULT_INFORMATION_CONDITION_LIMIT
    ),
) -> PoissonLocalCovarianceResult:
    """Estimate local parameter covariance for a Poisson reconvolution fit.

    The approximation uses the expected Poisson Fisher information

        I(theta) = J.T @ diag(1 / mu) @ J,

    where J is the Jacobian of the expected TCSPC counts with respect
    to amplitude, lifetime, background, and temporal shift.

    The information matrix is evaluated in dimensionless scaled
    parameter coordinates for numerical conditioning. The resulting
    covariance is then transformed back to physical parameter units.

    No covariance is reported when the fit failed, lies on a parameter
    bound, produces invalid expected counts, or has singular / badly
    conditioned local information.
    """

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    irf_array = np.asarray(
        irf,
        dtype=np.float64,
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    if irf_array.ndim != 1:
        raise ValueError(
            "irf must be one-dimensional."
        )

    if time_array.shape != irf_array.shape:
        raise ValueError(
            "time and irf must have the same shape."
        )

    if time_array.size < 5:
        raise ValueError(
            "at least five time bins are required."
        )

    if not np.all(
        np.isfinite(
            time_array
        )
    ):
        raise ValueError(
            "time must contain only finite values."
        )

    if not np.all(
        np.isfinite(
            irf_array
        )
    ):
        raise ValueError(
            "irf must contain only finite values."
        )

    if np.any(
        irf_array < 0.0
    ):
        raise ValueError(
            "irf must contain only non-negative values."
        )

    time_differences = np.diff(
        time_array
    )

    if np.any(
        time_differences <= 0.0
    ):
        raise ValueError(
            "time must be strictly increasing."
        )

    if not np.allclose(
        time_differences,
        time_differences[0],
    ):
        raise ValueError(
            "time must be uniformly spaced."
        )

    if (
        not np.isfinite(
            relative_step
        )
        or relative_step <= 0.0
    ):
        raise ValueError(
            "relative_step must be positive and finite."
        )

    if (
        not np.isfinite(
            condition_number_limit
        )
        or condition_number_limit < 1.0
    ):
        raise ValueError(
            "condition_number_limit must be finite "
            "and at least 1."
        )

    shift_lower, shift_upper = (
        temporal_shift_bounds
    )

    if not (
        np.isfinite(
            shift_lower
        )
        and np.isfinite(
            shift_upper
        )
    ):
        raise ValueError(
            "temporal shift bounds must be finite."
        )

    if shift_lower >= shift_upper:
        raise ValueError(
            "lower temporal shift bound must be "
            "smaller than upper temporal shift bound."
        )

    if not fit_result.success:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "fit_unsuccessful"
                ),
            )
        )

    parameters = np.asarray(
        [
            fit_result.amplitude,
            fit_result.lifetime,
            fit_result.background,
            fit_result.temporal_shift,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            parameters
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_finite_parameters"
                ),
            )
        )

    if (
        fit_result.amplitude < 0.0
        or fit_result.lifetime <= 0.0
        or fit_result.background <= 0.0
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_physical_parameters"
                ),
            )
        )

    bin_width = float(
        time_differences[0]
    )

    boundary_hit = (
        _poisson_fit_hits_boundary(
            parameters=parameters,
            temporal_shift_bounds=(
                temporal_shift_bounds
            ),
            bin_width=bin_width,
        )
    )

    if boundary_hit:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "parameter_at_bound"
                ),
                boundary_hit=True,
            )
        )

    expected_counts = (
        _reconvolution_model(
            time=time_array,
            irf=irf_array,
            amplitude=fit_result.amplitude,
            lifetime=fit_result.lifetime,
            background=fit_result.background,
            temporal_shift=(
                fit_result.temporal_shift
            ),
        )
    )

    if (
        not np.all(
            np.isfinite(
                expected_counts
            )
        )
        or np.any(
            expected_counts <= 0.0
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "invalid_expected_counts"
                ),
            )
        )

    lower_bounds = np.asarray(
        [
            0.0,
            1e-12,
            1e-12,
            shift_lower,
        ],
        dtype=np.float64,
    )

    upper_bounds = np.asarray(
        [
            np.inf,
            np.inf,
            np.inf,
            shift_upper,
        ],
        dtype=np.float64,
    )

    parameter_scales = (
        _parameter_scales(
            parameters=parameters,
            bin_width=bin_width,
        )
    )

    jacobian = (
        _expected_count_jacobian(
            time=time_array,
            irf=irf_array,
            parameters=parameters,
            parameter_scales=(
                parameter_scales
            ),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            relative_step=relative_step,
        )
    )

    if not np.all(
        np.isfinite(
            jacobian
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_finite_jacobian"
                ),
            )
        )

    weighted_jacobian = (
        jacobian
        / np.sqrt(
            expected_counts
        )[:, None]
    )

    information_matrix = (
        weighted_jacobian.T
        @ weighted_jacobian
    )

    scaled_information_matrix = (
        parameter_scales[:, None]
        * information_matrix
        * parameter_scales[None, :]
    )

    information_rank = int(
        np.linalg.matrix_rank(
            scaled_information_matrix
        )
    )

    n_parameters = (
        parameters.size
    )

    if information_rank < n_parameters:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "singular_information_matrix"
                ),
                condition_number=np.inf,
                information_rank=(
                    information_rank
                ),
            )
        )

    condition_number = float(
        np.linalg.cond(
            scaled_information_matrix
        )
    )

    if not np.isfinite(
        condition_number
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "singular_information_matrix"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    if (
        condition_number
        > condition_number_limit
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "ill_conditioned_information_matrix"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    try:
        scaled_covariance = (
            np.linalg.inv(
                scaled_information_matrix
            )
        )

    except np.linalg.LinAlgError:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "singular_information_matrix"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    covariance_matrix = (
        parameter_scales[:, None]
        * scaled_covariance
        * parameter_scales[None, :]
    )

    covariance_matrix = (
        0.5
        * (
            covariance_matrix
            + covariance_matrix.T
        )
    )

    if not np.all(
        np.isfinite(
            covariance_matrix
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_finite_covariance"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    parameter_variances = np.diag(
        covariance_matrix
    )

    if (
        not np.all(
            np.isfinite(
                parameter_variances
            )
        )
        or np.any(
            parameter_variances <= 0.0
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_positive_parameter_variance"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    parameter_standard_deviations = (
        np.sqrt(
            parameter_variances
        )
    )

    (
        amplitude_std,
        lifetime_std,
        background_std,
        temporal_shift_std,
    ) = parameter_standard_deviations

    return PoissonLocalCovarianceResult(
        covariance_matrix=(
            covariance_matrix
        ),
        amplitude_std=float(
            amplitude_std
        ),
        lifetime_std=float(
            lifetime_std
        ),
        background_std=float(
            background_std
        ),
        temporal_shift_std=float(
            temporal_shift_std
        ),
        covariance_valid=True,
        condition_number=(
            condition_number
        ),
        information_rank=(
            information_rank
        ),
        boundary_hit=False,
        failure_reason=None,
    )
