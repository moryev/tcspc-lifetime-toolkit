"""Paired, correctly specified Issue-4 uncertainty evaluation.

Each realization is sampled once and passed unchanged to classical Poisson
reconvolution and Bayesian raw-count inference. Intervals retain their own
inferential names. This module does not assess physical-model validity.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.bayesian import (
    BayesianPriorConfig,
    IRFModelRelation,
    fit_bayesian_monoexponential_reconvolution,
)
from tcspc_toolkit.bayesian_sampling import BayesianSamplingConfig, BayesianSamplingStatus
from tcspc_toolkit.classical_evaluation import fit_single_reconvolution_curve
from tcspc_toolkit.classical_uncertainty import (
    _reconstruct_reconvolution_fit_result,
    estimate_parametric_poisson_bootstrap,
    estimate_poisson_reconvolution_local_covariance,
)
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf_preparation import PreparedIRF
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.simulation import sample_photon_counts


@dataclass(frozen=True)
class BayesianClassicalCondition:
    """One synthetic matched-IRF condition; truth is not measurement metadata."""

    condition_id: str
    time_ns: NDArray[np.float64]
    generating_irf: PreparedIRF
    assumed_irf: PreparedIRF
    true_lifetime_ns: float
    signal_photon_count: int
    background_per_bin: float
    true_temporal_shift_ns: float
    n_repeats: int

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            raise ValueError("condition_id must be a nonempty string")
        if not isinstance(self.generating_irf, PreparedIRF) or not isinstance(
            self.assumed_irf, PreparedIRF
        ):
            raise TypeError("generating_irf and assumed_irf must be PreparedIRF objects")
        time_axis = np.asarray(self.time_ns, dtype=np.float64)
        if time_axis.ndim != 1 or time_axis.size < 5 or not np.all(np.isfinite(time_axis)):
            raise ValueError("time_ns must be a finite one-dimensional grid with at least five bins")
        if np.any(np.diff(time_axis) <= 0.0) or not np.allclose(
            np.diff(time_axis), np.diff(time_axis)[0]
        ):
            raise ValueError("time_ns must be uniformly spaced and increasing")
        if not np.array_equal(time_axis, self.generating_irf.time_ns) or not np.array_equal(
            time_axis, self.assumed_irf.time_ns
        ):
            raise ValueError("both prepared IRFs must use the condition time grid")
        if not np.array_equal(self.generating_irf.kernel, self.assumed_irf.kernel):
            raise ValueError("Stage 5 requires identical generating and assumed IRF kernels")
        if not math.isfinite(self.true_lifetime_ns) or self.true_lifetime_ns <= 0.0:
            raise ValueError("true_lifetime_ns must be finite and positive")
        if isinstance(self.signal_photon_count, (bool, np.bool_)) or not isinstance(
            self.signal_photon_count, (int, np.integer)
        ) or self.signal_photon_count <= 0:
            raise ValueError("signal_photon_count must be a positive integer")
        if not math.isfinite(self.background_per_bin) or self.background_per_bin < 0.0:
            raise ValueError("background_per_bin must be finite and non-negative")
        if not math.isfinite(self.true_temporal_shift_ns):
            raise ValueError("true_temporal_shift_ns must be finite")
        if isinstance(self.n_repeats, (bool, np.bool_)) or not isinstance(
            self.n_repeats, (int, np.integer)
        ) or self.n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer")
        copied = np.frombuffer(np.ascontiguousarray(time_axis).tobytes(), dtype=np.float64)
        object.__setattr__(self, "time_ns", copied)


@dataclass(frozen=True)
class BayesianClassicalEvaluationConfig:
    """Predeclared comparison policy; no prior is derived from observations."""

    profile: str
    prior_policy_id: str
    priors: BayesianPriorConfig
    sampling_config: BayesianSamplingConfig
    n_bootstrap_resamples: int
    nominal_interval_level: float
    temporal_shift_bounds_ns: tuple[float, float]
    random_seed: int
    background_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "scientific", "cost_pilot"}:
            raise ValueError("profile must be smoke, scientific, or cost_pilot")
        if not isinstance(self.prior_policy_id, str) or not self.prior_policy_id.strip():
            raise ValueError("prior_policy_id must be nonempty")
        if not isinstance(self.priors, BayesianPriorConfig):
            raise TypeError("priors must be a BayesianPriorConfig")
        if not isinstance(self.sampling_config, BayesianSamplingConfig):
            raise TypeError("sampling_config must be a BayesianSamplingConfig")
        if not self.priors.infer_temporal_shift:
            raise ValueError("Stage 5 compares free-shift models; classical fixed-shift fitting is unavailable")
        if self.priors.temporal_shift_prior is None:
            raise ValueError("free-shift prior is required")
        if tuple(self.temporal_shift_bounds_ns) != (
            self.priors.temporal_shift_prior.lower_ns,
            self.priors.temporal_shift_prior.upper_ns,
        ):
            raise ValueError("classical shift bounds must equal Bayesian shift-prior bounds")
        if isinstance(self.n_bootstrap_resamples, (bool, np.bool_)) or not isinstance(
            self.n_bootstrap_resamples, (int, np.integer)
        ) or self.n_bootstrap_resamples < 2:
            raise ValueError("n_bootstrap_resamples must be an integer >= 2")
        if not math.isfinite(self.nominal_interval_level) or not 0 < self.nominal_interval_level < 1:
            raise ValueError("nominal_interval_level must lie between zero and one")
        if self.nominal_interval_level != self.sampling_config.credible_interval_level:
            raise ValueError("Bayesian and classical interval levels must agree")
        if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(
            self.random_seed, (int, np.integer)
        ) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if not math.isfinite(self.background_fraction) or not 0 < self.background_fraction < 1:
            raise ValueError("background_fraction must lie between zero and one")


@dataclass(frozen=True)
class Issue4RealizationSeeds:
    observation: int
    bootstrap: int
    bayesian: int
    posterior_predictive: int  # reserved independent stream; not consumed in Stage 5


def derive_issue4_seed(
    base_seed: int, condition_id: str, realization_index: int, stream: str, *,
    prior_policy_id: str = "baseline",
) -> int:
    """Stable named streams; only Bayesian/predictive streams depend on prior ID."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("condition_id must be nonempty")
    if isinstance(realization_index, bool) or not isinstance(realization_index, int) or realization_index < 0:
        raise ValueError("realization_index must be a non-negative integer")
    if stream not in {"observation", "bootstrap", "bayesian", "posterior_predictive"}:
        raise ValueError("unknown Issue-4 random stream")
    policy = prior_policy_id if stream in {"bayesian", "posterior_predictive"} else "shared"
    key = f"issue4:v1:{base_seed}:{condition_id}:{realization_index}:{stream}:{policy}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def issue4_realization_seeds(
    config: BayesianClassicalEvaluationConfig, condition_id: str, realization_index: int,
) -> Issue4RealizationSeeds:
    return Issue4RealizationSeeds(**{
        name: derive_issue4_seed(
            config.random_seed, condition_id, realization_index, name,
            prior_policy_id=config.prior_policy_id,
        ) for name in ("observation", "bootstrap", "bayesian", "posterior_predictive")
    })


def build_issue4_expected_counts(
    condition: BayesianClassicalCondition,
) -> tuple[NDArray[np.float64], float]:
    """Convert finite-window signal budget into the true reconvolution scale A."""
    unit_signal = monoexponential_reconvolution_expected_counts(
        condition.time_ns, condition.generating_irf.kernel, 1.0,
        condition.true_lifetime_ns, 0.0, condition.true_temporal_shift_ns,
    )
    unit_sum = float(np.sum(unit_signal))
    if not math.isfinite(unit_sum) or unit_sum <= 0.0:
        raise ValueError("unit reconvolved signal has no finite positive window sum")
    true_amplitude = condition.signal_photon_count / unit_sum
    expected = monoexponential_reconvolution_expected_counts(
        condition.time_ns, condition.generating_irf.kernel, true_amplitude,
        condition.true_lifetime_ns, condition.background_per_bin,
        condition.true_temporal_shift_ns,
    )
    if not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        raise ValueError("generating expected counts must be finite and nonnegative")
    return expected, float(true_amplitude)


def sample_issue4_observation(
    condition: BayesianClassicalCondition, expected_counts: NDArray[np.float64], seed: int,
    realization_index: int,
) -> TCSPCMeasurement:
    """Sample one raw histogram; known truth remains outside the measurement."""
    counts = sample_photon_counts(expected_counts, np.random.default_rng(seed))
    return TCSPCMeasurement(
        time_ns=condition.time_ns, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id=f"issue4-{condition.condition_id}-{realization_index}",
        provenance={"origin": "issue4_independent_synthetic", "condition_id": condition.condition_id},
    )


@dataclass(frozen=True)
class Issue4ClassicalFit:
    optimizer_success: bool
    valid_fit: bool
    lifetime_ns: float
    amplitude: float
    background_per_bin: float
    temporal_shift_ns: float
    boundary_hit: bool
    failure_reason: str | None
    optimizer_seconds: float
    call_seconds: float
    numerical_validation_passed: bool | None = None
    max_coordinate_descent_nll: float = math.nan
    recovery_attempted: bool = False
    optimizer_status: int | None = None
    optimizer_message: str | None = None
    optimizer_nfev: int | None = None
    optimizer_njev: int | None = None


@dataclass(frozen=True)
class Issue4Covariance:
    valid_interval: bool
    lifetime_std_ns: float
    local_gaussian_lower_ns: float
    local_gaussian_upper_ns: float
    condition_number: float
    boundary_hit: bool
    failure_reason: str | None
    runtime_seconds: float


@dataclass(frozen=True)
class Issue4Bootstrap:
    valid_interval: bool
    lifetime_median_ns: float
    lifetime_std_ns: float
    percentile_lower_ns: float
    percentile_upper_ns: float
    n_requested: int
    n_valid_refits: int
    refit_failure_rate: float
    failure_reason: str | None
    runtime_seconds: float


@dataclass(frozen=True)
class Issue4Bayesian:
    status: BayesianSamplingStatus
    diagnostics_accepted: bool
    lifetime_mean_ns: float
    lifetime_median_ns: float
    posterior_lifetime_std_ns: float
    credible_lower_ns: float
    credible_upper_ns: float
    mean_acceptance_fraction: float
    minimum_effective_samples: float
    production_steps: int
    retained_samples: int
    extension_count: int
    minimum_autocorrelation_multiples: float
    maximum_autocorrelation_relative_change: float
    maximum_ensemble_mean_difference_sd: float
    maximum_ensemble_median_difference_sd: float
    lifetime_background_correlation: float
    lifetime_shift_correlation: float
    diagnostic_failure_reasons: tuple[str, ...]
    initialization_seconds: float
    sampling_seconds: float
    diagnostic_seconds: float
    inference_total_seconds: float
    call_seconds: float

    @property
    def valid_interval(self) -> bool:
        return (
            self.status is BayesianSamplingStatus.SUCCESS
            and self.diagnostics_accepted
            and math.isfinite(self.credible_lower_ns)
            and math.isfinite(self.credible_upper_ns)
            and self.credible_lower_ns <= self.credible_upper_ns
        )


@dataclass(frozen=True)
class BayesianClassicalRealizationResult:
    condition_id: str
    prior_policy_id: str
    profile: str
    realization_index: int
    true_lifetime_ns: float
    signal_photon_count: int
    true_reconvolution_amplitude: float
    background_per_bin: float
    true_temporal_shift_ns: float
    observed_total_counts: int
    observed_counts_sha256: str
    seeds: Issue4RealizationSeeds
    classical_fit: Issue4ClassicalFit
    covariance: Issue4Covariance
    bootstrap: Issue4Bootstrap
    bayesian: Issue4Bayesian
    orchestration_seconds: float


def _unavailable_covariance(reason: str) -> Issue4Covariance:
    return Issue4Covariance(False, math.nan, math.nan, math.nan, math.nan, False, reason, 0.0)


def _unavailable_bootstrap(reason: str, requested: int) -> Issue4Bootstrap:
    return Issue4Bootstrap(
        valid_interval=False,
        lifetime_median_ns=math.nan,
        lifetime_std_ns=math.nan,
        percentile_lower_ns=math.nan,
        percentile_upper_ns=math.nan,
        n_requested=requested,
        n_valid_refits=0,
        refit_failure_rate=math.nan,
        failure_reason=reason,
        runtime_seconds=0.0,
    )


def evaluate_issue4_realization(
    condition: BayesianClassicalCondition,
    config: BayesianClassicalEvaluationConfig,
    measurement: TCSPCMeasurement,
    *,
    realization_index: int,
    true_reconvolution_amplitude: float,
    seeds: Issue4RealizationSeeds,
) -> BayesianClassicalRealizationResult:
    """Evaluate the same raw count buffer with all four model-conditional methods."""
    started = time.perf_counter()
    counts = measurement.require_raw_counts()
    if not np.array_equal(measurement.time_ns, condition.time_ns):
        raise ValueError("measurement grid must equal condition grid")
    time_axis = condition.time_ns
    kernel = condition.assumed_irf.kernel
    fit_started = time.perf_counter()
    fitted = fit_single_reconvolution_curve(
        time=time_axis, counts=counts, irf=kernel,
        temporal_shift_bounds=config.temporal_shift_bounds_ns,
        objective="poisson", background_fraction=config.background_fraction,
    )
    fit_call_seconds = time.perf_counter() - fit_started
    classical = Issue4ClassicalFit(
        optimizer_success=fitted.optimizer_success,
        valid_fit=fitted.valid_fit, lifetime_ns=fitted.fitted_lifetime_ns,
        amplitude=fitted.fitted_amplitude, background_per_bin=fitted.fitted_background,
        temporal_shift_ns=fitted.fitted_temporal_shift_ns,
        boundary_hit=fitted.boundary_hit, failure_reason=fitted.failure_reason,
        optimizer_seconds=fitted.runtime_ms / 1000.0,
        call_seconds=fit_call_seconds,
        numerical_validation_passed=fitted.numerical_validation_passed,
        max_coordinate_descent_nll=fitted.max_coordinate_descent_nll,
        recovery_attempted=fitted.recovery_attempted,
        optimizer_status=fitted.optimizer_status,
        optimizer_message=fitted.optimizer_message,
        optimizer_nfev=fitted.optimizer_nfev,
        optimizer_njev=fitted.optimizer_njev,
    )

    if fitted.valid_fit:
        fit_result = _reconstruct_reconvolution_fit_result(
            time=time_axis, irf=kernel, curve_result=fitted,
        )
        covariance_started = time.perf_counter()
        covariance_result = estimate_poisson_reconvolution_local_covariance(
            time=time_axis, irf=kernel, fit_result=fit_result,
            temporal_shift_bounds=config.temporal_shift_bounds_ns,
        )
        covariance_seconds = time.perf_counter() - covariance_started
        covariance_valid = bool(
            covariance_result.covariance_valid
            and math.isfinite(covariance_result.lifetime_std)
            and covariance_result.lifetime_std > 0.0
        )
        z = NormalDist().inv_cdf(0.5 + config.nominal_interval_level / 2.0)
        lower = fitted.fitted_lifetime_ns - z * covariance_result.lifetime_std if covariance_valid else math.nan
        upper = fitted.fitted_lifetime_ns + z * covariance_result.lifetime_std if covariance_valid else math.nan
        covariance = Issue4Covariance(
            covariance_valid, covariance_result.lifetime_std, lower, upper,
            covariance_result.condition_number, covariance_result.boundary_hit,
            covariance_result.failure_reason, covariance_seconds,
        )

        bootstrap_started = time.perf_counter()
        bootstrap_result = estimate_parametric_poisson_bootstrap(
            time=time_axis, irf=kernel, fit_result=fit_result,
            temporal_shift_bounds=config.temporal_shift_bounds_ns,
            rng=np.random.default_rng(seeds.bootstrap),
            n_resamples=config.n_bootstrap_resamples,
            nominal_coverage=config.nominal_interval_level,
            background_fraction=config.background_fraction,
        )
        bootstrap_seconds = time.perf_counter() - bootstrap_started
        bootstrap = Issue4Bootstrap(
            valid_interval=bootstrap_result.bootstrap_valid,
            lifetime_median_ns=bootstrap_result.bootstrap_median_ns,
            lifetime_std_ns=bootstrap_result.bootstrap_std_ns,
            percentile_lower_ns=bootstrap_result.lower_ns,
            percentile_upper_ns=bootstrap_result.upper_ns,
            n_requested=bootstrap_result.n_resamples,
            n_valid_refits=bootstrap_result.n_successful_fits,
            refit_failure_rate=bootstrap_result.fit_failure_rate,
            failure_reason=bootstrap_result.failure_reason,
            runtime_seconds=bootstrap_seconds,
        )
    else:
        covariance = _unavailable_covariance("classical_fit_invalid")
        bootstrap = _unavailable_bootstrap("classical_fit_invalid", config.n_bootstrap_resamples)

    bayesian_started = time.perf_counter()
    inference = fit_bayesian_monoexponential_reconvolution(
        measurement, priors=config.priors,
        sampling_config=replace(config.sampling_config, random_seed=seeds.bayesian),
        prepared_irf=condition.assumed_irf,
        irf_model_relation=IRFModelRelation.MATCHED,
    )
    bayesian_call_seconds = time.perf_counter() - bayesian_started
    result = inference.result
    lifetime = result.parameter_summaries[1] if len(result.parameter_summaries) > 1 else None
    corr = result.correlation_matrix
    effective = result.diagnostics.approximate_effective_samples
    finite_effective = effective[np.isfinite(effective)]
    autocorrelation = result.diagnostics.autocorrelation_time_steps
    finite_autocorrelation = autocorrelation[np.isfinite(autocorrelation) & (autocorrelation > 0)]
    relative_change = result.diagnostics.autocorrelation_relative_change
    finite_change = relative_change[np.isfinite(relative_change)]
    bayesian = Issue4Bayesian(
        status=result.status, diagnostics_accepted=result.diagnostics.accepted,
        lifetime_mean_ns=lifetime.mean if lifetime is not None else math.nan,
        lifetime_median_ns=lifetime.median if lifetime is not None else math.nan,
        posterior_lifetime_std_ns=lifetime.standard_deviation if lifetime is not None else math.nan,
        credible_lower_ns=lifetime.credible_lower if lifetime is not None else math.nan,
        credible_upper_ns=lifetime.credible_upper if lifetime is not None else math.nan,
        mean_acceptance_fraction=result.diagnostics.mean_acceptance_fraction,
        minimum_effective_samples=float(np.min(finite_effective)) if finite_effective.size else math.nan,
        production_steps=result.diagnostics.production_steps,
        retained_samples=result.diagnostics.retained_samples,
        extension_count=result.diagnostics.extension_count,
        minimum_autocorrelation_multiples=(
            float(result.diagnostics.production_steps / np.max(finite_autocorrelation))
            if finite_autocorrelation.size else math.nan
        ),
        maximum_autocorrelation_relative_change=(
            float(np.max(finite_change)) if finite_change.size else math.nan
        ),
        maximum_ensemble_mean_difference_sd=result.diagnostics.maximum_ensemble_mean_difference_sd,
        maximum_ensemble_median_difference_sd=result.diagnostics.maximum_ensemble_median_difference_sd,
        lifetime_background_correlation=float(corr[1, 2]),
        lifetime_shift_correlation=float(corr[1, 3]),
        diagnostic_failure_reasons=result.diagnostics.failure_reasons,
        initialization_seconds=result.runtime.initialization_seconds,
        sampling_seconds=result.runtime.sampling_seconds,
        diagnostic_seconds=result.runtime.diagnostic_seconds,
        inference_total_seconds=result.runtime.total_seconds,
        call_seconds=bayesian_call_seconds,
    )
    return BayesianClassicalRealizationResult(
        condition_id=condition.condition_id, prior_policy_id=config.prior_policy_id,
        profile=config.profile, realization_index=realization_index,
        true_lifetime_ns=condition.true_lifetime_ns,
        signal_photon_count=condition.signal_photon_count,
        true_reconvolution_amplitude=true_reconvolution_amplitude,
        background_per_bin=condition.background_per_bin,
        true_temporal_shift_ns=condition.true_temporal_shift_ns,
        observed_total_counts=int(np.sum(counts)),
        observed_counts_sha256=hashlib.sha256(np.asarray(counts, dtype="<i8").tobytes()).hexdigest(),
        seeds=seeds, classical_fit=classical, covariance=covariance,
        bootstrap=bootstrap, bayesian=bayesian,
        orchestration_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class Issue4IntervalCoverage:
    method: str
    interval_kind: str
    n_attempted: int
    n_valid: int
    n_covering: int
    coverage_among_valid: float
    binomial_standard_error_among_valid: float
    wilson_95_lower: float
    wilson_95_upper: float
    failure_rate: float
    successful_and_covering_fraction: float
    mean_width_ns: float
    median_width_ns: float


@dataclass(frozen=True)
class Issue4EstimatorSpread:
    estimator: str
    n_valid: int
    empirical_std_ns: float
    empirical_bias_ns: float
    empirical_mae_ns: float
    empirical_rmse_ns: float


@dataclass(frozen=True)
class Issue4RuntimeSummary:
    metric: str
    n_valid: int
    median_seconds: float
    q25_seconds: float
    q75_seconds: float
    mean_seconds: float


@dataclass(frozen=True)
class Issue4UncertaintyScaleComparison:
    method: str
    mean_reported_lifetime_std_ns: float
    empirical_estimator: str
    empirical_estimator_spread_ns: float
    reported_to_empirical_spread_ratio: float


@dataclass(frozen=True)
class BayesianClassicalConditionSummary:
    condition_id: str
    prior_policy_id: str
    profile: str
    true_lifetime_ns: float
    n_realizations: int
    n_valid_classical_fits: int
    classical_fit_failure_rate: float
    intervals: tuple[Issue4IntervalCoverage, ...]
    estimator_spreads: tuple[Issue4EstimatorSpread, ...]
    mean_reported_lifetime_std_ns: tuple[tuple[str, float], ...]
    uncertainty_scale_comparisons: tuple[Issue4UncertaintyScaleComparison, ...]
    mean_bootstrap_refit_failure_rate: float
    runtimes: tuple[Issue4RuntimeSummary, ...]
    mean_lifetime_background_correlation: float
    mean_lifetime_shift_correlation: float


@dataclass(frozen=True)
class BayesianCalibrationReport:
    """Compact records; no posterior chains or raw histograms are retained."""

    condition: BayesianClassicalCondition
    config: BayesianClassicalEvaluationConfig
    per_realization: tuple[BayesianClassicalRealizationResult, ...]
    summary: BayesianClassicalConditionSummary


def _finite(values: list[float]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _finite_mean(values: list[float]) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else math.nan


def _interval_coverage(
    records: tuple[BayesianClassicalRealizationResult, ...], method: str,
) -> Issue4IntervalCoverage:
    if method == "covariance":
        kind = "local_gaussian_covariance_interval"
        extracted = [(r.covariance.valid_interval, r.covariance.local_gaussian_lower_ns,
                      r.covariance.local_gaussian_upper_ns) for r in records]
    elif method == "bootstrap":
        kind = "parametric_bootstrap_percentile_interval"
        extracted = [(r.bootstrap.valid_interval, r.bootstrap.percentile_lower_ns,
                      r.bootstrap.percentile_upper_ns) for r in records]
    elif method == "bayesian":
        kind = "bayesian_equal_tailed_credible_interval"
        extracted = [(r.bayesian.valid_interval, r.bayesian.credible_lower_ns,
                      r.bayesian.credible_upper_ns) for r in records]
    else:
        raise ValueError("unknown interval method")
    accepted = [(lo, hi, r.true_lifetime_ns) for r, (valid, lo, hi) in zip(records, extracted)
                if valid and math.isfinite(lo) and math.isfinite(hi) and lo <= hi]
    n_valid = len(accepted)
    n_covering = sum(lo <= truth <= hi for lo, hi, truth in accepted)
    coverage = n_covering / n_valid if n_valid else math.nan
    if n_valid:
        z95 = NormalDist().inv_cdf(0.975)
        denominator = 1.0 + z95**2 / n_valid
        centre = (coverage + z95**2 / (2.0 * n_valid)) / denominator
        half_width = z95 * math.sqrt(
            coverage * (1.0 - coverage) / n_valid + z95**2 / (4.0 * n_valid**2)
        ) / denominator
        wilson_lower, wilson_upper = centre - half_width, centre + half_width
    else:
        wilson_lower = wilson_upper = math.nan
    widths = [hi - lo for lo, hi, _ in accepted]
    n_attempted = len(records)
    return Issue4IntervalCoverage(
        method, kind, n_attempted, n_valid, n_covering, coverage,
        math.sqrt(coverage * (1.0 - coverage) / n_valid) if n_valid else math.nan,
        wilson_lower, wilson_upper,
        1.0 - n_valid / n_attempted,
        n_covering / n_attempted,
        float(np.mean(widths)) if widths else math.nan,
        float(np.median(widths)) if widths else math.nan,
    )


def _estimator_spread(
    records: tuple[BayesianClassicalRealizationResult, ...], estimator: str,
) -> Issue4EstimatorSpread:
    if estimator == "classical_fit":
        estimates = [r.classical_fit.lifetime_ns for r in records if r.classical_fit.valid_fit]
    elif estimator == "bayesian_posterior_median":
        estimates = [r.bayesian.lifetime_median_ns for r in records if r.bayesian.valid_interval]
    elif estimator == "bayesian_posterior_mean":
        estimates = [r.bayesian.lifetime_mean_ns for r in records if r.bayesian.valid_interval]
    else:
        raise ValueError("unknown estimator")
    arr = _finite(estimates)
    truth = records[0].true_lifetime_ns
    errors = arr - truth
    return Issue4EstimatorSpread(
        estimator, int(arr.size), float(np.std(arr, ddof=1)) if arr.size >= 2 else math.nan,
        float(np.mean(errors)) if arr.size else math.nan,
        float(np.mean(np.abs(errors))) if arr.size else math.nan,
        float(np.sqrt(np.mean(errors**2))) if arr.size else math.nan,
    )


def _runtime_summary(metric: str, values: list[float]) -> Issue4RuntimeSummary:
    arr = _finite(values)
    return Issue4RuntimeSummary(
        metric, int(arr.size), float(np.median(arr)) if arr.size else math.nan,
        float(np.quantile(arr, 0.25)) if arr.size else math.nan,
        float(np.quantile(arr, 0.75)) if arr.size else math.nan,
        float(np.mean(arr)) if arr.size else math.nan,
    )


def summarize_issue4_condition(
    records: tuple[BayesianClassicalRealizationResult, ...],
) -> BayesianClassicalConditionSummary:
    """Recompute all aggregates from compact per-realization records."""
    if not records:
        raise ValueError("at least one realization is required")
    first = records[0]
    if any((r.condition_id, r.prior_policy_id, r.profile, r.true_lifetime_ns) != (
        first.condition_id, first.prior_policy_id, first.profile, first.true_lifetime_ns
    ) for r in records):
        raise ValueError("records must belong to one condition and prior policy")
    intervals = tuple(_interval_coverage(records, name) for name in ("covariance", "bootstrap", "bayesian"))
    spreads = tuple(_estimator_spread(records, name) for name in (
        "classical_fit", "bayesian_posterior_median", "bayesian_posterior_mean"
    ))
    means = (
        ("covariance", _finite_mean([r.covariance.lifetime_std_ns for r in records if r.covariance.valid_interval])),
        ("bootstrap", _finite_mean([r.bootstrap.lifetime_std_ns for r in records if r.bootstrap.valid_interval])),
        ("bayesian", _finite_mean([r.bayesian.posterior_lifetime_std_ns for r in records if r.bayesian.valid_interval])),
    )
    classical_spread = spreads[0].empirical_std_ns
    bayesian_spread = spreads[1].empirical_std_ns
    comparisons = tuple(
        Issue4UncertaintyScaleComparison(
            method=name, mean_reported_lifetime_std_ns=value,
            empirical_estimator=estimator, empirical_estimator_spread_ns=denominator,
            reported_to_empirical_spread_ratio=(
                value / denominator if math.isfinite(denominator) and denominator > 0 else math.nan
            ),
        )
        for (name, value), (estimator, denominator) in zip(
            means,
            (("classical_fit", classical_spread), ("classical_fit", classical_spread),
             ("bayesian_posterior_median", bayesian_spread)),
        )
    )
    timing_fields = (
        ("classical_fit_call", [r.classical_fit.call_seconds for r in records]),
        ("classical_optimizer_only", [r.classical_fit.optimizer_seconds for r in records]),
        ("local_covariance", [r.covariance.runtime_seconds for r in records if r.classical_fit.valid_fit]),
        ("parametric_bootstrap", [r.bootstrap.runtime_seconds for r in records if r.classical_fit.valid_fit]),
        ("bayesian_initialization", [r.bayesian.initialization_seconds for r in records]),
        ("bayesian_sampling", [r.bayesian.sampling_seconds for r in records]),
        ("bayesian_diagnostics", [r.bayesian.diagnostic_seconds for r in records]),
        ("bayesian_inference_total", [r.bayesian.inference_total_seconds for r in records]),
        ("bayesian_call", [r.bayesian.call_seconds for r in records]),
        ("paired_orchestration", [r.orchestration_seconds for r in records]),
    )
    accepted_bayesian = [r for r in records if r.bayesian.valid_interval]
    return BayesianClassicalConditionSummary(
        condition_id=first.condition_id, prior_policy_id=first.prior_policy_id,
        profile=first.profile, true_lifetime_ns=first.true_lifetime_ns,
        n_realizations=len(records),
        n_valid_classical_fits=sum(r.classical_fit.valid_fit for r in records),
        classical_fit_failure_rate=1.0 - sum(r.classical_fit.valid_fit for r in records) / len(records),
        intervals=intervals, estimator_spreads=spreads,
        mean_reported_lifetime_std_ns=means,
        uncertainty_scale_comparisons=comparisons,
        mean_bootstrap_refit_failure_rate=_finite_mean(
            [r.bootstrap.refit_failure_rate for r in records if r.classical_fit.valid_fit]
        ),
        runtimes=tuple(_runtime_summary(name, values) for name, values in timing_fields),
        mean_lifetime_background_correlation=_finite_mean(
            [r.bayesian.lifetime_background_correlation for r in accepted_bayesian]
        ),
        mean_lifetime_shift_correlation=_finite_mean(
            [r.bayesian.lifetime_shift_correlation for r in accepted_bayesian]
        ),
    )


def evaluate_issue4_condition(
    condition: BayesianClassicalCondition,
    config: BayesianClassicalEvaluationConfig,
) -> BayesianCalibrationReport:
    """Run independent Poisson realizations under one correctly specified IRF."""
    lower, upper = config.temporal_shift_bounds_ns
    if not lower <= condition.true_temporal_shift_ns <= upper:
        raise ValueError("true shift must lie within common inference bounds")
    expected, true_amplitude = build_issue4_expected_counts(condition)
    records = []
    for realization_index in range(condition.n_repeats):
        seeds = issue4_realization_seeds(config, condition.condition_id, realization_index)
        measurement = sample_issue4_observation(
            condition, expected, seeds.observation, realization_index,
        )
        records.append(evaluate_issue4_realization(
            condition, config, measurement, realization_index=realization_index,
            true_reconvolution_amplitude=true_amplitude, seeds=seeds,
        ))
    per_realization = tuple(records)
    return BayesianCalibrationReport(
        condition, config, per_realization, summarize_issue4_condition(per_realization)
    )
