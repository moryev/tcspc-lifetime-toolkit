"""emcee sampling and operational diagnostics for the Stage-2 Poisson model.

These diagnostics concern numerical exploration of a posterior conditional on a
fixed IRF and mono-exponential forward model. They do not validate that model.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.bayesian import (
    BayesianModelContext,
    BayesianPhysicalParameters,
    BayesianPriorConfig,
    IRFModelRelation,
    log_posterior,
    physical_to_sampling_coordinates,
)
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import IRFSourceKind
from tcspc_toolkit.irf_preparation import PreparedIRF
from tcspc_toolkit.measurements import TCSPCMeasurement


PHYSICAL_PARAMETER_NAMES = (
    "amplitude", "lifetime_ns", "background_per_bin", "temporal_shift_ns"
)
SAMPLING_COORDINATE_NAMES = (
    "log_amplitude", "log_lifetime_ns", "log_background_per_bin", "temporal_shift_ns"
)
PARAMETER_UNITS = ("reconvolution_scale", "ns", "counts_per_bin", "ns")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _immutable_array(values: NDArray, *, dtype: np.dtype | type = np.float64) -> NDArray:
    """Copy into an immutable bytes-backed NumPy buffer."""
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class BayesianSamplingConfig:
    """Explicit emcee budget and operational, non-scientific acceptance policy.

    Default checks require at least 50 estimated autocorrelation times per
    ensemble, <=10% full-vs-half-chain tau change, >=100 approximate effective
    draws per parameter, acceptable mean acceptance, and agreement of separate
    ensemble means and medians within 0.5 pooled posterior SD. Tests may use
    explicitly looser settings for short regression chains; those do not
    establish scientific convergence.
    """

    random_seed: int
    n_walkers: int = 32
    warmup_steps: int = 500
    production_steps: int = 1000
    max_production_steps: int = 5000
    extension_steps: int = 500
    n_ensembles: int = 2
    credible_interval_level: float = 0.95
    min_autocorrelation_multiples: float = 50.0
    max_autocorrelation_relative_change: float = 0.10
    min_effective_samples: float = 100.0
    max_ensemble_location_difference_sd: float = 0.5
    min_mean_acceptance_fraction: float = 0.10
    max_mean_acceptance_fraction: float = 0.90

    def __post_init__(self) -> None:
        if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(
            self.random_seed, (int, np.integer)
        ) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        object.__setattr__(self, "random_seed", int(self.random_seed))
        for name in (
            "n_walkers", "warmup_steps", "production_steps", "max_production_steps",
            "extension_steps", "n_ensembles",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.n_ensembles < 2:
            raise ValueError("n_ensembles must be at least 2 independent runs")
        if self.max_production_steps < self.production_steps:
            raise ValueError("max_production_steps must be >= production_steps")
        level = _finite_float(self.credible_interval_level, "credible_interval_level")
        if not 0.0 < level < 1.0:
            raise ValueError("credible_interval_level must be between 0 and 1")
        object.__setattr__(self, "credible_interval_level", level)
        for name in (
            "min_autocorrelation_multiples", "min_effective_samples",
            "max_ensemble_location_difference_sd",
        ):
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        change = _finite_float(
            self.max_autocorrelation_relative_change,
            "max_autocorrelation_relative_change",
        )
        if change < 0.0:
            raise ValueError("max_autocorrelation_relative_change must be non-negative")
        object.__setattr__(self, "max_autocorrelation_relative_change", change)
        low = _finite_float(self.min_mean_acceptance_fraction, "min_mean_acceptance_fraction")
        high = _finite_float(self.max_mean_acceptance_fraction, "max_mean_acceptance_fraction")
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("mean acceptance bounds must satisfy 0 <= low < high <= 1")
        object.__setattr__(self, "min_mean_acceptance_fraction", low)
        object.__setattr__(self, "max_mean_acceptance_fraction", high)


class BayesianSamplingStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_SAMPLING = "insufficient_sampling"
    INITIALIZATION_FAILED = "initialization_failed"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class BayesianPosteriorSamples:
    """Unthinned production draws: (ensemble, step, walker, parameter).

    ``physical`` always has A, tau/ns, B/counts-per-bin, shift/ns (including a
    constant fixed shift). ``transformed`` has log A, log tau, log B, and the
    shift only when inferred. Warmup is excluded. ``log_probability`` is the
    Stage-2 transformed-coordinate posterior target at each retained draw.
    """

    physical: NDArray[np.float64]
    transformed: NDArray[np.float64]
    log_probability: NDArray[np.float64]
    physical_parameter_names: tuple[str, ...] = PHYSICAL_PARAMETER_NAMES
    sampling_coordinate_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        physical = np.asarray(self.physical, dtype=np.float64)
        transformed = np.asarray(self.transformed, dtype=np.float64)
        log_probability = np.asarray(self.log_probability, dtype=np.float64)
        if physical.ndim != 4 or physical.shape[-1] != 4:
            raise ValueError("physical samples must have shape (ensemble, step, walker, 4)")
        if transformed.ndim != 4 or transformed.shape[:3] != physical.shape[:3]:
            raise ValueError("transformed samples must match the physical sample axes")
        if transformed.shape[-1] not in (3, 4):
            raise ValueError("transformed samples must have three or four coordinates")
        if log_probability.shape != physical.shape[:3]:
            raise ValueError("log_probability must match ensemble, step, walker axes")
        if self.physical_parameter_names != PHYSICAL_PARAMETER_NAMES:
            raise ValueError("physical parameter order must be A, tau, B, shift")
        expected_names = SAMPLING_COORDINATE_NAMES[:transformed.shape[-1]]
        if self.sampling_coordinate_names and self.sampling_coordinate_names != expected_names:
            raise ValueError("sampling coordinate names do not match sample dimension")
        object.__setattr__(self, "sampling_coordinate_names", expected_names)
        object.__setattr__(self, "physical", _immutable_array(physical))
        object.__setattr__(self, "transformed", _immutable_array(transformed))
        object.__setattr__(self, "log_probability", _immutable_array(log_probability))


@dataclass(frozen=True)
class BayesianParameterSummary:
    name: str
    unit: str
    mean: float
    median: float
    standard_deviation: float
    credible_lower: float
    credible_upper: float


@dataclass(frozen=True)
class BayesianSamplingDiagnostics:
    """Operational chain checks, not a proof of convergence or model validity."""

    production_steps: int
    retained_samples: int
    extension_count: int
    acceptance_fraction: NDArray[np.float64]  # ensemble × walker
    mean_acceptance_fraction: float
    autocorrelation_time_steps: NDArray[np.float64]  # ensemble × parameter; NaN if unavailable
    autocorrelation_relative_change: NDArray[np.float64]  # full vs first half
    approximate_effective_samples: NDArray[np.float64]  # parameter; NaN if unavailable
    maximum_ensemble_mean_difference_sd: float
    maximum_ensemble_median_difference_sd: float
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "acceptance_fraction", "autocorrelation_time_steps",
            "autocorrelation_relative_change", "approximate_effective_samples",
        ):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))

    @property
    def accepted(self) -> bool:
        return not self.failure_reasons


@dataclass(frozen=True)
class BayesianRuntime:
    initialization_seconds: float
    sampling_seconds: float
    diagnostic_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class BayesianModelContextSummary:
    sample_id: str | None
    irf_selection: str
    irf_source_kind: IRFSourceKind
    irf_model_relation: IRFModelRelation
    fixed_irf_assumption: bool = True


@dataclass(frozen=True)
class BayesianReconvolutionResult:
    """Physical-space summaries conditional on one fixed reconvolution model."""

    status: BayesianSamplingStatus
    parameter_summaries: tuple[BayesianParameterSummary, ...]
    inferred_parameter_names: tuple[str, ...]
    inferred_parameter_units: tuple[str, ...]
    correlation_matrix: NDArray[np.float64]
    diagnostics: BayesianSamplingDiagnostics
    runtime: BayesianRuntime
    priors: BayesianPriorConfig
    sampling_config: BayesianSamplingConfig
    model_context: BayesianModelContextSummary
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.correlation_matrix, dtype=np.float64)
        n_parameters = len(self.inferred_parameter_names)
        if matrix.shape != (n_parameters, n_parameters):
            raise ValueError("correlation_matrix shape must match inferred parameters")
        object.__setattr__(self, "correlation_matrix", _immutable_array(matrix))


@dataclass(frozen=True)
class BayesianInferenceRun:
    result: BayesianReconvolutionResult
    samples: BayesianPosteriorSamples
    context: BayesianModelContext


class BayesianInitializationError(RuntimeError):
    """Unable to place distinct finite-probability walkers inside prior support."""


def _prior_draw(priors: BayesianPriorConfig, rng: np.random.Generator) -> BayesianPhysicalParameters:
    shift = priors.fixed_temporal_shift_ns
    if priors.infer_temporal_shift:
        assert priors.temporal_shift_prior is not None
        shift = rng.uniform(
            priors.temporal_shift_prior.lower_ns,
            priors.temporal_shift_prior.upper_ns,
        )
    assert shift is not None
    return BayesianPhysicalParameters(
        amplitude=float(rng.gamma(priors.amplitude.shape, 1.0 / priors.amplitude.rate)),
        lifetime_ns=float(rng.lognormal(
            priors.lifetime_ns.log_mean, priors.lifetime_ns.log_std
        )),
        background_per_bin=float(rng.gamma(
            priors.background_per_bin.shape, 1.0 / priors.background_per_bin.rate
        )),
        temporal_shift_ns=float(shift),
    )


def _initial_center(
    context: BayesianModelContext,
    priors: BayesianPriorConfig,
) -> NDArray[np.float64] | None:
    """Data-derived A/B scale and prior-median tau; initialization only."""
    counts = context.counts
    tail = counts[max(0, counts.size - max(5, counts.size // 10)):]
    background = max(float(np.mean(tail)), np.finfo(float).tiny)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lifetime = float(np.exp(priors.lifetime_ns.log_mean))
    shift = priors.fixed_temporal_shift_ns
    if priors.infer_temporal_shift:
        assert priors.temporal_shift_prior is not None
        shift = 0.5 * (
            priors.temporal_shift_prior.lower_ns + priors.temporal_shift_prior.upper_ns
        )
    assert shift is not None
    if not math.isfinite(lifetime) or lifetime <= 0.0:
        return None
    try:
        unit_curve = monoexponential_reconvolution_expected_counts(
            context.time_ns, context.irf_kernel, amplitude=1.0,
            lifetime=lifetime, background=0.0, temporal_shift=shift,
        )
        unit_peak = float(np.max(unit_curve))
        amplitude = max(
            (float(np.max(counts)) - background) / unit_peak,
            np.finfo(float).tiny,
        )
        parameters = BayesianPhysicalParameters(amplitude, lifetime, background, shift)
        return physical_to_sampling_coordinates(parameters, priors)
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return None


def _initialize_walkers(
    context: BayesianModelContext,
    priors: BayesianPriorConfig,
    n_walkers: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    center = _initial_center(context, priors)
    ndim = priors.n_sampling_coordinates
    scale = np.full(ndim, 0.20, dtype=np.float64)
    if priors.infer_temporal_shift:
        assert priors.temporal_shift_prior is not None
        scale[3] = 0.15 * (
            priors.temporal_shift_prior.upper_ns - priors.temporal_shift_prior.lower_ns
        )
    positions = np.empty((n_walkers, ndim), dtype=np.float64)
    for walker in range(n_walkers):
        accepted = False
        for _ in range(128):
            if center is not None:
                proposal = center + rng.normal(0.0, scale)
            else:
                try:
                    proposal = physical_to_sampling_coordinates(_prior_draw(priors, rng), priors)
                except ValueError:
                    continue
            if math.isfinite(log_posterior(proposal, context, priors)):
                positions[walker] = proposal
                accepted = True
                break
        if not accepted and center is not None:
            for _ in range(256):
                try:
                    proposal = physical_to_sampling_coordinates(_prior_draw(priors, rng), priors)
                except ValueError:
                    continue
                if math.isfinite(log_posterior(proposal, context, priors)):
                    positions[walker] = proposal
                    accepted = True
                    break
        if not accepted:
            raise BayesianInitializationError(
                f"could not initialize walker {walker} inside finite posterior support"
            )
    if np.linalg.matrix_rank(positions - np.mean(positions, axis=0)) < ndim:
        raise BayesianInitializationError("initial walker positions do not span parameter space")
    return positions


def _physical_draws(
    transformed: NDArray[np.float64], priors: BayesianPriorConfig
) -> NDArray[np.float64]:
    physical = np.empty((*transformed.shape[:3], 4), dtype=np.float64)
    with np.errstate(over="ignore", under="ignore"):
        physical[..., :3] = np.exp(transformed[..., :3])
    physical[..., 3] = (
        transformed[..., 3] if priors.infer_temporal_shift
        else priors.fixed_temporal_shift_ns
    )
    return physical


def _ensemble_location_difference(
    physical: NDArray[np.float64], n_parameters: int
) -> tuple[float, float]:
    if physical.shape[1] == 0:
        return math.inf, math.inf
    draws = physical[..., :n_parameters].reshape(physical.shape[0], -1, n_parameters)
    means = np.mean(draws, axis=1)
    medians = np.median(draws, axis=1)
    spread = np.std(draws.reshape(-1, n_parameters), axis=0, ddof=1)

    def maximum_difference(locations: NDArray[np.float64]) -> float:
        largest = 0.0
        for first in range(locations.shape[0]):
            for second in range(first + 1, locations.shape[0]):
                for parameter in range(n_parameters):
                    difference = abs(float(locations[first, parameter] - locations[second, parameter]))
                    if spread[parameter] <= np.finfo(float).eps * max(
                        1.0, abs(float(means[first, parameter])),
                        abs(float(means[second, parameter])),
                    ):
                        standardized = 0.0 if difference == 0.0 else math.inf
                    else:
                        standardized = difference / float(spread[parameter])
                    largest = max(largest, standardized)
        return largest

    return maximum_difference(means), maximum_difference(medians)


def _assess_sampling_diagnostics(
    transformed: NDArray[np.float64],
    physical: NDArray[np.float64],
    acceptance_fraction: NDArray[np.float64],
    config: BayesianSamplingConfig,
    *,
    extension_count: int,
    autocorrelation_estimator: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> BayesianSamplingDiagnostics:
    """Assess separate ensembles; never treat coupled walkers as R-hat chains."""
    if autocorrelation_estimator is None:
        from emcee.autocorr import integrated_time

        def autocorrelation_estimator(chain: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(integrated_time(chain, tol=0), dtype=np.float64)

    from emcee.autocorr import AutocorrError

    n_ensembles, n_steps, n_walkers, n_parameters = transformed.shape
    tau = np.full((n_ensembles, n_parameters), np.nan)
    change = np.full_like(tau, np.nan)
    failures: list[str] = []
    mean_acceptance = float(np.mean(acceptance_fraction)) if n_steps else math.nan
    if n_steps == 0:
        failures.append("no production samples")
    if not math.isfinite(mean_acceptance) or not (
        config.min_mean_acceptance_fraction
        <= mean_acceptance
        <= config.max_mean_acceptance_fraction
    ):
        failures.append("mean acceptance fraction outside configured bounds")
    for ensemble in range(n_ensembles):
        if n_steps < 4:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                full = np.asarray(autocorrelation_estimator(transformed[ensemble]), dtype=float)
                half = np.asarray(
                    autocorrelation_estimator(transformed[ensemble, :n_steps // 2]),
                    dtype=float,
                )
        except AutocorrError:
            continue
        if full.shape != (n_parameters,) or half.shape != (n_parameters,):
            raise ValueError("autocorrelation estimator must return one value per parameter")
        valid = np.isfinite(full) & np.isfinite(half) & (full > 0.0) & (half > 0.0)
        tau[ensemble, valid] = full[valid]
        change[ensemble, valid] = np.abs(full[valid] - half[valid]) / full[valid]
    available = bool(np.all(np.isfinite(tau)) and np.all(np.isfinite(change)))
    ess = np.full(n_parameters, np.nan)
    if not available:
        failures.append("autocorrelation estimate unavailable")
    else:
        if np.any(n_steps < config.min_autocorrelation_multiples * tau):
            failures.append("production shorter than configured autocorrelation multiple")
        if np.any(change > config.max_autocorrelation_relative_change):
            failures.append("autocorrelation estimate unstable across chain halves")
        ess = np.sum(n_steps * n_walkers / tau, axis=0)
        if np.any(ess < config.min_effective_samples):
            failures.append("approximate effective sample size below configured minimum")
    mean_diff, median_diff = _ensemble_location_difference(physical, n_parameters)
    if max(mean_diff, median_diff) > config.max_ensemble_location_difference_sd:
        failures.append("independent ensemble locations disagree")
    return BayesianSamplingDiagnostics(
        production_steps=n_steps,
        retained_samples=n_ensembles * n_steps * n_walkers,
        extension_count=extension_count,
        acceptance_fraction=acceptance_fraction,
        mean_acceptance_fraction=mean_acceptance,
        autocorrelation_time_steps=tau,
        autocorrelation_relative_change=change,
        approximate_effective_samples=ess,
        maximum_ensemble_mean_difference_sd=mean_diff,
        maximum_ensemble_median_difference_sd=median_diff,
        failure_reasons=tuple(failures),
    )


def _posterior_correlation(draws: NDArray[np.float64]) -> NDArray[np.float64]:
    n_parameters = draws.shape[-1]
    matrix = np.full((n_parameters, n_parameters), np.nan)
    flattened = draws.reshape(-1, n_parameters)
    if flattened.shape[0] < 2:
        return matrix
    standard_deviation = np.std(flattened, axis=0, ddof=1)
    nondegenerate = standard_deviation > (
        np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(flattened), axis=0))
    )
    if np.any(nondegenerate):
        selected = flattened[:, nondegenerate]
        calculated = np.atleast_2d(np.corrcoef(selected, rowvar=False))
        matrix[np.ix_(nondegenerate, nondegenerate)] = calculated
    return matrix


def _summarize_samples(
    physical: NDArray[np.float64],
    config: BayesianSamplingConfig,
    n_parameters: int,
) -> tuple[tuple[BayesianParameterSummary, ...], NDArray[np.float64]]:
    selected = physical[..., :n_parameters]
    correlation = _posterior_correlation(selected)
    if selected.shape[1] == 0:
        return (), correlation
    flattened = selected.reshape(-1, n_parameters)
    tail = (1.0 - config.credible_interval_level) / 2.0
    summaries = tuple(
        BayesianParameterSummary(
            name=PHYSICAL_PARAMETER_NAMES[index],
            unit=PARAMETER_UNITS[index],
            mean=float(np.mean(flattened[:, index])),
            median=float(np.median(flattened[:, index])),
            standard_deviation=float(np.std(flattened[:, index], ddof=1)),
            credible_lower=float(np.quantile(flattened[:, index], tail)),
            credible_upper=float(np.quantile(flattened[:, index], 1.0 - tail)),
        )
        for index in range(n_parameters)
    )
    return summaries, correlation


def _capture_samples(
    samplers: list,
    config: BayesianSamplingConfig,
    priors: BayesianPriorConfig,
) -> BayesianPosteriorSamples:
    # A numerical interruption can leave unequal lengths; retain the common
    # complete prefix without padding invalid draws into public samples.
    steps = min((sampler.iteration for sampler in samplers), default=0)
    ndim = priors.n_sampling_coordinates
    if steps:
        transformed = np.stack([sampler.get_chain()[:steps] for sampler in samplers])
        log_probability = np.stack([sampler.get_log_prob()[:steps] for sampler in samplers])
    else:
        transformed = np.empty((config.n_ensembles, 0, config.n_walkers, ndim))
        log_probability = np.empty((config.n_ensembles, 0, config.n_walkers))
    return BayesianPosteriorSamples(
        physical=_physical_draws(transformed, priors),
        transformed=transformed,
        log_probability=log_probability,
        sampling_coordinate_names=SAMPLING_COORDINATE_NAMES[:ndim],
    )


def _build_run(
    *,
    context: BayesianModelContext,
    priors: BayesianPriorConfig,
    config: BayesianSamplingConfig,
    samples: BayesianPosteriorSamples,
    diagnostics: BayesianSamplingDiagnostics,
    status: BayesianSamplingStatus,
    failure_reason: str | None,
    initialization_seconds: float,
    sampling_seconds: float,
    diagnostic_seconds: float,
    started_at: float,
) -> BayesianInferenceRun:
    n_parameters = priors.n_sampling_coordinates
    summaries, correlation = _summarize_samples(samples.physical, config, n_parameters)
    result = BayesianReconvolutionResult(
        status=status,
        parameter_summaries=summaries,
        inferred_parameter_names=PHYSICAL_PARAMETER_NAMES[:n_parameters],
        inferred_parameter_units=PARAMETER_UNITS[:n_parameters],
        correlation_matrix=correlation,
        diagnostics=diagnostics,
        runtime=BayesianRuntime(
            initialization_seconds=initialization_seconds,
            sampling_seconds=sampling_seconds,
            diagnostic_seconds=diagnostic_seconds,
            total_seconds=time.perf_counter() - started_at,
        ),
        priors=priors,
        sampling_config=config,
        model_context=BayesianModelContextSummary(
            sample_id=context.measurement.sample_id,
            irf_selection=context.irf_selection,
            irf_source_kind=context.irf_source_kind,
            irf_model_relation=context.irf_model_relation,
        ),
        failure_reason=failure_reason,
    )
    return BayesianInferenceRun(result=result, samples=samples, context=context)


def fit_bayesian_monoexponential_reconvolution(
    measurement: TCSPCMeasurement,
    *,
    priors: BayesianPriorConfig,
    sampling_config: BayesianSamplingConfig,
    prepared_irf: PreparedIRF | None = None,
    irf_model_relation: IRFModelRelation = IRFModelRelation.UNSPECIFIED,
) -> BayesianInferenceRun:
    """Sample the Stage-2 transformed posterior conditional on one fixed IRF.

    Numerical sampling status is not physical-model validation. The selected
    IRF remains a fixed plug-in assumption; no IRF uncertainty is propagated.
    """
    started_at = time.perf_counter()
    if not isinstance(priors, BayesianPriorConfig):
        raise TypeError("priors must be a BayesianPriorConfig")
    if not isinstance(sampling_config, BayesianSamplingConfig):
        raise TypeError("sampling_config must be a BayesianSamplingConfig")
    context = BayesianModelContext(
        measurement, prepared_irf=prepared_irf, irf_model_relation=irf_model_relation
    )
    ndim = priors.n_sampling_coordinates
    if sampling_config.n_walkers < 2 * ndim:
        raise ValueError(
            f"n_walkers must be at least {2 * ndim} for the {ndim}-parameter emcee stretch move"
        )
    try:
        import emcee
    except ModuleNotFoundError as exc:
        if exc.name != "emcee":
            raise
        raise ImportError(
            'Bayesian MCMC requires the optional emcee extra; install with '
            'python -m pip install -e ".[bayesian]"'
        ) from exc

    config = sampling_config
    seeds = np.random.SeedSequence(config.random_seed).spawn(2 * config.n_ensembles)
    samplers = []
    initialization_seconds = sampling_seconds = diagnostic_seconds = 0.0
    initialization_started = time.perf_counter()
    try:
        for ensemble in range(config.n_ensembles):
            rng = np.random.default_rng(seeds[2 * ensemble])
            positions = _initialize_walkers(context, priors, config.n_walkers, rng)
            sampler = emcee.EnsembleSampler(
                config.n_walkers, ndim, log_posterior, args=(context, priors)
            )
            emcee_seed = int(seeds[2 * ensemble + 1].generate_state(1)[0])
            sampler.random_state = np.random.RandomState(emcee_seed).get_state()
            samplers.append((sampler, positions))
    except BayesianInitializationError as exc:
        initialization_seconds = time.perf_counter() - initialization_started
        empty = _capture_samples([], config, priors)
        acceptance = np.full((config.n_ensembles, config.n_walkers), np.nan)
        diagnostic_started = time.perf_counter()
        diagnostics = _assess_sampling_diagnostics(
            empty.transformed, empty.physical, acceptance, config, extension_count=0,
        )
        diagnostic_seconds = time.perf_counter() - diagnostic_started
        return _build_run(
            context=context, priors=priors, config=config, samples=empty,
            diagnostics=diagnostics, status=BayesianSamplingStatus.INITIALIZATION_FAILED,
            failure_reason=str(exc), initialization_seconds=initialization_seconds,
            sampling_seconds=0.0, diagnostic_seconds=diagnostic_seconds,
            started_at=started_at,
        )
    initialization_seconds = time.perf_counter() - initialization_started

    sampler_objects = [item[0] for item in samplers]
    extension_count = 0
    numerical_failure: str | None = None
    try:
        sampling_started = time.perf_counter()
        states = []
        for sampler, positions in samplers:
            state = sampler.run_mcmc(positions, config.warmup_steps, progress=False)
            sampler.reset()
            states.append(state)
        for sampler, state in zip(sampler_objects, states):
            sampler.run_mcmc(state, config.production_steps, progress=False)
        sampling_seconds += time.perf_counter() - sampling_started

        while True:
            diagnostic_started = time.perf_counter()
            samples = _capture_samples(sampler_objects, config, priors)
            acceptance = np.stack([sampler.acceptance_fraction for sampler in sampler_objects])
            diagnostics = _assess_sampling_diagnostics(
                samples.transformed, samples.physical, acceptance, config,
                extension_count=extension_count,
            )
            diagnostic_seconds += time.perf_counter() - diagnostic_started
            if diagnostics.accepted or diagnostics.production_steps >= config.max_production_steps:
                break
            next_steps = min(
                config.extension_steps,
                config.max_production_steps - diagnostics.production_steps,
            )
            sampling_started = time.perf_counter()
            for sampler in sampler_objects:
                sampler.run_mcmc(None, next_steps, progress=False)
            sampling_seconds += time.perf_counter() - sampling_started
            extension_count += 1
    except (FloatingPointError, OverflowError) as exc:
        numerical_failure = str(exc)
    except ValueError as exc:
        if "Probability function returned NaN" not in str(exc):
            raise
        numerical_failure = str(exc)

    if numerical_failure is not None:
        # Preserve any common completed production prefix, even on a failure.
        sampling_seconds += time.perf_counter() - sampling_started
        diagnostic_started = time.perf_counter()
        samples = _capture_samples(sampler_objects, config, priors)
        acceptance = np.stack([sampler.acceptance_fraction for sampler in sampler_objects])
        diagnostics = _assess_sampling_diagnostics(
            samples.transformed, samples.physical, acceptance, config,
            extension_count=extension_count,
        )
        diagnostic_seconds += time.perf_counter() - diagnostic_started
        status = BayesianSamplingStatus.NUMERICAL_FAILURE
    else:
        status = (
            BayesianSamplingStatus.SUCCESS if diagnostics.accepted
            else BayesianSamplingStatus.INSUFFICIENT_SAMPLING
        )
    return _build_run(
        context=context, priors=priors, config=config, samples=samples,
        diagnostics=diagnostics, status=status, failure_reason=numerical_failure,
        initialization_seconds=initialization_seconds,
        sampling_seconds=sampling_seconds, diagnostic_seconds=diagnostic_seconds,
        started_at=started_at,
    )
