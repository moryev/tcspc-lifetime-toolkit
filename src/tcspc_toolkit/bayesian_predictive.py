"""Prior and posterior prediction conditional on one fixed TCSPC IRF.

Expected-count bands describe parameter uncertainty in the latent histogram.
Replicated-count bands additionally include new Poisson observation noise.
Neither includes uncertainty in the selected (possibly proxy) IRF or establishes
that the mono-exponential model is physically valid.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.bayesian import BayesianModelContext, BayesianPriorConfig
from tcspc_toolkit.bayesian_sampling import (
    BayesianInferenceRun,
    BayesianSamplingConfig,
    BayesianSamplingStatus,
    _immutable_array,
    _prior_draw,
)
from tcspc_toolkit.evaluation import calculate_poisson_deviance_residuals
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.simulation import sample_photon_counts


PHYSICAL_PARAMETER_ORDER = (
    "amplitude", "lifetime_ns", "background_per_bin", "temporal_shift_ns"
)


def _validate_request(n_draws: int, random_seed: int, interval_level: float) -> tuple[int, int, float]:
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(
        n_draws, (int, np.integer)
    ) or n_draws <= 0:
        raise ValueError("n_draws must be a positive integer")
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(
        random_seed, (int, np.integer)
    ) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if isinstance(interval_level, (bool, np.bool_)) or not isinstance(
        interval_level, (int, float, np.integer, np.floating)
    ) or not math.isfinite(float(interval_level)) or not 0.0 < float(interval_level) < 1.0:
        raise ValueError("interval_level must be finite and between 0 and 1")
    return int(n_draws), int(random_seed), float(interval_level)


def _validated_draw_arrays(
    physical: NDArray[np.float64],
    expected: NDArray[np.float64],
    replicated: NDArray[np.int64],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    physical_array = np.asarray(physical, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    replicated_array = np.asarray(replicated)
    if physical_array.ndim != 2 or physical_array.shape[1] != 4 or physical_array.shape[0] == 0:
        raise ValueError("physical_parameter_draws must have shape (n_draws, 4)")
    shape = (physical_array.shape[0], n_bins)
    if expected_array.shape != shape or replicated_array.shape != shape:
        raise ValueError("predictive count arrays must have shape (n_draws, n_bins)")
    if not np.all(np.isfinite(physical_array)) or np.any(physical_array[:, :3] <= 0.0):
        raise ValueError("physical parameter draws must be finite and positive where required")
    if not np.all(np.isfinite(expected_array)) or np.any(expected_array < 0.0):
        raise ValueError("expected counts must be finite and non-negative")
    if not np.issubdtype(replicated_array.dtype, np.integer) or np.any(replicated_array < 0):
        raise ValueError("replicated counts must be non-negative integers")
    return (
        _immutable_array(physical_array),
        _immutable_array(expected_array),
        _immutable_array(replicated_array, dtype=np.int64),
    )


@dataclass(frozen=True)
class BayesianPredictiveBand:
    """Per-bin median and equal-tailed quantiles at the stated probability level."""

    median: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    interval_level: float

    def __post_init__(self) -> None:
        _, _, level = _validate_request(1, 0, self.interval_level)
        object.__setattr__(self, "interval_level", level)
        arrays = [np.asarray(getattr(self, name), dtype=np.float64) for name in (
            "median", "lower", "upper"
        )]
        if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
            raise ValueError("predictive band arrays must be matching one-dimensional curves")
        if not np.all(np.isfinite(arrays)) or np.any(arrays[1] > arrays[0]) or np.any(
            arrays[0] > arrays[2]
        ):
            raise ValueError("predictive band quantiles must be finite and ordered")
        for name, array in zip(("median", "lower", "upper"), arrays):
            object.__setattr__(self, name, _immutable_array(array))


def summarize_predictive_band(
    count_draws: NDArray[np.float64] | NDArray[np.int64],
    *,
    interval_level: float,
) -> BayesianPredictiveBand:
    """Summarize draws without a Gaussian approximation (draws × time bins)."""
    _, _, level = _validate_request(1, 0, interval_level)
    draws = np.asarray(count_draws, dtype=np.float64)
    if draws.ndim != 2 or min(draws.shape) == 0:
        raise ValueError("count_draws must have shape (positive n_draws, positive n_bins)")
    if not np.all(np.isfinite(draws)) or np.any(draws < 0.0):
        raise ValueError("count_draws must be finite and non-negative")
    tail = (1.0 - level) / 2.0
    return BayesianPredictiveBand(
        median=np.median(draws, axis=0),
        lower=np.quantile(draws, tail, axis=0),
        upper=np.quantile(draws, 1.0 - tail, axis=0),
        interval_level=level,
    )


@dataclass(frozen=True)
class BayesianPredictiveDiscrepancy:
    """Observed and replicated statistics with descriptive >= tail fraction.

    ``observed`` is a scalar for count/peak/window statistics and a vector
    paired with the parameter draw for deviance/residual statistics. This
    posterior predictive tail probability is model-conditional, not a
    universally calibrated frequentist p-value.
    """

    observed: float | NDArray[np.float64]
    replicated: NDArray[np.float64]
    posterior_predictive_tail_probability: float

    def __post_init__(self) -> None:
        replicated = np.asarray(self.replicated, dtype=np.float64)
        if replicated.ndim != 1 or replicated.size == 0 or not np.all(np.isfinite(replicated)):
            raise ValueError("replicated discrepancy values must be a finite one-dimensional vector")
        observed = np.asarray(self.observed, dtype=np.float64)
        if observed.ndim == 0:
            if not math.isfinite(float(observed)):
                raise ValueError("observed discrepancy must be finite")
            object.__setattr__(self, "observed", float(observed))
        elif observed.shape == replicated.shape and np.all(np.isfinite(observed)):
            object.__setattr__(self, "observed", _immutable_array(observed))
        else:
            raise ValueError("observed discrepancy must be a scalar or match replicated draws")
        probability = float(self.posterior_predictive_tail_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("posterior predictive tail probability must lie in [0, 1]")
        object.__setattr__(self, "replicated", _immutable_array(replicated))
        object.__setattr__(self, "posterior_predictive_tail_probability", probability)


@dataclass(frozen=True)
class BayesianPredictiveWindowDiscrepancy:
    name: str
    lower_ns: float
    upper_ns: float
    total_counts: BayesianPredictiveDiscrepancy


@dataclass(frozen=True)
class BayesianPosteriorPredictiveDiagnostics:
    """Descriptive discrepancies; no combined score or model-validity flag."""

    poisson_deviance: BayesianPredictiveDiscrepancy
    rms_signed_deviance_residual: BayesianPredictiveDiscrepancy
    maximum_absolute_signed_deviance_residual: BayesianPredictiveDiscrepancy
    mean_observed_signed_deviance_residual_profile: NDArray[np.float64]
    total_counts: BayesianPredictiveDiscrepancy
    peak_counts: BayesianPredictiveDiscrepancy
    peak_time_ns: BayesianPredictiveDiscrepancy
    windows: tuple[BayesianPredictiveWindowDiscrepancy, ...]

    def __post_init__(self) -> None:
        profile = np.asarray(self.mean_observed_signed_deviance_residual_profile, dtype=float)
        if profile.ndim != 1 or not np.all(np.isfinite(profile)):
            raise ValueError("mean residual profile must be a finite one-dimensional curve")
        object.__setattr__(
            self, "mean_observed_signed_deviance_residual_profile",
            _immutable_array(profile),
        )


def _discrepancy(
    observed: float | NDArray[np.float64],
    replicated: NDArray[np.float64],
) -> BayesianPredictiveDiscrepancy:
    return BayesianPredictiveDiscrepancy(
        observed=observed,
        replicated=replicated,
        posterior_predictive_tail_probability=float(np.mean(replicated >= observed)),
    )


def _window_discrepancy(
    name: str,
    bounds: tuple[float, float],
    context: BayesianModelContext,
    replicated_counts: NDArray[np.int64],
) -> BayesianPredictiveWindowDiscrepancy:
    try:
        lower, upper = bounds
        lower = float(lower)
        upper = float(upper)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-element time window in ns") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must have finite increasing bounds")
    mask = (context.time_ns >= lower) & (context.time_ns < upper)
    if not np.any(mask):
        raise ValueError(f"{name} must contain at least one measurement time bin")
    observed = float(np.sum(context.counts[mask]))
    replicated = np.sum(replicated_counts[:, mask], axis=1, dtype=np.int64).astype(float)
    return BayesianPredictiveWindowDiscrepancy(
        name=name, lower_ns=lower, upper_ns=upper,
        total_counts=_discrepancy(observed, replicated),
    )


def calculate_bayesian_posterior_predictive_diagnostics(
    context: BayesianModelContext,
    expected_counts: NDArray[np.float64],
    replicated_counts: NDArray[np.int64],
    *,
    early_window_ns: tuple[float, float] | None = None,
    tail_window_ns: tuple[float, float] | None = None,
) -> BayesianPosteriorPredictiveDiagnostics:
    """Compare raw observed counts with paired fixed-IRF predictive replicas.

    Peak-time ties select the first maximum. Time windows are caller-supplied
    half-open intervals [lower, upper); none are inferred from the histogram.
    """
    if not isinstance(context, BayesianModelContext):
        raise TypeError("context must be a BayesianModelContext")
    observed = context.counts  # reassert the raw-count measurement boundary
    expected = np.asarray(expected_counts, dtype=np.float64)
    replicated = np.asarray(replicated_counts)
    if expected.ndim != 2 or expected.shape[0] == 0 or expected.shape[1] != observed.size:
        raise ValueError("expected_counts must have shape (positive n_draws, n_bins)")
    if replicated.shape != expected.shape or not np.issubdtype(replicated.dtype, np.integer):
        raise ValueError("replicated_counts must be integer histograms matching expected_counts")
    if not np.all(np.isfinite(expected)) or np.any(expected <= 0.0):
        raise ValueError("positive finite expected counts are required for deviance diagnostics")
    if np.any(replicated < 0):
        raise ValueError("replicated_counts must be non-negative")

    observed_residuals = np.stack([
        calculate_poisson_deviance_residuals(observed, curve) for curve in expected
    ])
    replicated_residuals = np.stack([
        calculate_poisson_deviance_residuals(counts, curve)
        for counts, curve in zip(replicated, expected)
    ])
    observed_deviance = np.sum(observed_residuals ** 2, axis=1)
    replicated_deviance = np.sum(replicated_residuals ** 2, axis=1)
    observed_rms = np.sqrt(np.mean(observed_residuals ** 2, axis=1))
    replicated_rms = np.sqrt(np.mean(replicated_residuals ** 2, axis=1))
    observed_max = np.max(np.abs(observed_residuals), axis=1)
    replicated_max = np.max(np.abs(replicated_residuals), axis=1)

    windows = []
    for name, bounds in (("early_window_ns", early_window_ns), ("tail_window_ns", tail_window_ns)):
        if bounds is not None:
            windows.append(_window_discrepancy(name, bounds, context, replicated))
    return BayesianPosteriorPredictiveDiagnostics(
        poisson_deviance=_discrepancy(observed_deviance, replicated_deviance),
        rms_signed_deviance_residual=_discrepancy(observed_rms, replicated_rms),
        maximum_absolute_signed_deviance_residual=_discrepancy(
            observed_max, replicated_max
        ),
        mean_observed_signed_deviance_residual_profile=np.mean(observed_residuals, axis=0),
        total_counts=_discrepancy(
            float(np.sum(observed)), np.sum(replicated, axis=1, dtype=np.int64).astype(float)
        ),
        peak_counts=_discrepancy(
            float(np.max(observed)), np.max(replicated, axis=1).astype(float)
        ),
        peak_time_ns=_discrepancy(
            float(context.time_ns[np.argmax(observed)]),
            context.time_ns[np.argmax(replicated, axis=1)],
        ),
        windows=tuple(windows),
    )


@dataclass(frozen=True)
class BayesianPriorPredictiveResult:
    """Prior draws and predictions, with no conditioning on histogram counts.

    Parameter draws have shape (n_draws, 4) in ``PHYSICAL_PARAMETER_ORDER``;
    expected and replicated histograms have shape (n_draws, n_bins).
    """

    physical_parameter_draws: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    replicated_counts: NDArray[np.int64]
    prior_expected_count_band: BayesianPredictiveBand
    prior_predictive_count_band: BayesianPredictiveBand
    priors: BayesianPriorConfig
    model_context: BayesianModelContext
    random_seed: int
    interval_level: float
    runtime_seconds: float
    fixed_irf_assumption: bool = True
    physical_parameter_order: tuple[str, ...] = PHYSICAL_PARAMETER_ORDER

    def __post_init__(self) -> None:
        physical, expected, replicated = _validated_draw_arrays(
            self.physical_parameter_draws, self.expected_counts,
            self.replicated_counts, self.model_context.time_ns.size,
        )
        object.__setattr__(self, "physical_parameter_draws", physical)
        object.__setattr__(self, "expected_counts", expected)
        object.__setattr__(self, "replicated_counts", replicated)


@dataclass(frozen=True)
class BayesianPosteriorPredictiveResult:
    """Posterior expected-count and new-observation predictions, kept separate.

    ``selected_posterior_indices`` has shape (n_draws, 3) and indexes the
    retained (ensemble, step, walker) axes. Parameter draws have shape
    (n_draws, 4); count arrays have shape (n_draws, n_bins). Selection is
    uniform without replacement when possible and with replacement when
    ``n_draws`` exceeds retained sample count. Repeated indices are not new
    independent posterior information.
    """

    selected_posterior_indices: NDArray[np.int64]
    selection_with_replacement: bool
    retained_posterior_sample_count: int
    physical_parameter_draws: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    replicated_counts: NDArray[np.int64]
    posterior_expected_count_band: BayesianPredictiveBand
    posterior_predictive_count_band: BayesianPredictiveBand
    diagnostics: BayesianPosteriorPredictiveDiagnostics
    model_context: BayesianModelContext
    priors: BayesianPriorConfig
    sampling_config: BayesianSamplingConfig
    inference_sampling_status: BayesianSamplingStatus
    random_seed: int
    interval_level: float
    runtime_seconds: float
    fixed_irf_assumption: bool = True
    physical_parameter_order: tuple[str, ...] = PHYSICAL_PARAMETER_ORDER

    def __post_init__(self) -> None:
        physical, expected, replicated = _validated_draw_arrays(
            self.physical_parameter_draws, self.expected_counts,
            self.replicated_counts, self.model_context.time_ns.size,
        )
        indices = np.asarray(self.selected_posterior_indices)
        if indices.shape != (physical.shape[0], 3) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selected_posterior_indices must have shape (n_draws, 3)")
        object.__setattr__(self, "selected_posterior_indices", _immutable_array(indices, dtype=np.int64))
        object.__setattr__(self, "physical_parameter_draws", physical)
        object.__setattr__(self, "expected_counts", expected)
        object.__setattr__(self, "replicated_counts", replicated)


def _propagate_physical_draws(
    context: BayesianModelContext,
    physical: NDArray[np.float64],
    poisson_rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    expected = np.stack([
        monoexponential_reconvolution_expected_counts(
            context.time_ns, context.irf_kernel,
            amplitude=float(amplitude), lifetime=float(lifetime),
            background=float(background), temporal_shift=float(shift),
        )
        for amplitude, lifetime, background, shift in physical
    ])
    if not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
        raise ValueError("predictive expected counts must be finite and non-negative")
    replicated = sample_photon_counts(expected, poisson_rng)
    return expected, replicated


def sample_bayesian_prior_predictive(
    model_context: BayesianModelContext,
    priors: BayesianPriorConfig,
    *,
    n_draws: int,
    random_seed: int,
    interval_level: float,
) -> BayesianPriorPredictiveResult:
    """Propagate explicit physical priors without using observed count values.

    This does not call the data-derived sampler initialization policy or emcee.
    The model context supplies only the time grid and fixed IRF assumption.
    """
    started = time.perf_counter()
    if not isinstance(model_context, BayesianModelContext):
        raise TypeError("model_context must be a BayesianModelContext")
    if not isinstance(priors, BayesianPriorConfig):
        raise TypeError("priors must be a BayesianPriorConfig")
    n_draws, random_seed, interval_level = _validate_request(
        n_draws, random_seed, interval_level
    )
    parameter_seed, poisson_seed = np.random.SeedSequence(random_seed).spawn(2)
    parameter_rng = np.random.default_rng(parameter_seed)
    poisson_rng = np.random.default_rng(poisson_seed)
    physical = np.empty((n_draws, 4), dtype=np.float64)
    for index in range(n_draws):
        draw = _prior_draw(priors, parameter_rng)
        physical[index] = (
            draw.amplitude, draw.lifetime_ns,
            draw.background_per_bin, draw.temporal_shift_ns,
        )
    if not np.all(np.isfinite(physical)) or np.any(physical[:, :3] <= 0.0):
        raise ValueError("prior sampling produced a non-finite or non-physical parameter draw")
    expected, replicated = _propagate_physical_draws(model_context, physical, poisson_rng)
    return BayesianPriorPredictiveResult(
        physical_parameter_draws=physical,
        expected_counts=expected,
        replicated_counts=replicated,
        prior_expected_count_band=summarize_predictive_band(
            expected, interval_level=interval_level
        ),
        prior_predictive_count_band=summarize_predictive_band(
            replicated, interval_level=interval_level
        ),
        priors=priors,
        model_context=model_context,
        random_seed=random_seed,
        interval_level=interval_level,
        runtime_seconds=time.perf_counter() - started,
    )


def sample_bayesian_posterior_predictive(
    inference_run: BayesianInferenceRun,
    *,
    n_draws: int,
    random_seed: int,
    interval_level: float,
    early_window_ns: tuple[float, float] | None = None,
    tail_window_ns: tuple[float, float] | None = None,
) -> BayesianPosteriorPredictiveResult:
    """Predict from retained physical draws; never rerun or thin the MCMC.

    If requested draws exceed retained draws, sample indices with replacement.
    The output remains conditional on the inference run's single fixed IRF.
    """
    started = time.perf_counter()
    if not isinstance(inference_run, BayesianInferenceRun):
        raise TypeError("inference_run must be a BayesianInferenceRun")
    n_draws, random_seed, interval_level = _validate_request(
        n_draws, random_seed, interval_level
    )
    samples = inference_run.samples.physical
    n_retained = math.prod(samples.shape[:3])
    if n_retained == 0:
        raise ValueError("inference_run has no retained posterior samples to predict from")
    index_seed, poisson_seed = np.random.SeedSequence(random_seed).spawn(2)
    index_rng = np.random.default_rng(index_seed)
    poisson_rng = np.random.default_rng(poisson_seed)
    replace = n_draws > n_retained
    flat_indices = index_rng.choice(n_retained, size=n_draws, replace=replace)
    selected_indices = np.column_stack(np.unravel_index(flat_indices, samples.shape[:3]))
    physical = samples.reshape(-1, 4)[flat_indices]
    context = inference_run.context
    expected, replicated = _propagate_physical_draws(context, physical, poisson_rng)
    diagnostics = calculate_bayesian_posterior_predictive_diagnostics(
        context, expected, replicated,
        early_window_ns=early_window_ns,
        tail_window_ns=tail_window_ns,
    )
    return BayesianPosteriorPredictiveResult(
        selected_posterior_indices=selected_indices,
        selection_with_replacement=replace,
        retained_posterior_sample_count=n_retained,
        physical_parameter_draws=physical,
        expected_counts=expected,
        replicated_counts=replicated,
        posterior_expected_count_band=summarize_predictive_band(
            expected, interval_level=interval_level
        ),
        posterior_predictive_count_band=summarize_predictive_band(
            replicated, interval_level=interval_level
        ),
        diagnostics=diagnostics,
        model_context=context,
        priors=inference_run.result.priors,
        sampling_config=inference_run.result.sampling_config,
        inference_sampling_status=inference_run.result.status,
        random_seed=random_seed,
        interval_level=interval_level,
        runtime_seconds=time.perf_counter() - started,
    )
