"""Controlled Issue-4 model-mismatch evaluation, separate from Stage-5 calibration.

Truth is benchmark metadata, never part of a raw TCSPCMeasurement. Every
inference remains conditional on a mono-exponential model and one fixed
prepared IRF; sampler diagnostics and posterior predictive checks do not
certify that either physical assumption is correct.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.bayesian import (
    BayesianPriorConfig,
    BoundedUniformPrior,
    GammaPrior,
    IRFModelRelation,
    LogNormalPrior,
    fit_bayesian_monoexponential_reconvolution,
)
from tcspc_toolkit.bayesian_evaluation import (
    Issue4Bayesian,
    Issue4Bootstrap,
    Issue4ClassicalFit,
    Issue4Covariance,
    _unavailable_bootstrap,
    _unavailable_covariance,
)
from tcspc_toolkit.bayesian_predictive import (
    BayesianPredictiveDiscrepancy,
    sample_bayesian_posterior_predictive,
)
from tcspc_toolkit.bayesian_sampling import (
    BayesianSamplingConfig,
    BayesianSamplingStatus,
)
from tcspc_toolkit.classical_evaluation import fit_single_reconvolution_curve
from tcspc_toolkit.classical_uncertainty import (
    _reconstruct_reconvolution_fit_result,
    estimate_parametric_poisson_bootstrap,
    estimate_poisson_reconvolution_local_covariance,
)
from tcspc_toolkit.fitting import (
    fit_monoexponential_reconvolution,
    poisson_negative_log_likelihood,
)
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import generate_emg_irf_profile, generate_gaussian_irf_profile
from tcspc_toolkit.irf_evaluation import _match_gaussian_to_emg_profile
from tcspc_toolkit.irf_preparation import PreparedIRF, prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.simulation import sample_photon_counts


Mechanism = Literal["decay", "irf"]
GeneratingModel = Literal["monoexponential", "biexponential"]
ReferenceName = Literal["generating_mono", "primary_component", "pseudo_true_mono"]


def _hash_array(values: NDArray, *, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


@dataclass(frozen=True)
class MismatchAssumption:
    assumption_id: str
    irf_id: str
    prepared_irf: PreparedIRF


@dataclass(frozen=True)
class MismatchCondition:
    condition_id: str
    mechanism: Mechanism
    generating_model: GeneratingModel
    time_ns: NDArray[np.float64]
    generating_irf_id: str
    generating_irf: PreparedIRF
    assumptions: tuple[MismatchAssumption, ...]
    primary_lifetime_ns: float
    secondary_lifetime_ns: float | None
    secondary_detected_fraction: float | None
    signal_photon_count: int
    background_per_bin: float
    true_temporal_shift_ns: float

    def __post_init__(self) -> None:
        grid = np.asarray(self.time_ns, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 5 or not np.all(np.isfinite(grid)):
            raise ValueError("time_ns must be a finite one-dimensional grid")
        if np.any(np.diff(grid) <= 0) or not np.allclose(np.diff(grid), np.diff(grid)[0]):
            raise ValueError("time_ns must be uniformly spaced and increasing")
        if self.mechanism not in ("decay", "irf"):
            raise ValueError("unknown mismatch mechanism")
        if self.generating_model not in ("monoexponential", "biexponential"):
            raise ValueError("unknown generating decay model")
        if not math.isfinite(self.primary_lifetime_ns) or self.primary_lifetime_ns <= 0:
            raise ValueError("primary lifetime must be finite and positive")
        if self.generating_model == "biexponential":
            if (
                self.mechanism != "decay"
                or self.secondary_lifetime_ns is None
                or not math.isfinite(self.secondary_lifetime_ns)
                or self.secondary_lifetime_ns <= 0
                or self.secondary_detected_fraction is None
                or not 0 < self.secondary_detected_fraction < 1
            ):
                raise ValueError("bi-exponential truth requires two lifetimes and a detected fraction")
        elif self.secondary_lifetime_ns is not None or self.secondary_detected_fraction is not None:
            raise ValueError("mono-exponential truth cannot have secondary-component metadata")
        if self.signal_photon_count <= 0 or not isinstance(self.signal_photon_count, int):
            raise ValueError("signal_photon_count must be a positive integer")
        if not math.isfinite(self.background_per_bin) or self.background_per_bin < 0:
            raise ValueError("background must be finite and nonnegative")
        if not math.isfinite(self.true_temporal_shift_ns):
            raise ValueError("true shift must be finite")
        if not np.array_equal(grid, self.generating_irf.time_ns):
            raise ValueError("generating IRF must use the condition grid")
        if not self.assumptions or len({a.assumption_id for a in self.assumptions}) != len(self.assumptions):
            raise ValueError("assumption IDs must be unique and nonempty")
        for assumption in self.assumptions:
            if not assumption.assumption_id or not np.array_equal(grid, assumption.prepared_irf.time_ns):
                raise ValueError("each assumed IRF must use the condition grid")
        matches = [
            np.array_equal(self.generating_irf.kernel, item.prepared_irf.kernel)
            for item in self.assumptions
        ]
        if self.mechanism == "decay" and (len(matches) != 1 or not matches[0]):
            raise ValueError("decay study requires one matched fixed-IRF assumption")
        if self.mechanism == "irf" and (
            self.generating_model != "monoexponential"
            or len(matches) != 2 or sorted(matches) != [False, True]
        ):
            raise ValueError("IRF study requires one matched and one misspecified assumption")
        copied = np.array(grid, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "time_ns", copied)

    @property
    def mono_generating_lifetime_ns(self) -> float | None:
        """No scalar generating mono lifetime exists for bi-exponential data."""
        return self.primary_lifetime_ns if self.generating_model == "monoexponential" else None


@dataclass(frozen=True)
class MismatchEvaluationConfig:
    profile: Literal["smoke", "scientific"]
    priors: BayesianPriorConfig
    sampling_config: BayesianSamplingConfig
    n_bootstrap_resamples: int
    n_posterior_predictive_draws: int
    nominal_interval_level: float
    temporal_shift_bounds_ns: tuple[float, float]
    early_window_ns: tuple[float, float]
    tail_window_ns: tuple[float, float]
    background_fraction: float
    seed_namespace: str
    base_seed: int
    n_repeats_per_condition: int

    def __post_init__(self) -> None:
        if self.profile not in ("smoke", "scientific"):
            raise ValueError("unknown mismatch profile")
        if self.n_bootstrap_resamples < 2 or self.n_posterior_predictive_draws < 1:
            raise ValueError("bootstrap and predictive budgets must be positive")
        if self.n_repeats_per_condition < 1 or self.base_seed < 0:
            raise ValueError("repeats and base seed must be valid")
        if self.seed_namespace != "issue4:stage6:v1":
            raise ValueError("Stage-6 seed namespace must be issue4:stage6:v1")
        if self.nominal_interval_level != self.sampling_config.credible_interval_level:
            raise ValueError("classical and Bayesian interval levels must agree")
        if not self.priors.infer_temporal_shift or self.priors.temporal_shift_prior is None:
            raise ValueError("Stage 6 requires a free residual shift")
        if self.temporal_shift_bounds_ns != (
            self.priors.temporal_shift_prior.lower_ns,
            self.priors.temporal_shift_prior.upper_ns,
        ):
            raise ValueError("classical and Bayesian shift bounds must agree")
        for name, window in (("early", self.early_window_ns), ("tail", self.tail_window_ns)):
            if len(window) != 2 or not all(map(math.isfinite, window)) or window[0] >= window[1]:
                raise ValueError(f"{name} window must have finite increasing bounds")


@dataclass(frozen=True)
class MismatchWorkflow:
    conditions: tuple[MismatchCondition, ...]
    config: MismatchEvaluationConfig


def build_mismatch_workflow(manifest: Mapping, profile: str) -> MismatchWorkflow:
    """Build prepared IRFs and typed Stage-6 design from its separate manifest."""
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported mismatch manifest schema")
    grid = manifest["time_grid_ns"]
    time_axis = np.arange(grid["start"], grid["stop"], grid["step"], dtype=np.float64)
    window = manifest["measurement_window_ns"]
    if window != [float(time_axis[0]), float(grid["stop"])]:
        raise ValueError("measurement window must equal the explicit half-open grid")
    early = tuple(manifest["predictive_windows_ns"]["early"])
    tail = tuple(manifest["predictive_windows_ns"]["tail"])
    for bounds in (early, tail):
        if bounds[0] < time_axis[0] or bounds[1] > grid["stop"]:
            raise ValueError("predictive windows must stay inside the measurement window")
        if not np.any((time_axis >= bounds[0]) & (time_axis < bounds[1])):
            raise ValueError("predictive window contains no bins")
    irf_spec = manifest["irfs"]
    gaussian_spec = irf_spec["gaussian"]
    emg_spec = irf_spec["emg"]
    gaussian_source = generate_gaussian_irf_profile(
        time_axis,
        gaussian_centre_ns=gaussian_spec["centre_ns"],
        gaussian_fwhm_ns=gaussian_spec["fwhm_ns"],
        provenance={"workflow": "issue4_stage6"},
    )
    emg_source = generate_emg_irf_profile(
        time_axis,
        gaussian_centre_ns=emg_spec["gaussian_component_centre_ns"],
        gaussian_fwhm_ns=emg_spec["gaussian_component_fwhm_ns"],
        tail_time_ns=emg_spec["tail_time_ns"],
        provenance={"workflow": "issue4_stage6"},
    )
    # The Issue-8 sampled peak/full-FWHM matcher is the single source of truth.
    comparison_source = _match_gaussian_to_emg_profile(emg_source)
    irfs = {
        "gaussian": prepare_irf(gaussian_source, time_axis),
        "emg": prepare_irf(emg_source, time_axis),
        "emg_peak_fwhm_gaussian": prepare_irf(comparison_source, time_axis),
    }
    common = manifest["common_model"]
    conditions = tuple(
        MismatchCondition(
            condition_id=spec["id"],
            mechanism=spec["mechanism"],
            generating_model=spec["generating_model"],
            time_ns=time_axis,
            generating_irf_id=spec["generating_irf"],
            generating_irf=irfs[spec["generating_irf"]],
            assumptions=tuple(
                MismatchAssumption(item["id"], item["irf"], irfs[item["irf"]])
                for item in spec["assumptions"]
            ),
            primary_lifetime_ns=spec["primary_lifetime_ns"],
            secondary_lifetime_ns=spec.get("secondary_lifetime_ns"),
            secondary_detected_fraction=spec.get("secondary_detected_fraction"),
            signal_photon_count=common["signal_photon_count"],
            background_per_bin=common["background_per_bin"],
            true_temporal_shift_ns=common["true_temporal_shift_ns"],
        )
        for spec in manifest["conditions"]
    )
    prior_spec = manifest["priors"]
    shift_lower, shift_upper = prior_spec["shift_uniform_bounds_ns"]
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(*prior_spec["amplitude_gamma_shape_rate"]),
        lifetime_ns=LogNormalPrior(*prior_spec["lifetime_lognormal_log_mean_std"]),
        background_per_bin=GammaPrior(*prior_spec["background_gamma_shape_rate"]),
        temporal_shift_prior=BoundedUniformPrior(shift_lower, shift_upper),
        fixed_temporal_shift_ns=None,
    )
    selected = manifest["profiles"][profile]
    sampler = BayesianSamplingConfig(
        random_seed=0,
        credible_interval_level=common["nominal_interval_level"],
        **selected["bayesian_sampling"],
    )
    config = MismatchEvaluationConfig(
        profile=profile,
        priors=priors,
        sampling_config=sampler,
        n_bootstrap_resamples=selected["n_bootstrap_resamples"],
        n_posterior_predictive_draws=selected["n_posterior_predictive_draws"],
        nominal_interval_level=common["nominal_interval_level"],
        temporal_shift_bounds_ns=tuple(common["temporal_shift_bounds_ns"]),
        early_window_ns=early,
        tail_window_ns=tail,
        background_fraction=common["classical_background_fraction"],
        seed_namespace=manifest["randomness"]["namespace"],
        base_seed=manifest["randomness"]["base_seed"],
        n_repeats_per_condition=selected["n_repeats_per_condition"],
    )
    if any(not config.temporal_shift_bounds_ns[0] <= c.true_temporal_shift_ns <=
           config.temporal_shift_bounds_ns[1] for c in conditions):
        raise ValueError("true shift lies outside shared inference bounds")
    return MismatchWorkflow(conditions=conditions, config=config)


@dataclass(frozen=True)
class MismatchSeeds:
    observation: int
    bootstrap: int
    bayesian: int
    posterior_predictive: int


def mismatch_seeds(config: MismatchEvaluationConfig, condition_id: str, index: int) -> MismatchSeeds:
    """Independent named streams, deliberately shared inside an IRF pair."""
    if index < 0 or not condition_id:
        raise ValueError("invalid condition or realization index")
    def derive(stream: str) -> int:
        key = f"{config.seed_namespace}:{config.base_seed}:{condition_id}:{index}:{stream}"
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    seeds = MismatchSeeds(*(derive(name) for name in (
        "observation", "bootstrap", "bayesian", "posterior_predictive"
    )))
    if len(set(vars(seeds).values())) != 4:
        raise RuntimeError("named random streams unexpectedly collided")
    return seeds


@dataclass(frozen=True)
class GeneratingCounts:
    expected_counts: NDArray[np.float64]
    primary_expected_signal: NDArray[np.float64]
    secondary_expected_signal: NDArray[np.float64]


def build_mismatch_expected_counts(condition: MismatchCondition) -> GeneratingCounts:
    """Apply Week-9's detected finite-window component-fraction convention."""
    def unit(lifetime: float) -> NDArray[np.float64]:
        curve = monoexponential_reconvolution_expected_counts(
            condition.time_ns, condition.generating_irf.kernel,
            amplitude=1.0, lifetime=lifetime, background=0.0,
            temporal_shift=condition.true_temporal_shift_ns,
        )
        if not np.all(np.isfinite(curve)) or np.any(curve < 0) or curve.sum() <= 0:
            raise ValueError("unit convolved signal is invalid")
        return curve / curve.sum()
    fraction = condition.secondary_detected_fraction or 0.0
    primary = condition.signal_photon_count * (1.0 - fraction) * unit(
        condition.primary_lifetime_ns
    )
    secondary = np.zeros_like(primary)
    if condition.generating_model == "biexponential":
        assert condition.secondary_lifetime_ns is not None
        secondary = condition.signal_photon_count * fraction * unit(
            condition.secondary_lifetime_ns
        )
    expected = primary + secondary + condition.background_per_bin
    return GeneratingCounts(expected, primary, secondary)


@dataclass(frozen=True)
class PseudoTrueReference:
    """Prior-free deterministic projection onto the assumed mono model."""
    condition_id: str
    assumption_id: str
    amplitude: float
    lifetime_ns: float
    background_per_bin: float
    temporal_shift_ns: float
    poisson_nll: float
    n_starts: int
    max_start_objective_gap: float
    max_start_lifetime_gap_ns: float
    active_bounds: tuple[str, ...]
    optimizer_status: int | None
    optimizer_message: str | None
    optimizer_nfev: int | None
    maximum_coordinate_descent_nll: float


def construct_pseudo_true_reference(
    condition: MismatchCondition,
    assumption: MismatchAssumption,
    expected_counts: NDArray[np.float64],
    temporal_shift_bounds_ns: tuple[float, float],
) -> PseudoTrueReference:
    """Fit fractional noise-free expectations with four deterministic starts.

    This is an offline, prior-free likelihood projection. All starts must pass
    Stage-5.5's Poisson local check and agree within 0.2 NLL and 0.005 ns;
    otherwise no reference is reported. The explicit bounds and fit diagnostics
    make ambiguity inspectable rather than silently selecting one basin.
    """
    expected = np.asarray(expected_counts, dtype=np.float64)
    if expected.shape != condition.time_ns.shape or not np.all(np.isfinite(expected)):
        raise ValueError("expected counts must match the finite measurement grid")
    if np.any(expected <= 0):
        raise ValueError("pseudo-true objective requires positive expectations")
    if not any(assumption is item for item in condition.assumptions):
        raise ValueError("assumption does not belong to condition")
    starts = (
        (0.75 * condition.primary_lifetime_ns, 0.0),
        (condition.primary_lifetime_ns, condition.true_temporal_shift_ns),
        (condition.secondary_lifetime_ns or 1.5 * condition.primary_lifetime_ns, 0.0),
        (2.0 * condition.primary_lifetime_ns, condition.true_temporal_shift_ns),
    )
    candidates = []
    for lifetime_start, shift_start in starts:
        unit = monoexponential_reconvolution_expected_counts(
            condition.time_ns, assumption.prepared_irf.kernel,
            amplitude=1.0, lifetime=lifetime_start, background=0.0,
            temporal_shift=shift_start,
        )
        amplitude_start = condition.signal_photon_count / float(np.sum(unit))
        candidate = fit_monoexponential_reconvolution(
            time=condition.time_ns, counts=expected,
            irf=assumption.prepared_irf.kernel,
            initial_guess=(
                amplitude_start, lifetime_start,
                max(condition.background_per_bin, 1e-8), shift_start,
            ),
            temporal_shift_bounds=temporal_shift_bounds_ns,
            objective="poisson",
        )
        if not candidate.success or candidate.numerical_validation_passed is not True:
            raise RuntimeError(
                f"pseudo-true optimization failed numerical validation for "
                f"{condition.condition_id}/{assumption.assumption_id}: "
                f"{candidate.optimizer_message}"
            )
        objective = poisson_negative_log_likelihood(expected, candidate.fitted_curve)
        if not math.isfinite(objective):
            raise RuntimeError("pseudo-true objective is nonfinite")
        candidates.append((objective, candidate))
    best_objective, best = min(candidates, key=lambda item: item[0])
    objective_gap = max(item[0] - best_objective for item in candidates)
    lifetime_gap = max(abs(item[1].lifetime - best.lifetime) for item in candidates)
    if objective_gap > 0.2 or lifetime_gap > 0.005:
        raise RuntimeError(
            f"pseudo-true starts disagree for {condition.condition_id}/"
            f"{assumption.assumption_id}: NLL gap={objective_gap:.6g}, "
            f"lifetime gap={lifetime_gap:.6g} ns"
        )
    active = []
    if best.amplitude <= 1e-10:
        active.append("amplitude_lower")
    if best.lifetime <= 1e-10:
        active.append("lifetime_lower")
    if best.background <= 1e-8:
        active.append("background_lower")
    if abs(best.temporal_shift - temporal_shift_bounds_ns[0]) <= 1e-6:
        active.append("shift_lower")
    if abs(best.temporal_shift - temporal_shift_bounds_ns[1]) <= 1e-6:
        active.append("shift_upper")
    return PseudoTrueReference(
        condition_id=condition.condition_id,
        assumption_id=assumption.assumption_id,
        amplitude=best.amplitude,
        lifetime_ns=best.lifetime,
        background_per_bin=best.background,
        temporal_shift_ns=best.temporal_shift,
        poisson_nll=best_objective,
        n_starts=len(starts),
        max_start_objective_gap=objective_gap,
        max_start_lifetime_gap_ns=lifetime_gap,
        active_bounds=tuple(active),
        optimizer_status=best.optimizer_status,
        optimizer_message=best.optimizer_message,
        optimizer_nfev=best.optimizer_nfev,
        maximum_coordinate_descent_nll=best.max_coordinate_descent_nll,
    )


@dataclass(frozen=True)
class MismatchReferenceComparison:
    """Per-realization deviation and interval inclusion, conditional on a named target."""

    method: Literal["covariance", "bootstrap", "bayesian"]
    interval_kind: str
    reference_name: ReferenceName
    reference_lifetime_ns: float
    point_lifetime_ns: float
    deviation_ns: float
    reported_std_ns: float
    interval_lower_ns: float
    interval_upper_ns: float
    interval_width_ns: float
    reference_included: bool | None
    deviation_to_reported_std: float


@dataclass(frozen=True)
class MismatchPredictiveSummary:
    """Compact descriptive PPC evidence, never a model-validity decision."""

    status: Literal["success", "not_available", "failed"]
    reason: str | None
    n_draws: int
    runtime_seconds: float
    discrepancy_summaries: tuple[tuple[str, float, float, float], ...]
    mean_signed_deviance_residual_profile: tuple[float, ...]


@dataclass(frozen=True)
class MismatchInferenceRecord:
    condition_id: str
    mechanism: Mechanism
    generating_model: GeneratingModel
    assumed_decay_model: Literal["monoexponential"]
    assumption_id: str
    assumed_irf_id: str
    generating_irf_id: str
    profile: str
    realization_index: int
    sample_id: str
    primary_lifetime_ns: float
    secondary_lifetime_ns: float | None
    secondary_detected_fraction: float | None
    mono_generating_lifetime_ns: float | None
    pseudo_true_mono_lifetime_ns: float
    physical_model_discrepancy_ns: float
    signal_photon_count: int
    background_per_bin: float
    true_temporal_shift_ns: float
    observed_total_counts: int
    observed_counts_sha256: str
    expected_counts_sha256: str
    time_grid_sha256: str
    generating_irf_sha256: str
    assumed_irf_sha256: str
    seeds: MismatchSeeds
    classical_fit: Issue4ClassicalFit
    covariance: Issue4Covariance
    bootstrap: Issue4Bootstrap
    bayesian: Issue4Bayesian
    posterior_predictive: MismatchPredictiveSummary
    comparisons: tuple[MismatchReferenceComparison, ...]
    orchestration_seconds: float


@dataclass(frozen=True)
class MismatchGroupSummary:
    condition_id: str
    assumption_id: str
    method: str
    empirical_estimator: str
    reference_name: ReferenceName
    n_attempted: int
    n_valid_point_estimates: int
    n_valid_intervals: int
    n_reference_included: int
    bias_ns: float
    mae_ns: float
    empirical_spread_ns: float
    mean_reported_std_ns: float
    mean_interval_width_ns: float
    inclusion_among_valid: float
    successful_and_including_fraction: float
    mean_deviation_to_mean_reported_std: float


@dataclass(frozen=True)
class MismatchReport:
    workflow: MismatchWorkflow
    generating_expected_counts: tuple[tuple[str, tuple[float, ...]], ...]
    pseudo_true_references: tuple[PseudoTrueReference, ...]
    per_inference: tuple[MismatchInferenceRecord, ...]
    summaries: tuple[MismatchGroupSummary, ...]
    runtime_seconds: float


def sample_mismatch_observation(
    condition: MismatchCondition, generating: GeneratingCounts,
    seeds: MismatchSeeds, realization_index: int,
) -> TCSPCMeasurement:
    """Sample once; physical truth stays in condition metadata, not measurement."""
    counts = sample_photon_counts(generating.expected_counts, np.random.default_rng(seeds.observation))
    return TCSPCMeasurement(
        time_ns=condition.time_ns,
        values=counts,
        data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id=f"issue4-stage6-{condition.condition_id}-{realization_index}",
        provenance={
            "origin": "issue4_stage6_independent_synthetic",
            "condition_id": condition.condition_id,
            "seed_namespace": "issue4:stage6:v1",
            "observation_seed": seeds.observation,
        },
    )


def _compact_bayesian(inference, call_seconds: float) -> Issue4Bayesian:
    result = inference.result
    lifetime = result.parameter_summaries[1] if len(result.parameter_summaries) > 1 else None
    corr = result.correlation_matrix
    effective = result.diagnostics.approximate_effective_samples
    finite_effective = effective[np.isfinite(effective)]
    autocorrelation = result.diagnostics.autocorrelation_time_steps
    finite_autocorrelation = autocorrelation[np.isfinite(autocorrelation) & (autocorrelation > 0)]
    relative_change = result.diagnostics.autocorrelation_relative_change
    finite_change = relative_change[np.isfinite(relative_change)]
    return Issue4Bayesian(
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
        call_seconds=call_seconds,
    )


def _compact_discrepancy(
    name: str, discrepancy: BayesianPredictiveDiscrepancy,
) -> tuple[str, float, float, float]:
    observed = float(np.mean(np.asarray(discrepancy.observed, dtype=float)))
    return (
        name, observed, float(np.median(discrepancy.replicated)),
        discrepancy.posterior_predictive_tail_probability,
    )


def _compact_predictive(predictive) -> MismatchPredictiveSummary:
    diagnostics = predictive.diagnostics
    named = (
        ("poisson_deviance", diagnostics.poisson_deviance),
        ("residual_rms", diagnostics.rms_signed_deviance_residual),
        ("maximum_absolute_residual", diagnostics.maximum_absolute_signed_deviance_residual),
        ("total_counts", diagnostics.total_counts),
        ("peak_counts", diagnostics.peak_counts),
        ("peak_time_ns", diagnostics.peak_time_ns),
    ) + tuple((f"{window.name.removesuffix('_window_ns')}_window_counts", window.total_counts)
              for window in diagnostics.windows)
    return MismatchPredictiveSummary(
        status="success", reason=None, n_draws=int(predictive.replicated_counts.shape[0]),
        runtime_seconds=predictive.runtime_seconds,
        discrepancy_summaries=tuple(_compact_discrepancy(name, value) for name, value in named),
        mean_signed_deviance_residual_profile=tuple(
            float(value) for value in diagnostics.mean_observed_signed_deviance_residual_profile
        ),
    )


def _reference_comparisons(
    condition: MismatchCondition, pseudo_true: PseudoTrueReference,
    classical: Issue4ClassicalFit, covariance: Issue4Covariance,
    bootstrap: Issue4Bootstrap, bayesian: Issue4Bayesian,
) -> tuple[MismatchReferenceComparison, ...]:
    physical_name: ReferenceName = (
        "generating_mono" if condition.generating_model == "monoexponential" else "primary_component"
    )
    references: tuple[tuple[ReferenceName, float], ...] = (
        (physical_name, condition.primary_lifetime_ns),
        ("pseudo_true_mono", pseudo_true.lifetime_ns),
    )
    methods = (
        ("covariance", "local_gaussian_covariance_interval", classical.lifetime_ns,
         covariance.lifetime_std_ns, covariance.local_gaussian_lower_ns,
         covariance.local_gaussian_upper_ns, covariance.valid_interval),
        ("bootstrap", "parametric_bootstrap_percentile_interval", classical.lifetime_ns,
         bootstrap.lifetime_std_ns, bootstrap.percentile_lower_ns,
         bootstrap.percentile_upper_ns, bootstrap.valid_interval),
        ("bayesian", "bayesian_equal_tailed_credible_interval", bayesian.lifetime_median_ns,
         bayesian.posterior_lifetime_std_ns, bayesian.credible_lower_ns,
         bayesian.credible_upper_ns, bayesian.valid_interval),
    )
    comparisons = []
    for reference_name, reference in references:
        for method, kind, estimate, std, lower, upper, valid in methods:
            point_valid = math.isfinite(estimate) and (
                bayesian.valid_interval if method == "bayesian" else classical.valid_fit
            )
            deviation = estimate - reference if point_valid else math.nan
            interval_valid = valid and all(map(math.isfinite, (lower, upper))) and lower <= upper
            comparisons.append(MismatchReferenceComparison(
                method=method, interval_kind=kind,
                reference_name=reference_name, reference_lifetime_ns=reference,
                point_lifetime_ns=estimate if point_valid else math.nan,
                deviation_ns=deviation,
                reported_std_ns=std if interval_valid else math.nan,
                interval_lower_ns=lower if interval_valid else math.nan,
                interval_upper_ns=upper if interval_valid else math.nan,
                interval_width_ns=upper - lower if interval_valid else math.nan,
                reference_included=(lower <= reference <= upper) if interval_valid else None,
                deviation_to_reported_std=(deviation / std)
                if interval_valid and std > 0 and math.isfinite(deviation) else math.nan,
            ))
    return tuple(comparisons)


def evaluate_mismatch_inference(
    condition: MismatchCondition, assumption: MismatchAssumption,
    config: MismatchEvaluationConfig, measurement: TCSPCMeasurement,
    generating: GeneratingCounts, pseudo_true: PseudoTrueReference,
    seeds: MismatchSeeds, realization_index: int,
) -> MismatchInferenceRecord:
    """Evaluate one raw buffer with all existing model-conditional methods."""
    started = time.perf_counter()
    counts = measurement.require_raw_counts()
    if not np.array_equal(measurement.time_ns, condition.time_ns):
        raise ValueError("measurement and condition grids differ")
    if not any(assumption is item for item in condition.assumptions):
        raise ValueError("assumption does not belong to condition")
    if (pseudo_true.condition_id, pseudo_true.assumption_id) != (
        condition.condition_id, assumption.assumption_id
    ):
        raise ValueError("pseudo-true reference and assumption differ")
    kernel = assumption.prepared_irf.kernel
    fit_started = time.perf_counter()
    fitted = fit_single_reconvolution_curve(
        time=condition.time_ns, counts=counts, irf=kernel,
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
        optimizer_seconds=fitted.runtime_ms / 1000.0, call_seconds=fit_call_seconds,
        numerical_validation_passed=fitted.numerical_validation_passed,
        max_coordinate_descent_nll=fitted.max_coordinate_descent_nll,
        recovery_attempted=fitted.recovery_attempted,
        optimizer_status=fitted.optimizer_status, optimizer_message=fitted.optimizer_message,
        optimizer_nfev=fitted.optimizer_nfev, optimizer_njev=fitted.optimizer_njev,
    )
    if fitted.valid_fit:
        fit_result = _reconstruct_reconvolution_fit_result(
            time=condition.time_ns, irf=kernel, curve_result=fitted,
        )
        covariance_started = time.perf_counter()
        covariance_result = estimate_poisson_reconvolution_local_covariance(
            time=condition.time_ns, irf=kernel, fit_result=fit_result,
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
            time=condition.time_ns, irf=kernel, fit_result=fit_result,
            temporal_shift_bounds=config.temporal_shift_bounds_ns,
            rng=np.random.default_rng(seeds.bootstrap),
            n_resamples=config.n_bootstrap_resamples,
            nominal_coverage=config.nominal_interval_level,
            background_fraction=config.background_fraction,
        )
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
            runtime_seconds=time.perf_counter() - bootstrap_started,
        )
    else:
        covariance = _unavailable_covariance("classical_fit_invalid")
        bootstrap = _unavailable_bootstrap("classical_fit_invalid", config.n_bootstrap_resamples)
    bayesian_started = time.perf_counter()
    relation = (
        IRFModelRelation.MATCHED if np.array_equal(kernel, condition.generating_irf.kernel)
        else IRFModelRelation.DELIBERATELY_MISSPECIFIED
    )
    inference = fit_bayesian_monoexponential_reconvolution(
        measurement, priors=config.priors,
        sampling_config=replace(config.sampling_config, random_seed=seeds.bayesian),
        prepared_irf=assumption.prepared_irf, irf_model_relation=relation,
    )
    bayesian = _compact_bayesian(inference, time.perf_counter() - bayesian_started)
    if bayesian.valid_interval:
        predictive_started = time.perf_counter()
        try:
            predictive = sample_bayesian_posterior_predictive(
                inference, n_draws=config.n_posterior_predictive_draws,
                random_seed=seeds.posterior_predictive,
                interval_level=config.nominal_interval_level,
                early_window_ns=config.early_window_ns,
                tail_window_ns=config.tail_window_ns,
            )
            predictive_summary = _compact_predictive(predictive)
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            predictive_summary = MismatchPredictiveSummary(
                "failed", f"{type(exc).__name__}: {exc}", 0,
                time.perf_counter() - predictive_started, (), (),
            )
    else:
        predictive_summary = MismatchPredictiveSummary(
            "not_available", f"bayesian_{bayesian.status.value}", 0, 0.0, (), (),
        )
    comparisons = _reference_comparisons(
        condition, pseudo_true, classical, covariance, bootstrap, bayesian,
    )
    return MismatchInferenceRecord(
        condition_id=condition.condition_id, mechanism=condition.mechanism,
        generating_model=condition.generating_model, assumed_decay_model="monoexponential",
        assumption_id=assumption.assumption_id,
        assumed_irf_id=assumption.irf_id, generating_irf_id=condition.generating_irf_id,
        profile=config.profile, realization_index=realization_index,
        sample_id=measurement.sample_id or "",
        primary_lifetime_ns=condition.primary_lifetime_ns,
        secondary_lifetime_ns=condition.secondary_lifetime_ns,
        secondary_detected_fraction=condition.secondary_detected_fraction,
        mono_generating_lifetime_ns=condition.mono_generating_lifetime_ns,
        pseudo_true_mono_lifetime_ns=pseudo_true.lifetime_ns,
        physical_model_discrepancy_ns=pseudo_true.lifetime_ns - condition.primary_lifetime_ns,
        signal_photon_count=condition.signal_photon_count,
        background_per_bin=condition.background_per_bin,
        true_temporal_shift_ns=condition.true_temporal_shift_ns,
        observed_total_counts=int(np.sum(counts)),
        observed_counts_sha256=_hash_array(counts, dtype="<i8"),
        expected_counts_sha256=_hash_array(generating.expected_counts, dtype="<f8"),
        time_grid_sha256=_hash_array(condition.time_ns, dtype="<f8"),
        generating_irf_sha256=_hash_array(condition.generating_irf.kernel, dtype="<f8"),
        assumed_irf_sha256=_hash_array(kernel, dtype="<f8"),
        seeds=seeds, classical_fit=classical, covariance=covariance,
        bootstrap=bootstrap, bayesian=bayesian,
        posterior_predictive=predictive_summary, comparisons=comparisons,
        orchestration_seconds=time.perf_counter() - started,
    )


def summarize_mismatch_records(
    records: tuple[MismatchInferenceRecord, ...],
) -> tuple[MismatchGroupSummary, ...]:
    """Reconstruct descriptive group metrics from compact inference records."""
    if not records:
        raise ValueError("at least one inference record is required")
    groups: dict[tuple[str, str, str, ReferenceName], list[MismatchReferenceComparison]] = {}
    for record in records:
        for comparison in record.comparisons:
            key = (record.condition_id, record.assumption_id,
                   comparison.method, comparison.reference_name)
            groups.setdefault(key, []).append(comparison)
    summaries = []
    for (condition_id, assumption_id, method, reference_name), values in groups.items():
        point = np.asarray([v.deviation_ns for v in values if math.isfinite(v.deviation_ns)])
        valid = [v for v in values if v.reference_included is not None]
        widths = [v.interval_width_ns for v in valid]
        stds = [v.reported_std_ns for v in valid if math.isfinite(v.reported_std_ns)]
        n_included = sum(v.reference_included is True for v in valid)
        mean_deviation = float(np.mean(point)) if point.size else math.nan
        mean_std = float(np.mean(stds)) if stds else math.nan
        summaries.append(MismatchGroupSummary(
            condition_id=condition_id, assumption_id=assumption_id,
            method=method,
            empirical_estimator=("bayesian_posterior_median" if method == "bayesian"
                                 else "classical_poisson_lifetime"),
            reference_name=reference_name,
            n_attempted=len(values), n_valid_point_estimates=int(point.size),
            n_valid_intervals=len(valid), n_reference_included=n_included,
            bias_ns=mean_deviation,
            mae_ns=float(np.mean(np.abs(point))) if point.size else math.nan,
            empirical_spread_ns=float(np.std(point, ddof=1)) if point.size >= 2 else math.nan,
            mean_reported_std_ns=mean_std,
            mean_interval_width_ns=float(np.mean(widths)) if widths else math.nan,
            inclusion_among_valid=n_included / len(valid) if valid else math.nan,
            successful_and_including_fraction=n_included / len(values),
            mean_deviation_to_mean_reported_std=(mean_deviation / mean_std)
            if math.isfinite(mean_std) and mean_std > 0 else math.nan,
        ))
    return tuple(summaries)


def evaluate_mismatch_workflow(
    workflow: MismatchWorkflow,
    *,
    progress: Callable[[MismatchInferenceRecord, int, int], None] | None = None,
) -> MismatchReport:
    """Run a selected profile; the scientific profile is manual-run only."""
    started = time.perf_counter()
    curves = []
    references = []
    records = []
    total = workflow.config.n_repeats_per_condition * sum(
        len(condition.assumptions) for condition in workflow.conditions
    )
    for condition in workflow.conditions:
        generating = build_mismatch_expected_counts(condition)
        curves.append((condition.condition_id, tuple(float(x) for x in generating.expected_counts)))
        by_assumption = {}
        for assumption in condition.assumptions:
            reference = construct_pseudo_true_reference(
                condition, assumption, generating.expected_counts,
                workflow.config.temporal_shift_bounds_ns,
            )
            references.append(reference)
            by_assumption[assumption.assumption_id] = reference
        for index in range(workflow.config.n_repeats_per_condition):
            seeds = mismatch_seeds(workflow.config, condition.condition_id, index)
            measurement = sample_mismatch_observation(condition, generating, seeds, index)
            for assumption in condition.assumptions:
                record = evaluate_mismatch_inference(
                    condition, assumption, workflow.config, measurement,
                    generating, by_assumption[assumption.assumption_id], seeds, index,
                )
                records.append(record)
                if progress is not None:
                    progress(record, len(records), total)
    return MismatchReport(
        workflow=workflow, generating_expected_counts=tuple(curves),
        pseudo_true_references=tuple(references), per_inference=tuple(records),
        summaries=summarize_mismatch_records(tuple(records)),
        runtime_seconds=time.perf_counter() - started,
    )
