"""Development K=3 Poisson reconvolution; explicit settings, no selected default.

Numerical acceptance is neither component recoverability nor scientific qualification.
The entry point asserts raw-count semantics; measurement adapters remain separate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import ArrayLike, NDArray
import numpy as np

from tcspc_toolkit.estimation._fixed_k import (
    _FixedKParameters,
    _DerivativePolicy,
    _positive_integer,
    _scalar,
)
from tcspc_toolkit.evaluation.diagnostics import PoissonFitStatistics
from tcspc_toolkit.irf.preparation import PreparedIRF, IRFPreparationDiagnostics


@dataclass(frozen=True)
class TriexponentialParameters:
    """Three pre-exponential amplitudes, ns lifetimes/shift, counts/bin background."""

    amplitudes: tuple[float, float, float]
    lifetimes_ns: tuple[float, float, float]
    background: float = 0.0
    temporal_shift_ns: float = 0.0

    def __post_init__(self):
        checked = _FixedKParameters(
            self.amplitudes, self.lifetimes_ns, self.background, self.temporal_shift_ns
        )
        if checked.n_components != 3:
            raise ValueError("exactly three components required")
        for name in ("amplitudes", "lifetimes_ns", "background", "temporal_shift_ns"):
            object.__setattr__(self, name, getattr(checked, name))


@dataclass(frozen=True)
class TriexponentialFitSettings:
    """Explicit development work allocations. No production policy is selected.

    Start limits count nonlinear invocations, including conditional/recovery solves.
    NNLS projections have their own counter. Twenty percent of optimization model
    calls are reserved for required recovery; optional continuations are disabled.
    """

    max_seed_starts: int
    reserved_recovery_starts: int
    max_optimization_calls: int
    wall_seconds: float
    policy: str = "custom"
    max_initialization_calls: int = 512
    max_validation_calls: int = 1024
    max_assessment_calls: int = 32
    max_diagnostic_calls: int = 256
    max_iterations: int = 1000
    max_line_search: int = 40
    ftol: float = 1e-12
    gtol: float = 1e-6
    derivative_step: float = 1e-4
    derivative_rtol: float = 2e-4
    derivative_atol: float = 1e-6
    derivative_reductions: int = 3
    validation_steps: tuple[float, ...] = (1e-4, 1e-2)
    descent_tolerance: float = 0.001
    solver: str = "lbfgsb"
    recovery_solver: str | None = None

    def __post_init__(self):
        if self.solver not in ("lbfgsb", "trf") or self.recovery_solver not in (
            None,
            "lbfgsb",
            "trf",
        ):
            raise ValueError("solver must be lbfgsb or trf")
        for name in (
            "max_seed_starts",
            "max_optimization_calls",
            "max_iterations",
            "max_line_search",
            "derivative_reductions",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in (
            "reserved_recovery_starts",
            "max_initialization_calls",
            "max_validation_calls",
            "max_assessment_calls",
            "max_diagnostic_calls",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        if not 10 <= self.max_seed_starts <= 109:
            raise ValueError("seed capacity must be between 10 and 109")
        if self.policy not in ("custom", "compact", "balanced", "exhaustive"):
            raise ValueError("unknown development policy")
        for name in ("wall_seconds", "ftol", "gtol", "derivative_step"):
            value = _scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("derivative_rtol", "derivative_atol", "descent_tolerance"):
            object.__setattr__(
                self, name, _scalar(getattr(self, name), name, minimum=0)
            )
        steps = tuple(_scalar(s, "validation step") for s in self.validation_steps)
        if not steps or any(s <= 0 for s in steps):
            raise ValueError("positive validation steps required")
        object.__setattr__(self, "validation_steps", steps)

    @classmethod
    def development(cls, policy: str):
        """Construct an explicitly requested, unqualified candidate policy."""
        choices = {
            "compact": (16, 4, 60000, 120, 1024),
            "balanced": (32, 8, 120000, 300, 1024),
            "exhaustive": (109, 8, 300000, 900, 2048),
        }
        if policy not in choices:
            raise ValueError("unknown development policy")
        seeds, recovery, calls, seconds, validation = choices[policy]
        return cls(
            seeds, recovery, calls, seconds, policy, max_validation_calls=validation
        )

    @property
    def total_start_limit(self):
        return self.max_seed_starts + self.reserved_recovery_starts

    @property
    def recovery_calls(self):
        return self.max_optimization_calls // 5

    @property
    def derivative_policy(self):
        return _DerivativePolicy(
            self.derivative_step,
            self.derivative_rtol,
            self.derivative_atol,
            self.derivative_reductions,
        )


@dataclass(frozen=True)
class TriexponentialStartDiagnostics:
    index: int
    kind: str
    seed_id: str | None
    initial_parameters: TriexponentialParameters
    best_parameters: TriexponentialParameters | None
    best_centered_nll: float | None
    endpoint_centered_nll: float | None
    optimizer_reported_success: bool | None
    solver_status: int | None
    termination: str
    fixed_coordinates: tuple[str, ...]
    model_call_limit: int
    model_calls: int
    elapsed_seconds: float
    derivative_statuses: tuple[str, ...]
    safeguard_trials: int
    solver: str = "lbfgsb"
    residual_evaluations: int = 0
    jacobian_evaluations: int = 0
    inward_start_coordinates: tuple[float, ...] | None = None
    failure_events: int = 0
    failure_diagnostics: tuple = ()


@dataclass(frozen=True)
class TriexponentialWork:
    requests: int
    objective_attempts: int
    objective_calls: int
    model_calls: int
    charged_component_convolutions: int
    completed_component_convolutions: int
    failures_with_unknown_partial_work: int
    phases: Mapping[str, Mapping[str, int]]
    phase_seconds: Mapping[str, float]
    blocked: tuple[tuple[str, str], ...]
    statistics_calculations: int
    total_seconds: float
    overhead_seconds: float
    failure_events: int = 0
    recent_failures: tuple = ()


@dataclass(frozen=True)
class TriexponentialValidationProbe:
    direction: str
    coordinate_direction: tuple[float, ...]
    signed_step: float
    feasible: bool
    parameters: TriexponentialParameters | None
    centered_nll: float | None
    status: str


@dataclass(frozen=True)
class TriexponentialLocalValidation:
    candidate: TriexponentialParameters
    submitted_objective: float
    tolerance: float
    probes: tuple[TriexponentialValidationProbe, ...]
    best_probe: TriexponentialParameters | None
    best_improvement: float
    complete: bool
    passed: bool | None


@dataclass(frozen=True)
class TriexponentialNumericalAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    required_stages_complete: bool
    candidate: TriexponentialParameters | None
    recomputed_centered_nll: float | None
    signed_objective_gap: float | None
    consistent: bool
    validation: TriexponentialLocalValidation | None
    fresh_prediction_consistent: bool | None = None
    fresh_prediction_reasons: tuple[str, ...] = ()
    validation_matches_candidate: bool = False
    required_recovery_complete: bool = False
    objective_gradient_available: bool | None = None
    residual_jacobian_available: bool | None = None
    diagnostic_improvement: float | None = None
    diagnostic_best_parameters: TriexponentialParameters | None = None
    diagnostic_best_centered_nll: float | None = None


@dataclass(frozen=True)
class TriexponentialOptimizationDiagnostics:
    generated_templates: int
    screened_candidates: int
    rejected_candidates: int
    duplicate_candidates: int
    available_seeds: int
    selected_seed_ids: tuple[str, ...]
    unattempted_seed_ids: tuple[str, ...]
    missing_coverage: tuple[str, ...]
    initialization_complete: bool
    nnls_projections: int
    nnls_failures: int
    background_repairs: int
    scaling_fallback: bool
    starts: tuple[TriexponentialStartDiagnostics, ...]
    recovery_starts_reserved: int
    recovery_starts_used: int
    recovery_calls_reserved: int
    recovery_calls_used: int
    recovery_unavailable_reason: str | None
    selected_candidate_origin: str | None
    best_observed_parameters: TriexponentialParameters | None
    best_observed_centered_nll: float | None
    promotions: tuple[str, ...]
    stopping_reason: str
    work: TriexponentialWork


@dataclass(frozen=True)
class TriexponentialComponentSummary:
    amplitude_fractions: tuple[float, ...] | None
    detected_signal_photons: tuple[float, ...]
    detected_signal_fractions: tuple[float, ...] | None
    ideal_photon_fractions: tuple[float, ...] | None


@dataclass(frozen=True)
class TriexponentialIdentifiabilityDiagnostics:
    """Descriptive model-conditional sensitivity, not physical identifiability."""

    nominal_parameters: int
    zero_amplitude_components: tuple[int, ...]
    equal_lifetime_pairs: tuple[tuple[int, int], ...]
    lifetime_ratios: tuple[tuple[int, int, float], ...]
    active_bounds: tuple[str, ...]
    conditioning_status: str
    weighted_jacobian_rank: int | None
    singular_values: tuple[float, ...] | None
    condition_number: float | None
    rank_tolerance: float | None
    derivative_statuses: tuple[str, ...]
    candidate: TriexponentialParameters | None = None
    candidate_evaluation_id: int | None = None
    restricted: Any = None
    derivative_details: tuple = ()


@dataclass(frozen=True)
class TriexponentialFitAssumptions:
    lifetime_bounds_ns: tuple[float, float]
    temporal_shift_bounds_ns: tuple[float, float]
    settings: TriexponentialFitSettings
    amplitude_scale: float
    background_scale: float
    target_kind: str
    irf_input_kind: str
    source_kind: str | None
    source_parameters: Mapping[str, Any]
    source_metadata: Mapping[str, Any]
    source_provenance: Mapping[str, Any]
    preparation_diagnostics: IRFPreparationDiagnostics | None
    policy_version: str = "issue37-pr2a-development-v1"


@dataclass(frozen=True)
class MultiexponentialFitResult:
    """K=3 result, including rejected finite estimates and explicit unavailability."""

    parameters: TriexponentialParameters | None
    expected_counts: NDArray[np.float64] | None
    component_counts: NDArray[np.float64] | None
    centered_nll: float | None
    components: TriexponentialComponentSummary | None
    statistics: PoissonFitStatistics | None
    optimization: TriexponentialOptimizationDiagnostics
    numerical: TriexponentialNumericalAssessment
    identifiability: TriexponentialIdentifiabilityDiagnostics
    assumptions: TriexponentialFitAssumptions


def fit_triexponential_reconvolution(
    time_ns: ArrayLike,
    counts: ArrayLike,
    irf: PreparedIRF | ArrayLike,
    *,
    lifetime_bounds_ns: tuple[float, float],
    temporal_shift_bounds_ns: tuple[float, float],
    initial_guesses: Sequence[TriexponentialParameters] | None = None,
    settings: TriexponentialFitSettings | None = None,
) -> MultiexponentialFitResult:
    """Fit K=3 to caller-asserted raw counts with explicitly supplied settings.

    IRFs must already be normalized on the supplied grid. No preprocessing,
    expectation floor, component removal, root export or production default is
    introduced. Positive shift means a later IRF. Failure retains finite evidence.
    """
    from tcspc_toolkit.estimation._fixed_k_search import _fit_triexponential

    return _fit_triexponential(
        time_ns,
        counts,
        irf,
        lifetime_bounds_ns=lifetime_bounds_ns,
        temporal_shift_bounds_ns=temporal_shift_bounds_ns,
        initial_guesses=initial_guesses,
        settings=settings,
        target_kind="raw_counts",
    )
