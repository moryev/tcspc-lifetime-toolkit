"""Bayesian contracts for mono-exponential Poisson reconvolution.

The densities below are deterministic and conditional on one fixed IRF kernel.
The Poisson likelihood omits the count-factorial constant exactly as the
classical objective does. The optional sampler is imported only when called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.experimental import _resolve_reconvolution_irf
from tcspc_toolkit.fitting import poisson_negative_log_likelihood
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import IRFProfile, IRFSourceKind
from tcspc_toolkit.irf_preparation import PreparedIRF
from tcspc_toolkit.measurements import SampledIRF, TCSPCMeasurement


def _finite_real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return converted


@dataclass(frozen=True)
class GammaPrior:
    """Proper Gamma prior with shape and rate (inverse physical units)."""

    shape: float
    rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _finite_real(self.shape, "shape", positive=True))
        object.__setattr__(self, "rate", _finite_real(self.rate, "rate", positive=True))

    def log_density(self, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return -math.inf
        return (
            self.shape * math.log(self.rate)
            - math.lgamma(self.shape)
            + (self.shape - 1.0) * math.log(value)
            - self.rate * value
        )


@dataclass(frozen=True)
class LogNormalPrior:
    """Proper lifetime prior: log(lifetime / 1 ns) is Normal(log_mean, log_std)."""

    log_mean: float
    log_std: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_mean", _finite_real(self.log_mean, "log_mean"))
        object.__setattr__(
            self, "log_std", _finite_real(self.log_std, "log_std", positive=True)
        )

    def log_density(self, lifetime_ns: float) -> float:
        if not math.isfinite(lifetime_ns) or lifetime_ns <= 0.0:
            return -math.inf
        log_lifetime = math.log(lifetime_ns)
        standardized = (log_lifetime - self.log_mean) / self.log_std
        return (
            -log_lifetime
            - math.log(self.log_std)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * standardized * standardized
        )


@dataclass(frozen=True)
class BoundedUniformPrior:
    """Proper uniform prior on a finite residual shift interval in ns."""

    lower_ns: float
    upper_ns: float

    def __post_init__(self) -> None:
        lower = _finite_real(self.lower_ns, "lower_ns")
        upper = _finite_real(self.upper_ns, "upper_ns")
        if lower >= upper or not math.isfinite(upper - lower):
            raise ValueError("shift bounds must form a finite increasing interval")
        object.__setattr__(self, "lower_ns", lower)
        object.__setattr__(self, "upper_ns", upper)

    def log_density(self, shift_ns: float) -> float:
        if not math.isfinite(shift_ns) or not self.lower_ns <= shift_ns <= self.upper_ns:
            return -math.inf
        return -math.log(self.upper_ns - self.lower_ns)


@dataclass(frozen=True)
class BayesianPriorConfig:
    """Explicit physical-space priors; exactly one shift mode is required.

    ``amplitude`` is the reconvolution scale, not finite-window signal photons.
    ``background_per_bin`` is expected detector background counts per bin.
    No prior scale is inferred from observed counts.
    """

    amplitude: GammaPrior
    lifetime_ns: LogNormalPrior
    background_per_bin: GammaPrior
    temporal_shift_prior: BoundedUniformPrior | None
    fixed_temporal_shift_ns: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.amplitude, GammaPrior):
            raise TypeError("amplitude must be a GammaPrior")
        if not isinstance(self.lifetime_ns, LogNormalPrior):
            raise TypeError("lifetime_ns must be a LogNormalPrior")
        if not isinstance(self.background_per_bin, GammaPrior):
            raise TypeError("background_per_bin must be a GammaPrior")
        if (self.temporal_shift_prior is None) == (self.fixed_temporal_shift_ns is None):
            raise ValueError("provide exactly one of temporal_shift_prior or fixed_temporal_shift_ns")
        if self.temporal_shift_prior is not None and not isinstance(
            self.temporal_shift_prior, BoundedUniformPrior
        ):
            raise TypeError("temporal_shift_prior must be a BoundedUniformPrior")
        if self.fixed_temporal_shift_ns is not None:
            object.__setattr__(
                self,
                "fixed_temporal_shift_ns",
                _finite_real(self.fixed_temporal_shift_ns, "fixed_temporal_shift_ns"),
            )

    @property
    def infer_temporal_shift(self) -> bool:
        return self.temporal_shift_prior is not None

    @property
    def n_sampling_coordinates(self) -> int:
        return 4 if self.infer_temporal_shift else 3


@dataclass(frozen=True)
class BayesianPhysicalParameters:
    """Physical parameters in the same units as the shared forward model.

    This carrier may also hold an out-of-support proposal; the density functions
    return ``-inf`` for such a state.
    """

    amplitude: float
    lifetime_ns: float
    background_per_bin: float
    temporal_shift_ns: float


class IRFModelRelation(str, Enum):
    """Caller-declared relation to known generating physics, never inferred."""

    UNSPECIFIED = "unspecified"
    MATCHED = "matched"
    DELIBERATELY_MISSPECIFIED = "deliberately_misspecified"


@dataclass(frozen=True)
class BayesianModelContext:
    """One raw histogram conditioned on one fixed IRF assumption.

    The selected source retains its provenance and, for a prepared IRF, its
    preparation diagnostics. A leading-edge-derived proxy remains a fixed
    plug-in assumption even when derived from this same measurement; no IRF
    uncertainty is integrated into the densities below.
    """

    measurement: TCSPCMeasurement
    prepared_irf: PreparedIRF | None = None
    irf_model_relation: IRFModelRelation = IRFModelRelation.UNSPECIFIED
    _irf_kernel: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, TCSPCMeasurement):
            raise TypeError("measurement must be a TCSPCMeasurement")
        if not isinstance(self.irf_model_relation, IRFModelRelation):
            raise ValueError("irf_model_relation must be an IRFModelRelation")
        counts = self.measurement.require_raw_counts()
        kernel, _ = _resolve_reconvolution_irf(self.measurement, counts, self.prepared_irf)
        if counts.size < 5:
            raise InvalidMeasurementError("reconvolution requires at least five time bins")
        kernel_copy = np.array(kernel, dtype=np.float64, copy=True)
        kernel_copy.setflags(write=False)
        object.__setattr__(self, "_irf_kernel", kernel_copy)

    @property
    def time_ns(self) -> NDArray[np.float64]:
        return self.measurement.time_ns

    @property
    def counts(self) -> NDArray[np.int64]:
        return self.measurement.require_raw_counts()

    @property
    def irf_kernel(self) -> NDArray[np.float64]:
        return self._irf_kernel

    @property
    def irf_selection(self) -> Literal["explicit_prepared", "attached_sampled"]:
        return "explicit_prepared" if self.prepared_irf is not None else "attached_sampled"

    @property
    def irf_source(self) -> IRFProfile | SampledIRF:
        if self.prepared_irf is not None:
            return self.prepared_irf.source
        attached_irf = self.measurement.irf
        assert attached_irf is not None
        return attached_irf

    @property
    def irf_source_kind(self) -> IRFSourceKind:
        if self.prepared_irf is not None:
            return self.prepared_irf.source.source_kind
        return IRFSourceKind.IMPORTED_SAMPLED


class _InvalidProposal(ValueError):
    """Numerically or physically unusable transformed sampler proposal."""


def _sampling_vector(coordinates: ArrayLike, prior: BayesianPriorConfig) -> NDArray[np.float64]:
    if not isinstance(prior, BayesianPriorConfig):
        raise TypeError("prior must be a BayesianPriorConfig")
    try:
        vector = np.asarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("sampling coordinates must be numeric") from exc
    if vector.shape != (prior.n_sampling_coordinates,):
        raise ValueError(f"sampling coordinates must have shape ({prior.n_sampling_coordinates},)")
    return vector


def sampling_to_physical_parameters(
    coordinates: ArrayLike,
    prior: BayesianPriorConfig,
) -> BayesianPhysicalParameters:
    """Map log(A), log(tau/ns), log(B), [shift/ns] to physical values."""
    vector = _sampling_vector(coordinates, prior)
    if not np.all(np.isfinite(vector)):
        raise _InvalidProposal("sampling coordinates must be finite")
    with np.errstate(over="ignore", under="ignore"):
        positive_values = np.exp(vector[:3])
    if not np.all(np.isfinite(positive_values)) or np.any(positive_values <= 0.0):
        raise _InvalidProposal("sampling coordinates produce non-physical values")
    shift = float(vector[3]) if prior.infer_temporal_shift else prior.fixed_temporal_shift_ns
    assert shift is not None
    return BayesianPhysicalParameters(
        amplitude=float(positive_values[0]),
        lifetime_ns=float(positive_values[1]),
        background_per_bin=float(positive_values[2]),
        temporal_shift_ns=shift,
    )


def physical_to_sampling_coordinates(
    parameters: BayesianPhysicalParameters,
    prior: BayesianPriorConfig,
) -> NDArray[np.float64]:
    """Map positive physical values to the selected three or four coordinates."""
    if not isinstance(parameters, BayesianPhysicalParameters):
        raise TypeError("parameters must be BayesianPhysicalParameters")
    if not isinstance(prior, BayesianPriorConfig):
        raise TypeError("prior must be a BayesianPriorConfig")
    positive_values = (
        parameters.amplitude,
        parameters.lifetime_ns,
        parameters.background_per_bin,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
        raise ValueError("amplitude, lifetime_ns, and background_per_bin must be finite and positive")
    if not math.isfinite(parameters.temporal_shift_ns):
        raise ValueError("temporal_shift_ns must be finite")
    if not prior.infer_temporal_shift and parameters.temporal_shift_ns != prior.fixed_temporal_shift_ns:
        raise ValueError("temporal_shift_ns differs from the fixed shift")
    coordinates = [math.log(value) for value in positive_values]
    if prior.infer_temporal_shift:
        coordinates.append(parameters.temporal_shift_ns)
    return np.asarray(coordinates, dtype=np.float64)


def log_transformation_jacobian(
    coordinates: ArrayLike,
    prior: BayesianPriorConfig,
) -> float:
    """Log |d(A,tau,B[,shift])/d(log A,log tau,log B[,shift])|."""
    vector = _sampling_vector(coordinates, prior)
    if not np.all(np.isfinite(vector)):
        return -math.inf
    jacobian = float(np.sum(vector[:3]))
    return jacobian if math.isfinite(jacobian) else -math.inf


def log_prior(
    parameters: BayesianPhysicalParameters,
    prior: BayesianPriorConfig,
) -> float:
    """Normalized physical-space prior density; fixed shift adds no density."""
    if not isinstance(parameters, BayesianPhysicalParameters):
        raise TypeError("parameters must be BayesianPhysicalParameters")
    if not isinstance(prior, BayesianPriorConfig):
        raise TypeError("prior must be a BayesianPriorConfig")
    terms = [
        prior.amplitude.log_density(parameters.amplitude),
        prior.lifetime_ns.log_density(parameters.lifetime_ns),
        prior.background_per_bin.log_density(parameters.background_per_bin),
    ]
    if prior.infer_temporal_shift:
        assert prior.temporal_shift_prior is not None
        terms.append(prior.temporal_shift_prior.log_density(parameters.temporal_shift_ns))
    elif parameters.temporal_shift_ns != prior.fixed_temporal_shift_ns:
        return -math.inf
    return float(sum(terms)) if all(math.isfinite(term) for term in terms) else -math.inf


def log_likelihood(
    parameters: BayesianPhysicalParameters,
    context: BayesianModelContext,
) -> float:
    """Raw-count Poisson log likelihood, omitting the count-factorial constant."""
    if not isinstance(parameters, BayesianPhysicalParameters):
        raise TypeError("parameters must be BayesianPhysicalParameters")
    if not isinstance(context, BayesianModelContext):
        raise TypeError("context must be a BayesianModelContext")
    if not all(
        math.isfinite(value)
        for value in (
            parameters.amplitude,
            parameters.lifetime_ns,
            parameters.background_per_bin,
            parameters.temporal_shift_ns,
        )
    ) or (
        parameters.amplitude < 0.0
        or parameters.lifetime_ns <= 0.0
        or parameters.background_per_bin < 0.0
    ):
        return -math.inf

    with np.errstate(over="ignore", invalid="ignore"):
        try:
            expected_counts = monoexponential_reconvolution_expected_counts(
                time=context.time_ns,
                irf=context.irf_kernel,
                amplitude=parameters.amplitude,
                lifetime=parameters.lifetime_ns,
                background=parameters.background_per_bin,
                temporal_shift=parameters.temporal_shift_ns,
            )
        except ValueError as exc:
            if "decay must contain only finite values" in str(exc):
                return -math.inf
            raise
    if not np.all(np.isfinite(expected_counts)):
        return -math.inf
    negative_log_likelihood = poisson_negative_log_likelihood(context.counts, expected_counts)
    return -negative_log_likelihood if math.isfinite(negative_log_likelihood) else -math.inf


def log_posterior(
    coordinates: ArrayLike,
    context: BayesianModelContext,
    prior: BayesianPriorConfig,
) -> float:
    """Unnormalized density in log-parameter sampling coordinates."""
    if not isinstance(context, BayesianModelContext):
        raise TypeError("context must be a BayesianModelContext")
    vector = _sampling_vector(coordinates, prior)
    try:
        parameters = sampling_to_physical_parameters(vector, prior)
    except _InvalidProposal:
        return -math.inf
    physical_log_prior = log_prior(parameters, prior)
    if not math.isfinite(physical_log_prior):
        return -math.inf
    likelihood = log_likelihood(parameters, context)
    if not math.isfinite(likelihood):
        return -math.inf
    jacobian = log_transformation_jacobian(vector, prior)
    total = physical_log_prior + likelihood + jacobian
    return total if math.isfinite(total) else -math.inf


def fit_bayesian_monoexponential_reconvolution(
    measurement: TCSPCMeasurement,
    *,
    priors: BayesianPriorConfig,
    sampling_config: "BayesianSamplingConfig",
    prepared_irf: PreparedIRF | None = None,
    irf_model_relation: IRFModelRelation = IRFModelRelation.UNSPECIFIED,
) -> "BayesianInferenceRun":
    """Run optional emcee sampling against this module's posterior target."""
    from tcspc_toolkit.bayesian_sampling import (
        fit_bayesian_monoexponential_reconvolution as _fit,
    )

    return _fit(
        measurement,
        priors=priors,
        sampling_config=sampling_config,
        prepared_irf=prepared_irf,
        irf_model_relation=irf_model_relation,
    )
