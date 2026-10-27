"""Small adapters from imported measurements to existing estimators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.estimation.classical import (
    ReconvolutionCurveResult,
    fit_single_reconvolution_curve,
)
from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitResult,
    BiexponentialFitSettings,
    BiexponentialParameters,
    fit_biexponential_reconvolution,
)
from tcspc_toolkit.evaluation.reconvolution import (
    ReconvolutionStatisticsComparison,
    compare_reconvolution_statistics,
)
from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.evaluation.diagnostics import calculate_lifetime_errors
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.preprocessing.features import FEATURE_NAMES, extract_features
from tcspc_toolkit.irf.preparation import PreparedIRF
from tcspc_toolkit.irf.profiles import IRFProfile
from tcspc_toolkit.irf.selection import _resolve_reconvolution_irf
from tcspc_toolkit.measurement.records import SampledIRF, TCSPCMeasurement
from tcspc_toolkit.physics.forward_model import (
    monoexponential_reconvolution_expected_counts,
)
from tcspc_toolkit.preprocessing.histograms import time_axes_compatible


@dataclass(frozen=True)
class ReferenceLifetimeEvaluation:
    """One estimate compared with an explicitly supplied trusted reference.

    ``signed_error_ns`` is ``estimated_lifetime_ns - reference_lifetime_ns``.
    A single comparison does not establish estimator bias or coverage.
    """

    estimated_lifetime_ns: float
    reference_lifetime_ns: float
    signed_error_ns: float
    absolute_error_ns: float
    relative_error: float
    reference_metadata: Mapping[str, Any] = field(default_factory=dict)


def fit_experimental_reconvolution(
    measurement: TCSPCMeasurement,
    *,
    temporal_shift_bounds_ns: tuple[float, float],
    background_fraction: float = 0.10,
    prepared_irf: PreparedIRF | None = None,
) -> ReconvolutionCurveResult:
    """Fit raw counts with an explicit prepared IRF or the attached sampled IRF.

    An explicit ``prepared_irf`` takes precedence over ``measurement.irf``.
    It must already be on the measurement grid; no preparation occurs here.
    Shift bounds must include zero, the reused fitter's current initial shift.
    """
    counts = measurement.require_raw_counts()
    fitting_irf, _ = _resolve_reconvolution_irf(measurement, counts, prepared_irf)
    if counts.size < 5:
        raise InvalidMeasurementError("reconvolution requires at least five time bins")
    try:
        lower, upper = temporal_shift_bounds_ns
        valid_bounds = np.isfinite(lower) and np.isfinite(upper) and lower < upper
    except (TypeError, ValueError):
        valid_bounds = False
    if not valid_bounds:
        raise InvalidMeasurementError(
            "temporal_shift_bounds_ns must be finite and increasing"
        )
    if not lower <= 0.0 <= upper:
        raise InvalidMeasurementError(
            "temporal_shift_bounds_ns must include 0.0 ns because the current "
            "reconvolution fitter initializes temporal shift at zero"
        )
    try:
        valid_fraction = (
            np.isfinite(background_fraction) and 0.0 < background_fraction < 1.0
        )
    except TypeError:
        valid_fraction = False
    if not valid_fraction:
        raise InvalidMeasurementError("background_fraction must lie between 0 and 1")

    return fit_single_reconvolution_curve(
        time=measurement.time_ns,
        counts=counts,
        irf=fitting_irf,
        temporal_shift_bounds=(float(lower), float(upper)),
        objective="poisson",
        background_fraction=background_fraction,
    )


def _require_training_grid(
    measurement: TCSPCMeasurement, training_time_ns: ArrayLike
) -> None:
    try:
        compatible = time_axes_compatible(measurement.time_ns, training_time_ns)
    except ValueError as exc:
        raise InvalidMeasurementError(f"invalid training time grid: {exc}") from exc
    if not compatible:
        raise InvalidMeasurementError(
            "measurement time grid differs from the model training grid"
        )


def measurement_to_feature_table(
    measurement: TCSPCMeasurement,
    *,
    training_time_ns: ArrayLike,
    feature_config: FeatureConfig,
    training_feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Build the ordered one-row feature input for a compatible fitted model."""
    counts = measurement.require_raw_counts()
    _require_training_grid(measurement, training_time_ns)
    if tuple(training_feature_names) != FEATURE_NAMES:
        raise InvalidMeasurementError(
            "ordered training feature names differ from FEATURE_NAMES"
        )
    return extract_features(measurement.time_ns, counts, feature_config)


def measurement_to_histogram_batch(
    measurement: TCSPCMeasurement,
    *,
    training_time_ns: ArrayLike,
) -> NDArray[np.int64]:
    """Build a one-row raw histogram input for a fitted ML pipeline."""
    counts = measurement.require_raw_counts()
    _require_training_grid(measurement, training_time_ns)
    return np.asarray(counts[None, :], dtype=np.int64).copy()


def evaluate_estimate_against_reference(
    estimated_lifetime_ns: float,
    *,
    reference_lifetime_ns: float,
    reference_metadata: Mapping[str, Any] | None = None,
) -> ReferenceLifetimeEvaluation:
    """Calculate single-sample errors only for a trusted supplied reference."""
    try:
        valid_estimate = (
            np.isfinite(estimated_lifetime_ns) and estimated_lifetime_ns > 0
        )
    except TypeError:
        valid_estimate = False
    if not valid_estimate:
        raise InvalidMeasurementError(
            "estimated_lifetime_ns must be finite and positive"
        )
    try:
        valid_reference = (
            np.isfinite(reference_lifetime_ns) and reference_lifetime_ns > 0
        )
    except TypeError:
        valid_reference = False
    if not valid_reference:
        raise InvalidMeasurementError(
            "reference_lifetime_ns must be finite and positive"
        )
    if reference_metadata is None:
        reference_metadata = {}
    if not isinstance(reference_metadata, Mapping):
        raise InvalidMeasurementError("reference_metadata must be a mapping")
    signed, absolute, relative = calculate_lifetime_errors(
        true_lifetimes=np.array([reference_lifetime_ns], dtype=np.float64),
        estimated_lifetimes=np.array([estimated_lifetime_ns], dtype=np.float64),
    )
    return ReferenceLifetimeEvaluation(
        estimated_lifetime_ns=float(estimated_lifetime_ns),
        reference_lifetime_ns=float(reference_lifetime_ns),
        signed_error_ns=float(signed[0]),
        absolute_error_ns=float(absolute[0]),
        relative_error=float(relative[0]),
        reference_metadata=MappingProxyType(deepcopy(dict(reference_metadata))),
    )


@dataclass(frozen=True)
class ExperimentalReconvolutionContext:
    """Defensive snapshots, with numerical preparation separate from provenance.

    As in the existing measurement records, arrays are read-only and metadata
    are deep copies protected at the top level, not recursively immutable.
    Neither normalization nor recorded provenance certifies a physical IRF.
    """

    measurement: TCSPCMeasurement
    assumed_irf: PreparedIRF | SampledIRF
    kernel: NDArray[np.float64]
    irf_selection: str
    normalization_performed: bool
    area_before_normalization: float
    area_after_normalization: float


@dataclass(frozen=True)
class ExperimentalBiexponentialResult:
    fit: BiexponentialFitResult
    context: ExperimentalReconvolutionContext


@dataclass(frozen=True)
class ReconvolutionComparisonBounds:
    """Actual bounds differ: the unchanged mono fitter has no upper tau bound."""

    bi_lifetime_bounds_ns: tuple[float, float]
    temporal_shift_bounds_ns: tuple[float, float]
    mono_lifetime_bounds_ns: tuple[float, None] = (1e-12, None)
    mono_background_lower: float = 1e-12
    bi_background_lower: float = 0.0
    amplitude_lower: float = 0.0
    mono_lifetime_outside_bi_bounds: bool | None = None


@dataclass(frozen=True)
class ExperimentalReconvolutionComparison:
    context: ExperimentalReconvolutionContext
    mono: ReconvolutionCurveResult
    bi: BiexponentialFitResult
    mono_expected_counts: NDArray[np.float64] | None
    statistics: ReconvolutionStatisticsComparison
    bounds: ReconvolutionComparisonBounds


def _sampled_snapshot(source: SampledIRF) -> SampledIRF:
    return SampledIRF(source.time_ns, source.values, source.metadata, source.provenance)


def _reconvolution_context(measurement, prepared_irf):
    counts = measurement.require_raw_counts()
    kernel, source = _resolve_reconvolution_irf(measurement, counts, prepared_irf)
    snapshot = TCSPCMeasurement(
        measurement.time_ns,
        counts,
        measurement.data_kind,
        None if measurement.irf is None else _sampled_snapshot(measurement.irf),
        measurement.sample_id,
        measurement.metadata,
        measurement.provenance,
    )
    if isinstance(source, PreparedIRF):
        profile = source.source
        selected = PreparedIRF(
            IRFProfile(
                profile.time_ns,
                profile.values,
                profile.source_kind,
                source_parameters=profile.source_parameters,
                metadata=profile.metadata,
                provenance=profile.provenance,
            ),
            source.time_ns,
            source.kernel,
            source.diagnostics,
        )
        path = "explicit_prepared_irf"
        normalized = False
        before = float(np.trapezoid(source.kernel, x=source.time_ns))
    else:
        selected = _sampled_snapshot(source)
        path = "attached_sampled_irf"
        normalized = True
        before = float(np.trapezoid(source.values, x=source.time_ns))
    copied_kernel = np.array(kernel, dtype=float, copy=True)
    copied_kernel.setflags(write=False)
    return ExperimentalReconvolutionContext(
        snapshot,
        selected,
        copied_kernel,
        path,
        normalized,
        before,
        float(np.trapezoid(copied_kernel, x=snapshot.time_ns)),
    )


def _fit_context_biexponential(
    context, lifetime_bounds_ns, temporal_shift_bounds_ns, initial_guesses, settings
):
    assumed = context.assumed_irf
    return fit_biexponential_reconvolution(
        context.measurement.time_ns,
        context.measurement.require_raw_counts(),
        assumed if isinstance(assumed, PreparedIRF) else context.kernel,
        lifetime_bounds_ns=lifetime_bounds_ns,
        temporal_shift_bounds_ns=temporal_shift_bounds_ns,
        initial_guesses=initial_guesses,
        settings=settings,
    )


def fit_experimental_biexponential_reconvolution(
    measurement: TCSPCMeasurement,
    *,
    lifetime_bounds_ns: tuple[float, float],
    temporal_shift_bounds_ns: tuple[float, float],
    prepared_irf: PreparedIRF | None = None,
    initial_guesses: Sequence[BiexponentialParameters] | None = None,
    settings: BiexponentialFitSettings | None = None,
) -> ExperimentalBiexponentialResult:
    """Fit canonical raw counts without transforming the observations.

    An explicit PreparedIRF takes precedence. Otherwise the attached same-grid
    sampled IRF is normalized on a defensive copy using the existing resolver.
    This latter path records adapter normalization, not verified preparation
    metadata, and retains the fitter's declared-array assumption.
    """
    context = _reconvolution_context(measurement, prepared_irf)
    result = _fit_context_biexponential(
        context, lifetime_bounds_ns, temporal_shift_bounds_ns, initial_guesses, settings
    )
    return ExperimentalBiexponentialResult(result, context)


def compare_experimental_reconvolution(
    measurement: TCSPCMeasurement,
    *,
    lifetime_bounds_ns: tuple[float, float],
    temporal_shift_bounds_ns: tuple[float, float],
    prepared_irf: PreparedIRF | None = None,
    initial_guesses: Sequence[BiexponentialParameters] | None = None,
    settings: BiexponentialFitSettings | None = None,
    background_fraction: float = 0.10,
) -> ExperimentalReconvolutionComparison:
    """Descriptively compare unchanged mono/bi fits on identical raw inputs.

    Mono requires at least five bins and shift bounds including zero. Its
    lifetime is unbounded above and background bounded below by 1e-12; bi has
    explicit finite lifetime bounds and permits exact zero background. Thus
    these are not silently presented as fits with identical feasible sets.
    """
    context = _reconvolution_context(measurement, prepared_irf)
    selected = context.assumed_irf
    mono = fit_experimental_reconvolution(
        context.measurement,
        temporal_shift_bounds_ns=temporal_shift_bounds_ns,
        background_fraction=background_fraction,
        prepared_irf=selected if isinstance(selected, PreparedIRF) else None,
    )
    bi = _fit_context_biexponential(
        context, lifetime_bounds_ns, temporal_shift_bounds_ns, initial_guesses, settings
    )
    parameters = [
        mono.fitted_amplitude,
        mono.fitted_lifetime_ns,
        mono.fitted_background,
        mono.fitted_temporal_shift_ns,
    ]
    curve = None
    outside = None
    if (
        np.all(np.isfinite(parameters))
        and parameters[0] >= 0
        and parameters[1] > 0
        and parameters[2] >= 0
    ):
        prediction = monoexponential_reconvolution_expected_counts(
            context.measurement.time_ns, context.kernel, *parameters
        )
        if np.all(np.isfinite(prediction)) and np.all(prediction >= 0):
            curve = np.array(prediction, copy=True)
            curve.setflags(write=False)
        outside = not lifetime_bounds_ns[0] <= parameters[1] <= lifetime_bounds_ns[1]
    return ExperimentalReconvolutionComparison(
        context,
        mono,
        bi,
        curve,
        compare_reconvolution_statistics(
            context.measurement.values, curve, bi.expected_counts
        ),
        ReconvolutionComparisonBounds(
            tuple(lifetime_bounds_ns),
            tuple(temporal_shift_bounds_ns),
            mono_lifetime_outside_bi_bounds=outside,
        ),
    )
