"""Small adapters from imported measurements to existing estimators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.classical_evaluation import (
    ReconvolutionCurveResult,
    fit_single_reconvolution_curve,
)
from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.evaluation import calculate_lifetime_errors
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.features import FEATURE_NAMES, extract_features
from tcspc_toolkit.irf import normalize_irf
from tcspc_toolkit.irf_preparation import PreparedIRF
from tcspc_toolkit.measurements import TCSPCMeasurement
from tcspc_toolkit.preprocessing import time_axes_compatible


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
    if prepared_irf is None and measurement.irf is None:
        raise InvalidMeasurementError(
            "a measured IRF on the histogram grid is required for experimental reconvolution"
        )
    if prepared_irf is not None:
        if not isinstance(prepared_irf, PreparedIRF):
            raise InvalidMeasurementError("prepared_irf must be a PreparedIRF")
        if not time_axes_compatible(measurement.time_ns, prepared_irf.time_ns):
            raise InvalidMeasurementError(
                "prepared_irf grid differs from measurement time grid; "
                "prepare the IRF explicitly on the measurement grid"
            )
        if (
            prepared_irf.kernel.shape != counts.shape
            or not np.all(np.isfinite(prepared_irf.kernel))
            or np.any(prepared_irf.kernel < 0.0)
        ):
            raise InvalidMeasurementError(
                "prepared_irf kernel must be finite, nonnegative, and unit-area"
            )
        kernel_area = float(np.trapezoid(prepared_irf.kernel, x=prepared_irf.time_ns))
        if not np.isfinite(kernel_area) or not np.isclose(
            kernel_area, 1.0, rtol=1e-10, atol=1e-12
        ):
            raise InvalidMeasurementError("prepared_irf kernel must have unit area")
    if counts.size < 5:
        raise InvalidMeasurementError("reconvolution requires at least five time bins")
    try:
        lower, upper = temporal_shift_bounds_ns
        valid_bounds = np.isfinite(lower) and np.isfinite(upper) and lower < upper
    except (TypeError, ValueError):
        valid_bounds = False
    if not valid_bounds:
        raise InvalidMeasurementError("temporal_shift_bounds_ns must be finite and increasing")
    if not lower <= 0.0 <= upper:
        raise InvalidMeasurementError(
            "temporal_shift_bounds_ns must include 0.0 ns because the current "
            "reconvolution fitter initializes temporal shift at zero"
        )
    try:
        valid_fraction = np.isfinite(background_fraction) and 0.0 < background_fraction < 1.0
    except TypeError:
        valid_fraction = False
    if not valid_fraction:
        raise InvalidMeasurementError("background_fraction must lie between 0 and 1")

    if prepared_irf is None:
        fitting_irf = normalize_irf(measurement.time_ns, measurement.irf.values)
    else:
        fitting_irf = prepared_irf.kernel
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
        valid_estimate = np.isfinite(estimated_lifetime_ns) and estimated_lifetime_ns > 0
    except TypeError:
        valid_estimate = False
    if not valid_estimate:
        raise InvalidMeasurementError("estimated_lifetime_ns must be finite and positive")
    try:
        valid_reference = np.isfinite(reference_lifetime_ns) and reference_lifetime_ns > 0
    except TypeError:
        valid_reference = False
    if not valid_reference:
        raise InvalidMeasurementError("reference_lifetime_ns must be finite and positive")
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
