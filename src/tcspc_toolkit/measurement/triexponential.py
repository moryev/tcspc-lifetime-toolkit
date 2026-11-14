"""Experimental K=3 measurement adapter with the existing numerical result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tcspc_toolkit.estimation.triexponential import (
    MultiexponentialFitResult,
    TriexponentialFitSettings,
    TriexponentialParameters,
    fit_triexponential_reconvolution,
)
from tcspc_toolkit.irf.preparation import PreparedIRF
from tcspc_toolkit.measurement.adapters import (
    ExperimentalReconvolutionContext,
    _reconvolution_context,
)
from tcspc_toolkit.measurement.records import TCSPCMeasurement


@dataclass(frozen=True)
class ExperimentalTriexponentialResult:
    """Existing K=3 fit record with defensive measurement and IRF context."""

    fit: MultiexponentialFitResult
    context: ExperimentalReconvolutionContext


def fit_experimental_triexponential_reconvolution(
    measurement: TCSPCMeasurement,
    *,
    lifetime_bounds_ns: tuple[float, float],
    temporal_shift_bounds_ns: tuple[float, float],
    settings: TriexponentialFitSettings,
    prepared_irf: PreparedIRF | None = None,
    initial_guesses: Sequence[TriexponentialParameters] | None = None,
) -> ExperimentalTriexponentialResult:
    """Experimental research K=3 fitting with explicit settings and raw counts.

    The explicit PreparedIRF takes precedence. Otherwise the existing resolver
    normalizes a defensive copy of the attached same-grid sampled IRF. The
    context records that normalization and preserves the source provenance.
    ``fit`` is the existing structured result, including rejected finite
    candidates, numerical assessment and conditional sensitivity diagnostics.
    Inspect those diagnostics before interpreting individual components.
    """
    context = _reconvolution_context(measurement, prepared_irf)
    assumed = context.assumed_irf
    fit = fit_triexponential_reconvolution(
        context.measurement.time_ns,
        context.measurement.require_raw_counts(),
        assumed if isinstance(assumed, PreparedIRF) else context.kernel,
        lifetime_bounds_ns=lifetime_bounds_ns,
        temporal_shift_bounds_ns=temporal_shift_bounds_ns,
        initial_guesses=initial_guesses,
        settings=settings,
    )
    return ExperimentalTriexponentialResult(fit, context)
