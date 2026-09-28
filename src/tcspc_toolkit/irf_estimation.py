"""Explicit, failure-aware derivative proxies for unavailable measured IRFs.

For a mono-exponential decay, ``S'(t) = A h(t) - S(t)/tau``. Consequently a
leading-edge derivative is not the true IRF; this module does not estimate the
missing ``S(t)/tau`` term or assess physical model validity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter

from tcspc_toolkit.irf import IRFProfile, IRFSourceKind, normalize_irf
from tcspc_toolkit.irf_preparation import _sampled_fwhm
from tcspc_toolkit.measurements import TCSPCMeasurement
from tcspc_toolkit.preprocessing import (
    estimate_background,
    subtract_background,
    validate_time_axis,
)


TimeWindow = tuple[float, float]
BinRange = tuple[int, int]


@dataclass(frozen=True)
class LeadingEdgeIRFDiagnostics:
    """Facts about numerical proxy construction, not physical IRF validity.

    Bin ranges and requested ns windows are half-open. Counts diagnostics use
    the entire requested rising-edge window. Negative derivative fraction is
    discarded negative area divided by positive plus negative absolute area.
    Both areas use trapezoidal integration after zero extension outside the
    selected pre-peak bins, matching the proxy's normalization convention.
    The fraction is ``None`` when its denominator is zero; derivative area
    has count units. A peak is flagged as near a window boundary within half
    the smoothing-window length; the width-resolution flag means FWHM is below
    two sample spacings. These are numerical observations, not validity tests.
    """

    status: Literal["proxy_constructed", "failed"]
    failure_reason: str | None
    flags: tuple[str, ...]
    requested_background_window_ns: TimeWindow
    background_bin_range: BinRange
    requested_rising_edge_window_ns: TimeWindow
    rising_edge_bin_range: BinRange
    selected_rising_segment_bin_range: BinRange | None
    background_estimate_counts_per_bin: float
    smoothing_window_bins: int
    polynomial_order: int
    sample_spacing_ns: float
    derivative_order: int
    boundary_mode: str
    smoothed_signal_peak_time_ns: float | None
    proxy_peak_time_ns: float | None
    proxy_fwhm_ns: float | None
    proxy_area_before_normalization: float | None
    normalization_factor: float | None
    final_normalized_area: float | None
    negative_derivative_area_discarded_counts: float | None
    negative_derivative_fraction: float | None
    observed_counts_in_rising_window: float
    estimated_background_counts_in_rising_window: float
    net_signal_counts_in_rising_window: float
    estimated_background_fraction_of_rising_counts: float | None


@dataclass(frozen=True)
class LeadingEdgeIRFResult:
    """A constructed leading-edge proxy, or an explained numerical failure."""

    profile: IRFProfile | None
    diagnostics: LeadingEdgeIRFDiagnostics

    def __post_init__(self) -> None:
        if (self.profile is None) != (self.diagnostics.status == "failed"):
            raise ValueError("profile presence must agree with diagnostic status")


def _validated_time_window(window: TimeWindow, name: str) -> TimeWindow:
    try:
        start, stop = window
        if isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_)):
            raise ValueError
        start, stop = float(start), float(stop)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite ns boundaries") from exc
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError(f"{name} must have finite increasing ns boundaries")
    return start, stop


def _bin_range(time_ns: np.ndarray, window_ns: TimeWindow, name: str) -> BinRange:
    start = int(np.searchsorted(time_ns, window_ns[0], side="left"))
    stop = int(np.searchsorted(time_ns, window_ns[1], side="left"))
    if start == stop:
        raise ValueError(f"{name} selects no histogram bins")
    return start, stop


def estimate_irf_from_leading_edge(
    measurement: TCSPCMeasurement,
    *,
    background_window_ns: TimeWindow,
    rising_edge_window_ns: TimeWindow,
    smoothing_window_bins: int,
    polynomial_order: int = 3,
) -> LeadingEdgeIRFResult:
    """Construct a labelled derivative proxy from raw fluorescence counts.

    Windows select bin coordinates in ``[start, stop)`` ns, as in
    ``crop_time_window``. The caller chooses both windows and the Savitzky--
    Golay settings. No zero time, lifetime, missing physical correction term,
    fitted IRF, or forward-model kernel is inferred here. A numerically valid
    proxy is not evidence that the fluorescence rise equals the true IRF.
    """
    if not isinstance(measurement, TCSPCMeasurement):
        raise TypeError("measurement must be a TCSPCMeasurement")
    counts = measurement.require_raw_counts()
    time_ns = validate_time_axis(measurement.time_ns)
    background_window = _validated_time_window(background_window_ns, "background_window_ns")
    rising_window = _validated_time_window(rising_edge_window_ns, "rising_edge_window_ns")
    if background_window[1] > rising_window[0]:
        raise ValueError("background window must precede and not overlap rising-edge window")

    if (
        isinstance(smoothing_window_bins, (bool, np.bool_))
        or not isinstance(smoothing_window_bins, (int, np.integer))
        or smoothing_window_bins < 3
        or smoothing_window_bins % 2 == 0
    ):
        raise ValueError("smoothing_window_bins must be an odd integer of at least 3")
    if (
        isinstance(polynomial_order, (bool, np.bool_))
        or not isinstance(polynomial_order, (int, np.integer))
        or polynomial_order < 1
        or polynomial_order >= smoothing_window_bins
    ):
        raise ValueError("polynomial_order must be an integer from 1 to window length - 1")

    background_start, background_stop = _bin_range(
        time_ns, background_window, "background_window_ns"
    )
    edge_start, edge_stop = _bin_range(time_ns, rising_window, "rising_edge_window_ns")
    if background_stop - background_start < 2:
        raise ValueError("background_window_ns must select at least two bins")
    if edge_stop - edge_start < smoothing_window_bins:
        raise ValueError(
            "rising_edge_window_ns must select at least smoothing_window_bins bins"
        )

    spacing_ns = float(np.mean(np.diff(time_ns)))
    background = estimate_background(counts, background_start, background_stop)
    working_trace = subtract_background(counts, background)
    observed_counts = float(np.sum(counts[edge_start:edge_stop], dtype=np.float64))
    background_counts = background * (edge_stop - edge_start)
    net_counts = observed_counts - background_counts
    background_fraction = (
        background_counts / observed_counts if observed_counts > 0.0 else None
    )
    flags: list[str] = []
    if net_counts <= 0.0:
        flags.append("nonpositive_net_rising_window_signal")
    if background_fraction is not None and background_fraction >= 0.5:
        flags.append("background_at_least_half_observed_rising_counts")

    diagnostics = LeadingEdgeIRFDiagnostics(
        status="failed",
        failure_reason="not_constructed",
        flags=tuple(flags),
        requested_background_window_ns=background_window,
        background_bin_range=(background_start, background_stop),
        requested_rising_edge_window_ns=rising_window,
        rising_edge_bin_range=(edge_start, edge_stop),
        selected_rising_segment_bin_range=None,
        background_estimate_counts_per_bin=background,
        smoothing_window_bins=int(smoothing_window_bins),
        polynomial_order=int(polynomial_order),
        sample_spacing_ns=spacing_ns,
        derivative_order=1,
        boundary_mode="interp",
        smoothed_signal_peak_time_ns=None,
        proxy_peak_time_ns=None,
        proxy_fwhm_ns=None,
        proxy_area_before_normalization=None,
        normalization_factor=None,
        final_normalized_area=None,
        negative_derivative_area_discarded_counts=None,
        negative_derivative_fraction=None,
        observed_counts_in_rising_window=observed_counts,
        estimated_background_counts_in_rising_window=background_counts,
        net_signal_counts_in_rising_window=net_counts,
        estimated_background_fraction_of_rising_counts=background_fraction,
    )
    if not np.isfinite(observed_counts) or not np.isfinite(background_counts):
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="nonfinite_window_count_diagnostic")
        )

    smoothed = savgol_filter(
        working_trace,
        window_length=smoothing_window_bins,
        polyorder=polynomial_order,
        deriv=0,
        mode="interp",
    )
    derivative = savgol_filter(
        working_trace,
        window_length=smoothing_window_bins,
        polyorder=polynomial_order,
        deriv=1,
        delta=spacing_ns,
        mode="interp",
    )
    if not np.all(np.isfinite(smoothed)) or not np.all(np.isfinite(derivative)):
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="nonfinite_smoothed_signal_or_derivative")
        )

    peak_index = edge_start + int(np.argmax(smoothed[edge_start:edge_stop]))
    diagnostics = replace(
        diagnostics, smoothed_signal_peak_time_ns=float(time_ns[peak_index])
    )
    if smoothed[peak_index] <= 0.0:
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="nonpositive_smoothed_signal_peak")
        )
    if peak_index == edge_start or peak_index == edge_stop - 1:
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="smoothed_peak_at_rising_window_boundary")
        )
    if peak_index - edge_start < 3:
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="insufficient_pre_peak_samples")
        )
    if min(peak_index - edge_start, edge_stop - 1 - peak_index) <= smoothing_window_bins // 2:
        flags.append("smoothed_peak_near_rising_window_boundary")

    # Restrict the derivative to the caller's pre-peak segment. Negative
    # fluctuations were retained through subtraction and smoothing; only the
    # final nonnegative proxy discards them.
    selected = slice(edge_start, peak_index)
    positive_proxy = np.zeros_like(derivative)
    discarded_negative = np.zeros_like(derivative)
    positive_proxy[selected] = np.clip(derivative[selected], 0.0, None)
    discarded_negative[selected] = np.clip(-derivative[selected], 0.0, None)
    positive_area = float(np.trapezoid(positive_proxy, x=time_ns))
    negative_area = float(np.trapezoid(discarded_negative, x=time_ns))
    absolute_area = positive_area + negative_area
    negative_fraction = negative_area / absolute_area if absolute_area > 0.0 else None
    if negative_area > 0.0:
        flags.append("negative_derivative_clipped_for_proxy")
    diagnostics = replace(
        diagnostics,
        flags=tuple(flags),
        selected_rising_segment_bin_range=(edge_start, peak_index),
        proxy_area_before_normalization=positive_area,
        negative_derivative_area_discarded_counts=negative_area,
        negative_derivative_fraction=negative_fraction,
    )
    if not np.isfinite(positive_area) or positive_area <= 0.0:
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="no_positive_rising_derivative_area")
        )
    normalization_factor = 1.0 / positive_area
    if not np.isfinite(normalization_factor):
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="unusable_proxy_normalization")
        )

    try:
        normalized_proxy = normalize_irf(time_ns, positive_proxy)
    except ValueError:
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="unusable_proxy_normalization")
        )
    normalized_area = float(np.trapezoid(normalized_proxy, x=time_ns))
    if not np.all(np.isfinite(normalized_proxy)) or not np.isfinite(normalized_area):
        return LeadingEdgeIRFResult(
            None, replace(diagnostics, failure_reason="unusable_proxy_normalization")
        )
    proxy_width = _sampled_fwhm(time_ns, normalized_proxy)
    proxy_peak_time = float(time_ns[int(np.argmax(normalized_proxy))])
    if proxy_width is None:
        flags.append("proxy_fwhm_unavailable")
    elif proxy_width < 2.0 * spacing_ns:
        flags.append("proxy_fwhm_below_two_sample_spacings")

    profile = IRFProfile(
        time_ns=time_ns,
        values=normalized_proxy,
        source_kind=IRFSourceKind.LEADING_EDGE_ESTIMATE,
        source_parameters={
            "background_window_ns": background_window,
            "rising_edge_window_ns": rising_window,
            "smoothing_window_bins": int(smoothing_window_bins),
            "polynomial_order": int(polynomial_order),
            "derivative_order": 1,
            "derivative_delta_ns": spacing_ns,
            "savgol_boundary_mode": "interp",
            "negative_derivative_handling": "clip_to_zero_for_proxy_only",
        },
        metadata={
            "interpretation": "leading-edge derivative proxy; not a measured or true IRF",
            "omitted_physical_term": "S(t)/tau",
        },
        provenance={
            "estimator": "estimate_irf_from_leading_edge",
            "derived_from": "fluorescence_histogram",
            "measurement_sample_id": measurement.sample_id,
            "measurement_provenance": dict(measurement.provenance),
        },
    )
    return LeadingEdgeIRFResult(
        profile,
        replace(
            diagnostics,
            status="proxy_constructed",
            failure_reason=None,
            flags=tuple(flags),
            proxy_peak_time_ns=proxy_peak_time,
            proxy_fwhm_ns=proxy_width,
            normalization_factor=normalization_factor,
            final_normalized_area=normalized_area,
        ),
    )
