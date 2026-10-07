"""Compatibility facade for the historical histogram-preprocessing API.

These direct exports remain supported through at least Issue #12. Feature and
representation implementations are explicit submodules, not eager exports here."""

from tcspc_toolkit.preprocessing.histograms import (
    CountNormalization,
    validate_time_axis,
    time_axes_compatible,
    validate_histogram,
    estimate_background,
    subtract_background,
    detect_peak,
    align_to_irf,
    crop_time_window,
    rebin_histogram,
    normalize_counts,
)

__all__ = [
    "CountNormalization",
    "validate_time_axis",
    "time_axes_compatible",
    "validate_histogram",
    "estimate_background",
    "subtract_background",
    "detect_peak",
    "align_to_irf",
    "crop_time_window",
    "rebin_histogram",
    "normalize_counts",
]
