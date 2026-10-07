"""Compatibility imports for :mod:`tcspc_toolkit.preprocessing.features`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings."""

from tcspc_toolkit.preprocessing.features import (
    FEATURE_NAMES,
    extract_features,
    extract_feature_table,
    quantile_arrival_time,
    half_decay_time,
    tail_log_slope,
    integrated_tail_fraction,
    early_late_count_ratio,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "extract_feature_table",
    "quantile_arrival_time",
    "half_decay_time",
    "tail_log_slope",
    "integrated_tail_fraction",
    "early_late_count_ratio",
]
