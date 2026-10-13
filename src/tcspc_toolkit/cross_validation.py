"""Compatibility exports for :mod:`tcspc_toolkit.ml.cross_validation`.

Historical flat imports remain supported through at least Issue #12.
"""

from tcspc_toolkit.ml.cross_validation import (
    DEFAULT_CV_N_SPLITS,
    DEFAULT_CV_N_REPEATS,
    DEFAULT_CV_RANDOM_STATE,
    RepeatedCVConfig,
    RepeatedCVBenchmarkResult,
    make_repeated_kfold,
    evaluate_regressor_repeated_cv,
    CV_METRIC_COLUMNS,
    summarize_repeated_cv,
    evaluate_estimators_repeated_cv,
)

__all__ = [
    "DEFAULT_CV_N_SPLITS",
    "DEFAULT_CV_N_REPEATS",
    "DEFAULT_CV_RANDOM_STATE",
    "RepeatedCVConfig",
    "RepeatedCVBenchmarkResult",
    "make_repeated_kfold",
    "evaluate_regressor_repeated_cv",
    "CV_METRIC_COLUMNS",
    "summarize_repeated_cv",
    "evaluate_estimators_repeated_cv",
]
