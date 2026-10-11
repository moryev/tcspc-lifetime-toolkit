"""Compatibility exports; use :mod:`tcspc_toolkit.benchmarks.mismatch`."""

from tcspc_toolkit.benchmarks.mismatch import (
    EstimatorMismatchResult,
    ClassicalMismatchResult,
    generate_matched_biexponential_mismatch_dataset,
    evaluate_ml_mismatch_benchmark,
    evaluate_classical_mismatch_benchmark,
    summarize_mismatch_benchmark,
)

__all__ = (
    "EstimatorMismatchResult",
    "ClassicalMismatchResult",
    "generate_matched_biexponential_mismatch_dataset",
    "evaluate_ml_mismatch_benchmark",
    "evaluate_classical_mismatch_benchmark",
    "summarize_mismatch_benchmark",
)
