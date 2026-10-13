"""Compatibility exports for :mod:`tcspc_toolkit.ml.evaluation`.

Historical flat imports remain supported through at least Issue #12.
"""

from tcspc_toolkit.ml.evaluation import (
    FloatArray,
    IntArray,
    BenchmarkMeasurements,
    BenchmarkDataset,
    BenchmarkSplit,
    RegressionMetrics,
    RegressionBenchmarkResult,
    HistogramRepresentations,
    generate_benchmark_measurements,
    build_benchmark_dataset,
    split_benchmark_dataset,
    build_histogram_representations,
    summarize_split_coverage,
    summarize_split_level_balance,
    evaluate_regression,
    evaluate_regressor,
    evaluate_ridge_representation_benchmark,
    evaluate_photon_count_ablation,
    evaluate_nonlinear_representation_benchmark,
    evaluate_baselines,
)

__all__ = [
    "FloatArray",
    "IntArray",
    "BenchmarkMeasurements",
    "BenchmarkDataset",
    "BenchmarkSplit",
    "RegressionMetrics",
    "RegressionBenchmarkResult",
    "HistogramRepresentations",
    "generate_benchmark_measurements",
    "build_benchmark_dataset",
    "split_benchmark_dataset",
    "build_histogram_representations",
    "summarize_split_coverage",
    "summarize_split_level_balance",
    "evaluate_regression",
    "evaluate_regressor",
    "evaluate_ridge_representation_benchmark",
    "evaluate_photon_count_ablation",
    "evaluate_nonlinear_representation_benchmark",
    "evaluate_baselines",
]
