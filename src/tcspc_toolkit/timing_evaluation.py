"""Compatibility exports; use :mod:`tcspc_toolkit.benchmarks.timing`."""

from tcspc_toolkit.benchmarks.timing import (
    FloatArray,
    InferenceTimingResult,
    benchmark_fitted_regressor_runtime,
    benchmark_mean_arrival_runtime,
    benchmark_ml_inference_runtime,
    summarize_reconvolution_runtime,
    benchmark_inference_runtime,
    summarize_inference_timing,
)

__all__ = (
    "FloatArray",
    "InferenceTimingResult",
    "benchmark_fitted_regressor_runtime",
    "benchmark_mean_arrival_runtime",
    "benchmark_ml_inference_runtime",
    "summarize_reconvolution_runtime",
    "benchmark_inference_runtime",
    "summarize_inference_timing",
)
