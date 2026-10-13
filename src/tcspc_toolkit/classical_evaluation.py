"""Compatibility exports for canonical classical execution and diagnostics."""

from tcspc_toolkit.estimation.classical import (
    ReconvolutionInitialGuess,
    ReconvolutionCurveResult,
    ReconvolutionBenchmarkSummary,
    ReconvolutionBenchmarkResult,
    estimate_reconvolution_initial_guess,
    fit_single_reconvolution_curve,
    evaluate_reconvolution_benchmark,
)

__all__ = [
    "ReconvolutionInitialGuess",
    "ReconvolutionCurveResult",
    "ReconvolutionBenchmarkSummary",
    "ReconvolutionBenchmarkResult",
    "estimate_reconvolution_initial_guess",
    "fit_single_reconvolution_curve",
    "evaluate_reconvolution_benchmark",
]
