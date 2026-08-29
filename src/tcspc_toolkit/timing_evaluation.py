"""Inference-timing benchmarks for TCSPC lifetime estimators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tcspc_toolkit.baselines import (
    estimate_lifetime_from_mean_arrival,
)
from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult,
)
from tcspc_toolkit.ml_evaluation import (
    BenchmarkSplit,
)
from tcspc_toolkit.ml_models import (
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InferenceTimingResult:
    """Timing measurements for one lifetime estimator.

    Attributes
    ----------
    estimator_name:
        Stable benchmark identifier.
    timing_mode:
        Description of how runtime measurements were obtained.
    n_curves_per_call:
        Number of curves processed by one timed call.
    n_measurements:
        Number of recorded timing measurements.
    warmup_runs:
        Number of untimed warm-up calls performed before measurement.
    runtime_per_curve_ms:
        Runtime per curve for every timing measurement.
    mean_time_per_curve_ms:
        Mean runtime per curve.
    median_time_per_curve_ms:
        Median runtime per curve.
    p90_time_per_curve_ms:
        90th-percentile runtime per curve.
    median_throughput_curves_per_s:
        Throughput corresponding to the median time per curve.
    """

    estimator_name: str
    timing_mode: str

    n_curves_per_call: int
    n_measurements: int
    warmup_runs: int

    runtime_per_curve_ms: FloatArray

    mean_time_per_curve_ms: float
    median_time_per_curve_ms: float
    p90_time_per_curve_ms: float

    median_throughput_curves_per_s: float


def _validate_timing_parameters(
    *,
    n_repeats: int,
    warmup_runs: int,
) -> None:
    """Validate repeated timing configuration."""
    if (
        isinstance(n_repeats, bool)
        or not isinstance(
            n_repeats,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "n_repeats must be an integer."
        )

    if n_repeats < 1:
        raise ValueError(
            "n_repeats must be at least 1."
        )

    if (
        isinstance(warmup_runs, bool)
        or not isinstance(
            warmup_runs,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "warmup_runs must be an integer."
        )

    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs must be non-negative."
        )


def _build_timing_result(
    *,
    estimator_name: str,
    timing_mode: str,
    elapsed_times_ms: FloatArray,
    n_curves_per_call: int,
    warmup_runs: int,
) -> InferenceTimingResult:
    """Convert elapsed call times into per-curve timing statistics."""
    elapsed_array = np.asarray(
        elapsed_times_ms,
        dtype=np.float64,
    )

    if elapsed_array.ndim != 1:
        raise ValueError(
            "elapsed_times_ms must be one-dimensional."
        )

    if elapsed_array.size == 0:
        raise ValueError(
            "elapsed_times_ms must contain at least one value."
        )

    if not np.all(
        np.isfinite(elapsed_array)
    ):
        raise ValueError(
            "elapsed_times_ms must contain only finite values."
        )

    if np.any(
        elapsed_array < 0.0
    ):
        raise ValueError(
            "elapsed_times_ms must be non-negative."
        )

    if n_curves_per_call < 1:
        raise ValueError(
            "n_curves_per_call must be at least 1."
        )

    runtime_per_curve_ms = (
        elapsed_array
        / n_curves_per_call
    )

    mean_time_per_curve_ms = float(
        np.mean(
            runtime_per_curve_ms
        )
    )

    median_time_per_curve_ms = float(
        np.median(
            runtime_per_curve_ms
        )
    )

    p90_time_per_curve_ms = float(
        np.percentile(
            runtime_per_curve_ms,
            90.0,
        )
    )

    if median_time_per_curve_ms > 0.0:
        median_throughput_curves_per_s = (
            1000.0
            / median_time_per_curve_ms
        )
    else:
        median_throughput_curves_per_s = np.inf

    return InferenceTimingResult(
        estimator_name=estimator_name,
        timing_mode=timing_mode,
        n_curves_per_call=int(
            n_curves_per_call
        ),
        n_measurements=int(
            elapsed_array.size
        ),
        warmup_runs=int(
            warmup_runs
        ),
        runtime_per_curve_ms=(
            runtime_per_curve_ms
        ),
        mean_time_per_curve_ms=(
            mean_time_per_curve_ms
        ),
        median_time_per_curve_ms=(
            median_time_per_curve_ms
        ),
        p90_time_per_curve_ms=(
            p90_time_per_curve_ms
        ),
        median_throughput_curves_per_s=float(
            median_throughput_curves_per_s
        ),
    )


def _benchmark_repeated_batch_call(
    *,
    estimator_name: str,
    predict_callable: Callable[[], Any],
    n_curves: int,
    n_repeats: int,
    warmup_runs: int,
) -> InferenceTimingResult:
    """Benchmark a prediction callable using repeated batch calls."""
    _validate_timing_parameters(
        n_repeats=n_repeats,
        warmup_runs=warmup_runs,
    )

    if n_curves < 1:
        raise ValueError(
            "n_curves must be at least 1."
        )

    for _ in range(
        warmup_runs
    ):
        predict_callable()

    elapsed_times_ms = np.empty(
        n_repeats,
        dtype=np.float64,
    )

    for repeat_index in range(
        n_repeats
    ):
        start = perf_counter()

        predict_callable()

        elapsed_times_ms[
            repeat_index
        ] = (
            perf_counter()
            - start
        ) * 1000.0

    return _build_timing_result(
        estimator_name=estimator_name,
        timing_mode="repeated_batch",
        elapsed_times_ms=elapsed_times_ms,
        n_curves_per_call=n_curves,
        warmup_runs=warmup_runs,
    )


def benchmark_fitted_regressor_runtime(
    *,
    estimator_name: str,
    estimator: Any,
    X_test: Any,
    n_repeats: int = 100,
    warmup_runs: int = 5,
) -> InferenceTimingResult:
    """Benchmark ``predict`` for an already-fitted regressor.

    Notes
    -----
    Model fitting is deliberately excluded. The supplied estimator
    must already have been fitted before this function is called.
    """
    if not hasattr(
        X_test,
        "shape",
    ):
        raise ValueError(
            "X_test must provide a shape attribute."
        )

    if len(
        X_test.shape
    ) != 2:
        raise ValueError(
            "X_test must be two-dimensional."
        )

    n_curves = int(
        X_test.shape[0]
    )

    if n_curves < 1:
        raise ValueError(
            "X_test must contain at least one sample."
        )

    return _benchmark_repeated_batch_call(
        estimator_name=estimator_name,
        predict_callable=lambda: estimator.predict(
            X_test
        ),
        n_curves=n_curves,
        n_repeats=n_repeats,
        warmup_runs=warmup_runs,
    )


def benchmark_mean_arrival_runtime(
    *,
    split: BenchmarkSplit,
    n_repeats: int = 100,
    warmup_runs: int = 5,
) -> InferenceTimingResult:
    """Benchmark the physics-inspired mean-arrival estimator."""
    required_columns = (
        "mean_arrival_time_ns",
        "peak_time_ns",
    )

    missing_columns = [
        column
        for column in required_columns
        if column
        not in split.X_features_test.columns
    ]

    if missing_columns:
        raise ValueError(
            "X_features_test is missing required features: "
            + ", ".join(
                missing_columns
            )
        )

    mean_arrival_time_ns = (
        split.X_features_test[
            "mean_arrival_time_ns"
        ].to_numpy(
            dtype=np.float64
        )
    )

    peak_time_ns = (
        split.X_features_test[
            "peak_time_ns"
        ].to_numpy(
            dtype=np.float64
        )
    )

    return _benchmark_repeated_batch_call(
        estimator_name="mean_arrival_time",
        predict_callable=lambda: (
            estimate_lifetime_from_mean_arrival(
                mean_arrival_time_ns=(
                    mean_arrival_time_ns
                ),
                peak_time_ns=peak_time_ns,
            )
        ),
        n_curves=split.y_test.size,
        n_repeats=n_repeats,
        warmup_runs=warmup_runs,
    )


def benchmark_ml_inference_runtime(
    *,
    split: BenchmarkSplit,
    n_repeats: int = 100,
    warmup_runs: int = 5,
) -> dict[
    str,
    InferenceTimingResult,
]:
    """Fit ML estimators once and benchmark prediction runtime.

    Training is performed before any timing measurement. Only repeated
    ``predict`` calls on the test feature table contribute to reported
    inference runtime.
    """
    model_factories = {
        "ridge": make_ridge_pipeline,
        "random_forest": (
            make_random_forest_pipeline
        ),
        "hist_gradient_boosting": (
            make_hist_gradient_boosting_pipeline
        ),
    }

    timing_results: dict[
        str,
        InferenceTimingResult,
    ] = {}

    for (
        estimator_name,
        make_estimator,
    ) in model_factories.items():
        estimator = make_estimator()

        # Training is intentionally outside
        # the timed prediction benchmark.
        estimator.fit(
            split.X_features_train,
            split.y_train,
        )

        timing_results[
            estimator_name
        ] = (
            benchmark_fitted_regressor_runtime(
                estimator_name=estimator_name,
                estimator=estimator,
                X_test=split.X_features_test,
                n_repeats=n_repeats,
                warmup_runs=warmup_runs,
            )
        )

    return timing_results


def summarize_reconvolution_runtime(
    *,
    result: ReconvolutionBenchmarkResult,
) -> InferenceTimingResult:
    """Summarize existing per-curve reconvolution optimization runtimes.

    Day 47 records optimization time independently for every fitted
    histogram. These measurements are reused here rather than running
    the expensive benchmark again solely for timing.
    """
    if "runtime_ms" not in result.per_curve.columns:
        raise ValueError(
            "Reconvolution result must contain runtime_ms."
        )

    runtime_values = (
        result.per_curve[
            "runtime_ms"
        ].to_numpy(
            dtype=np.float64
        )
    )

    finite_runtime_values = (
        runtime_values[
            np.isfinite(
                runtime_values
            )
        ]
    )

    if finite_runtime_values.size == 0:
        raise ValueError(
            "Reconvolution result contains no finite runtimes."
        )

    return _build_timing_result(
        estimator_name="reconvolution",
        timing_mode="per_curve_optimization",
        elapsed_times_ms=(
            finite_runtime_values
        ),
        n_curves_per_call=1,
        warmup_runs=0,
    )


def benchmark_inference_runtime(
    *,
    split: BenchmarkSplit,
    reconvolution_result: (
        ReconvolutionBenchmarkResult
        | None
    ) = None,
    n_repeats: int = 100,
    warmup_runs: int = 5,
) -> dict[
    str,
    InferenceTimingResult,
]:
    """Benchmark all Week 7 estimators used in timing comparisons."""
    results = {
        "mean_arrival_time": (
            benchmark_mean_arrival_runtime(
                split=split,
                n_repeats=n_repeats,
                warmup_runs=warmup_runs,
            )
        ),
        **benchmark_ml_inference_runtime(
            split=split,
            n_repeats=n_repeats,
            warmup_runs=warmup_runs,
        ),
    }

    if reconvolution_result is not None:
        results[
            "reconvolution"
        ] = summarize_reconvolution_runtime(
            result=reconvolution_result
        )

    return results


def summarize_inference_timing(
    timing_results: dict[
        str,
        InferenceTimingResult,
    ],
) -> pd.DataFrame:
    """Create a compact comparison table of inference cost."""
    rows: list[
        dict[str, float | int | str]
    ] = []

    for result in timing_results.values():
        rows.append(
            {
                "estimator": (
                    result.estimator_name
                ),
                "timing_mode": (
                    result.timing_mode
                ),
                "n_curves_per_call": (
                    result.n_curves_per_call
                ),
                "n_measurements": (
                    result.n_measurements
                ),
                "mean_time_per_curve_ms": (
                    result.mean_time_per_curve_ms
                ),
                "median_time_per_curve_ms": (
                    result.median_time_per_curve_ms
                ),
                "p90_time_per_curve_ms": (
                    result.p90_time_per_curve_ms
                ),
                "median_throughput_curves_per_s": (
                    result
                    .median_throughput_curves_per_s
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "median_time_per_curve_ms"
        )
        .reset_index(
            drop=True
        )
    )
