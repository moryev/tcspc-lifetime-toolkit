import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult,
    ReconvolutionBenchmarkSummary,
)
from tcspc_toolkit.ml_evaluation import (
    BenchmarkSplit,
)
from tcspc_toolkit.ml_models import (
    make_ridge_pipeline,
)
from tcspc_toolkit.timing_evaluation import (
    benchmark_fitted_regressor_runtime,
    benchmark_inference_runtime,
    benchmark_mean_arrival_runtime,
    benchmark_ml_inference_runtime,
    summarize_inference_timing,
    summarize_reconvolution_runtime,
)


def _make_timing_split() -> BenchmarkSplit:
    rng = np.random.default_rng(42)

    n_train = 20
    n_test = 8
    n_features = 5
    n_bins = 32

    X_features_train = pd.DataFrame(
        rng.normal(
            size=(
                n_train,
                n_features,
            )
        ),
        columns=[
            "mean_arrival_time_ns",
            "peak_time_ns",
            "feature_2",
            "feature_3",
            "feature_4",
        ],
    )

    X_features_test = pd.DataFrame(
        rng.normal(
            size=(
                n_test,
                n_features,
            )
        ),
        columns=(
            X_features_train.columns
        ),
    )

    y_train = np.linspace(
        1.0,
        4.0,
        n_train,
        dtype=np.float64,
    )

    y_test = np.linspace(
        1.0,
        4.0,
        n_test,
        dtype=np.float64,
    )

    X_histograms_train = np.ones(
        (
            n_train,
            n_bins,
        ),
        dtype=np.int64,
    )

    X_histograms_test = np.ones(
        (
            n_test,
            n_bins,
        ),
        dtype=np.int64,
    )

    return BenchmarkSplit(
        train_indices=np.arange(
            n_train,
            dtype=np.int64,
        ),
        test_indices=np.arange(
            n_train,
            n_train + n_test,
            dtype=np.int64,
        ),
        X_features_train=(
            X_features_train
        ),
        X_features_test=(
            X_features_test
        ),
        X_histograms_train=(
            X_histograms_train
        ),
        X_histograms_test=(
            X_histograms_test
        ),
        y_train=y_train,
        y_test=y_test,
        metadata_train=pd.DataFrame(
            index=range(n_train)
        ),
        metadata_test=pd.DataFrame(
            index=range(n_test)
        ),
    )


def test_fitted_regressor_runtime_returns_measurements() -> None:
    split = _make_timing_split()

    estimator = make_ridge_pipeline()

    estimator.fit(
        split.X_features_train,
        split.y_train,
    )

    result = (
        benchmark_fitted_regressor_runtime(
            estimator_name="ridge",
            estimator=estimator,
            X_test=split.X_features_test,
            n_repeats=5,
            warmup_runs=2,
        )
    )

    assert result.estimator_name == "ridge"
    assert result.timing_mode == "repeated_batch"

    assert result.n_curves_per_call == (
        split.y_test.size
    )

    assert result.n_measurements == 5
    assert result.warmup_runs == 2

    assert (
        result.runtime_per_curve_ms.shape
        == (5,)
    )

    assert np.all(
        result.runtime_per_curve_ms
        >= 0.0
    )


def test_fitted_regressor_timing_does_not_call_fit() -> None:
    class TrackingEstimator:
        def __init__(self) -> None:
            self.fit_calls = 0
            self.predict_calls = 0

        def fit(
            self,
            X,
            y,
        ):
            self.fit_calls += 1
            return self

        def predict(
            self,
            X,
        ):
            self.predict_calls += 1

            return np.zeros(
                X.shape[0],
                dtype=np.float64,
            )

    estimator = TrackingEstimator()

    X_test = np.ones(
        (6, 3),
        dtype=np.float64,
    )

    benchmark_fitted_regressor_runtime(
        estimator_name="tracking",
        estimator=estimator,
        X_test=X_test,
        n_repeats=4,
        warmup_runs=2,
    )

    assert estimator.fit_calls == 0

    assert estimator.predict_calls == (
        4 + 2
    )


def test_mean_arrival_runtime_uses_test_batch() -> None:
    split = _make_timing_split()

    result = benchmark_mean_arrival_runtime(
        split=split,
        n_repeats=4,
        warmup_runs=1,
    )

    assert (
        result.estimator_name
        == "mean_arrival_time"
    )

    assert result.n_curves_per_call == (
        split.y_test.size
    )

    assert result.n_measurements == 4


def test_ml_inference_runtime_returns_all_models() -> None:
    split = _make_timing_split()

    results = benchmark_ml_inference_runtime(
        split=split,
        n_repeats=3,
        warmup_runs=1,
    )

    assert set(
        results
    ) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    for result in results.values():
        assert result.n_measurements == 3

        assert (
            result.n_curves_per_call
            == split.y_test.size
        )


def test_reconvolution_runtime_uses_existing_per_curve_timings() -> None:
    per_curve = pd.DataFrame(
        {
            "runtime_ms": [
                2.0,
                4.0,
                6.0,
                np.nan,
            ],
        }
    )

    summary = ReconvolutionBenchmarkSummary(
        n_samples=4,
        n_successful_fits=3,
        n_failed_fits=1,
        success_rate=0.75,
        failure_rate=0.25,
        mae_valid_ns=0.1,
        median_absolute_error_valid_ns=0.1,
        rmse_valid_ns=0.1,
        mean_runtime_ms=4.0,
        median_runtime_ms=4.0,
    )

    reconvolution_result = (
        ReconvolutionBenchmarkResult(
            per_curve=per_curve,
            summary=summary,
        )
    )

    timing = summarize_reconvolution_runtime(
        result=reconvolution_result
    )

    np.testing.assert_allclose(
        timing.runtime_per_curve_ms,
        np.array(
            [2.0, 4.0, 6.0]
        ),
    )

    assert (
        timing.median_time_per_curve_ms
        == pytest.approx(4.0)
    )

    assert (
        timing.mean_time_per_curve_ms
        == pytest.approx(4.0)
    )

    assert (
        timing.median_throughput_curves_per_s
        == pytest.approx(250.0)
    )


def test_integrated_inference_timing_contains_fast_estimators() -> None:
    split = _make_timing_split()

    results = benchmark_inference_runtime(
        split=split,
        n_repeats=3,
        warmup_runs=1,
    )

    assert set(
        results
    ) == {
        "mean_arrival_time",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }


def test_inference_timing_summary_has_expected_columns() -> None:
    split = _make_timing_split()

    results = benchmark_inference_runtime(
        split=split,
        n_repeats=3,
        warmup_runs=1,
    )

    summary = summarize_inference_timing(
        results
    )

    expected_columns = {
        "estimator",
        "timing_mode",
        "n_curves_per_call",
        "n_measurements",
        "mean_time_per_curve_ms",
        "median_time_per_curve_ms",
        "p90_time_per_curve_ms",
        "median_throughput_curves_per_s",
    }

    assert expected_columns.issubset(
        summary.columns
    )

    assert summary.shape[0] == 4


