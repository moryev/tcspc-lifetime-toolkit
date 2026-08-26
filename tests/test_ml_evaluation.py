import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.features import FEATURE_NAMES
from tcspc_toolkit.ml_evaluation import (
    BenchmarkDataset,
    BenchmarkSplit,
    build_benchmark_dataset,
    build_histogram_representations,
    evaluate_baselines,
    evaluate_nonlinear_representation_benchmark,
    evaluate_regression,
    evaluate_regressor,
    evaluate_ridge_representation_benchmark,
    evaluate_photon_count_ablation,
    generate_benchmark_measurements,
    split_benchmark_dataset,
    summarize_split_coverage,
    summarize_split_level_balance,
)
from tcspc_toolkit.ml_models import (
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)

def _make_benchmark_dataset() -> BenchmarkDataset:
    n_samples = 10

    X_features = pd.DataFrame(
        {
            "feature_a": np.arange(
                n_samples,
                dtype=np.float64,
            ),
            "feature_b": np.arange(
                100,
                100 + n_samples,
                dtype=np.float64,
            ),
        }
    )

    X_histograms = np.column_stack(
        (
            np.arange(n_samples),
            np.arange(n_samples) + 10,
            np.arange(n_samples) + 20,
        )
    ).astype(np.int64)

    y = np.arange(
        1,
        n_samples + 1,
        dtype=np.float64,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": np.arange(
                n_samples,
                dtype=np.int64,
            ),
            "background_per_bin": np.arange(
                n_samples,
                dtype=np.float64,
            ),
        }
    )

    return BenchmarkDataset(
        X_features=X_features,
        X_histograms=X_histograms,
        y=y,
        metadata=metadata,
    )


def _make_benchmark_measurements():
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    return generate_benchmark_measurements(
        time=time,
        lifetimes_ns=np.array(
            [1.0, 2.0],
            dtype=np.float64,
        ),
        signal_photon_counts=np.array(
            [100_000],
            dtype=np.int64,
        ),
        background_levels=np.array(
            [1.0],
            dtype=np.float64,
        ),
        irf_centre_ns=1.0,
        irf_fwhm_values_ns=np.array(
            [0.3],
            dtype=np.float64,
        ),
        irf_shift_values_ns=np.array(
            [0.0],
            dtype=np.float64,
        ),
        random_seed=42,
    )


def _make_benchmark_feature_config() -> FeatureConfig:
    return FeatureConfig(
        tail_start_ns=5.0,
        early_stop_ns=3.0,
        late_start_ns=7.0,
    )


def test_perfect_predictions_have_zero_error() -> None:
    y_true = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    metrics = evaluate_regression(
        y_true=y_true,
        y_pred=y_true.copy(),
    )

    assert metrics.mae_ns == pytest.approx(0.0)
    assert metrics.median_absolute_error_ns == pytest.approx(0.0)
    assert metrics.rmse_ns == pytest.approx(0.0)
    assert metrics.mean_relative_error == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)


def test_benchmark_split_is_reproducible() -> None:
    dataset = _make_benchmark_dataset()

    split_a = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    split_b = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    np.testing.assert_array_equal(
        split_a.train_indices,
        split_b.train_indices,
    )

    np.testing.assert_array_equal(
        split_a.test_indices,
        split_b.test_indices,
    )


def test_benchmark_split_has_no_sample_overlap() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    overlap = np.intersect1d(
        split.train_indices,
        split.test_indices,
    )

    assert overlap.size == 0


def test_benchmark_split_preserves_target_alignment() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    np.testing.assert_array_equal(
        split.y_train,
        dataset.y[split.train_indices],
    )

    np.testing.assert_array_equal(
        split.y_test,
        dataset.y[split.test_indices],
    )


def test_benchmark_split_preserves_metadata_alignment() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    expected_train_sample_ids = (
        dataset.metadata.iloc[
            split.train_indices
        ]["sample_id"]
        .to_numpy()
    )

    expected_test_sample_ids = (
        dataset.metadata.iloc[
            split.test_indices
        ]["sample_id"]
        .to_numpy()
    )

    np.testing.assert_array_equal(
        split.metadata_train["sample_id"].to_numpy(),
        expected_train_sample_ids,
    )

    np.testing.assert_array_equal(
        split.metadata_test["sample_id"].to_numpy(),
        expected_test_sample_ids,
    )


def test_benchmark_split_uses_identical_sample_membership_across_representations() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    train_sample_ids_from_features = (
        split.X_features_train["feature_a"]
        .to_numpy(dtype=np.int64)
    )

    train_sample_ids_from_histograms = (
        split.X_histograms_train[:, 0]
    )

    train_sample_ids_from_metadata = (
        split.metadata_train["sample_id"]
        .to_numpy()
    )

    np.testing.assert_array_equal(
        train_sample_ids_from_features,
        train_sample_ids_from_histograms,
    )

    np.testing.assert_array_equal(
        train_sample_ids_from_features,
        train_sample_ids_from_metadata,
    )


def test_benchmark_split_rejects_misaligned_dataset() -> None:
    dataset = _make_benchmark_dataset()

    misaligned_dataset = BenchmarkDataset(
        X_features=dataset.X_features.iloc[:-1],
        X_histograms=dataset.X_histograms,
        y=dataset.y,
        metadata=dataset.metadata,
    )

    with pytest.raises(
        ValueError,
        match="X_features and y",
    ):
        split_benchmark_dataset(
            misaligned_dataset,
        )


def test_generate_benchmark_measurements_has_expected_shape() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    measurements = generate_benchmark_measurements(
        time=time,
        lifetimes_ns=np.array(
            [1.0, 2.0],
            dtype=np.float64,
        ),
        signal_photon_counts=np.array(
            [1_000, 5_000],
            dtype=np.int64,
        ),
        background_levels=np.array(
            [0.0, 0.5],
            dtype=np.float64,
        ),
        irf_centre_ns=1.0,
        irf_fwhm_values_ns=np.array(
            [0.2],
            dtype=np.float64,
        ),
        irf_shift_values_ns=np.array(
            [-0.1, 0.1],
            dtype=np.float64,
        ),
        random_seed=42,
    )

    expected_n_samples = (
        2
        * 2
        * 2
        * 1
        * 2
    )

    assert measurements.X_histograms.shape == (
        expected_n_samples,
        time.size,
    )

    assert measurements.y.shape == (
        expected_n_samples,
    )

    assert measurements.metadata.shape[0] == (
        expected_n_samples
    )

    np.testing.assert_array_equal(
        measurements.metadata[
            "sample_id"
        ].to_numpy(),
        np.arange(
            expected_n_samples,
            dtype=np.int64,
        ),
    )


def test_generate_benchmark_measurements_is_reproducible() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    common_arguments = {
        "time": time,
        "lifetimes_ns": np.array(
            [1.0, 2.0],
            dtype=np.float64,
        ),
        "signal_photon_counts": np.array(
            [1_000],
            dtype=np.int64,
        ),
        "background_levels": np.array(
            [0.0, 0.5],
            dtype=np.float64,
        ),
        "irf_centre_ns": 1.0,
        "irf_fwhm_values_ns": np.array(
            [0.2],
            dtype=np.float64,
        ),
        "irf_shift_values_ns": np.array(
            [0.0],
            dtype=np.float64,
        ),
        "random_seed": 42,
    }

    measurements_a = generate_benchmark_measurements(
        **common_arguments
    )

    measurements_b = generate_benchmark_measurements(
        **common_arguments
    )

    np.testing.assert_array_equal(
        measurements_a.X_histograms,
        measurements_b.X_histograms,
    )

    np.testing.assert_array_equal(
        measurements_a.y,
        measurements_b.y,
    )

    pd.testing.assert_frame_equal(
        measurements_a.metadata,
        measurements_b.metadata,
    )


def test_benchmark_metadata_does_not_contain_lifetime_target() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    measurements = generate_benchmark_measurements(
        time=time,
        lifetimes_ns=np.array(
            [1.0, 2.0],
            dtype=np.float64,
        ),
        signal_photon_counts=np.array(
            [1_000],
            dtype=np.int64,
        ),
        background_levels=np.array(
            [0.0],
            dtype=np.float64,
        ),
        irf_centre_ns=1.0,
        irf_fwhm_values_ns=np.array(
            [0.2],
            dtype=np.float64,
        ),
        irf_shift_values_ns=np.array(
            [0.0],
            dtype=np.float64,
        ),
        random_seed=42,
    )

    assert "lifetime_true_ns" not in (
        measurements.metadata.columns
    )


def test_build_benchmark_dataset_has_aligned_shapes() -> None:
    measurements = _make_benchmark_measurements()

    dataset = build_benchmark_dataset(
        measurements,
        feature_config=_make_benchmark_feature_config(),
    )

    n_samples = measurements.X_histograms.shape[0]

    assert dataset.X_features.shape == (
        n_samples,
        len(FEATURE_NAMES),
    )

    assert dataset.X_histograms.shape == (
        n_samples,
        measurements.time.size,
    )

    assert dataset.y.shape == (
        n_samples,
    )

    assert dataset.metadata.shape[0] == (
        n_samples
    )


def test_build_benchmark_dataset_uses_stable_feature_schema() -> None:
    measurements = _make_benchmark_measurements()

    dataset = build_benchmark_dataset(
        measurements,
        feature_config=_make_benchmark_feature_config(),
    )

    assert list(
        dataset.X_features.columns
    ) == list(
        FEATURE_NAMES
    )


def test_build_benchmark_dataset_preserves_histograms() -> None:
    measurements = _make_benchmark_measurements()

    dataset = build_benchmark_dataset(
        measurements,
        feature_config=_make_benchmark_feature_config(),
    )

    np.testing.assert_array_equal(
        dataset.X_histograms,
        measurements.X_histograms,
    )


def test_build_benchmark_dataset_preserves_targets_and_metadata() -> None:
    measurements = _make_benchmark_measurements()

    dataset = build_benchmark_dataset(
        measurements,
        feature_config=_make_benchmark_feature_config(),
    )

    np.testing.assert_array_equal(
        dataset.y,
        measurements.y,
    )

    pd.testing.assert_frame_equal(
        dataset.metadata,
        measurements.metadata,
    )


def test_engineered_total_counts_match_corresponding_histograms() -> None:
    measurements = _make_benchmark_measurements()

    dataset = build_benchmark_dataset(
        measurements,
        feature_config=_make_benchmark_feature_config(),
    )

    histogram_total_counts = (
        dataset.X_histograms.sum(
            axis=1
        )
    )

    feature_total_counts = (
        dataset.X_features[
            "total_counts"
        ].to_numpy()
    )

    np.testing.assert_allclose(
        feature_total_counts,
        histogram_total_counts,
    )


def _make_coverage_split() -> BenchmarkSplit:
    X_features_train = pd.DataFrame(
        {
            "feature": [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    X_features_test = pd.DataFrame(
        {
            "feature": [
                4.0,
                5.0,
            ],
        }
    )

    X_histograms_train = np.zeros(
        (4, 3),
        dtype=np.int64,
    )

    X_histograms_test = np.zeros(
        (2, 3),
        dtype=np.int64,
    )

    y_train = np.array(
        [
            1.0,
            2.0,
            1.0,
            2.0,
        ],
        dtype=np.float64,
    )

    y_test = np.array(
        [
            1.0,
            2.0,
        ],
        dtype=np.float64,
    )

    metadata_train = pd.DataFrame(
        {
            "sample_id": [
                0,
                1,
                2,
                3,
            ],
            "signal_photon_count_target": [
                1_000,
                5_000,
                1_000,
                5_000,
            ],
            "background_per_bin": [
                0.0,
                0.5,
                0.0,
                0.5,
            ],
            "irf_fwhm_ns": [
                0.2,
                0.4,
                0.2,
                0.4,
            ],
            "irf_shift_ns": [
                -0.1,
                0.1,
                -0.1,
                0.1,
            ],
        }
    )

    metadata_test = pd.DataFrame(
        {
            "sample_id": [
                4,
                5,
            ],
            "signal_photon_count_target": [
                1_000,
                5_000,
            ],
            "background_per_bin": [
                0.0,
                0.5,
            ],
            "irf_fwhm_ns": [
                0.2,
                0.4,
            ],
            "irf_shift_ns": [
                -0.1,
                0.1,
            ],
        }
    )

    return BenchmarkSplit(
        train_indices=np.array(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
        test_indices=np.array(
            [4, 5],
            dtype=np.int64,
        ),
        X_features_train=X_features_train,
        X_features_test=X_features_test,
        X_histograms_train=X_histograms_train,
        X_histograms_test=X_histograms_test,
        y_train=y_train,
        y_test=y_test,
        metadata_train=metadata_train,
        metadata_test=metadata_test,
    )


def test_split_coverage_summarizes_target_and_nuisance_variables() -> None:
    split = _make_coverage_split()

    summary = summarize_split_coverage(
        split
    )

    assert list(
        summary["parameter"]
    ) == [
        "lifetime_ns",
        "signal_photon_count_target",
        "background_per_bin",
        "irf_fwhm_ns",
        "irf_shift_ns",
    ]


def test_split_coverage_detects_matching_parameter_support() -> None:
    split = _make_coverage_split()

    summary = summarize_split_coverage(
        split
    )

    assert summary[
        "same_support"
    ].all()


def test_split_coverage_reports_correct_lifetime_range() -> None:
    split = _make_coverage_split()

    summary = summarize_split_coverage(
        split
    )

    lifetime_row = (
        summary
        .set_index("parameter")
        .loc["lifetime_ns"]
    )

    assert lifetime_row[
        "train_min"
    ] == pytest.approx(1.0)

    assert lifetime_row[
        "train_max"
    ] == pytest.approx(2.0)

    assert lifetime_row[
        "test_min"
    ] == pytest.approx(1.0)

    assert lifetime_row[
        "test_max"
    ] == pytest.approx(2.0)

    assert lifetime_row[
        "train_n_unique"
    ] == 2

    assert lifetime_row[
        "test_n_unique"
    ] == 2


def test_split_level_balance_reports_counts_and_fractions() -> None:
    split = _make_coverage_split()

    balance = summarize_split_level_balance(
        split
    )

    lifetime_rows = (
        balance[
            balance["parameter"]
            == "lifetime_ns"
        ]
        .set_index("value")
    )

    assert lifetime_rows.loc[
        1.0,
        "train_count",
    ] == 2

    assert lifetime_rows.loc[
        1.0,
        "test_count",
    ] == 1

    assert lifetime_rows.loc[
        1.0,
        "train_fraction",
    ] == pytest.approx(0.5)

    assert lifetime_rows.loc[
        1.0,
        "test_fraction",
    ] == pytest.approx(0.5)


def _make_baseline_benchmark_split() -> BenchmarkSplit:
    X_features_train = pd.DataFrame(
        {
            "mean_arrival_time_ns": [
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "peak_time_ns": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        }
    )

    X_features_test = pd.DataFrame(
        {
            "mean_arrival_time_ns": [
                2.0,
                3.5,
                5.0,
            ],
            "peak_time_ns": [
                1.0,
                1.0,
                1.0,
            ],
        }
    )

    X_histograms_train = np.zeros(
        (4, 3),
        dtype=np.int64,
    )

    X_histograms_test = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    y_train = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        dtype=np.float64,
    )

    y_test = np.array(
        [
            1.0,
            2.5,
            4.0,
        ],
        dtype=np.float64,
    )

    metadata_train = pd.DataFrame(
        {
            "sample_id": [
                0,
                1,
                2,
                3,
            ],
        }
    )

    metadata_test = pd.DataFrame(
        {
            "sample_id": [
                4,
                5,
                6,
            ],
        }
    )

    return BenchmarkSplit(
        train_indices=np.array(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
        test_indices=np.array(
            [4, 5, 6],
            dtype=np.int64,
        ),
        X_features_train=X_features_train,
        X_features_test=X_features_test,
        X_histograms_train=X_histograms_train,
        X_histograms_test=X_histograms_test,
        y_train=y_train,
        y_test=y_test,
        metadata_train=metadata_train,
        metadata_test=metadata_test,
    )


def test_evaluate_baselines_produces_one_prediction_per_test_sample() -> None:
    split = _make_baseline_benchmark_split()

    results = evaluate_baselines(
        split
    )

    assert set(results) == {
        "constant_mean",
        "mean_arrival_time",
    }

    for result in results.values():
        assert result.y_pred.shape == (
            split.y_test.shape
        )


def test_evaluate_baselines_uses_training_mean_for_constant_baseline() -> None:
    split = _make_baseline_benchmark_split()

    results = evaluate_baselines(
        split
    )

    expected_prediction = np.mean(
        split.y_train
    )

    np.testing.assert_allclose(
        results[
            "constant_mean"
        ].y_pred,
        expected_prediction,
    )


def test_evaluate_baselines_uses_mean_arrival_features() -> None:
    split = _make_baseline_benchmark_split()

    results = evaluate_baselines(
        split
    )

    expected_predictions = (
        split.X_features_test[
            "mean_arrival_time_ns"
        ].to_numpy()
        - split.X_features_test[
            "peak_time_ns"
        ].to_numpy()
    )

    np.testing.assert_allclose(
        results[
            "mean_arrival_time"
        ].y_pred,
        expected_predictions,
    )


def test_evaluate_baselines_attaches_regression_metrics() -> None:
    split = _make_baseline_benchmark_split()

    results = evaluate_baselines(
        split
    )

    for result in results.values():
        expected_metrics = evaluate_regression(
            y_true=split.y_test,
            y_pred=result.y_pred,
        )

        assert (
            result.metrics.mae_ns
            == pytest.approx(
                expected_metrics.mae_ns
            )
        )

        assert (
            result.metrics.median_absolute_error_ns
            == pytest.approx(
                expected_metrics.median_absolute_error_ns
            )
        )

        assert (
            result.metrics.mean_relative_error
            == pytest.approx(
                expected_metrics.mean_relative_error
            )
        )

        assert (
            result.metrics.median_relative_error
            == pytest.approx(
                expected_metrics.median_relative_error
            )
        )

        assert (
            result.metrics.r2
            == pytest.approx(
                expected_metrics.r2
            )
        )


def test_regression_metrics_are_computed_correctly() -> None:
    y_true = np.array(
        [1.0, 2.0, 4.0],
        dtype=np.float64,
    )

    y_pred = np.array(
        [1.5, 1.0, 5.0],
        dtype=np.float64,
    )

    metrics = evaluate_regression(
        y_true=y_true,
        y_pred=y_pred,
    )

    expected_absolute_errors = np.array(
        [0.5, 1.0, 1.0],
        dtype=np.float64,
    )

    expected_rmse = np.sqrt(
        np.mean(
            expected_absolute_errors ** 2
        )
    )

    assert metrics.mae_ns == pytest.approx(
        np.mean(expected_absolute_errors)
    )

    assert metrics.median_absolute_error_ns == pytest.approx(
        np.median(expected_absolute_errors)
    )

    assert metrics.rmse_ns == pytest.approx(
        expected_rmse
    )


def _make_regression_test_data():
    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(80, 4)
    )

    y = (
        3.0
        + 0.5 * X[:, 0]
        - 0.2 * X[:, 1]
        + 0.1 * X[:, 2]
    )

    X_train = X[:60]
    X_test = X[60:]

    y_train = y[:60]
    y_test = y[60:]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


@pytest.mark.parametrize(
    "estimator_name, estimator_factory",
    [
        (
            "ridge",
            make_ridge_pipeline,
        ),
        (
            "random_forest",
            make_random_forest_pipeline,
        ),
        (
            "hist_gradient_boosting",
            make_hist_gradient_boosting_pipeline,
        ),
    ],
)
def test_evaluate_regressor_trains_and_predicts(
    estimator_name,
    estimator_factory,
) -> None:
    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = _make_regression_test_data()

    estimator = estimator_factory()

    result = evaluate_regressor(
        estimator_name=estimator_name,
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    assert result.estimator_name == estimator_name

    assert result.y_pred.shape == (
        y_test.shape
    )

    assert np.all(
        np.isfinite(result.y_pred)
    )

    assert result.relative_errors.shape == (
        y_test.shape
    )

    assert np.all(
        np.isfinite(result.relative_errors)
    )


def test_random_forest_is_deterministic_with_fixed_random_state() -> None:
    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = _make_regression_test_data()

    estimator_a = make_random_forest_pipeline(
        random_state=42,
    )

    estimator_b = make_random_forest_pipeline(
        random_state=42,
    )

    result_a = evaluate_regressor(
        estimator_name="random_forest",
        estimator=estimator_a,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    result_b = evaluate_regressor(
        estimator_name="random_forest",
        estimator=estimator_b,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    np.testing.assert_allclose(
        result_a.y_pred,
        result_b.y_pred,
    )


class _FixedPredictionRegressor:
    def __init__(
        self,
        predictions: np.ndarray,
    ) -> None:
        self.predictions = predictions

    def fit(
        self,
        X,
        y,
    ):
        return self

    def predict(
        self,
        X,
    ) -> np.ndarray:
        return self.predictions


def test_evaluate_regressor_computes_metrics_from_predictions() -> None:
    X_train = np.zeros(
        (3, 2),
        dtype=np.float64,
    )

    y_train = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    X_test = np.zeros(
        (3, 2),
        dtype=np.float64,
    )

    y_test = np.array(
        [1.0, 2.0, 4.0],
        dtype=np.float64,
    )

    predictions = np.array(
        [1.5, 1.0, 5.0],
        dtype=np.float64,
    )

    estimator = _FixedPredictionRegressor(
        predictions
    )

    result = evaluate_regressor(
        estimator_name="fixed",
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    expected_metrics = evaluate_regression(
        y_true=y_test,
        y_pred=predictions,
    )

    assert result.metrics == expected_metrics


def test_test_targets_do_not_affect_predictions() -> None:
    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = _make_regression_test_data()

    alternative_y_test = (
        y_test + 10.0
    )

    result_a = evaluate_regressor(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    result_b = evaluate_regressor(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=alternative_y_test,
    )

    np.testing.assert_allclose(
        result_a.y_pred,
        result_b.y_pred,
    )


def test_evaluate_regressor_rejects_misaligned_training_samples() -> None:
    X_train = np.zeros(
        (4, 2),
        dtype=np.float64,
    )

    y_train = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    X_test = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    y_test = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="X_train and y_train",
    ):
        evaluate_regressor(
            estimator_name="ridge",
            estimator=make_ridge_pipeline(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )


def test_build_histogram_representations_has_expected_shapes() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    assert representations.X_normalized_train.shape == (
        split.X_histograms_train.shape
    )

    assert representations.X_normalized_test.shape == (
        split.X_histograms_test.shape
    )

    assert representations.X_pca_train.shape == (
        split.y_train.size,
        2,
    )

    assert representations.X_pca_test.shape == (
        split.y_test.size,
        2,
    )

    np.testing.assert_allclose(
        representations.X_normalized_train.sum(axis=1),
        1.0,
    )

    np.testing.assert_allclose(
        representations.X_normalized_test.sum(axis=1),
        1.0,
    )


def test_build_histogram_representations_fits_pca_on_training_data_only() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    expected_training_mean = (
        representations.X_normalized_train.mean(axis=0)
    )

    np.testing.assert_allclose(
        representations.pca.mean_,
        expected_training_mean,
    )


def test_ridge_representation_benchmark_evaluates_all_representations() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    results = evaluate_ridge_representation_benchmark(
        split,
        representations,
    )

    assert set(results) == {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    for result in results.values():
        assert result.y_pred.shape == split.y_test.shape
        assert np.all(np.isfinite(result.y_pred))


def test_ridge_representation_benchmark_uses_stable_names() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    results = evaluate_ridge_representation_benchmark(
        split,
        representations,
    )

    assert (
        results["engineered_features"].estimator_name
        == "ridge_engineered_features"
    )

    assert (
        results["normalized_histogram"].estimator_name
        == "ridge_normalized_histogram"
    )

    assert (
        results["pca_histogram"].estimator_name
        == "ridge_pca_histogram"
    )


def test_photon_count_ablation_evaluates_all_representations() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    results = evaluate_photon_count_ablation(
        split,
        representations,
    )

    assert set(results) == {
        "normalized_histogram",
        "normalized_histogram_with_total_counts",
        "pca_histogram",
        "pca_histogram_with_total_counts",
    }

    for result in results.values():
        assert result.y_pred.shape == split.y_test.shape
        assert np.all(np.isfinite(result.y_pred))


def test_nonlinear_representation_benchmark_evaluates_all_combinations() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    results = evaluate_nonlinear_representation_benchmark(
        split,
        representations,
    )

    assert set(results) == {
        "random_forest",
        "hist_gradient_boosting",
    }

    expected_representations = {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    for model_results in results.values():
        assert set(model_results) == (
            expected_representations
        )

        for result in model_results.values():
            assert result.y_pred.shape == (
                split.y_test.shape
            )

            assert np.all(
                np.isfinite(result.y_pred)
            )



def test_nonlinear_representation_benchmark_uses_stable_names() -> None:
    dataset = _make_benchmark_dataset()

    split = split_benchmark_dataset(
        dataset,
        test_size=0.3,
        random_state=42,
    )

    representations = build_histogram_representations(
        split,
        n_pca_components=2,
    )

    results = evaluate_nonlinear_representation_benchmark(
        split,
        representations,
    )

    assert (
        results[
            "random_forest"
        ][
            "engineered_features"
        ].estimator_name
        == "random_forest_engineered_features"
    )

    assert (
        results[
            "hist_gradient_boosting"
        ][
            "pca_histogram"
        ].estimator_name
        == "hist_gradient_boosting_pca_histogram"
    )


