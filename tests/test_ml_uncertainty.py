from sklearn.decomposition import PCA

from tcspc_toolkit.generalization import (
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    GeneralizationTestMeasurements,
)
from tcspc_toolkit.generalization_evaluation import (
    GeneralizationPreparedData,
)

import numpy as np
import pytest

import pandas as pd

from tcspc_toolkit.ml_evaluation import (
    BenchmarkDataset,
)
from tcspc_toolkit.ml_uncertainty import (
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_MEDIAN_QUANTILE,
    DEFAULT_QUANTILE_NOMINAL_COVERAGE,
    DEFAULT_UPPER_QUANTILE,
    QUANTILE_GRADIENT_BOOSTING_METHOD_ID,
    QuantileGradientBoostingExternalResult,
    QuantileGradientBoostingIntervalEstimator,
    evaluate_frozen_quantile_gradient_boosting,
    fit_and_evaluate_quantile_gradient_boosting,
    QuantileGradientBoostingRobustnessResult,
    evaluate_week9_quantile_robustness_conditions,
    QuantileGradientBoostingPairedRobustnessResult,
    evaluate_week9_paired_quantile_response,
)
from tcspc_toolkit.uncertainty_evaluation import (
    PredictionIntervalResult,
    UncertaintyDevelopmentSplit,
)


def _make_synthetic_generalization_prepared_data(
) -> GeneralizationPreparedData:
    development = (
        _make_synthetic_development_dataset(
            n_samples=40,
        )
    )

    n_samples = 6

    pair_ids = np.arange(
        n_samples,
        dtype=np.int64,
    )

    time = np.array(
        [
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )

    histograms = np.zeros(
        (
            n_samples,
            2,
        ),
        dtype=np.int64,
    )

    y = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            3.0,
        ],
        dtype=np.float64,
    )

    X_features = pd.DataFrame(
        {
            "feature_1": np.linspace(
                0.0,
                1.0,
                n_samples,
            ),
            "feature_2": np.linspace(
                0.0,
                1.0,
                n_samples,
            ) ** 2,
            "feature_3": np.sin(
                np.linspace(
                    0.0,
                    1.0,
                    n_samples,
                )
            ),
        }
    )

    test_a = GeneralizationTestMeasurements(
        test_id="A",
        time=time,
        X_histograms=histograms.copy(),
        y=y.copy(),
        metadata=pd.DataFrame(
            {
                "pair_id": pair_ids,
            }
        ),
    )

    test_b = GeneralizationTestMeasurements(
        test_id="B",
        time=time,
        X_histograms=histograms.copy(),
        y=y.copy(),
        metadata=pd.DataFrame(
            {
                "pair_id": pair_ids,
                "signal_photon_count_target": [
                    250,
                    500,
                    50_000,
                    250,
                    50_000,
                    500,
                ],
            }
        ),
    )

    test_d = GeneralizationTestMeasurements(
        test_id="D",
        time=time,
        X_histograms=histograms.copy(),
        y=y.copy(),
        metadata=pd.DataFrame(
            {
                "pair_id": pair_ids,
                "background_per_bin": np.full(
                    n_samples,
                    5.0,
                ),
            }
        ),
    )

    test_f = GeneralizationTestMeasurements(
        test_id="F",
        time=time,
        X_histograms=histograms.copy(),
        y=y.copy(),
        metadata=pd.DataFrame(
            {
                "pair_id": pair_ids,
                "primary_lifetime_ns": y.copy(),
                "secondary_fraction": [
                    0.05,
                    0.15,
                    0.05,
                    0.15,
                    0.05,
                    0.15,
                ],
                "model_mismatch_severity": [
                    "weak",
                    "moderate",
                    "weak",
                    "moderate",
                    "weak",
                    "moderate",
                ],
            }
        ),
    )

    return GeneralizationPreparedData(
        development=development,
        tests={
            "A": test_a,
            "B": test_b,
            "D": test_d,
            "F": test_f,
        },
        X_features={
            "A": X_features.copy(),
            "B": X_features.copy(),
            "D": X_features.copy(),
            "F": X_features.copy(),
        },
        X_normalized_development=np.zeros(
            (
                development.y.size,
                2,
            )
        ),
        X_normalized={
            "B": np.zeros(
                (
                    n_samples,
                    2,
                )
            ),
            "D": np.zeros(
                (
                    n_samples,
                    2,
                )
            ),
            "F": np.zeros(
                (
                    n_samples,
                    2,
                )
            ),
        },
        X_pca_development=np.zeros(
            (
                development.y.size,
                1,
            )
        ),
        X_pca={
            "B": np.zeros(
                (
                    n_samples,
                    1,
                )
            ),
            "D": np.zeros(
                (
                    n_samples,
                    1,
                )
            ),
            "F": np.zeros(
                (
                    n_samples,
                    1,
                )
            ),
        },
        pca=PCA(
            n_components=1,
        ),
    )


def _make_synthetic_development_dataset(
    n_samples: int = 64,
) -> BenchmarkDataset:
    x = np.linspace(
        0.0,
        1.0,
        n_samples,
        dtype=np.float64,
    )

    X_features = pd.DataFrame(
        {
            "feature_1": x,
            "feature_2": x**2,
            "feature_3": np.sin(x),
        }
    )

    y = (
        1.0
        + 3.0 * x
    )

    X_histograms = np.zeros(
        (
            n_samples,
            4,
        ),
        dtype=np.int64,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": np.arange(
                n_samples,
                dtype=np.int64,
            )
        }
    )

    return BenchmarkDataset(
        X_features=X_features,
        X_histograms=X_histograms,
        y=y,
        metadata=metadata,
    )


def test_quantile_interval_estimator_constructs_three_models() -> None:
    estimator = (
        QuantileGradientBoostingIntervalEstimator()
    )

    lower_model = estimator.lower_model.named_steps[
        "model"
    ]
    median_model = estimator.median_model.named_steps[
        "model"
    ]
    upper_model = estimator.upper_model.named_steps[
        "model"
    ]

    assert lower_model.loss == "quantile"
    assert median_model.loss == "quantile"
    assert upper_model.loss == "quantile"

    assert lower_model.quantile == pytest.approx(
        DEFAULT_LOWER_QUANTILE
    )
    assert median_model.quantile == pytest.approx(
        DEFAULT_MEDIAN_QUANTILE
    )
    assert upper_model.quantile == pytest.approx(
        DEFAULT_UPPER_QUANTILE
    )


def test_quantile_interval_estimator_fit_returns_self() -> None:
    X = np.linspace(
        0.0,
        1.0,
        60,
    ).reshape(-1, 1)

    y = (
        1.0
        + 2.0 * X[:, 0]
    )

    estimator = (
        QuantileGradientBoostingIntervalEstimator()
    )

    fitted = estimator.fit(
        X,
        y,
    )

    assert fitted is estimator


def test_quantile_interval_estimator_returns_prediction_interval() -> None:
    X = np.linspace(
        0.0,
        1.0,
        80,
    ).reshape(-1, 1)

    y = (
        1.0
        + 2.0 * X[:, 0]
    )

    estimator = (
        QuantileGradientBoostingIntervalEstimator()
    )

    estimator.fit(
        X,
        y,
    )

    intervals = estimator.predict(
        X
    )

    assert isinstance(
        intervals,
        PredictionIntervalResult,
    )

    assert intervals.prediction.shape == y.shape
    assert intervals.lower.shape == y.shape
    assert intervals.upper.shape == y.shape

    assert np.all(
        np.isfinite(intervals.prediction)
    )
    assert np.all(
        np.isfinite(intervals.lower)
    )
    assert np.all(
        np.isfinite(intervals.upper)
    )

    assert intervals.nominal_coverage == pytest.approx(
        DEFAULT_QUANTILE_NOMINAL_COVERAGE
    )

    assert (
        intervals.method_id
        == QUANTILE_GRADIENT_BOOSTING_METHOD_ID
    )


def test_quantile_interval_estimator_does_not_reorder_crossed_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = (
        QuantileGradientBoostingIntervalEstimator()
    )

    X = np.zeros(
        (2, 1),
        dtype=np.float64,
    )

    monkeypatch.setattr(
        estimator.lower_model,
        "predict",
        lambda X: np.array(
            [2.0, 1.0],
            dtype=np.float64,
        ),
    )

    monkeypatch.setattr(
        estimator.median_model,
        "predict",
        lambda X: np.array(
            [1.5, 1.5],
            dtype=np.float64,
        ),
    )

    monkeypatch.setattr(
        estimator.upper_model,
        "predict",
        lambda X: np.array(
            [1.0, 2.0],
            dtype=np.float64,
        ),
    )

    intervals = estimator.predict(
        X
    )

    assert intervals.lower[0] == pytest.approx(
        2.0
    )
    assert intervals.upper[0] == pytest.approx(
        1.0
    )

    assert not intervals.valid_interval_mask[0]
    assert intervals.valid_interval_mask[1]


def test_quantile_calibration_workflow_uses_held_out_split() -> None:
    development = (
        _make_synthetic_development_dataset()
    )

    result = (
        fit_and_evaluate_quantile_gradient_boosting(
            development
        )
    )

    assert len(
        result.split.training_indices
    ) == 48

    assert len(
        result.split.calibration_indices
    ) == 16

    assert set(
        result.split.training_indices
    ).isdisjoint(
        result.split.calibration_indices
    )

    assert (
        result.calibration_intervals.prediction.shape
        == (16,)
    )

    assert (
        result.interval_metrics.n_samples
        == 16
    )

    assert (
        result.quantile_metrics.n_samples
        == 16
    )

    assert result.feature_names == (
        "feature_1",
        "feature_2",
        "feature_3",
    )


def test_quantile_calibration_workflow_fits_only_training_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = (
        _make_synthetic_development_dataset(
            n_samples=20,
        )
    )

    captured: dict[
        str,
        np.ndarray,
    ] = {}

    def fake_fit(
        self: QuantileGradientBoostingIntervalEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> QuantileGradientBoostingIntervalEstimator:
        captured["X_fit"] = X.copy()
        captured["y_fit"] = y.copy()

        return self

    def fake_predict(
        self: QuantileGradientBoostingIntervalEstimator,
        X: np.ndarray,
    ) -> PredictionIntervalResult:
        captured[
            "X_predict"
        ] = X.copy()

        prediction = np.full(
            X.shape[0],
            2.0,
            dtype=np.float64,
        )

        return PredictionIntervalResult(
            prediction=prediction,
            lower=prediction - 0.5,
            upper=prediction + 0.5,
            nominal_coverage=0.90,
            method_id=(
                QUANTILE_GRADIENT_BOOSTING_METHOD_ID
            ),
        )

    monkeypatch.setattr(
        QuantileGradientBoostingIntervalEstimator,
        "fit",
        fake_fit,
    )

    monkeypatch.setattr(
        QuantileGradientBoostingIntervalEstimator,
        "predict",
        fake_predict,
    )

    result = (
        fit_and_evaluate_quantile_gradient_boosting(
            development
        )
    )

    training_indices = np.asarray(
        result.split.training_indices,
        dtype=np.int64,
    )

    calibration_indices = np.asarray(
        result.split.calibration_indices,
        dtype=np.int64,
    )

    X_all = (
        development.X_features.to_numpy(
            dtype=np.float64,
        )
    )

    y_all = np.asarray(
        development.y,
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        captured["X_fit"],
        X_all[
            training_indices
        ],
    )

    np.testing.assert_array_equal(
        captured["y_fit"],
        y_all[
            training_indices
        ],
    )

    np.testing.assert_array_equal(
        captured["X_predict"],
        X_all[
            calibration_indices
        ],
    )


def test_quantile_calibration_workflow_rejects_incomplete_split() -> None:
    development = (
        _make_synthetic_development_dataset(
            n_samples=10,
        )
    )

    split = UncertaintyDevelopmentSplit(
        training_indices=(
            0,
            1,
            2,
            3,
            4,
        ),
        calibration_indices=(
            5,
            6,
            7,
            8,
        ),
        random_seed=57,
        calibration_fraction=0.4,
    )

    with pytest.raises(
        ValueError,
        match="partition all development samples",
    ):
        fit_and_evaluate_quantile_gradient_boosting(
            development,
            split=split,
        )


def test_external_quantile_evaluation_uses_frozen_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = (
        _make_synthetic_development_dataset(
            n_samples=40,
        )
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            development
        )
    )

    external_features = (
        development.X_features.iloc[
            :6
        ].copy()
    )

    external_y = (
        development.y[
            :6
        ].copy()
    )

    predict_calls = 0

    def fake_predict(
        X: np.ndarray,
    ) -> PredictionIntervalResult:
        nonlocal predict_calls

        predict_calls += 1

        prediction = np.full(
            X.shape[0],
            2.0,
            dtype=np.float64,
        )

        return PredictionIntervalResult(
            prediction=prediction,
            lower=prediction - 0.5,
            upper=prediction + 0.5,
            nominal_coverage=0.90,
            method_id=(
                QUANTILE_GRADIENT_BOOSTING_METHOD_ID
            ),
        )

    monkeypatch.setattr(
        calibration_result.fitted_estimator,
        "predict",
        fake_predict,
    )

    result = (
        evaluate_frozen_quantile_gradient_boosting(
            calibration_result,
            external_features,
            external_y,
            condition_id="external_test",
        )
    )

    assert isinstance(
        result,
        QuantileGradientBoostingExternalResult,
    )

    assert predict_calls == 1

    assert result.condition_id == (
        "external_test"
    )

    assert (
        result.interval_metrics.n_samples
        == 6
    )

    assert (
        result.quantile_metrics.n_samples
        == 6
    )


def test_external_quantile_evaluation_reports_median_mae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = (
        _make_synthetic_development_dataset(
            n_samples=40,
        )
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            development
        )
    )

    X_external = (
        development.X_features.iloc[
            :3
        ].copy()
    )

    y_external = np.array(
        [
            1.0,
            2.0,
            3.0,
        ],
        dtype=np.float64,
    )

    def fake_predict(
        X: np.ndarray,
    ) -> PredictionIntervalResult:
        prediction = np.array(
            [
                1.1,
                1.8,
                3.3,
            ],
            dtype=np.float64,
        )

        return PredictionIntervalResult(
            prediction=prediction,
            lower=prediction - 0.5,
            upper=prediction + 0.5,
            nominal_coverage=0.90,
            method_id=(
                QUANTILE_GRADIENT_BOOSTING_METHOD_ID
            ),
        )

    monkeypatch.setattr(
        calibration_result.fitted_estimator,
        "predict",
        fake_predict,
    )

    result = (
        evaluate_frozen_quantile_gradient_boosting(
            calibration_result,
            X_external,
            y_external,
            condition_id="mae_test",
        )
    )

    expected_mae = (
        0.1
        + 0.2
        + 0.3
    ) / 3

    assert (
        result.median_prediction_mae_ns
        == pytest.approx(
            expected_mae
        )
    )


def test_external_quantile_evaluation_rejects_feature_schema_change() -> None:
    development = (
        _make_synthetic_development_dataset(
            n_samples=40,
        )
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            development
        )
    )

    X_external = (
        development.X_features.iloc[
            :4
        ].copy()
    )

    X_external = X_external[
        [
            "feature_3",
            "feature_2",
            "feature_1",
        ]
    ]

    with pytest.raises(
        ValueError,
        match="feature schema",
    ):
        evaluate_frozen_quantile_gradient_boosting(
            calibration_result,
            X_external,
            development.y[:4],
            condition_id="wrong_schema",
        )


def test_week9_quantile_robustness_uses_five_frozen_conditions() -> None:
    prepared = (
        _make_synthetic_generalization_prepared_data()
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            prepared.development
        )
    )

    result = (
        evaluate_week9_quantile_robustness_conditions(
            calibration_result,
            prepared,
            definition=(
                default_generalization_suite()
            ),
        )
    )

    assert isinstance(
        result,
        QuantileGradientBoostingRobustnessResult,
    )

    assert tuple(
        result.condition_results
    ) == (
        "B_low_photon",
        "B_high_photon",
        "D_high_background",
        "F_weak_mismatch",
        "F_moderate_mismatch",
    )

    assert (
        result.condition_results[
            "B_low_photon"
        ].interval_metrics.n_samples
        == 4
    )

    assert (
        result.condition_results[
            "B_high_photon"
        ].interval_metrics.n_samples
        == 2
    )

    assert (
        result.condition_results[
            "D_high_background"
        ].interval_metrics.n_samples
        == 6
    )

    assert (
        result.condition_results[
            "F_weak_mismatch"
        ].interval_metrics.n_samples
        == 3
    )

    assert (
        result.condition_results[
            "F_moderate_mismatch"
        ].interval_metrics.n_samples
        == 3
    )


def test_week9_quantile_robustness_summary_contains_response_ratios() -> None:
    prepared = (
        _make_synthetic_generalization_prepared_data()
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            prepared.development
        )
    )

    result = (
        evaluate_week9_quantile_robustness_conditions(
            calibration_result,
            prepared,
            definition=(
                default_generalization_suite()
            ),
        )
    )

    assert list(
        result.summary[
            "condition_id"
        ]
    ) == [
        "B_low_photon",
        "B_high_photon",
        "D_high_background",
        "F_weak_mismatch",
        "F_moderate_mismatch",
    ]

    required_columns = {
        "empirical_coverage",
        "mean_interval_width_ns",
        "median_interval_width_ns",
        "median_prediction_mae_ns",
        "quantile_crossing_rate",
        "mean_width_ratio_to_calibration",
        "median_width_ratio_to_calibration",
        "mae_ratio_to_calibration",
    }

    assert required_columns.issubset(
        result.summary.columns
    )


def test_week9_paired_quantile_response_uses_test_a_pair_ids() -> None:
    prepared = (
        _make_synthetic_generalization_prepared_data()
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            prepared.development
        )
    )

    result = (
        evaluate_week9_paired_quantile_response(
            calibration_result,
            prepared,
            definition=(
                default_generalization_suite()
            ),
        )
    )

    assert isinstance(
        result,
        QuantileGradientBoostingPairedRobustnessResult,
    )

    assert tuple(
        result.condition_results
    ) == (
        "B_low_photon",
        "B_high_photon",
        "D_high_background",
        "F_weak_mismatch",
        "F_moderate_mismatch",
    )

    low_photon = (
        result.condition_results[
            "B_low_photon"
        ].diagnostics
    )

    assert set(
        low_photon[
            "pair_id"
        ]
    ) == {
        0,
        1,
        3,
        5,
    }

    assert np.allclose(
        low_photon[
            "reference_absolute_error_ns"
        ],
        low_photon[
            "shifted_absolute_error_ns"
        ],
    )


def test_week9_paired_quantile_summary_reports_response_metrics() -> None:
    prepared = (
        _make_synthetic_generalization_prepared_data()
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            prepared.development
        )
    )

    result = (
        evaluate_week9_paired_quantile_response(
            calibration_result,
            prepared,
            definition=(
                default_generalization_suite()
            ),
        )
    )

    required_columns = {
        "condition_id",
        "reference_mae_ns",
        "shifted_mae_ns",
        "mae_change_ns",
        "reference_mean_interval_width_ns",
        "shifted_mean_interval_width_ns",
        "mean_interval_width_change_ns",
        "median_absolute_error_change_ns",
        "median_interval_width_change_ns",
        "spearman_width_error_change",
        "fraction_error_increased",
        "fraction_width_increased",
    }

    assert required_columns.issubset(
        result.summary.columns
    )

    assert len(
        result.summary
    ) == 5


def test_week9_paired_quantile_response_requires_pair_ids() -> None:
    prepared = (
        _make_synthetic_generalization_prepared_data()
    )

    prepared.tests[
        "A"
    ].metadata.drop(
        columns="pair_id",
        inplace=True,
    )

    calibration_result = (
        fit_and_evaluate_quantile_gradient_boosting(
            prepared.development
        )
    )

    with pytest.raises(
        ValueError,
        match="pair_id",
    ):
        evaluate_week9_paired_quantile_response(
            calibration_result,
            prepared,
            definition=(
                default_generalization_suite()
            ),
        )


