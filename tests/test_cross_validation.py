import numpy as np
import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression

from tcspc_toolkit.cross_validation import (
    DEFAULT_CV_N_REPEATS,
    DEFAULT_CV_N_SPLITS,
    DEFAULT_CV_RANDOM_STATE,
    RepeatedCVBenchmarkResult,
    RepeatedCVConfig,
    evaluate_estimators_repeated_cv,
    evaluate_regressor_repeated_cv,
    make_repeated_kfold,
    summarize_repeated_cv,
)
from tcspc_toolkit.ml_models import (
    make_hist_gradient_boosting_pipeline,
    make_pca_histogram_ridge_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)

def test_default_repeated_cv_config() -> None:
    config = RepeatedCVConfig()

    assert config.n_splits == DEFAULT_CV_N_SPLITS
    assert config.n_repeats == DEFAULT_CV_N_REPEATS
    assert config.random_state == DEFAULT_CV_RANDOM_STATE
    assert config.n_evaluations == 25


def test_make_repeated_kfold_produces_expected_number_of_splits(
) -> None:
    X = np.arange(100).reshape(50, 2)

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=5,
        random_state=52_001,
    )

    splitter = make_repeated_kfold(config)

    splits = list(splitter.split(X))

    assert len(splits) == 25


def test_repeated_kfold_is_reproducible() -> None:
    X = np.arange(100).reshape(50, 2)

    splitter_1 = make_repeated_kfold(
        RepeatedCVConfig(
            n_splits=5,
            n_repeats=3,
            random_state=123,
        )
    )

    splitter_2 = make_repeated_kfold(
        RepeatedCVConfig(
            n_splits=5,
            n_repeats=3,
            random_state=123,
        )
    )

    splits_1 = list(splitter_1.split(X))
    splits_2 = list(splitter_2.split(X))

    assert len(splits_1) == len(splits_2)

    for (
        train_indices_1,
        validation_indices_1,
    ), (
        train_indices_2,
        validation_indices_2,
    ) in zip(
        splits_1,
        splits_2,
        strict=True,
    ):
        assert np.array_equal(
            train_indices_1,
            train_indices_2,
        )

        assert np.array_equal(
            validation_indices_1,
            validation_indices_2,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "exception_type"),
    [
        ("n_splits", 1, ValueError),
        ("n_splits", 2.5, TypeError),
        ("n_repeats", 0, ValueError),
        ("n_repeats", 2.5, TypeError),
        ("random_state", -1, ValueError),
        ("random_state", 2.5, TypeError),
    ],
)
def test_repeated_cv_config_rejects_invalid_values(
    field_name: str,
    value: object,
    exception_type: type[Exception],
) -> None:
    kwargs = {
        field_name: value,
    }

    with pytest.raises(exception_type):
        RepeatedCVConfig(**kwargs)


def test_repeated_cv_returns_one_row_per_fold() -> None:
    X = np.column_stack(
        [
            np.linspace(
                0.0,
                1.0,
                50,
            ),
            np.linspace(
                1.0,
                2.0,
                50,
            ),
        ]
    )

    y = np.linspace(
        1.0,
        4.0,
        50,
    )

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=3,
        random_state=123,
    )

    results = evaluate_regressor_repeated_cv(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X=X,
        y=y,
        config=config,
    )

    assert results.shape[0] == 15

    assert list(results.columns) == [
        "model",
        "repeat",
        "fold",
        "n_train",
        "n_validation",
        "mae_ns",
        "median_absolute_error_ns",
        "rmse_ns",
        "bias_ns",
        "r2",
    ]

    assert set(results["model"]) == {
        "ridge"
    }

    assert set(results["repeat"]) == {
        1,
        2,
        3,
    }

    assert set(results["fold"]) == {
        1,
        2,
        3,
        4,
        5,
    }


def test_each_repeat_contains_all_folds() -> None:
    X = np.arange(
        120,
        dtype=np.float64,
    ).reshape(
        60,
        2,
    )

    y = np.linspace(
        1.0,
        4.0,
        60,
    )

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=4,
        random_state=123,
    )

    results = evaluate_regressor_repeated_cv(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X=X,
        y=y,
        config=config,
    )

    for repeat in range(
        1,
        config.n_repeats + 1,
    ):
        repeat_results = results[
            results["repeat"] == repeat
        ]

        assert set(
            repeat_results["fold"]
        ) == {
            1,
            2,
            3,
            4,
            5,
        }


def test_repeated_cv_metrics_are_finite() -> None:
    X = np.arange(
        120,
        dtype=np.float64,
    ).reshape(
        60,
        2,
    )

    y = np.linspace(
        1.0,
        4.0,
        60,
    )

    results = evaluate_regressor_repeated_cv(
        estimator_name="linear",
        estimator=LinearRegression(),
        X=X,
        y=y,
        config=RepeatedCVConfig(
            n_splits=5,
            n_repeats=2,
            random_state=123,
        ),
    )

    metric_columns = [
        "mae_ns",
        "median_absolute_error_ns",
        "rmse_ns",
        "bias_ns",
        "r2",
    ]

    assert np.all(
        np.isfinite(
            results[
                metric_columns
            ].to_numpy()
        )
    )


def test_repeated_cv_is_reproducible_for_fixed_seed() -> None:
    X = np.arange(
        120,
        dtype=np.float64,
    ).reshape(
        60,
        2,
    )

    y = np.linspace(
        1.0,
        4.0,
        60,
    )

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=3,
        random_state=123,
    )

    results_1 = evaluate_regressor_repeated_cv(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X=X,
        y=y,
        config=config,
    )

    results_2 = evaluate_regressor_repeated_cv(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X=X,
        y=y,
        config=config,
    )

    pd.testing.assert_frame_equal(
        results_1,
        results_2,
    )


def test_repeated_cv_does_not_fit_original_estimator() -> None:
    X = np.arange(
        120,
        dtype=np.float64,
    ).reshape(
        60,
        2,
    )

    y = np.linspace(
        1.0,
        4.0,
        60,
    )

    estimator = make_ridge_pipeline()

    evaluate_regressor_repeated_cv(
        estimator_name="ridge",
        estimator=estimator,
        X=X,
        y=y,
        config=RepeatedCVConfig(
            n_splits=5,
            n_repeats=2,
            random_state=123,
        ),
    )

    scaler = estimator.named_steps[
        "scaler"
    ]

    ridge = estimator.named_steps[
        "model"
    ]

    assert not hasattr(
        scaler,
        "mean_",
    )

    assert not hasattr(
        ridge,
        "coef_",
    )


def test_repeated_cv_rejects_mismatched_sample_counts() -> None:
    X = np.ones(
        (20, 3),
        dtype=np.float64,
    )

    y = np.ones(
        19,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "X and y must contain the same "
            "number of samples"
        ),
    ):
        evaluate_regressor_repeated_cv(
            estimator_name="ridge",
            estimator=make_ridge_pipeline(),
            X=X,
            y=y,
        )


def test_summarize_repeated_cv_computes_mean_and_std() -> None:
    cv_results = pd.DataFrame(
        {
            "model": [
                "ridge",
                "ridge",
                "ridge",
            ],
            "mae_ns": [
                0.10,
                0.20,
                0.30,
            ],
            "median_absolute_error_ns": [
                0.08,
                0.16,
                0.24,
            ],
            "rmse_ns": [
                0.15,
                0.25,
                0.35,
            ],
            "bias_ns": [
                -0.05,
                0.00,
                0.05,
            ],
            "r2": [
                0.90,
                0.92,
                0.94,
            ],
        }
    )

    summary = summarize_repeated_cv(
        cv_results
    )

    assert summary.shape[0] == 1

    row = summary.iloc[0]

    assert row["model"] == "ridge"
    assert row["n_evaluations"] == 3

    assert row["mean_mae_ns"] == pytest.approx(
        0.20
    )

    assert row["std_mae_ns"] == pytest.approx(
        0.10
    )

    assert row["mean_bias_ns"] == pytest.approx(
        0.0
    )

    assert row["mean_r2"] == pytest.approx(
        0.92
    )


def test_summarize_repeated_cv_groups_by_model() -> None:
    cv_results = pd.DataFrame(
        {
            "model": [
                "ridge",
                "ridge",
                "forest",
                "forest",
            ],
            "mae_ns": [
                0.10,
                0.20,
                0.05,
                0.07,
            ],
            "median_absolute_error_ns": [
                0.08,
                0.16,
                0.04,
                0.06,
            ],
            "rmse_ns": [
                0.15,
                0.25,
                0.08,
                0.10,
            ],
            "bias_ns": [
                -0.02,
                0.02,
                -0.01,
                0.01,
            ],
            "r2": [
                0.90,
                0.92,
                0.97,
                0.98,
            ],
        }
    )

    summary = summarize_repeated_cv(
        cv_results
    )

    assert list(summary["model"]) == [
        "ridge",
        "forest",
    ]

    ridge_row = summary[
        summary["model"] == "ridge"
    ].iloc[0]

    forest_row = summary[
        summary["model"] == "forest"
    ].iloc[0]

    assert ridge_row[
        "n_evaluations"
    ] == 2

    assert forest_row[
        "n_evaluations"
    ] == 2

    assert forest_row[
        "mean_mae_ns"
    ] < ridge_row[
        "mean_mae_ns"
    ]


def test_summarize_repeated_cv_accepts_evaluator_output() -> None:
    X = np.arange(
        120,
        dtype=np.float64,
    ).reshape(
        60,
        2,
    )

    y = np.linspace(
        1.0,
        4.0,
        60,
    )

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=3,
        random_state=123,
    )

    cv_results = evaluate_regressor_repeated_cv(
        estimator_name="ridge",
        estimator=make_ridge_pipeline(),
        X=X,
        y=y,
        config=config,
    )

    summary = summarize_repeated_cv(
        cv_results
    )

    assert summary.shape[0] == 1

    assert summary.iloc[0][
        "n_evaluations"
    ] == 15

    assert summary.iloc[0][
        "model"
    ] == "ridge"


def test_summarize_repeated_cv_rejects_empty_results() -> None:
    cv_results = pd.DataFrame(
        columns=[
            "model",
            "mae_ns",
            "median_absolute_error_ns",
            "rmse_ns",
            "bias_ns",
            "r2",
        ]
    )

    with pytest.raises(
        ValueError,
        match="must contain at least one CV result",
    ):
        summarize_repeated_cv(
            cv_results
        )


def test_summarize_repeated_cv_rejects_missing_columns() -> None:
    cv_results = pd.DataFrame(
        {
            "model": [
                "ridge",
            ],
            "mae_ns": [
                0.1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        summarize_repeated_cv(
            cv_results
        )


def test_summarize_repeated_cv_requires_two_evaluations_per_model(
) -> None:
    cv_results = pd.DataFrame(
        {
            "model": [
                "ridge",
            ],
            "mae_ns": [
                0.1,
            ],
            "median_absolute_error_ns": [
                0.08,
            ],
            "rmse_ns": [
                0.15,
            ],
            "bias_ns": [
                0.01,
            ],
            "r2": [
                0.95,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two CV evaluations",
    ):
        summarize_repeated_cv(
            cv_results
        )


def test_repeated_cv_keeps_original_pca_pipeline_unfitted(
) -> None:
    rng = np.random.default_rng(
        123
    )

    X = rng.poisson(
        lam=20.0,
        size=(
            60,
            20,
        ),
    ).astype(
        np.float64
    )

    y = np.linspace(
        1.0,
        4.0,
        60,
    )

    estimator = (
        make_pca_histogram_ridge_pipeline(
            n_components=5,
        )
    )

    results = evaluate_regressor_repeated_cv(
        estimator_name="ridge_pca_histogram",
        estimator=estimator,
        X=X,
        y=y,
        config=RepeatedCVConfig(
            n_splits=5,
            n_repeats=2,
            random_state=123,
        ),
    )

    assert results.shape[0] == 10

    pca = estimator.named_steps[
        "pca"
    ]

    scaler = estimator.named_steps[
        "scaler"
    ]

    model = estimator.named_steps[
        "model"
    ]

    assert not hasattr(
        pca,
        "components_",
    )

    assert not hasattr(
        scaler,
        "mean_",
    )

    assert not hasattr(
        model,
        "coef_",
    )


def test_multiple_estimators_return_combined_cv_results(
) -> None:
    rng = np.random.default_rng(
        123
    )

    X = rng.normal(
        size=(60, 5)
    )

    y = (
        2.5
        + 0.4 * X[:, 0]
        - 0.2 * X[:, 1]
    )

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=2,
        random_state=123,
    )

    result = evaluate_estimators_repeated_cv(
        estimators={
            "ridge": make_ridge_pipeline(),
            "random_forest": (
                make_random_forest_pipeline(
                    random_state=123,
                )
            ),
        },
        X=X,
        y=y,
        config=config,
    )

    assert isinstance(
        result,
        RepeatedCVBenchmarkResult,
    )

    assert result.fold_results.shape[0] == 20

    assert set(
        result.fold_results["model"]
    ) == {
        "ridge",
        "random_forest",
    }

    assert result.summary.shape[0] == 2


def test_multiple_estimators_share_cv_grid() -> None:
    rng = np.random.default_rng(
        123
    )

    X = rng.normal(
        size=(50, 4)
    )

    y = (
        2.0
        + 0.3 * X[:, 0]
    )

    config = RepeatedCVConfig(
        n_splits=5,
        n_repeats=3,
        random_state=456,
    )

    result = evaluate_estimators_repeated_cv(
        estimators={
            "ridge": make_ridge_pipeline(),
            "random_forest": (
                make_random_forest_pipeline(
                    random_state=123,
                )
            ),
        },
        X=X,
        y=y,
        config=config,
    )

    ridge_grid = (
        result.fold_results[
            result.fold_results["model"]
            == "ridge"
        ][
            [
                "repeat",
                "fold",
                "n_train",
                "n_validation",
            ]
        ]
        .reset_index(drop=True)
    )

    forest_grid = (
        result.fold_results[
            result.fold_results["model"]
            == "random_forest"
        ][
            [
                "repeat",
                "fold",
                "n_train",
                "n_validation",
            ]
        ]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        ridge_grid,
        forest_grid,
    )


def test_canonical_ml_models_run_through_repeated_cv(
) -> None:
    rng = np.random.default_rng(
        123
    )

    X = rng.normal(
        size=(60, 6)
    )

    y = (
        2.5
        + 0.5 * X[:, 0]
        - 0.25 * X[:, 1]
    )

    config = RepeatedCVConfig(
        n_splits=2,
        n_repeats=1,
        random_state=123,
    )

    result = evaluate_estimators_repeated_cv(
        estimators={
            "ridge": (
                make_ridge_pipeline()
            ),
            "random_forest": (
                make_random_forest_pipeline(
                    random_state=123,
                )
            ),
            "hist_gradient_boosting": (
                make_hist_gradient_boosting_pipeline(
                    random_state=123,
                )
            ),
        },
        X=X,
        y=y,
        config=config,
    )

    assert set(
        result.summary["model"]
    ) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    assert np.all(
        result.summary[
            "n_evaluations"
        ].to_numpy()
        == 2
    )

    metric_columns = [
        "mean_mae_ns",
        "std_mae_ns",
        "mean_rmse_ns",
        "std_rmse_ns",
        "mean_bias_ns",
        "std_bias_ns",
        "mean_r2",
        "std_r2",
    ]

    assert np.all(
        np.isfinite(
            result.summary[
                metric_columns
            ].to_numpy()
        )
    )


def test_multiple_estimator_cv_rejects_empty_mapping(
) -> None:
    X = np.ones(
        (20, 3),
        dtype=np.float64,
    )

    y = np.linspace(
        1.0,
        4.0,
        20,
    )

    with pytest.raises(
        ValueError,
        match="at least one estimator",
    ):
        evaluate_estimators_repeated_cv(
            estimators={},
            X=X,
            y=y,
        )


