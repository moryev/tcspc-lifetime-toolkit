import numpy as np
import pytest

from dataclasses import replace

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.generalization import (
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    generate_generalization_test,
    GeneralizationTestMeasurements,
)
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)
from tcspc_toolkit.generalization_evaluation import (
    calculate_mae_degradation,
    calculate_robustness_metrics,
    DEFAULT_DEVELOPMENT_RANDOM_SEED,
    build_generalization_development_measurements,
    fit_generalization_ml_estimators,
    prepare_generalization_ab_data,
    build_mae_degradation_table,
    evaluate_principal_ab_benchmark,
    summarize_generalization_predictions,
    build_ab_comparison_table,
    evaluate_ml_representation_ab_benchmark,
    add_test_b_photon_count_regime,
    build_day53_ab_report,
    build_generalization_plot_diagnostics,
    summarize_test_b_photon_count_ood,
)


FEATURE_CONFIG = FeatureConfig(
    tail_start_ns=2.0,
    early_stop_ns=2.0,
    late_start_ns=3.0,
)


def test_calculate_robustness_metrics() -> None:
    y_true = np.asarray(
        [1.0, 2.0, 3.0, 4.0]
    )

    y_pred = np.asarray(
        [1.1, 1.8, 3.3, 3.6]
    )

    metrics = calculate_robustness_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    errors = y_pred - y_true
    absolute_errors = np.abs(errors)

    assert metrics.n_samples == 4

    assert metrics.mae_ns == pytest.approx(
        np.mean(absolute_errors)
    )

    assert (
        metrics.median_absolute_error_ns
        == pytest.approx(
            np.median(absolute_errors)
        )
    )

    assert metrics.rmse_ns == pytest.approx(
        np.sqrt(
            np.mean(
                errors**2
            )
        )
    )

    assert metrics.bias_ns == pytest.approx(
        np.mean(errors)
    )

    assert (
        metrics.p90_absolute_error_ns
        == pytest.approx(
            np.percentile(
                absolute_errors,
                90.0,
            )
        )
    )

    assert (
        metrics.p95_absolute_error_ns
        == pytest.approx(
            np.percentile(
                absolute_errors,
                95.0,
            )
        )
    )


def test_calculate_mae_degradation() -> None:
    reference_metrics = (
        calculate_robustness_metrics(
            y_true=[1.0, 2.0],
            y_pred=[1.1, 1.9],
        )
    )

    ood_metrics = (
        calculate_robustness_metrics(
            y_true=[1.0, 2.0],
            y_pred=[1.2, 1.8],
        )
    )

    degradation = (
        calculate_mae_degradation(
            reference_metrics=(
                reference_metrics
            ),
            ood_metrics=ood_metrics,
        )
    )

    assert (
        degradation.reference_test_id
        == "A"
    )

    assert (
        degradation.ood_test_id
        == "B"
    )

    assert (
        degradation.mae_degradation
        == pytest.approx(2.0)
    )


def test_mae_degradation_rejects_zero_reference_error() -> None:
    reference_metrics = (
        calculate_robustness_metrics(
            y_true=[1.0, 2.0],
            y_pred=[1.0, 2.0],
        )
    )

    ood_metrics = (
        calculate_robustness_metrics(
            y_true=[1.0, 2.0],
            y_pred=[1.1, 2.1],
        )
    )

    with pytest.raises(
        ValueError,
        match="Reference MAE",
    ):
        calculate_mae_degradation(
            reference_metrics=(
                reference_metrics
            ),
            ood_metrics=ood_metrics,
        )


def test_robustness_metrics_require_aligned_predictions() -> None:
    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        calculate_robustness_metrics(
            y_true=[1.0, 2.0],
            y_pred=[1.0],
        )


def test_development_measurements_use_familiar_domain() -> None:
    definition = (
        default_generalization_suite()
    )

    measurements = (
        build_generalization_development_measurements(
            definition=definition
        )
    )

    familiar = definition.familiar

    expected_n_samples = (
        len(familiar.lifetime_values_ns)
        * len(familiar.signal_photon_counts)
        * len(familiar.background_levels)
        * len(familiar.irf_fwhm_values_ns)
        * len(familiar.irf_shift_values_ns)
    )

    assert (
        measurements.y.size
        == expected_n_samples
    )

    assert set(
        measurements.y
    ) == set(
        familiar.lifetime_values_ns
    )

    assert set(
        measurements.metadata[
            "signal_photon_count_target"
        ]
    ) == set(
        familiar.signal_photon_counts
    )

    assert set(
        measurements.metadata[
            "background_per_bin"
        ]
    ) == set(
        familiar.background_levels
    )

    assert set(
        measurements.metadata[
            "irf_fwhm_ns"
        ]
    ) == set(
        familiar.irf_fwhm_values_ns
    )

    assert set(
        measurements.metadata[
            "irf_shift_ns"
        ]
    ) == set(
        familiar.irf_shift_values_ns
    )


def test_development_measurements_are_reproducible() -> None:
    first = (
        build_generalization_development_measurements()
    )

    second = (
        build_generalization_development_measurements()
    )

    np.testing.assert_array_equal(
        first.X_histograms,
        second.X_histograms,
    )

    np.testing.assert_array_equal(
        first.y,
        second.y,
    )


def test_development_seed_is_separate_from_final_test_seeds() -> None:
    definition = (
        default_generalization_suite()
    )

    assert (
        DEFAULT_DEVELOPMENT_RANDOM_SEED
        not in definition.numerics.test_seeds
    )


@pytest.fixture(scope="module")
def prepared_ab_data():
    definition = (
        default_generalization_suite()
    )

    development = (
        build_generalization_development_measurements(
            definition=definition
        )
    )

    test_a = generate_generalization_test(
        definition,
        "A",
    )

    test_b = generate_generalization_test(
        definition,
        "B",
    )

    return prepare_generalization_ab_data(
        development_measurements=development,
        test_a=test_a,
        test_b=test_b,
        feature_config=FEATURE_CONFIG,
    )


def test_ab_representation_shapes(
    prepared_ab_data,
) -> None:
    prepared = prepared_ab_data

    n_development = (
        prepared.development.y.size
    )

    n_a = prepared.test_a.y.size
    n_b = prepared.test_b.y.size

    assert (
        prepared.development.X_features.shape[0]
        == n_development
    )

    assert (
        prepared.X_normalized_development.shape[0]
        == n_development
    )

    assert (
        prepared.X_pca_development.shape[0]
        == n_development
    )

    assert (
        prepared.X_features_a.shape[0]
        == n_a
    )

    assert (
        prepared.X_normalized_a.shape[0]
        == n_a
    )

    assert (
        prepared.X_pca_a.shape[0]
        == n_a
    )

    assert (
        prepared.X_features_b.shape[0]
        == n_b
    )

    assert (
        prepared.X_normalized_b.shape[0]
        == n_b
    )

    assert (
        prepared.X_pca_b.shape[0]
        == n_b
    )


def test_pca_is_fitted_only_on_development_data(
    prepared_ab_data,
) -> None:
    prepared = prepared_ab_data

    assert (
        prepared.pca.n_samples_
        == prepared.development.y.size
    )

    np.testing.assert_allclose(
        prepared.pca.mean_,
        np.mean(
            prepared.X_normalized_development,
            axis=0,
        ),
    )


def test_fit_generalization_ml_estimators(
    prepared_ab_data,
) -> None:
    fitted = (
        fit_generalization_ml_estimators(
            prepared_ab_data
        )
    )

    assert set(fitted) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    expected_representations = {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    X_a_by_representation = {
        "engineered_features": (
            prepared_ab_data.X_features_a
        ),
        "normalized_histogram": (
            prepared_ab_data.X_normalized_a
        ),
        "pca_histogram": (
            prepared_ab_data.X_pca_a
        ),
    }

    for model_estimators in fitted.values():
        assert (
            set(model_estimators)
            == expected_representations
        )

        for (
            representation_name,
            estimator,
        ) in model_estimators.items():
            predictions = estimator.predict(
                X_a_by_representation[
                    representation_name
                ]
            )

            assert predictions.shape == (
                prepared_ab_data.test_a.y.shape
            )

            assert np.all(
                np.isfinite(predictions)
            )


@pytest.fixture(scope="module")
def small_prepared_ab_data(
    prepared_ab_data,
):
    prepared = prepared_ab_data

    lifetimes = np.unique(
        prepared.test_a.y
    )

    indices = np.asarray(
        [
            np.flatnonzero(
                prepared.test_a.y
                == lifetime
            )[0]
            for lifetime in lifetimes
        ],
        dtype=np.int64,
    )

    def subset_test(
        test: GeneralizationTestMeasurements,
    ) -> GeneralizationTestMeasurements:
        return GeneralizationTestMeasurements(
            test_id=test.test_id,
            time=test.time.copy(),
            X_histograms=(
                test.X_histograms[
                    indices
                ].copy()
            ),
            y=test.y[
                indices
            ].copy(),
            metadata=(
                test.metadata.iloc[
                    indices
                ]
                .reset_index(
                    drop=True
                )
            ),
        )

    return replace(
        prepared,
        test_a=subset_test(
            prepared.test_a
        ),
        test_b=subset_test(
            prepared.test_b
        ),
        X_features_a=(
            prepared.X_features_a.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        ),
        X_features_b=(
            prepared.X_features_b.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        ),
        X_normalized_a=(
            prepared.X_normalized_a[
                indices
            ].copy()
        ),
        X_normalized_b=(
            prepared.X_normalized_b[
                indices
            ].copy()
        ),
        X_pca_a=(
            prepared.X_pca_a[
                indices
            ].copy()
        ),
        X_pca_b=(
            prepared.X_pca_b[
                indices
            ].copy()
        ),
    )


@pytest.fixture(scope="module")
def nominal_irf(
    small_prepared_ab_data,
):
    time = (
        small_prepared_ab_data
        .test_a
        .time
    )

    raw_irf = generate_gaussian_irf(
        time=time,
        centre=1.0,
        fwhm=0.30,
    )

    return normalize_irf(
        time=time,
        irf=raw_irf,
    )


@pytest.fixture(scope="module")
def fitted_ab_estimators(
    prepared_ab_data,
):
    return fit_generalization_ml_estimators(
        prepared_ab_data
    )


def test_principal_ab_benchmark_contains_all_estimators(
    small_prepared_ab_data,
    fitted_ab_estimators,
    nominal_irf,
) -> None:
    result = (
        evaluate_principal_ab_benchmark(
            prepared=(
                small_prepared_ab_data
            ),
            fitted_estimators=(
                fitted_ab_estimators
            ),
            classical_irf=nominal_irf,
        )
    )

    expected_estimators = {
        "constant_mean",
        "mean_arrival_time",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
        "classical_reconvolution",
    }

    assert set(
        result.summary[
            "estimator"
        ]
    ) == expected_estimators

    assert set(
        result.summary[
            "test_id"
        ]
    ) == {
        "A",
        "B",
    }

    assert len(
        result.summary
    ) == 12

    assert len(
        result.degradation
    ) == 6


def test_principal_summary_contains_required_metrics(
    small_prepared_ab_data,
    fitted_ab_estimators,
    nominal_irf,
) -> None:
    result = (
        evaluate_principal_ab_benchmark(
            prepared=(
                small_prepared_ab_data
            ),
            fitted_estimators=(
                fitted_ab_estimators
            ),
            classical_irf=nominal_irf,
        )
    )

    required_columns = {
        "mae_ns",
        "median_absolute_error_ns",
        "rmse_ns",
        "bias_ns",
        "p90_absolute_error_ns",
        "p95_absolute_error_ns",
        "classical_failure_rate",
    }

    assert required_columns.issubset(
        result.summary.columns
    )


def test_constant_baseline_has_unit_mae_degradation(
    small_prepared_ab_data,
    fitted_ab_estimators,
    nominal_irf,
) -> None:
    result = (
        evaluate_principal_ab_benchmark(
            prepared=(
                small_prepared_ab_data
            ),
            fitted_estimators=(
                fitted_ab_estimators
            ),
            classical_irf=nominal_irf,
        )
    )

    row = result.degradation.loc[
        result.degradation[
            "estimator"
        ] == "constant_mean"
    ].iloc[0]

    assert row[
        "mae_degradation"
    ] == pytest.approx(
        1.0
    )


def test_mae_degradation_matches_absolute_metrics(
    small_prepared_ab_data,
    fitted_ab_estimators,
    nominal_irf,
) -> None:
    result = (
        evaluate_principal_ab_benchmark(
            prepared=(
                small_prepared_ab_data
            ),
            fitted_estimators=(
                fitted_ab_estimators
            ),
            classical_irf=nominal_irf,
        )
    )

    finite_rows = (
        result.degradation[
            np.isfinite(
                result.degradation[
                    "mae_degradation"
                ]
            )
        ]
    )

    for _, row in finite_rows.iterrows():
        assert row[
            "mae_degradation"
        ] == pytest.approx(
            row["mae_b_ns"]
            / row["mae_a_ns"]
        )


def test_classical_failure_rate_matches_reconvolution_result(
    small_prepared_ab_data,
    fitted_ab_estimators,
    nominal_irf,
) -> None:
    result = (
        evaluate_principal_ab_benchmark(
            prepared=(
                small_prepared_ab_data
            ),
            fitted_estimators=(
                fitted_ab_estimators
            ),
            classical_irf=nominal_irf,
        )
    )

    for test_id in (
        "A",
        "B",
    ):
        summary_row = (
            result.summary.loc[
                (
                    result.summary[
                        "estimator"
                    ]
                    == "classical_reconvolution"
                )
                & (
                    result.summary[
                        "test_id"
                    ]
                    == test_id
                )
            ]
            .iloc[0]
        )

        classical_result = (
            result.classical_results[
                test_id
            ]
        )

        assert summary_row[
            "classical_failure_rate"
        ] == pytest.approx(
            classical_result
            .summary
            .failure_rate
        )


def test_classical_failure_rate_is_na_for_nonclassical_estimators(
    small_prepared_ab_data,
    fitted_ab_estimators,
    nominal_irf,
) -> None:
    result = (
        evaluate_principal_ab_benchmark(
            prepared=(
                small_prepared_ab_data
            ),
            fitted_estimators=(
                fitted_ab_estimators
            ),
            classical_irf=nominal_irf,
        )
    )

    nonclassical = result.summary[
        result.summary[
            "estimator"
        ] != "classical_reconvolution"
    ]

    assert (
        nonclassical[
            "classical_failure_rate"
        ]
        .isna()
        .all()
    )


def test_representation_ab_benchmark_has_expected_structure(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    assert len(
        result.summary
    ) == 18

    assert len(
        result.degradation
    ) == 9

    assert len(
        result.comparison
    ) == 9

    assert set(
        result.summary[
            "estimator"
        ]
    ) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    assert set(
        result.summary[
            "representation"
        ]
    ) == {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    assert set(
        result.summary[
            "test_id"
        ]
    ) == {
        "A",
        "B",
    }


def test_every_ml_model_uses_all_representations(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    expected_representations = {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    for model_name in (
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    ):
        model_rows = (
            result.degradation.loc[
                result.degradation[
                    "estimator"
                ] == model_name
            ]
        )

        assert set(
            model_rows[
                "representation"
            ]
        ) == expected_representations


def test_every_representation_combination_has_a_and_b(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    grouped = (
        result.summary.groupby(
            [
                "estimator",
                "representation",
            ]
        )
    )

    for _, group in grouped:
        assert set(
            group["test_id"]
        ) == {
            "A",
            "B",
        }

        assert len(group) == 2


def test_representation_comparison_retains_absolute_metrics(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    required_columns = {
        "mae_a",
        "mae_b",
        "mae_degradation",
        "median_absolute_error_a",
        "median_absolute_error_b",
        "rmse_a",
        "rmse_b",
        "bias_a",
        "bias_b",
        "p90_absolute_error_a",
        "p90_absolute_error_b",
        "p95_absolute_error_a",
        "p95_absolute_error_b",
    }

    assert required_columns.issubset(
        result.comparison.columns
    )


def test_representation_degradation_matches_mae_ratio(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    for _, row in (
        result.comparison.iterrows()
    ):
        assert row[
            "mae_degradation"
        ] == pytest.approx(
            row["mae_b"]
            / row["mae_a"]
        )


def test_engineered_feature_results_use_existing_fitted_models(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    ridge = (
        fitted_ab_estimators[
            "ridge"
        ][
            "engineered_features"
        ]
    )

    expected_predictions = (
        ridge.predict(
            prepared_ab_data.X_features_a
        )
    )

    actual_predictions = (
        result.predictions.loc[
            (
                result.predictions[
                    "estimator"
                ] == "ridge"
            )
            & (
                result.predictions[
                    "representation"
                ]
                == "engineered_features"
            )
            & (
                result.predictions[
                    "test_id"
                ] == "A"
            ),
            "predicted_lifetime_ns",
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    np.testing.assert_allclose(
        actual_predictions,
        expected_predictions,
    )


def test_test_b_photon_count_regimes_are_low_or_high(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    definition = (
        default_generalization_suite()
    )

    familiar = definition.familiar

    test_b = result.predictions.loc[
        result.predictions[
            "test_id"
        ] == "B"
    ]

    labeled = add_test_b_photon_count_regime(
        test_b,
        familiar_min_photons=min(
            familiar.signal_photon_counts
        ),
        familiar_max_photons=max(
            familiar.signal_photon_counts
        ),
    )

    assert set(
        labeled[
            "photon_count_regime"
        ]
    ) == {
        "low_photon_ood",
        "high_photon_ood",
    }


def test_test_b_photon_count_summary_contains_both_ood_directions(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    familiar = (
        default_generalization_suite()
        .familiar
    )

    summary = (
        summarize_test_b_photon_count_ood(
            predictions=result.predictions,
            familiar_min_photons=min(
                familiar.signal_photon_counts
            ),
            familiar_max_photons=max(
                familiar.signal_photon_counts
            ),
        )
    )

    assert len(summary) == 18

    assert set(
        summary[
            "photon_count_regime"
        ]
    ) == {
        "low_photon_ood",
        "high_photon_ood",
    }


def test_generalization_plot_adapter_uses_benchmark_schema(
    prepared_ab_data,
    fitted_ab_estimators,
) -> None:
    result = (
        evaluate_ml_representation_ab_benchmark(
            prepared=prepared_ab_data,
            fitted_estimators=(
                fitted_ab_estimators
            ),
        )
    )

    diagnostics = (
        build_generalization_plot_diagnostics(
            result.predictions
        )
    )

    assert (
        "estimator_name"
        in diagnostics.columns
    )

    assert (
        "valid_estimate"
        in diagnostics.columns
    )

    assert (
        "estimator"
        not in diagnostics.columns
    )

    assert (
        "valid_prediction"
        not in diagnostics.columns
    )


