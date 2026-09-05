"""Evaluation utilities for TCSPC generalization tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from collections.abc import Callable
from typing import Any

from sklearn.decomposition import PCA

from tcspc_toolkit.baselines import (
    estimate_lifetime_from_mean_arrival,
    predict_constant_mean_baseline,
)
from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult,
    evaluate_reconvolution_benchmark,
)
from tcspc_toolkit.config import (
    CountNormalization,
    FeatureConfig,
)
from tcspc_toolkit.features import (
    extract_feature_table,
)
from tcspc_toolkit.generalization import (
    GeneralizationSuiteDefinition,
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    GeneralizationTestMeasurements,
    build_generalization_time_axis,
)
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)
from tcspc_toolkit.ml_evaluation import (
    BenchmarkDataset,
    BenchmarkMeasurements,
    build_benchmark_dataset,
    generate_benchmark_measurements,
)
from tcspc_toolkit.ml_models import (
    DEFAULT_PCA_COMPONENTS,
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)
from tcspc_toolkit.representations import (
    fit_pca_representation,
    normalize_histogram_batch,
    transform_pca_representation,
)


DEFAULT_DEVELOPMENT_RANDOM_SEED = 42


@dataclass(frozen=True)
class RobustnessMetrics:
    """Aggregate regression metrics for one robustness test."""

    n_samples: int

    mae_ns: float
    median_absolute_error_ns: float
    rmse_ns: float
    bias_ns: float

    p90_absolute_error_ns: float
    p95_absolute_error_ns: float


@dataclass(frozen=True)
class RobustnessDegradation:
    """Relative degradation from familiar to OOD performance."""

    reference_test_id: str
    ood_test_id: str

    reference_mae_ns: float
    ood_mae_ns: float

    mae_degradation: float


def calculate_robustness_metrics(
    *,
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> RobustnessMetrics:
    """Calculate aggregate metrics for one robustness evaluation."""

    y_true_array = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true_array.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if y_pred_array.ndim != 1:
        raise ValueError(
            "y_pred must be one-dimensional."
        )

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true_array.size == 0:
        raise ValueError(
            "y_true and y_pred must not be empty."
        )

    if not np.all(
        np.isfinite(y_true_array)
    ):
        raise ValueError(
            "y_true must contain only finite values."
        )

    if not np.all(
        np.isfinite(y_pred_array)
    ):
        raise ValueError(
            "y_pred must contain only finite values."
        )

    errors = (
        y_pred_array
        - y_true_array
    )

    absolute_errors = np.abs(
        errors
    )

    return RobustnessMetrics(
        n_samples=int(
            y_true_array.size
        ),
        mae_ns=float(
            np.mean(
                absolute_errors
            )
        ),
        median_absolute_error_ns=float(
            np.median(
                absolute_errors
            )
        ),
        rmse_ns=float(
            np.sqrt(
                np.mean(
                    errors**2
                )
            )
        ),
        bias_ns=float(
            np.mean(
                errors
            )
        ),
        p90_absolute_error_ns=float(
            np.percentile(
                absolute_errors,
                90.0,
            )
        ),
        p95_absolute_error_ns=float(
            np.percentile(
                absolute_errors,
                95.0,
            )
        ),
    )


def calculate_mae_degradation(
    *,
    reference_metrics: RobustnessMetrics,
    ood_metrics: RobustnessMetrics,
    reference_test_id: str = "A",
    ood_test_id: str = "B",
) -> RobustnessDegradation:
    """Calculate MAE degradation between reference and OOD tests."""

    if reference_metrics.mae_ns <= 0.0:
        raise ValueError(
            "Reference MAE must be strictly positive "
            "to calculate degradation."
        )

    mae_degradation = (
        ood_metrics.mae_ns
        / reference_metrics.mae_ns
    )

    return RobustnessDegradation(
        reference_test_id=(
            reference_test_id
        ),
        ood_test_id=(
            ood_test_id
        ),
        reference_mae_ns=(
            reference_metrics.mae_ns
        ),
        ood_mae_ns=(
            ood_metrics.mae_ns
        ),
        mae_degradation=float(
            mae_degradation
        ),
    )


def build_generalization_development_measurements(
    *,
    definition: GeneralizationSuiteDefinition | None = None,
    random_seed: int = DEFAULT_DEVELOPMENT_RANDOM_SEED,
) -> BenchmarkMeasurements:
    """Generate the compact development pool for Week 8.

    The development measurements are drawn exclusively from the
    familiar physical domain frozen by the Week 8 generalization
    protocol.

    Final Tests A-F are not used here.
    """

    if definition is None:
        definition = default_generalization_suite()

    if random_seed in definition.numerics.test_seeds:
        raise ValueError(
            "Development random seed must differ from "
            "all final A-F test seeds."
        )

    familiar = definition.familiar

    time = build_generalization_time_axis(
        definition
    )

    return generate_benchmark_measurements(
        time=time,
        lifetimes_ns=(
            familiar.lifetime_values_ns
        ),
        signal_photon_counts=(
            familiar.signal_photon_counts
        ),
        background_levels=(
            familiar.background_levels
        ),
        irf_centre_ns=(
            familiar.irf_centre_ns
        ),
        irf_fwhm_values_ns=(
            familiar.irf_fwhm_values_ns
        ),
        irf_shift_values_ns=(
            familiar.irf_shift_values_ns
        ),
        random_seed=random_seed,
    )


@dataclass(frozen=True)
class GeneralizationABPreparedData:
    """Development and final A/B representations.

    All fitted representation parameters, especially PCA,
    are estimated exclusively from development data.
    """

    development: BenchmarkDataset

    test_a: GeneralizationTestMeasurements
    test_b: GeneralizationTestMeasurements

    X_features_a: pd.DataFrame
    X_features_b: pd.DataFrame

    X_normalized_development: np.ndarray
    X_normalized_a: np.ndarray
    X_normalized_b: np.ndarray

    X_pca_development: np.ndarray
    X_pca_a: np.ndarray
    X_pca_b: np.ndarray

    pca: PCA


@dataclass(frozen=True)
class GeneralizationPreparedData:
    """Development data and final robustness-test representations.

    All fitted representation parameters are estimated exclusively
    from development data.

    Final robustness Tests A-F are transformed only after all
    development fitting is complete.
    """

    development: BenchmarkDataset

    tests: dict[
        str,
        GeneralizationTestMeasurements,
    ]

    X_features: dict[
        str,
        pd.DataFrame,
    ]

    X_normalized_development: np.ndarray

    X_normalized: dict[
        str,
        np.ndarray,
    ]

    X_pca_development: np.ndarray

    X_pca: dict[
        str,
        np.ndarray,
    ]

    pca: PCA


def prepare_generalization_ab_data(
    *,
    development_measurements: BenchmarkMeasurements,
    test_a: GeneralizationTestMeasurements,
    test_b: GeneralizationTestMeasurements,
    feature_config: FeatureConfig,
    n_pca_components: int = DEFAULT_PCA_COMPONENTS,
) -> GeneralizationABPreparedData:
    """Prepare development, Test-A, and Test-B representations.

    Engineered features are extracted independently from every
    histogram.

    TOTAL normalization is sample-wise and therefore stateless.

    PCA is fitted exclusively on normalized development histograms.
    Tests A and B are transformed using that already fitted PCA.
    """

    if test_a.test_id != "A":
        raise ValueError(
            "test_a must contain robustness Test A."
        )

    if test_b.test_id != "B":
        raise ValueError(
            "test_b must contain robustness Test B."
        )

    if not np.array_equal(
        development_measurements.time,
        test_a.time,
    ):
        raise ValueError(
            "Development data and Test A must share "
            "the same time axis."
        )

    if not np.array_equal(
        development_measurements.time,
        test_b.time,
    ):
        raise ValueError(
            "Development data and Test B must share "
            "the same time axis."
        )

    if not np.array_equal(
        test_a.y,
        test_b.y,
    ):
        raise ValueError(
            "Tests A and B must preserve paired "
            "lifetime targets."
        )

    development = build_benchmark_dataset(
        development_measurements,
        feature_config=feature_config,
    )

    X_features_a = extract_feature_table(
        histograms=test_a.X_histograms,
        time=test_a.time,
        config=feature_config,
    )

    X_features_b = extract_feature_table(
        histograms=test_b.X_histograms,
        time=test_b.time,
        config=feature_config,
    )

    if tuple(
        development.X_features.columns
    ) != tuple(
        X_features_a.columns
    ):
        raise RuntimeError(
            "Development and Test-A engineered "
            "feature schemas do not match."
        )

    if tuple(
        development.X_features.columns
    ) != tuple(
        X_features_b.columns
    ):
        raise RuntimeError(
            "Development and Test-B engineered "
            "feature schemas do not match."
        )

    X_normalized_development = (
        normalize_histogram_batch(
            histograms=(
                development.X_histograms
            ),
            mode=CountNormalization.TOTAL,
        )
    )

    X_normalized_a = (
        normalize_histogram_batch(
            histograms=test_a.X_histograms,
            mode=CountNormalization.TOTAL,
        )
    )

    X_normalized_b = (
        normalize_histogram_batch(
            histograms=test_b.X_histograms,
            mode=CountNormalization.TOTAL,
        )
    )

    pca = fit_pca_representation(
        X_train=X_normalized_development,
        n_components=n_pca_components,
    )

    X_pca_development = (
        transform_pca_representation(
            pca=pca,
            X=X_normalized_development,
        )
    )

    X_pca_a = (
        transform_pca_representation(
            pca=pca,
            X=X_normalized_a,
        )
    )

    X_pca_b = (
        transform_pca_representation(
            pca=pca,
            X=X_normalized_b,
        )
    )

    return GeneralizationABPreparedData(
        development=development,
        test_a=test_a,
        test_b=test_b,
        X_features_a=X_features_a,
        X_features_b=X_features_b,
        X_normalized_development=(
            X_normalized_development
        ),
        X_normalized_a=X_normalized_a,
        X_normalized_b=X_normalized_b,
        X_pca_development=(
            X_pca_development
        ),
        X_pca_a=X_pca_a,
        X_pca_b=X_pca_b,
        pca=pca,
    )


def prepare_generalization_data(
    *,
    development_measurements: BenchmarkMeasurements,
    tests: dict[
        str,
        GeneralizationTestMeasurements,
    ],
    feature_config: FeatureConfig,
    n_pca_components: int = DEFAULT_PCA_COMPONENTS,
) -> GeneralizationPreparedData:
    """Prepare representations for final robustness tests.

    PCA is fitted exclusively on normalized development
    histograms. Final test sets are never used for fitting.
    """

    if not tests:
        raise ValueError(
            "tests must contain at least one "
            "generalization test."
        )

    normalized_test_ids = {
        test_id.strip().upper()
        for test_id in tests
    }

    if len(normalized_test_ids) != len(tests):
        raise ValueError(
            "Generalization test IDs must be unique."
        )

    development = build_benchmark_dataset(
        development_measurements,
        feature_config=feature_config,
    )

    X_features: dict[
        str,
        pd.DataFrame,
    ] = {}

    X_normalized: dict[
        str,
        np.ndarray,
    ] = {}

    normalized_tests: dict[
        str,
        GeneralizationTestMeasurements,
    ] = {}

    for test_id, test in tests.items():
        normalized_id = (
            test_id.strip().upper()
        )

        if test.test_id != normalized_id:
            raise ValueError(
                "Dictionary key and test.test_id "
                "must match."
            )

        if not np.array_equal(
            development_measurements.time,
            test.time,
        ):
            raise ValueError(
                "Development data and all final "
                "tests must share the same time axis."
            )

        features = extract_feature_table(
            histograms=test.X_histograms,
            time=test.time,
            config=feature_config,
        )

        if tuple(
            development.X_features.columns
        ) != tuple(
            features.columns
        ):
            raise RuntimeError(
                "Development and final-test "
                "feature schemas do not match."
            )

        X_features[normalized_id] = (
            features
        )

        X_normalized[normalized_id] = (
            normalize_histogram_batch(
                histograms=(
                    test.X_histograms
                ),
                mode=(
                    CountNormalization.TOTAL
                ),
            )
        )

        normalized_tests[
            normalized_id
        ] = test

    X_normalized_development = (
        normalize_histogram_batch(
            histograms=(
                development.X_histograms
            ),
            mode=CountNormalization.TOTAL,
        )
    )

    pca = fit_pca_representation(
        X_train=(
            X_normalized_development
        ),
        n_components=n_pca_components,
    )

    X_pca_development = (
        transform_pca_representation(
            pca=pca,
            X=X_normalized_development,
        )
    )

    X_pca = {
        test_id: (
            transform_pca_representation(
                pca=pca,
                X=X_test,
            )
        )
        for test_id, X_test
        in X_normalized.items()
    }

    return GeneralizationPreparedData(
        development=development,
        tests=normalized_tests,
        X_features=X_features,
        X_normalized_development=(
            X_normalized_development
        ),
        X_normalized=X_normalized,
        X_pca_development=(
            X_pca_development
        ),
        X_pca=X_pca,
        pca=pca,
    )


def fit_generalization_ml_estimators(
    prepared: GeneralizationPreparedData,
) -> dict[str, dict[str, Any]]:
    """Fit Week-8 ML estimators exclusively on development data."""

    X_development_by_representation = {
        "engineered_features": (
            prepared.development.X_features
        ),
        "normalized_histogram": (
            prepared.X_normalized_development
        ),
        "pca_histogram": (
            prepared.X_pca_development
        ),
    }

    model_factories: dict[
        str,
        Callable[[], Any],
    ] = {
        "ridge": make_ridge_pipeline,
        "random_forest": (
            make_random_forest_pipeline
        ),
        "hist_gradient_boosting": (
            make_hist_gradient_boosting_pipeline
        ),
    }

    fitted_estimators: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        model_name,
        make_estimator,
    ) in model_factories.items():

        representation_estimators: dict[
            str,
            Any,
        ] = {}

        for (
            representation_name,
            X_development,
        ) in (
            X_development_by_representation.items()
        ):
            estimator = make_estimator()

            estimator.fit(
                X_development,
                prepared.development.y,
            )

            representation_estimators[
                representation_name
            ] = estimator

        fitted_estimators[
            model_name
        ] = representation_estimators

    return fitted_estimators


@dataclass(frozen=True)
class PrincipalABBenchmarkResult:
    """Principal-estimator comparison between Tests A and B.

    Attributes
    ----------
    predictions:
        Long-form per-sample predictions and errors for every
        principal estimator on Tests A and B.
    summary:
        Aggregate absolute-performance metrics for every
        estimator and test.
    degradation:
        Test-B versus Test-A MAE degradation for every estimator.
    classical_results:
        Original failure-aware reconvolution benchmark results
        for Tests A and B.
    """

    predictions: pd.DataFrame
    summary: pd.DataFrame
    degradation: pd.DataFrame

    classical_results: dict[
        str,
        ReconvolutionBenchmarkResult,
    ]


def _build_generalization_prediction_table(
    *,
    estimator_name: str,
    representation_name: str,
    test: GeneralizationTestMeasurements,
    y_pred: ArrayLike,
    valid_mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Build aligned per-sample robustness diagnostics."""

    y_pred_array = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_pred_array.shape != test.y.shape:
        raise ValueError(
            "Predictions must contain exactly one "
            "value per robustness-test sample."
        )

    if valid_mask is None:
        valid_array = np.ones(
            test.y.size,
            dtype=bool,
        )

    else:
        valid_array = np.asarray(
            valid_mask,
            dtype=bool,
        )

        if valid_array.shape != test.y.shape:
            raise ValueError(
                "valid_mask must have the same shape "
                "as the lifetime targets."
            )

    if np.any(
        valid_array
        & ~np.isfinite(y_pred_array)
    ):
        raise ValueError(
            "All valid predictions must be finite."
        )

    errors = np.full(
        test.y.size,
        np.nan,
        dtype=np.float64,
    )

    errors[
        valid_array
    ] = (
        y_pred_array[valid_array]
        - test.y[valid_array]
    )

    absolute_errors = np.abs(
        errors
    )

    diagnostics = test.metadata.copy(
        deep=True
    ).reset_index(
        drop=True
    )

    diagnostics[
        "estimator"
    ] = estimator_name

    diagnostics[
        "representation"
    ] = representation_name

    diagnostics[
        "true_lifetime_ns"
    ] = test.y

    diagnostics[
        "predicted_lifetime_ns"
    ] = y_pred_array

    diagnostics[
        "valid_prediction"
    ] = valid_array

    diagnostics[
        "error_ns"
    ] = errors

    diagnostics[
        "absolute_error_ns"
    ] = absolute_errors

    return diagnostics


def summarize_generalization_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize robustness predictions by estimator and test."""

    required_columns = {
        "estimator",
        "representation",
        "test_id",
        "true_lifetime_ns",
        "predicted_lifetime_ns",
        "valid_prediction",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    rows: list[
        dict[str, str | int | float]
    ] = []

    grouped = predictions.groupby(
        [
            "estimator",
            "representation",
            "test_id",
        ],
        sort=False,
        dropna=False,
    )

    for (
        estimator_name,
        representation_name,
        test_id,
    ), group in grouped:

        valid_mask = group[
            "valid_prediction"
        ].to_numpy(
            dtype=bool
        )

        n_total = int(
            len(group)
        )

        n_valid = int(
            np.sum(valid_mask)
        )

        if n_valid > 0:
            metrics = (
                calculate_robustness_metrics(
                    y_true=(
                        group.loc[
                            valid_mask,
                            "true_lifetime_ns",
                        ]
                    ),
                    y_pred=(
                        group.loc[
                            valid_mask,
                            "predicted_lifetime_ns",
                        ]
                    ),
                )
            )

            mae_ns = metrics.mae_ns

            median_absolute_error_ns = (
                metrics.median_absolute_error_ns
            )

            rmse_ns = metrics.rmse_ns
            bias_ns = metrics.bias_ns

            p90_absolute_error_ns = (
                metrics.p90_absolute_error_ns
            )

            p95_absolute_error_ns = (
                metrics.p95_absolute_error_ns
            )

        else:
            mae_ns = np.nan

            median_absolute_error_ns = (
                np.nan
            )

            rmse_ns = np.nan
            bias_ns = np.nan

            p90_absolute_error_ns = np.nan
            p95_absolute_error_ns = np.nan

        if str(
                estimator_name
        ).startswith(
            "classical_reconvolution"
        ):
            classical_failure_rate = (
                1.0
                - n_valid / n_total
            )

        else:
            classical_failure_rate = np.nan

        rows.append(
            {
                "estimator": (
                    estimator_name
                ),
                "representation": (
                    representation_name
                ),
                "test_id": test_id,
                "n_total_samples": n_total,
                "n_valid_predictions": n_valid,
                "mae_ns": mae_ns,
                "median_absolute_error_ns": (
                    median_absolute_error_ns
                ),
                "rmse_ns": rmse_ns,
                "bias_ns": bias_ns,
                "p90_absolute_error_ns": (
                    p90_absolute_error_ns
                ),
                "p95_absolute_error_ns": (
                    p95_absolute_error_ns
                ),
                "classical_failure_rate": (
                    classical_failure_rate
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_mae_degradation_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare Test-B MAE against familiar Test-A MAE."""

    required_columns = {
        "estimator",
        "representation",
        "test_id",
        "mae_ns",
    }

    missing_columns = (
        required_columns
        - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Summary table is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    rows: list[
        dict[str, str | float]
    ] = []

    grouped = summary.groupby(
        [
            "estimator",
            "representation",
        ],
        sort=False,
        dropna=False,
    )

    for (
        estimator_name,
        representation_name,
    ), group in grouped:

        test_ids = set(
            group["test_id"]
        )

        if test_ids != {
            "A",
            "B",
        }:
            raise ValueError(
                "Every estimator must contain exactly "
                "Tests A and B."
            )

        reference_mae_ns = float(
            group.loc[
                group["test_id"] == "A",
                "mae_ns",
            ].iloc[0]
        )

        ood_mae_ns = float(
            group.loc[
                group["test_id"] == "B",
                "mae_ns",
            ].iloc[0]
        )

        if (
            np.isfinite(reference_mae_ns)
            and reference_mae_ns > 0.0
            and np.isfinite(ood_mae_ns)
        ):
            mae_degradation = (
                ood_mae_ns
                / reference_mae_ns
            )

        else:
            mae_degradation = np.nan

        rows.append(
            {
                "estimator": (
                    estimator_name
                ),
                "representation": (
                    representation_name
                ),
                "mae_a_ns": (
                    reference_mae_ns
                ),
                "mae_b_ns": (
                    ood_mae_ns
                ),
                "mae_degradation": float(
                    mae_degradation
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_reference_mae_degradation_table(
    summary: pd.DataFrame,
    *,
    reference_test_id: str = "A",
) -> pd.DataFrame:
    """Compare every OOD test against one reference test.

    One degradation row is produced for every
    estimator/representation/OOD-test combination.
    """

    required_columns = {
        "estimator",
        "representation",
        "test_id",
        "mae_ns",
    }

    missing_columns = (
        required_columns
        - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Summary table is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    reference_test_id = (
        reference_test_id
        .strip()
        .upper()
    )

    if not reference_test_id:
        raise ValueError(
            "reference_test_id must not be empty."
        )

    rows: list[
        dict[str, str | float]
    ] = []

    grouped = summary.groupby(
        [
            "estimator",
            "representation",
        ],
        sort=False,
        dropna=False,
    )

    for (
        estimator_name,
        representation_name,
    ), group in grouped:

        reference_rows = group.loc[
            group["test_id"]
            == reference_test_id
        ]

        if len(reference_rows) != 1:
            raise ValueError(
                "Every estimator/representation "
                "combination must contain exactly one "
                f"reference Test {reference_test_id} row."
            )

        reference_mae_ns = float(
            reference_rows[
                "mae_ns"
            ].iloc[0]
        )

        ood_rows = group.loc[
            group["test_id"]
            != reference_test_id
        ]

        if ood_rows.empty:
            raise ValueError(
                "Every estimator/representation "
                "combination must contain at least "
                "one OOD test."
            )

        for _, ood_row in (
            ood_rows.iterrows()
        ):
            ood_test_id = str(
                ood_row["test_id"]
            )

            ood_mae_ns = float(
                ood_row["mae_ns"]
            )

            if (
                np.isfinite(
                    reference_mae_ns
                )
                and reference_mae_ns > 0.0
                and np.isfinite(
                    ood_mae_ns
                )
            ):
                mae_degradation = (
                    ood_mae_ns
                    / reference_mae_ns
                )

            else:
                mae_degradation = np.nan

            rows.append(
                {
                    "estimator": (
                        estimator_name
                    ),
                    "representation": (
                        representation_name
                    ),
                    "reference_test_id": (
                        reference_test_id
                    ),
                    "ood_test_id": (
                        ood_test_id
                    ),
                    "reference_mae_ns": (
                        reference_mae_ns
                    ),
                    "ood_mae_ns": (
                        ood_mae_ns
                    ),
                    "mae_degradation": float(
                        mae_degradation
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def _get_generalization_test_representations(
    *,
    prepared: GeneralizationPreparedData,
    test_id: str,
) -> dict[str, Any]:
    """Return the three canonical representations for one test."""

    normalized_id = (
        test_id
        .strip()
        .upper()
    )

    if normalized_id not in prepared.tests:
        raise KeyError(
            f"Prepared data do not contain "
            f"Test {normalized_id!r}."
        )

    return {
        "engineered_features": (
            prepared.X_features[
                normalized_id
            ]
        ),
        "normalized_histogram": (
            prepared.X_normalized[
                normalized_id
            ]
        ),
        "pca_histogram": (
            prepared.X_pca[
                normalized_id
            ]
        ),
    }


def _evaluate_nonclassical_generalization_tests(
    *,
    prepared: GeneralizationPreparedData,
    fitted_estimators: dict[
        str,
        dict[str, Any],
    ],
    test_ids: tuple[str, ...],
) -> list[pd.DataFrame]:
    """Evaluate baselines and ML estimators on selected final tests."""

    if not test_ids:
        raise ValueError(
            "test_ids must not be empty."
        )

    normalized_test_ids = tuple(
        test_id.strip().upper()
        for test_id in test_ids
    )

    if len(
        set(normalized_test_ids)
    ) != len(normalized_test_ids):
        raise ValueError(
            "test_ids must not contain duplicates."
        )

    expected_models = {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    expected_representations = {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    if set(
        fitted_estimators
    ) != expected_models:
        raise ValueError(
            "fitted_estimators must contain exactly "
            "Ridge, Random Forest, and "
            "HistGradientBoosting."
        )

    for model_name in expected_models:
        if (
            set(
                fitted_estimators[
                    model_name
                ]
            )
            != expected_representations
        ):
            raise ValueError(
                f"{model_name!r} must contain exactly "
                "the three canonical representations."
            )

    prediction_tables: list[
        pd.DataFrame
    ] = []

    for test_id in normalized_test_ids:
        if test_id not in prepared.tests:
            raise KeyError(
                f"Prepared data do not contain "
                f"Test {test_id!r}."
            )

        test = prepared.tests[
            test_id
        ]

        representations = (
            _get_generalization_test_representations(
                prepared=prepared,
                test_id=test_id,
            )
        )

        #
        # Constant development-mean baseline.
        #
        constant_predictions = (
            predict_constant_mean_baseline(
                y_train=(
                    prepared.development.y
                ),
                n_predictions=(
                    test.y.size
                ),
            )
        )

        prediction_tables.append(
            _build_generalization_prediction_table(
                estimator_name=(
                    "constant_mean"
                ),
                representation_name="none",
                test=test,
                y_pred=constant_predictions,
            )
        )

        #
        # Mean-arrival-time baseline.
        #
        X_features = representations[
            "engineered_features"
        ]

        mean_arrival_predictions = (
            estimate_lifetime_from_mean_arrival(
                mean_arrival_time_ns=(
                    X_features[
                        "mean_arrival_time_ns"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
                peak_time_ns=(
                    X_features[
                        "peak_time_ns"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
            )
        )

        prediction_tables.append(
            _build_generalization_prediction_table(
                estimator_name=(
                    "mean_arrival_time"
                ),
                representation_name=(
                    "engineered_features"
                ),
                test=test,
                y_pred=(
                    mean_arrival_predictions
                ),
            )
        )

        #
        # Three ML models × three representations.
        #
        for model_name in (
            "ridge",
            "random_forest",
            "hist_gradient_boosting",
        ):
            for representation_name in (
                "engineered_features",
                "normalized_histogram",
                "pca_histogram",
            ):
                estimator = (
                    fitted_estimators[
                        model_name
                    ][
                        representation_name
                    ]
                )

                X_test = representations[
                    representation_name
                ]

                predictions = np.asarray(
                    estimator.predict(
                        X_test
                    ),
                    dtype=np.float64,
                )

                prediction_tables.append(
                    _build_generalization_prediction_table(
                        estimator_name=(
                            model_name
                        ),
                        representation_name=(
                            representation_name
                        ),
                        test=test,
                        y_pred=predictions,
                    )
                )

    return prediction_tables


def _evaluate_principal_nonclassical_estimators(
    *,
    prepared: GeneralizationABPreparedData,
    fitted_estimators: dict[
        str,
        dict[str, Any],
    ],
) -> list[pd.DataFrame]:
    """Evaluate baselines and primary ML estimators on Tests A/B."""

    prediction_tables: list[
        pd.DataFrame
    ] = []

    test_inputs = (
        (
            prepared.test_a,
            prepared.X_features_a,
        ),
        (
            prepared.test_b,
            prepared.X_features_b,
        ),
    )

    for (
            test,
            X_features,
    ) in test_inputs:

        # Constant baseline:
        # training/development target mean only.
        constant_predictions = (
            predict_constant_mean_baseline(
                y_train=(
                    prepared.development.y
                ),
                n_predictions=(
                    test.y.size
                ),
            )
        )

        prediction_tables.append(
            _build_generalization_prediction_table(
                estimator_name=(
                    "constant_mean"
                ),
                representation_name=(
                    "none"
                ),
                test=test,
                y_pred=constant_predictions,
            )
        )

        # Physics-inspired mean-arrival-time baseline.
        mean_arrival_predictions = (
            estimate_lifetime_from_mean_arrival(
                mean_arrival_time_ns=(
                    X_features[
                        "mean_arrival_time_ns"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
                peak_time_ns=(
                    X_features[
                        "peak_time_ns"
                    ].to_numpy(
                        dtype=np.float64
                    )
                ),
            )
        )

        prediction_tables.append(
            _build_generalization_prediction_table(
                estimator_name=(
                    "mean_arrival_time"
                ),
                representation_name=(
                    "engineered_features"
                ),
                test=test,
                y_pred=(
                    mean_arrival_predictions
                ),
            )
        )

        # Principal ML estimator set.
        for model_name in (
            "ridge",
            "random_forest",
            "hist_gradient_boosting",
        ):
            try:
                estimator = (
                    fitted_estimators[
                        model_name
                    ][
                        "engineered_features"
                    ]
                )

            except KeyError as exc:
                raise KeyError(
                    "Missing fitted engineered-feature "
                    f"estimator for {model_name!r}."
                ) from exc

            predictions = np.asarray(
                estimator.predict(
                    X_features
                ),
                dtype=np.float64,
            )

            prediction_tables.append(
                _build_generalization_prediction_table(
                    estimator_name=(
                        model_name
                    ),
                    representation_name=(
                        "engineered_features"
                    ),
                    test=test,
                    y_pred=predictions,
                )
            )

    return prediction_tables


def _evaluate_classical_ab(
    *,
    prepared: GeneralizationABPreparedData,
    irf: ArrayLike,
    temporal_shift_bounds: tuple[
        float,
        float,
    ],
    objective: str = "poisson",
) -> tuple[
    list[pd.DataFrame],
    dict[
        str,
        ReconvolutionBenchmarkResult,
    ],
]:
    """Evaluate classical reconvolution on Tests A and B."""

    prediction_tables: list[
        pd.DataFrame
    ] = []

    classical_results: dict[
        str,
        ReconvolutionBenchmarkResult,
    ] = {}

    for test in (
        prepared.test_a,
        prepared.test_b,
    ):
        result = (
            evaluate_reconvolution_benchmark(
                time=test.time,
                X_histograms=(
                    test.X_histograms
                ),
                y_true=test.y,
                metadata=test.metadata,
                irf=irf,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective=objective,
            )
        )

        classical_results[
            test.test_id
        ] = result

        fitted_lifetimes = (
            result.per_curve[
                "fitted_lifetime_ns"
            ].to_numpy(
                dtype=np.float64
            )
        )

        valid_mask = (
            result.per_curve[
                "valid_fit"
            ].to_numpy(
                dtype=bool
            )
        )

        prediction_tables.append(
            _build_generalization_prediction_table(
                estimator_name=(
                    "classical_reconvolution"
                ),
                representation_name=(
                    "raw_histogram"
                ),
                test=test,
                y_pred=fitted_lifetimes,
                valid_mask=valid_mask,
            )
        )

    return (
        prediction_tables,
        classical_results,
    )


def evaluate_principal_ab_benchmark(
    *,
    prepared: GeneralizationABPreparedData,
    fitted_estimators: dict[
        str,
        dict[str, Any],
    ],
    classical_irf: ArrayLike,
    temporal_shift_bounds: tuple[
        float,
        float,
    ] = (-0.5, 0.5),
    classical_objective: str = "poisson",
) -> PrincipalABBenchmarkResult:
    """Evaluate the principal estimator set on Tests A and B."""

    prediction_tables = (
        _evaluate_principal_nonclassical_estimators(
            prepared=prepared,
            fitted_estimators=(
                fitted_estimators
            ),
        )
    )

    (
        classical_prediction_tables,
        classical_results,
    ) = _evaluate_classical_ab(
        prepared=prepared,
        irf=classical_irf,
        temporal_shift_bounds=(
            temporal_shift_bounds
        ),
        objective=classical_objective,
    )

    prediction_tables.extend(
        classical_prediction_tables
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    summary = (
        summarize_generalization_predictions(
            predictions
        )
    )

    degradation = (
        build_mae_degradation_table(
            summary
        )
    )

    return PrincipalABBenchmarkResult(
        predictions=predictions,
        summary=summary,
        degradation=degradation,
        classical_results=(
            classical_results
        ),
    )


@dataclass(frozen=True)
class RepresentationABBenchmarkResult:
    """A/B robustness benchmark across ML representations."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    degradation: pd.DataFrame
    comparison: pd.DataFrame


def build_ab_comparison_table(
    *,
    summary: pd.DataFrame,
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    """Build a side-by-side Test-A versus Test-B comparison."""

    required_summary_columns = {
        "estimator",
        "representation",
        "test_id",
        "mae_ns",
        "median_absolute_error_ns",
        "rmse_ns",
        "bias_ns",
        "p90_absolute_error_ns",
        "p95_absolute_error_ns",
        "classical_failure_rate",
    }

    missing_summary_columns = (
        required_summary_columns
        - set(summary.columns)
    )

    if missing_summary_columns:
        raise ValueError(
            "Summary table is missing required columns: "
            + ", ".join(
                sorted(
                    missing_summary_columns
                )
            )
        )

    required_degradation_columns = {
        "estimator",
        "representation",
        "mae_degradation",
    }

    missing_degradation_columns = (
        required_degradation_columns
        - set(degradation.columns)
    )

    if missing_degradation_columns:
        raise ValueError(
            "Degradation table is missing required columns: "
            + ", ".join(
                sorted(
                    missing_degradation_columns
                )
            )
        )

    test_a = (
        summary.loc[
            summary["test_id"] == "A"
        ]
        .copy()
    )

    test_b = (
        summary.loc[
            summary["test_id"] == "B"
        ]
        .copy()
    )

    join_columns = [
        "estimator",
        "representation",
    ]

    a_keys = set(
        map(
            tuple,
            test_a[
                join_columns
            ].to_numpy(),
        )
    )

    b_keys = set(
        map(
            tuple,
            test_b[
                join_columns
            ].to_numpy(),
        )
    )

    if a_keys != b_keys:
        raise ValueError(
            "Tests A and B must contain identical "
            "estimator/representation combinations."
        )

    metric_columns = {
        "mae_ns": "mae",
        "median_absolute_error_ns": (
            "median_absolute_error"
        ),
        "rmse_ns": "rmse",
        "bias_ns": "bias",
        "p90_absolute_error_ns": (
            "p90_absolute_error"
        ),
        "p95_absolute_error_ns": (
            "p95_absolute_error"
        ),
        "classical_failure_rate": (
            "classical_failure_rate"
        ),
    }

    a_columns = join_columns.copy()
    b_columns = join_columns.copy()

    for column in metric_columns:
        a_columns.append(column)
        b_columns.append(column)

    test_a = test_a[
        a_columns
    ]

    test_b = test_b[
        b_columns
    ]

    test_a = test_a.rename(
        columns={
            column: (
                f"{base_name}_a"
            )
            for (
                column,
                base_name,
            ) in metric_columns.items()
        }
    )

    test_b = test_b.rename(
        columns={
            column: (
                f"{base_name}_b"
            )
            for (
                column,
                base_name,
            ) in metric_columns.items()
        }
    )

    comparison = test_a.merge(
        test_b,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    comparison = comparison.merge(
        degradation[
            [
                "estimator",
                "representation",
                "mae_degradation",
            ]
        ],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    return comparison


def evaluate_ml_representation_ab_benchmark(
    *,
    prepared: GeneralizationABPreparedData,
    fitted_estimators: dict[
        str,
        dict[str, Any],
    ],
) -> RepresentationABBenchmarkResult:
    """Evaluate ML representation robustness on Tests A and B.

    All estimators must already be fitted exclusively on
    development data.

    This function performs prediction and evaluation only.
    """

    development_representations = {
        "engineered_features": (
            prepared.development.X_features
        ),
        "normalized_histogram": (
            prepared.X_normalized_development
        ),
        "pca_histogram": (
            prepared.X_pca_development
        ),
    }

    test_a_representations = {
        "engineered_features": (
            prepared.X_features_a
        ),
        "normalized_histogram": (
            prepared.X_normalized_a
        ),
        "pca_histogram": (
            prepared.X_pca_a
        ),
    }

    test_b_representations = {
        "engineered_features": (
            prepared.X_features_b
        ),
        "normalized_histogram": (
            prepared.X_normalized_b
        ),
        "pca_histogram": (
            prepared.X_pca_b
        ),
    }

    expected_models = {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    expected_representations = {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    if set(fitted_estimators) != expected_models:
        raise ValueError(
            "fitted_estimators must contain exactly "
            "Ridge, Random Forest, and "
            "HistGradientBoosting."
        )

    for model_name in expected_models:
        if (
            set(
                fitted_estimators[
                    model_name
                ]
            )
            != expected_representations
        ):
            raise ValueError(
                f"{model_name!r} must contain exactly "
                "the three canonical representations."
            )

    # Basic alignment check against the development data
    # on which the estimators were fitted.
    for (
        representation_name,
        X_development,
    ) in development_representations.items():
        if (
            X_development.shape[0]
            != prepared.development.y.size
        ):
            raise ValueError(
                "Development representation "
                f"{representation_name!r} is misaligned."
            )

    prediction_tables: list[
        pd.DataFrame
    ] = []

    test_definitions = (
        (
            prepared.test_a,
            test_a_representations,
        ),
        (
            prepared.test_b,
            test_b_representations,
        ),
    )

    for (
        test,
        representations,
    ) in test_definitions:

        for model_name in (
            "ridge",
            "random_forest",
            "hist_gradient_boosting",
        ):
            for representation_name in (
                "engineered_features",
                "normalized_histogram",
                "pca_histogram",
            ):
                estimator = (
                    fitted_estimators[
                        model_name
                    ][
                        representation_name
                    ]
                )

                X_test = representations[
                    representation_name
                ]

                predictions = np.asarray(
                    estimator.predict(
                        X_test
                    ),
                    dtype=np.float64,
                )

                prediction_tables.append(
                    _build_generalization_prediction_table(
                        estimator_name=(
                            model_name
                        ),
                        representation_name=(
                            representation_name
                        ),
                        test=test,
                        y_pred=predictions,
                    )
                )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    summary = (
        summarize_generalization_predictions(
            predictions
        )
    )

    degradation = (
        build_mae_degradation_table(
            summary
        )
    )

    comparison = (
        build_ab_comparison_table(
            summary=summary,
            degradation=degradation,
        )
    )

    return RepresentationABBenchmarkResult(
        predictions=predictions,
        summary=summary,
        degradation=degradation,
        comparison=comparison,
    )


def add_test_b_photon_count_regime(
    predictions: pd.DataFrame,
    *,
    familiar_min_photons: int,
    familiar_max_photons: int,
) -> pd.DataFrame:
    """Label Test-B predictions by photon-count OOD direction."""

    required_columns = {
        "test_id",
        "signal_photon_count_target",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if familiar_min_photons <= 0:
        raise ValueError(
            "familiar_min_photons must be positive."
        )

    if (
        familiar_max_photons
        <= familiar_min_photons
    ):
        raise ValueError(
            "familiar_max_photons must be greater "
            "than familiar_min_photons."
        )

    if not (
        predictions["test_id"] == "B"
    ).all():
        raise ValueError(
            "Photon-count OOD labeling requires "
            "Test-B predictions only."
        )

    result = predictions.copy(
        deep=True
    )

    photon_counts = result[
        "signal_photon_count_target"
    ].to_numpy()

    regimes = np.full(
        len(result),
        "familiar",
        dtype=object,
    )

    regimes[
        photon_counts
        < familiar_min_photons
    ] = "low_photon_ood"

    regimes[
        photon_counts
        > familiar_max_photons
    ] = "high_photon_ood"

    result[
        "photon_count_regime"
    ] = regimes

    return result


def summarize_test_b_photon_count_ood(
    *,
    predictions: pd.DataFrame,
    familiar_min_photons: int,
    familiar_max_photons: int,
) -> pd.DataFrame:
    """Summarize low- and high-photon Test-B robustness."""

    required_columns = {
        "estimator",
        "representation",
        "test_id",
        "true_lifetime_ns",
        "predicted_lifetime_ns",
        "valid_prediction",
        "signal_photon_count_target",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    test_b = predictions.loc[
        predictions[
            "test_id"
        ] == "B"
    ].copy()

    if test_b.empty:
        raise ValueError(
            "Predictions do not contain Test B."
        )

    test_b = add_test_b_photon_count_regime(
        test_b,
        familiar_min_photons=(
            familiar_min_photons
        ),
        familiar_max_photons=(
            familiar_max_photons
        ),
    )

    rows: list[
        dict[str, str | int | float]
    ] = []

    grouped = test_b.groupby(
        [
            "estimator",
            "representation",
            "photon_count_regime",
        ],
        sort=False,
    )

    for (
        estimator_name,
        representation_name,
        regime_name,
    ), group in grouped:

        valid_mask = group[
            "valid_prediction"
        ].to_numpy(
            dtype=bool
        )

        n_total = int(
            len(group)
        )

        n_valid = int(
            np.sum(valid_mask)
        )

        if n_valid > 0:
            metrics = (
                calculate_robustness_metrics(
                    y_true=group.loc[
                        valid_mask,
                        "true_lifetime_ns",
                    ],
                    y_pred=group.loc[
                        valid_mask,
                        "predicted_lifetime_ns",
                    ],
                )
            )

            mae_ns = metrics.mae_ns
            median_ae_ns = (
                metrics
                .median_absolute_error_ns
            )
            rmse_ns = metrics.rmse_ns
            bias_ns = metrics.bias_ns
            p90_ae_ns = (
                metrics
                .p90_absolute_error_ns
            )
            p95_ae_ns = (
                metrics
                .p95_absolute_error_ns
            )

        else:
            mae_ns = np.nan
            median_ae_ns = np.nan
            rmse_ns = np.nan
            bias_ns = np.nan
            p90_ae_ns = np.nan
            p95_ae_ns = np.nan

        rows.append(
            {
                "estimator": (
                    estimator_name
                ),
                "representation": (
                    representation_name
                ),
                "photon_count_regime": (
                    regime_name
                ),
                "n_total_samples": n_total,
                "n_valid_predictions": n_valid,
                "mae_ns": mae_ns,
                "median_absolute_error_ns": (
                    median_ae_ns
                ),
                "rmse_ns": rmse_ns,
                "bias_ns": bias_ns,
                "p90_absolute_error_ns": (
                    p90_ae_ns
                ),
                "p95_absolute_error_ns": (
                    p95_ae_ns
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


@dataclass(frozen=True)
class Day53ABReport:
    """Complete reporting tables for the Day-53 A/B experiment."""

    principal_comparison: pd.DataFrame
    representation_comparison: pd.DataFrame
    photon_count_ood_summary: pd.DataFrame


def build_day53_ab_report(
    *,
    principal_result: PrincipalABBenchmarkResult,
    representation_result: RepresentationABBenchmarkResult,
    familiar_min_photons: int,
    familiar_max_photons: int,
) -> Day53ABReport:
    """Build final Day-53 familiar-vs-photon-OOD report tables."""

    principal_comparison = (
        build_ab_comparison_table(
            summary=(
                principal_result.summary
            ),
            degradation=(
                principal_result.degradation
            ),
        )
    )

    representation_comparison = (
        representation_result
        .comparison
        .copy(
            deep=True
        )
    )

    photon_count_ood_summary = (
        summarize_test_b_photon_count_ood(
            predictions=(
                representation_result
                .predictions
            ),
            familiar_min_photons=(
                familiar_min_photons
            ),
            familiar_max_photons=(
                familiar_max_photons
            ),
        )
    )

    return Day53ABReport(
        principal_comparison=(
            principal_comparison
        ),
        representation_comparison=(
            representation_comparison
        ),
        photon_count_ood_summary=(
            photon_count_ood_summary
        ),
    )


def build_generalization_plot_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Adapt Week-8 predictions to the common benchmark plotting schema."""

    required_columns = {
        "estimator",
        "true_lifetime_ns",
        "predicted_lifetime_ns",
        "error_ns",
        "absolute_error_ns",
        "valid_prediction",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    diagnostics = predictions.copy(
        deep=True
    )

    diagnostics = diagnostics.rename(
        columns={
            "estimator": (
                "estimator_name"
            ),
            "valid_prediction": (
                "valid_estimate"
            ),
        }
    )

    return diagnostics


@dataclass(frozen=True)
class InstrumentAcquisitionBenchmarkResult:
    """Non-classical robustness benchmark for Tests A/C/D/E."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    degradation: pd.DataFrame


@dataclass(frozen=True)
class ModelMismatchBenchmarkResult:
    """Non-classical robustness benchmark for Tests A and F.

    Test A provides the familiar mono-exponential reference.

    Test F introduces a controlled bi-exponential decay-model
    mismatch while preserving the paired physical conditions.

    Test-F predictions are interpreted relative to both:

    - the dominant-component lifetime ``tau_1`` used as the
      primary scoring target;
    - the descriptive signal-photon-weighted component lifetime.
    """

    predictions: pd.DataFrame
    summary: pd.DataFrame
    degradation: pd.DataFrame

    test_f_reference_diagnostics: pd.DataFrame
    severity_summary: pd.DataFrame


def build_test_f_reference_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Interpret Test-F predictions against two lifetime references.

    The primary reference is the dominant-component lifetime
    ``tau_1``.

    The signal-photon-weighted component lifetime is retained only
    as a secondary descriptive reference. It must not be interpreted
    as a unique true lifetime of the bi-exponential decay.
    """

    required_columns = {
        "estimator",
        "representation",
        "test_id",
        "predicted_lifetime_ns",
        "valid_prediction",
        "primary_lifetime_ns",
        "secondary_lifetime_ns",
        "secondary_fraction",
        "model_mismatch_severity",
        "signal_photon_weighted_lifetime_ns",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions are missing required Test-F columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    diagnostics = predictions.loc[
        predictions["test_id"] == "F"
    ].copy(
        deep=True
    )

    if diagnostics.empty:
        raise ValueError(
            "Predictions do not contain Test F."
        )

    primary_lifetime = diagnostics[
        "primary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    secondary_lifetime = diagnostics[
        "secondary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    weighted_lifetime = diagnostics[
        "signal_photon_weighted_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(primary_lifetime)
    ):
        raise ValueError(
            "Test-F primary lifetimes must be finite."
        )

    if not np.all(
        np.isfinite(secondary_lifetime)
    ):
        raise ValueError(
            "Test-F secondary lifetimes must be finite."
        )

    if not np.all(
        np.isfinite(weighted_lifetime)
    ):
        raise ValueError(
            "Test-F weighted lifetime references must be finite."
        )

    if np.any(
        secondary_lifetime
        <= primary_lifetime
    ):
        raise ValueError(
            "Test-F secondary lifetimes must exceed "
            "the primary lifetimes."
        )

    predicted = diagnostics[
        "predicted_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    valid = diagnostics[
        "valid_prediction"
    ].to_numpy(
        dtype=bool
    )

    error_to_primary = np.full(
        len(diagnostics),
        np.nan,
        dtype=np.float64,
    )

    error_to_weighted = np.full(
        len(diagnostics),
        np.nan,
        dtype=np.float64,
    )

    effective_position = np.full(
        len(diagnostics),
        np.nan,
        dtype=np.float64,
    )

    error_to_primary[valid] = (
        predicted[valid]
        - primary_lifetime[valid]
    )

    error_to_weighted[valid] = (
        predicted[valid]
        - weighted_lifetime[valid]
    )

    effective_position[valid] = (
        (
            predicted[valid]
            - primary_lifetime[valid]
        )
        / (
            secondary_lifetime[valid]
            - primary_lifetime[valid]
        )
    )

    diagnostics[
        "error_to_primary_ns"
    ] = error_to_primary

    diagnostics[
        "absolute_error_to_primary_ns"
    ] = np.abs(
        error_to_primary
    )

    diagnostics[
        "error_to_signal_photon_weighted_ns"
    ] = error_to_weighted

    diagnostics[
        "absolute_error_to_signal_photon_weighted_ns"
    ] = np.abs(
        error_to_weighted
    )

    diagnostics[
        "effective_mixture_position"
    ] = effective_position

    return diagnostics


def summarize_test_f_model_mismatch(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize Test-F predictions by mismatch severity."""

    diagnostics = (
        build_test_f_reference_diagnostics(
            predictions
        )
    )

    rows: list[
        dict[str, str | int | float]
    ] = []

    grouped = diagnostics.groupby(
        [
            "estimator",
            "representation",
            "model_mismatch_severity",
        ],
        sort=False,
        dropna=False,
    )

    for (
        estimator_name,
        representation_name,
        severity,
    ), group in grouped:

        valid_mask = group[
            "valid_prediction"
        ].to_numpy(
            dtype=bool
        )

        n_total = int(
            len(group)
        )

        n_valid = int(
            np.sum(valid_mask)
        )

        secondary_fractions = np.unique(
            group[
                "secondary_fraction"
            ].to_numpy(
                dtype=np.float64
            )
        )

        if secondary_fractions.size != 1:
            raise ValueError(
                "Each Test-F severity must correspond "
                "to exactly one secondary fraction."
            )

        secondary_fraction = float(
            secondary_fractions[0]
        )

        if n_valid > 0:
            valid_group = group.loc[
                valid_mask
            ]

            primary_metrics = (
                calculate_robustness_metrics(
                    y_true=valid_group[
                        "primary_lifetime_ns"
                    ],
                    y_pred=valid_group[
                        "predicted_lifetime_ns"
                    ],
                )
            )

            weighted_metrics = (
                calculate_robustness_metrics(
                    y_true=valid_group[
                        "signal_photon_weighted_lifetime_ns"
                    ],
                    y_pred=valid_group[
                        "predicted_lifetime_ns"
                    ],
                )
            )

            predicted = valid_group[
                "predicted_lifetime_ns"
            ].to_numpy(
                dtype=np.float64
            )

            mixture_position = valid_group[
                "effective_mixture_position"
            ].to_numpy(
                dtype=np.float64
            )

            mean_prediction = float(
                np.mean(predicted)
            )

            median_prediction = float(
                np.median(predicted)
            )

            mean_mixture_position = float(
                np.mean(mixture_position)
            )

            median_mixture_position = float(
                np.median(mixture_position)
            )

        else:
            primary_metrics = None
            weighted_metrics = None

            mean_prediction = np.nan
            median_prediction = np.nan

            mean_mixture_position = np.nan
            median_mixture_position = np.nan

        rows.append(
            {
                "estimator": (
                    estimator_name
                ),
                "representation": (
                    representation_name
                ),
                "model_mismatch_severity": (
                    severity
                ),
                "secondary_fraction": (
                    secondary_fraction
                ),
                "n_total_samples": (
                    n_total
                ),
                "n_valid_predictions": (
                    n_valid
                ),
                "mean_predicted_lifetime_ns": (
                    mean_prediction
                ),
                "median_predicted_lifetime_ns": (
                    median_prediction
                ),
                "mae_to_primary_ns": (
                    (
                        primary_metrics.mae_ns
                    )
                    if primary_metrics is not None
                    else np.nan
                ),
                "bias_to_primary_ns": (
                    (
                        primary_metrics.bias_ns
                    )
                    if primary_metrics is not None
                    else np.nan
                ),
                "mae_to_signal_photon_weighted_ns": (
                    (
                        weighted_metrics.mae_ns
                    )
                    if weighted_metrics is not None
                    else np.nan
                ),
                "bias_to_signal_photon_weighted_ns": (
                    (
                        weighted_metrics.bias_ns
                    )
                    if weighted_metrics is not None
                    else np.nan
                ),
                "mean_effective_mixture_position": (
                    mean_mixture_position
                ),
                "median_effective_mixture_position": (
                    median_mixture_position
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_model_mismatch_benchmark(
    *,
    prepared: GeneralizationPreparedData,
    fitted_estimators: dict[
        str,
        dict[str, Any],
    ],
) -> ModelMismatchBenchmarkResult:
    """Evaluate non-classical estimators on Tests A and F.

    Test A is the familiar mono-exponential reference.

    Test F changes only the decay model by introducing the
    controlled weak/moderate bi-exponential component.

    All estimators must already be fitted exclusively on
    development data.
    """

    required_test_ids = (
        "A",
        "F",
    )

    missing_test_ids = (
        set(required_test_ids)
        - set(prepared.tests)
    )

    if missing_test_ids:
        raise ValueError(
            "Prepared data are missing required "
            "model-mismatch tests: "
            + ", ".join(
                sorted(missing_test_ids)
            )
        )

    prediction_tables = (
        _evaluate_nonclassical_generalization_tests(
            prepared=prepared,
            fitted_estimators=(
                fitted_estimators
            ),
            test_ids=required_test_ids,
        )
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    summary = (
        summarize_generalization_predictions(
            predictions
        )
    )

    degradation = (
        build_reference_mae_degradation_table(
            summary,
            reference_test_id="A",
        )
    )

    test_f_reference_diagnostics = (
        build_test_f_reference_diagnostics(
            predictions
        )
    )

    severity_summary = (
        summarize_test_f_model_mismatch(
            predictions
        )
    )

    return ModelMismatchBenchmarkResult(
        predictions=predictions,
        summary=summary,
        degradation=degradation,
        test_f_reference_diagnostics=(
            test_f_reference_diagnostics
        ),
        severity_summary=(
            severity_summary
        ),
    )


def evaluate_instrument_acquisition_benchmark(
    *,
    prepared: GeneralizationPreparedData,
    fitted_estimators: dict[
        str,
        dict[str, Any],
    ],
) -> InstrumentAcquisitionBenchmarkResult:
    """Evaluate non-classical estimators on Tests A, C, D, and E.

    Test A provides the familiar reference.

    Tests C-E probe:

    - IRF-width mismatch;
    - elevated background;
    - temporal misalignment.

    All estimators must already be fitted exclusively
    on development data.
    """

    required_test_ids = (
        "A",
        "C",
        "D",
        "E",
    )

    missing_test_ids = (
        set(required_test_ids)
        - set(prepared.tests)
    )

    if missing_test_ids:
        raise ValueError(
            "Prepared data are missing required "
            "instrument/acquisition tests: "
            + ", ".join(
                sorted(missing_test_ids)
            )
        )

    prediction_tables = (
        _evaluate_nonclassical_generalization_tests(
            prepared=prepared,
            fitted_estimators=(
                fitted_estimators
            ),
            test_ids=required_test_ids,
        )
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    summary = (
        summarize_generalization_predictions(
            predictions
        )
    )

    degradation = (
        build_reference_mae_degradation_table(
            summary,
            reference_test_id="A",
        )
    )

    return (
        InstrumentAcquisitionBenchmarkResult(
            predictions=predictions,
            summary=summary,
            degradation=degradation,
        )
    )


def _evaluate_classical_generalization_test(
    *,
    test: GeneralizationTestMeasurements,
    irf_centre_ns: float,
    temporal_shift_bounds: tuple[
        float,
        float,
    ],
    objective: str = "poisson",
    background_fraction: float = 0.10,
    assumed_irf_fwhm_ns: float | None = None,
    irf_mode: str,
) -> pd.DataFrame:
    """Evaluate one robustness test with a specified IRF policy.

    If assumed_irf_fwhm_ns is None, every curve is fitted using
    an IRF whose width matches that curve's simulation metadata.

    Otherwise all curves are fitted using the same fixed assumed
    IRF width.
    """

    required_metadata_columns = {
        "sample_id",
        "irf_fwhm_ns",
        "irf_shift_ns",
    }

    missing_columns = (
        required_metadata_columns
        - set(test.metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Test metadata are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if not np.isfinite(
        irf_centre_ns
    ):
        raise ValueError(
            "irf_centre_ns must be finite."
        )

    if assumed_irf_fwhm_ns is not None:
        if (
            not np.isfinite(
                assumed_irf_fwhm_ns
            )
            or assumed_irf_fwhm_ns <= 0.0
        ):
            raise ValueError(
                "assumed_irf_fwhm_ns must be "
                "finite and positive."
            )

        assumed_widths = np.full(
            test.y.size,
            float(
                assumed_irf_fwhm_ns
            ),
            dtype=np.float64,
        )

    else:
        assumed_widths = (
            test.metadata[
                "irf_fwhm_ns"
            ].to_numpy(
                dtype=np.float64
            )
        )

    diagnostic_tables: list[
        pd.DataFrame
    ] = []

    for assumed_width_ns in np.unique(
        assumed_widths
    ):
        mask = np.isclose(
            assumed_widths,
            assumed_width_ns,
            rtol=0.0,
            atol=1e-12,
        )

        indices = np.flatnonzero(
            mask
        )

        irf = generate_gaussian_irf(
            time=test.time,
            centre=irf_centre_ns,
            fwhm=float(
                assumed_width_ns
            ),
        )

        irf = normalize_irf(
            time=test.time,
            irf=irf,
        )

        result = (
            evaluate_reconvolution_benchmark(
                time=test.time,
                X_histograms=(
                    test.X_histograms[
                        indices
                    ]
                ),
                y_true=(
                    test.y[
                        indices
                    ]
                ),
                metadata=(
                    test.metadata.iloc[
                        indices
                    ].reset_index(
                        drop=True
                    )
                ),
                irf=irf,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective=objective,
                background_fraction=(
                    background_fraction
                ),
            )
        )

        diagnostics = (
            result.per_curve.copy(
                deep=True
            )
        )

        diagnostics[
            "classical_irf_mode"
        ] = irf_mode

        diagnostics[
            "assumed_irf_fwhm_ns"
        ] = float(
            assumed_width_ns
        )

        diagnostics[
            "irf_fwhm_error_ns"
        ] = (
            diagnostics[
                "assumed_irf_fwhm_ns"
            ]
            - diagnostics[
                "irf_fwhm_ns"
            ]
        )

        diagnostics[
            "temporal_shift_error_ns"
        ] = (
            diagnostics[
                "fitted_temporal_shift_ns"
            ]
            - diagnostics[
                "irf_shift_ns"
            ]
        )

        diagnostic_tables.append(
            diagnostics
        )

    combined = pd.concat(
        diagnostic_tables,
        ignore_index=True,
    )

    combined = (
        combined.sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    return combined


def _build_classical_generalization_prediction_table(
    *,
    test: GeneralizationTestMeasurements,
    diagnostics: pd.DataFrame,
    estimator_name: str,
) -> pd.DataFrame:
    """Convert classical fit diagnostics to robustness predictions."""

    if len(
        diagnostics
    ) != test.y.size:
        raise ValueError(
            "Classical diagnostics must contain "
            "one row per test sample."
        )

    diagnostic_sample_ids = (
        diagnostics[
            "sample_id"
        ].to_numpy()
    )

    test_sample_ids = (
        test.metadata[
            "sample_id"
        ].to_numpy()
    )

    if not np.array_equal(
        diagnostic_sample_ids,
        test_sample_ids,
    ):
        raise ValueError(
            "Classical diagnostics are not aligned "
            "with the robustness test."
        )

    predictions = (
        _build_generalization_prediction_table(
            estimator_name=(
                estimator_name
            ),
            representation_name=(
                "raw_histogram"
            ),
            test=test,
            y_pred=(
                diagnostics[
                    "fitted_lifetime_ns"
                ].to_numpy(
                    dtype=np.float64
                )
            ),
            valid_mask=(
                diagnostics[
                    "valid_fit"
                ].to_numpy(
                    dtype=bool
                )
            ),
        )
    )

    predictions[
        "classical_irf_mode"
    ] = diagnostics[
        "classical_irf_mode"
    ].to_numpy()

    predictions[
        "assumed_irf_fwhm_ns"
    ] = diagnostics[
        "assumed_irf_fwhm_ns"
    ].to_numpy(
        dtype=np.float64
    )

    predictions[
        "fitted_temporal_shift_ns"
    ] = diagnostics[
        "fitted_temporal_shift_ns"
    ].to_numpy(
        dtype=np.float64
    )

    predictions[
        "temporal_shift_error_ns"
    ] = diagnostics[
        "temporal_shift_error_ns"
    ].to_numpy(
        dtype=np.float64
    )

    return predictions


def _build_classical_instrument_degradation_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare all classical C-E experiments with correct Test A."""

    reference_rows = summary.loc[
        (
            summary["estimator"]
            == (
                "classical_reconvolution_correct_irf"
            )
        )
        & (
            summary["test_id"]
            == "A"
        )
    ]

    if len(
        reference_rows
    ) != 1:
        raise ValueError(
            "Exactly one correct-IRF Test-A "
            "reference row is required."
        )

    reference_mae_ns = float(
        reference_rows[
            "mae_ns"
        ].iloc[0]
    )

    ood_rows = summary.loc[
        ~(
            (
                summary["estimator"]
                == (
                    "classical_reconvolution_correct_irf"
                )
            )
            & (
                summary["test_id"]
                == "A"
            )
        )
    ]

    rows: list[
        dict[str, str | float]
    ] = []

    for _, row in (
        ood_rows.iterrows()
    ):
        ood_mae_ns = float(
            row["mae_ns"]
        )

        if (
            np.isfinite(
                reference_mae_ns
            )
            and reference_mae_ns > 0.0
            and np.isfinite(
                ood_mae_ns
            )
        ):
            degradation = (
                ood_mae_ns
                / reference_mae_ns
            )

        else:
            degradation = np.nan

        rows.append(
            {
                "reference_estimator": (
                    "classical_reconvolution_correct_irf"
                ),
                "reference_test_id": "A",
                "ood_estimator": (
                    row["estimator"]
                ),
                "ood_test_id": (
                    row["test_id"]
                ),
                "reference_mae_ns": (
                    reference_mae_ns
                ),
                "ood_mae_ns": (
                    ood_mae_ns
                ),
                "mae_degradation": float(
                    degradation
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _build_test_c_classical_irf_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare correct and deliberately incorrect IRFs on Test C."""

    correct = summary.loc[
        (
            summary["estimator"]
            == (
                "classical_reconvolution_correct_irf"
            )
        )
        & (
            summary["test_id"]
            == "C"
        )
    ]

    nominal = summary.loc[
        (
            summary["estimator"]
            == (
                "classical_reconvolution_nominal_irf"
            )
        )
        & (
            summary["test_id"]
            == "C"
        )
    ]

    if (
        len(correct) != 1
        or len(nominal) != 1
    ):
        raise ValueError(
            "Test C requires exactly one correct-IRF "
            "and one nominal-IRF summary row."
        )

    correct_row = correct.iloc[0]
    nominal_row = nominal.iloc[0]

    correct_mae = float(
        correct_row["mae_ns"]
    )

    nominal_mae = float(
        nominal_row["mae_ns"]
    )

    if (
        np.isfinite(correct_mae)
        and correct_mae > 0.0
        and np.isfinite(nominal_mae)
    ):
        mismatch_penalty = (
            nominal_mae
            / correct_mae
        )
    else:
        mismatch_penalty = np.nan

    return pd.DataFrame(
        [
            {
                "mae_correct_irf_ns": (
                    correct_mae
                ),
                "mae_nominal_irf_ns": (
                    nominal_mae
                ),
                "nominal_to_correct_mae_ratio": (
                    float(
                        mismatch_penalty
                    )
                ),
                "bias_correct_irf_ns": float(
                    correct_row[
                        "bias_ns"
                    ]
                ),
                "bias_nominal_irf_ns": float(
                    nominal_row[
                        "bias_ns"
                    ]
                ),
                "p95_correct_irf_ns": float(
                    correct_row[
                        "p95_absolute_error_ns"
                    ]
                ),
                "p95_nominal_irf_ns": float(
                    nominal_row[
                        "p95_absolute_error_ns"
                    ]
                ),
                "failure_rate_correct_irf": float(
                    correct_row[
                        "classical_failure_rate"
                    ]
                ),
                "failure_rate_nominal_irf": float(
                    nominal_row[
                        "classical_failure_rate"
                    ]
                ),
            }
        ]
    )


@dataclass(frozen=True)
class ClassicalInstrumentAcquisitionBenchmarkResult:
    """Classical robustness evaluation for Tests A/C/D/E."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    degradation: pd.DataFrame

    fit_diagnostics: pd.DataFrame

    test_c_irf_comparison: pd.DataFrame


def evaluate_classical_instrument_acquisition_benchmark(
    *,
    tests: dict[
        str,
        GeneralizationTestMeasurements,
    ],
    irf_centre_ns: float,
    nominal_irf_fwhm_ns: float = 0.40,
    temporal_shift_bounds: tuple[
        float,
        float,
    ] = (-0.5, 0.5),
    objective: str = "poisson",
    background_fraction: float = 0.10,
) -> ClassicalInstrumentAcquisitionBenchmarkResult:
    """Evaluate classical reconvolution on Tests A, C, D, and E.

    A, C, D, and E are first fitted with the correct test IRF
    width.

    Test C is additionally fitted using one deliberately incorrect
    familiar IRF width to quantify instrument-model mismatch.
    """

    required_test_ids = {
        "A",
        "C",
        "D",
        "E",
    }

    missing_test_ids = (
        required_test_ids
        - set(tests)
    )

    if missing_test_ids:
        raise ValueError(
            "Missing required classical robustness tests: "
            + ", ".join(
                sorted(missing_test_ids)
            )
        )

    prediction_tables: list[
        pd.DataFrame
    ] = []

    diagnostic_tables: list[
        pd.DataFrame
    ] = []

    #
    # A/C/D/E with the physically correct IRF width.
    #
    for test_id in (
        "A",
        "C",
        "D",
        "E",
    ):
        test = tests[
            test_id
        ]

        diagnostics = (
            _evaluate_classical_generalization_test(
                test=test,
                irf_centre_ns=(
                    irf_centre_ns
                ),
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective=objective,
                background_fraction=(
                    background_fraction
                ),
                assumed_irf_fwhm_ns=None,
                irf_mode=(
                    "correct_test_irf"
                ),
            )
        )

        diagnostics[
            "estimator"
        ] = (
            "classical_reconvolution_correct_irf"
        )

        diagnostic_tables.append(
            diagnostics
        )

        prediction_tables.append(
            _build_classical_generalization_prediction_table(
                test=test,
                diagnostics=diagnostics,
                estimator_name=(
                    "classical_reconvolution_correct_irf"
                ),
            )
        )

    #
    # Test C again, but deliberately using the wrong
    # familiar/nominal IRF.
    #
    test_c = tests[
        "C"
    ]

    nominal_diagnostics = (
        _evaluate_classical_generalization_test(
            test=test_c,
            irf_centre_ns=(
                irf_centre_ns
            ),
            temporal_shift_bounds=(
                temporal_shift_bounds
            ),
            objective=objective,
            background_fraction=(
                background_fraction
            ),
            assumed_irf_fwhm_ns=(
                nominal_irf_fwhm_ns
            ),
            irf_mode=(
                "nominal_familiar_irf"
            ),
        )
    )

    nominal_diagnostics[
        "estimator"
    ] = (
        "classical_reconvolution_nominal_irf"
    )

    diagnostic_tables.append(
        nominal_diagnostics
    )

    prediction_tables.append(
        _build_classical_generalization_prediction_table(
            test=test_c,
            diagnostics=(
                nominal_diagnostics
            ),
            estimator_name=(
                "classical_reconvolution_nominal_irf"
            ),
        )
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    fit_diagnostics = pd.concat(
        diagnostic_tables,
        ignore_index=True,
    )

    summary = (
        summarize_generalization_predictions(
            predictions
        )
    )

    degradation = (
        _build_classical_instrument_degradation_table(
            summary
        )
    )

    test_c_irf_comparison = (
        _build_test_c_classical_irf_comparison(
            summary
        )
    )

    return (
        ClassicalInstrumentAcquisitionBenchmarkResult(
            predictions=predictions,
            summary=summary,
            degradation=degradation,
            fit_diagnostics=(
                fit_diagnostics
            ),
            test_c_irf_comparison=(
                test_c_irf_comparison
            ),
        )
    )


def build_instrument_representation_comparison(
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    """Compare ML representation robustness across Tests C-E."""

    required_columns = {
        "estimator",
        "representation",
        "reference_test_id",
        "ood_test_id",
        "reference_mae_ns",
        "ood_mae_ns",
        "mae_degradation",
    }

    missing_columns = (
        required_columns
        - set(degradation.columns)
    )

    if missing_columns:
        raise ValueError(
            "Degradation table is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    ml_estimators = {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    representations = {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }

    expected_ood_tests = {
        "C",
        "D",
        "E",
    }

    data = degradation.loc[
        degradation[
            "estimator"
        ].isin(
            ml_estimators
        )
    ].copy()

    if set(
        data["ood_test_id"]
    ) != expected_ood_tests:
        raise ValueError(
            "Representation comparison requires "
            "Tests C, D, and E."
        )

    if set(
        data["representation"]
    ) != representations:
        raise ValueError(
            "Representation comparison requires "
            "the three canonical ML representations."
        )

    comparison = (
        data.pivot(
            index=[
                "estimator",
                "ood_test_id",
            ],
            columns="representation",
            values="mae_degradation",
        )
        .reset_index()
    )

    comparison.columns.name = None

    comparison = comparison.rename(
        columns={
            "engineered_features": (
                "degradation_engineered_features"
            ),
            "normalized_histogram": (
                "degradation_normalized_histogram"
            ),
            "pca_histogram": (
                "degradation_pca_histogram"
            ),
        }
    )

    comparison[
        "normalized_minus_engineered"
    ] = (
        comparison[
            "degradation_normalized_histogram"
        ]
        - comparison[
            "degradation_engineered_features"
        ]
    )

    comparison[
        "pca_minus_engineered"
    ] = (
        comparison[
            "degradation_pca_histogram"
        ]
        - comparison[
            "degradation_engineered_features"
        ]
    )

    return (
        comparison.sort_values(
            [
                "ood_test_id",
                "estimator",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_test_c_paired_irf_diagnostics(
    fit_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare correct- and nominal-IRF fits curve by curve on Test C."""

    required_columns = {
        "sample_id",
        "test_id",
        "estimator",
        "true_lifetime_ns",
        "fitted_lifetime_ns",
        "absolute_error_ns",
        "valid_fit",
        "fitted_temporal_shift_ns",
        "poisson_deviance",
    }

    missing_columns = (
        required_columns
        - set(fit_diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical diagnostics are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    test_c = fit_diagnostics.loc[
        fit_diagnostics[
            "test_id"
        ] == "C"
    ].copy()

    correct = test_c.loc[
        test_c[
            "estimator"
        ]
        == (
            "classical_reconvolution_correct_irf"
        )
    ].copy()

    nominal = test_c.loc[
        test_c[
            "estimator"
        ]
        == (
            "classical_reconvolution_nominal_irf"
        )
    ].copy()

    if correct.empty or nominal.empty:
        raise ValueError(
            "Test C must contain both correct-IRF "
            "and nominal-IRF classical fits."
        )

    correct = correct[
        [
            "sample_id",
            "true_lifetime_ns",
            "fitted_lifetime_ns",
            "absolute_error_ns",
            "valid_fit",
            "fitted_temporal_shift_ns",
            "poisson_deviance",
        ]
    ].rename(
        columns={
            "fitted_lifetime_ns": (
                "fitted_lifetime_correct_irf_ns"
            ),
            "absolute_error_ns": (
                "absolute_error_correct_irf_ns"
            ),
            "valid_fit": (
                "valid_fit_correct_irf"
            ),
            "fitted_temporal_shift_ns": (
                "fitted_shift_correct_irf_ns"
            ),
            "poisson_deviance": (
                "poisson_deviance_correct_irf"
            ),
        }
    )

    nominal = nominal[
        [
            "sample_id",
            "true_lifetime_ns",
            "fitted_lifetime_ns",
            "absolute_error_ns",
            "valid_fit",
            "fitted_temporal_shift_ns",
            "poisson_deviance",
        ]
    ].rename(
        columns={
            "fitted_lifetime_ns": (
                "fitted_lifetime_nominal_irf_ns"
            ),
            "absolute_error_ns": (
                "absolute_error_nominal_irf_ns"
            ),
            "valid_fit": (
                "valid_fit_nominal_irf"
            ),
            "fitted_temporal_shift_ns": (
                "fitted_shift_nominal_irf_ns"
            ),
            "poisson_deviance": (
                "poisson_deviance_nominal_irf"
            ),
        }
    )

    paired = correct.merge(
        nominal,
        on=[
            "sample_id",
            "true_lifetime_ns",
        ],
        how="inner",
        validate="one_to_one",
    )

    paired[
        "absolute_error_penalty_ns"
    ] = (
        paired[
            "absolute_error_nominal_irf_ns"
        ]
        - paired[
            "absolute_error_correct_irf_ns"
        ]
    )

    paired[
        "fitted_shift_change_ns"
    ] = (
        paired[
            "fitted_shift_nominal_irf_ns"
        ]
        - paired[
            "fitted_shift_correct_irf_ns"
        ]
    )

    paired[
        "poisson_deviance_change"
    ] = (
        paired[
            "poisson_deviance_nominal_irf"
        ]
        - paired[
            "poisson_deviance_correct_irf"
        ]
    )

    return paired.sort_values(
        "sample_id"
    ).reset_index(
        drop=True
    )


def summarize_test_c_paired_irf_diagnostics(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paired Test-C correct-vs-nominal IRF behavior."""

    required_columns = {
        "valid_fit_correct_irf",
        "valid_fit_nominal_irf",
        "absolute_error_penalty_ns",
        "fitted_shift_change_ns",
        "poisson_deviance_change",
    }

    missing_columns = (
        required_columns
        - set(paired.columns)
    )

    if missing_columns:
        raise ValueError(
            "Paired Test-C diagnostics are missing "
            "required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    both_valid = (
        paired[
            "valid_fit_correct_irf"
        ].to_numpy(
            dtype=bool
        )
        & paired[
            "valid_fit_nominal_irf"
        ].to_numpy(
            dtype=bool
        )
    )

    valid = paired.loc[
        both_valid
    ]

    if valid.empty:
        return pd.DataFrame(
            [
                {
                    "n_pairs": len(paired),
                    "n_both_valid": 0,
                    "mean_absolute_error_penalty_ns": (
                        np.nan
                    ),
                    "median_absolute_error_penalty_ns": (
                        np.nan
                    ),
                    "mean_fitted_shift_change_ns": (
                        np.nan
                    ),
                    "mean_poisson_deviance_change": (
                        np.nan
                    ),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "n_pairs": len(paired),
                "n_both_valid": len(valid),
                "mean_absolute_error_penalty_ns": float(
                    valid[
                        "absolute_error_penalty_ns"
                    ].mean()
                ),
                "median_absolute_error_penalty_ns": float(
                    valid[
                        "absolute_error_penalty_ns"
                    ].median()
                ),
                "mean_fitted_shift_change_ns": float(
                    valid[
                        "fitted_shift_change_ns"
                    ].mean()
                ),
                "mean_poisson_deviance_change": float(
                    valid[
                        "poisson_deviance_change"
                    ].mean()
                ),
            }
        ]
    )



def summarize_test_e_temporal_shift_recovery(
    fit_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fitted temporal-shift recovery on Test E."""

    required_columns = {
        "test_id",
        "estimator",
        "irf_shift_ns",
        "fitted_temporal_shift_ns",
        "temporal_shift_error_ns",
        "valid_fit",
        "boundary_hit",
        "absolute_error_ns",
    }

    missing_columns = (
        required_columns
        - set(fit_diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical diagnostics are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    test_e = fit_diagnostics.loc[
        (
            fit_diagnostics[
                "test_id"
            ] == "E"
        )
        & (
            fit_diagnostics[
                "estimator"
            ]
            == (
                "classical_reconvolution_correct_irf"
            )
        )
    ].copy()

    if test_e.empty:
        raise ValueError(
            "Correct-IRF Test-E diagnostics "
            "are required."
        )

    rows: list[
        dict[str, int | float]
    ] = []

    grouped = test_e.groupby(
        "irf_shift_ns",
        sort=True,
    )

    for true_shift_ns, group in grouped:
        valid_mask = group[
            "valid_fit"
        ].to_numpy(
            dtype=bool
        )

        n_total = len(group)

        n_valid = int(
            np.sum(
                valid_mask
            )
        )

        failure_rate = (
            1.0
            - n_valid / n_total
        )

        boundary_hit_rate = float(
            group[
                "boundary_hit"
            ].mean()
        )

        if n_valid > 0:
            valid = group.loc[
                valid_mask
            ]

            shift_errors = valid[
                "temporal_shift_error_ns"
            ].to_numpy(
                dtype=np.float64
            )

            shift_absolute_errors = (
                np.abs(
                    shift_errors
                )
            )

            lifetime_absolute_errors = (
                valid[
                    "absolute_error_ns"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            shift_bias_ns = float(
                np.mean(
                    shift_errors
                )
            )

            shift_mae_ns = float(
                np.mean(
                    shift_absolute_errors
                )
            )

            shift_rmse_ns = float(
                np.sqrt(
                    np.mean(
                        shift_errors**2
                    )
                )
            )

            lifetime_mae_ns = float(
                np.mean(
                    lifetime_absolute_errors
                )
            )

            mean_fitted_shift_ns = float(
                valid[
                    "fitted_temporal_shift_ns"
                ].mean()
            )

        else:
            shift_bias_ns = np.nan
            shift_mae_ns = np.nan
            shift_rmse_ns = np.nan
            lifetime_mae_ns = np.nan
            mean_fitted_shift_ns = np.nan

        rows.append(
            {
                "true_shift_ns": float(
                    true_shift_ns
                ),
                "n_total": n_total,
                "n_valid": n_valid,
                "failure_rate": float(
                    failure_rate
                ),
                "boundary_hit_rate": (
                    boundary_hit_rate
                ),
                "mean_fitted_shift_ns": (
                    mean_fitted_shift_ns
                ),
                "shift_bias_ns": (
                    shift_bias_ns
                ),
                "shift_mae_ns": (
                    shift_mae_ns
                ),
                "shift_rmse_ns": (
                    shift_rmse_ns
                ),
                "lifetime_mae_ns": (
                    lifetime_mae_ns
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


@dataclass(frozen=True)
class InstrumentAcquisitionDiagnostics:
    """Analysis-ready diagnostics for Day 54."""

    representation_comparison: pd.DataFrame

    test_c_paired_irf: pd.DataFrame
    test_c_paired_irf_summary: pd.DataFrame

    test_e_shift_recovery: pd.DataFrame


def build_instrument_acquisition_diagnostics(
    *,
    nonclassical_result: (
        InstrumentAcquisitionBenchmarkResult
    ),
    classical_result: (
        ClassicalInstrumentAcquisitionBenchmarkResult
    ),
) -> InstrumentAcquisitionDiagnostics:
    """Build the principal Day-54 diagnostic tables."""

    representation_comparison = (
        build_instrument_representation_comparison(
            nonclassical_result.degradation
        )
    )

    test_c_paired_irf = (
        build_test_c_paired_irf_diagnostics(
            classical_result.fit_diagnostics
        )
    )

    test_c_paired_irf_summary = (
        summarize_test_c_paired_irf_diagnostics(
            test_c_paired_irf
        )
    )

    test_e_shift_recovery = (
        summarize_test_e_temporal_shift_recovery(
            classical_result.fit_diagnostics
        )
    )

    return InstrumentAcquisitionDiagnostics(
        representation_comparison=(
            representation_comparison
        ),
        test_c_paired_irf=(
            test_c_paired_irf
        ),
        test_c_paired_irf_summary=(
            test_c_paired_irf_summary
        ),
        test_e_shift_recovery=(
            test_e_shift_recovery
        ),
    )


@dataclass(frozen=True)
class ClassicalModelMismatchBenchmarkResult:
    """Classical mono-exponential evaluation for Tests A and F."""

    predictions: pd.DataFrame
    summary: pd.DataFrame
    degradation: pd.DataFrame

    fit_diagnostics: pd.DataFrame
    paired_diagnostics: pd.DataFrame
    severity_summary: pd.DataFrame


def build_classical_af_paired_diagnostics(
    fit_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Build paired classical diagnostics for Tests A and F."""

    required_columns = {
        "sample_id",
        "pair_id",
        "test_id",
        "true_lifetime_ns",
        "fitted_lifetime_ns",
        "valid_fit",
        "boundary_hit",
        "poisson_nll",
        "poisson_deviance",
        "poisson_deviance_per_bin",
    }

    missing_columns = (
        required_columns
        - set(fit_diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical diagnostics are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    test_a = fit_diagnostics.loc[
        fit_diagnostics["test_id"] == "A"
    ].copy()

    test_f = fit_diagnostics.loc[
        fit_diagnostics["test_id"] == "F"
    ].copy()

    if test_a.empty:
        raise ValueError(
            "Classical diagnostics do not contain Test A."
        )

    if test_f.empty:
        raise ValueError(
            "Classical diagnostics do not contain Test F."
        )

    required_f_columns = {
        "primary_lifetime_ns",
        "secondary_lifetime_ns",
        "secondary_fraction",
        "model_mismatch_severity",
        "signal_photon_weighted_lifetime_ns",
    }

    missing_f_columns = (
        required_f_columns
        - set(test_f.columns)
    )

    if missing_f_columns:
        raise ValueError(
            "Test-F classical diagnostics are missing columns: "
            + ", ".join(
                sorted(missing_f_columns)
            )
        )

    test_a = test_a[
        [
            "sample_id",
            "pair_id",
            "true_lifetime_ns",
            "fitted_lifetime_ns",
            "valid_fit",
            "boundary_hit",
            "poisson_nll",
            "poisson_deviance",
            "poisson_deviance_per_bin",
        ]
    ].rename(
        columns={
            "true_lifetime_ns": (
                "true_lifetime_a_ns"
            ),
            "fitted_lifetime_ns": (
                "fitted_lifetime_a_ns"
            ),
            "valid_fit": (
                "valid_fit_a"
            ),
            "boundary_hit": (
                "boundary_hit_a"
            ),
            "poisson_nll": (
                "poisson_nll_a"
            ),
            "poisson_deviance": (
                "poisson_deviance_a"
            ),
            "poisson_deviance_per_bin": (
                "poisson_deviance_per_bin_a"
            ),
        }
    )

    test_f = test_f[
        [
            "sample_id",
            "pair_id",
            "primary_lifetime_ns",
            "secondary_lifetime_ns",
            "secondary_fraction",
            "model_mismatch_severity",
            "signal_photon_weighted_lifetime_ns",
            "fitted_lifetime_ns",
            "valid_fit",
            "boundary_hit",
            "poisson_nll",
            "poisson_deviance",
            "poisson_deviance_per_bin",
        ]
    ].rename(
        columns={
            "fitted_lifetime_ns": (
                "fitted_lifetime_f_ns"
            ),
            "valid_fit": (
                "valid_fit_f"
            ),
            "boundary_hit": (
                "boundary_hit_f"
            ),
            "poisson_nll": (
                "poisson_nll_f"
            ),
            "poisson_deviance": (
                "poisson_deviance_f"
            ),
            "poisson_deviance_per_bin": (
                "poisson_deviance_per_bin_f"
            ),
        }
    )

    paired = test_a.merge(
        test_f,
        on=[
            "sample_id",
            "pair_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(test_a):
        raise ValueError(
            "Tests A and F do not preserve complete "
            "classical sample pairing."
        )

    if not np.allclose(
        paired[
            "true_lifetime_a_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        paired[
            "primary_lifetime_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    ):
        raise ValueError(
            "Test-A lifetime and Test-F primary lifetime "
            "must be paired."
        )

    valid_a = paired[
        "valid_fit_a"
    ].to_numpy(
        dtype=bool
    )

    valid_f = paired[
        "valid_fit_f"
    ].to_numpy(
        dtype=bool
    )

    paired_valid = (
        valid_a
        & valid_f
    )

    paired[
        "paired_valid_fit"
    ] = paired_valid

    primary = paired[
        "primary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    secondary = paired[
        "secondary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    weighted = paired[
        "signal_photon_weighted_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    fitted_a = paired[
        "fitted_lifetime_a_ns"
    ].to_numpy(
        dtype=np.float64
    )

    fitted_f = paired[
        "fitted_lifetime_f_ns"
    ].to_numpy(
        dtype=np.float64
    )

    paired[
        "absolute_error_to_primary_a_ns"
    ] = np.where(
        valid_a,
        np.abs(
            fitted_a
            - primary
        ),
        np.nan,
    )

    paired[
        "absolute_error_to_primary_f_ns"
    ] = np.where(
        valid_f,
        np.abs(
            fitted_f
            - primary
        ),
        np.nan,
    )

    paired[
        "absolute_error_to_signal_photon_weighted_f_ns"
    ] = np.where(
        valid_f,
        np.abs(
            fitted_f
            - weighted
        ),
        np.nan,
    )

    paired[
        "effective_mixture_position_f"
    ] = np.where(
        valid_f,
        (
            fitted_f
            - primary
        )
        / (
            secondary
            - primary
        ),
        np.nan,
    )

    paired[
        "fitted_lifetime_change_ns"
    ] = np.where(
        paired_valid,
        fitted_f
        - fitted_a,
        np.nan,
    )

    paired[
        "poisson_deviance_change"
    ] = np.where(
        paired_valid,
        paired[
            "poisson_deviance_f"
        ].to_numpy(
            dtype=np.float64
        )
        - paired[
            "poisson_deviance_a"
        ].to_numpy(
            dtype=np.float64
        ),
        np.nan,
    )

    paired[
        "poisson_deviance_per_bin_change"
    ] = np.where(
        paired_valid,
        paired[
            "poisson_deviance_per_bin_f"
        ].to_numpy(
            dtype=np.float64
        )
        - paired[
            "poisson_deviance_per_bin_a"
        ].to_numpy(
            dtype=np.float64
        ),
        np.nan,
    )

    # Retained as a descriptive optimizer diagnostic.
    # The current implementation is a reduced Poisson NLL.
    paired[
        "poisson_nll_change"
    ] = np.where(
        paired_valid,
        paired[
            "poisson_nll_f"
        ].to_numpy(
            dtype=np.float64
        )
        - paired[
            "poisson_nll_a"
        ].to_numpy(
            dtype=np.float64
        ),
        np.nan,
    )

    return paired


def summarize_classical_test_f_model_mismatch(
    fit_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize classical Test-F fitting by mismatch severity."""

    required_columns = {
        "test_id",
        "model_mismatch_severity",
        "secondary_fraction",
        "primary_lifetime_ns",
        "secondary_lifetime_ns",
        "signal_photon_weighted_lifetime_ns",
        "fitted_lifetime_ns",
        "valid_fit",
        "boundary_hit",
        "poisson_nll",
        "poisson_deviance",
        "poisson_deviance_per_bin",
    }

    missing_columns = (
        required_columns
        - set(fit_diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical diagnostics are missing required "
            "Test-F columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    test_f = fit_diagnostics.loc[
        fit_diagnostics["test_id"] == "F"
    ].copy()

    if test_f.empty:
        raise ValueError(
            "Classical diagnostics do not contain Test F."
        )

    rows: list[
        dict[str, str | int | float]
    ] = []

    grouped = test_f.groupby(
        "model_mismatch_severity",
        sort=False,
        dropna=False,
    )

    for severity, group in grouped:
        valid_mask = group[
            "valid_fit"
        ].to_numpy(
            dtype=bool
        )

        n_total = int(
            len(group)
        )

        n_valid = int(
            np.sum(valid_mask)
        )

        failure_rate = (
            1.0
            - n_valid / n_total
        )

        boundary_hit_rate = float(
            np.mean(
                group[
                    "boundary_hit"
                ].to_numpy(
                    dtype=bool
                )
            )
        )

        fractions = np.unique(
            group[
                "secondary_fraction"
            ].to_numpy(
                dtype=np.float64
            )
        )

        if fractions.size != 1:
            raise ValueError(
                "Each mismatch severity must correspond "
                "to exactly one secondary fraction."
            )

        secondary_fraction = float(
            fractions[0]
        )

        if n_valid > 0:
            valid_group = group.loc[
                valid_mask
            ]

            fitted = valid_group[
                "fitted_lifetime_ns"
            ].to_numpy(
                dtype=np.float64
            )

            primary = valid_group[
                "primary_lifetime_ns"
            ].to_numpy(
                dtype=np.float64
            )

            secondary = valid_group[
                "secondary_lifetime_ns"
            ].to_numpy(
                dtype=np.float64
            )

            weighted = valid_group[
                "signal_photon_weighted_lifetime_ns"
            ].to_numpy(
                dtype=np.float64
            )

            error_primary = (
                fitted
                - primary
            )

            error_weighted = (
                fitted
                - weighted
            )

            mixture_position = (
                fitted
                - primary
            ) / (
                secondary
                - primary
            )

            poisson_nll = valid_group[
                "poisson_nll"
            ].to_numpy(
                dtype=np.float64
            )

            poisson_deviance = valid_group[
                "poisson_deviance"
            ].to_numpy(
                dtype=np.float64
            )

            deviance_per_bin = valid_group[
                "poisson_deviance_per_bin"
            ].to_numpy(
                dtype=np.float64
            )

            finite_nll = poisson_nll[
                np.isfinite(
                    poisson_nll
                )
            ]

            finite_deviance = poisson_deviance[
                np.isfinite(
                    poisson_deviance
                )
            ]

            finite_deviance_per_bin = (
                deviance_per_bin[
                    np.isfinite(
                        deviance_per_bin
                    )
                ]
            )

            mean_nll = (
                float(
                    np.mean(
                        finite_nll
                    )
                )
                if finite_nll.size
                else np.nan
            )

            median_nll = (
                float(
                    np.median(
                        finite_nll
                    )
                )
                if finite_nll.size
                else np.nan
            )

            mean_deviance = (
                float(
                    np.mean(
                        finite_deviance
                    )
                )
                if finite_deviance.size
                else np.nan
            )

            median_deviance = (
                float(
                    np.median(
                        finite_deviance
                    )
                )
                if finite_deviance.size
                else np.nan
            )

            mean_deviance_per_bin = (
                float(
                    np.mean(
                        finite_deviance_per_bin
                    )
                )
                if finite_deviance_per_bin.size
                else np.nan
            )

            median_deviance_per_bin = (
                float(
                    np.median(
                        finite_deviance_per_bin
                    )
                )
                if finite_deviance_per_bin.size
                else np.nan
            )

            mae_primary = float(
                np.mean(
                    np.abs(
                        error_primary
                    )
                )
            )

            bias_primary = float(
                np.mean(
                    error_primary
                )
            )

            mae_weighted = float(
                np.mean(
                    np.abs(
                        error_weighted
                    )
                )
            )

            bias_weighted = float(
                np.mean(
                    error_weighted
                )
            )

            mean_mixture_position = float(
                np.mean(
                    mixture_position
                )
            )

            median_mixture_position = float(
                np.median(
                    mixture_position
                )
            )

        else:
            mae_primary = np.nan
            bias_primary = np.nan

            mae_weighted = np.nan
            bias_weighted = np.nan

            mean_mixture_position = np.nan
            median_mixture_position = np.nan

            mean_nll = np.nan
            median_nll = np.nan

            mean_deviance = np.nan
            median_deviance = np.nan

            mean_deviance_per_bin = np.nan
            median_deviance_per_bin = np.nan

        rows.append(
            {
                "model_mismatch_severity": (
                    severity
                ),
                "secondary_fraction": (
                    secondary_fraction
                ),
                "n_total_samples": (
                    n_total
                ),
                "n_valid_fits": (
                    n_valid
                ),
                "failure_rate": (
                    float(
                        failure_rate
                    )
                ),
                "boundary_hit_rate": (
                    boundary_hit_rate
                ),
                "mae_to_primary_ns": (
                    mae_primary
                ),
                "bias_to_primary_ns": (
                    bias_primary
                ),
                "mae_to_signal_photon_weighted_ns": (
                    mae_weighted
                ),
                "bias_to_signal_photon_weighted_ns": (
                    bias_weighted
                ),
                "mean_effective_mixture_position": (
                    mean_mixture_position
                ),
                "median_effective_mixture_position": (
                    median_mixture_position
                ),
                "mean_poisson_nll": (
                    mean_nll
                ),
                "median_poisson_nll": (
                    median_nll
                ),
                "mean_poisson_deviance": (
                    mean_deviance
                ),
                "median_poisson_deviance": (
                    median_deviance
                ),
                "mean_poisson_deviance_per_bin": (
                    mean_deviance_per_bin
                ),
                "median_poisson_deviance_per_bin": (
                    median_deviance_per_bin
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_classical_model_mismatch_benchmark(
    *,
    tests: dict[
        str,
        GeneralizationTestMeasurements,
    ],
    irf_centre_ns: float,
    temporal_shift_bounds: tuple[
        float,
        float,
    ] = (-0.5, 0.5),
    objective: str = "poisson",
    background_fraction: float = 0.10,
) -> ClassicalModelMismatchBenchmarkResult:
    """Evaluate mono-exponential reconvolution on Tests A and F.

    Both tests are fitted using the correct per-curve IRF width.

    Therefore Test F isolates decay-model mismatch rather than
    instrument-response mismatch.
    """

    required_test_ids = {
        "A",
        "F",
    }

    missing_test_ids = (
        required_test_ids
        - set(tests)
    )

    if missing_test_ids:
        raise ValueError(
            "Missing required classical model-mismatch tests: "
            + ", ".join(
                sorted(missing_test_ids)
            )
        )

    for test_id in (
        "A",
        "F",
    ):
        if tests[
            test_id
        ].test_id != test_id:
            raise ValueError(
                "Dictionary key and test.test_id must match."
            )

    estimator_name = (
        "classical_reconvolution_mono_model"
    )

    prediction_tables: list[
        pd.DataFrame
    ] = []

    diagnostic_tables: list[
        pd.DataFrame
    ] = []

    for test_id in (
        "A",
        "F",
    ):
        test = tests[
            test_id
        ]

        diagnostics = (
            _evaluate_classical_generalization_test(
                test=test,
                irf_centre_ns=(
                    irf_centre_ns
                ),
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective=objective,
                background_fraction=(
                    background_fraction
                ),

                # Important:
                # use each curve's true IRF width.
                assumed_irf_fwhm_ns=None,

                irf_mode=(
                    "correct_test_irf"
                ),
            )
        )

        diagnostics = diagnostics.copy(
            deep=True
        )

        diagnostics[
            "classical_decay_model"
        ] = (
            "monoexponential_reconvolution"
        )

        diagnostics[
            "poisson_deviance_per_bin"
        ] = (
            diagnostics[
                "poisson_deviance"
            ].to_numpy(
                dtype=np.float64
            )
            / float(
                test.time.size
            )
        )

        diagnostic_tables.append(
            diagnostics
        )

        prediction_tables.append(
            _build_classical_generalization_prediction_table(
                test=test,
                diagnostics=diagnostics,
                estimator_name=(
                    estimator_name
                ),
            )
        )

    fit_diagnostics = pd.concat(
        diagnostic_tables,
        ignore_index=True,
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    summary = (
        summarize_generalization_predictions(
            predictions
        )
    )

    degradation = (
        build_reference_mae_degradation_table(
            summary,
            reference_test_id="A",
        )
    )

    paired_diagnostics = (
        build_classical_af_paired_diagnostics(
            fit_diagnostics
        )
    )

    severity_summary = (
        summarize_classical_test_f_model_mismatch(
            fit_diagnostics
        )
    )

    return (
        ClassicalModelMismatchBenchmarkResult(
            predictions=predictions,
            summary=summary,
            degradation=degradation,
            fit_diagnostics=(
                fit_diagnostics
            ),
            paired_diagnostics=(
                paired_diagnostics
            ),
            severity_summary=(
                severity_summary
            ),
        )
    )


def build_af_comparison_table(
    *,
    summary: pd.DataFrame,
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    """Build a side-by-side Test-A versus Test-F comparison."""

    required_summary_columns = {
        "estimator",
        "representation",
        "test_id",
        "mae_ns",
        "median_absolute_error_ns",
        "rmse_ns",
        "bias_ns",
        "p90_absolute_error_ns",
        "p95_absolute_error_ns",
        "classical_failure_rate",
    }

    missing_summary_columns = (
        required_summary_columns
        - set(summary.columns)
    )

    if missing_summary_columns:
        raise ValueError(
            "Summary table is missing required columns: "
            + ", ".join(
                sorted(
                    missing_summary_columns
                )
            )
        )

    required_degradation_columns = {
        "estimator",
        "representation",
        "reference_test_id",
        "ood_test_id",
        "mae_degradation",
    }

    missing_degradation_columns = (
        required_degradation_columns
        - set(degradation.columns)
    )

    if missing_degradation_columns:
        raise ValueError(
            "Degradation table is missing required columns: "
            + ", ".join(
                sorted(
                    missing_degradation_columns
                )
            )
        )

    if set(
        summary["test_id"]
    ) != {
        "A",
        "F",
    }:
        raise ValueError(
            "A/F comparison requires exactly Tests A and F."
        )

    if not (
        degradation[
            "reference_test_id"
        ] == "A"
    ).all():
        raise ValueError(
            "A/F degradation must use Test A as the reference."
        )

    if not (
        degradation[
            "ood_test_id"
        ] == "F"
    ).all():
        raise ValueError(
            "A/F degradation must use Test F as the OOD test."
        )

    test_a = summary.loc[
        summary[
            "test_id"
        ] == "A"
    ].copy()

    test_f = summary.loc[
        summary[
            "test_id"
        ] == "F"
    ].copy()

    join_columns = [
        "estimator",
        "representation",
    ]

    a_keys = set(
        map(
            tuple,
            test_a[
                join_columns
            ].to_numpy(),
        )
    )

    f_keys = set(
        map(
            tuple,
            test_f[
                join_columns
            ].to_numpy(),
        )
    )

    if a_keys != f_keys:
        raise ValueError(
            "Tests A and F must contain identical "
            "estimator/representation combinations."
        )

    metric_columns = {
        "mae_ns": "mae",
        "median_absolute_error_ns": (
            "median_absolute_error"
        ),
        "rmse_ns": "rmse",
        "bias_ns": "bias",
        "p90_absolute_error_ns": (
            "p90_absolute_error"
        ),
        "p95_absolute_error_ns": (
            "p95_absolute_error"
        ),
        "classical_failure_rate": (
            "classical_failure_rate"
        ),
    }

    test_a = test_a[
        join_columns
        + list(
            metric_columns
        )
    ]

    test_f = test_f[
        join_columns
        + list(
            metric_columns
        )
    ]

    test_a = test_a.rename(
        columns={
            column: (
                f"{base_name}_a"
            )
            for (
                column,
                base_name,
            ) in metric_columns.items()
        }
    )

    test_f = test_f.rename(
        columns={
            column: (
                f"{base_name}_f"
            )
            for (
                column,
                base_name,
            ) in metric_columns.items()
        }
    )

    comparison = test_a.merge(
        test_f,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    comparison = comparison.merge(
        degradation[
            [
                "estimator",
                "representation",
                "mae_degradation",
            ]
        ],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    return comparison


def build_day55_severity_comparison(
    *,
    nonclassical_summary: pd.DataFrame,
    classical_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine Test-F severity results across all estimators."""

    required_nonclassical_columns = {
        "estimator",
        "representation",
        "model_mismatch_severity",
        "secondary_fraction",
        "n_total_samples",
        "n_valid_predictions",
        "mae_to_primary_ns",
        "bias_to_primary_ns",
        "mae_to_signal_photon_weighted_ns",
        "bias_to_signal_photon_weighted_ns",
        "mean_effective_mixture_position",
        "median_effective_mixture_position",
    }

    missing_nonclassical = (
        required_nonclassical_columns
        - set(nonclassical_summary.columns)
    )

    if missing_nonclassical:
        raise ValueError(
            "Non-classical severity summary is missing columns: "
            + ", ".join(
                sorted(
                    missing_nonclassical
                )
            )
        )

    required_classical_columns = {
        "model_mismatch_severity",
        "secondary_fraction",
        "n_total_samples",
        "n_valid_fits",
        "mae_to_primary_ns",
        "bias_to_primary_ns",
        "mae_to_signal_photon_weighted_ns",
        "bias_to_signal_photon_weighted_ns",
        "mean_effective_mixture_position",
        "median_effective_mixture_position",
    }

    missing_classical = (
        required_classical_columns
        - set(classical_summary.columns)
    )

    if missing_classical:
        raise ValueError(
            "Classical severity summary is missing columns: "
            + ", ".join(
                sorted(
                    missing_classical
                )
            )
        )

    nonclassical = (
        nonclassical_summary.copy(
            deep=True
        )
    )

    nonclassical[
        "n_valid_estimates"
    ] = nonclassical[
        "n_valid_predictions"
    ].to_numpy(
        dtype=np.int64
    )

    nonclassical[
        "failure_rate"
    ] = (
        1.0
        - nonclassical[
            "n_valid_estimates"
        ].to_numpy(
            dtype=np.float64
        )
        / nonclassical[
            "n_total_samples"
        ].to_numpy(
            dtype=np.float64
        )
    )

    classical = (
        classical_summary.copy(
            deep=True
        )
    )

    classical.insert(
        0,
        "representation",
        "raw_histogram",
    )

    classical.insert(
        0,
        "estimator",
        "classical_reconvolution_mono_model",
    )

    classical[
        "n_valid_estimates"
    ] = classical[
        "n_valid_fits"
    ].to_numpy(
        dtype=np.int64
    )

    common_columns = [
        "estimator",
        "representation",
        "model_mismatch_severity",
        "secondary_fraction",
        "n_total_samples",
        "n_valid_estimates",
        "failure_rate",
        "mae_to_primary_ns",
        "bias_to_primary_ns",
        "mae_to_signal_photon_weighted_ns",
        "bias_to_signal_photon_weighted_ns",
        "mean_effective_mixture_position",
        "median_effective_mixture_position",
    ]

    combined = pd.concat(
        [
            nonclassical[
                common_columns
            ],
            classical[
                common_columns
            ],
        ],
        ignore_index=True,
    )

    return combined


def build_classical_model_mismatch_gof_summary(
    severity_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Extract classical goodness-of-fit reporting for Test F."""

    required_columns = {
        "model_mismatch_severity",
        "secondary_fraction",
        "n_total_samples",
        "n_valid_fits",
        "failure_rate",
        "boundary_hit_rate",
        "mean_poisson_nll",
        "median_poisson_nll",
        "mean_poisson_deviance",
        "median_poisson_deviance",
        "mean_poisson_deviance_per_bin",
        "median_poisson_deviance_per_bin",
    }

    missing_columns = (
        required_columns
        - set(severity_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical severity summary is missing GOF columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    return (
        severity_summary[
            [
                "model_mismatch_severity",
                "secondary_fraction",
                "n_total_samples",
                "n_valid_fits",
                "failure_rate",
                "boundary_hit_rate",
                "mean_poisson_nll",
                "median_poisson_nll",
                "mean_poisson_deviance",
                "median_poisson_deviance",
                "mean_poisson_deviance_per_bin",
                "median_poisson_deviance_per_bin",
            ]
        ]
        .copy(
            deep=True
        )
        .reset_index(
            drop=True
        )
    )


@dataclass(frozen=True)
class Day55ModelMismatchReport:
    """Reporting tables for the Day-55 Test-F experiment."""

    nonclassical_comparison: pd.DataFrame
    classical_comparison: pd.DataFrame

    severity_comparison: pd.DataFrame

    classical_gof_summary: pd.DataFrame
    classical_paired_diagnostics: pd.DataFrame


def build_day55_model_mismatch_report(
    *,
    nonclassical_result: ModelMismatchBenchmarkResult,
    classical_result: ClassicalModelMismatchBenchmarkResult,
) -> Day55ModelMismatchReport:
    """Build the final reporting tables for Day 55."""

    nonclassical_comparison = (
        build_af_comparison_table(
            summary=(
                nonclassical_result.summary
            ),
            degradation=(
                nonclassical_result.degradation
            ),
        )
    )

    classical_comparison = (
        build_af_comparison_table(
            summary=(
                classical_result.summary
            ),
            degradation=(
                classical_result.degradation
            ),
        )
    )

    severity_comparison = (
        build_day55_severity_comparison(
            nonclassical_summary=(
                nonclassical_result
                .severity_summary
            ),
            classical_summary=(
                classical_result
                .severity_summary
            ),
        )
    )

    classical_gof_summary = (
        build_classical_model_mismatch_gof_summary(
            classical_result
            .severity_summary
        )
    )

    classical_paired_diagnostics = (
        classical_result
        .paired_diagnostics
        .copy(
            deep=True
        )
    )

    return Day55ModelMismatchReport(
        nonclassical_comparison=(
            nonclassical_comparison
        ),
        classical_comparison=(
            classical_comparison
        ),
        severity_comparison=(
            severity_comparison
        ),
        classical_gof_summary=(
            classical_gof_summary
        ),
        classical_paired_diagnostics=(
            classical_paired_diagnostics
        ),
    )

