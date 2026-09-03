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


def fit_generalization_ml_estimators(
    prepared: GeneralizationABPreparedData,
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

        if (
            estimator_name
            == "classical_reconvolution"
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


