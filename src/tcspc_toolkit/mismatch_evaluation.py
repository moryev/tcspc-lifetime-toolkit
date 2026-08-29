"""Controlled model-mismatch evaluation for TCSPC lifetime estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult,
    evaluate_reconvolution_benchmark,
)
from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.features import extract_feature_table
from tcspc_toolkit.ml_evaluation import (
    BenchmarkDataset,
    BenchmarkSplit,
    RegressionBenchmarkResult,
    evaluate_regression,
)
from tcspc_toolkit.ml_models import (
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)
from tcspc_toolkit.simulation import (
    simulate_irf_convolved_biexponential_histogram,
)


@dataclass(frozen=True)
class EstimatorMismatchResult:
    """In-distribution and mismatched results for one ML estimator."""

    in_distribution: RegressionBenchmarkResult
    mismatch: RegressionBenchmarkResult


@dataclass(frozen=True)
class ClassicalMismatchResult:
    """In-distribution and mismatched reconvolution results."""

    in_distribution: ReconvolutionBenchmarkResult
    mismatch: ReconvolutionBenchmarkResult


def generate_matched_biexponential_mismatch_dataset(
    *,
    time: ArrayLike,
    reference_split: BenchmarkSplit,
    feature_config: FeatureConfig,
    irf_centre_ns: float,
    secondary_fraction: float = 0.10,
    secondary_lifetime_factor: float = 2.0,
    random_seed: int | None = 42,
) -> BenchmarkDataset:
    """Generate a bi-exponential test set matched to an existing test split.

    Each mismatch histogram preserves the primary lifetime, photon-count
    target, background level, IRF width, and IRF shift of the corresponding
    mono-exponential test sample.

    The only systematic change is the addition of a secondary exponential
    lifetime component.

    Parameters
    ----------
    time:
        Shared TCSPC time axis.
    reference_split:
        Original mono-exponential benchmark split.
    feature_config:
        Feature-extraction configuration used for the benchmark.
    irf_centre_ns:
        Centre of the unshifted Gaussian IRF.
    secondary_fraction:
        Fraction of expected signal photons assigned to the secondary
        lifetime component.
    secondary_lifetime_factor:
        Secondary lifetime expressed as a multiple of the primary lifetime.
        For example, ``2.0`` means ``tau_secondary = 2 * tau_primary``.
    random_seed:
        Seed controlling Poisson sampling.

    Returns
    -------
    BenchmarkDataset
        Matched bi-exponential histograms, engineered features, original
        primary-lifetime targets, and mismatch metadata.
    """
    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    if time_array.size != reference_split.X_histograms_test.shape[1]:
        raise ValueError(
            "time must match the histogram bin count."
        )

    if not np.isfinite(irf_centre_ns):
        raise ValueError(
            "irf_centre_ns must be finite."
        )

    if not np.isfinite(secondary_fraction):
        raise ValueError(
            "secondary_fraction must be finite."
        )

    if not (
        0.0
        <= secondary_fraction
        < 1.0
    ):
        raise ValueError(
            "secondary_fraction must lie in [0, 1)."
        )

    if not np.isfinite(
        secondary_lifetime_factor
    ):
        raise ValueError(
            "secondary_lifetime_factor must be finite."
        )

    if secondary_lifetime_factor <= 0.0:
        raise ValueError(
            "secondary_lifetime_factor must be positive."
        )

    required_metadata_columns = (
        "signal_photon_count_target",
        "background_per_bin",
        "irf_fwhm_ns",
        "irf_shift_ns",
    )

    missing_columns = [
        column
        for column in required_metadata_columns
        if column
        not in reference_split.metadata_test.columns
    ]

    if missing_columns:
        raise ValueError(
            "metadata_test is missing required columns: "
            + ", ".join(missing_columns)
        )

    n_samples = reference_split.y_test.size

    if reference_split.metadata_test.shape[0] != n_samples:
        raise ValueError(
            "metadata_test and y_test must contain "
            "the same number of samples."
        )

    X_histograms = np.empty(
        (
            n_samples,
            time_array.size,
        ),
        dtype=np.int64,
    )

    metadata_rows: list[
        dict[str, object]
    ] = []

    rng = np.random.default_rng(
        random_seed
    )

    for sample_index in range(
        n_samples
    ):
        primary_lifetime_ns = float(
            reference_split.y_test[
                sample_index
            ]
        )

        secondary_lifetime_ns = (
            secondary_lifetime_factor
            * primary_lifetime_ns
        )

        reference_metadata = (
            reference_split.metadata_test.iloc[
                sample_index
            ]
        )

        counts, simulation_metadata = (
            simulate_irf_convolved_biexponential_histogram(
                time=time_array,
                primary_lifetime_ns=primary_lifetime_ns,
                secondary_lifetime_ns=secondary_lifetime_ns,
                secondary_fraction=secondary_fraction,
                signal_photon_count=int(
                    reference_metadata[
                        "signal_photon_count_target"
                    ]
                ),
                background_per_bin=float(
                    reference_metadata[
                        "background_per_bin"
                    ]
                ),
                irf_centre_ns=irf_centre_ns,
                irf_fwhm_ns=float(
                    reference_metadata[
                        "irf_fwhm_ns"
                    ]
                ),
                irf_shift_ns=float(
                    reference_metadata[
                        "irf_shift_ns"
                    ]
                ),
                rng=rng,
            )
        )

        X_histograms[
            sample_index
        ] = counts

        metadata_rows.append(
            {
                **reference_metadata.to_dict(),
                **simulation_metadata,
                "mismatch_type": (
                    "biexponential_contamination"
                ),
                "secondary_lifetime_factor": float(
                    secondary_lifetime_factor
                ),
            }
        )

    X_features = extract_feature_table(
        histograms=X_histograms,
        time=time_array,
        config=feature_config,
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    return BenchmarkDataset(
        X_features=X_features,
        X_histograms=X_histograms,
        y=reference_split.y_test.copy(),
        metadata=metadata,
    )


def _build_regression_result(
    *,
    estimator_name: str,
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> RegressionBenchmarkResult:
    """Construct a regression result from already-generated predictions."""
    y_true_array = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if not np.all(
        np.isfinite(y_pred_array)
    ):
        raise ValueError(
            "Predictions must contain only finite values."
        )

    metrics = evaluate_regression(
        y_true=y_true_array,
        y_pred=y_pred_array,
    )

    relative_errors = (
        np.abs(
            y_pred_array
            - y_true_array
        )
        / y_true_array
    )

    return RegressionBenchmarkResult(
        estimator_name=estimator_name,
        y_pred=y_pred_array,
        relative_errors=relative_errors,
        metrics=metrics,
    )


def evaluate_ml_mismatch_benchmark(
    *,
    reference_split: BenchmarkSplit,
    mismatch_dataset: BenchmarkDataset,
) -> dict[
    str,
    EstimatorMismatchResult,
]:
    """Evaluate fitted ML estimators in- and out-of-distribution.

    Each estimator is fitted exactly once on the mono-exponential
    training data. The fitted estimator then predicts both the original
    mono-exponential test set and the matched bi-exponential test set.
    """
    if mismatch_dataset.y.shape != reference_split.y_test.shape:
        raise ValueError(
            "Mismatch targets must match reference test targets."
        )

    if not np.allclose(
        mismatch_dataset.y,
        reference_split.y_test,
    ):
        raise ValueError(
            "Mismatch targets must preserve the reference "
            "primary lifetimes."
        )

    if (
        mismatch_dataset.X_features.shape[1]
        != reference_split.X_features_test.shape[1]
    ):
        raise ValueError(
            "Mismatch and reference feature tables must "
            "contain the same number of features."
        )

    model_factories = {
        "ridge": make_ridge_pipeline,
        "random_forest": (
            make_random_forest_pipeline
        ),
        "hist_gradient_boosting": (
            make_hist_gradient_boosting_pipeline
        ),
    }

    results: dict[
        str,
        EstimatorMismatchResult,
    ] = {}

    for (
        estimator_name,
        make_estimator,
    ) in model_factories.items():
        estimator = make_estimator()

        estimator.fit(
            reference_split.X_features_train,
            reference_split.y_train,
        )

        in_distribution_predictions = (
            estimator.predict(
                reference_split.X_features_test
            )
        )

        mismatch_predictions = (
            estimator.predict(
                mismatch_dataset.X_features
            )
        )

        in_distribution_result = (
            _build_regression_result(
                estimator_name=estimator_name,
                y_true=reference_split.y_test,
                y_pred=in_distribution_predictions,
            )
        )

        mismatch_result = (
            _build_regression_result(
                estimator_name=estimator_name,
                y_true=mismatch_dataset.y,
                y_pred=mismatch_predictions,
            )
        )

        results[
            estimator_name
        ] = EstimatorMismatchResult(
            in_distribution=(
                in_distribution_result
            ),
            mismatch=mismatch_result,
        )

    return results


def evaluate_classical_mismatch_benchmark(
    *,
    time: ArrayLike,
    reference_split: BenchmarkSplit,
    mismatch_dataset: BenchmarkDataset,
    irf: ArrayLike,
    temporal_shift_bounds: tuple[
        float,
        float,
    ],
    objective: str = "poisson",
    background_fraction: float = 0.10,
) -> ClassicalMismatchResult:
    """Evaluate mono-exponential reconvolution before and after mismatch."""
    in_distribution_result = (
        evaluate_reconvolution_benchmark(
            time=time,
            X_histograms=(
                reference_split.X_histograms_test
            ),
            y_true=reference_split.y_test,
            metadata=reference_split.metadata_test,
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

    mismatch_result = (
        evaluate_reconvolution_benchmark(
            time=time,
            X_histograms=(
                mismatch_dataset.X_histograms
            ),
            y_true=mismatch_dataset.y,
            metadata=mismatch_dataset.metadata,
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

    return ClassicalMismatchResult(
        in_distribution=(
            in_distribution_result
        ),
        mismatch=mismatch_result,
    )


def summarize_mismatch_benchmark(
    *,
    y_true: ArrayLike,
    ml_results: dict[
        str,
        EstimatorMismatchResult,
    ],
    classical_result: (
        ClassicalMismatchResult
        | None
    ) = None,
) -> pd.DataFrame:
    """Summarize accuracy degradation caused by model mismatch."""
    y_true_array = np.asarray(
        y_true,
        dtype=np.float64,
    )

    summary_rows: list[
        dict[str, float | str]
    ] = []

    for (
        estimator_name,
        result,
    ) in ml_results.items():
        in_distribution_predictions = (
            result.in_distribution.y_pred
        )

        mismatch_predictions = (
            result.mismatch.y_pred
        )

        in_distribution_mae = (
            result.in_distribution.metrics.mae_ns
        )

        mismatch_mae = (
            result.mismatch.metrics.mae_ns
        )

        if in_distribution_mae > 0.0:
            mae_ratio = (
                mismatch_mae
                / in_distribution_mae
            )
        else:
            mae_ratio = np.inf

        summary_rows.append(
            {
                "estimator": estimator_name,
                "in_distribution_mae_ns": (
                    in_distribution_mae
                ),
                "mismatch_mae_ns": (
                    mismatch_mae
                ),
                "mae_change_ns": (
                    mismatch_mae
                    - in_distribution_mae
                ),
                "mae_ratio": float(
                    mae_ratio
                ),
                "in_distribution_bias_ns": float(
                    np.mean(
                        in_distribution_predictions
                        - y_true_array
                    )
                ),
                "mismatch_bias_ns": float(
                    np.mean(
                        mismatch_predictions
                        - y_true_array
                    )
                ),
                "in_distribution_failure_rate": 0.0,
                "mismatch_failure_rate": 0.0,
            }
        )

    if classical_result is not None:
        in_distribution_classical = (
            classical_result.in_distribution
        )

        mismatch_classical = (
            classical_result.mismatch
        )

        in_distribution_mae = (
            in_distribution_classical.summary.mae_valid_ns
        )

        mismatch_mae = (
            mismatch_classical.summary.mae_valid_ns
        )

        if (
            np.isfinite(in_distribution_mae)
            and in_distribution_mae > 0.0
        ):
            mae_ratio = (
                mismatch_mae
                / in_distribution_mae
            )
        else:
            mae_ratio = np.nan

        in_distribution_valid = (
            in_distribution_classical.per_curve[
                "valid_fit"
            ].to_numpy(
                dtype=bool
            )
        )

        mismatch_valid = (
            mismatch_classical.per_curve[
                "valid_fit"
            ].to_numpy(
                dtype=bool
            )
        )

        if np.any(
            in_distribution_valid
        ):
            in_distribution_bias = float(
                in_distribution_classical.per_curve.loc[
                    in_distribution_valid,
                    "error_ns",
                ].mean()
            )
        else:
            in_distribution_bias = np.nan

        if np.any(
            mismatch_valid
        ):
            mismatch_bias = float(
                mismatch_classical.per_curve.loc[
                    mismatch_valid,
                    "error_ns",
                ].mean()
            )
        else:
            mismatch_bias = np.nan

        summary_rows.append(
            {
                "estimator": "reconvolution",
                "in_distribution_mae_ns": (
                    in_distribution_mae
                ),
                "mismatch_mae_ns": (
                    mismatch_mae
                ),
                "mae_change_ns": (
                    mismatch_mae
                    - in_distribution_mae
                ),
                "mae_ratio": float(
                    mae_ratio
                ),
                "in_distribution_bias_ns": (
                    in_distribution_bias
                ),
                "mismatch_bias_ns": (
                    mismatch_bias
                ),
                "in_distribution_failure_rate": (
                    in_distribution_classical
                    .summary.failure_rate
                ),
                "mismatch_failure_rate": (
                    mismatch_classical
                    .summary.failure_rate
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )
