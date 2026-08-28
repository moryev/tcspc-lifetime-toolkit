"""Conditional performance analysis for TCSPC lifetime benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tcspc_toolkit.evaluation import (
    calculate_lifetime_errors,
)
from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult,
)
from tcspc_toolkit.ml_evaluation import (
    BenchmarkSplit,
    RegressionBenchmarkResult,
)


def build_prediction_diagnostics(
    *,
    estimator_name: str,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metadata: pd.DataFrame,
    valid_mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """Build aligned per-sample lifetime-estimation diagnostics."""

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

    if y_pred_array.shape != y_true_array.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    n_samples = y_true_array.size

    if metadata.shape[0] != n_samples:
        raise ValueError(
            "metadata must contain one row per sample."
        )

    if not np.all(
        np.isfinite(y_true_array)
    ):
        raise ValueError(
            "y_true must contain only finite values."
        )

    if np.any(
        y_true_array <= 0.0
    ):
        raise ValueError(
            "True lifetimes must be strictly positive."
        )

    if valid_mask is None:
        valid_array = np.isfinite(
            y_pred_array
        )

    else:
        valid_array = np.asarray(
            valid_mask,
            dtype=bool,
        )

        if valid_array.shape != (
            n_samples,
        ):
            raise ValueError(
                "valid_mask must contain one value per sample."
            )

    (
        error_ns,
        absolute_error_ns,
        relative_error,
    ) = calculate_lifetime_errors(
        true_lifetimes=y_true_array,
        estimated_lifetimes=y_pred_array,
    )

    diagnostics = metadata.copy(
        deep=True
    )

    diagnostics.insert(
        0,
        "estimator_name",
        estimator_name,
    )

    diagnostics["true_lifetime_ns"] = (
        y_true_array
    )

    diagnostics["predicted_lifetime_ns"] = (
        y_pred_array
    )

    diagnostics["error_ns"] = (
        error_ns
    )

    diagnostics["absolute_error_ns"] = (
        absolute_error_ns
    )

    diagnostics["relative_error"] = (
        relative_error
    )

    diagnostics["valid_estimate"] = (
        valid_array
    )

    return diagnostics


def build_ml_prediction_diagnostics(
    *,
    split: BenchmarkSplit,
    result: RegressionBenchmarkResult,
) -> pd.DataFrame:
    """Build per-sample diagnostics for one ML benchmark result."""

    return build_prediction_diagnostics(
        estimator_name=(
            result.estimator_name
        ),
        y_true=split.y_test,
        y_pred=result.y_pred,
        metadata=split.metadata_test,
    )


def build_classical_prediction_diagnostics(
    result: ReconvolutionBenchmarkResult,
    *,
    estimator_name: str = "classical_reconvolution",
) -> pd.DataFrame:
    """Standardize classical reconvolution results for conditional analysis."""

    per_curve = result.per_curve.copy(
        deep=True
    )

    required_columns = {
        "true_lifetime_ns",
        "fitted_lifetime_ns",
        "valid_fit",
    }

    missing_columns = (
        required_columns
        - set(per_curve.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical benchmark results are missing "
            "required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    per_curve.insert(
        0,
        "estimator_name",
        estimator_name,
    )

    per_curve[
        "predicted_lifetime_ns"
    ] = per_curve[
        "fitted_lifetime_ns"
    ]

    per_curve[
        "valid_estimate"
    ] = per_curve[
        "valid_fit"
    ].astype(bool)

    return per_curve


def summarize_conditional_performance(
    diagnostics: pd.DataFrame,
    *,
    condition_column: str,
) -> pd.DataFrame:
    """Summarize estimator performance within benchmark conditions.

    Error metrics are calculated only for valid estimates.
    Failure rate is calculated over all samples in each condition.

    Parameters
    ----------
    diagnostics:
        Per-sample prediction diagnostics containing estimator names,
        signed errors, absolute errors, validity flags, and benchmark
        metadata.
    condition_column:
        Column used to define the evaluation groups.

    Returns
    -------
    pandas.DataFrame
        One row per estimator and condition containing sample counts,
        failure rate, MAE, median absolute error, bias, and the
        90th and 95th percentiles of absolute error.
    """

    required_columns = {
        "estimator_name",
        "error_ns",
        "absolute_error_ns",
        "valid_estimate",
        condition_column,
    }

    missing_columns = (
        required_columns
        - set(diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Diagnostics are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if diagnostics.empty:
        raise ValueError(
            "diagnostics must contain at least one sample."
        )

    if diagnostics[
        condition_column
    ].isna().any():
        raise ValueError(
            f"{condition_column} must not contain missing values."
        )

    if diagnostics[
        "valid_estimate"
    ].isna().any():
        raise ValueError(
            "valid_estimate must not contain missing values."
        )

    summary_rows: list[
        dict[str, object]
    ] = []

    grouped = diagnostics.groupby(
        [
            "estimator_name",
            condition_column,
        ],
        sort=False,
        observed=True,
    )

    for (
        estimator_name,
        condition_value,
    ), group in grouped:

        valid_mask = group[
            "valid_estimate"
        ].to_numpy(
            dtype=bool
        )

        n_samples = group.shape[0]

        n_valid_estimates = int(
            np.sum(
                valid_mask
            )
        )

        n_failed_estimates = (
            n_samples
            - n_valid_estimates
        )

        failure_rate = (
            n_failed_estimates
            / n_samples
        )

        valid_errors = group.loc[
            valid_mask,
            "error_ns",
        ].to_numpy(
            dtype=np.float64
        )

        valid_absolute_errors = group.loc[
            valid_mask,
            "absolute_error_ns",
        ].to_numpy(
            dtype=np.float64
        )

        if (
            not np.all(
                np.isfinite(
                    valid_errors
                )
            )
            or not np.all(
                np.isfinite(
                    valid_absolute_errors
                )
            )
        ):
            raise ValueError(
                "Valid estimates must have finite errors."
            )

        if n_valid_estimates > 0:
            mae_ns = float(
                np.mean(
                    valid_absolute_errors
                )
            )

            median_absolute_error_ns = float(
                np.median(
                    valid_absolute_errors
                )
            )

            bias_ns = float(
                np.mean(
                    valid_errors
                )
            )

            p90_absolute_error_ns = float(
                np.quantile(
                    valid_absolute_errors,
                    0.90,
                )
            )

            p95_absolute_error_ns = float(
                np.quantile(
                    valid_absolute_errors,
                    0.95,
                )
            )

        else:
            mae_ns = np.nan
            median_absolute_error_ns = np.nan
            bias_ns = np.nan
            p90_absolute_error_ns = np.nan
            p95_absolute_error_ns = np.nan

        summary_rows.append(
            {
                "estimator_name": (
                    estimator_name
                ),
                condition_column: (
                    condition_value
                ),
                "n_samples": (
                    n_samples
                ),
                "n_valid_estimates": (
                    n_valid_estimates
                ),
                "n_failed_estimates": (
                    n_failed_estimates
                ),
                "failure_rate": float(
                    failure_rate
                ),
                "mae_ns": (
                    mae_ns
                ),
                "median_absolute_error_ns": (
                    median_absolute_error_ns
                ),
                "bias_ns": (
                    bias_ns
                ),
                "p90_absolute_error_ns": (
                    p90_absolute_error_ns
                ),
                "p95_absolute_error_ns": (
                    p95_absolute_error_ns
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def assign_numeric_regimes(
    values: ArrayLike,
    *,
    bin_edges: ArrayLike,
    labels: tuple[str, ...],
) -> pd.Categorical:
    """Assign numeric values to ordered benchmark regimes.

    Parameters
    ----------
    values:
        One-dimensional numeric values to classify.
    bin_edges:
        Strictly increasing bin boundaries. For ``n`` regimes,
        ``n + 1`` edges are required.
    labels:
        Ordered regime labels.

    Returns
    -------
    pandas.Categorical
        Ordered categorical regime labels.
    """

    values_array = np.asarray(
        values,
        dtype=np.float64,
    )

    edges_array = np.asarray(
        bin_edges,
        dtype=np.float64,
    )

    if values_array.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if edges_array.ndim != 1:
        raise ValueError(
            "bin_edges must be one-dimensional."
        )

    if edges_array.size < 2:
        raise ValueError(
            "bin_edges must contain at least two values."
        )

    if len(labels) != (
        edges_array.size - 1
    ):
        raise ValueError(
            "Number of labels must equal "
            "number of bins."
        )

    if not np.all(
        np.isfinite(values_array)
    ):
        raise ValueError(
            "values must contain only finite values."
        )

    if np.any(
        np.isnan(edges_array)
    ):
        raise ValueError(
            "bin_edges must not contain NaN."
        )

    if not np.all(
        np.diff(edges_array) > 0.0
    ):
        raise ValueError(
            "bin_edges must be strictly increasing."
        )

    regimes = pd.cut(
        values_array,
        bins=edges_array,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    )

    if regimes.isna().any():
        raise ValueError(
            "Some values fall outside the supplied bin edges."
        )

    return regimes


def add_benchmark_regimes(
    diagnostics: pd.DataFrame,
    *,
    lifetime_edges: ArrayLike,
    photon_count_edges: ArrayLike,
    background_edges: ArrayLike,
    irf_width_edges: ArrayLike,
    irf_misalignment_edges: ArrayLike,
    irf_misalignment_labels: tuple[str, ...] = (
        "small",
        "medium",
        "large",
    ),
) -> pd.DataFrame:
    """Add standard TCSPC benchmark regime labels.

    The original continuous benchmark variables are preserved.
    IRF misalignment is classified using the magnitude of the
    temporal shift while the original signed shift is retained.
    """

    required_columns = {
        "true_lifetime_ns",
        "signal_photon_count_target",
        "background_per_bin",
        "irf_fwhm_ns",
        "irf_shift_ns",
    }

    missing_columns = (
        required_columns
        - set(diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Diagnostics are missing required benchmark columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    result = diagnostics.copy(
        deep=True
    )

    result[
        "lifetime_regime"
    ] = assign_numeric_regimes(
        result[
            "true_lifetime_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        bin_edges=lifetime_edges,
        labels=(
            "short",
            "medium",
            "long",
        ),
    )

    result[
        "photon_count_regime"
    ] = assign_numeric_regimes(
        result[
            "signal_photon_count_target"
        ].to_numpy(
            dtype=np.float64
        ),
        bin_edges=photon_count_edges,
        labels=(
            "low",
            "medium",
            "high",
        ),
    )

    result[
        "background_regime"
    ] = assign_numeric_regimes(
        result[
            "background_per_bin"
        ].to_numpy(
            dtype=np.float64
        ),
        bin_edges=background_edges,
        labels=(
            "low",
            "medium",
            "high",
        ),
    )

    result[
        "irf_width_regime"
    ] = assign_numeric_regimes(
        result[
            "irf_fwhm_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        bin_edges=irf_width_edges,
        labels=(
            "narrow",
            "medium",
            "broad",
        ),
    )

    absolute_irf_shift_ns = np.abs(
        result[
            "irf_shift_ns"
        ].to_numpy(
            dtype=np.float64
        )
    )

    result[
        "absolute_irf_shift_ns"
    ] = absolute_irf_shift_ns

    result[
        "irf_misalignment_regime"
    ] = assign_numeric_regimes(
        absolute_irf_shift_ns,
        bin_edges=irf_misalignment_edges,
        labels=irf_misalignment_labels,
    )

    return result


def summarize_standard_regimes(
    diagnostics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Summarize performance across standard TCSPC benchmark regimes.

    Parameters
    ----------
    diagnostics:
        Prediction diagnostics containing the standard regime columns
        produced by ``add_benchmark_regimes``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Conditional performance summaries keyed by physical condition.
    """

    regime_columns = {
        "lifetime": "lifetime_regime",
        "photon_count": "photon_count_regime",
        "background": "background_regime",
        "irf_width": "irf_width_regime",
        "irf_misalignment": (
            "irf_misalignment_regime"
        ),
    }

    missing_columns = {
        column
        for column in regime_columns.values()
        if column not in diagnostics.columns
    }

    if missing_columns:
        raise ValueError(
            "Diagnostics are missing standard regime columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return {
        regime_name: (
            summarize_conditional_performance(
                diagnostics,
                condition_column=condition_column,
            )
        )
        for (
            regime_name,
            condition_column,
        ) in regime_columns.items()
    }
