"""Conditional performance analysis for TCSPC lifetime benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.evaluation import (
    calculate_lifetime_errors,
)
from tcspc_toolkit.classical_evaluation import (
    ReconvolutionBenchmarkResult,
)
from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import (
    EvaluationBatch, LifetimeReference, MethodDescriptor, PointEvaluationResult,
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
    method: MethodDescriptor | None = None,
    reference: LifetimeReference | None = None,
) -> pd.DataFrame:
    """Build aligned diagnostics, retaining finite errors on invalid rows.

    Supply both ``method`` and ``reference`` to use canonical point facts.
    The descriptor's ID replaces ``estimator_name`` in that path; its family
    and representation remain explicit in the internal facts, not extra columns.
    The reference must be fully available and exactly match ``y_true`` in order.
    No reference kind or method family is inferred from names or target values.

    Omitting both preserves the historical compatibility-only path, including
    empty inputs and caller-supplied validity masks. Explicit inputs obey the
    stricter core contract: nonempty batches and finite estimates when valid.
    Metadata is aligned positionally; its index (including duplicates) survives.
    Infinite estimates retain legacy infinite-error display in this array view,
    although their canonical comparisons have NaN errors and are invalid.
    Visible errors do not determine eligibility for valid-only summaries.
    """
    if (method is None) != (reference is None):
        raise ValueError("method and reference must be supplied together.")

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

    if method is not None and reference is not None:
        result = _build_conditional_point_evaluation(
            method=method, reference=reference, y_true=y_true_array,
            y_pred=y_pred_array, valid_mask=valid_array,
        )
        return _project_conditional_predictions(
            result=result, reference_id=reference.reference_id, metadata=metadata,
        )

    return _build_legacy_prediction_diagnostics(
        estimator_name=estimator_name, y_true_array=y_true_array,
        y_pred_array=y_pred_array, metadata=metadata, valid_array=valid_array,
    )


def _build_legacy_prediction_diagnostics(
    *, estimator_name: str, y_true_array: NDArray[np.float64],
    y_pred_array: NDArray[np.float64], metadata: pd.DataFrame,
    valid_array: NDArray[np.bool_],
) -> pd.DataFrame:
    """Historical array presentation with no invented method/reference semantics."""
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


def _build_conditional_point_evaluation(
    *, method: MethodDescriptor, reference: LifetimeReference,
    y_true: ArrayLike, y_pred: ArrayLike, valid_mask: ArrayLike,
    failure_reasons: Sequence[str | None] | None = None,
) -> PointEvaluationResult:
    """Adapt explicit semantics without treating metadata labels as identities.

    IDs are local positions within this single conditional invocation. Historical
    metadata (including repeated sample labels/indexes) stays in the projection.
    No representation matrix, fitting policy or synthetic model is inferred.
    """
    targets = np.asarray(y_true, dtype=np.float64)
    if (
        not np.array_equal(reference.values_ns, targets)
        or not reference.available.all()
    ):
        raise ValueError("reference must be fully available and match the target vector exactly.")
    batch = EvaluationBatch(
        evaluation_id="conditional", sample_ids=range(targets.size),
        references={reference.reference_id: reference},
    )
    return build_point_evaluation(
        batch=batch, method=method, lifetime_estimates_ns=y_pred,
        is_valid=valid_mask, failure_reasons=failure_reasons,
    )


def _project_conditional_predictions(
    *, result: PointEvaluationResult, reference_id: str, metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Project row-aligned facts without masking finite scientifically invalid errors.

    Metadata rows correspond positionally to point rows, not pandas index labels.
    Only the legacy array presentation restores infinite-estimate errors; canonical
    comparisons remain untouched. Classical diagnostic columns stay in per_curve.
    """
    points = result.points
    comparisons = result.reference_comparisons
    key = ["evaluation_id", "sample_id", "method_id", "representation_id"]
    rows = points.merge(
        comparisons.loc[comparisons.reference_id == reference_id,
                        key + ["reference_available", "reference_value_ns",
                               "error_ns", "absolute_error_ns", "relative_error"]],
        on=key, how="left", sort=False, validate="one_to_one",
    )
    if len(metadata) != len(rows) or not rows.reference_available.eq(True).all():
        raise ValueError("Conditional projection requires row-aligned metadata and fully referenced points.")
    # Compatibility-only display: the strict core leaves nonfinite errors absent.
    infinite = np.isinf(rows.lifetime_estimate_ns)
    rows.loc[infinite, "error_ns"] = rows.loc[infinite, "lifetime_estimate_ns"]
    rows.loc[infinite, ["absolute_error_ns", "relative_error"]] = np.inf
    table = metadata.copy(deep=True)
    table.insert(0, "estimator_name", rows.method_id.to_numpy())
    for name, source in (
        ("true_lifetime_ns", "reference_value_ns"),
        ("predicted_lifetime_ns", "lifetime_estimate_ns"),
        ("error_ns", "error_ns"), ("absolute_error_ns", "absolute_error_ns"),
        ("relative_error", "relative_error"), ("valid_estimate", "is_valid"),
    ):
        table[name] = rows[source].to_numpy()
    return table


def build_ml_prediction_diagnostics(
    *,
    split: BenchmarkSplit,
    result: RegressionBenchmarkResult,
    method: MethodDescriptor | None = None,
    reference: LifetimeReference | None = None,
) -> pd.DataFrame:
    """Adapt a regression result without fitting or inferring method/reference kind.

    ``RegressionBenchmarkResult`` also carries historical baseline outputs, so
    its name cannot establish family. Supply both ``method`` and ``reference``
    for factual evaluation, or omit both for exact historical table behavior.
    See ``build_prediction_diagnostics`` for alignment and presentation policies.
    """

    return build_prediction_diagnostics(
        estimator_name=(
            result.estimator_name
        ),
        y_true=split.y_test,
        y_pred=result.y_pred,
        metadata=split.metadata_test,
        method=method,
        reference=reference,
    )


def build_classical_prediction_diagnostics(
    result: ReconvolutionBenchmarkResult,
    *,
    estimator_name: str = "classical_reconvolution",
    reference: LifetimeReference | None = None,
) -> pd.DataFrame:
    """Standardize classical diagnostics without recomputing fits or acceptance.

    With an explicit, fully aligned ``reference``, adapt point facts using family
    ``classical`` and representation ``raw_histogram``. Without it, retain the
    historical table-only path: a column named true_lifetime_ns cannot establish
    reference kind. Original diagnostic/error columns remain authoritative in
    both paths, including finite-invalid visibility and nonfinite-error handling.
    """

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

    estimates = per_curve["fitted_lifetime_ns"]
    valid = per_curve["valid_fit"].astype(bool)
    if reference is not None:
        facts = _build_conditional_point_evaluation(
            method=MethodDescriptor(estimator_name, "classical", "raw_histogram"),
            reference=reference, y_true=per_curve["true_lifetime_ns"],
            y_pred=estimates, valid_mask=valid,
            failure_reasons=per_curve["failure_reason"].tolist() if "failure_reason" in per_curve else None,
        ).points
        estimates = pd.Series(
            facts.lifetime_estimate_ns.to_numpy(), index=per_curve.index,
            dtype=estimates.dtype,
        )
        valid = facts.is_valid.to_numpy()

    per_curve.insert(
        0,
        "estimator_name",
        estimator_name,
    )

    per_curve[
        "predicted_lifetime_ns"
    ] = estimates

    per_curve[
        "valid_estimate"
    ] = valid

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
