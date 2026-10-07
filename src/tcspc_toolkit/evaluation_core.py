"""Small factual builders and explicit-mask metric primitives.

Generalization baseline/ML execution consumes these facts through a legacy table
projection. Execution, preparation, grouped reporting and uncertainty stay outside
this module.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.evaluation_results import (
    EvaluationBatch, MethodDescriptor, PointEvaluationResult,
    _COMPARISON_COLUMNS, _POINT_COLUMNS, _POINT_KEY, _error_values, _float_vector, _mask,
)

__all__ = [
    "build_point_evaluation", "combine_point_evaluations", "calculate_reference_errors",
    "summarize_point_errors", "calculate_degradation_ratio",
]


def calculate_reference_errors(
    lifetime_estimates_ns: ArrayLike, reference_values_ns: ArrayLike, *,
    reference_available: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return signed, absolute and absolute-relative errors in positional order.

    Available references must be finite and positive. Missing references and
    nonfinite estimates yield NaN errors. Finite estimates, including negative
    values, are not clipped. Scientific validity is deliberately not a filter:
    callers select a metric population separately. Relative error is abs(error)
    divided by reference. Unavailable reference placeholders are ignored.
    """
    estimates = _float_vector(lifetime_estimates_ns, "lifetime_estimates_ns")
    references = _float_vector(reference_values_ns, "reference_values_ns")
    if references.shape != estimates.shape:
        raise ValueError("Estimates and references must be row-aligned.")
    available = _mask(reference_available, estimates.size, "reference_available")
    if np.any(available & (~np.isfinite(references) | (references <= 0))):
        raise ValueError("Available references must be finite and positive.")
    return _error_values(estimates, references, available)


def build_point_evaluation(
    *, batch: EvaluationBatch, method: MethodDescriptor,
    lifetime_estimates_ns: ArrayLike, is_valid: ArrayLike,
    failure_reasons: Sequence[str | None] | None = None,
) -> PointEvaluationResult:
    """Build facts for one method on one batch; never fit, predict or summarize.

    Validity is mandatory and supplied by the caller. Nonfinite outputs are
    allowed only when invalid; finite invalid estimates remain inspectable.
    Metadata stays in the batch, avoiding collisions with factual columns.
    A representation label identifies the source representation; building results
    does not require its matrix to remain resident in the batch.

    Point rows follow sample order. Comparisons follow reference mapping order,
    then sample order, including unavailable rows with NaN reference/error values.
    With no references the comparison table has its normal schema and zero rows.
    """
    estimates = _float_vector(lifetime_estimates_ns, "lifetime_estimates_ns")
    n = len(batch.sample_ids)
    if estimates.size != n:
        raise ValueError("Estimates must be row-aligned with the batch.")
    valid = _mask(is_valid, n, "is_valid")
    reasons = [None] * n if failure_reasons is None else list(failure_reasons)
    if isinstance(failure_reasons, (str, bytes)) or len(reasons) != n:
        raise ValueError("failure_reasons must be a row-aligned sequence.")
    points = pd.DataFrame({
        "evaluation_id": batch.evaluation_id, "sample_id": list(batch.sample_ids),
        "method_id": method.method_id, "method_family": method.family,
        "representation_id": method.representation_id,
        "lifetime_estimate_ns": estimates, "is_valid": valid, "failure_reason": reasons,
    }, columns=_POINT_COLUMNS)
    tables = []
    for reference in batch.references.values():
        errors = calculate_reference_errors(
            estimates, reference.values_ns, reference_available=reference.available,
        )
        table = points.loc[:, list(_POINT_KEY)].copy()
        table["reference_id"] = reference.reference_id
        table["reference_kind"] = reference.kind
        table["reference_version"] = reference.reference_version
        table["assumption_label"] = reference.assumption_label
        table["reference_available"] = reference.available
        table["reference_value_ns"] = np.where(reference.available, reference.values_ns, np.nan)
        for name, values in zip(("error_ns", "absolute_error_ns", "relative_error"), errors):
            table[name] = values
        table["is_valid_comparison"] = valid & reference.available & np.isfinite(errors).all(axis=0)
        tables.append(table)
    comparisons = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame({
        name: pd.Series(dtype=bool if name in ("reference_available", "is_valid_comparison")
                        else float if name in ("reference_value_ns", "error_ns", "absolute_error_ns", "relative_error")
                        else object)
        for name in _COMPARISON_COLUMNS
    })
    return PointEvaluationResult(points, comparisons)


def combine_point_evaluations(results: Sequence[PointEvaluationResult]) -> PointEvaluationResult:
    """Concatenate in supplied order, revalidating identities and linked facts.

    Repeated identities are rejected, never overwritten or silently deduplicated.
    The supplied result snapshots are unchanged; their public tables are copies.
    """
    if not results:
        raise ValueError("At least one point evaluation is required.")
    return PointEvaluationResult(
        pd.concat([r.points for r in results], ignore_index=True),
        pd.concat([r.reference_comparisons for r in results], ignore_index=True),
    )


def summarize_point_errors(
    errors_ns: ArrayLike, *, is_valid: ArrayLike,
    reference_available: ArrayLike, eligible: ArrayLike,
) -> dict[str, int | float]:
    """Summarize an explicitly selected population, with independent counts.

    One row means one attempted estimate against one selected reference; do not
    pool rows from several reference IDs as if they were independent observations.
    ``is_valid`` is the scientific/method validity fact, ``reference_available``
    is the reference fact, and ``eligible`` is the caller-selected metric population.
    Contributors must have available references and finite errors; unusable selected
    inputs raise rather than being discarded. ``eligible`` is NOT implicitly
    intersected with ``is_valid``: callers can deliberately choose valid-fit or
    finite-estimate populations. No failure/coverage rate is inferred. The four
    counts are attempted rows, caller-valid rows, available references, and selected
    contributors. Empty selections give NaN metrics. An attempted-population
    policy that propagates missing errors must be handled explicitly by a later
    wrapper, not by silently dropping values here. Percentiles use linear interpolation.
    """
    errors = _float_vector(errors_ns, "errors_ns")
    valid = _mask(is_valid, errors.size, "is_valid")
    available = _mask(reference_available, errors.size, "reference_available")
    selected = _mask(eligible, errors.size, "eligible")
    if np.any(selected & (~available | ~np.isfinite(errors))):
        raise ValueError("Eligible rows must have available references and finite errors.")
    values = errors[selected]
    absolute = np.abs(values)
    return {
        "n_attempted": int(errors.size), "n_valid": int(valid.sum()),
        "n_available": int(available.sum()), "n_contributing": int(selected.sum()),
        "mae_ns": float(np.mean(absolute)) if values.size else np.nan,
        "rmse_ns": float(np.sqrt(np.mean(values**2))) if values.size else np.nan,
        "bias_ns": float(np.mean(values)) if values.size else np.nan,
        "median_absolute_error_ns": float(np.median(absolute)) if values.size else np.nan,
        "p90_absolute_error_ns": float(np.percentile(absolute, 90, method="linear")) if values.size else np.nan,
        "p95_absolute_error_ns": float(np.percentile(absolute, 95, method="linear")) if values.size else np.nan,
    }


def calculate_degradation_ratio(
    *, reference_error: float, shifted_error: float, minimum_reference_error: float,
) -> float:
    """Return shifted/reference error, or NaN if undefined, with explicit tolerance.

    The tolerance must be finite and nonnegative. A denominator at or below it,
    negative error, nonfinite input or nonfinite quotient yields NaN. Zero shifted
    error over a usable reference yields zero. No legacy exception/infinity policy
    is changed, and this function does not choose reference populations.
    """
    if not np.isfinite(minimum_reference_error) or minimum_reference_error < 0:
        raise ValueError("minimum_reference_error must be finite and nonnegative.")
    if not np.isfinite(reference_error) or not np.isfinite(shifted_error) or (
        reference_error <= minimum_reference_error or shifted_error < 0
    ):
        return float("nan")
    ratio = float(shifted_error) / float(reference_error)
    return ratio if np.isfinite(ratio) else float("nan")
