"""Factual evaluation contracts, independent of inference and benchmark policy.

Matrices remain caller-owned. Small reference arrays are copied and read-only.
Canonical tables are private snapshots exposed through defensive-copy properties;
frozen dataclasses alone do not make NumPy or pandas contents immutable.
No training, summary, uncertainty, persistence, or frozen-protocol policy lives
here. See ``docs/evaluation.md`` for identity and reference semantics.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


__all__ = ["MethodDescriptor", "LifetimeReference", "EvaluationBatch", "PointEvaluationResult"]

_FAMILIES = ("baseline", "ml", "classical", "bayesian")
_REFERENCE_KINDS = (
    "generating_mono", "primary_component", "trusted_experimental", "pseudo_true_mono",
)
_POINT_KEY = ("evaluation_id", "sample_id", "method_id", "representation_id")
_POINT_COLUMNS = (
    "evaluation_id", "sample_id", "method_id", "method_family", "representation_id",
    "lifetime_estimate_ns", "is_valid", "failure_reason",
)
_COMPARISON_COLUMNS = (
    *_POINT_KEY, "reference_id", "reference_kind", "reference_version", "assumption_label",
    "reference_available", "reference_value_ns", "error_ns", "absolute_error_ns",
    "relative_error", "is_valid_comparison",
)


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string.")


def _sample_ids(values: Sequence[str | int]) -> tuple[str | int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("sample_ids must be an ordered sequence of identifiers.")
    result = tuple(values)
    for value in result:
        if isinstance(value, str):
            _identifier(value, "sample_id")
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("sample_id must be a string or integer, not a boolean.")
    return result


def _float_vector(values: ArrayLike, label: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    return result


def _mask(values: ArrayLike, size: int, label: str) -> NDArray[np.bool_]:
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype.kind != "b":
        raise ValueError(f"{label} must be a row-aligned boolean vector.")
    return result


def _reference_scope(kind: str, version: str | None, assumption_label: str | None) -> None:
    if kind not in _REFERENCE_KINDS:
        raise ValueError("Unsupported reference kind.")
    if version is not None:
        _identifier(version, "reference_version")
    if kind == "pseudo_true_mono" or assumption_label is not None:
        _identifier(assumption_label, "assumption_label")


@dataclass(frozen=True)
class MethodDescriptor:
    """An opaque method ID and explicit family, never inferred from its name.

    ``representation_id=None`` means not applicable. The pair of method and
    representation IDs identifies a method variant within a result collection.
    One method ID must retain the same family across all representations/batches.
    Different inference/summary variants must use distinct caller-chosen IDs.
    Configuration and scientific assumptions remain with domain-specific objects.
    """

    method_id: str
    family: Literal["baseline", "ml", "classical", "bayesian"]
    representation_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.method_id, "method_id")
        if self.family not in _FAMILIES:
            raise ValueError("Unsupported method family.")
        if self.representation_id is not None:
            _identifier(self.representation_id, "representation_id")


@dataclass(frozen=True, init=False)
class LifetimeReference:
    """One explicitly identified, positionally aligned lifetime reference.

    Available values must be positive and finite. Unavailable numeric values
    are retained in this carrier but never used or exposed as comparisons.
    ``assumption_label`` describes runtime scope: required for a pseudo-true
    projection, optional otherwise. It is not a database key, full provenance,
    or a claim that assumptions were checked. Versions are optional; unknown
    versions are not invented. Arrays are copied and read-only; inputs may be
    array-like. Deliberately bypassing NumPy's write protection is unsupported.
    """

    reference_id: str
    kind: Literal["generating_mono", "primary_component", "trusted_experimental", "pseudo_true_mono"]
    values_ns: NDArray[np.float64]
    available: NDArray[np.bool_]
    reference_version: str | None
    assumption_label: str | None

    def __init__(
        self, reference_id: str,
        kind: Literal["generating_mono", "primary_component", "trusted_experimental", "pseudo_true_mono"],
        values_ns: ArrayLike, available: ArrayLike, *,
        reference_version: str | None = None, assumption_label: str | None = None,
    ) -> None:
        _identifier(reference_id, "reference_id")
        _reference_scope(kind, reference_version, assumption_label)
        values = _float_vector(values_ns, "values_ns").copy()
        mask = _mask(available, values.size, "available").copy()
        if np.any(mask & (~np.isfinite(values) | (values <= 0))):
            raise ValueError("Available references must be finite and positive.")
        values.setflags(write=False)
        mask.setflags(write=False)
        for name, value in (
            ("reference_id", reference_id), ("kind", kind), ("values_ns", values),
            ("available", mask), ("reference_version", reference_version),
            ("assumption_label", assumption_label),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, init=False)
class EvaluationBatch:
    """Already-prepared observations; no training or representation policy.

    IDs are preserved verbatim and unique within the evaluation. Metadata,
    references, and matrix-like representations share positional row order.
    Counts can be validated, not whether a caller reordered matrix contents.
    A metadata index is not a sample identity; optional sample_id/evaluation_id
    columns must agree with the explicit IDs. Metadata is copied at construction
    and access; nested user objects in metadata cells are not recursively copied.
    Mapping shells are copied/read-only, but matrices pass through unchanged.
    Empty representation/reference mappings are valid. Batches are nonempty.
    """

    evaluation_id: str
    sample_ids: tuple[str | int, ...]
    representations: Mapping[str, Any]
    references: Mapping[str, LifetimeReference]
    _metadata: pd.DataFrame = field(repr=False)

    def __init__(
        self, evaluation_id: str, sample_ids: Sequence[str | int],
        representations: Mapping[str, Any] | None = None,
        metadata: pd.DataFrame | None = None,
        references: Mapping[str, LifetimeReference] | None = None,
    ) -> None:
        _identifier(evaluation_id, "evaluation_id")
        ids = _sample_ids(sample_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("sample_ids must be nonempty and unique within a batch.")
        representations = {} if representations is None else dict(representations)
        references = {} if references is None else dict(references)
        for name, matrix in representations.items():
            _identifier(name, "representation_id")
            shape = getattr(matrix, "shape", ())
            if not shape or isinstance(shape[0], (bool, np.bool_)) or (
                not isinstance(shape[0], (int, np.integer)) or shape[0] != len(ids)
            ):
                raise ValueError("Representations must expose a row-aligned shape[0].")
        if metadata is None:
            metadata = pd.DataFrame(index=range(len(ids)))
        if not isinstance(metadata, pd.DataFrame) or len(metadata) != len(ids):
            raise ValueError("metadata must be a row-aligned DataFrame.")
        if not metadata.columns.is_unique:
            raise ValueError("metadata column names must be unique.")
        if "sample_id" in metadata and tuple(metadata.sample_id) != ids:
            raise ValueError("metadata sample_id order differs from sample_ids.")
        if "evaluation_id" in metadata and not metadata.evaluation_id.eq(evaluation_id).all():
            raise ValueError("metadata evaluation_id conflicts with the batch.")
        for name, reference in references.items():
            if not isinstance(reference, LifetimeReference) or name != reference.reference_id:
                raise ValueError("Reference mapping keys must match LifetimeReference IDs.")
            if reference.values_ns.size != len(ids):
                raise ValueError("References must be row-aligned with sample_ids.")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "_metadata", metadata.copy(deep=True))
        object.__setattr__(self, "representations", MappingProxyType(representations))
        object.__setattr__(self, "references", MappingProxyType(references))

    @property
    def metadata(self) -> pd.DataFrame:
        """Return an editable table copy, not the batch's alignment snapshot."""
        return self._metadata.copy(deep=True)


def _error_values(
    estimates: NDArray[np.float64], references: NDArray[np.float64],
    available: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Arithmetic shared by factual validation and the public error primitive."""
    error = np.full(estimates.size, np.nan, dtype=np.float64)
    computable = available & np.isfinite(estimates)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error[computable] = estimates[computable] - references[computable]
        absolute = np.abs(error)
        relative = np.full(estimates.size, np.nan, dtype=np.float64)
        relative[computable] = absolute[computable] / references[computable]
    return error, absolute, relative


def _table(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or set(frame) != set(columns):
        raise ValueError(f"Table must have exactly these columns: {columns}.")
    result = frame.loc[:, list(columns)].copy(deep=True).reset_index(drop=True)
    # pandas string inference may turn None into NaN/pd.NA. Normalize nullable
    # labels before using them as dictionary keys or validating scope fields.
    for name in ("representation_id", "failure_reason", "reference_version", "assumption_label"):
        if name in result:
            values = result[name].astype(object)
            result[name] = values.where(values.notna(), None)
    for name in ("lifetime_estimate_ns", "reference_value_ns", "error_ns", "absolute_error_ns", "relative_error"):
        if name in result:
            result[name] = _float_vector(result[name], name)
    return result


@dataclass(frozen=True, init=False)
class PointEvaluationResult:
    """Validated point facts and independent reference comparisons, not summaries.

    Point identity is (evaluation_id, sample_id, method_id, representation_id).
    Family cannot vary for a method ID, even across representations/evaluations.
    Comparison identity adds reference_id; a reference ID has one kind/scope
    throughout the result, with aligned values per evaluation/sample. Null
    representation IDs are real key components.

    Invalid finite estimates and nonfinite failed outputs are preserved. Valid
    estimates must be finite. Errors remain visible for finite invalid estimates;
    is_valid_comparison additionally requires point validity and an available,
    numerically usable reference. Tables are private validated snapshots; public
    properties return editable copies. Construct a new result from edited copies
    to validate changes. No deeply immutable pandas implementation is implied.
    """

    _points: pd.DataFrame = field(repr=False)
    _reference_comparisons: pd.DataFrame = field(repr=False)

    def __init__(self, points: pd.DataFrame, reference_comparisons: pd.DataFrame) -> None:
        points = _table(points, _POINT_COLUMNS)
        comparisons = _table(reference_comparisons, _COMPARISON_COLUMNS)
        if points.duplicated(list(_POINT_KEY)).any():
            raise ValueError("Duplicate point identity.")
        if comparisons.duplicated([*_POINT_KEY, "reference_id"]).any():
            raise ValueError("Duplicate reference-comparison identity.")
        estimates = _float_vector(points.lifetime_estimate_ns, "lifetime_estimate_ns")
        valid = _mask(points.is_valid, len(points), "is_valid")
        if np.any(valid & ~np.isfinite(estimates)):
            raise ValueError("Valid estimates must be finite.")
        point_lookup, families = {}, {}
        for row in points.itertuples(index=False):
            _identifier(row.evaluation_id, "evaluation_id")
            _sample_ids((row.sample_id,))
            method = MethodDescriptor(row.method_id, row.method_family, row.representation_id)
            if families.setdefault(method.method_id, method.family) != method.family:
                raise ValueError("Conflicting method family for one method_id.")
            if row.failure_reason is not None:
                _identifier(row.failure_reason, "failure_reason")
                if row.is_valid:
                    raise ValueError("Valid estimates cannot have a failure_reason.")
            point_lookup[tuple(getattr(row, key) for key in _POINT_KEY)] = row
        reference_facts, definitions = {}, {}
        _mask(comparisons.reference_available, len(comparisons), "reference_available")
        _mask(comparisons.is_valid_comparison, len(comparisons), "is_valid_comparison")
        for row in comparisons.itertuples(index=False):
            _sample_ids((row.sample_id,))
            key = tuple(getattr(row, name) for name in _POINT_KEY)
            if key not in point_lookup:
                raise ValueError("Reference comparison has no matching point.")
            _identifier(row.reference_id, "reference_id")
            _reference_scope(row.reference_kind, row.reference_version, row.assumption_label)
            definition = (row.reference_kind, row.reference_version, row.assumption_label)
            if definitions.setdefault(row.reference_id, definition) != definition:
                raise ValueError("Conflicting reference definition.")
            value = float(row.reference_value_ns)
            if row.reference_available:
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("Available reference values must be finite and positive.")
            elif not np.isnan(value):
                raise ValueError("Unavailable comparison references must be NaN.")
            fact = (row.reference_available, value if row.reference_available else None)
            ref_sample_key = (row.evaluation_id, row.reference_id, row.sample_id)
            if reference_facts.setdefault(ref_sample_key, fact) != fact:
                raise ValueError("Conflicting reference values for one sample.")
            point = point_lookup[key]
            expected = _error_values(
                np.array([point.lifetime_estimate_ns]), np.array([value]),
                np.array([row.reference_available]),
            )
            actual = np.array([row.error_ns, row.absolute_error_ns, row.relative_error], dtype=float)
            if not np.array_equal(actual, np.array(expected).ravel(), equal_nan=True):
                raise ValueError("Comparison errors conflict with the point/reference facts.")
            expected_valid = point.is_valid and row.reference_available and np.isfinite(actual).all()
            if row.is_valid_comparison != expected_valid:
                raise ValueError("is_valid_comparison conflicts with point/reference validity.")
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_reference_comparisons", comparisons)

    @property
    def points(self) -> pd.DataFrame:
        """Return an editable copy of the validated point facts."""
        return self._points.copy(deep=True)

    @property
    def reference_comparisons(self) -> pd.DataFrame:
        """Return an editable copy of the validated reference comparisons."""
        return self._reference_comparisons.copy(deep=True)
