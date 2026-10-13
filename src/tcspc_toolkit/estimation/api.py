"""Small, estimator-independent execution boundary for lifetime regression.

Representations are prepared by the caller. This module does not select
benchmarks, fit representations, split data, calibrate uncertainty, or compute
evaluation metrics. It has no dependency on sklearn or canonical model names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = [
    "RegressorProtocol", "EstimatorSpec", "fit_regressors", "predict_regressors",
]


class RegressorProtocol(Protocol):
    """Structural point-regression interface; no sklearn inheritance required.

    ``fit`` must update the instance in place; its return value is ignored.
    ``predict`` must return one finite scalar per input row. Adapters may own
    other training machinery internally; no cloning or ``get_params`` is used.
    Execution helpers pass through matrix-like ``X`` objects with a row axis
    in ``shape[0]``; ``Any`` does not mean arbitrary Python objects are accepted.
    """

    def fit(self, X: Any, y: NDArray[np.float64], /) -> object:
        """Fit using only the supplied training rows and lifetime targets."""
        ...

    def predict(self, X: Any, /) -> ArrayLike:
        """Predict lifetimes in nanoseconds, in input-row order."""
        ...


@dataclass(frozen=True)
class EstimatorSpec:
    """An estimator identifier, fresh-instance factory, and representation IDs.

    Names are arbitrary nonblank strings, preserved verbatim. ``representations``
    is a nonempty tuple of unique names selecting caller-supplied inputs, not
    instructions to construct or fit transforms. The zero-argument ``factory``
    must return a fresh, unfitted regressor on every call. The specification is
    immutable; it does not freeze state captured by a user-supplied callable.
    """

    name: str
    factory: Callable[[], RegressorProtocol]
    representations: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "Estimator name")
        if not callable(self.factory):
            raise TypeError("factory must be callable without arguments.")
        if not isinstance(self.representations, tuple):
            raise TypeError("representations must be an immutable tuple.")
        if not self.representations:
            raise ValueError("An estimator must select at least one representation.")
        for name in self.representations:
            _validate_identifier(name, "Representation name")
        if len(set(self.representations)) != len(self.representations):
            raise ValueError(
                "Representation names must be unique within a specification."
            )


def _validate_identifier(name: str, label: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"{label} must be a string.")
    if not name.strip():
        raise ValueError(f"{label} must not be blank.")


def _validate_representation_rows(
    representations: Mapping[str, Any],
    required: Sequence[str],
    expected_rows: int | None = None,
) -> int:
    """Check positional row counts without coercing representation objects."""
    for name in dict.fromkeys(required):
        if name not in representations:
            raise ValueError(f"Missing representation: {name!r}.")
        shape = getattr(representations[name], "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError(
                f"Representation {name!r} must expose a sample axis in shape."
            )
        n_rows = shape[0]
        if (
            isinstance(n_rows, (bool, np.bool_))
            or not isinstance(n_rows, (int, np.integer))
            or n_rows < 1
        ):
            raise ValueError(
                f"Representation {name!r} must contain at least one sample."
            )
        if expected_rows is None:
            expected_rows = int(n_rows)
        elif n_rows != expected_rows:
            raise ValueError(
                f"Representation {name!r} must contain {expected_rows} samples; "
                "all selected representations and targets must be row-aligned."
            )
    if expected_rows is None:
        raise ValueError("At least one representation must be selected.")
    return expected_rows


def fit_regressors(
    *,
    estimator_specs: Sequence[EstimatorSpec],
    X_train_by_representation: Mapping[str, Any],
    y_train: ArrayLike,
) -> dict[str, dict[str, RegressorProtocol]]:
    """Fit independent regressors for each specified representation.

    ``estimator_specs`` must be nonempty with unique estimator names. Each
    selected input must be matrix-like, with an indexable ``shape`` whose first
    entry is a positive integer row count (not a boolean). It must contain the
    same training rows, in the same order, as ``y_train``. Arrays, DataFrames,
    and backend-specific matrix/tensor inputs pass through unchanged.
    Extra, unselected mapping entries are ignored. Feature validation belongs
    to the estimator; this layer only validates sample-axis counts.

    Targets are converted from array-like input to a nonempty, finite, positive
    one-dimensional float64 array, matching the existing lifetime ML contract.
    Row-count validation cannot detect reordered samples; alignment is the
    caller's responsibility. Representation fitting must use training data
    only (or live inside a factory-created pipeline), never calibration or
    external evaluation data. This function performs no cross-validation.

    Returns a nested dictionary indexed by estimator name, then representation
    name, in specification order. Each pair receives a fresh factory instance;
    no cloning, implicit random-state assignment, or name-based dispatch occurs.

    Raises ``ValueError`` for invalid targets, duplicate names, missing inputs,
    row-count mismatches, or factories reusing an instance within this call;
    ``TypeError`` for invalid specifications or missing fit/predict methods.
    All input/factory checks precede fitting. Estimator exceptions propagate;
    a later fit failure does not roll back earlier fits or adapter side effects.
    """
    specs = tuple(estimator_specs)
    if not specs:
        raise ValueError("At least one estimator specification is required.")
    if any(not isinstance(spec, EstimatorSpec) for spec in specs):
        raise TypeError("estimator_specs must contain EstimatorSpec objects.")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Estimator names must be unique.")

    try:
        y = np.asarray(y_train, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_train must contain numeric lifetime values.") from exc
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_train must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("Training lifetimes must be finite and strictly positive.")
    _validate_representation_rows(
        X_train_by_representation,
        [name for spec in specs for name in spec.representations],
        expected_rows=y.size,
    )

    fitted: dict[str, dict[str, RegressorProtocol]] = {}
    instance_ids: set[int] = set()
    for spec in specs:
        fitted[spec.name] = {}
        for name in spec.representations:
            estimator = spec.factory()
            if not all(
                callable(getattr(estimator, method, None))
                for method in ("fit", "predict")
            ):
                raise TypeError(
                    f"Factory for {spec.name!r} must provide fit and predict methods."
                )
            if id(estimator) in instance_ids:
                raise ValueError(
                    "Each factory call must return an independent estimator instance."
                )
            instance_ids.add(id(estimator))
            fitted[spec.name][name] = estimator

    for representation_estimators in fitted.values():
        for name, estimator in representation_estimators.items():
            estimator.fit(X_train_by_representation[name], y)
    return fitted


def predict_regressors(
    *,
    fitted_estimators: Mapping[str, Mapping[str, RegressorProtocol]],
    X_by_representation: Mapping[str, Any],
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Predict without fitting, scoring, or supplying evaluation targets.

    Accepts the nonempty nested mapping returned by ``fit_regressors`` (or an
    equivalent caller-owned mapping). Selected representations must contain
    the same nonempty batch of samples in the same order; only their row counts
    can be checked. Matrix-like inputs require the same ``shape[0]`` contract
    as in ``fit_regressors`` and pass through unchanged; extras are ignored.

    Returns the same estimator/representation keys with one-dimensional float64
    NumPy arrays. Predictions must contain exactly one finite value per row.
    Negative/zero predictions are preserved, not clipped or rejected, matching
    the existing point-regression benchmark contract. Multioutput and interval
    predictions require a separate interface, not silent squeezing or averaging.

    Empty mappings, missing representations, or misaligned row counts raise
    ``ValueError``. Invalid identifier types raise ``TypeError``. Non-numeric,
    incorrectly shaped, or nonfinite predictions raise ``RuntimeError``;
    exceptions raised by the estimator itself propagate.
    """
    if not fitted_estimators:
        raise ValueError("At least one fitted estimator is required.")
    required: list[str] = []
    for name, representation_estimators in fitted_estimators.items():
        _validate_identifier(name, "Estimator name")
        if not representation_estimators:
            raise ValueError(
                f"Estimator {name!r} must select at least one representation."
            )
        for representation in representation_estimators:
            _validate_identifier(representation, "Representation name")
            required.append(representation)
    n_rows = _validate_representation_rows(X_by_representation, required)

    predictions: dict[str, dict[str, NDArray[np.float64]]] = {}
    for name, representation_estimators in fitted_estimators.items():
        predictions[name] = {}
        for representation, estimator in representation_estimators.items():
            raw_prediction = estimator.predict(X_by_representation[representation])
            try:
                prediction = np.asarray(raw_prediction, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Predictions for {name!r}/{representation!r} must be numeric."
                ) from exc
            if prediction.shape != (n_rows,):
                raise RuntimeError(
                    f"Estimator {name!r}/{representation!r} must produce exactly "
                    "one prediction per sample in a one-dimensional array."
                )
            if not np.all(np.isfinite(prediction)):
                raise RuntimeError(
                    f"Predictions for {name!r}/{representation!r} must be finite."
                )
            predictions[name][representation] = prediction
    return predictions
