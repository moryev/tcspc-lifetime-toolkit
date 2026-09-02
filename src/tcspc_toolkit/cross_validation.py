"""Repeated cross-validation for TCSPC lifetime estimation."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold

from tcspc_toolkit.ml_evaluation import (
    evaluate_regression,
)


DEFAULT_CV_N_SPLITS = 5
DEFAULT_CV_N_REPEATS = 5
DEFAULT_CV_RANDOM_STATE = 52_001


@dataclass(frozen=True)
class RepeatedCVConfig:
    """Configuration for repeated development-set cross-validation.

    Repeated cross-validation is used only on development data.
    The final Week 8 robustness Tests A-F must never participate
    in cross-validation, preprocessing fitting, model fitting,
    feature selection, or hyperparameter selection.

    Parameters
    ----------
    n_splits:
        Number of folds in each cross-validation repeat.
    n_repeats:
        Number of independently randomized repetitions.
    random_state:
        Seed controlling the repeated K-fold partitions.
    """

    n_splits: int = DEFAULT_CV_N_SPLITS
    n_repeats: int = DEFAULT_CV_N_REPEATS
    random_state: int = DEFAULT_CV_RANDOM_STATE

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_splits, (bool, np.bool_))
            or not isinstance(
                self.n_splits,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "n_splits must be an integer."
            )

        if self.n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        if (
            isinstance(self.n_repeats, (bool, np.bool_))
            or not isinstance(
                self.n_repeats,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "n_repeats must be an integer."
            )

        if self.n_repeats < 1:
            raise ValueError(
                "n_repeats must be at least 1."
            )

        if (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(
                self.random_state,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "random_state must be an integer."
            )

        if self.random_state < 0:
            raise ValueError(
                "random_state must be non-negative."
            )

    @property
    def n_evaluations(self) -> int:
        """Return the total number of train/validation fits."""

        return self.n_splits * self.n_repeats


@dataclass(frozen=True)
class RepeatedCVBenchmarkResult:
    """Combined repeated-CV results for multiple estimators.

    Attributes
    ----------
    fold_results:
        Long-form table containing one row per model and
        cross-validation evaluation.
    summary:
        Aggregate mean and standard-deviation metrics for
        every model.
    """

    fold_results: pd.DataFrame
    summary: pd.DataFrame


def make_repeated_kfold(
    config: RepeatedCVConfig | None = None,
) -> RepeatedKFold:
    """Construct the repeated K-fold splitter.

    Parameters
    ----------
    config:
        Repeated cross-validation configuration. If omitted,
        the canonical Week 8 configuration is used.

    Returns
    -------
    RepeatedKFold
        Reproducible scikit-learn repeated K-fold splitter.
    """

    if config is None:
        config = RepeatedCVConfig()

    return RepeatedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )


def _subset_rows(
    X: Any,
    indices: np.ndarray,
) -> Any:
    """Return selected sample rows while preserving input type.

    pandas DataFrames are subset using ``iloc`` so that column
    names and DataFrame semantics are retained. NumPy-like
    inputs are indexed directly.
    """

    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]

    return X[indices]


def evaluate_regressor_repeated_cv(
    *,
    estimator_name: str,
    estimator: Any,
    X: Any,
    y: ArrayLike,
    config: RepeatedCVConfig | None = None,
) -> pd.DataFrame:
    """Evaluate one regression estimator using repeated K-fold CV.

    Parameters
    ----------
    estimator_name:
        Stable identifier stored in the long-form result table.
    estimator:
        Unfitted scikit-learn-compatible regression estimator
        implementing ``fit`` and ``predict``. A fresh clone is
        fitted independently inside every fold.
    X:
        Development-set input representation with one sample
        per row.
    y:
        Positive true fluorescence lifetimes for the
        development samples.
    config:
        Repeated cross-validation configuration. If omitted,
        the canonical Week 8 configuration is used.

    Returns
    -------
    pandas.DataFrame
        One row per validation fold containing repeat and fold
        identifiers, sample counts, and regression metrics.

    Notes
    -----
    This function is intended exclusively for development data.

    Final robustness Tests A-F must never be passed to this
    function.

    Any fitted preprocessing steps contained inside ``estimator``,
    such as ``StandardScaler`` or ``PCA``, are refitted from
    scratch on each training fold because the estimator is
    cloned before fitting.
    """

    if (
        not isinstance(estimator_name, str)
        or not estimator_name.strip()
    ):
        raise ValueError(
            "estimator_name must be a non-empty string."
        )

    if config is None:
        config = RepeatedCVConfig()

    try:
        y_array = np.asarray(
            y,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "y must contain numeric values."
        ) from exc

    if y_array.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )

    if y_array.size == 0:
        raise ValueError(
            "y must contain at least one lifetime."
        )

    if not np.all(
        np.isfinite(y_array)
    ):
        raise ValueError(
            "y must contain only finite values."
        )

    if np.any(
        y_array <= 0.0
    ):
        raise ValueError(
            "Lifetimes must be strictly positive."
        )

    if not hasattr(X, "shape"):
        raise TypeError(
            "X must provide a two-dimensional shape."
        )

    if len(X.shape) != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if X.shape[0] != y_array.size:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X must contain at least one feature."
        )

    if config.n_splits > y_array.size:
        raise ValueError(
            "n_splits must not exceed the number of samples."
        )

    splitter = make_repeated_kfold(
        config
    )

    sample_indices = np.arange(
        y_array.size,
        dtype=np.int64,
    )

    rows: list[
        dict[str, str | int | float]
    ] = []

    for split_index, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(sample_indices)
    ):
        repeat = (
            split_index // config.n_splits
        ) + 1

        fold = (
            split_index % config.n_splits
        ) + 1

        X_train = _subset_rows(
            X,
            train_indices,
        )

        X_validation = _subset_rows(
            X,
            validation_indices,
        )

        y_train = y_array[
            train_indices
        ]

        y_validation = y_array[
            validation_indices
        ]

        fold_estimator = clone(
            estimator
        )

        fold_estimator.fit(
            X_train,
            y_train,
        )

        y_pred = np.asarray(
            fold_estimator.predict(
                X_validation
            ),
            dtype=np.float64,
        )

        if y_pred.shape != y_validation.shape:
            raise RuntimeError(
                "Estimator must produce exactly one "
                "prediction per validation sample."
            )

        if not np.all(
            np.isfinite(y_pred)
        ):
            raise RuntimeError(
                "Estimator predictions must be finite."
            )

        metrics = evaluate_regression(
            y_true=y_validation,
            y_pred=y_pred,
        )

        bias_ns = float(
            np.mean(
                y_pred - y_validation
            )
        )

        rows.append(
            {
                "model": estimator_name,
                "repeat": repeat,
                "fold": fold,
                "n_train": int(
                    train_indices.size
                ),
                "n_validation": int(
                    validation_indices.size
                ),
                "mae_ns": metrics.mae_ns,
                "median_absolute_error_ns": (
                    metrics.median_absolute_error_ns
                ),
                "rmse_ns": metrics.rmse_ns,
                "bias_ns": bias_ns,
                "r2": metrics.r2,
            }
        )

    return pd.DataFrame(
        rows
    )


CV_METRIC_COLUMNS = (
    "mae_ns",
    "median_absolute_error_ns",
    "rmse_ns",
    "bias_ns",
    "r2",
)


def summarize_repeated_cv(
    cv_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize repeated cross-validation performance by model.

    Parameters
    ----------
    cv_results:
        Long-form repeated-CV table produced by
        ``evaluate_regressor_repeated_cv``.

    Returns
    -------
    pandas.DataFrame
        One row per model containing the number of CV evaluations
        and the mean and standard deviation of each regression
        metric across folds and repeats.

    Notes
    -----
    The reported standard deviation describes variation across
    repeated train/validation splits. It is therefore a measure
    of development-set performance stability, not uncertainty
    on the final untouched Tests A-F.
    """

    required_columns = {
        "model",
        *CV_METRIC_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(cv_results.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "cv_results is missing required columns: "
            f"{missing_text}."
        )

    if cv_results.empty:
        raise ValueError(
            "cv_results must contain at least one CV result."
        )

    if cv_results["model"].isna().any():
        raise ValueError(
            "model must not contain missing values."
        )

    metric_values = cv_results[
        list(CV_METRIC_COLUMNS)
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(metric_values)
    ):
        raise ValueError(
            "CV metrics must contain only finite values."
        )

    summary_rows: list[
        dict[str, str | int | float]
    ] = []

    for model_name, model_results in (
        cv_results.groupby(
            "model",
            sort=False,
        )
    ):
        if len(model_results) < 2:
            raise ValueError(
                "Each model must contain at least two "
                "CV evaluations to compute a standard deviation."
            )
        row: dict[
            str,
            str | int | float
        ] = {
            "model": str(model_name),
            "n_evaluations": int(
                len(model_results)
            ),
        }

        for metric_name in CV_METRIC_COLUMNS:
            values = model_results[
                metric_name
            ].to_numpy(
                dtype=np.float64
            )

            row[
                f"mean_{metric_name}"
            ] = float(
                np.mean(values)
            )

            row[
                f"std_{metric_name}"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def evaluate_estimators_repeated_cv(
    *,
    estimators: Mapping[str, Any],
    X: Any,
    y: ArrayLike,
    config: RepeatedCVConfig | None = None,
) -> RepeatedCVBenchmarkResult:
    """Evaluate multiple estimators using identical repeated CV.

    Parameters
    ----------
    estimators:
        Mapping from stable model names to unfitted
        scikit-learn-compatible regression estimators.
    X:
        Common development-set input representation.
    y:
        Positive true fluorescence lifetimes.
    config:
        Repeated cross-validation configuration. If omitted,
        the canonical Week 8 configuration is used.

    Returns
    -------
    RepeatedCVBenchmarkResult
        Combined fold-level results and aggregate summary.

    Notes
    -----
    Every estimator is evaluated using the same repeated
    K-fold configuration and therefore the same deterministic
    train/validation partitions.

    Each fold still receives a fresh clone of the supplied
    estimator through ``evaluate_regressor_repeated_cv``.

    This function is intended only for development data.
    Final robustness Tests A-F must remain untouched.
    """

    if not estimators:
        raise ValueError(
            "estimators must contain at least one estimator."
        )

    if config is None:
        config = RepeatedCVConfig()

    fold_tables: list[pd.DataFrame] = []

    for estimator_name, estimator in (
        estimators.items()
    ):
        model_results = (
            evaluate_regressor_repeated_cv(
                estimator_name=estimator_name,
                estimator=estimator,
                X=X,
                y=y,
                config=config,
            )
        )

        fold_tables.append(
            model_results
        )

    fold_results = pd.concat(
        fold_tables,
        ignore_index=True,
    )

    summary = summarize_repeated_cv(
        fold_results
    )

    return RepeatedCVBenchmarkResult(
        fold_results=fold_results,
        summary=summary,
    )
