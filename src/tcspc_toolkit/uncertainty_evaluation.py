"""Shared protocol and metrics for Week 9 uncertainty evaluation.

The module deliberately separates calibrated prediction intervals from
uncertainty scores and empirical reference variability. The frozen Week 8
Tests A-F are external evaluation data only and must not be used for fitting,
uncertainty calibration, or threshold selection.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from tcspc_toolkit.generalization import FINAL_ROBUSTNESS_TEST_IDS


DEFAULT_UNCERTAINTY_CALIBRATION_FRACTION = 0.25
DEFAULT_UNCERTAINTY_SPLIT_SEED = 57

WEEK9_HYPOTHESES = (
    "Acquisition degradation should generally increase uncertainty or reduce calibration.",
    "OOD errors may grow faster than internal ML uncertainty scores.",
    "Classical uncertainty should be most meaningful when the assumed physics is correct.",
    "Physical model mismatch can produce confident bias.",
)


class UncertaintyOutputKind(str, Enum):
    """Scientific interpretation of an uncertainty method output."""

    PREDICTION_INTERVAL = "prediction_interval"
    UNCERTAINTY_SCORE = "uncertainty_score"
    EMPIRICAL_REFERENCE = "empirical_reference"


class UncertaintyDataRole(str, Enum):
    """Allowed roles for data used during Week 9."""

    TRAINING = "uncertainty_training"
    CALIBRATION = "uncertainty_calibration"
    EXTERNAL_EVALUATION = "external_evaluation"


@dataclass(frozen=True)
class UncertaintyMethodDefinition:
    """Describe what an uncertainty method claims to estimate."""

    method_id: str
    output_kind: UncertaintyOutputKind
    interpretation: str
    calibration_scope: str

    def __post_init__(self) -> None:
        if not self.method_id.strip():
            raise ValueError("method_id must not be empty")
        if not self.interpretation.strip():
            raise ValueError("interpretation must not be empty")
        if not self.calibration_scope.strip():
            raise ValueError("calibration_scope must not be empty")


@dataclass(frozen=True)
class UncertaintyDataProvenance:
    """Record the role of one named data source in Week 9.

    Tests A-F are frozen external evaluation sources. Constructing a training
    or calibration provenance record for one of those identifiers raises
    immediately, making leakage an explicit protocol violation.
    """

    source_id: str
    role: UncertaintyDataRole

    def __post_init__(self) -> None:
        normalized_source = self.source_id.strip().upper()
        if not normalized_source:
            raise ValueError("source_id must not be empty")

        if (
            normalized_source in FINAL_ROBUSTNESS_TEST_IDS
            and self.role is not UncertaintyDataRole.EXTERNAL_EVALUATION
        ):
            raise ValueError(
                "Frozen Tests A-F may be used only for external uncertainty evaluation."
            )


@dataclass(frozen=True)
class UncertaintyDevelopmentSplit:
    """Indices for leakage-safe uncertainty training and calibration subsets."""

    training_indices: tuple[int, ...]
    calibration_indices: tuple[int, ...]
    random_seed: int
    calibration_fraction: float

    def __post_init__(self) -> None:
        if not self.training_indices:
            raise ValueError("training_indices must not be empty")
        if not self.calibration_indices:
            raise ValueError("calibration_indices must not be empty")
        if any(index < 0 for index in self.training_indices):
            raise ValueError("training_indices must be non-negative")
        if any(index < 0 for index in self.calibration_indices):
            raise ValueError("calibration_indices must be non-negative")
        if len(set(self.training_indices)) != len(self.training_indices):
            raise ValueError("training_indices must not contain duplicates")
        if len(set(self.calibration_indices)) != len(self.calibration_indices):
            raise ValueError("calibration_indices must not contain duplicates")
        if set(self.training_indices) & set(self.calibration_indices):
            raise ValueError("training and calibration indices must be disjoint")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie strictly between 0 and 1")


def make_uncertainty_development_split(
    n_samples: int,
    *,
    calibration_fraction: float = DEFAULT_UNCERTAINTY_CALIBRATION_FRACTION,
    random_seed: int = DEFAULT_UNCERTAINTY_SPLIT_SEED,
) -> UncertaintyDevelopmentSplit:
    """Split development samples into uncertainty-training and calibration sets."""

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")

    n_calibration = int(round(n_samples * calibration_fraction))
    n_calibration = min(max(n_calibration, 1), n_samples - 1)

    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(n_samples)

    calibration_indices = tuple(
        sorted(int(index) for index in permutation[:n_calibration])
    )
    training_indices = tuple(
        sorted(int(index) for index in permutation[n_calibration:])
    )

    return UncertaintyDevelopmentSplit(
        training_indices=training_indices,
        calibration_indices=calibration_indices,
        random_seed=random_seed,
        calibration_fraction=calibration_fraction,
    )


@dataclass(frozen=True)
class PredictionIntervalResult:
    """Per-sample prediction intervals with an explicit nominal coverage."""

    prediction: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    nominal_coverage: float
    method_id: str

    def __post_init__(self) -> None:
        _validate_1d_same_shape(
            self.prediction,
            self.lower,
            self.upper,
            names=("prediction", "lower", "upper"),
        )
        if self.prediction.size == 0:
            raise ValueError("prediction intervals must not be empty")
        if not 0.0 < self.nominal_coverage < 1.0:
            raise ValueError("nominal_coverage must lie strictly between 0 and 1")
        if not self.method_id.strip():
            raise ValueError("method_id must not be empty")

    @property
    def valid_interval_mask(self) -> NDArray[np.bool_]:
        """Return intervals that are finite, ordered, and have a finite prediction."""

        return (
            np.isfinite(self.prediction)
            & np.isfinite(self.lower)
            & np.isfinite(self.upper)
            & (self.lower <= self.upper)
        )


@dataclass(frozen=True)
class UncertaintyScoreResult:
    """Per-sample uncertainty ranking without a nominal coverage claim."""

    prediction: NDArray[np.float64]
    uncertainty_score: NDArray[np.float64]
    method_id: str

    def __post_init__(self) -> None:
        _validate_1d_same_shape(
            self.prediction,
            self.uncertainty_score,
            names=("prediction", "uncertainty_score"),
        )
        if self.prediction.size == 0:
            raise ValueError("uncertainty scores must not be empty")
        negative_finite_scores = (
            np.isfinite(self.uncertainty_score)
            & (self.uncertainty_score < 0.0)
        )
        if np.any(negative_finite_scores):
            raise ValueError("finite uncertainty scores must be non-negative")
        if not self.method_id.strip():
            raise ValueError("method_id must not be empty")

    @property
    def valid_score_mask(self) -> NDArray[np.bool_]:
        """Return samples with finite predictions and finite scores."""

        return np.isfinite(self.prediction) & np.isfinite(self.uncertainty_score)


@dataclass(frozen=True)
class IntervalEvaluationMetrics:
    """Calibration and sharpness metrics for prediction intervals."""

    n_samples: int
    n_valid_intervals: int
    empirical_coverage: float
    coverage_error: float
    mean_interval_width: float
    median_interval_width: float
    mean_interval_score: float
    interval_failure_rate: float


@dataclass(frozen=True)
class UncertaintyScoreMetrics:
    """Ranking diagnostics for non-interval uncertainty outputs."""

    n_samples: int
    n_valid_scores: int
    score_failure_rate: float
    mean_absolute_error_ns: float
    spearman_error_correlation: float
    low_uncertainty_mae_ns: float
    high_uncertainty_mae_ns: float
    tail_fraction: float


@dataclass(frozen=True)
class SelectivePredictionMetrics:
    """Error before and after rejecting the most uncertain predictions."""

    rejection_fraction: float
    retained_fraction: float
    n_valid_scores: int
    n_retained: int
    mae_all_ns: float
    mae_retained_ns: float
    mae_improvement_ns: float


def evaluate_prediction_intervals(
    true_lifetimes_ns: NDArray[np.float64],
    intervals: PredictionIntervalResult,
) -> IntervalEvaluationMetrics:
    """Evaluate interval coverage, width, failures, and the mean interval score."""

    true_lifetimes_ns = _validate_true_lifetimes(
        true_lifetimes_ns,
        expected_shape=intervals.prediction.shape,
    )

    valid = intervals.valid_interval_mask
    n_samples = int(true_lifetimes_ns.size)
    n_valid = int(np.count_nonzero(valid))
    failure_rate = 1.0 - (n_valid / n_samples)

    if n_valid == 0:
        return IntervalEvaluationMetrics(
            n_samples=n_samples,
            n_valid_intervals=0,
            empirical_coverage=float("nan"),
            coverage_error=float("nan"),
            mean_interval_width=float("nan"),
            median_interval_width=float("nan"),
            mean_interval_score=float("nan"),
            interval_failure_rate=failure_rate,
        )

    y = true_lifetimes_ns[valid]
    lower = intervals.lower[valid]
    upper = intervals.upper[valid]
    widths = upper - lower

    covered = (lower <= y) & (y <= upper)
    empirical_coverage = float(np.mean(covered))
    alpha = 1.0 - intervals.nominal_coverage

    interval_score = widths.copy()
    below = y < lower
    above = y > upper
    interval_score[below] += (2.0 / alpha) * (lower[below] - y[below])
    interval_score[above] += (2.0 / alpha) * (y[above] - upper[above])

    return IntervalEvaluationMetrics(
        n_samples=n_samples,
        n_valid_intervals=n_valid,
        empirical_coverage=empirical_coverage,
        coverage_error=empirical_coverage - intervals.nominal_coverage,
        mean_interval_width=float(np.mean(widths)),
        median_interval_width=float(np.median(widths)),
        mean_interval_score=float(np.mean(interval_score)),
        interval_failure_rate=failure_rate,
    )


def evaluate_uncertainty_scores(
    true_lifetimes_ns: NDArray[np.float64],
    scores: UncertaintyScoreResult,
    *,
    tail_fraction: float = 0.20,
) -> UncertaintyScoreMetrics:
    """Test whether larger uncertainty scores are associated with larger errors."""

    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError("tail_fraction must lie in (0, 0.5]")

    true_lifetimes_ns = _validate_true_lifetimes(
        true_lifetimes_ns,
        expected_shape=scores.prediction.shape,
    )

    valid = scores.valid_score_mask
    n_samples = int(true_lifetimes_ns.size)
    n_valid = int(np.count_nonzero(valid))
    failure_rate = 1.0 - (n_valid / n_samples)

    if n_valid == 0:
        return UncertaintyScoreMetrics(
            n_samples=n_samples,
            n_valid_scores=0,
            score_failure_rate=failure_rate,
            mean_absolute_error_ns=float("nan"),
            spearman_error_correlation=float("nan"),
            low_uncertainty_mae_ns=float("nan"),
            high_uncertainty_mae_ns=float("nan"),
            tail_fraction=tail_fraction,
        )

    y = true_lifetimes_ns[valid]
    prediction = scores.prediction[valid]
    uncertainty = scores.uncertainty_score[valid]
    absolute_error = np.abs(prediction - y)

    if (
        n_valid < 2
        or np.all(uncertainty == uncertainty[0])
        or np.all(absolute_error == absolute_error[0])
    ):
        correlation = float("nan")
    else:
        correlation = float(spearmanr(uncertainty, absolute_error).statistic)

    n_tail = max(1, int(np.floor(n_valid * tail_fraction)))
    order = np.argsort(uncertainty, kind="stable")
    low_indices = order[:n_tail]
    high_indices = order[-n_tail:]

    return UncertaintyScoreMetrics(
        n_samples=n_samples,
        n_valid_scores=n_valid,
        score_failure_rate=failure_rate,
        mean_absolute_error_ns=float(np.mean(absolute_error)),
        spearman_error_correlation=correlation,
        low_uncertainty_mae_ns=float(np.mean(absolute_error[low_indices])),
        high_uncertainty_mae_ns=float(np.mean(absolute_error[high_indices])),
        tail_fraction=tail_fraction,
    )


def evaluate_selective_prediction(
    true_lifetimes_ns: NDArray[np.float64],
    scores: UncertaintyScoreResult,
    *,
    rejection_fraction: float,
) -> SelectivePredictionMetrics:
    """Reject the highest-uncertainty predictions and measure MAE improvement."""

    if not 0.0 <= rejection_fraction < 1.0:
        raise ValueError("rejection_fraction must lie in [0, 1)")

    true_lifetimes_ns = _validate_true_lifetimes(
        true_lifetimes_ns,
        expected_shape=scores.prediction.shape,
    )

    valid = scores.valid_score_mask
    if not np.any(valid):
        raise ValueError("at least one valid uncertainty score is required")

    y = true_lifetimes_ns[valid]
    prediction = scores.prediction[valid]
    uncertainty = scores.uncertainty_score[valid]
    absolute_error = np.abs(prediction - y)

    n_valid = int(absolute_error.size)
    n_retained = max(1, int(np.ceil(n_valid * (1.0 - rejection_fraction))))
    retained_indices = np.argsort(uncertainty, kind="stable")[:n_retained]

    mae_all = float(np.mean(absolute_error))
    mae_retained = float(np.mean(absolute_error[retained_indices]))

    return SelectivePredictionMetrics(
        rejection_fraction=rejection_fraction,
        retained_fraction=n_retained / n_valid,
        n_valid_scores=n_valid,
        n_retained=n_retained,
        mae_all_ns=mae_all,
        mae_retained_ns=mae_retained,
        mae_improvement_ns=mae_all - mae_retained,
    )


def _validate_true_lifetimes(
    values: NDArray[np.float64],
    *,
    expected_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("true_lifetimes_ns must be one-dimensional")
    if values.shape != expected_shape:
        raise ValueError("true lifetimes and uncertainty outputs must have the same shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("true_lifetimes_ns must contain only finite values")
    if np.any(values <= 0.0):
        raise ValueError("true_lifetimes_ns must be positive")
    return values


def _validate_1d_same_shape(
    *arrays: NDArray[np.float64],
    names: tuple[str, ...],
) -> None:
    if len(arrays) != len(names):
        raise ValueError("names must match the number of arrays")
    shapes: list[tuple[int, ...]] = []
    for array, name in zip(arrays, names, strict=True):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        shapes.append(array.shape)
    if len(set(shapes)) != 1:
        raise ValueError("uncertainty output arrays must have the same shape")
