"""Method definitions for machine-learning uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pandas as pd

from numpy.typing import NDArray

from typing import Any, TYPE_CHECKING

from sklearn.base import clone

from scipy.stats import spearmanr

from tcspc_toolkit.ml_evaluation import (
    BenchmarkDataset,
)
from tcspc_toolkit.ml_models import (
    DEFAULT_RANDOM_STATE,
    make_quantile_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)
from tcspc_toolkit.uncertainty_evaluation import (
    IntervalEvaluationMetrics,
    PredictionIntervalResult,
    QuantileIntervalEvaluationMetrics,
    UncertaintyDevelopmentSplit,
    UncertaintyMethodDefinition,
    UncertaintyOutputKind,
    UncertaintyScoreMetrics,
    UncertaintyScoreResult,
    evaluate_prediction_intervals,
    evaluate_quantile_prediction_intervals,
    evaluate_uncertainty_scores,
    make_uncertainty_development_split,
)


if TYPE_CHECKING:
    from tcspc_toolkit.generalization import (
        GeneralizationSuiteDefinition,
    )
    from tcspc_toolkit.generalization_evaluation import (
        GeneralizationPreparedData,
    )


QUANTILE_GRADIENT_BOOSTING_METHOD_ID = (
    "quantile_gradient_boosting"
)

DEFAULT_LOWER_QUANTILE = 0.05
DEFAULT_MEDIAN_QUANTILE = 0.50
DEFAULT_UPPER_QUANTILE = 0.95

DEFAULT_QUANTILE_NOMINAL_COVERAGE = (
    DEFAULT_UPPER_QUANTILE
    - DEFAULT_LOWER_QUANTILE
)


RANDOM_FOREST_TREE_SPREAD_METHOD_ID = (
    "random_forest_tree_spread"
)

ML_TRAINING_BOOTSTRAP_METHOD_ID = (
    "ml_training_bootstrap"
)

DEFAULT_BOOTSTRAP_REPLICATES = 200


class QuantileGradientBoostingIntervalEstimator:
    """Three-model quantile estimator for lifetime prediction intervals.

    Three independent histogram gradient-boosting regressors estimate
    the conditional 0.05, 0.50, and 0.95 lifetime quantiles.

    The median model provides the central lifetime prediction, while
    the lower and upper models define a nominal 90% prediction
    interval.

    Quantile predictions are returned exactly as produced by the
    fitted models. They are never reordered, clipped, or otherwise
    repaired when quantile crossing occurs.
    """

    def __init__(
        self,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.random_state = random_state

        self.lower_model = (
            make_quantile_hist_gradient_boosting_pipeline(
                quantile=DEFAULT_LOWER_QUANTILE,
                random_state=random_state,
            )
        )

        self.median_model = (
            make_quantile_hist_gradient_boosting_pipeline(
                quantile=DEFAULT_MEDIAN_QUANTILE,
                random_state=random_state,
            )
        )

        self.upper_model = (
            make_quantile_hist_gradient_boosting_pipeline(
                quantile=DEFAULT_UPPER_QUANTILE,
                random_state=random_state,
            )
        )

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> QuantileGradientBoostingIntervalEstimator:
        """Fit all three quantile models on the same training data."""

        self.lower_model.fit(
            X,
            y,
        )

        self.median_model.fit(
            X,
            y,
        )

        self.upper_model.fit(
            X,
            y,
        )

        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> PredictionIntervalResult:
        """Predict the median lifetime and nominal 90% interval."""

        lower = np.asarray(
            self.lower_model.predict(X),
            dtype=np.float64,
        )

        prediction = np.asarray(
            self.median_model.predict(X),
            dtype=np.float64,
        )

        upper = np.asarray(
            self.upper_model.predict(X),
            dtype=np.float64,
        )

        return PredictionIntervalResult(
            prediction=prediction,
            lower=lower,
            upper=upper,
            nominal_coverage=(
                DEFAULT_QUANTILE_NOMINAL_COVERAGE
            ),
            method_id=(
                QUANTILE_GRADIENT_BOOSTING_METHOD_ID
            ),
        )


class RandomForestTreeSpreadEstimator:
    """Random-Forest lifetime estimator with tree-disagreement scores.

    The central prediction is the mean prediction across the fitted
    regression trees.

    The uncertainty score is the population standard deviation of
    those tree predictions.

    The tree spread is an ensemble-disagreement heuristic. It is not
    interpreted as a calibrated prediction standard deviation or
    prediction interval.
    """

    def __init__(
        self,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.random_state = random_state

        self.pipeline = (
            make_random_forest_pipeline(
                random_state=random_state,
            )
        )

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RandomForestTreeSpreadEstimator:
        """Fit the Random Forest on uncertainty-training data."""

        X, y = _validate_ml_uncertainty_training_data(
            X,
            y,
        )

        self.pipeline.fit(
            X,
            y,
        )

        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> UncertaintyScoreResult:
        """Return mean RF predictions and tree-to-tree spread."""

        X = _validate_ml_uncertainty_prediction_features(
            X
        )

        forest = self.pipeline.named_steps[
            "model"
        ]

        tree_predictions = np.stack(
            [
                np.asarray(
                    tree.predict(X),
                    dtype=np.float64,
                )
                for tree in forest.estimators_
            ],
            axis=0,
        )

        prediction = np.mean(
            tree_predictions,
            axis=0,
        )

        tree_spread = np.std(
            tree_predictions,
            axis=0,
            ddof=0,
        )

        return UncertaintyScoreResult(
            prediction=prediction,
            uncertainty_score=tree_spread,
            method_id=(
                RANDOM_FOREST_TREE_SPREAD_METHOD_ID
            ),
        )


class BootstrapPredictionSpreadEstimator:
    """Estimate training-sample sensitivity by non-parametric bootstrap.

    A reference estimator is fitted once on the complete
    uncertainty-training subset and supplies the central prediction.

    Additional estimator clones are fitted to bootstrap resamples of
    the uncertainty-training samples. The standard deviation of their
    predictions is returned as an uncertainty score.

    Bootstrap spread is not automatically interpreted as a calibrated
    prediction interval.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        if (
            isinstance(
                n_bootstrap,
                (bool, np.bool_),
            )
            or not isinstance(
                n_bootstrap,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "n_bootstrap must be an integer."
            )

        if n_bootstrap < 2:
            raise ValueError(
                "n_bootstrap must be at least 2."
            )

        self.base_estimator = base_estimator
        self.n_bootstrap = int(
            n_bootstrap
        )
        self.random_state = random_state

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> BootstrapPredictionSpreadEstimator:
        """Fit the reference model and bootstrap estimator replicas."""

        X, y = _validate_ml_uncertainty_training_data(
            X,
            y,
        )

        self.reference_estimator_ = clone(
            self.base_estimator
        )

        self.reference_estimator_.fit(
            X,
            y,
        )

        rng = np.random.default_rng(
            self.random_state
        )

        n_samples = int(
            y.size
        )

        self.bootstrap_estimators_: list[
            Any
        ] = []

        for _ in range(
            self.n_bootstrap
        ):
            sample_indices = rng.integers(
                0,
                n_samples,
                size=n_samples,
            )

            estimator = clone(
                self.base_estimator
            )

            estimator.fit(
                X[
                    sample_indices
                ],
                y[
                    sample_indices
                ],
            )

            self.bootstrap_estimators_.append(
                estimator
            )

        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> UncertaintyScoreResult:
        """Return reference predictions and bootstrap prediction spread."""

        X = _validate_ml_uncertainty_prediction_features(
            X
        )

        if not hasattr(
            self,
            "reference_estimator_",
        ):
            raise RuntimeError(
                "BootstrapPredictionSpreadEstimator must be "
                "fitted before prediction."
            )

        prediction = np.asarray(
            self.reference_estimator_.predict(
                X
            ),
            dtype=np.float64,
        )

        bootstrap_predictions = np.stack(
            [
                np.asarray(
                    estimator.predict(X),
                    dtype=np.float64,
                )
                for estimator
                in self.bootstrap_estimators_
            ],
            axis=0,
        )

        bootstrap_spread = np.std(
            bootstrap_predictions,
            axis=0,
            ddof=1,
        )

        return UncertaintyScoreResult(
            prediction=prediction,
            uncertainty_score=(
                bootstrap_spread
            ),
            method_id=(
                ML_TRAINING_BOOTSTRAP_METHOD_ID
            ),
        )


@dataclass(frozen=True)
class MLUncertaintyScoreCalibrationResult:
    """Held-out development evaluation for an ML uncertainty score."""

    fitted_estimator: Any

    split: UncertaintyDevelopmentSplit

    calibration_scores: UncertaintyScoreResult

    score_metrics: UncertaintyScoreMetrics

    feature_names: tuple[str, ...]


def fit_and_evaluate_random_forest_tree_spread(
    development: BenchmarkDataset,
    *,
    split: UncertaintyDevelopmentSplit | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MLUncertaintyScoreCalibrationResult:
    """Fit RF tree-spread estimator and evaluate held-out ranking."""

    estimator = RandomForestTreeSpreadEstimator(
        random_state=random_state,
    )

    return _fit_and_evaluate_ml_uncertainty_score(
        development,
        estimator=estimator,
        split=split,
    )


def fit_and_evaluate_ridge_bootstrap_spread(
    development: BenchmarkDataset,
    *,
    split: UncertaintyDevelopmentSplit | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MLUncertaintyScoreCalibrationResult:
    """Fit Ridge plus bootstrap replicas and evaluate held-out spread."""

    estimator = BootstrapPredictionSpreadEstimator(
        make_ridge_pipeline(),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    return _fit_and_evaluate_ml_uncertainty_score(
        development,
        estimator=estimator,
        split=split,
    )


def _fit_and_evaluate_ml_uncertainty_score(
    development: BenchmarkDataset,
    *,
    estimator: Any,
    split: UncertaintyDevelopmentSplit | None,
) -> MLUncertaintyScoreCalibrationResult:
    """Fit one score-producing estimator on the Week-9 split."""

    n_samples = int(
        development.y.size
    )

    if n_samples < 2:
        raise ValueError(
            "development must contain at least two samples."
        )

    if (
        development.X_features.shape[0]
        != n_samples
    ):
        raise ValueError(
            "X_features and y must contain the same "
            "number of development samples."
        )

    X = development.X_features.to_numpy(
        dtype=np.float64,
    )

    y = np.asarray(
        development.y,
        dtype=np.float64,
    )

    X, y = _validate_ml_uncertainty_training_data(
        X,
        y,
    )

    if split is None:
        split = make_uncertainty_development_split(
            n_samples
        )

    _validate_uncertainty_split_for_dataset(
        split,
        n_samples=n_samples,
    )

    training_indices = np.asarray(
        split.training_indices,
        dtype=np.int64,
    )

    calibration_indices = np.asarray(
        split.calibration_indices,
        dtype=np.int64,
    )

    estimator.fit(
        X[
            training_indices
        ],
        y[
            training_indices
        ],
    )

    calibration_scores = estimator.predict(
        X[
            calibration_indices
        ]
    )

    score_metrics = evaluate_uncertainty_scores(
        y[
            calibration_indices
        ],
        calibration_scores,
    )

    return MLUncertaintyScoreCalibrationResult(
        fitted_estimator=estimator,
        split=split,
        calibration_scores=(
            calibration_scores
        ),
        score_metrics=score_metrics,
        feature_names=tuple(
            development.X_features.columns
        ),
    )


@dataclass(frozen=True)
class QuantileGradientBoostingCalibrationResult:
    """Development calibration result for quantile lifetime intervals.

    The fitted estimator has seen only the uncertainty-training
    subset. The calibration subset is used only for held-out
    evaluation.

    After this result is created, the fitted estimator should be
    treated as frozen for subsequent external robustness evaluation.
    """

    fitted_estimator: (
        QuantileGradientBoostingIntervalEstimator
    )

    split: UncertaintyDevelopmentSplit

    calibration_intervals: PredictionIntervalResult

    interval_metrics: IntervalEvaluationMetrics

    quantile_metrics: (
        QuantileIntervalEvaluationMetrics
    )

    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class QuantileGradientBoostingExternalResult:
    """Evaluation of one frozen quantile model on external data."""

    condition_id: str
    intervals: PredictionIntervalResult
    interval_metrics: IntervalEvaluationMetrics
    quantile_metrics: QuantileIntervalEvaluationMetrics
    median_prediction_mae_ns: float
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class QuantileGradientBoostingRobustnessResult:
    """Conditional external uncertainty evaluation for Day 58.

    The five external conditions are evaluated using the same
    frozen q05/q50/q95 estimator produced by the development
    calibration workflow.
    """

    condition_results: dict[
        str,
        QuantileGradientBoostingExternalResult,
    ]

    summary: pd.DataFrame


@dataclass(frozen=True)
class QuantileGradientBoostingPairedConditionResult:
    """Matched Test-A versus shifted-condition uncertainty response."""

    condition_id: str
    shifted_test_id: str

    reference_result: (
        QuantileGradientBoostingExternalResult
    )

    shifted_result: (
        QuantileGradientBoostingExternalResult
    )

    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class QuantileGradientBoostingPairedRobustnessResult:
    """Paired Day-58 uncertainty-response analysis."""

    condition_results: dict[
        str,
        QuantileGradientBoostingPairedConditionResult,
    ]

    summary: pd.DataFrame


def fit_and_evaluate_quantile_gradient_boosting(
    development: BenchmarkDataset,
    *,
    split: UncertaintyDevelopmentSplit | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> QuantileGradientBoostingCalibrationResult:
    """Fit quantile models and evaluate them on held-out calibration data.

    Only engineered features are used.

    The development dataset is divided into uncertainty-training
    and calibration subsets using the Week 9 development split.
    All three quantile models are fitted exclusively on the
    uncertainty-training subset.

    The calibration subset is used only after fitting to measure
    interval calibration, sharpness, pinball loss, and quantile
    crossing.

    Final robustness Tests A-F must not be used by this function.
    """

    n_samples = int(
        development.y.size
    )

    if n_samples < 2:
        raise ValueError(
            "development must contain at least two samples."
        )

    if (
        development.X_features.shape[0]
        != n_samples
    ):
        raise ValueError(
            "X_features and y must contain the same "
            "number of development samples."
        )

    X_features = (
        development.X_features.to_numpy(
            dtype=np.float64,
        )
    )

    y = np.asarray(
        development.y,
        dtype=np.float64,
    )

    if X_features.ndim != 2:
        raise ValueError(
            "X_features must be two-dimensional."
        )

    if X_features.shape[1] == 0:
        raise ValueError(
            "X_features must contain at least one feature."
        )

    if not np.all(
        np.isfinite(X_features)
    ):
        raise ValueError(
            "X_features must contain only finite values."
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            "development lifetimes must contain only "
            "finite values."
        )

    if np.any(
        y <= 0.0
    ):
        raise ValueError(
            "development lifetimes must be positive."
        )

    if split is None:
        split = (
            make_uncertainty_development_split(
                n_samples
            )
        )

    _validate_uncertainty_split_for_dataset(
        split,
        n_samples=n_samples,
    )

    training_indices = np.asarray(
        split.training_indices,
        dtype=np.int64,
    )

    calibration_indices = np.asarray(
        split.calibration_indices,
        dtype=np.int64,
    )

    X_training = X_features[
        training_indices
    ]

    y_training = y[
        training_indices
    ]

    X_calibration = X_features[
        calibration_indices
    ]

    y_calibration = y[
        calibration_indices
    ]

    estimator = (
        QuantileGradientBoostingIntervalEstimator(
            random_state=random_state,
        )
    )

    estimator.fit(
        X_training,
        y_training,
    )

    calibration_intervals = (
        estimator.predict(
            X_calibration
        )
    )

    interval_metrics = (
        evaluate_prediction_intervals(
            y_calibration,
            calibration_intervals,
        )
    )

    quantile_metrics = (
        evaluate_quantile_prediction_intervals(
            y_calibration,
            calibration_intervals,
            lower_quantile=(
                DEFAULT_LOWER_QUANTILE
            ),
            median_quantile=(
                DEFAULT_MEDIAN_QUANTILE
            ),
            upper_quantile=(
                DEFAULT_UPPER_QUANTILE
            ),
        )
    )

    return (
        QuantileGradientBoostingCalibrationResult(
            fitted_estimator=estimator,
            split=split,
            calibration_intervals=(
                calibration_intervals
            ),
            interval_metrics=interval_metrics,
            quantile_metrics=quantile_metrics,
            feature_names=tuple(
                development.X_features.columns
            ),
        )
    )


def evaluate_frozen_quantile_gradient_boosting(
    calibration_result: QuantileGradientBoostingCalibrationResult,
    X_features: pd.DataFrame,
    y_true_ns: NDArray[np.float64],
    *,
    condition_id: str,
) -> QuantileGradientBoostingExternalResult:
    """Evaluate the frozen quantile estimator on external features.

    The estimator stored in ``calibration_result`` is used only for
    prediction. No model fitting or calibration is performed.

    The engineered-feature schema must match the feature schema used
    during Week 9 development.
    """

    if not condition_id.strip():
        raise ValueError(
            "condition_id must not be empty."
        )

    if not isinstance(
        X_features,
        pd.DataFrame,
    ):
        raise TypeError(
            "X_features must be a pandas DataFrame."
        )

    feature_names = tuple(
        X_features.columns
    )

    if (
        feature_names
        != calibration_result.feature_names
    ):
        raise ValueError(
            "External feature schema must match "
            "the uncertainty-development feature schema."
        )

    X = X_features.to_numpy(
        dtype=np.float64,
    )

    y = np.asarray(
        y_true_ns,
        dtype=np.float64,
    )

    if X.ndim != 2:
        raise ValueError(
            "X_features must be two-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "y_true_ns must be one-dimensional."
        )

    if X.shape[0] != y.size:
        raise ValueError(
            "X_features and y_true_ns must contain "
            "the same number of samples."
        )

    if y.size == 0:
        raise ValueError(
            "External evaluation data must not be empty."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "X_features must contain only finite values."
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            "y_true_ns must contain only finite values."
        )

    if np.any(
        y <= 0.0
    ):
        raise ValueError(
            "y_true_ns must be positive."
        )

    intervals = (
        calibration_result
        .fitted_estimator
        .predict(
            X
        )
    )

    interval_metrics = (
        evaluate_prediction_intervals(
            y,
            intervals,
        )
    )

    quantile_metrics = (
        evaluate_quantile_prediction_intervals(
            y,
            intervals,
            lower_quantile=(
                DEFAULT_LOWER_QUANTILE
            ),
            median_quantile=(
                DEFAULT_MEDIAN_QUANTILE
            ),
            upper_quantile=(
                DEFAULT_UPPER_QUANTILE
            ),
        )
    )

    median_prediction_mae_ns = float(
        np.mean(
            np.abs(
                intervals.prediction
                - y
            )
        )
    )

    return (
        QuantileGradientBoostingExternalResult(
            condition_id=(
                condition_id.strip()
            ),
            intervals=intervals,
            interval_metrics=(
                interval_metrics
            ),
            quantile_metrics=(
                quantile_metrics
            ),
            median_prediction_mae_ns=(
                median_prediction_mae_ns
            ),
            feature_names=feature_names,
        )
    )


def evaluate_week9_quantile_robustness_conditions(
    calibration_result: QuantileGradientBoostingCalibrationResult,
    prepared: GeneralizationPreparedData,
    *,
    definition: GeneralizationSuiteDefinition,
) -> QuantileGradientBoostingRobustnessResult:
    """Evaluate the frozen quantile estimator on Day-58 OOD conditions.

    The evaluation is restricted to the five conditional regimes
    frozen by the Week 9 protocol:

    - Test B low-photon OOD;
    - Test B high-photon OOD;
    - Test D elevated background;
    - Test F weak bi-exponential mismatch;
    - Test F moderate bi-exponential mismatch.

    Test-F coverage and error are evaluated relative to the primary
    dominant-component lifetime tau_1.

    No estimator fitting or uncertainty calibration is performed.
    """

    required_test_ids = {
        "B",
        "D",
        "F",
    }

    missing_tests = (
        required_test_ids
        - set(prepared.tests)
    )

    if missing_tests:
        raise ValueError(
            "Prepared generalization data are missing tests: "
            + ", ".join(
                sorted(missing_tests)
            )
        )

    missing_features = (
        required_test_ids
        - set(prepared.X_features)
    )

    if missing_features:
        raise ValueError(
            "Prepared engineered features are missing tests: "
            + ", ".join(
                sorted(missing_features)
            )
        )

    test_b = prepared.tests["B"]
    test_d = prepared.tests["D"]
    test_f = prepared.tests["F"]

    X_b = prepared.X_features["B"]
    X_d = prepared.X_features["D"]
    X_f = prepared.X_features["F"]

    _validate_external_test_alignment(
        test_b.y,
        X_b,
        test_id="B",
    )

    _validate_external_test_alignment(
        test_d.y,
        X_d,
        test_id="D",
    )

    _validate_external_test_alignment(
        test_f.y,
        X_f,
        test_id="F",
    )

    numerics = definition.numerics

    # ---------------------------------------------------------
    # Test B: low- versus high-photon OOD
    # ---------------------------------------------------------

    b_photon_counts = (
        test_b.metadata[
            "signal_photon_count_target"
        ].to_numpy(
            dtype=np.int64,
        )
    )

    b_low_mask = np.isin(
        b_photon_counts,
        np.asarray(
            numerics.test_b_low_photon_counts,
            dtype=np.int64,
        ),
    )

    b_high_mask = np.isin(
        b_photon_counts,
        np.asarray(
            numerics.test_b_high_photon_counts,
            dtype=np.int64,
        ),
    )

    if np.any(
        b_low_mask
        & b_high_mask
    ):
        raise ValueError(
            "Test-B low- and high-photon regimes must be disjoint."
        )

    if not np.all(
        b_low_mask
        | b_high_mask
    ):
        raise ValueError(
            "Every Test-B sample must belong to either "
            "the low- or high-photon OOD regime."
        )

    # ---------------------------------------------------------
    # Test D: elevated background
    # ---------------------------------------------------------

    d_background = (
        test_d.metadata[
            "background_per_bin"
        ].to_numpy(
            dtype=np.float64,
        )
    )

    d_background_mask = np.isin(
        d_background,
        np.asarray(
            numerics.test_d_background_levels,
            dtype=np.float64,
        ),
    )

    if not np.all(
        d_background_mask
    ):
        raise ValueError(
            "Every Test-D sample must belong to the "
            "frozen elevated-background regime."
        )

    # ---------------------------------------------------------
    # Test F: weak versus moderate model mismatch
    # ---------------------------------------------------------

    required_f_columns = {
        "primary_lifetime_ns",
        "secondary_fraction",
        "model_mismatch_severity",
    }

    missing_f_columns = (
        required_f_columns
        - set(test_f.metadata.columns)
    )

    if missing_f_columns:
        raise ValueError(
            "Test-F metadata are missing columns: "
            + ", ".join(
                sorted(missing_f_columns)
            )
        )

    primary_lifetime = (
        test_f.metadata[
            "primary_lifetime_ns"
        ].to_numpy(
            dtype=np.float64,
        )
    )

    if not np.allclose(
        primary_lifetime,
        test_f.y,
    ):
        raise ValueError(
            "Test-F targets must equal the primary "
            "dominant-component lifetime."
        )

    f_severity = (
        test_f.metadata[
            "model_mismatch_severity"
        ].astype(str).to_numpy()
    )

    f_weak_mask = (
        f_severity == "weak"
    )

    f_moderate_mask = (
        f_severity == "moderate"
    )

    if not np.all(
        f_weak_mask
        | f_moderate_mask
    ):
        raise ValueError(
            "Every Test-F sample must have weak or "
            "moderate mismatch severity."
        )

    weak_fractions = np.unique(
        test_f.metadata.loc[
            f_weak_mask,
            "secondary_fraction",
        ].to_numpy(
            dtype=np.float64,
        )
    )

    moderate_fractions = np.unique(
        test_f.metadata.loc[
            f_moderate_mask,
            "secondary_fraction",
        ].to_numpy(
            dtype=np.float64,
        )
    )

    if (
        weak_fractions.size != 1
        or not np.isclose(
            weak_fractions[0],
            numerics.test_f_secondary_fractions[0],
        )
    ):
        raise ValueError(
            "Weak Test-F samples do not match the "
            "frozen secondary fraction."
        )

    if (
        moderate_fractions.size != 1
        or not np.isclose(
            moderate_fractions[0],
            numerics.test_f_secondary_fractions[1],
        )
    ):
        raise ValueError(
            "Moderate Test-F samples do not match the "
            "frozen secondary fraction."
        )

    condition_definitions = (
        (
            "B_low_photon",
            "B",
            X_b,
            test_b.y,
            b_low_mask,
        ),
        (
            "B_high_photon",
            "B",
            X_b,
            test_b.y,
            b_high_mask,
        ),
        (
            "D_high_background",
            "D",
            X_d,
            test_d.y,
            d_background_mask,
        ),
        (
            "F_weak_mismatch",
            "F",
            X_f,
            test_f.y,
            f_weak_mask,
        ),
        (
            "F_moderate_mismatch",
            "F",
            X_f,
            test_f.y,
            f_moderate_mask,
        ),
    )

    condition_results: dict[
        str,
        QuantileGradientBoostingExternalResult,
    ] = {}

    condition_test_ids: dict[
        str,
        str,
    ] = {}

    for (
        condition_id,
        test_id,
        X_features,
        y_true,
        mask,
    ) in condition_definitions:

        indices = np.flatnonzero(
            mask
        )

        if indices.size == 0:
            raise ValueError(
                f"{condition_id} contains no samples."
            )

        condition_features = (
            X_features.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        )

        condition_y = np.asarray(
            y_true[
                indices
            ],
            dtype=np.float64,
        )

        condition_results[
            condition_id
        ] = (
            evaluate_frozen_quantile_gradient_boosting(
                calibration_result,
                condition_features,
                condition_y,
                condition_id=condition_id,
            )
        )

        condition_test_ids[
            condition_id
        ] = test_id

    summary = (
        _build_quantile_robustness_summary(
            calibration_result,
            condition_results,
            condition_test_ids=(
                condition_test_ids
            ),
        )
    )

    return (
        QuantileGradientBoostingRobustnessResult(
            condition_results=(
                condition_results
            ),
            summary=summary,
        )
    )


def evaluate_week9_paired_quantile_response(
    calibration_result: QuantileGradientBoostingCalibrationResult,
    prepared: GeneralizationPreparedData,
    *,
    definition: GeneralizationSuiteDefinition,
) -> QuantileGradientBoostingPairedRobustnessResult:
    """Compare each Day-58 shifted curve with its matched Test-A curve.

    Matching is performed by the frozen Week-8 ``pair_id``.

    The same frozen quantile estimator is used for both Test A and
    the shifted condition. No fitting, calibration, or threshold
    selection is performed.

    Test F continues to use the dominant-component lifetime tau_1
    as the primary scoring reference.
    """

    required_test_ids = {
        "A",
        "B",
        "D",
        "F",
    }

    missing_tests = (
        required_test_ids
        - set(prepared.tests)
    )

    if missing_tests:
        raise ValueError(
            "Prepared generalization data are missing tests: "
            + ", ".join(
                sorted(missing_tests)
            )
        )

    missing_features = (
        required_test_ids
        - set(prepared.X_features)
    )

    if missing_features:
        raise ValueError(
            "Prepared engineered features are missing tests: "
            + ", ".join(
                sorted(missing_features)
            )
        )

    test_a = prepared.tests["A"]
    test_b = prepared.tests["B"]
    test_d = prepared.tests["D"]
    test_f = prepared.tests["F"]

    X_a = prepared.X_features["A"]
    X_b = prepared.X_features["B"]
    X_d = prepared.X_features["D"]
    X_f = prepared.X_features["F"]

    for test_id, test, X_features in (
        (
            "A",
            test_a,
            X_a,
        ),
        (
            "B",
            test_b,
            X_b,
        ),
        (
            "D",
            test_d,
            X_d,
        ),
        (
            "F",
            test_f,
            X_f,
        ),
    ):
        _validate_external_test_alignment(
            test.y,
            X_features,
            test_id=test_id,
        )

        if "pair_id" not in test.metadata.columns:
            raise ValueError(
                f"Test {test_id} metadata must contain pair_id."
            )

        pair_ids = test.metadata[
            "pair_id"
        ].to_numpy(
            dtype=np.int64,
        )

        if np.unique(
            pair_ids
        ).size != pair_ids.size:
            raise ValueError(
                f"Test {test_id} pair_id values must be unique."
            )

    numerics = definition.numerics

    # ---------------------------------------------------------
    # Build the same five frozen Day-58 condition masks.
    # ---------------------------------------------------------

    b_photon_counts = (
        test_b.metadata[
            "signal_photon_count_target"
        ].to_numpy(
            dtype=np.int64,
        )
    )

    b_low_mask = np.isin(
        b_photon_counts,
        np.asarray(
            numerics.test_b_low_photon_counts,
            dtype=np.int64,
        ),
    )

    b_high_mask = np.isin(
        b_photon_counts,
        np.asarray(
            numerics.test_b_high_photon_counts,
            dtype=np.int64,
        ),
    )

    d_background = (
        test_d.metadata[
            "background_per_bin"
        ].to_numpy(
            dtype=np.float64,
        )
    )

    d_background_mask = np.isin(
        d_background,
        np.asarray(
            numerics.test_d_background_levels,
            dtype=np.float64,
        ),
    )

    f_severity = (
        test_f.metadata[
            "model_mismatch_severity"
        ]
        .astype(str)
        .to_numpy()
    )

    f_weak_mask = (
        f_severity == "weak"
    )

    f_moderate_mask = (
        f_severity == "moderate"
    )

    condition_definitions = (
        (
            "B_low_photon",
            "B",
            test_b,
            X_b,
            b_low_mask,
        ),
        (
            "B_high_photon",
            "B",
            test_b,
            X_b,
            b_high_mask,
        ),
        (
            "D_high_background",
            "D",
            test_d,
            X_d,
            d_background_mask,
        ),
        (
            "F_weak_mismatch",
            "F",
            test_f,
            X_f,
            f_weak_mask,
        ),
        (
            "F_moderate_mismatch",
            "F",
            test_f,
            X_f,
            f_moderate_mask,
        ),
    )

    a_pair_ids = test_a.metadata[
        "pair_id"
    ].to_numpy(
        dtype=np.int64,
    )

    a_index_by_pair = {
        int(pair_id): index
        for index, pair_id
        in enumerate(a_pair_ids)
    }

    condition_results: dict[
        str,
        QuantileGradientBoostingPairedConditionResult,
    ] = {}

    for (
        condition_id,
        shifted_test_id,
        shifted_test,
        shifted_features,
        mask,
    ) in condition_definitions:

        shifted_indices = np.flatnonzero(
            mask
        )

        if shifted_indices.size == 0:
            raise ValueError(
                f"{condition_id} contains no samples."
            )

        shifted_pair_ids = (
            shifted_test.metadata.iloc[
                shifted_indices
            ][
                "pair_id"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        try:
            reference_indices = np.asarray(
                [
                    a_index_by_pair[
                        int(pair_id)
                    ]
                    for pair_id
                    in shifted_pair_ids
                ],
                dtype=np.int64,
            )

        except KeyError as exc:
            raise ValueError(
                "Every shifted pair_id must have a "
                "matching Test-A reference."
            ) from exc

        reference_y = np.asarray(
            test_a.y[
                reference_indices
            ],
            dtype=np.float64,
        )

        shifted_y = np.asarray(
            shifted_test.y[
                shifted_indices
            ],
            dtype=np.float64,
        )

        if not np.allclose(
            reference_y,
            shifted_y,
        ):
            raise ValueError(
                f"{condition_id} and Test A must preserve "
                "paired lifetime targets."
            )

        reference_features = (
            X_a.iloc[
                reference_indices
            ]
            .reset_index(
                drop=True
            )
        )

        condition_features = (
            shifted_features.iloc[
                shifted_indices
            ]
            .reset_index(
                drop=True
            )
        )

        reference_result = (
            evaluate_frozen_quantile_gradient_boosting(
                calibration_result,
                reference_features,
                reference_y,
                condition_id=(
                    f"A_reference_for_{condition_id}"
                ),
            )
        )

        shifted_result = (
            evaluate_frozen_quantile_gradient_boosting(
                calibration_result,
                condition_features,
                shifted_y,
                condition_id=condition_id,
            )
        )

        diagnostics = (
            _build_paired_quantile_diagnostics(
                pair_ids=shifted_pair_ids,
                true_lifetimes_ns=(
                    shifted_y
                ),
                reference_intervals=(
                    reference_result.intervals
                ),
                shifted_intervals=(
                    shifted_result.intervals
                ),
            )
        )

        condition_results[
            condition_id
        ] = (
            QuantileGradientBoostingPairedConditionResult(
                condition_id=condition_id,
                shifted_test_id=(
                    shifted_test_id
                ),
                reference_result=(
                    reference_result
                ),
                shifted_result=(
                    shifted_result
                ),
                diagnostics=diagnostics,
            )
        )

    summary = (
        _build_paired_quantile_response_summary(
            condition_results
        )
    )

    return (
        QuantileGradientBoostingPairedRobustnessResult(
            condition_results=(
                condition_results
            ),
            summary=summary,
        )
    )


@dataclass(frozen=True)
class MLUncertaintyScoreExternalResult:
    """Evaluation of one frozen ML uncertainty-score estimator."""

    condition_id: str

    scores: UncertaintyScoreResult

    score_metrics: UncertaintyScoreMetrics

    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class MLUncertaintyScorePairedConditionResult:
    """Matched Test-A versus shifted-condition score response."""

    condition_id: str
    shifted_test_id: str

    reference_result: MLUncertaintyScoreExternalResult

    shifted_result: MLUncertaintyScoreExternalResult

    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class MLUncertaintyScorePairedRobustnessResult:
    """Paired robustness analysis for one ML uncertainty score."""

    condition_results: dict[
        str,
        MLUncertaintyScorePairedConditionResult,
    ]

    summary: pd.DataFrame


def evaluate_week9_paired_ml_uncertainty_response(
    calibration_result: MLUncertaintyScoreCalibrationResult,
    prepared: GeneralizationPreparedData,
    *,
    definition: GeneralizationSuiteDefinition,
) -> MLUncertaintyScorePairedRobustnessResult:
    """Compare ML uncertainty scores under matched A-to-shift pairs.

    The frozen score-producing estimator is applied to matched
    Test-A and shifted Test-B/D/F samples.

    No fitting, bootstrap resampling, calibration, or threshold
    selection is performed using final robustness Tests A-F.

    The analysis asks two related questions:

    1. Within each condition, do larger uncertainty scores tend to
       accompany larger absolute errors?

    2. For matched physical samples, does a shift-induced increase
       in uncertainty score accompany the corresponding increase
       in absolute error?

    Test F continues to use the dominant-component lifetime tau_1
    as the primary scoring reference.
    """

    required_test_ids = {
        "A",
        "B",
        "D",
        "F",
    }

    missing_tests = (
        required_test_ids
        - set(prepared.tests)
    )

    if missing_tests:
        raise ValueError(
            "Prepared generalization data are missing tests: "
            + ", ".join(
                sorted(missing_tests)
            )
        )

    missing_features = (
        required_test_ids
        - set(prepared.X_features)
    )

    if missing_features:
        raise ValueError(
            "Prepared engineered features are missing tests: "
            + ", ".join(
                sorted(missing_features)
            )
        )

    test_a = prepared.tests["A"]
    test_b = prepared.tests["B"]
    test_d = prepared.tests["D"]
    test_f = prepared.tests["F"]

    X_a = prepared.X_features["A"]
    X_b = prepared.X_features["B"]
    X_d = prepared.X_features["D"]
    X_f = prepared.X_features["F"]

    for test_id, test, X_features in (
        ("A", test_a, X_a),
        ("B", test_b, X_b),
        ("D", test_d, X_d),
        ("F", test_f, X_f),
    ):
        _validate_external_test_alignment(
            test.y,
            X_features,
            test_id=test_id,
        )

        if "pair_id" not in test.metadata.columns:
            raise ValueError(
                f"Test {test_id} metadata must contain pair_id."
            )

        pair_ids = (
            test.metadata[
                "pair_id"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        if (
            np.unique(pair_ids).size
            != pair_ids.size
        ):
            raise ValueError(
                f"Test {test_id} pair_id values must be unique."
            )

    numerics = definition.numerics

    # ---------------------------------------------------------
    # Test B: low- and high-photon OOD
    # ---------------------------------------------------------

    b_photon_counts = (
        test_b.metadata[
            "signal_photon_count_target"
        ]
        .to_numpy(
            dtype=np.int64,
        )
    )

    b_low_mask = np.isin(
        b_photon_counts,
        np.asarray(
            numerics.test_b_low_photon_counts,
            dtype=np.int64,
        ),
    )

    b_high_mask = np.isin(
        b_photon_counts,
        np.asarray(
            numerics.test_b_high_photon_counts,
            dtype=np.int64,
        ),
    )

    if np.any(
        b_low_mask
        & b_high_mask
    ):
        raise ValueError(
            "Test-B low- and high-photon regimes must be disjoint."
        )

    if not np.all(
        b_low_mask
        | b_high_mask
    ):
        raise ValueError(
            "Every Test-B sample must belong to either "
            "the low- or high-photon OOD regime."
        )

    # ---------------------------------------------------------
    # Test D: elevated background
    # ---------------------------------------------------------

    d_background = (
        test_d.metadata[
            "background_per_bin"
        ]
        .to_numpy(
            dtype=np.float64,
        )
    )

    d_background_mask = np.isin(
        d_background,
        np.asarray(
            numerics.test_d_background_levels,
            dtype=np.float64,
        ),
    )

    if not np.all(
        d_background_mask
    ):
        raise ValueError(
            "Every Test-D sample must belong to the "
            "frozen elevated-background regime."
        )

    # ---------------------------------------------------------
    # Test F: controlled model mismatch
    # ---------------------------------------------------------

    required_f_columns = {
        "primary_lifetime_ns",
        "secondary_fraction",
        "model_mismatch_severity",
    }

    missing_f_columns = (
        required_f_columns
        - set(test_f.metadata.columns)
    )

    if missing_f_columns:
        raise ValueError(
            "Test-F metadata are missing columns: "
            + ", ".join(
                sorted(missing_f_columns)
            )
        )

    primary_lifetime = (
        test_f.metadata[
            "primary_lifetime_ns"
        ]
        .to_numpy(
            dtype=np.float64,
        )
    )

    if not np.allclose(
        primary_lifetime,
        test_f.y,
    ):
        raise ValueError(
            "Test-F targets must equal the primary "
            "dominant-component lifetime."
        )

    f_severity = (
        test_f.metadata[
            "model_mismatch_severity"
        ]
        .astype(str)
        .to_numpy()
    )

    f_weak_mask = (
        f_severity == "weak"
    )

    f_moderate_mask = (
        f_severity == "moderate"
    )

    if not np.all(
        f_weak_mask
        | f_moderate_mask
    ):
        raise ValueError(
            "Every Test-F sample must have weak or "
            "moderate mismatch severity."
        )

    # ---------------------------------------------------------
    # Five Day-59 conditions
    # ---------------------------------------------------------

    condition_definitions = (
        (
            "B_low_photon",
            "B",
            test_b,
            X_b,
            b_low_mask,
        ),
        (
            "B_high_photon",
            "B",
            test_b,
            X_b,
            b_high_mask,
        ),
        (
            "D_high_background",
            "D",
            test_d,
            X_d,
            d_background_mask,
        ),
        (
            "F_weak_mismatch",
            "F",
            test_f,
            X_f,
            f_weak_mask,
        ),
        (
            "F_moderate_mismatch",
            "F",
            test_f,
            X_f,
            f_moderate_mask,
        ),
    )

    a_pair_ids = (
        test_a.metadata[
            "pair_id"
        ]
        .to_numpy(
            dtype=np.int64,
        )
    )

    a_index_by_pair = {
        int(pair_id): index
        for index, pair_id
        in enumerate(a_pair_ids)
    }

    condition_results: dict[
        str,
        MLUncertaintyScorePairedConditionResult,
    ] = {}

    for (
        condition_id,
        shifted_test_id,
        shifted_test,
        shifted_features,
        mask,
    ) in condition_definitions:

        shifted_indices = np.flatnonzero(
            mask
        )

        if shifted_indices.size == 0:
            raise ValueError(
                f"{condition_id} contains no samples."
            )

        shifted_pair_ids = (
            shifted_test.metadata.iloc[
                shifted_indices
            ][
                "pair_id"
            ]
            .to_numpy(
                dtype=np.int64,
            )
        )

        try:
            reference_indices = np.asarray(
                [
                    a_index_by_pair[
                        int(pair_id)
                    ]
                    for pair_id
                    in shifted_pair_ids
                ],
                dtype=np.int64,
            )

        except KeyError as exc:
            raise ValueError(
                "Every shifted pair_id must have a "
                "matching Test-A reference."
            ) from exc

        reference_y = np.asarray(
            test_a.y[
                reference_indices
            ],
            dtype=np.float64,
        )

        shifted_y = np.asarray(
            shifted_test.y[
                shifted_indices
            ],
            dtype=np.float64,
        )

        if not np.allclose(
            reference_y,
            shifted_y,
        ):
            raise ValueError(
                f"{condition_id} and Test A must preserve "
                "paired lifetime targets."
            )

        reference_features = (
            X_a.iloc[
                reference_indices
            ]
            .reset_index(
                drop=True
            )
        )

        condition_features = (
            shifted_features.iloc[
                shifted_indices
            ]
            .reset_index(
                drop=True
            )
        )

        reference_result = (
            evaluate_frozen_ml_uncertainty_score(
                calibration_result,
                reference_features,
                reference_y,
                condition_id=(
                    f"A_reference_for_{condition_id}"
                ),
            )
        )

        shifted_result = (
            evaluate_frozen_ml_uncertainty_score(
                calibration_result,
                condition_features,
                shifted_y,
                condition_id=condition_id,
            )
        )

        diagnostics = (
            _build_paired_ml_uncertainty_diagnostics(
                pair_ids=shifted_pair_ids,
                true_lifetimes_ns=shifted_y,
                reference_scores=(
                    reference_result.scores
                ),
                shifted_scores=(
                    shifted_result.scores
                ),
            )
        )

        condition_results[
            condition_id
        ] = (
            MLUncertaintyScorePairedConditionResult(
                condition_id=condition_id,
                shifted_test_id=(
                    shifted_test_id
                ),
                reference_result=(
                    reference_result
                ),
                shifted_result=(
                    shifted_result
                ),
                diagnostics=diagnostics,
            )
        )

    summary = (
        _build_paired_ml_uncertainty_summary(
            condition_results
        )
    )

    return (
        MLUncertaintyScorePairedRobustnessResult(
            condition_results=(
                condition_results
            ),
            summary=summary,
        )
    )


def evaluate_frozen_ml_uncertainty_score(
    calibration_result: MLUncertaintyScoreCalibrationResult,
    X_features: pd.DataFrame,
    y_true_ns: NDArray[np.float64],
    *,
    condition_id: str,
) -> MLUncertaintyScoreExternalResult:
    """Evaluate a frozen score estimator without refitting."""

    if not condition_id.strip():
        raise ValueError(
            "condition_id must not be empty."
        )

    if not isinstance(
        X_features,
        pd.DataFrame,
    ):
        raise TypeError(
            "X_features must be a pandas DataFrame."
        )

    feature_names = tuple(
        X_features.columns
    )

    if (
        feature_names
        != calibration_result.feature_names
    ):
        raise ValueError(
            "External feature schema must match "
            "the uncertainty-development feature schema."
        )

    X = X_features.to_numpy(
        dtype=np.float64,
    )

    y = np.asarray(
        y_true_ns,
        dtype=np.float64,
    )

    X, y = _validate_ml_uncertainty_training_data(
        X,
        y,
    )

    scores = (
        calibration_result
        .fitted_estimator
        .predict(X)
    )

    metrics = evaluate_uncertainty_scores(
        y,
        scores,
    )

    return MLUncertaintyScoreExternalResult(
        condition_id=condition_id.strip(),
        scores=scores,
        score_metrics=metrics,
        feature_names=feature_names,
    )


ML_UNCERTAINTY_METHODS = {
    QUANTILE_GRADIENT_BOOSTING_METHOD_ID: UncertaintyMethodDefinition(
        method_id=QUANTILE_GRADIENT_BOOSTING_METHOD_ID,
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=(
            "Conditional predictive quantiles learned from the uncertainty-training distribution."
        ),
        calibration_scope=(
            "Nominal coverage is not guaranteed under finite-sample or distribution shift and must be measured on held-out data."
        ),
    ),
    RANDOM_FOREST_TREE_SPREAD_METHOD_ID: UncertaintyMethodDefinition(
        method_id=RANDOM_FOREST_TREE_SPREAD_METHOD_ID,
        output_kind=UncertaintyOutputKind.UNCERTAINTY_SCORE,
        interpretation=(
            "Tree-to-tree ensemble disagreement used as an uncertainty ranking heuristic."
        ),
        calibration_scope=(
            "No nominal prediction-interval coverage claim is made unless a separate calibration procedure is applied."
        ),
    ),
    ML_TRAINING_BOOTSTRAP_METHOD_ID: UncertaintyMethodDefinition(
        method_id=ML_TRAINING_BOOTSTRAP_METHOD_ID,
        output_kind=UncertaintyOutputKind.UNCERTAINTY_SCORE,
        interpretation=(
            "Sensitivity of predictions to uncertainty-training sample variation."
        ),
        calibration_scope=(
            "Bootstrap spread is not automatically a calibrated predictive interval for lifetime error."
        ),
    ),
    "split_conformal": UncertaintyMethodDefinition(
        method_id="split_conformal",
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=(
            "Residual-based marginal prediction intervals calibrated on a dedicated development calibration subset."
        ),
        calibration_scope=(
            "Finite-sample marginal coverage is targeted under exchangeability; coverage can degrade under distribution shift."
        ),
    ),
}


def _validate_uncertainty_split_for_dataset(
    split: UncertaintyDevelopmentSplit,
    *,
    n_samples: int,
) -> None:
    """Validate that an uncertainty split partitions one dataset."""

    expected_indices = set(
        range(n_samples)
    )

    training_indices = set(
        split.training_indices
    )

    calibration_indices = set(
        split.calibration_indices
    )

    observed_indices = (
        training_indices
        | calibration_indices
    )

    if observed_indices != expected_indices:
        raise ValueError(
            "Uncertainty split must partition all "
            "development samples exactly once."
        )


def _validate_external_test_alignment(
    y: NDArray[np.float64],
    X_features: pd.DataFrame,
    *,
    test_id: str,
) -> None:
    """Validate feature/target alignment for one external test."""

    if X_features.shape[0] != y.size:
        raise ValueError(
            f"Test {test_id} engineered features and "
            "targets must contain the same number of samples."
        )

    if y.size == 0:
        raise ValueError(
            f"Test {test_id} must not be empty."
        )


def _build_quantile_robustness_summary(
    calibration_result: QuantileGradientBoostingCalibrationResult,
    condition_results: dict[
        str,
        QuantileGradientBoostingExternalResult,
    ],
    *,
    condition_test_ids: dict[str, str],
) -> pd.DataFrame:
    """Build the Day-58 conditional uncertainty summary."""

    calibration_interval = (
        calibration_result.interval_metrics
    )

    calibration_quantile = (
        calibration_result.quantile_metrics
    )

    calibration_mean_width = (
        calibration_interval
        .mean_interval_width
    )

    calibration_median_width = (
        calibration_interval
        .median_interval_width
    )

    # For q = 0.50, pinball loss is exactly
    # one half of mean absolute error.
    calibration_median_mae = (
        2.0
        * calibration_quantile
        .median_pinball_loss
    )

    rows: list[
        dict[str, str | int | float]
    ] = []

    for (
        condition_id,
        result,
    ) in condition_results.items():

        interval = (
            result.interval_metrics
        )

        quantile = (
            result.quantile_metrics
        )

        mean_width_ratio = (
            _safe_positive_ratio(
                interval.mean_interval_width,
                calibration_mean_width,
            )
        )

        median_width_ratio = (
            _safe_positive_ratio(
                interval.median_interval_width,
                calibration_median_width,
            )
        )

        mae_ratio = (
            _safe_positive_ratio(
                result.median_prediction_mae_ns,
                calibration_median_mae,
            )
        )

        rows.append(
            {
                "condition_id": condition_id,
                "test_id": (
                    condition_test_ids[
                        condition_id
                    ]
                ),
                "n_samples": (
                    interval.n_samples
                ),
                "empirical_coverage": (
                    interval.empirical_coverage
                ),
                "coverage_error": (
                    interval.coverage_error
                ),
                "mean_interval_width_ns": (
                    interval.mean_interval_width
                ),
                "median_interval_width_ns": (
                    interval.median_interval_width
                ),
                "median_prediction_mae_ns": (
                    result
                    .median_prediction_mae_ns
                ),
                "lower_pinball_loss": (
                    quantile.lower_pinball_loss
                ),
                "median_pinball_loss": (
                    quantile.median_pinball_loss
                ),
                "upper_pinball_loss": (
                    quantile.upper_pinball_loss
                ),
                "mean_pinball_loss": (
                    quantile.mean_pinball_loss
                ),
                "quantile_crossing_rate": (
                    quantile.quantile_crossing_rate
                ),
                "interval_failure_rate": (
                    interval.interval_failure_rate
                ),
                "calibration_coverage": (
                    calibration_interval
                    .empirical_coverage
                ),
                "calibration_mean_width_ns": (
                    calibration_mean_width
                ),
                "calibration_median_width_ns": (
                    calibration_median_width
                ),
                "calibration_median_prediction_mae_ns": (
                    calibration_median_mae
                ),
                "mean_width_ratio_to_calibration": (
                    mean_width_ratio
                ),
                "median_width_ratio_to_calibration": (
                    median_width_ratio
                ),
                "mae_ratio_to_calibration": (
                    mae_ratio
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _safe_positive_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a finite ratio when the reference is usable."""

    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return float("nan")

    return float(
        numerator
        / denominator
    )


def _build_paired_quantile_diagnostics(
    *,
    pair_ids: NDArray[np.int64],
    true_lifetimes_ns: NDArray[np.float64],
    reference_intervals: PredictionIntervalResult,
    shifted_intervals: PredictionIntervalResult,
) -> pd.DataFrame:
    """Build matched Test-A versus shifted per-pair diagnostics."""

    pair_ids = np.asarray(
        pair_ids,
        dtype=np.int64,
    )

    y = np.asarray(
        true_lifetimes_ns,
        dtype=np.float64,
    )

    n_samples = y.size

    if pair_ids.shape != (
        n_samples,
    ):
        raise ValueError(
            "pair_ids and true lifetimes must have the same shape."
        )

    if (
        reference_intervals.prediction.shape
        != (n_samples,)
        or shifted_intervals.prediction.shape
        != (n_samples,)
    ):
        raise ValueError(
            "Paired interval predictions must match "
            "the number of paired samples."
        )

    reference_valid = (
        reference_intervals.valid_interval_mask
    )

    shifted_valid = (
        shifted_intervals.valid_interval_mask
    )

    paired_valid = (
        reference_valid
        & shifted_valid
    )

    reference_absolute_error = np.abs(
        reference_intervals.prediction
        - y
    )

    shifted_absolute_error = np.abs(
        shifted_intervals.prediction
        - y
    )

    reference_width = (
        reference_intervals.upper
        - reference_intervals.lower
    )

    shifted_width = (
        shifted_intervals.upper
        - shifted_intervals.lower
    )

    absolute_error_change = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    interval_width_change = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    absolute_error_change[
        paired_valid
    ] = (
        shifted_absolute_error[
            paired_valid
        ]
        - reference_absolute_error[
            paired_valid
        ]
    )

    interval_width_change[
        paired_valid
    ] = (
        shifted_width[
            paired_valid
        ]
        - reference_width[
            paired_valid
        ]
    )

    return pd.DataFrame(
        {
            "pair_id": pair_ids,
            "true_lifetime_ns": y,
            "reference_prediction_ns": (
                reference_intervals.prediction
            ),
            "shifted_prediction_ns": (
                shifted_intervals.prediction
            ),
            "reference_absolute_error_ns": (
                reference_absolute_error
            ),
            "shifted_absolute_error_ns": (
                shifted_absolute_error
            ),
            "absolute_error_change_ns": (
                absolute_error_change
            ),
            "reference_interval_width_ns": (
                reference_width
            ),
            "shifted_interval_width_ns": (
                shifted_width
            ),
            "interval_width_change_ns": (
                interval_width_change
            ),
            "reference_valid_interval": (
                reference_valid
            ),
            "shifted_valid_interval": (
                shifted_valid
            ),
            "paired_valid_interval": (
                paired_valid
            ),
        }
    )


def _build_paired_quantile_response_summary(
    condition_results: dict[
        str,
        QuantileGradientBoostingPairedConditionResult,
    ],
) -> pd.DataFrame:
    """Summarize paired error and uncertainty response."""

    rows: list[
        dict[str, str | int | float]
    ] = []

    for (
        condition_id,
        result,
    ) in condition_results.items():

        diagnostics = (
            result.diagnostics
        )

        valid = diagnostics[
            "paired_valid_interval"
        ].to_numpy(
            dtype=bool,
        )

        n_pairs = int(
            len(diagnostics)
        )

        n_valid_pairs = int(
            np.count_nonzero(
                valid
            )
        )

        if n_valid_pairs == 0:
            rows.append(
                {
                    "condition_id": (
                        condition_id
                    ),
                    "shifted_test_id": (
                        result.shifted_test_id
                    ),
                    "n_pairs": n_pairs,
                    "n_valid_pairs": 0,
                    "reference_mae_ns": np.nan,
                    "shifted_mae_ns": np.nan,
                    "mae_change_ns": np.nan,
                    "reference_mean_interval_width_ns": np.nan,
                    "shifted_mean_interval_width_ns": np.nan,
                    "mean_interval_width_change_ns": np.nan,
                    "median_absolute_error_change_ns": np.nan,
                    "median_interval_width_change_ns": np.nan,
                    "spearman_width_error_change": np.nan,
                    "fraction_error_increased": np.nan,
                    "fraction_width_increased": np.nan,
                }
            )

            continue

        valid_diagnostics = (
            diagnostics.loc[
                valid
            ]
        )

        reference_error = (
            valid_diagnostics[
                "reference_absolute_error_ns"
            ].to_numpy(
                dtype=np.float64,
            )
        )

        shifted_error = (
            valid_diagnostics[
                "shifted_absolute_error_ns"
            ].to_numpy(
                dtype=np.float64,
            )
        )

        error_change = (
            valid_diagnostics[
                "absolute_error_change_ns"
            ].to_numpy(
                dtype=np.float64,
            )
        )

        reference_width = (
            valid_diagnostics[
                "reference_interval_width_ns"
            ].to_numpy(
                dtype=np.float64,
            )
        )

        shifted_width = (
            valid_diagnostics[
                "shifted_interval_width_ns"
            ].to_numpy(
                dtype=np.float64,
            )
        )

        width_change = (
            valid_diagnostics[
                "interval_width_change_ns"
            ].to_numpy(
                dtype=np.float64,
            )
        )

        correlation = (
            _safe_spearman_correlation(
                width_change,
                error_change,
            )
        )

        rows.append(
            {
                "condition_id": (
                    condition_id
                ),
                "shifted_test_id": (
                    result.shifted_test_id
                ),
                "n_pairs": n_pairs,
                "n_valid_pairs": (
                    n_valid_pairs
                ),
                "reference_mae_ns": float(
                    np.mean(
                        reference_error
                    )
                ),
                "shifted_mae_ns": float(
                    np.mean(
                        shifted_error
                    )
                ),
                "mae_change_ns": float(
                    np.mean(
                        error_change
                    )
                ),
                "reference_mean_interval_width_ns": float(
                    np.mean(
                        reference_width
                    )
                ),
                "shifted_mean_interval_width_ns": float(
                    np.mean(
                        shifted_width
                    )
                ),
                "mean_interval_width_change_ns": float(
                    np.mean(
                        width_change
                    )
                ),
                "median_absolute_error_change_ns": float(
                    np.median(
                        error_change
                    )
                ),
                "median_interval_width_change_ns": float(
                    np.median(
                        width_change
                    )
                ),
                "spearman_width_error_change": (
                    correlation
                ),
                "fraction_error_increased": float(
                    np.mean(
                        error_change > 0.0
                    )
                ),
                "fraction_width_increased": float(
                    np.mean(
                        width_change > 0.0
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _safe_spearman_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Return Spearman correlation when both variables vary."""

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.shape != y.shape:
        raise ValueError(
            "Spearman inputs must have the same shape."
        )

    if x.size < 2:
        return float("nan")

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    if x.size < 2:
        return float("nan")

    if (
        np.all(
            x == x[0]
        )
        or np.all(
            y == y[0]
        )
    ):
        return float("nan")

    return float(
        spearmanr(
            x,
            y,
        ).statistic
    )


def _validate_ml_uncertainty_training_data(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate ML inputs used to fit an uncertainty estimator."""

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )

    if X.shape[0] != y.size:
        raise ValueError(
            "X and y must contain the same "
            "number of samples."
        )

    if y.size == 0:
        raise ValueError(
            "training data must not be empty."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X must contain at least one feature."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "X must contain only finite values."
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise ValueError(
            "y must contain only finite values."
        )

    if np.any(
        y <= 0.0
    ):
        raise ValueError(
            "lifetimes must be positive."
        )

    return X, y


def _validate_ml_uncertainty_prediction_features(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate features passed to a fitted uncertainty estimator."""

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "X must contain at least one sample."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X must contain at least one feature."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "X must contain only finite values."
        )

    return X


def _build_paired_ml_uncertainty_diagnostics(
    *,
    pair_ids: NDArray[np.int64],
    true_lifetimes_ns: NDArray[np.float64],
    reference_scores: UncertaintyScoreResult,
    shifted_scores: UncertaintyScoreResult,
) -> pd.DataFrame:
    """Build matched Test-A versus shifted score diagnostics."""

    pair_ids = np.asarray(
        pair_ids,
        dtype=np.int64,
    )

    y = np.asarray(
        true_lifetimes_ns,
        dtype=np.float64,
    )

    n_samples = int(
        y.size
    )

    if pair_ids.shape != (
        n_samples,
    ):
        raise ValueError(
            "pair_ids and true lifetimes must have the same shape."
        )

    if (
        reference_scores.prediction.shape
        != (n_samples,)
        or shifted_scores.prediction.shape
        != (n_samples,)
    ):
        raise ValueError(
            "Paired score predictions must match "
            "the number of paired samples."
        )

    if (
        reference_scores.method_id
        != shifted_scores.method_id
    ):
        raise ValueError(
            "Reference and shifted uncertainty scores "
            "must come from the same method."
        )

    reference_valid = (
        reference_scores.valid_score_mask
    )

    shifted_valid = (
        shifted_scores.valid_score_mask
    )

    paired_valid = (
        reference_valid
        & shifted_valid
    )

    reference_absolute_error = np.abs(
        reference_scores.prediction
        - y
    )

    shifted_absolute_error = np.abs(
        shifted_scores.prediction
        - y
    )

    absolute_error_change = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    uncertainty_score_change = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    absolute_error_change[
        paired_valid
    ] = (
        shifted_absolute_error[
            paired_valid
        ]
        - reference_absolute_error[
            paired_valid
        ]
    )

    uncertainty_score_change[
        paired_valid
    ] = (
        shifted_scores.uncertainty_score[
            paired_valid
        ]
        - reference_scores.uncertainty_score[
            paired_valid
        ]
    )

    return pd.DataFrame(
        {
            "pair_id": pair_ids,
            "true_lifetime_ns": y,
            "reference_prediction_ns": (
                reference_scores.prediction
            ),
            "shifted_prediction_ns": (
                shifted_scores.prediction
            ),
            "reference_absolute_error_ns": (
                reference_absolute_error
            ),
            "shifted_absolute_error_ns": (
                shifted_absolute_error
            ),
            "absolute_error_change_ns": (
                absolute_error_change
            ),
            "reference_uncertainty_score": (
                reference_scores
                .uncertainty_score
            ),
            "shifted_uncertainty_score": (
                shifted_scores
                .uncertainty_score
            ),
            "uncertainty_score_change": (
                uncertainty_score_change
            ),
            "reference_valid_score": (
                reference_valid
            ),
            "shifted_valid_score": (
                shifted_valid
            ),
            "paired_valid_score": (
                paired_valid
            ),
        }
    )


def _build_paired_ml_uncertainty_summary(
    condition_results: dict[
        str,
        MLUncertaintyScorePairedConditionResult,
    ],
) -> pd.DataFrame:
    """Summarize paired error and uncertainty-score response."""

    rows: list[
        dict[str, str | int | float]
    ] = []

    for (
        condition_id,
        result,
    ) in condition_results.items():

        diagnostics = (
            result.diagnostics
        )

        valid = (
            diagnostics[
                "paired_valid_score"
            ]
            .to_numpy(
                dtype=bool,
            )
        )

        n_pairs = int(
            len(diagnostics)
        )

        n_valid_pairs = int(
            np.count_nonzero(
                valid
            )
        )

        method_id = (
            result
            .shifted_result
            .scores
            .method_id
        )

        if n_valid_pairs == 0:
            rows.append(
                {
                    "method_id": method_id,
                    "condition_id": condition_id,
                    "shifted_test_id": (
                        result.shifted_test_id
                    ),
                    "n_pairs": n_pairs,
                    "n_valid_pairs": 0,
                    "reference_mae_ns": np.nan,
                    "shifted_mae_ns": np.nan,
                    "mae_change_ns": np.nan,
                    "mae_ratio": np.nan,
                    "reference_mean_uncertainty_score": np.nan,
                    "shifted_mean_uncertainty_score": np.nan,
                    "mean_uncertainty_score_change": np.nan,
                    "mean_uncertainty_score_ratio": np.nan,
                    "reference_score_error_correlation": np.nan,
                    "shifted_score_error_correlation": np.nan,
                    "median_absolute_error_change_ns": np.nan,
                    "median_uncertainty_score_change": np.nan,
                    "spearman_score_error_change": np.nan,
                    "fraction_error_increased": np.nan,
                    "fraction_score_increased": np.nan,
                }
            )

            continue

        valid_diagnostics = (
            diagnostics.loc[
                valid
            ]
        )

        reference_error = (
            valid_diagnostics[
                "reference_absolute_error_ns"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        shifted_error = (
            valid_diagnostics[
                "shifted_absolute_error_ns"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        error_change = (
            valid_diagnostics[
                "absolute_error_change_ns"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        reference_score = (
            valid_diagnostics[
                "reference_uncertainty_score"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        shifted_score = (
            valid_diagnostics[
                "shifted_uncertainty_score"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        score_change = (
            valid_diagnostics[
                "uncertainty_score_change"
            ]
            .to_numpy(
                dtype=np.float64,
            )
        )

        reference_mae = float(
            np.mean(
                reference_error
            )
        )

        shifted_mae = float(
            np.mean(
                shifted_error
            )
        )

        reference_mean_score = float(
            np.mean(
                reference_score
            )
        )

        shifted_mean_score = float(
            np.mean(
                shifted_score
            )
        )

        rows.append(
            {
                "method_id": method_id,
                "condition_id": condition_id,
                "shifted_test_id": (
                    result.shifted_test_id
                ),
                "n_pairs": n_pairs,
                "n_valid_pairs": (
                    n_valid_pairs
                ),
                "reference_mae_ns": (
                    reference_mae
                ),
                "shifted_mae_ns": (
                    shifted_mae
                ),
                "mae_change_ns": float(
                    np.mean(
                        error_change
                    )
                ),
                "mae_ratio": (
                    _safe_positive_ratio(
                        shifted_mae,
                        reference_mae,
                    )
                ),
                "reference_mean_uncertainty_score": (
                    reference_mean_score
                ),
                "shifted_mean_uncertainty_score": (
                    shifted_mean_score
                ),
                "mean_uncertainty_score_change": float(
                    np.mean(
                        score_change
                    )
                ),
                "mean_uncertainty_score_ratio": (
                    _safe_positive_ratio(
                        shifted_mean_score,
                        reference_mean_score,
                    )
                ),
                "reference_score_error_correlation": (
                    result
                    .reference_result
                    .score_metrics
                    .spearman_error_correlation
                ),
                "shifted_score_error_correlation": (
                    result
                    .shifted_result
                    .score_metrics
                    .spearman_error_correlation
                ),
                "median_absolute_error_change_ns": float(
                    np.median(
                        error_change
                    )
                ),
                "median_uncertainty_score_change": float(
                    np.median(
                        score_change
                    )
                ),
                "spearman_score_error_change": (
                    _safe_spearman_correlation(
                        score_change,
                        error_change,
                    )
                ),
                "fraction_error_increased": float(
                    np.mean(
                        error_change > 0.0
                    )
                ),
                "fraction_score_increased": float(
                    np.mean(
                        score_change > 0.0
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


