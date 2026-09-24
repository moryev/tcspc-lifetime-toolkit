"""Week 9 uncertainty robustness evaluation across frozen Tests A-F.

This module provides the final uncertainty-calibration analogue of the
Week 8 robustness scorecard.

All estimators and calibration corrections must be frozen before Tests
A-F are evaluated. Final robustness tests are external evaluation data
only and must never be used for model fitting, uncertainty calibration,
or threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from statistics import NormalDist

from tcspc_toolkit.evaluation import (
    calculate_poisson_deviance_residuals,
)
from tcspc_toolkit.classical_evaluation import (
    fit_single_reconvolution_curve,
)
from tcspc_toolkit.classical_uncertainty import (
    DEFAULT_CLASSICAL_BOOTSTRAP_REPLICATES,
    DEFAULT_CLASSICAL_NOMINAL_COVERAGE,
    PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID,
    POISSON_LOCAL_COVARIANCE_METHOD_ID,
    _reconstruct_reconvolution_fit_result,
    estimate_parametric_poisson_bootstrap,
    estimate_poisson_reconvolution_local_covariance,
)
from tcspc_toolkit.generalization import (
    FINAL_ROBUSTNESS_TEST_IDS,
    GeneralizationSuiteDefinition,
)
from tcspc_toolkit.generalization_datasets import (
    GeneralizationTestMeasurements,
)
from tcspc_toolkit.generalization_evaluation import (
    GeneralizationPreparedData,
)
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)
from tcspc_toolkit.ml_evaluation import (
    BenchmarkDataset,
)
from tcspc_toolkit.ml_uncertainty import (
    MLUncertaintyScoreCalibrationResult,
    QuantileGradientBoostingCalibrationResult,
    QuantileGradientBoostingExternalResult,
    evaluate_frozen_ml_uncertainty_score,
    evaluate_frozen_quantile_gradient_boosting,
)
from tcspc_toolkit.uncertainty_evaluation import (
    IntervalEvaluationMetrics,
    PredictionIntervalResult,
    evaluate_prediction_intervals,
)


DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED = 62_001


CONFORMALIZED_QUANTILE_METHOD_ID = (
    "conformalized_quantile_gradient_boosting"
)


@dataclass(frozen=True)
class ConformalizedQuantileCalibrationResult:
    """Split-conformal correction for the frozen quantile estimator.

    The correction is estimated exclusively from the held-out
    uncertainty-calibration subset.

    The underlying q05/q50/q95 models are not refitted.
    """

    base_calibration: (
        QuantileGradientBoostingCalibrationResult
    )

    correction_ns: float

    n_calibration_scores: int

    calibration_scores_ns: NDArray[np.float64]

    calibration_intervals: PredictionIntervalResult

    interval_metrics: IntervalEvaluationMetrics


def _finite_sample_conformal_quantile(
    scores: NDArray[np.float64],
    *,
    nominal_coverage: float,
) -> float:
    """Return the finite-sample split-conformal correction.

    For calibration size n and desired coverage 1-alpha, use

        k = ceil((n + 1) * (1 - alpha))

    and return the k-th ordered nonconformity score.

    The rank is capped at n for small calibration sets.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    if scores.size == 0:
        raise ValueError(
            "scores must not be empty."
        )

    if not np.all(
        np.isfinite(scores)
    ):
        raise ValueError(
            "scores must contain only finite values."
        )

    if np.any(
        scores < 0.0
    ):
        raise ValueError(
            "conformal nonconformity scores must be non-negative."
        )

    if not (
        0.0
        < nominal_coverage
        < 1.0
    ):
        raise ValueError(
            "nominal_coverage must lie strictly between 0 and 1."
        )

    n_scores = int(
        scores.size
    )

    rank = ceil(
        (n_scores + 1)
        * nominal_coverage
    )

    rank = min(
        rank,
        n_scores,
    )

    ordered = np.sort(
        scores
    )

    return float(
        ordered[
            rank - 1
        ]
    )


def conformalize_quantile_gradient_boosting(
    calibration_result: (
        QuantileGradientBoostingCalibrationResult
    ),
    development: BenchmarkDataset,
) -> ConformalizedQuantileCalibrationResult:
    """Conformalize the frozen q05/q50/q95 prediction interval.

    The quantile models have already been fitted on the
    uncertainty-training subset.

    Only the held-out calibration subset is used here.

    No final Test A-F data are involved.
    """

    y = np.asarray(
        development.y,
        dtype=np.float64,
    )

    calibration_indices = np.asarray(
        calibration_result
        .split
        .calibration_indices,
        dtype=np.int64,
    )

    if (
        calibration_indices.size
        != calibration_result
        .calibration_intervals
        .prediction
        .size
    ):
        raise ValueError(
            "Calibration indices and stored calibration "
            "intervals must contain the same number of samples."
        )

    y_calibration = y[
        calibration_indices
    ]

    intervals = (
        calibration_result
        .calibration_intervals
    )

    valid = (
        intervals.valid_interval_mask
        & np.isfinite(
            y_calibration
        )
    )

    if not np.any(
        valid
    ):
        raise ValueError(
            "At least one valid calibration interval is required."
        )

    y_valid = y_calibration[
        valid
    ]

    lower_valid = intervals.lower[
        valid
    ]

    upper_valid = intervals.upper[
        valid
    ]

    scores = np.maximum.reduce(
        [
            lower_valid
            - y_valid,

            y_valid
            - upper_valid,

            np.zeros_like(
                y_valid
            ),
        ]
    )

    correction = (
        _finite_sample_conformal_quantile(
            scores,
            nominal_coverage=(
                intervals.nominal_coverage
            ),
        )
    )

    conformal_intervals = (
        PredictionIntervalResult(
            prediction=(
                intervals.prediction.copy()
            ),
            lower=(
                intervals.lower
                - correction
            ),
            upper=(
                intervals.upper
                + correction
            ),
            nominal_coverage=(
                intervals.nominal_coverage
            ),
            method_id=(
                CONFORMALIZED_QUANTILE_METHOD_ID
            ),
        )
    )

    interval_metrics = (
        evaluate_prediction_intervals(
            y_calibration,
            conformal_intervals,
        )
    )

    return (
        ConformalizedQuantileCalibrationResult(
            base_calibration=(
                calibration_result
            ),
            correction_ns=(
                correction
            ),
            n_calibration_scores=int(
                scores.size
            ),
            calibration_scores_ns=(
                scores
            ),
            calibration_intervals=(
                conformal_intervals
            ),
            interval_metrics=(
                interval_metrics
            ),
        )
    )


@dataclass(frozen=True)
class ConformalizedQuantileExternalResult:
    """Frozen conformalized-quantile evaluation on one external test."""

    test_id: str

    intervals: PredictionIntervalResult

    interval_metrics: IntervalEvaluationMetrics

    median_prediction_mae_ns: float


def evaluate_frozen_conformalized_quantile(
    calibration_result: (
        ConformalizedQuantileCalibrationResult
    ),
    X_features: pd.DataFrame,
    y_true_ns: NDArray[np.float64],
    *,
    test_id: str,
) -> ConformalizedQuantileExternalResult:
    """Apply the frozen conformal correction to external data."""

    base_result: (
        QuantileGradientBoostingExternalResult
    ) = (
        evaluate_frozen_quantile_gradient_boosting(
            calibration_result
            .base_calibration,
            X_features,
            y_true_ns,
            condition_id=test_id,
        )
    )

    correction = (
        calibration_result
        .correction_ns
    )

    base_intervals = (
        base_result.intervals
    )

    intervals = PredictionIntervalResult(
        prediction=(
            base_intervals.prediction.copy()
        ),
        lower=(
            base_intervals.lower
            - correction
        ),
        upper=(
            base_intervals.upper
            + correction
        ),
        nominal_coverage=(
            base_intervals.nominal_coverage
        ),
        method_id=(
            CONFORMALIZED_QUANTILE_METHOD_ID
        ),
    )

    y = np.asarray(
        y_true_ns,
        dtype=np.float64,
    )

    metrics = (
        evaluate_prediction_intervals(
            y,
            intervals,
        )
    )

    mae = float(
        np.mean(
            np.abs(
                intervals.prediction
                - y
            )
        )
    )

    return (
        ConformalizedQuantileExternalResult(
            test_id=test_id,
            intervals=intervals,
            interval_metrics=metrics,
            median_prediction_mae_ns=mae,
        )
    )


def build_week9_interval_scorecard(
    *,
    prepared: GeneralizationPreparedData,
    quantile_calibration: (
        QuantileGradientBoostingCalibrationResult
    ),
    conformal_calibration: (
        ConformalizedQuantileCalibrationResult
    ),
) -> pd.DataFrame:
    """Build the final A-F interval-calibration scorecard.

    Test F is evaluated against the primary dominant-component
    lifetime tau_1 stored in the frozen Week-8 target vector.
    """

    rows: list[
        dict[str, object]
    ] = []

    for test_id in (
        FINAL_ROBUSTNESS_TEST_IDS
    ):
        if test_id not in prepared.tests:
            raise ValueError(
                f"Prepared data are missing Test {test_id}."
            )

        if test_id not in prepared.X_features:
            raise ValueError(
                f"Prepared features are missing Test {test_id}."
            )

        test = prepared.tests[
            test_id
        ]

        X_features = prepared.X_features[
            test_id
        ]

        y = np.asarray(
            test.y,
            dtype=np.float64,
        )

        quantile_result = (
            evaluate_frozen_quantile_gradient_boosting(
                quantile_calibration,
                X_features,
                y,
                condition_id=test_id,
            )
        )

        conformal_result = (
            evaluate_frozen_conformalized_quantile(
                conformal_calibration,
                X_features,
                y,
                test_id=test_id,
            )
        )

        for (
            method_id,
            mae,
            metrics,
        ) in (
            (
                quantile_result
                .intervals
                .method_id,

                quantile_result
                .median_prediction_mae_ns,

                quantile_result
                .interval_metrics,
            ),
            (
                conformal_result
                .intervals
                .method_id,

                conformal_result
                .median_prediction_mae_ns,

                conformal_result
                .interval_metrics,
            ),
        ):
            rows.append(
                {
                    "method": method_id,
                    "test_id": test_id,
                    "target_reference": (
                        "dominant_component_tau_1"
                        if test_id == "F"
                        else "monoexponential_lifetime"
                    ),
                    "mae_ns": mae,
                    "nominal_coverage": (
                        quantile_result
                        .intervals
                        .nominal_coverage
                    ),
                    "empirical_coverage": (
                        metrics
                        .empirical_coverage
                    ),
                    "coverage_gap": (
                        metrics
                        .coverage_error
                    ),
                    "mean_width_ns": (
                        metrics
                        .mean_interval_width
                    ),
                    "median_width_ns": (
                        metrics
                        .median_interval_width
                    ),
                    "interval_score": (
                        metrics
                        .mean_interval_score
                    ),
                    "interval_failure_rate": (
                        metrics
                        .interval_failure_rate
                    ),
                }
            )

    return (
        pd.DataFrame(
            rows
        )
    )


def build_week9_score_only_scorecard(
    *,
    prepared: GeneralizationPreparedData,
    calibration_results: dict[
        str,
        MLUncertaintyScoreCalibrationResult,
    ],
) -> pd.DataFrame:
    """Evaluate frozen score-only uncertainty methods across Tests A-F."""

    if not calibration_results:
        raise ValueError(
            "calibration_results must not be empty."
        )

    rows: list[
        dict[str, object]
    ] = []

    for (
        method_name,
        calibration_result,
    ) in calibration_results.items():

        for test_id in (
            FINAL_ROBUSTNESS_TEST_IDS
        ):
            test = prepared.tests[
                test_id
            ]

            X_features = (
                prepared.X_features[
                    test_id
                ]
            )

            result = (
                evaluate_frozen_ml_uncertainty_score(
                    calibration_result,
                    X_features,
                    np.asarray(
                        test.y,
                        dtype=np.float64,
                    ),
                    condition_id=test_id,
                )
            )

            metrics = (
                result.score_metrics
            )

            valid = (
                result.scores
                .valid_score_mask
            )

            if np.any(
                valid
            ):
                mean_spread = float(
                    np.mean(
                        result
                        .scores
                        .uncertainty_score[
                            valid
                        ]
                    )
                )
            else:
                mean_spread = float(
                    "nan"
                )

            rows.append(
                {
                    "method": method_name,
                    "test_id": test_id,
                    "target_reference": (
                        "dominant_component_tau_1"
                        if test_id == "F"
                        else "monoexponential_lifetime"
                    ),
                    "mae_ns": (
                        metrics
                        .mean_absolute_error_ns
                    ),
                    "mean_uncertainty_score": (
                        mean_spread
                    ),
                    "error_score_spearman": (
                        metrics
                        .spearman_error_correlation
                    ),
                    "low_uncertainty_mae_ns": (
                        metrics
                        .low_uncertainty_mae_ns
                    ),
                    "high_uncertainty_mae_ns": (
                        metrics
                        .high_uncertainty_mae_ns
                    ),
                    "score_failure_rate": (
                        metrics
                        .score_failure_rate
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


@dataclass(frozen=True)
class Week9MLUncertaintyRobustnessReport:
    """Final frozen ML uncertainty evaluation across Tests A-F."""

    interval_scorecard: pd.DataFrame

    score_only_scorecard: pd.DataFrame

    conformal_correction_ns: float


def build_week9_ml_uncertainty_robustness_report(
    *,
    prepared: GeneralizationPreparedData,
    quantile_calibration: (
        QuantileGradientBoostingCalibrationResult
    ),
    conformal_calibration: (
        ConformalizedQuantileCalibrationResult
    ),
    score_calibrations: dict[
        str,
        MLUncertaintyScoreCalibrationResult,
    ],
) -> Week9MLUncertaintyRobustnessReport:
    """Build the final frozen ML uncertainty scorecards."""

    interval_scorecard = (
        build_week9_interval_scorecard(
            prepared=prepared,
            quantile_calibration=(
                quantile_calibration
            ),
            conformal_calibration=(
                conformal_calibration
            ),
        )
    )

    score_only_scorecard = (
        build_week9_score_only_scorecard(
            prepared=prepared,
            calibration_results=(
                score_calibrations
            ),
        )
    )

    return (
        Week9MLUncertaintyRobustnessReport(
            interval_scorecard=(
                interval_scorecard
            ),
            score_only_scorecard=(
                score_only_scorecard
            ),
            conformal_correction_ns=(
                conformal_calibration
                .correction_ns
            ),
        )
    )


def _evaluate_classical_uncertainty_test(
    *,
    test: GeneralizationTestMeasurements,
    irf_centre_ns: float,
    temporal_shift_bounds: tuple[
        float,
        float,
    ],
    rng: np.random.Generator,
    n_bootstrap_resamples: int,
    nominal_coverage: float,
    background_fraction: float,
) -> pd.DataFrame:
    """Evaluate classical uncertainty for every curve in one A-F test.

    Every curve is fitted with the correct IRF width stored in the
    frozen Week-8 simulation metadata.

    The fitted decay model remains mono-exponential for all tests,
    including bi-exponential Test F.

    The returned table contains both local-covariance and parametric
    Poisson-bootstrap uncertainty diagnostics.
    """

    required_metadata_columns = {
        "sample_id",
        "irf_fwhm_ns",
        "irf_shift_ns",
    }

    missing_columns = (
        required_metadata_columns
        - set(test.metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Test metadata are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if (
        isinstance(
            n_bootstrap_resamples,
            (bool, np.bool_),
        )
        or not isinstance(
            n_bootstrap_resamples,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "n_bootstrap_resamples must be an integer."
        )

    if n_bootstrap_resamples < 2:
        raise ValueError(
            "n_bootstrap_resamples must be at least 2."
        )

    if not (
        0.0
        < nominal_coverage
        < 1.0
    ):
        raise ValueError(
            "nominal_coverage must lie strictly between 0 and 1."
        )

    if not np.isfinite(
        irf_centre_ns
    ):
        raise ValueError(
            "irf_centre_ns must be finite."
        )

    normal_quantile = (
        NormalDist().inv_cdf(
            0.5
            + nominal_coverage / 2.0
        )
    )

    # Reuse an IRF for all samples having the same width.
    irf_cache: dict[
        float,
        NDArray[np.float64],
    ] = {}

    rows: list[
        dict[str, object]
    ] = []

    for sample_index in range(
        test.y.size
    ):
        metadata_row = (
            test.metadata.iloc[
                sample_index
            ]
            .to_dict()
        )

        true_lifetime_ns = float(
            test.y[
                sample_index
            ]
        )

        counts = (
            test.X_histograms[
                sample_index
            ]
        )

        irf_fwhm_ns = float(
            metadata_row[
                "irf_fwhm_ns"
            ]
        )

        if irf_fwhm_ns not in irf_cache:
            irf = generate_gaussian_irf(
                time=test.time,
                centre=irf_centre_ns,
                fwhm=irf_fwhm_ns,
            )

            irf = normalize_irf(
                time=test.time,
                irf=irf,
            )

            irf_cache[
                irf_fwhm_ns
            ] = irf

        irf = irf_cache[
            irf_fwhm_ns
        ]

        curve_result = (
            fit_single_reconvolution_curve(
                time=test.time,
                counts=counts,
                irf=irf,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective="poisson",
                background_fraction=(
                    background_fraction
                ),
            )
        )

        row: dict[
            str,
            object,
        ] = {
            **metadata_row,

            "true_lifetime_ns": (
                true_lifetime_ns
            ),

            "fitted_lifetime_ns": (
                curve_result
                .fitted_lifetime_ns
            ),

            "classical_fit_valid": (
                curve_result.valid_fit
            ),

            "classical_boundary_hit": (
                curve_result.boundary_hit
            ),

            "poisson_nll": (
                curve_result.poisson_nll
            ),

            "poisson_deviance": (
                curve_result
                .poisson_deviance
            ),

            "covariance_valid": False,
            "covariance_std_ns": np.nan,
            "covariance_lower_ns": np.nan,
            "covariance_upper_ns": np.nan,
            "covariance_condition_number": np.nan,
            "covariance_failure_reason": None,

            "bootstrap_valid": False,
            "bootstrap_std_ns": np.nan,
            "bootstrap_lower_ns": np.nan,
            "bootstrap_upper_ns": np.nan,
            "bootstrap_fit_failure_rate": np.nan,
            "bootstrap_failure_reason": None,
        }

        if not curve_result.valid_fit:
            rows.append(
                row
            )
            continue

        fit_result = (
            _reconstruct_reconvolution_fit_result(
                time=test.time,
                irf=irf,
                curve_result=(
                    curve_result
                ),
            )
        )

        # -----------------------------------------------------
        # Day 60:
        # local Poisson/Fisher covariance
        # -----------------------------------------------------

        covariance_result = (
            estimate_poisson_reconvolution_local_covariance(
                time=test.time,
                irf=irf,
                fit_result=fit_result,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
            )
        )

        row[
            "covariance_valid"
        ] = (
            covariance_result
            .covariance_valid
        )

        row[
            "covariance_condition_number"
        ] = (
            covariance_result
            .condition_number
        )

        row[
            "covariance_failure_reason"
        ] = (
            covariance_result
            .failure_reason
        )

        if (
            covariance_result
            .covariance_valid
        ):
            lifetime_std = float(
                covariance_result
                .lifetime_std
            )

            fitted_lifetime = float(
                curve_result
                .fitted_lifetime_ns
            )

            row[
                "covariance_std_ns"
            ] = lifetime_std

            row[
                "covariance_lower_ns"
            ] = (
                fitted_lifetime
                - normal_quantile
                * lifetime_std
            )

            row[
                "covariance_upper_ns"
            ] = (
                fitted_lifetime
                + normal_quantile
                * lifetime_std
            )

        # -----------------------------------------------------
        # Day 61:
        # parametric Poisson bootstrap
        # -----------------------------------------------------

        bootstrap_result = (
            estimate_parametric_poisson_bootstrap(
                time=test.time,
                irf=irf,
                fit_result=fit_result,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                rng=rng,
                n_resamples=(
                    n_bootstrap_resamples
                ),
                nominal_coverage=(
                    nominal_coverage
                ),
                background_fraction=(
                    background_fraction
                ),
            )
        )

        row[
            "bootstrap_valid"
        ] = (
            bootstrap_result
            .bootstrap_valid
        )

        row[
            "bootstrap_fit_failure_rate"
        ] = (
            bootstrap_result
            .fit_failure_rate
        )

        row[
            "bootstrap_failure_reason"
        ] = (
            bootstrap_result
            .failure_reason
        )

        if (
            bootstrap_result
            .bootstrap_valid
        ):
            row[
                "bootstrap_std_ns"
            ] = (
                bootstrap_result
                .bootstrap_std_ns
            )

            row[
                "bootstrap_lower_ns"
            ] = (
                bootstrap_result
                .lower_ns
            )

            row[
                "bootstrap_upper_ns"
            ] = (
                bootstrap_result
                .upper_ns
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def _summarize_classical_interval_method(
    data: pd.DataFrame,
    *,
    method_id: str,
    lower_column: str,
    upper_column: str,
    std_column: str,
    nominal_coverage: float,
) -> dict[str, object]:
    """Summarize one classical uncertainty method."""

    if data.empty:
        raise ValueError(
            "data must not be empty."
        )

    y = data[
        "true_lifetime_ns"
    ].to_numpy(
        dtype=np.float64,
    )

    prediction = data[
        "fitted_lifetime_ns"
    ].to_numpy(
        dtype=np.float64,
    )

    lower = data[
        lower_column
    ].to_numpy(
        dtype=np.float64,
    )

    upper = data[
        upper_column
    ].to_numpy(
        dtype=np.float64,
    )

    uncertainty_std = data[
        std_column
    ].to_numpy(
        dtype=np.float64,
    )

    intervals = (
        PredictionIntervalResult(
            prediction=prediction,
            lower=lower,
            upper=upper,
            nominal_coverage=(
                nominal_coverage
            ),
            method_id=method_id,
        )
    )

    interval_metrics = (
        evaluate_prediction_intervals(
            y,
            intervals,
        )
    )

    valid_prediction = (
        np.isfinite(
            prediction
        )
    )

    n_valid_predictions = int(
        np.count_nonzero(
            valid_prediction
        )
    )

    if n_valid_predictions > 0:
        absolute_error = np.abs(
            prediction[
                valid_prediction
            ]
            - y[
                valid_prediction
            ]
        )

        mae_ns = float(
            np.mean(
                absolute_error
            )
        )

    else:
        mae_ns = np.nan

    valid_std = (
        valid_prediction
        & np.isfinite(
            uncertainty_std
        )
        & (
            uncertainty_std
            > 0.0
        )
    )

    if np.any(
        valid_std
    ):
        normalized_error = (
            np.abs(
                prediction[
                    valid_std
                ]
                - y[
                    valid_std
                ]
            )
            / uncertainty_std[
                valid_std
            ]
        )

        mean_uncertainty_std_ns = float(
            np.mean(
                uncertainty_std[
                    valid_std
                ]
            )
        )

        mean_abs_error_over_std = float(
            np.mean(
                normalized_error
            )
        )

        median_abs_error_over_std = float(
            np.median(
                normalized_error
            )
        )

    else:
        mean_uncertainty_std_ns = np.nan
        mean_abs_error_over_std = np.nan
        median_abs_error_over_std = np.nan

    return {
        "method": method_id,

        "n_samples": int(
            y.size
        ),

        "n_valid_predictions": (
            n_valid_predictions
        ),

        "fit_failure_rate": (
            1.0
            - n_valid_predictions
            / y.size
        ),

        "mae_ns": mae_ns,

        "nominal_coverage": (
            nominal_coverage
        ),

        "empirical_coverage": (
            interval_metrics
            .empirical_coverage
        ),

        "coverage_gap": (
            interval_metrics
            .coverage_error
        ),

        "mean_width_ns": (
            interval_metrics
            .mean_interval_width
        ),

        "median_width_ns": (
            interval_metrics
            .median_interval_width
        ),

        "interval_score": (
            interval_metrics
            .mean_interval_score
        ),

        "interval_failure_rate": (
            interval_metrics
            .interval_failure_rate
        ),

        "mean_uncertainty_std_ns": (
            mean_uncertainty_std_ns
        ),

        "mean_abs_error_over_std": (
            mean_abs_error_over_std
        ),

        "median_abs_error_over_std": (
            median_abs_error_over_std
        ),
    }


def build_week9_classical_uncertainty_scorecard(
    per_curve: pd.DataFrame,
    *,
    nominal_coverage: float,
) -> pd.DataFrame:
    """Build covariance/bootstrap uncertainty scorecards for Tests A-F."""

    required_columns = {
        "test_id",
        "true_lifetime_ns",
        "fitted_lifetime_ns",
        "covariance_std_ns",
        "covariance_lower_ns",
        "covariance_upper_ns",
        "bootstrap_std_ns",
        "bootstrap_lower_ns",
        "bootstrap_upper_ns",
    }

    missing_columns = (
        required_columns
        - set(per_curve.columns)
    )

    if missing_columns:
        raise ValueError(
            "Classical uncertainty table is missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    rows: list[
        dict[str, object]
    ] = []

    for test_id in (
        FINAL_ROBUSTNESS_TEST_IDS
    ):
        test_rows = (
            per_curve.loc[
                per_curve[
                    "test_id"
                ] == test_id
            ]
            .reset_index(
                drop=True
            )
        )

        if test_rows.empty:
            raise ValueError(
                f"Classical uncertainty results are missing Test {test_id}."
            )

        for (
            method_id,
            lower_column,
            upper_column,
            std_column,
        ) in (
            (
                POISSON_LOCAL_COVARIANCE_METHOD_ID,
                "covariance_lower_ns",
                "covariance_upper_ns",
                "covariance_std_ns",
            ),
            (
                PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID,
                "bootstrap_lower_ns",
                "bootstrap_upper_ns",
                "bootstrap_std_ns",
            ),
        ):
            summary = (
                _summarize_classical_interval_method(
                    test_rows,
                    method_id=method_id,
                    lower_column=(
                        lower_column
                    ),
                    upper_column=(
                        upper_column
                    ),
                    std_column=(
                        std_column
                    ),
                    nominal_coverage=(
                        nominal_coverage
                    ),
                )
            )

            summary[
                "test_id"
            ] = test_id

            summary[
                "target_reference"
            ] = (
                "dominant_component_tau_1"
                if test_id == "F"
                else "monoexponential_lifetime"
            )

            if (
                method_id
                == PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID
            ):
                failure_rates = (
                    test_rows[
                        "bootstrap_fit_failure_rate"
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                finite_failure_rates = (
                    failure_rates[
                        np.isfinite(
                            failure_rates
                        )
                    ]
                )

                if (
                    finite_failure_rates
                    .size
                    > 0
                ):
                    summary[
                        "mean_bootstrap_refit_failure_rate"
                    ] = float(
                        np.mean(
                            finite_failure_rates
                        )
                    )

                else:
                    summary[
                        "mean_bootstrap_refit_failure_rate"
                    ] = np.nan

            else:
                summary[
                    "mean_bootstrap_refit_failure_rate"
                ] = np.nan

            rows.append(
                summary
            )

    result = pd.DataFrame(
        rows
    )

    column_order = [
        "method",
        "test_id",
        "target_reference",
        "n_samples",
        "n_valid_predictions",
        "fit_failure_rate",
        "mae_ns",
        "nominal_coverage",
        "empirical_coverage",
        "coverage_gap",
        "mean_width_ns",
        "median_width_ns",
        "mean_uncertainty_std_ns",
        "mean_abs_error_over_std",
        "median_abs_error_over_std",
        "interval_score",
        "interval_failure_rate",
        "mean_bootstrap_refit_failure_rate",
    ]

    return (
        result[
            column_order
        ]
    )


def build_week9_classical_conditional_scorecard(
    per_curve: pd.DataFrame,
    *,
    definition: GeneralizationSuiteDefinition,
    nominal_coverage: float,
) -> pd.DataFrame:
    """Evaluate classical uncertainty in the key B/D/F sub-regimes."""

    numerics = (
        definition.numerics
    )

    b_rows = (
        per_curve.loc[
            per_curve[
                "test_id"
            ] == "B"
        ]
    )

    d_rows = (
        per_curve.loc[
            per_curve[
                "test_id"
            ] == "D"
        ]
    )

    f_rows = (
        per_curve.loc[
            per_curve[
                "test_id"
            ] == "F"
        ]
    )

    required_b_column = (
        "signal_photon_count_target"
    )

    if (
        required_b_column
        not in b_rows.columns
    ):
        raise ValueError(
            "Test B results must contain "
            "signal_photon_count_target."
        )

    if (
        "background_per_bin"
        not in d_rows.columns
    ):
        raise ValueError(
            "Test D results must contain background_per_bin."
        )

    if (
        "model_mismatch_severity"
        not in f_rows.columns
    ):
        raise ValueError(
            "Test F results must contain model_mismatch_severity."
        )

    b_photons = (
        b_rows[
            required_b_column
        ].to_numpy(
            dtype=np.int64
        )
    )

    b_low_mask = np.isin(
        b_photons,
        np.asarray(
            numerics
            .test_b_low_photon_counts,
            dtype=np.int64,
        ),
    )

    b_high_mask = np.isin(
        b_photons,
        np.asarray(
            numerics
            .test_b_high_photon_counts,
            dtype=np.int64,
        ),
    )

    f_severity = (
        f_rows[
            "model_mismatch_severity"
        ]
        .astype(str)
        .to_numpy()
    )

    condition_tables = (
        (
            "B_low_photon",
            "B",
            b_rows.loc[
                b_low_mask
            ],
        ),
        (
            "B_high_photon",
            "B",
            b_rows.loc[
                b_high_mask
            ],
        ),
        (
            "D_high_background",
            "D",
            d_rows,
        ),
        (
            "F_weak_mismatch",
            "F",
            f_rows.loc[
                f_severity
                == "weak"
            ],
        ),
        (
            "F_moderate_mismatch",
            "F",
            f_rows.loc[
                f_severity
                == "moderate"
            ],
        ),
    )

    rows: list[
        dict[str, object]
    ] = []

    for (
        condition_id,
        test_id,
        condition_rows,
    ) in condition_tables:

        condition_rows = (
            condition_rows
            .reset_index(
                drop=True
            )
        )

        if condition_rows.empty:
            raise ValueError(
                f"{condition_id} contains no samples."
            )

        for (
            method_id,
            lower_column,
            upper_column,
            std_column,
        ) in (
            (
                POISSON_LOCAL_COVARIANCE_METHOD_ID,
                "covariance_lower_ns",
                "covariance_upper_ns",
                "covariance_std_ns",
            ),
            (
                PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID,
                "bootstrap_lower_ns",
                "bootstrap_upper_ns",
                "bootstrap_std_ns",
            ),
        ):
            summary = (
                _summarize_classical_interval_method(
                    condition_rows,
                    method_id=(
                        method_id
                    ),
                    lower_column=(
                        lower_column
                    ),
                    upper_column=(
                        upper_column
                    ),
                    std_column=(
                        std_column
                    ),
                    nominal_coverage=(
                        nominal_coverage
                    ),
                )
            )

            summary[
                "condition_id"
            ] = condition_id

            summary[
                "test_id"
            ] = test_id

            summary[
                "target_reference"
            ] = (
                "dominant_component_tau_1"
                if test_id == "F"
                else "monoexponential_lifetime"
            )

            rows.append(
                summary
            )

    result = pd.DataFrame(
        rows
    )

    return result[
        [
            "method",
            "condition_id",
            "test_id",
            "target_reference",
            "n_samples",
            "mae_ns",
            "empirical_coverage",
            "coverage_gap",
            "mean_width_ns",
            "mean_uncertainty_std_ns",
            "mean_abs_error_over_std",
            "median_abs_error_over_std",
            "fit_failure_rate",
            "interval_failure_rate",
        ]
    ]


@dataclass(frozen=True)
class Week9ClassicalUncertaintyRobustnessReport:
    """Final classical uncertainty evaluation across frozen Tests A-F."""

    per_curve: pd.DataFrame

    scorecard: pd.DataFrame

    conditional_scorecard: pd.DataFrame

    nominal_coverage: float

    n_bootstrap_resamples: int


def evaluate_week9_classical_uncertainty_robustness(
    *,
    prepared: GeneralizationPreparedData,
    definition: GeneralizationSuiteDefinition,
    temporal_shift_bounds: tuple[
        float,
        float,
    ] = (-0.5, 0.5),
    n_bootstrap_resamples: int = (
        DEFAULT_CLASSICAL_BOOTSTRAP_REPLICATES
    ),
    nominal_coverage: float = (
        DEFAULT_CLASSICAL_NOMINAL_COVERAGE
    ),
    bootstrap_random_seed: int = (
        DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED
    ),
    background_fraction: float = 0.10,
) -> Week9ClassicalUncertaintyRobustnessReport:
    """Run frozen classical uncertainty evaluation across Tests A-F.

    Every histogram is fitted with the same Poisson reconvolution
    estimator used by the Week-8 classical benchmark.

    The correct per-curve IRF width is supplied for every test.

    Test F is intentionally fitted with the wrong decay family:
    a mono-exponential reconvolution model is applied to the
    bi-exponential measurement. Uncertainty is scored against the
    dominant-component lifetime tau_1.
    """

    missing_tests = (
        set(
            FINAL_ROBUSTNESS_TEST_IDS
        )
        - set(
            prepared.tests
        )
    )

    if missing_tests:
        raise ValueError(
            "Prepared data are missing final robustness tests: "
            + ", ".join(
                sorted(
                    missing_tests
                )
            )
        )

    rng = np.random.default_rng(
        bootstrap_random_seed
    )

    tables: list[
        pd.DataFrame
    ] = []

    for test_id in (
        FINAL_ROBUSTNESS_TEST_IDS
    ):
        test = (
            prepared.tests[
                test_id
            ]
        )

        if test.test_id != test_id:
            raise ValueError(
                "Dictionary key and test.test_id must match."
            )

        if test_id == "F":
            if (
                "primary_lifetime_ns"
                not in test.metadata.columns
            ):
                raise ValueError(
                    "Test F metadata must contain primary_lifetime_ns."
                )

            primary_lifetime = (
                test.metadata[
                    "primary_lifetime_ns"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            if not np.allclose(
                primary_lifetime,
                test.y,
            ):
                raise ValueError(
                    "Test-F targets must equal the primary "
                    "dominant-component lifetime tau_1."
                )

        result = (
            _evaluate_classical_uncertainty_test(
                test=test,
                irf_centre_ns=(
                    definition
                    .familiar
                    .irf_centre_ns
                ),
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                rng=rng,
                n_bootstrap_resamples=(
                    n_bootstrap_resamples
                ),
                nominal_coverage=(
                    nominal_coverage
                ),
                background_fraction=(
                    background_fraction
                ),
            )
        )

        tables.append(
            result
        )

    per_curve = pd.concat(
        tables,
        ignore_index=True,
    )

    scorecard = (
        build_week9_classical_uncertainty_scorecard(
            per_curve,
            nominal_coverage=(
                nominal_coverage
            ),
        )
    )

    conditional_scorecard = (
        build_week9_classical_conditional_scorecard(
            per_curve,
            definition=definition,
            nominal_coverage=(
                nominal_coverage
            ),
        )
    )

    return (
        Week9ClassicalUncertaintyRobustnessReport(
            per_curve=per_curve,
            scorecard=scorecard,
            conditional_scorecard=(
                conditional_scorecard
            ),
            nominal_coverage=float(
                nominal_coverage
            ),
            n_bootstrap_resamples=int(
                n_bootstrap_resamples
            ),
        )
    )


@dataclass(frozen=True)
class ClassicalResidualMatrixResult:
    """Signed deviance residuals for one classical robustness test."""

    metadata: pd.DataFrame

    residuals: NDArray[np.float64]

    valid_fit_mask: NDArray[np.bool_]

    poisson_deviance_per_bin: NDArray[np.float64]



def _evaluate_classical_signed_residuals(
    *,
    test: GeneralizationTestMeasurements,
    irf_centre_ns: float,
    temporal_shift_bounds: tuple[
        float,
        float,
    ] = (-0.5, 0.5),
    background_fraction: float = 0.10,
) -> ClassicalResidualMatrixResult:
    """Fit one robustness test and retain signed Poisson residuals."""

    required_metadata_columns = {
        "sample_id",
        "pair_id",
        "irf_fwhm_ns",
    }

    missing_columns = (
        required_metadata_columns
        - set(
            test.metadata.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Residual analysis metadata are missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    n_samples = int(
        test.y.size
    )

    n_bins = int(
        test.time.size
    )

    residuals = np.full(
        (
            n_samples,
            n_bins,
        ),
        np.nan,
        dtype=np.float64,
    )

    valid_fit_mask = np.zeros(
        n_samples,
        dtype=bool,
    )

    poisson_deviance_per_bin = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )

    diagnostics = (
        test.metadata.copy(
            deep=True
        )
    )

    diagnostics[
        "classical_fit_valid"
    ] = False

    diagnostics[
        "fitted_lifetime_ns"
    ] = np.nan

    diagnostics[
        "poisson_deviance"
    ] = np.nan

    diagnostics[
        "poisson_deviance_per_bin"
    ] = np.nan

    irf_cache: dict[
        float,
        NDArray[np.float64],
    ] = {}

    for sample_index in range(
        n_samples
    ):
        irf_fwhm_ns = float(
            test.metadata.iloc[
                sample_index
            ][
                "irf_fwhm_ns"
            ]
        )

        if (
            irf_fwhm_ns
            not in irf_cache
        ):
            irf = (
                generate_gaussian_irf(
                    time=test.time,
                    centre=(
                        irf_centre_ns
                    ),
                    fwhm=(
                        irf_fwhm_ns
                    ),
                )
            )

            irf = normalize_irf(
                time=test.time,
                irf=irf,
            )

            irf_cache[
                irf_fwhm_ns
            ] = irf

        irf = irf_cache[
            irf_fwhm_ns
        ]

        counts = (
            test.X_histograms[
                sample_index
            ]
        )

        curve_result = (
            fit_single_reconvolution_curve(
                time=test.time,
                counts=counts,
                irf=irf,
                temporal_shift_bounds=(
                    temporal_shift_bounds
                ),
                objective="poisson",
                background_fraction=(
                    background_fraction
                ),
            )
        )

        diagnostics.loc[
            sample_index,
            "classical_fit_valid",
        ] = (
            curve_result.valid_fit
        )

        diagnostics.loc[
            sample_index,
            "fitted_lifetime_ns",
        ] = (
            curve_result
            .fitted_lifetime_ns
        )

        diagnostics.loc[
            sample_index,
            "poisson_deviance",
        ] = (
            curve_result
            .poisson_deviance
        )

        if not curve_result.valid_fit:
            continue

        fit_result = (
            _reconstruct_reconvolution_fit_result(
                time=test.time,
                irf=irf,
                curve_result=(
                    curve_result
                ),
            )
        )

        signed_residuals = (
            calculate_poisson_deviance_residuals(
                observed=(
                    counts
                ),
                expected=(
                    fit_result
                    .fitted_curve
                ),
            )
        )

        residuals[
            sample_index
        ] = signed_residuals

        valid_fit_mask[
            sample_index
        ] = True

        deviance_per_bin = float(
            np.mean(
                signed_residuals**2
            )
        )

        poisson_deviance_per_bin[
            sample_index
        ] = (
            deviance_per_bin
        )

        diagnostics.loc[
            sample_index,
            "poisson_deviance_per_bin",
        ] = (
            deviance_per_bin
        )

    return (
        ClassicalResidualMatrixResult(
            metadata=diagnostics,
            residuals=residuals,
            valid_fit_mask=(
                valid_fit_mask
            ),
            poisson_deviance_per_bin=(
                poisson_deviance_per_bin
            ),
        )
    )


def _summarize_signed_residual_profile(
    residuals: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    float,
    float,
    float,
]:
    """Summarize persistent signed residual structure.

    Returns
    -------
    mean_profile:
        Mean signed residual at every time bin.

    profile_rms:
        RMS magnitude of the mean residual profile.

    mean_absolute_profile:
        Mean absolute magnitude of the profile.

    max_absolute_profile:
        Maximum absolute mean residual.
    """

    values = np.asarray(
        residuals,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "residuals must be two-dimensional."
        )

    if values.shape[0] == 0:
        raise ValueError(
            "residuals must contain at least one curve."
        )

    valid_rows = np.all(
        np.isfinite(
            values
        ),
        axis=1,
    )

    if not np.any(
        valid_rows
    ):
        raise ValueError(
            "At least one finite residual curve is required."
        )

    valid = values[
        valid_rows
    ]

    mean_profile = np.mean(
        valid,
        axis=0,
    )

    profile_rms = float(
        np.sqrt(
            np.mean(
                mean_profile**2
            )
        )
    )

    mean_absolute_profile = float(
        np.mean(
            np.abs(
                mean_profile
            )
        )
    )

    max_absolute_profile = float(
        np.max(
            np.abs(
                mean_profile
            )
        )
    )

    return (
        mean_profile,
        profile_rms,
        mean_absolute_profile,
        max_absolute_profile,
    )



@dataclass(frozen=True)
class Week9ResidualMismatchReport:
    """Residual-structure diagnostics for Test A versus Test F."""

    profile_table: pd.DataFrame

    summary: pd.DataFrame

    paired_profile_table: pd.DataFrame


def evaluate_week9_residual_mismatch_diagnostics(
    *,
    prepared: GeneralizationPreparedData,
    definition: GeneralizationSuiteDefinition,
    temporal_shift_bounds: tuple[
        float,
        float,
    ] = (-0.5, 0.5),
    background_fraction: float = 0.10,
) -> Week9ResidualMismatchReport:
    """Compare signed residual structure for A, weak F, and moderate F."""

    if "A" not in prepared.tests:
        raise ValueError(
            "Prepared data must contain Test A."
        )

    if "F" not in prepared.tests:
        raise ValueError(
            "Prepared data must contain Test F."
        )

    test_a = prepared.tests[
        "A"
    ]

    test_f = prepared.tests[
        "F"
    ]

    if (
        "model_mismatch_severity"
        not in test_f.metadata.columns
    ):
        raise ValueError(
            "Test F metadata must contain model_mismatch_severity."
        )

    if not np.array_equal(
        test_a.time,
        test_f.time,
    ):
        raise ValueError(
            "Tests A and F must share the same time axis."
        )

    residual_a = (
        _evaluate_classical_signed_residuals(
            test=test_a,
            irf_centre_ns=(
                definition
                .familiar
                .irf_centre_ns
            ),
            temporal_shift_bounds=(
                temporal_shift_bounds
            ),
            background_fraction=(
                background_fraction
            ),
        )
    )

    residual_f = (
        _evaluate_classical_signed_residuals(
            test=test_f,
            irf_centre_ns=(
                definition
                .familiar
                .irf_centre_ns
            ),
            temporal_shift_bounds=(
                temporal_shift_bounds
            ),
            background_fraction=(
                background_fraction
            ),
        )
    )

    f_severity = (
        test_f.metadata[
            "model_mismatch_severity"
        ]
        .astype(str)
        .to_numpy()
    )

    weak_mask = (
        f_severity
        == "weak"
    )

    moderate_mask = (
        f_severity
        == "moderate"
    )

    if not np.any(
        weak_mask
    ):
        raise ValueError(
            "Test F contains no weak-mismatch curves."
        )

    if not np.any(
        moderate_mask
    ):
        raise ValueError(
            "Test F contains no moderate-mismatch curves."
        )

    (
        profile_a,
        rms_a,
        mean_abs_a,
        max_abs_a,
    ) = (
        _summarize_signed_residual_profile(
            residual_a.residuals
        )
    )

    (
        profile_f_weak,
        rms_f_weak,
        mean_abs_f_weak,
        max_abs_f_weak,
    ) = (
        _summarize_signed_residual_profile(
            residual_f.residuals[
                weak_mask
            ]
        )
    )

    (
        profile_f_moderate,
        rms_f_moderate,
        mean_abs_f_moderate,
        max_abs_f_moderate,
    ) = (
        _summarize_signed_residual_profile(
            residual_f.residuals[
                moderate_mask
            ]
        )
    )

    profile_table = pd.DataFrame(
        {
            "time_ns": (
                test_a.time.copy()
            ),

            "A_mean_signed_residual": (
                profile_a
            ),

            "F_weak_mean_signed_residual": (
                profile_f_weak
            ),

            "F_moderate_mean_signed_residual": (
                profile_f_moderate
            ),
        }
    )

    # --------------------------------------------------------
    # Scalar GOF summary
    # --------------------------------------------------------

    condition_data = (
        (
            "A",
            residual_a,
            np.ones(
                test_a.y.size,
                dtype=bool,
            ),
            rms_a,
            mean_abs_a,
            max_abs_a,
        ),
        (
            "F_weak_mismatch",
            residual_f,
            weak_mask,
            rms_f_weak,
            mean_abs_f_weak,
            max_abs_f_weak,
        ),
        (
            "F_moderate_mismatch",
            residual_f,
            moderate_mask,
            rms_f_moderate,
            mean_abs_f_moderate,
            max_abs_f_moderate,
        ),
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for (
        condition_id,
        result,
        condition_mask,
        profile_rms,
        mean_absolute_profile,
        max_absolute_profile,
    ) in condition_data:

        valid = (
            condition_mask
            & result.valid_fit_mask
        )

        deviance = (
            result
            .poisson_deviance_per_bin[
                valid
            ]
        )

        summary_rows.append(
            {
                "condition_id": (
                    condition_id
                ),

                "n_curves": int(
                    np.count_nonzero(
                        condition_mask
                    )
                ),

                "n_valid_fits": int(
                    np.count_nonzero(
                        valid
                    )
                ),

                "fit_failure_rate": (
                    1.0
                    - np.count_nonzero(
                        valid
                    )
                    / np.count_nonzero(
                        condition_mask
                    )
                ),

                "mean_poisson_deviance_per_bin": float(
                    np.mean(
                        deviance
                    )
                ),

                "median_poisson_deviance_per_bin": float(
                    np.median(
                        deviance
                    )
                ),

                "mean_residual_profile_rms": (
                    profile_rms
                ),

                "mean_absolute_residual_profile": (
                    mean_absolute_profile
                ),

                "max_absolute_mean_residual": (
                    max_absolute_profile
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Matched F - A residual profiles
    # --------------------------------------------------------

    a_pair_ids = (
        residual_a.metadata[
            "pair_id"
        ].to_numpy()
    )

    f_pair_ids = (
        residual_f.metadata[
            "pair_id"
        ].to_numpy()
    )

    if (
        len(
            set(
                a_pair_ids
            )
        )
        != a_pair_ids.size
    ):
        raise ValueError(
            "Test A pair_id values must be unique."
        )

    if (
        len(
            set(
                f_pair_ids
            )
        )
        != f_pair_ids.size
    ):
        raise ValueError(
            "Test F pair_id values must be unique."
        )

    a_index_by_pair = {
        pair_id: index
        for index, pair_id
        in enumerate(
            a_pair_ids
        )
    }

    paired_profiles: dict[
        str,
        NDArray[np.float64],
    ] = {}

    for (
        severity_name,
        severity_mask,
    ) in (
        (
            "weak",
            weak_mask,
        ),
        (
            "moderate",
            moderate_mask,
        ),
    ):
        differences: list[
            NDArray[np.float64]
        ] = []

        f_indices = np.flatnonzero(
            severity_mask
        )

        for f_index in (
            f_indices
        ):
            pair_id = (
                f_pair_ids[
                    f_index
                ]
            )

            if (
                pair_id
                not in a_index_by_pair
            ):
                raise ValueError(
                    "Every Test-F pair_id must have "
                    "a matching Test-A curve."
                )

            a_index = (
                a_index_by_pair[
                    pair_id
                ]
            )

            if not (
                residual_f
                .valid_fit_mask[
                    f_index
                ]
                and residual_a
                .valid_fit_mask[
                    a_index
                ]
            ):
                continue

            differences.append(
                residual_f.residuals[
                    f_index
                ]
                - residual_a.residuals[
                    a_index
                ]
            )

        if not differences:
            raise ValueError(
                f"No valid paired residuals for {severity_name} mismatch."
            )

        difference_matrix = (
            np.vstack(
                differences
            )
        )

        (
            mean_difference,
            _,
            _,
            _,
        ) = (
            _summarize_signed_residual_profile(
                difference_matrix
            )
        )

        paired_profiles[
            severity_name
        ] = mean_difference

    paired_profile_table = (
        pd.DataFrame(
            {
                "time_ns": (
                    test_a.time.copy()
                ),

                "F_weak_minus_A_mean_residual": (
                    paired_profiles[
                        "weak"
                    ]
                ),

                "F_moderate_minus_A_mean_residual": (
                    paired_profiles[
                        "moderate"
                    ]
                ),
            }
        )
    )

    return (
        Week9ResidualMismatchReport(
            profile_table=(
                profile_table
            ),
            summary=summary,
            paired_profile_table=(
                paired_profile_table
            ),
        )
    )


