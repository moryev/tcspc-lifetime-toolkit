"""Scikit-learn models for TCSPC lifetime benchmarking."""

from __future__ import annotations

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42


def make_ridge_pipeline() -> Pipeline:
    """Create the Ridge-regression benchmark pipeline.

    Engineered features are standardized before fitting because
    Ridge regularization is sensitive to feature scale.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted StandardScaler + Ridge regression pipeline.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(),
            ),
        ]
    )


def make_random_forest_pipeline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create the Random Forest benchmark pipeline.

    Parameters
    ----------
    random_state:
        Seed controlling stochastic model construction.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted Random Forest regression pipeline.
    """
    return Pipeline(
        steps=[
            (
                "model",
                RandomForestRegressor(
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_hist_gradient_boosting_pipeline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create the histogram gradient-boosting benchmark pipeline.

    Parameters
    ----------
    random_state:
        Seed controlling stochastic operations performed by the
        estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted histogram gradient-boosting regression pipeline.
    """
    return Pipeline(
        steps=[
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=random_state,
                ),
            ),
        ]
    )
