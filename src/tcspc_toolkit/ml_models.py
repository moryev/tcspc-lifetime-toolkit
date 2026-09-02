"""Scikit-learn models for TCSPC lifetime benchmarking."""

from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
)

from tcspc_toolkit.config import CountNormalization
from tcspc_toolkit.representations import (
    normalize_histogram_batch,
)


DEFAULT_RANDOM_STATE = 42


DEFAULT_PCA_COMPONENTS = 10


def _make_total_histogram_normalizer(
) -> FunctionTransformer:
    """Create a stateless TOTAL-normalization transformer.

    The transformer normalizes every histogram independently
    by its own total count. It therefore contains no fitted
    statistics shared between samples.
    """

    return FunctionTransformer(
        func=normalize_histogram_batch,
        kw_args={
            "mode": CountNormalization.TOTAL,
        },
        validate=False,
    )


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


def make_normalized_histogram_ridge_pipeline(
) -> Pipeline:
    """Create a Ridge pipeline for raw TCSPC histograms.

    Raw histograms are TOTAL-normalized independently,
    standardized feature-wise using statistics fitted on the
    training data, and passed to Ridge regression.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted TOTAL normalization + StandardScaler + Ridge
        pipeline.

    Notes
    -----
    The pipeline accepts raw histograms directly.

    When used inside cross-validation, StandardScaler is fitted
    independently on each training fold. TOTAL normalization is
    stateless and operates independently on every histogram.
    """

    return Pipeline(
        steps=[
            (
                "normalize",
                _make_total_histogram_normalizer(),
            ),
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


def make_pca_histogram_ridge_pipeline(
    *,
    n_components: int = DEFAULT_PCA_COMPONENTS,
) -> Pipeline:
    """Create a leakage-safe PCA + Ridge histogram pipeline.

    Parameters
    ----------
    n_components:
        Number of principal components retained from the
        TOTAL-normalized histogram representation.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted TOTAL normalization + PCA + StandardScaler +
        Ridge pipeline.

    Notes
    -----
    When this pipeline is used inside repeated cross-validation,
    PCA is fitted exclusively on each training fold.

    Validation-fold histograms are normalized independently and
    transformed using the PCA model fitted on the corresponding
    training fold. They never contribute to PCA fitting.
    """

    if (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(
            n_components,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "n_components must be an integer."
        )

    if n_components < 1:
        raise ValueError(
            "n_components must be at least 1."
        )

    return Pipeline(
        steps=[
            (
                "normalize",
                _make_total_histogram_normalizer(),
            ),
            (
                "pca",
                PCA(
                    n_components=int(
                        n_components
                    ),
                    svd_solver="full",
                ),
            ),
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


