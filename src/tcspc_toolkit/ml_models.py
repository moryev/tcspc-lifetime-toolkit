"""Compatibility exports for :mod:`tcspc_toolkit.ml.models`.

Historical flat imports remain supported through at least Issue #12.
"""

from tcspc_toolkit.ml.models import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_PCA_COMPONENTS,
    make_ridge_pipeline,
    make_random_forest_pipeline,
    make_hist_gradient_boosting_pipeline,
    make_canonical_ml_estimator_specs,
    make_quantile_hist_gradient_boosting_pipeline,
    make_normalized_histogram_ridge_pipeline,
    make_pca_histogram_ridge_pipeline,
)

__all__ = [
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_PCA_COMPONENTS",
    "make_ridge_pipeline",
    "make_random_forest_pipeline",
    "make_hist_gradient_boosting_pipeline",
    "make_canonical_ml_estimator_specs",
    "make_quantile_hist_gradient_boosting_pipeline",
    "make_normalized_histogram_ridge_pipeline",
    "make_pca_histogram_ridge_pipeline",
]

