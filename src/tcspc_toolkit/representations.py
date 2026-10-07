"""Compatibility imports for :mod:`tcspc_toolkit.preprocessing.representations`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings."""

from tcspc_toolkit.preprocessing.representations import (
    normalize_histogram_batch,
    fit_pca_representation,
    transform_pca_representation,
    cumulative_explained_variance,
    augment_with_total_counts,
)

__all__ = [
    "normalize_histogram_batch",
    "fit_pca_representation",
    "transform_pca_representation",
    "cumulative_explained_variance",
    "augment_with_total_counts",
]
