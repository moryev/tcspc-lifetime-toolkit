"""Machine-learning representations of TCSPC histograms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from tcspc_toolkit.config import CountNormalization
from tcspc_toolkit.preprocessing import normalize_counts


def normalize_histogram_batch(
    histograms: ArrayLike,
    mode: CountNormalization = CountNormalization.TOTAL,
) -> NDArray[np.float64]:
    """Normalize multiple TCSPC histograms independently.

    Parameters
    ----------
    histograms:
        Two-dimensional array with one histogram per row and one
        time bin per column.
    mode:
        Normalization strategy applied independently to each histogram.

    Returns
    -------
    NDArray[np.float64]
        Two-dimensional array of normalized histograms with the same
        shape as the input.

    Raises
    ------
    ValueError
        If ``histograms`` cannot be converted to a numeric array, is not
        two-dimensional, contains no histograms, contains no bins, or if
        an individual histogram cannot be normalized.
    TypeError
        If ``mode`` is not a ``CountNormalization`` value.

    Notes
    -----
    Each histogram is normalized independently using
    ``preprocessing.normalize_counts``.

    With total-count normalization, every histogram is divided by its
    own total count. Therefore absolute photon-count information is
    deliberately removed from the resulting representation.

    The input array is not modified.
    """
    try:
        histograms_array = np.asarray(
            histograms,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "histograms must contain numeric values."
        ) from exc

    if histograms_array.ndim != 2:
        raise ValueError(
            "histograms must be a two-dimensional array."
        )

    if histograms_array.shape[0] == 0:
        raise ValueError(
            "histograms must contain at least one histogram."
        )

    if histograms_array.shape[1] == 0:
        raise ValueError(
            "histograms must contain at least one bin."
        )

    normalized_histograms = [
        normalize_counts(
            counts=counts,
            mode=mode,
        )
        for counts in histograms_array
    ]

    return np.vstack(normalized_histograms)


def fit_pca_representation(
    X_train: ArrayLike,
    n_components: int,
) -> PCA:
    """Fit PCA to a training histogram representation.

    Parameters
    ----------
    X_train:
        Two-dimensional training matrix with one histogram per row
        and one histogram bin per column. The matrix should already
        contain the desired normalized histogram representation.
    n_components:
        Number of principal components to retain.

    Returns
    -------
    PCA
        Fitted scikit-learn PCA transformer.

    Raises
    ------
    ValueError
        If ``X_train`` cannot be converted to a finite two-dimensional
        numeric array, contains fewer than two histograms, contains no
        bins, or if ``n_components`` is outside the valid range.
    TypeError
        If ``n_components`` is not an integer.

    Notes
    -----
    PCA must be fitted only on training data. Test data must not
    contribute to the fitted PCA mean, components, or explained
    variance.

    This function does not perform histogram normalization. The input
    should already have been prepared using the desired representation,
    for example with ``normalize_histogram_batch``.

    The exact full SVD solver is used to provide deterministic PCA
    fitting for a fixed input matrix.
    """
    try:
        X_train_array = np.asarray(
            X_train,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "X_train must contain numeric values."
        ) from exc

    if X_train_array.ndim != 2:
        raise ValueError(
            "X_train must be a two-dimensional array."
        )

    if X_train_array.shape[0] < 2:
        raise ValueError(
            "X_train must contain at least two histograms."
        )

    if X_train_array.shape[1] == 0:
        raise ValueError(
            "X_train must contain at least one feature."
        )

    if not np.all(np.isfinite(X_train_array)):
        raise ValueError(
            "X_train must contain only finite values."
        )

    if (
        isinstance(n_components, (bool, np.bool_))
        or not isinstance(n_components, (int, np.integer))
    ):
        raise TypeError(
            "n_components must be an integer."
        )

    if n_components < 1:
        raise ValueError(
            "n_components must be at least 1."
        )

    max_components = min(X_train_array.shape)

    if n_components > max_components:
        raise ValueError(
            "n_components must not exceed the smaller "
            "of the number of training histograms and features."
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="full",
    )

    pca.fit(X_train_array)

    return pca


def transform_pca_representation(
    pca: PCA,
    X: ArrayLike,
) -> NDArray[np.float64]:
    """Transform histogram representations using a fitted PCA model.

    Parameters
    ----------
    pca:
        Fitted scikit-learn PCA transformer.
    X:
        Two-dimensional matrix with one histogram per row and the same
        number of features used when fitting ``pca``.

    Returns
    -------
    NDArray[np.float64]
        PCA-compressed representation with one row per histogram and one
        column per retained principal component.

    Raises
    ------
    TypeError
        If ``pca`` is not a scikit-learn ``PCA`` instance.
    ValueError
        If ``X`` cannot be converted to a finite two-dimensional numeric
        array, contains no histograms or features, or has a different
        number of features from the data used to fit ``pca``.

    Notes
    -----
    This function only transforms data using an already fitted PCA
    model. It never refits PCA.

    In a leakage-safe train/test workflow, PCA must first be fitted on
    training data only. The same fitted PCA object is then used to
    transform both training and test data.
    """
    if not isinstance(pca, PCA):
        raise TypeError(
            "pca must be a scikit-learn PCA instance."
        )

    if not hasattr(pca, "components_"):
        raise ValueError(
            "pca must be fitted before transformation."
        )

    try:
        X_array = np.asarray(
            X,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "X must contain numeric values."
        ) from exc

    if X_array.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional array."
        )

    if X_array.shape[0] == 0:
        raise ValueError(
            "X must contain at least one histogram."
        )

    if X_array.shape[1] == 0:
        raise ValueError(
            "X must contain at least one feature."
        )

    if not np.all(np.isfinite(X_array)):
        raise ValueError(
            "X must contain only finite values."
        )

    if X_array.shape[1] != pca.n_features_in_:
        raise ValueError(
            "X must contain the same number of features "
            "used to fit pca."
        )

    transformed = pca.transform(X_array)

    return np.asarray(
        transformed,
        dtype=np.float64,
    )


def cumulative_explained_variance(
    pca: PCA,
) -> NDArray[np.float64]:
    """Return cumulative explained variance for a fitted PCA model.

    Parameters
    ----------
    pca:
        Fitted scikit-learn PCA transformer.

    Returns
    -------
    NDArray[np.float64]
        One-dimensional array containing the cumulative fraction of
        variance explained by the retained principal components.

    Raises
    ------
    TypeError
        If ``pca`` is not a scikit-learn ``PCA`` instance.
    ValueError
        If ``pca`` has not been fitted or if its explained-variance
        ratios contain non-finite values.

    Notes
    -----
    The kth returned value gives the fraction of total variance
    explained by the first k + 1 principal components.
    """
    if not isinstance(pca, PCA):
        raise TypeError(
            "pca must be a scikit-learn PCA instance."
        )

    if not hasattr(pca, "explained_variance_ratio_"):
        raise ValueError(
            "pca must be fitted before explained variance is evaluated."
        )

    explained_variance_ratio = np.asarray(
        pca.explained_variance_ratio_,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(explained_variance_ratio)):
        raise ValueError(
            "pca explained variance ratios must be finite."
        )

    return np.cumsum(explained_variance_ratio)


# TODO: After the main functions are implemented, try performing "PCA-compressed engineered features"
#       (kind of Principal Component Regression, PCR, analysis)


def augment_with_total_counts(
    X_representation: ArrayLike,
    histograms: ArrayLike,
) -> NDArray[np.float64]:
    """Append measured total photon count to an ML representation.

    Parameters
    ----------
    X_representation:
        Two-dimensional representation matrix with one sample per row.
    histograms:
        Raw histogram matrix corresponding to the same samples.

    Returns
    -------
    NDArray[np.float64]
        Representation matrix with one additional column containing
        the measured total photon count of each raw histogram.

    Raises
    ------
    ValueError
        If either input is not a finite two-dimensional numeric array,
        if their sample counts differ, or if histograms contain
        negative values.

    Notes
    -----
    The appended count is calculated directly from the measured raw
    histogram and therefore does not use simulation metadata.

    When ``X_representation`` contains TOTAL-normalized histogram bins,
    adding the total count restores the absolute count scale removed
    by normalization.
    """
    try:
        X_array = np.asarray(
            X_representation,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "X_representation must contain numeric values."
        ) from exc

    try:
        histograms_array = np.asarray(
            histograms,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "histograms must contain numeric values."
        ) from exc

    if X_array.ndim != 2:
        raise ValueError(
            "X_representation must be a two-dimensional array."
        )

    if histograms_array.ndim != 2:
        raise ValueError(
            "histograms must be a two-dimensional array."
        )

    if X_array.shape[0] != histograms_array.shape[0]:
        raise ValueError(
            "X_representation and histograms must contain "
            "the same number of samples."
        )

    if not np.all(np.isfinite(X_array)):
        raise ValueError(
            "X_representation must contain only finite values."
        )

    if not np.all(np.isfinite(histograms_array)):
        raise ValueError(
            "histograms must contain only finite values."
        )

    if np.any(histograms_array < 0.0):
        raise ValueError(
            "histograms must contain non-negative values."
        )

    total_counts = np.sum(
        histograms_array,
        axis=1,
        dtype=np.float64,
    )

    return np.column_stack(
        (
            X_array,
            total_counts,
        )
    )
