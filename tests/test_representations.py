import numpy as np
import pytest
from sklearn.decomposition import PCA

from tcspc_toolkit.config import CountNormalization
from tcspc_toolkit.representations import (
    cumulative_explained_variance,
    fit_pca_representation,
    normalize_histogram_batch,
    transform_pca_representation,
)


def test_normalize_histogram_batch_preserves_shape() -> None:
    histograms = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    result = normalize_histogram_batch(
        histograms=histograms,
    )

    assert result.shape == histograms.shape


def test_total_normalization_produces_unit_row_sums() -> None:
    histograms = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [5.0, 5.0, 10.0, 20.0],
        ],
        dtype=np.float64,
    )

    result = normalize_histogram_batch(
        histograms=histograms,
        mode=CountNormalization.TOTAL,
    )

    np.testing.assert_allclose(
        result.sum(axis=1),
        np.ones(histograms.shape[0]),
    )


def test_total_normalization_removes_count_scaling() -> None:
    reference_histogram = np.array(
        [1.0, 2.0, 4.0, 2.0, 1.0],
        dtype=np.float64,
    )

    histograms = np.vstack(
        [
            reference_histogram,
            10.0 * reference_histogram,
            100.0 * reference_histogram,
        ]
    )

    result = normalize_histogram_batch(
        histograms=histograms,
        mode=CountNormalization.TOTAL,
    )

    np.testing.assert_allclose(
        result[1],
        result[0],
    )

    np.testing.assert_allclose(
        result[2],
        result[0],
    )


def test_peak_normalization_produces_unit_row_peaks() -> None:
    histograms = np.array(
        [
            [1.0, 2.0, 4.0, 2.0],
            [3.0, 9.0, 6.0, 3.0],
        ],
        dtype=np.float64,
    )

    result = normalize_histogram_batch(
        histograms=histograms,
        mode=CountNormalization.PEAK,
    )

    np.testing.assert_allclose(
        np.max(result, axis=1),
        np.ones(histograms.shape[0]),
    )


def test_normalize_histogram_batch_does_not_modify_input() -> None:
    histograms = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )

    original_histograms = histograms.copy()

    normalize_histogram_batch(
        histograms=histograms,
    )

    np.testing.assert_array_equal(
        histograms,
        original_histograms,
    )


def test_normalize_histogram_batch_rejects_one_dimensional_input() -> None:
    histogram = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="histograms must be a two-dimensional array",
    ):
        normalize_histogram_batch(
            histograms=histogram,
        )


def test_normalize_histogram_batch_rejects_empty_batch() -> None:
    histograms = np.empty(
        (0, 4),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="histograms must contain at least one histogram",
    ):
        normalize_histogram_batch(
            histograms=histograms,
        )


def test_normalize_histogram_batch_rejects_zero_total_histogram() -> None:
    histograms = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="normalization factor must be positive",
    ):
        normalize_histogram_batch(
            histograms=histograms,
            mode=CountNormalization.TOTAL,
        )


def test_fit_pca_representation_returns_fitted_pca() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.2, 0.2, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.3, 0.2, 0.2],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    assert isinstance(pca, PCA)
    assert pca.n_components_ == 2


def test_fit_pca_representation_has_expected_component_shape() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.2, 0.2, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.3, 0.2, 0.2],
            [0.1, 0.4, 0.3, 0.2],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=3,
    )

    assert pca.components_.shape == (
        3,
        X_train.shape[1],
    )


def test_fit_pca_representation_uses_training_mean_only() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.3, 0.2],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    expected_mean = X_train.mean(axis=0)

    np.testing.assert_allclose(
        pca.mean_,
        expected_mean,
    )


def test_fit_pca_representation_is_deterministic() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    first_pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    second_pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    np.testing.assert_allclose(
        first_pca.components_,
        second_pca.components_,
    )

    np.testing.assert_allclose(
        first_pca.explained_variance_ratio_,
        second_pca.explained_variance_ratio_,
    )


def test_fit_pca_representation_accepts_normalized_histograms() -> None:
    histograms = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 20.0, 20.0, 40.0],
            [40.0, 30.0, 20.0, 10.0],
            [30.0, 40.0, 20.0, 10.0],
        ],
        dtype=np.float64,
    )

    X_train = normalize_histogram_batch(
        histograms=histograms,
        mode=CountNormalization.TOTAL,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    assert pca.n_components_ == 2

    np.testing.assert_allclose(
        X_train.sum(axis=1),
        np.ones(X_train.shape[0]),
    )


def test_fit_pca_representation_does_not_modify_input() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.3, 0.2],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    original_X_train = X_train.copy()

    fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    np.testing.assert_array_equal(
        X_train,
        original_X_train,
    )


def test_fit_pca_representation_rejects_one_dimensional_input() -> None:
    X_train = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="X_train must be a two-dimensional array",
    ):
        fit_pca_representation(
            X_train=X_train,
            n_components=2,
        )


def test_fit_pca_representation_rejects_too_many_components() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.5],
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="n_components must not exceed",
    ):
        fit_pca_representation(
            X_train=X_train,
            n_components=4,
        )


def test_fit_pca_representation_rejects_zero_components() -> None:
    X_train = np.ones(
        (4, 5),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="n_components must be at least 1",
    ):
        fit_pca_representation(
            X_train=X_train,
            n_components=0,
        )


def test_transform_pca_representation_returns_expected_shape() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    result = transform_pca_representation(
        pca=pca,
        X=X_train,
    )

    assert result.shape == (
        X_train.shape[0],
        2,
    )


def test_transform_pca_representation_handles_train_and_test_data() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
        ],
        dtype=np.float64,
    )

    X_test = np.array(
        [
            [0.15, 0.25, 0.25, 0.35],
            [0.35, 0.25, 0.25, 0.15],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    X_train_pca = transform_pca_representation(
        pca=pca,
        X=X_train,
    )

    X_test_pca = transform_pca_representation(
        pca=pca,
        X=X_test,
    )

    assert X_train_pca.shape == (4, 2)
    assert X_test_pca.shape == (2, 2)


def test_transform_pca_representation_does_not_refit_on_test_data() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.3, 0.2],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    X_test = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.8, 0.10, 0.05, 0.05],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    original_mean = pca.mean_.copy()
    original_components = pca.components_.copy()

    transform_pca_representation(
        pca=pca,
        X=X_test,
    )

    np.testing.assert_allclose(
        pca.mean_,
        original_mean,
    )

    np.testing.assert_allclose(
        pca.components_,
        original_components,
    )


def test_pca_mean_is_based_only_on_training_data() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.2, 0.2],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    X_test = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.8, 0.10, 0.05, 0.05],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    transform_pca_representation(
        pca=pca,
        X=X_test,
    )

    np.testing.assert_allclose(
        pca.mean_,
        X_train.mean(axis=0),
    )

    assert not np.allclose(
        pca.mean_,
        np.vstack(
            [X_train, X_test]
        ).mean(axis=0),
    )


def test_pca_workflow_uses_normalized_histogram_representations() -> None:
    train_histograms = np.array(
        [
            [10.0, 20.0, 40.0, 30.0],
            [20.0, 30.0, 30.0, 20.0],
            [40.0, 30.0, 20.0, 10.0],
            [30.0, 40.0, 20.0, 10.0],
        ],
        dtype=np.float64,
    )

    test_histograms = np.array(
        [
            [15.0, 25.0, 35.0, 25.0],
            [35.0, 30.0, 20.0, 15.0],
        ],
        dtype=np.float64,
    )

    X_train = normalize_histogram_batch(
        histograms=train_histograms,
        mode=CountNormalization.TOTAL,
    )

    X_test = normalize_histogram_batch(
        histograms=test_histograms,
        mode=CountNormalization.TOTAL,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    X_train_pca = transform_pca_representation(
        pca=pca,
        X=X_train,
    )

    X_test_pca = transform_pca_representation(
        pca=pca,
        X=X_test,
    )

    assert X_train_pca.shape == (4, 2)
    assert X_test_pca.shape == (2, 2)

    np.testing.assert_allclose(
        X_train.sum(axis=1),
        np.ones(X_train.shape[0]),
    )

    np.testing.assert_allclose(
        X_test.sum(axis=1),
        np.ones(X_test.shape[0]),
    )


def test_transform_pca_representation_rejects_unfitted_pca() -> None:
    pca = PCA(
        n_components=2,
        svd_solver="full",
    )

    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="pca must be fitted before transformation",
    ):
        transform_pca_representation(
            pca=pca,
            X=X,
        )


def test_transform_pca_representation_rejects_feature_mismatch() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    X_test = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.5, 0.3, 0.2],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="same number of features",
    ):
        transform_pca_representation(
            pca=pca,
            X=X_test,
        )


def test_transform_pca_representation_does_not_modify_input() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=2,
    )

    X_test = np.array(
        [
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.2, 0.2],
        ],
        dtype=np.float64,
    )

    original_X_test = X_test.copy()

    transform_pca_representation(
        pca=pca,
        X=X_test,
    )

    np.testing.assert_array_equal(
        X_test,
        original_X_test,
    )


def test_cumulative_explained_variance_has_expected_shape() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=3,
    )

    cumulative = cumulative_explained_variance(
        pca=pca,
    )

    assert cumulative.shape == (3,)


def test_cumulative_explained_variance_matches_cumsum() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=3,
    )

    cumulative = cumulative_explained_variance(
        pca=pca,
    )

    expected = np.cumsum(
        pca.explained_variance_ratio_
    )

    np.testing.assert_allclose(
        cumulative,
        expected,
    )


def test_cumulative_explained_variance_is_monotonic() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=3,
    )

    cumulative = cumulative_explained_variance(
        pca=pca,
    )

    assert np.all(
        np.diff(cumulative) >= 0.0
    )


def test_cumulative_explained_variance_does_not_exceed_one() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=3,
    )

    cumulative = cumulative_explained_variance(
        pca=pca,
    )

    assert cumulative[-1] <= 1.0 + 1e-12


def test_final_cumulative_variance_equals_sum_of_ratios() -> None:
    X_train = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.2, 0.3],
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=3,
    )

    cumulative = cumulative_explained_variance(
        pca=pca,
    )

    assert cumulative[-1] == pytest.approx(
        np.sum(pca.explained_variance_ratio_)
    )


def test_cumulative_explained_variance_rejects_unfitted_pca() -> None:
    pca = PCA(
        n_components=2,
        svd_solver="full",
    )

    with pytest.raises(
        ValueError,
        match="pca must be fitted",
    ):
        cumulative_explained_variance(
            pca=pca,
        )


def test_total_normalized_histograms_have_redundant_pca_direction() -> None:
    histograms = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 30.0, 40.0, 10.0],
            [40.0, 20.0, 10.0, 30.0],
            [30.0, 40.0, 20.0, 10.0],
            [25.0, 15.0, 35.0, 25.0],
        ],
        dtype=np.float64,
    )

    X_train = normalize_histogram_batch(
        histograms=histograms,
        mode=CountNormalization.TOTAL,
    )

    pca = fit_pca_representation(
        X_train=X_train,
        n_components=4,
    )

    assert pca.explained_variance_ratio_[-1] == pytest.approx(
        0.0,
        abs=1e-12,
    )


