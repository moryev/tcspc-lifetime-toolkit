import numpy as np

from tcspc_toolkit.config import (
    CountNormalization,
    FeatureConfig,
)
from tcspc_toolkit.features import (
    FEATURE_NAMES,
    extract_feature_table,
)
from tcspc_toolkit.representations import (
    fit_pca_representation,
    normalize_histogram_batch,
    transform_pca_representation,
)


def _make_integer_exponential_histogram(
    time: np.ndarray,
    lifetime_ns: float,
    amplitude: int = 1_000_000,
) -> np.ndarray:
    """Create a deterministic integer-valued exponential histogram."""
    expected_counts = (
        amplitude
        * np.exp(-time / lifetime_ns)
    )

    return np.rint(
        expected_counts
    ).astype(np.int64)


def _feature_config() -> FeatureConfig:
    """Return a common feature definition for integration tests."""
    return FeatureConfig(
        tail_start_ns=4.0,
        early_stop_ns=2.0,
        late_start_ns=5.0,
        min_tail_points=5,
    )


def test_feature_table_matches_number_of_histograms() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.1,
        dtype=np.float64,
    )

    histograms = np.vstack(
        [
            _make_integer_exponential_histogram(
                time,
                lifetime_ns=1.0,
            ),
            _make_integer_exponential_histogram(
                time,
                lifetime_ns=2.0,
            ),
            _make_integer_exponential_histogram(
                time,
                lifetime_ns=4.0,
            ),
        ]
    )

    features = extract_feature_table(
        histograms=histograms,
        time=time,
        config=_feature_config(),
    )

    assert len(features) == len(histograms)

    assert list(features.columns) == list(FEATURE_NAMES)


def test_engineered_features_have_expected_count_scaling_behaviour() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.1,
        dtype=np.float64,
    )

    histogram = _make_integer_exponential_histogram(
        time,
        lifetime_ns=2.0,
    )

    scale_factor = 7

    histograms = np.vstack(
        [
            histogram,
            scale_factor * histogram,
        ]
    )

    features = extract_feature_table(
        histograms=histograms,
        time=time,
        config=_feature_config(),
    )

    scale_invariant_features = [
        "mean_arrival_time_ns",
        "arrival_time_variance_ns2",
        "arrival_time_skewness",
        "t10_ns",
        "t25_ns",
        "t50_ns",
        "t75_ns",
        "t90_ns",
    ]

    np.testing.assert_allclose(
        features.loc[0, scale_invariant_features],
        features.loc[1, scale_invariant_features],
    )

    np.testing.assert_allclose(
        features.loc[1, "total_counts"],
        scale_factor
        * features.loc[0, "total_counts"],
    )

    np.testing.assert_allclose(
        features.loc[1, "peak_height"],
        scale_factor
        * features.loc[0, "peak_height"],
    )


def test_engineered_features_are_sensitive_to_lifetime() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.1,
        dtype=np.float64,
    )

    lifetimes_ns = np.array(
        [1.0, 2.0, 4.0],
        dtype=np.float64,
    )

    histograms = np.vstack(
        [
            _make_integer_exponential_histogram(
                time,
                lifetime_ns=lifetime_ns,
            )
            for lifetime_ns in lifetimes_ns
        ]
    )

    features = extract_feature_table(
        histograms=histograms,
        time=time,
        config=_feature_config(),
    )

    mean_arrival_times = features[
        "mean_arrival_time_ns"
    ].to_numpy()

    t50_values = features[
        "t50_ns"
    ].to_numpy()

    tail_slopes = features[
        "tail_log_slope_per_ns"
    ].to_numpy()

    assert np.all(
        np.diff(mean_arrival_times) > 0.0
    )

    assert np.all(
        np.diff(t50_values) > 0.0
    )

    assert np.all(
        tail_slopes < 0.0
    )

    assert np.all(
        np.diff(tail_slopes) > 0.0
    )


def test_three_representations_preserve_sample_alignment() -> None:
    time = np.arange(
        0.0,
        20.0,
        0.1,
        dtype=np.float64,
    )

    lifetimes_ns = np.array(
        [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
        ],
        dtype=np.float64,
    )

    histograms = np.vstack(
        [
            _make_integer_exponential_histogram(
                time,
                lifetime_ns=lifetime_ns,
            )
            for lifetime_ns in lifetimes_ns
        ]
    )

    X_engineered = extract_feature_table(
        histograms=histograms,
        time=time,
        config=_feature_config(),
    ).to_numpy()

    X_histogram = normalize_histogram_batch(
        histograms=histograms,
        mode=CountNormalization.TOTAL,
    )

    n_train = 6
    n_components = 3

    pca = fit_pca_representation(
        X_train=X_histogram[:n_train],
        n_components=n_components,
    )

    X_pca = transform_pca_representation(
        pca=pca,
        X=X_histogram,
    )

    n_samples = histograms.shape[0]

    assert X_engineered.shape[0] == n_samples
    assert X_histogram.shape[0] == n_samples
    assert X_pca.shape == (
        n_samples,
        n_components,
    )
