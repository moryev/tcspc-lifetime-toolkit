"""Tests for TCSPC feature extraction."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.exceptions import (
    FeatureExtractionError,
    InvalidHistogramError,
)
from tcspc_toolkit.features import (
    FEATURE_NAMES,
    early_late_count_ratio,
    extract_feature_table,
    extract_features,
    half_decay_time,
    integrated_tail_fraction,
    quantile_arrival_time,
    tail_log_slope,
)


@pytest.fixture
def simple_histogram() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    time = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, 4.0, 9.0, 3.0, 1.0],
        dtype=np.float64,
    )

    return time, counts


@pytest.fixture
def feature_config() -> FeatureConfig:
    return FeatureConfig(
        tail_start_ns=0.2,
        early_stop_ns=0.1,
        late_start_ns=0.2,
    )


def test_extract_features_returns_dataframe(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (
        1,
        len(FEATURE_NAMES),
    )


def test_extract_features_has_stable_columns(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    assert list(result.columns) == list(FEATURE_NAMES)


def test_extract_features_calculates_total_counts(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    assert result.loc[0, "total_counts"] == 18.0


def test_extract_features_calculates_peak_height(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    assert result.loc[0, "peak_height"] == 9.0


def test_extract_features_calculates_peak_time(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    assert result.loc[0, "peak_time_ns"] == 0.2


def test_extract_features_does_not_modify_inputs(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    original_time = time.copy()
    original_counts = counts.copy()

    extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    np.testing.assert_array_equal(
        time,
        original_time,
    )
    np.testing.assert_array_equal(
        counts,
        original_counts,
    )


def test_extract_features_rejects_invalid_histogram(
    feature_config: FeatureConfig,
) -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, -2.0, 3.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="counts must be non-negative",
    ):
        extract_features(
            time=time,
            counts=counts,
            config=feature_config,
        )


@pytest.fixture
def moment_histogram() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    time = np.array(
        [0.0, 1.0, 2.0],
        dtype=np.float64,
    )
    counts = np.array(
        [3.0, 1.0, 1.0],
        dtype=np.float64,
    )

    return time, counts


@pytest.fixture
def moment_feature_config() -> FeatureConfig:
    return FeatureConfig(
        tail_start_ns=0.0,
        early_stop_ns=0.0,
        late_start_ns=1.0,
    )


def test_extract_features_calculates_mean_arrival_time(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    moment_feature_config: FeatureConfig,
) -> None:
    time, counts = moment_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=moment_feature_config,
    )

    assert result.loc[
        0,
        "mean_arrival_time_ns",
    ] == pytest.approx(0.6)


def test_extract_features_calculates_arrival_time_variance(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    moment_feature_config: FeatureConfig,
) -> None:
    time, counts = moment_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=moment_feature_config,
    )

    assert result.loc[
        0,
        "arrival_time_variance_ns2",
    ] == pytest.approx(0.64)


def test_extract_features_calculates_arrival_time_skewness(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    moment_feature_config: FeatureConfig,
) -> None:
    time, counts = moment_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=moment_feature_config,
    )

    expected_skewness = 0.84375

    assert result.loc[
        0,
        "arrival_time_skewness",
    ] == pytest.approx(expected_skewness)


def test_arrival_moments_are_invariant_to_count_scaling(
    moment_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    moment_feature_config: FeatureConfig,
) -> None:
    time, counts = moment_histogram

    original = extract_features(
        time=time,
        counts=counts,
        config=moment_feature_config,
    )

    scaled = extract_features(
        time=time,
        counts=10.0 * counts,
        config=moment_feature_config,
    )

    moment_columns = [
        "mean_arrival_time_ns",
        "arrival_time_variance_ns2",
        "arrival_time_skewness",
    ]

    np.testing.assert_allclose(
        original.loc[0, moment_columns].to_numpy(
            dtype=np.float64
        ),
        scaled.loc[0, moment_columns].to_numpy(
            dtype=np.float64
        ),
    )


def test_extract_features_rejects_all_zero_histogram(
    feature_config: FeatureConfig,
) -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.zeros(
        3,
        dtype=np.float64,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="all-zero histogram",
    ):
        extract_features(
            time=time,
            counts=counts,
            config=feature_config,
        )


def test_extract_features_rejects_zero_arrival_time_variance(
    feature_config: FeatureConfig,
) -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [0.0, 10.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="variance is zero",
    ):
        extract_features(
            time=time,
            counts=counts,
            config=feature_config,
        )


def test_extract_features_rejects_non_finite_counts(
    feature_config: FeatureConfig,
) -> None:
    time = np.array(
        [0.0, 0.1, 0.2],
        dtype=np.float64,
    )
    counts = np.array(
        [1.0, np.nan, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidHistogramError,
        match="counts must contain only finite values",
    ):
        extract_features(
            time=time,
            counts=counts,
            config=feature_config,
        )


def test_quantile_arrival_time_exact_cumulative_position() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    counts = np.array([1.0, 1.0, 2.0, 0.0])

    result = quantile_arrival_time(
        time,
        counts,
        quantile=0.5,
    )

    assert result == pytest.approx(1.0)


def test_quantile_arrival_time_interpolates() -> None:
    time = np.array([0.0, 1.0, 2.0])
    counts = np.array([0.0, 1.0, 1.0])

    result = quantile_arrival_time(
        time,
        counts,
        quantile=0.25,
    )

    assert result == pytest.approx(0.5)


def test_quantile_arrival_times_are_ordered() -> None:
    time = np.arange(6, dtype=float)

    counts = np.array([
        1.0,
        2.0,
        5.0,
        4.0,
        2.0,
        1.0,
    ])

    t10 = quantile_arrival_time(time, counts, 0.10)
    t25 = quantile_arrival_time(time, counts, 0.25)
    t50 = quantile_arrival_time(time, counts, 0.50)
    t75 = quantile_arrival_time(time, counts, 0.75)
    t90 = quantile_arrival_time(time, counts, 0.90)

    assert t10 <= t25 <= t50 <= t75 <= t90


def test_half_decay_time_for_ideal_exponential() -> None:
    lifetime = 2.0
    amplitude = 100_000

    time = np.linspace(
        0.0,
        10.0,
        1001,
    )

    counts = np.rint(
        amplitude * np.exp(-time / lifetime)
    ).astype(int)

    result = half_decay_time(
        time,
        counts,
    )

    expected = lifetime * np.log(2.0)

    assert result == pytest.approx(
        expected,
        abs=0.01,
    )


def test_half_decay_time_uses_post_peak_crossing() -> None:
    time = np.arange(7, dtype=float)

    counts = np.array([
        1.0,
        2.0,
        4.0,
        10.0,
        8.0,
        4.0,
        3.0,
    ])

    result = half_decay_time(
        time,
        counts,
    )

    assert result == pytest.approx(1.75)


def test_half_decay_time_raises_when_no_crossing_occurs() -> None:
    time = np.arange(4, dtype=float)

    counts = np.array([
        10.0,
        9.0,
        8.0,
        7.0,
    ])

    with pytest.raises(
        FeatureExtractionError,
        match="No post-peak half-height crossing",
    ):
        half_decay_time(time, counts)


def test_half_decay_time_raises_when_peak_is_last_bin() -> None:
    time = np.arange(3, dtype=float)

    counts = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(
        FeatureExtractionError,
        match="final bin",
    ):
        half_decay_time(time, counts)


def test_tail_log_slope_for_ideal_exponential() -> None:
    lifetime = 2.0
    amplitude = 1_000_000

    time = np.linspace(
        0.0,
        10.0,
        1001,
    )

    counts = np.rint(
        amplitude * np.exp(-time / lifetime)
    ).astype(int)

    result = tail_log_slope(
        time=time,
        counts=counts,
        tail_start_ns=2.0,
    )

    expected = -1.0 / lifetime

    assert result == pytest.approx(
        expected,
        abs=0.01,
    )


def test_tail_log_slope_ignores_zero_count_bins() -> None:
    time = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )

    counts = np.array(
        [16.0, 8.0, 0.0, 2.0, 0.0]
    )

    result = tail_log_slope(
        time=time,
        counts=counts,
        tail_start_ns=0.0,
    )

    assert result == pytest.approx(
        -np.log(2.0)
    )


def test_tail_log_slope_rejects_insufficient_positive_bins() -> None:
    time = np.arange(
        5,
        dtype=float,
    )

    counts = np.array([
        8.0,
        4.0,
        0.0,
        0.0,
        0.0,
    ])

    with pytest.raises(
        FeatureExtractionError,
        match="Insufficient positive-count bins",
    ):
        tail_log_slope(
            time=time,
            counts=counts,
            tail_start_ns=0.0,
        )


def test_tail_log_slope_rejects_missing_tail_region() -> None:
    time = np.arange(
        4,
        dtype=float,
    )

    counts = np.array([
        8.0,
        4.0,
        2.0,
        1.0,
    ])

    with pytest.raises(
        FeatureExtractionError,
        match="Tail region contains no histogram bins",
    ):
        tail_log_slope(
            time=time,
            counts=counts,
            tail_start_ns=10.0,
        )


def test_integrated_tail_fraction_matches_expected_value() -> None:
    time = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    counts = np.array([
        10.0,
        10.0,
        5.0,
        5.0,
    ])

    result = integrated_tail_fraction(
        time=time,
        counts=counts,
        tail_start_ns=2.0,
    )

    assert result == pytest.approx(
        1.0 / 3.0
    )


def test_integrated_tail_fraction_is_bounded() -> None:
    time = np.arange(
        5,
        dtype=float,
    )

    counts = np.array([
        10.0,
        7.0,
        3.0,
        1.0,
        0.0,
    ])

    result = integrated_tail_fraction(
        time=time,
        counts=counts,
        tail_start_ns=2.0,
    )

    assert 0.0 <= result <= 1.0


def test_integrated_tail_fraction_allows_zero_count_tail() -> None:
    time = np.arange(
        5,
        dtype=float,
    )

    counts = np.array([
        5.0,
        2.0,
        1.0,
        0.0,
        0.0,
    ])

    result = integrated_tail_fraction(
        time=time,
        counts=counts,
        tail_start_ns=3.0,
    )

    assert result == 0.0


def test_early_late_count_ratio_matches_expected_value() -> None:
    time = np.arange(
        7,
        dtype=float,
    )

    counts = np.array([
        20.0,
        15.0,
        10.0,
        5.0,
        3.0,
        2.0,
        1.0,
    ])

    result = early_late_count_ratio(
        time=time,
        counts=counts,
        early_stop_ns=2.0,
        late_start_ns=4.0,
    )

    assert result == pytest.approx(
        7.5
    )


def test_early_late_count_ratio_allows_zero_early_counts() -> None:
    time = np.arange(
        5,
        dtype=float,
    )

    counts = np.array([
        0.0,
        0.0,
        4.0,
        2.0,
        1.0,
    ])

    result = early_late_count_ratio(
        time=time,
        counts=counts,
        early_stop_ns=1.0,
        late_start_ns=2.0,
    )

    assert result == 0.0


def test_early_late_count_ratio_rejects_zero_late_counts() -> None:
    time = np.arange(
        5,
        dtype=float,
    )

    counts = np.array([
        10.0,
        5.0,
        2.0,
        0.0,
        0.0,
    ])

    with pytest.raises(
        FeatureExtractionError,
        match="late region contains zero counts",
    ):
        early_late_count_ratio(
            time=time,
            counts=counts,
            early_stop_ns=1.0,
            late_start_ns=3.0,
        )


def test_extract_features_includes_tail_features(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    expected_slope = tail_log_slope(
        time=time,
        counts=counts,
        tail_start_ns=feature_config.tail_start_ns,
        min_points=feature_config.min_tail_points,
    )

    expected_fraction = integrated_tail_fraction(
        time=time,
        counts=counts,
        tail_start_ns=feature_config.tail_start_ns,
    )

    expected_ratio = early_late_count_ratio(
        time=time,
        counts=counts,
        early_stop_ns=feature_config.early_stop_ns,
        late_start_ns=feature_config.late_start_ns,
    )

    assert result.loc[
        0,
        "tail_log_slope_per_ns",
    ] == pytest.approx(expected_slope)

    assert result.loc[
        0,
        "integrated_tail_fraction",
    ] == pytest.approx(expected_fraction)

    assert result.loc[
        0,
        "early_late_count_ratio",
    ] == pytest.approx(expected_ratio)


def test_feature_names_define_public_schema() -> None:
    assert FEATURE_NAMES == (
        "total_counts",
        "peak_height",
        "peak_time_ns",
        "mean_arrival_time_ns",
        "arrival_time_variance_ns2",
        "arrival_time_skewness",
        "t10_ns",
        "t25_ns",
        "t50_ns",
        "t75_ns",
        "t90_ns",
        "half_decay_time_ns",
        "tail_log_slope_per_ns",
        "integrated_tail_fraction",
        "early_late_count_ratio",
    )


def test_extract_feature_table_returns_expected_shape(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    histograms = np.vstack([
        counts,
        counts,
        counts,
    ])

    result = extract_feature_table(
        histograms=histograms,
        time=time,
        config=feature_config,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (
        3,
        len(FEATURE_NAMES),
    )


def test_extract_feature_table_has_stable_columns(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    histograms = np.vstack([
        counts,
        counts,
    ])

    result = extract_feature_table(
        histograms=histograms,
        time=time,
        config=feature_config,
    )

    assert list(result.columns) == list(FEATURE_NAMES)


def test_single_and_batch_feature_extraction_are_identical(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    single_result = extract_features(
        time=time,
        counts=counts,
        config=feature_config,
    )

    batch_result = extract_feature_table(
        histograms=counts[np.newaxis, :],
        time=time,
        config=feature_config,
    )

    pd.testing.assert_frame_equal(
        single_result,
        batch_result,
    )


def test_extract_feature_table_does_not_modify_inputs(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    histograms = np.vstack([
        counts,
        2.0 * counts,
    ])

    original_time = time.copy()
    original_histograms = histograms.copy()

    extract_feature_table(
        histograms=histograms,
        time=time,
        config=feature_config,
    )

    np.testing.assert_array_equal(
        time,
        original_time,
    )

    np.testing.assert_array_equal(
        histograms,
        original_histograms,
    )


def test_extract_feature_table_handles_heterogeneous_count_levels(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    histograms = np.vstack([
        counts,
        10.0 * counts,
        100.0 * counts,
    ])

    result = extract_feature_table(
        histograms=histograms,
        time=time,
        config=feature_config,
    )

    np.testing.assert_allclose(
        result["total_counts"].to_numpy(),
        np.array([
            18.0,
            180.0,
            1800.0,
        ]),
    )

    np.testing.assert_allclose(
        result["peak_height"].to_numpy(),
        np.array([
            9.0,
            90.0,
            900.0,
        ]),
    )

    scale_invariant_columns = [
        name
        for name in FEATURE_NAMES
        if name not in {
            "total_counts",
            "peak_height",
        }
    ]

    reference = result.loc[
        0,
        scale_invariant_columns,
    ].to_numpy(dtype=np.float64)

    for row_index in range(1, len(result)):
        np.testing.assert_allclose(
            result.loc[
                row_index,
                scale_invariant_columns,
            ].to_numpy(dtype=np.float64),
            reference,
        )


def test_extract_feature_table_rejects_one_dimensional_histograms(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    with pytest.raises(
        ValueError,
        match="histograms must be a two-dimensional array",
    ):
        extract_feature_table(
            histograms=counts,
            time=time,
            config=feature_config,
        )


def test_extract_feature_table_rejects_bin_count_mismatch(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    histograms = np.vstack([
        counts[:-1],
        counts[:-1],
    ])

    with pytest.raises(
        ValueError,
        match="same number of bins as time",
    ):
        extract_feature_table(
            histograms=histograms,
            time=time,
            config=feature_config,
        )


def test_extract_feature_table_rejects_empty_batch(
    simple_histogram: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    feature_config: FeatureConfig,
) -> None:
    time, counts = simple_histogram

    histograms = np.empty(
        (0, counts.size),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="at least one histogram",
    ):
        extract_feature_table(
            histograms=histograms,
            time=time,
            config=feature_config,
        )
