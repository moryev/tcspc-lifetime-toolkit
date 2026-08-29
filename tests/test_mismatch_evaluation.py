import numpy as np
import pytest

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.mismatch_evaluation import (
    evaluate_ml_mismatch_benchmark,
    generate_matched_biexponential_mismatch_dataset,
    summarize_mismatch_benchmark,
)
from tcspc_toolkit.ml_evaluation import (
    build_benchmark_dataset,
    generate_benchmark_measurements,
    split_benchmark_dataset,
)


def _make_reference_split():
    time = np.arange(
        0.0,
        20.0,
        0.05,
        dtype=np.float64,
    )

    measurements = generate_benchmark_measurements(
        time=time,
        lifetimes_ns=np.array(
            [1.0, 2.0, 3.0, 4.0],
            dtype=np.float64,
        ),
        signal_photon_counts=np.array(
            [5_000, 20_000],
            dtype=np.int64,
        ),
        background_levels=np.array(
            [0.5, 2.0],
            dtype=np.float64,
        ),
        irf_centre_ns=1.0,
        irf_fwhm_values_ns=np.array(
            [0.25, 0.40],
            dtype=np.float64,
        ),
        irf_shift_values_ns=np.array(
            [-0.05, 0.05],
            dtype=np.float64,
        ),
        random_seed=42,
    )

    feature_config = FeatureConfig(
        tail_start_ns=5.0,
        early_stop_ns=3.0,
        late_start_ns=7.0,
    )

    dataset = build_benchmark_dataset(
        measurements,
        feature_config=feature_config,
    )

    split = split_benchmark_dataset(
        dataset,
        test_size=0.25,
        random_state=42,
    )

    return (
        time,
        feature_config,
        split,
    )


def test_matched_mismatch_preserves_targets_and_sample_count() -> None:
    (
        time,
        feature_config,
        split,
    ) = _make_reference_split()

    mismatch = (
        generate_matched_biexponential_mismatch_dataset(
            time=time,
            reference_split=split,
            feature_config=feature_config,
            irf_centre_ns=1.0,
            secondary_fraction=0.10,
            secondary_lifetime_factor=2.0,
            random_seed=123,
        )
    )

    np.testing.assert_array_equal(
        mismatch.y,
        split.y_test,
    )

    assert (
        mismatch.X_histograms.shape
        == split.X_histograms_test.shape
    )

    assert (
        mismatch.X_features.shape
        == split.X_features_test.shape
    )


def test_matched_mismatch_preserves_nuisance_parameters() -> None:
    (
        time,
        feature_config,
        split,
    ) = _make_reference_split()

    mismatch = (
        generate_matched_biexponential_mismatch_dataset(
            time=time,
            reference_split=split,
            feature_config=feature_config,
            irf_centre_ns=1.0,
            secondary_fraction=0.10,
            secondary_lifetime_factor=2.0,
            random_seed=123,
        )
    )

    nuisance_columns = (
        "signal_photon_count_target",
        "background_per_bin",
        "irf_fwhm_ns",
        "irf_shift_ns",
    )

    for column in nuisance_columns:
        np.testing.assert_allclose(
            mismatch.metadata[
                column
            ].to_numpy(),
            split.metadata_test[
                column
            ].to_numpy(),
        )


def test_matched_mismatch_secondary_lifetime_uses_requested_factor() -> None:
    (
        time,
        feature_config,
        split,
    ) = _make_reference_split()

    mismatch = (
        generate_matched_biexponential_mismatch_dataset(
            time=time,
            reference_split=split,
            feature_config=feature_config,
            irf_centre_ns=1.0,
            secondary_fraction=0.10,
            secondary_lifetime_factor=2.0,
            random_seed=123,
        )
    )

    expected_secondary_lifetimes = (
        2.0
        * split.y_test
    )

    np.testing.assert_allclose(
        mismatch.metadata[
            "secondary_lifetime_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        expected_secondary_lifetimes,
    )

    np.testing.assert_allclose(
        mismatch.metadata[
            "secondary_fraction"
        ].to_numpy(
            dtype=np.float64
        ),
        0.10,
    )


def test_matched_mismatch_is_reproducible() -> None:
    (
        time,
        feature_config,
        split,
    ) = _make_reference_split()

    kwargs = {
        "time": time,
        "reference_split": split,
        "feature_config": feature_config,
        "irf_centre_ns": 1.0,
        "secondary_fraction": 0.10,
        "secondary_lifetime_factor": 2.0,
        "random_seed": 123,
    }

    mismatch_a = (
        generate_matched_biexponential_mismatch_dataset(
            **kwargs
        )
    )

    mismatch_b = (
        generate_matched_biexponential_mismatch_dataset(
            **kwargs
        )
    )

    np.testing.assert_array_equal(
        mismatch_a.X_histograms,
        mismatch_b.X_histograms,
    )


def test_ml_mismatch_benchmark_returns_all_estimators() -> None:
    (
        time,
        feature_config,
        split,
    ) = _make_reference_split()

    mismatch = (
        generate_matched_biexponential_mismatch_dataset(
            time=time,
            reference_split=split,
            feature_config=feature_config,
            irf_centre_ns=1.0,
            secondary_fraction=0.10,
            secondary_lifetime_factor=2.0,
            random_seed=123,
        )
    )

    results = evaluate_ml_mismatch_benchmark(
        reference_split=split,
        mismatch_dataset=mismatch,
    )

    assert set(
        results
    ) == {
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
    }

    for result in results.values():
        assert (
            result.in_distribution.y_pred.shape
            == split.y_test.shape
        )

        assert (
            result.mismatch.y_pred.shape
            == split.y_test.shape
        )


def test_mismatch_summary_contains_degradation_metrics() -> None:
    (
        time,
        feature_config,
        split,
    ) = _make_reference_split()

    mismatch = (
        generate_matched_biexponential_mismatch_dataset(
            time=time,
            reference_split=split,
            feature_config=feature_config,
            irf_centre_ns=1.0,
            secondary_fraction=0.10,
            secondary_lifetime_factor=2.0,
            random_seed=123,
        )
    )

    results = evaluate_ml_mismatch_benchmark(
        reference_split=split,
        mismatch_dataset=mismatch,
    )

    summary = summarize_mismatch_benchmark(
        y_true=split.y_test,
        ml_results=results,
    )

    assert summary.shape[0] == 3

    expected_columns = {
        "estimator",
        "in_distribution_mae_ns",
        "mismatch_mae_ns",
        "mae_change_ns",
        "mae_ratio",
        "in_distribution_bias_ns",
        "mismatch_bias_ns",
        "in_distribution_failure_rate",
        "mismatch_failure_rate",
    }

    assert expected_columns.issubset(
        summary.columns
    )


