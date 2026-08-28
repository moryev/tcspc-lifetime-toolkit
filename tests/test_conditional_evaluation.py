import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.conditional_evaluation import (
    add_benchmark_regimes,
    assign_numeric_regimes,
    build_prediction_diagnostics,
    summarize_conditional_performance,
    summarize_standard_regimes
)


def test_build_prediction_diagnostics_distinguishes_signed_and_absolute_error():
    y_true = np.array(
        [1.0, 2.0, 4.0],
        dtype=np.float64,
    )

    y_pred = np.array(
        [1.2, 1.5, 4.1],
        dtype=np.float64,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "background_per_bin": [
                0.0,
                1.0,
                2.0,
            ],
        }
    )

    diagnostics = (
        build_prediction_diagnostics(
            estimator_name="test_estimator",
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
        )
    )

    expected_error = np.array(
        [0.2, -0.5, 0.1],
        dtype=np.float64,
    )

    expected_absolute_error = np.array(
        [0.2, 0.5, 0.1],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        diagnostics[
            "error_ns"
        ].to_numpy(),
        expected_error,
    )

    np.testing.assert_allclose(
        diagnostics[
            "absolute_error_ns"
        ].to_numpy(),
        expected_absolute_error,
    )


def test_build_prediction_diagnostics_preserves_metadata_and_validity():
    y_true = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    y_pred = np.array(
        [1.1, np.nan, 2.8],
        dtype=np.float64,
    )

    valid_mask = np.array(
        [True, False, True],
        dtype=bool,
    )

    metadata = pd.DataFrame(
        {
            "sample_id": [10, 11, 12],
            "irf_fwhm_ns": [
                0.2,
                0.4,
                0.6,
            ],
        }
    )

    diagnostics = (
        build_prediction_diagnostics(
            estimator_name="test_estimator",
            y_true=y_true,
            y_pred=y_pred,
            metadata=metadata,
            valid_mask=valid_mask,
        )
    )

    assert diagnostics[
        "sample_id"
    ].tolist() == [
        10,
        11,
        12,
    ]

    assert diagnostics[
        "valid_estimate"
    ].tolist() == [
        True,
        False,
        True,
    ]

    assert np.isnan(
        diagnostics.loc[
            1,
            "error_ns",
        ]
    )


def test_summarize_conditional_performance_reports_error_statistics():
    diagnostics = pd.DataFrame(
        {
            "estimator_name": [
                "test_estimator",
                "test_estimator",
                "test_estimator",
                "test_estimator",
            ],
            "photon_regime": [
                "low",
                "low",
                "high",
                "high",
            ],
            "error_ns": [
                0.2,
                -0.2,
                0.1,
                0.5,
            ],
            "absolute_error_ns": [
                0.2,
                0.2,
                0.1,
                0.5,
            ],
            "valid_estimate": [
                True,
                True,
                True,
                True,
            ],
        }
    )

    summary = (
        summarize_conditional_performance(
            diagnostics,
            condition_column="photon_regime",
        )
    )

    low = summary.loc[
        summary[
            "photon_regime"
        ] == "low"
    ].iloc[0]

    assert low["n_samples"] == 2
    assert low["n_valid_estimates"] == 2
    assert low["n_failed_estimates"] == 0
    assert low["failure_rate"] == 0.0

    assert np.isclose(
        low["mae_ns"],
        0.2,
    )

    assert np.isclose(
        low["median_absolute_error_ns"],
        0.2,
    )

    assert np.isclose(
        low["bias_ns"],
        0.0,
    )


def test_summarize_conditional_performance_reports_failure_rate():
    diagnostics = pd.DataFrame(
        {
            "estimator_name": [
                "classical_reconvolution",
                "classical_reconvolution",
                "classical_reconvolution",
                "classical_reconvolution",
            ],
            "photon_regime": [
                "low",
                "low",
                "high",
                "high",
            ],
            "error_ns": [
                0.4,
                np.nan,
                0.1,
                -0.1,
            ],
            "absolute_error_ns": [
                0.4,
                np.nan,
                0.1,
                0.1,
            ],
            "valid_estimate": [
                True,
                False,
                True,
                True,
            ],
        }
    )

    summary = (
        summarize_conditional_performance(
            diagnostics,
            condition_column="photon_regime",
        )
    )

    low = summary.loc[
        summary[
            "photon_regime"
        ] == "low"
    ].iloc[0]

    assert low["n_samples"] == 2
    assert low["n_valid_estimates"] == 1
    assert low["n_failed_estimates"] == 1

    assert np.isclose(
        low["failure_rate"],
        0.5,
    )

    assert np.isclose(
        low["mae_ns"],
        0.4,
    )

    assert np.isclose(
        low["bias_ns"],
        0.4,
    )


def test_summarize_conditional_performance_reports_tail_errors():
    absolute_errors = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        dtype=np.float64,
    )

    diagnostics = pd.DataFrame(
        {
            "estimator_name": [
                "test_estimator"
            ] * 4,
            "lifetime_regime": [
                "medium"
            ] * 4,
            "error_ns": absolute_errors,
            "absolute_error_ns": (
                absolute_errors
            ),
            "valid_estimate": [
                True
            ] * 4,
        }
    )

    summary = (
        summarize_conditional_performance(
            diagnostics,
            condition_column="lifetime_regime",
        )
    )

    row = summary.iloc[0]

    assert np.isclose(
        row[
            "p90_absolute_error_ns"
        ],
        np.quantile(
            absolute_errors,
            0.90,
        ),
    )

    assert np.isclose(
        row[
            "p95_absolute_error_ns"
        ],
        np.quantile(
            absolute_errors,
            0.95,
        ),
    )


def test_assign_numeric_regimes_assigns_expected_labels():
    values = np.array(
        [
            0.5,
            1.5,
            2.5,
        ],
        dtype=np.float64,
    )

    regimes = assign_numeric_regimes(
        values,
        bin_edges=np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            dtype=np.float64,
        ),
        labels=(
            "low",
            "medium",
            "high",
        ),
    )

    assert regimes.tolist() == [
        "low",
        "medium",
        "high",
    ]


def test_add_benchmark_regimes_adds_standard_regime_columns():
    diagnostics = pd.DataFrame(
        {
            "true_lifetime_ns": [
                1.0,
                2.0,
                4.0,
            ],
            "signal_photon_count_target": [
                1_000,
                10_000,
                100_000,
            ],
            "background_per_bin": [
                0.1,
                1.0,
                10.0,
            ],
            "irf_fwhm_ns": [
                0.2,
                0.5,
                1.0,
            ],
            "irf_shift_ns": [
                0.02,
                0.15,
                0.40,
            ],
        }
    )

    result = add_benchmark_regimes(
        diagnostics,
        lifetime_edges=[
            0.0,
            1.5,
            3.0,
            np.inf,
        ],
        photon_count_edges=[
            0.0,
            5_000,
            50_000,
            np.inf,
        ],
        background_edges=[
            0.0,
            0.5,
            5.0,
            np.inf,
        ],
        irf_width_edges=[
            0.0,
            0.3,
            0.7,
            np.inf,
        ],
        irf_misalignment_edges=[
            0.0,
            0.10,
            0.30,
            np.inf,
        ],
    )

    assert result[
        "lifetime_regime"
    ].tolist() == [
        "short",
        "medium",
        "long",
    ]

    assert result[
        "photon_count_regime"
    ].tolist() == [
        "low",
        "medium",
        "high",
    ]

    assert result[
        "background_regime"
    ].tolist() == [
        "low",
        "medium",
        "high",
    ]

    assert result[
        "irf_width_regime"
    ].tolist() == [
        "narrow",
        "medium",
        "broad",
    ]

    assert result[
        "irf_misalignment_regime"
    ].tolist() == [
        "small",
        "medium",
        "large",
    ]


def test_irf_misalignment_regime_uses_absolute_shift():
    diagnostics = pd.DataFrame(
        {
            "true_lifetime_ns": [
                2.0,
                2.0,
                2.0,
            ],
            "signal_photon_count_target": [
                10_000,
                10_000,
                10_000,
            ],
            "background_per_bin": [
                1.0,
                1.0,
                1.0,
            ],
            "irf_fwhm_ns": [
                0.5,
                0.5,
                0.5,
            ],
            "irf_shift_ns": [
                -0.4,
                0.0,
                0.4,
            ],
        }
    )

    result = add_benchmark_regimes(
        diagnostics,
        lifetime_edges=[
            0.0,
            1.5,
            3.0,
            np.inf,
        ],
        photon_count_edges=[
            0.0,
            5_000,
            50_000,
            np.inf,
        ],
        background_edges=[
            0.0,
            0.5,
            5.0,
            np.inf,
        ],
        irf_width_edges=[
            0.0,
            0.3,
            0.7,
            np.inf,
        ],
        irf_misalignment_edges=[
            0.0,
            0.10,
            0.30,
            np.inf,
        ],
    )

    np.testing.assert_allclose(
        result[
            "absolute_irf_shift_ns"
        ].to_numpy(),
        np.array(
            [
                0.4,
                0.0,
                0.4,
            ]
        ),
    )

    assert result[
        "irf_misalignment_regime"
    ].tolist() == [
        "large",
        "small",
        "large",
    ]


def test_assign_numeric_regimes_rejects_values_outside_edges():
    values = np.array(
        [
            1.0,
            5.0,
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        assign_numeric_regimes(
            values,
            bin_edges=[
                0.0,
                2.0,
                3.0,
                4.0,
            ],
            labels=(
                "low",
                "medium",
                "high",
            ),
        )


def test_standard_regime_workflow_compares_estimators():
    metadata = pd.DataFrame(
        {
            "sample_id": [
                0,
                1,
                2,
            ],
            "signal_photon_count_target": [
                5_000,
                25_000,
                100_000,
            ],
            "background_per_bin": [
                0.0,
                0.5,
                2.0,
            ],
            "irf_fwhm_ns": [
                0.15,
                0.30,
                0.60,
            ],
            "irf_shift_ns": [
                -0.20,
                0.00,
                0.20,
            ],
        }
    )

    y_true = np.array(
        [
            1.0,
            2.5,
            4.5,
        ],
        dtype=np.float64,
    )

    ridge_diagnostics = (
        build_prediction_diagnostics(
            estimator_name="ridge",
            y_true=y_true,
            y_pred=np.array(
                [
                    1.1,
                    2.4,
                    4.6,
                ],
                dtype=np.float64,
            ),
            metadata=metadata,
        )
    )

    classical_diagnostics = (
        build_prediction_diagnostics(
            estimator_name=(
                "classical_reconvolution"
            ),
            y_true=y_true,
            y_pred=np.array(
                [
                    1.2,
                    2.5,
                    np.nan,
                ],
                dtype=np.float64,
            ),
            metadata=metadata,
        )
    )

    diagnostics = pd.concat(
        [
            ridge_diagnostics,
            classical_diagnostics,
        ],
        ignore_index=True,
    )

    diagnostics = add_benchmark_regimes(
        diagnostics,
        lifetime_edges=[
            0.0,
            1.75,
            3.25,
            np.inf,
        ],
        photon_count_edges=[
            0.0,
            10_000,
            50_000,
            np.inf,
        ],
        background_edges=[
            0.0,
            0.25,
            1.25,
            np.inf,
        ],
        irf_width_edges=[
            0.0,
            0.225,
            0.45,
            np.inf,
        ],
        irf_misalignment_edges=[
            0.0,
            0.10,
            np.inf,
        ],
        irf_misalignment_labels=(
            "aligned",
            "shifted",
        ),
    )

    summaries = (
        summarize_standard_regimes(
            diagnostics
        )
    )

    assert set(
        summaries
    ) == {
        "lifetime",
        "photon_count",
        "background",
        "irf_width",
        "irf_misalignment",
    }

    lifetime_summary = (
        summaries[
            "lifetime"
        ]
    )

    assert set(
        lifetime_summary[
            "estimator_name"
        ]
    ) == {
        "ridge",
        "classical_reconvolution",
    }

    classical_long = (
        lifetime_summary.loc[
            (
                lifetime_summary[
                    "estimator_name"
                ]
                == "classical_reconvolution"
            )
            & (
                lifetime_summary[
                    "lifetime_regime"
                ]
                == "long"
            )
        ]
        .iloc[0]
    )

    assert (
        classical_long[
            "n_samples"
        ]
        == 1
    )

    assert (
        classical_long[
            "n_failed_estimates"
        ]
        == 1
    )

    assert np.isclose(
        classical_long[
            "failure_rate"
        ],
        1.0,
    )

    assert np.isnan(
        classical_long[
            "mae_ns"
        ]
    )


def test_standard_regime_summary_keeps_estimators_separate():
    diagnostics = pd.DataFrame(
        {
            "estimator_name": [
                "model_a",
                "model_b",
            ],
            "lifetime_regime": [
                "short",
                "short",
            ],
            "photon_count_regime": [
                "low",
                "low",
            ],
            "background_regime": [
                "low",
                "low",
            ],
            "irf_width_regime": [
                "narrow",
                "narrow",
            ],
            "irf_misalignment_regime": [
                "aligned",
                "aligned",
            ],
            "error_ns": [
                0.1,
                1.0,
            ],
            "absolute_error_ns": [
                0.1,
                1.0,
            ],
            "valid_estimate": [
                True,
                True,
            ],
        }
    )

    summaries = (
        summarize_standard_regimes(
            diagnostics
        )
    )

    lifetime_summary = (
        summaries["lifetime"]
    )

    assert lifetime_summary.shape[0] == 2

    model_a = lifetime_summary.loc[
        lifetime_summary[
            "estimator_name"
        ] == "model_a"
    ].iloc[0]

    model_b = lifetime_summary.loc[
        lifetime_summary[
            "estimator_name"
        ] == "model_b"
    ].iloc[0]

    assert np.isclose(
        model_a["mae_ns"],
        0.1,
    )

    assert np.isclose(
        model_b["mae_ns"],
        1.0,
    )
