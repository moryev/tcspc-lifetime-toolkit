import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tcspc_toolkit.benchmark_plots import (
    plot_error_histogram,
    plot_error_vs_condition,
    plot_paired_absolute_error_comparison,
    plot_true_vs_predicted,
)


def _make_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                0,
                1,
                2,
            ],
            "estimator_name": [
                "test_estimator"
            ] * 3,
            "true_lifetime_ns": [
                1.0,
                2.0,
                3.0,
            ],
            "predicted_lifetime_ns": [
                1.1,
                1.8,
                3.3,
            ],
            "error_ns": [
                0.1,
                -0.2,
                0.3,
            ],
            "absolute_error_ns": [
                0.1,
                0.2,
                0.3,
            ],
            "valid_estimate": [
                True,
                True,
                True,
            ],
            "signal_photon_count_target": [
                5_000,
                25_000,
                100_000,
            ],
        }
    )


def test_plot_true_vs_predicted_returns_axes():
    diagnostics = (
        _make_diagnostics()
    )

    ax = plot_true_vs_predicted(
        diagnostics
    )

    assert ax.get_xlabel() == (
        "True lifetime (ns)"
    )

    assert ax.get_ylabel() == (
        "Predicted lifetime (ns)"
    )

    plt.close(
        ax.figure
    )


def test_plot_error_vs_condition_uses_requested_error_type():
    diagnostics = (
        _make_diagnostics()
    )

    ax = plot_error_vs_condition(
        diagnostics,
        condition_column=(
            "true_lifetime_ns"
        ),
        absolute=True,
    )

    plotted_values = (
        ax.collections[0]
        .get_offsets()[
            :,
            1,
        ]
    )

    np.testing.assert_allclose(
        plotted_values,
        diagnostics[
            "absolute_error_ns"
        ].to_numpy(),
    )

    plt.close(
        ax.figure
    )


def test_plot_error_vs_condition_supports_logarithmic_x_axis():
    diagnostics = (
        _make_diagnostics()
    )

    ax = plot_error_vs_condition(
        diagnostics,
        condition_column=(
            "signal_photon_count_target"
        ),
        log_x=True,
    )

    assert ax.get_xscale() == "log"

    plt.close(
        ax.figure
    )


def test_plot_error_vs_condition_excludes_failed_estimates():
    diagnostics = (
        _make_diagnostics()
    )

    diagnostics.loc[
        1,
        "valid_estimate",
    ] = False

    diagnostics.loc[
        1,
        "error_ns",
    ] = np.nan

    ax = plot_error_vs_condition(
        diagnostics,
        condition_column=(
            "true_lifetime_ns"
        ),
    )

    plotted_values = (
        ax.collections[0]
        .get_offsets()
    )

    assert plotted_values.shape[0] == 2

    plt.close(
        ax.figure
    )


def test_paired_error_comparison_aligns_samples_by_id():
    diagnostics_x = pd.DataFrame(
        {
            "sample_id": [
                10,
                11,
                12,
            ],
            "estimator_name": [
                "classical"
            ] * 3,
            "absolute_error_ns": [
                0.3,
                0.1,
                0.2,
            ],
            "valid_estimate": [
                True,
                True,
                True,
            ],
        }
    )

    diagnostics_y = pd.DataFrame(
        {
            "sample_id": [
                12,
                10,
                11,
            ],
            "estimator_name": [
                "ridge"
            ] * 3,
            "absolute_error_ns": [
                0.1,
                0.2,
                0.05,
            ],
            "valid_estimate": [
                True,
                True,
                True,
            ],
        }
    )

    ax = (
        plot_paired_absolute_error_comparison(
            diagnostics_x,
            diagnostics_y,
        )
    )

    plotted_points = (
        ax.collections[0]
        .get_offsets()
    )

    expected_points = np.array(
        [
            [0.3, 0.2],
            [0.1, 0.05],
            [0.2, 0.1],
        ]
    )

    np.testing.assert_allclose(
        plotted_points,
        expected_points,
    )

    plt.close(
        ax.figure
    )


