"""Reusable plotting utilities for TCSPC lifetime benchmarks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


_CONDITION_LABELS = {
    "true_lifetime_ns": "True lifetime (ns)",
    "signal_photon_count_target": "Signal photon count",
    "background_per_bin": "Background (counts/bin)",
    "irf_fwhm_ns": "IRF FWHM (ns)",
    "irf_shift_ns": "IRF shift (ns)",
    "absolute_irf_shift_ns": "Absolute IRF shift (ns)",
}


def _require_columns(
    diagnostics: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Validate required plotting columns."""

    missing_columns = (
        required_columns
        - set(diagnostics.columns)
    )

    if missing_columns:
        raise ValueError(
            "Diagnostics are missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


def _get_axes(
    ax: Axes | None,
) -> Axes:
    """Return supplied axes or create a new figure and axes."""

    if ax is None:
        _, ax = plt.subplots()

    return ax


def _valid_plot_rows(
    diagnostics: pd.DataFrame,
    *,
    numeric_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return valid estimates with finite plotting values."""

    _require_columns(
        diagnostics,
        {
            "valid_estimate",
            *numeric_columns,
        },
    )

    valid_mask = diagnostics[
        "valid_estimate"
    ].to_numpy(
        dtype=bool
    )

    valid_rows = diagnostics.loc[
        valid_mask
    ].copy()

    if valid_rows.empty:
        raise ValueError(
            "No valid estimates are available for plotting."
        )

    values = valid_rows.loc[
        :,
        list(numeric_columns),
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Valid estimates must contain finite plotting values."
        )

    return valid_rows


def plot_true_vs_predicted(
    diagnostics: pd.DataFrame,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot predicted lifetime against true lifetime."""

    _require_columns(
        diagnostics,
        {
            "estimator_name",
            "true_lifetime_ns",
            "predicted_lifetime_ns",
            "valid_estimate",
        },
    )

    valid_rows = _valid_plot_rows(
        diagnostics,
        numeric_columns=(
            "true_lifetime_ns",
            "predicted_lifetime_ns",
        ),
    )

    ax = _get_axes(
        ax
    )

    for estimator_name, group in (
        valid_rows.groupby(
            "estimator_name",
            sort=False,
        )
    ):
        ax.scatter(
            group[
                "true_lifetime_ns"
            ],
            group[
                "predicted_lifetime_ns"
            ],
            label=estimator_name,
        )

    lower_limit = float(
        min(
            valid_rows[
                "true_lifetime_ns"
            ].min(),
            valid_rows[
                "predicted_lifetime_ns"
            ].min(),
        )
    )

    upper_limit = float(
        max(
            valid_rows[
                "true_lifetime_ns"
            ].max(),
            valid_rows[
                "predicted_lifetime_ns"
            ].max(),
        )
    )

    ax.plot(
        [
            lower_limit,
            upper_limit,
        ],
        [
            lower_limit,
            upper_limit,
        ],
        linestyle="--",
    )

    ax.set_xlabel(
        "True lifetime (ns)"
    )

    ax.set_ylabel(
        "Predicted lifetime (ns)"
    )

    ax.set_title(
        "True vs predicted lifetime"
    )

    ax.legend()

    return ax


def plot_error_histogram(
    diagnostics: pd.DataFrame,
    *,
    absolute: bool = False,
    bins: int = 30,
    ax: Axes | None = None,
) -> Axes:
    """Plot the distribution of signed or absolute lifetime errors."""

    error_column = (
        "absolute_error_ns"
        if absolute
        else "error_ns"
    )

    _require_columns(
        diagnostics,
        {
            "estimator_name",
            error_column,
            "valid_estimate",
        },
    )

    valid_rows = _valid_plot_rows(
        diagnostics,
        numeric_columns=(
            error_column,
        ),
    )

    ax = _get_axes(
        ax
    )

    for estimator_name, group in (
        valid_rows.groupby(
            "estimator_name",
            sort=False,
        )
    ):
        ax.hist(
            group[
                error_column
            ],
            bins=bins,
            alpha=0.6,
            label=estimator_name,
        )

    if absolute:
        ax.set_xlabel(
            "Absolute lifetime error |Δτ| (ns)"
        )

        ax.set_title(
            "Absolute lifetime error distribution"
        )

    else:
        ax.axvline(
            0.0,
            linestyle="--",
        )

        ax.set_xlabel(
            "Signed lifetime error Δτ (ns)"
        )

        ax.set_title(
            "Signed lifetime error distribution"
        )

    ax.set_ylabel(
        "Count"
    )

    ax.legend()

    return ax


def plot_error_vs_condition(
    diagnostics: pd.DataFrame,
    *,
    condition_column: str,
    absolute: bool = False,
    log_x: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot lifetime estimation error against a benchmark condition."""

    error_column = (
        "absolute_error_ns"
        if absolute
        else "error_ns"
    )

    _require_columns(
        diagnostics,
        {
            "estimator_name",
            condition_column,
            error_column,
            "valid_estimate",
        },
    )

    valid_rows = _valid_plot_rows(
        diagnostics,
        numeric_columns=(
            condition_column,
            error_column,
        ),
    )

    ax = _get_axes(
        ax
    )

    for estimator_name, group in (
        valid_rows.groupby(
            "estimator_name",
            sort=False,
        )
    ):
        ax.scatter(
            group[
                condition_column
            ],
            group[
                error_column
            ],
            label=estimator_name,
        )

    if absolute:
        ax.set_ylabel(
            "Absolute lifetime error |Δτ| (ns)"
        )

    else:
        ax.axhline(
            0.0,
            linestyle="--",
        )

        ax.set_ylabel(
            "Signed lifetime error Δτ (ns)"
        )

    x_label = _CONDITION_LABELS.get(
        condition_column,
        condition_column,
    )

    ax.set_xlabel(
        x_label
    )

    if log_x:
        ax.set_xscale(
            "log"
        )

    ax.legend()

    return ax


def plot_paired_absolute_error_comparison(
    diagnostics_x: pd.DataFrame,
    diagnostics_y: pd.DataFrame,
    *,
    sample_id_column: str = "sample_id",
    ax: Axes | None = None,
) -> Axes:
    """Compare absolute errors from two estimators on matched samples."""

    required_columns = {
        "estimator_name",
        sample_id_column,
        "absolute_error_ns",
        "valid_estimate",
    }

    _require_columns(
        diagnostics_x,
        required_columns,
    )

    _require_columns(
        diagnostics_y,
        required_columns,
    )

    estimator_names_x = (
        diagnostics_x[
            "estimator_name"
        ].unique()
    )

    estimator_names_y = (
        diagnostics_y[
            "estimator_name"
        ].unique()
    )

    if estimator_names_x.size != 1:
        raise ValueError(
            "diagnostics_x must contain exactly one estimator."
        )

    if estimator_names_y.size != 1:
        raise ValueError(
            "diagnostics_y must contain exactly one estimator."
        )

    estimator_x = str(
        estimator_names_x[0]
    )

    estimator_y = str(
        estimator_names_y[0]
    )

    valid_x = _valid_plot_rows(
        diagnostics_x,
        numeric_columns=(
            "absolute_error_ns",
        ),
    )

    valid_y = _valid_plot_rows(
        diagnostics_y,
        numeric_columns=(
            "absolute_error_ns",
        ),
    )

    if valid_x[
        sample_id_column
    ].duplicated().any():
        raise ValueError(
            "diagnostics_x contains duplicate sample IDs."
        )

    if valid_y[
        sample_id_column
    ].duplicated().any():
        raise ValueError(
            "diagnostics_y contains duplicate sample IDs."
        )

    paired = valid_x[
        [
            sample_id_column,
            "absolute_error_ns",
        ]
    ].merge(
        valid_y[
            [
                sample_id_column,
                "absolute_error_ns",
            ]
        ],
        on=sample_id_column,
        how="inner",
        suffixes=(
            "_x",
            "_y",
        ),
    )

    if paired.empty:
        raise ValueError(
            "No matched valid samples are available "
            "for estimator comparison."
        )

    ax = _get_axes(
        ax
    )

    x_errors = paired[
        "absolute_error_ns_x"
    ].to_numpy(
        dtype=np.float64
    )

    y_errors = paired[
        "absolute_error_ns_y"
    ].to_numpy(
        dtype=np.float64
    )

    ax.scatter(
        x_errors,
        y_errors,
    )

    upper_limit = float(
        max(
            np.max(
                x_errors
            ),
            np.max(
                y_errors
            ),
        )
    )

    ax.plot(
        [
            0.0,
            upper_limit,
        ],
        [
            0.0,
            upper_limit,
        ],
        linestyle="--",
    )

    ax.set_xlabel(
        f"{estimator_x} |Δτ| (ns)"
    )

    ax.set_ylabel(
        f"{estimator_y} |Δτ| (ns)"
    )

    ax.set_title(
        "Paired absolute-error comparison"
    )

    return ax
