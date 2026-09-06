import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.generalization_evaluation import (
    ClassicalGeneralizationSuiteBenchmarkResult,
    GeneralizationSuiteBenchmarkResult,
    build_reference_mae_degradation_table,
    build_week8_robustness_report,
)


FINAL_TEST_IDS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
)


def _make_summary_rows(
    *,
    estimator: str,
    representation: str,
    mae_values: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    failure_rates: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ] | None = None,
) -> pd.DataFrame:
    """Build one synthetic A-F robustness summary."""

    if failure_rates is None:
        failure_rates = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    rows = []

    for (
        test_id,
        mae_ns,
        failure_rate,
    ) in zip(
        FINAL_TEST_IDS,
        mae_values,
        failure_rates,
        strict=True,
    ):
        rows.append(
            {
                "estimator": estimator,
                "representation": representation,
                "test_id": test_id,
                "n_total_samples": 100,
                "n_valid_predictions": (
                    100
                    if np.isnan(
                        failure_rate
                    )
                    else round(
                        100
                        * (
                            1.0
                            - failure_rate
                        )
                    )
                ),
                "mae_ns": mae_ns,
                "median_absolute_error_ns": (
                    0.8
                    * mae_ns
                ),
                "rmse_ns": (
                    1.2
                    * mae_ns
                ),
                "bias_ns": (
                    0.1
                    * mae_ns
                ),
                "p90_absolute_error_ns": (
                    1.5
                    * mae_ns
                ),
                "p95_absolute_error_ns": (
                    1.8
                    * mae_ns
                ),
                "classical_failure_rate": (
                    failure_rate
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


@pytest.fixture
def synthetic_nonclassical_result(
) -> GeneralizationSuiteBenchmarkResult:
    """Return complete synthetic A-F non-classical results."""

    summary = pd.concat(
        [
            _make_summary_rows(
                estimator="constant_mean",
                representation="none",
                mae_values=(
                    1.00,
                    1.10,
                    1.20,
                    1.30,
                    1.40,
                    1.50,
                ),
            ),
            _make_summary_rows(
                estimator="mean_arrival_time",
                representation=(
                    "engineered_features"
                ),
                mae_values=(
                    0.50,
                    1.00,
                    0.75,
                    0.625,
                    0.55,
                    1.00,
                ),
            ),
            _make_summary_rows(
                estimator="ridge",
                representation=(
                    "engineered_features"
                ),
                mae_values=(
                    0.20,
                    0.40,
                    0.30,
                    0.50,
                    0.25,
                    0.60,
                ),
            ),
            _make_summary_rows(
                estimator="random_forest",
                representation=(
                    "engineered_features"
                ),
                mae_values=(
                    0.10,
                    0.15,
                    0.20,
                    0.12,
                    0.30,
                    0.25,
                ),
            ),
            _make_summary_rows(
                estimator=(
                    "hist_gradient_boosting"
                ),
                representation=(
                    "engineered_features"
                ),
                mae_values=(
                    0.25,
                    0.50,
                    0.25,
                    0.75,
                    0.125,
                    0.375,
                ),
            ),
        ],
        ignore_index=True,
    )

    degradation = (
        build_reference_mae_degradation_table(
            summary,
            reference_test_id="A",
        )
    )

    return GeneralizationSuiteBenchmarkResult(
        predictions=pd.DataFrame(),
        summary=summary,
        degradation=degradation,
    )


@pytest.fixture
def synthetic_classical_result(
) -> ClassicalGeneralizationSuiteBenchmarkResult:
    """Return complete synthetic A-F classical results."""

    summary = _make_summary_rows(
        estimator=(
            "classical_reconvolution_mono_model"
        ),
        representation="raw_histogram",
        mae_values=(
            0.05,
            0.10,
            0.075,
            0.05,
            0.20,
            0.15,
        ),
        failure_rates=(
            0.00,
            0.01,
            0.00,
            0.03,
            0.02,
            0.05,
        ),
    )

    degradation = (
        build_reference_mae_degradation_table(
            summary,
            reference_test_id="A",
        )
    )

    return (
        ClassicalGeneralizationSuiteBenchmarkResult(
            predictions=pd.DataFrame(),
            summary=summary,
            degradation=degradation,
            fit_diagnostics=pd.DataFrame(),
        )
    )


@pytest.fixture
def week8_report(
    synthetic_nonclassical_result,
    synthetic_classical_result,
):
    """Build the synthetic final Week-8 report."""

    return build_week8_robustness_report(
        nonclassical_result=(
            synthetic_nonclassical_result
        ),
        classical_result=(
            synthetic_classical_result
        ),
    )


def test_week8_mae_matrix_contains_tests_a_to_f(
    week8_report,
) -> None:
    """The final MAE matrix must contain all six frozen tests."""

    assert tuple(
        week8_report.mae_matrix.columns
    ) == FINAL_TEST_IDS

    assert week8_report.mae_matrix.shape == (
        6,
        6,
    )


def test_week8_mae_matrix_contains_principal_estimators(
    week8_report,
) -> None:
    """The final matrix must contain the principal estimator set."""

    expected_estimators = [
        "constant_mean",
        "mean_arrival_time",
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
        "classical_reconvolution",
    ]

    assert list(
        week8_report.mae_matrix.index
    ) == expected_estimators

    assert (
        week8_report.mae_matrix.loc[
            "ridge",
            "A",
        ]
        == pytest.approx(
            0.20
        )
    )

    assert (
        week8_report.mae_matrix.loc[
            "random_forest",
            "F",
        ]
        == pytest.approx(
            0.25
        )
    )

    assert (
        week8_report.mae_matrix.loc[
            "classical_reconvolution",
            "C",
        ]
        == pytest.approx(
            0.075
        )
    )


def test_week8_degradation_uses_test_a_as_reference(
    week8_report,
) -> None:
    """Every OOD degradation value must be relative to Test A."""

    degradation = (
        week8_report.degradation_matrix
    )

    assert list(
        degradation.columns
    ) == [
        "B_over_A",
        "C_over_A",
        "D_over_A",
        "E_over_A",
        "F_over_A",
    ]

    # Ridge:
    #
    # A = 0.20 ns
    # B = 0.40 ns
    # C = 0.30 ns
    # D = 0.50 ns
    # E = 0.25 ns
    # F = 0.60 ns
    assert (
        degradation.loc[
            "ridge",
            "B_over_A",
        ]
        == pytest.approx(
            2.0
        )
    )

    assert (
        degradation.loc[
            "ridge",
            "C_over_A",
        ]
        == pytest.approx(
            1.5
        )
    )

    assert (
        degradation.loc[
            "ridge",
            "D_over_A",
        ]
        == pytest.approx(
            2.5
        )
    )

    assert (
        degradation.loc[
            "ridge",
            "E_over_A",
        ]
        == pytest.approx(
            1.25
        )
    )

    assert (
        degradation.loc[
            "ridge",
            "F_over_A",
        ]
        == pytest.approx(
            3.0
        )
    )

    # Classical reconvolution:
    #
    # A = 0.05 ns
    # F = 0.15 ns
    assert (
        degradation.loc[
            "classical_reconvolution",
            "F_over_A",
        ]
        == pytest.approx(
            3.0
        )
    )


def test_week8_test_f_is_labelled_with_primary_lifetime_reference(
    week8_report,
) -> None:
    """Test F must explicitly use tau_1 as the scoring reference."""

    summary = (
        week8_report.principal_summary
    )

    test_f = summary.loc[
        summary[
            "test_id"
        ] == "F"
    ]

    assert not test_f.empty

    assert set(
        test_f[
            "target_reference"
        ]
    ) == {
        "dominant_component_tau_1"
    }

    familiar_and_other_ood = summary.loc[
        summary[
            "test_id"
        ] != "F"
    ]

    assert set(
        familiar_and_other_ood[
            "target_reference"
        ]
    ) == {
        "monoexponential_lifetime"
    }


def test_week8_classical_failures_are_preserved(
    week8_report,
) -> None:
    """Classical fit failures must survive final aggregation."""

    failure_matrix = (
        week8_report
        .classical_failure_rate_matrix
    )

    assert list(
        failure_matrix.index
    ) == [
        "classical_reconvolution"
    ]

    assert tuple(
        failure_matrix.columns
    ) == FINAL_TEST_IDS

    assert (
        failure_matrix.loc[
            "classical_reconvolution",
            "A",
        ]
        == pytest.approx(
            0.00
        )
    )

    assert (
        failure_matrix.loc[
            "classical_reconvolution",
            "B",
        ]
        == pytest.approx(
            0.01
        )
    )

    assert (
        failure_matrix.loc[
            "classical_reconvolution",
            "D",
        ]
        == pytest.approx(
            0.03
        )
    )

    assert (
        failure_matrix.loc[
            "classical_reconvolution",
            "F",
        ]
        == pytest.approx(
            0.05
        )
    )


def test_near_zero_reference_mae_does_not_produce_huge_degradation(
) -> None:
    """Numerically negligible reference MAE must yield NaN."""

    summary = pd.DataFrame(
        {
            "estimator": [
                "ridge",
                "ridge",
            ],
            "representation": [
                "engineered_features",
                "engineered_features",
            ],
            "test_id": [
                "A",
                "B",
            ],
            "mae_ns": [
                1e-15,
                0.20,
            ],
        }
    )

    degradation = (
        build_reference_mae_degradation_table(
            summary,
            reference_test_id="A",
        )
    )

    assert degradation.shape == (
        1,
        7,
    )

    assert (
        degradation[
            "reference_test_id"
        ].iloc[0]
        == "A"
    )

    assert (
        degradation[
            "ood_test_id"
        ].iloc[0]
        == "B"
    )

    assert (
        degradation[
            "reference_mae_ns"
        ].iloc[0]
        == pytest.approx(
            1e-15
        )
    )

    assert (
        degradation[
            "ood_mae_ns"
        ].iloc[0]
        == pytest.approx(
            0.20
        )
    )

    assert np.isnan(
        degradation[
            "mae_degradation"
        ].iloc[0]
    )
