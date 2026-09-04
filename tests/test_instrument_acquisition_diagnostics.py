import numpy as np

from tcspc_toolkit.generalization_evaluation import (
    build_instrument_representation_comparison,
    build_test_c_paired_irf_diagnostics,
    summarize_test_e_temporal_shift_recovery,
)


def test_representation_comparison_has_one_row_per_model_and_ood_test(
    instrument_benchmark,
) -> None:
    comparison = (
        build_instrument_representation_comparison(
            instrument_benchmark.degradation
        )
    )

    assert len(
        comparison
    ) == 9

    assert set(
        comparison[
            "ood_test_id"
        ]
    ) == {
        "C",
        "D",
        "E",
    }


def test_representation_comparison_contains_all_degradation_columns(
    instrument_benchmark,
) -> None:
    comparison = (
        build_instrument_representation_comparison(
            instrument_benchmark.degradation
        )
    )

    expected_columns = {
        "degradation_engineered_features",
        "degradation_normalized_histogram",
        "degradation_pca_histogram",
        "normalized_minus_engineered",
        "pca_minus_engineered",
    }

    assert expected_columns.issubset(
        comparison.columns
    )


def test_test_c_irf_diagnostics_preserve_sample_pairing(
    classical_instrument_result,
) -> None:
    paired = (
        build_test_c_paired_irf_diagnostics(
            classical_instrument_result.fit_diagnostics
        )
    )

    assert paired[
        "sample_id"
    ].is_unique

    assert len(
        paired
    ) > 0


def test_test_c_irf_diagnostics_contain_both_irf_modes(
    classical_instrument_result,
) -> None:
    paired = (
        build_test_c_paired_irf_diagnostics(
            classical_instrument_result.fit_diagnostics
        )
    )

    assert {
        "absolute_error_correct_irf_ns",
        "absolute_error_nominal_irf_ns",
        "absolute_error_penalty_ns",
        "fitted_shift_change_ns",
    }.issubset(
        paired.columns
    )


def test_test_e_shift_recovery_contains_both_misalignments(
    classical_instrument_result,
) -> None:
    summary = (
        summarize_test_e_temporal_shift_recovery(
            classical_instrument_result.fit_diagnostics
        )
    )

    np.testing.assert_allclose(
        np.sort(
            summary[
                "true_shift_ns"
            ].to_numpy()
        ),
        np.asarray(
            [
                -0.15,
                0.15,
            ]
        ),
    )


def test_test_e_shift_recovery_reports_fit_statistics(
    classical_instrument_result,
) -> None:
    summary = (
        summarize_test_e_temporal_shift_recovery(
            classical_instrument_result.fit_diagnostics
        )
    )

    expected_columns = {
        "failure_rate",
        "boundary_hit_rate",
        "mean_fitted_shift_ns",
        "shift_bias_ns",
        "shift_mae_ns",
        "shift_rmse_ns",
        "lifetime_mae_ns",
    }

    assert expected_columns.issubset(
        summary.columns
    )


