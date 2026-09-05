import numpy as np

from tcspc_toolkit.generalization_evaluation import (
    build_classical_af_paired_diagnostics,
    summarize_classical_test_f_model_mismatch,
)


def test_classical_model_mismatch_contains_tests_a_and_f(
    classical_model_mismatch_result,
) -> None:
    assert set(
        classical_model_mismatch_result.predictions[
            "test_id"
        ]
    ) == {
        "A",
        "F",
    }

    assert len(
        classical_model_mismatch_result.summary
    ) == 2


def test_classical_model_mismatch_uses_correct_irf_width(
    classical_model_mismatch_result,
) -> None:
    diagnostics = (
        classical_model_mismatch_result
        .fit_diagnostics
    )

    assert set(
        diagnostics[
            "classical_irf_mode"
        ]
    ) == {
        "correct_test_irf",
    }

    np.testing.assert_allclose(
        diagnostics[
            "assumed_irf_fwhm_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        diagnostics[
            "irf_fwhm_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    )


def test_classical_model_mismatch_reports_goodness_of_fit(
    classical_model_mismatch_result,
) -> None:
    diagnostics = (
        classical_model_mismatch_result
        .fit_diagnostics
    )

    expected_columns = {
        "valid_fit",
        "boundary_hit",
        "poisson_nll",
        "poisson_deviance",
        "poisson_deviance_per_bin",
    }

    assert expected_columns.issubset(
        diagnostics.columns
    )


def test_classical_poisson_deviance_per_bin_is_consistent(
    classical_model_mismatch_result,
) -> None:
    diagnostics = (
        classical_model_mismatch_result
        .fit_diagnostics
    )

    finite = np.isfinite(
        diagnostics[
            "poisson_deviance"
        ].to_numpy(
            dtype=np.float64
        )
    )

    # Canonical Week-8 geometry contains 400 bins.
    np.testing.assert_allclose(
        diagnostics.loc[
            finite,
            "poisson_deviance_per_bin",
        ].to_numpy(),
        diagnostics.loc[
            finite,
            "poisson_deviance",
        ].to_numpy()
        / 400.0,
    )


def test_classical_test_f_contains_both_mismatch_severities(
    classical_model_mismatch_result,
) -> None:
    diagnostics = (
        classical_model_mismatch_result
        .fit_diagnostics
    )

    test_f = diagnostics.loc[
        diagnostics[
            "test_id"
        ] == "F"
    ]

    assert set(
        test_f[
            "model_mismatch_severity"
        ]
    ) == {
        "weak",
        "moderate",
    }


def test_classical_af_paired_diagnostics_preserve_pairing(
    classical_model_mismatch_result,
) -> None:
    paired = (
        build_classical_af_paired_diagnostics(
            classical_model_mismatch_result
            .fit_diagnostics
        )
    )

    assert len(
        paired
    ) == 8

    assert paired[
        "sample_id"
    ].is_unique

    assert paired[
        "pair_id"
    ].is_unique


def test_classical_af_paired_diagnostics_preserve_primary_target(
    classical_model_mismatch_result,
) -> None:
    paired = (
        classical_model_mismatch_result
        .paired_diagnostics
    )

    np.testing.assert_allclose(
        paired[
            "true_lifetime_a_ns"
        ].to_numpy(),
        paired[
            "primary_lifetime_ns"
        ].to_numpy(),
    )


def test_classical_af_paired_diagnostics_report_fit_and_gof_changes(
    classical_model_mismatch_result,
) -> None:
    paired = (
        classical_model_mismatch_result
        .paired_diagnostics
    )

    expected_columns = {
        "paired_valid_fit",
        "absolute_error_to_primary_a_ns",
        "absolute_error_to_primary_f_ns",
        "absolute_error_to_signal_photon_weighted_f_ns",
        "effective_mixture_position_f",
        "fitted_lifetime_change_ns",
        "poisson_nll_change",
        "poisson_deviance_change",
        "poisson_deviance_per_bin_change",
    }

    assert expected_columns.issubset(
        paired.columns
    )


def test_classical_model_mismatch_degradation_uses_a_as_reference(
    classical_model_mismatch_result,
) -> None:
    degradation = (
        classical_model_mismatch_result
        .degradation
    )

    assert len(
        degradation
    ) == 1

    assert set(
        degradation[
            "reference_test_id"
        ]
    ) == {
        "A",
    }

    assert set(
        degradation[
            "ood_test_id"
        ]
    ) == {
        "F",
    }


def test_classical_test_f_severity_summary_has_two_rows(
    classical_model_mismatch_result,
) -> None:
    summary = (
        summarize_classical_test_f_model_mismatch(
            classical_model_mismatch_result
            .fit_diagnostics
        )
    )

    assert len(
        summary
    ) == 2

    assert set(
        summary[
            "model_mismatch_severity"
        ]
    ) == {
        "weak",
        "moderate",
    }


def test_classical_test_f_severity_summary_reports_gof(
    classical_model_mismatch_result,
) -> None:
    summary = (
        classical_model_mismatch_result
        .severity_summary
    )

    expected_columns = {
        "failure_rate",
        "boundary_hit_rate",
        "mae_to_primary_ns",
        "mae_to_signal_photon_weighted_ns",
        "mean_effective_mixture_position",
        "mean_poisson_nll",
        "mean_poisson_deviance",
        "mean_poisson_deviance_per_bin",
    }

    assert expected_columns.issubset(
        summary.columns
    )
