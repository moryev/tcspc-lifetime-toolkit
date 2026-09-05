import numpy as np


def test_day55_nonclassical_comparison_has_all_estimators(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .nonclassical_comparison
    )

    # Constant + mean-arrival +
    # 3 ML models x 3 representations.
    assert len(
        comparison
    ) == 11


def test_day55_nonclassical_comparison_contains_a_f_metrics(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .nonclassical_comparison
    )

    expected_columns = {
        "mae_a",
        "mae_f",
        "bias_a",
        "bias_f",
        "p95_absolute_error_a",
        "p95_absolute_error_f",
        "mae_degradation",
    }

    assert expected_columns.issubset(
        comparison.columns
    )


def test_day55_classical_comparison_has_one_estimator(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .classical_comparison
    )

    assert len(
        comparison
    ) == 1

    assert set(
        comparison[
            "estimator"
        ]
    ) == {
        "classical_reconvolution_mono_model",
    }


def test_day55_severity_comparison_contains_all_estimators(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .severity_comparison
    )

    # 11 non-classical estimator/representation
    # combinations + 1 classical estimator,
    # each evaluated at two severities.
    assert len(
        comparison
    ) == 24


def test_day55_severity_comparison_contains_two_severities(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .severity_comparison
    )

    assert set(
        comparison[
            "model_mismatch_severity"
        ]
    ) == {
        "weak",
        "moderate",
    }

    np.testing.assert_allclose(
        np.sort(
            comparison[
                "secondary_fraction"
            ].unique()
        ),
        np.asarray(
            [
                0.05,
                0.15,
            ]
        ),
    )


def test_day55_severity_comparison_uses_common_valid_count(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .severity_comparison
    )

    expected_columns = {
        "n_total_samples",
        "n_valid_estimates",
        "failure_rate",
    }

    assert expected_columns.issubset(
        comparison.columns
    )

    assert np.all(
        comparison[
            "n_valid_estimates"
        ]
        <= comparison[
            "n_total_samples"
        ]
    )


def test_day55_severity_comparison_reports_both_lifetime_references(
    day55_model_mismatch_report,
) -> None:
    comparison = (
        day55_model_mismatch_report
        .severity_comparison
    )

    expected_columns = {
        "mae_to_primary_ns",
        "bias_to_primary_ns",
        "mae_to_signal_photon_weighted_ns",
        "bias_to_signal_photon_weighted_ns",
        "mean_effective_mixture_position",
        "median_effective_mixture_position",
    }

    assert expected_columns.issubset(
        comparison.columns
    )


def test_day55_classical_gof_is_kept_separate(
    day55_model_mismatch_report,
) -> None:
    gof = (
        day55_model_mismatch_report
        .classical_gof_summary
    )

    assert len(
        gof
    ) == 2

    expected_columns = {
        "mean_poisson_nll",
        "median_poisson_nll",
        "mean_poisson_deviance",
        "median_poisson_deviance",
        "mean_poisson_deviance_per_bin",
        "median_poisson_deviance_per_bin",
    }

    assert expected_columns.issubset(
        gof.columns
    )


def test_day55_classical_paired_diagnostics_are_preserved(
    day55_model_mismatch_report,
    classical_model_mismatch_result,
) -> None:
    report_paired = (
        day55_model_mismatch_report
        .classical_paired_diagnostics
    )

    original_paired = (
        classical_model_mismatch_result
        .paired_diagnostics
    )

    assert len(
        report_paired
    ) == len(
        original_paired
    )

    np.testing.assert_array_equal(
        report_paired[
            "pair_id"
        ].to_numpy(),
        original_paired[
            "pair_id"
        ].to_numpy(),
    )
