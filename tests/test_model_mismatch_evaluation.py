import numpy as np

from tcspc_toolkit.generalization_evaluation import (
    build_test_f_reference_diagnostics,
    summarize_test_f_model_mismatch,
)


def test_model_mismatch_benchmark_contains_tests_a_and_f(
    model_mismatch_benchmark,
) -> None:
    assert set(
        model_mismatch_benchmark.predictions[
            "test_id"
        ]
    ) == {
        "A",
        "F",
    }


def test_model_mismatch_summary_contains_all_estimators(
    model_mismatch_benchmark,
) -> None:
    summary = (
        model_mismatch_benchmark.summary
    )

    # Per test:
    # 1 constant baseline
    # 1 mean-arrival baseline
    # 3 ML models x 3 representations
    # = 11 estimator/representation combinations.
    #
    # Tests A and F therefore produce 22 rows.
    assert len(
        summary
    ) == 22

    assert set(
        summary[
            "test_id"
        ]
    ) == {
        "A",
        "F",
    }


def test_model_mismatch_degradation_uses_test_a_as_reference(
    model_mismatch_benchmark,
) -> None:
    degradation = (
        model_mismatch_benchmark.degradation
    )

    assert len(
        degradation
    ) == 11

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


def test_test_f_reference_diagnostics_contains_both_severities(
    model_mismatch_benchmark,
) -> None:
    diagnostics = (
        build_test_f_reference_diagnostics(
            model_mismatch_benchmark.predictions
        )
    )

    assert set(
        diagnostics[
            "model_mismatch_severity"
        ]
    ) == {
        "weak",
        "moderate",
    }

    np.testing.assert_allclose(
        np.sort(
            diagnostics[
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


def test_test_f_primary_reference_matches_standard_error(
    model_mismatch_benchmark,
) -> None:
    diagnostics = (
        model_mismatch_benchmark
        .test_f_reference_diagnostics
    )

    valid = diagnostics[
        "valid_prediction"
    ].to_numpy(
        dtype=bool
    )

    np.testing.assert_allclose(
        diagnostics.loc[
            valid,
            "error_to_primary_ns",
        ].to_numpy(),
        diagnostics.loc[
            valid,
            "error_ns",
        ].to_numpy(),
    )


def test_test_f_weighted_reference_lies_between_components(
    model_mismatch_benchmark,
) -> None:
    diagnostics = (
        model_mismatch_benchmark
        .test_f_reference_diagnostics
    )

    primary = diagnostics[
        "primary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    secondary = diagnostics[
        "secondary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    weighted = diagnostics[
        "signal_photon_weighted_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    assert np.all(
        weighted > primary
    )

    assert np.all(
        weighted < secondary
    )


def test_test_f_effective_mixture_position_is_consistent(
    model_mismatch_benchmark,
) -> None:
    diagnostics = (
        model_mismatch_benchmark
        .test_f_reference_diagnostics
    )

    valid = diagnostics[
        "valid_prediction"
    ].to_numpy(
        dtype=bool
    )

    expected = (
        (
            diagnostics.loc[
                valid,
                "predicted_lifetime_ns",
            ].to_numpy()
            - diagnostics.loc[
                valid,
                "primary_lifetime_ns",
            ].to_numpy()
        )
        / (
            diagnostics.loc[
                valid,
                "secondary_lifetime_ns",
            ].to_numpy()
            - diagnostics.loc[
                valid,
                "primary_lifetime_ns",
            ].to_numpy()
        )
    )

    np.testing.assert_allclose(
        diagnostics.loc[
            valid,
            "effective_mixture_position",
        ].to_numpy(),
        expected,
    )


def test_test_f_severity_summary_has_one_row_per_estimator_and_severity(
    model_mismatch_benchmark,
) -> None:
    summary = (
        summarize_test_f_model_mismatch(
            model_mismatch_benchmark.predictions
        )
    )

    # 11 estimator/representation combinations
    # x 2 mismatch severities.
    assert len(
        summary
    ) == 22

    assert set(
        summary[
            "model_mismatch_severity"
        ]
    ) == {
        "weak",
        "moderate",
    }


def test_test_f_severity_summary_reports_both_references(
    model_mismatch_benchmark,
) -> None:
    summary = (
        model_mismatch_benchmark
        .severity_summary
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
        summary.columns
    )
