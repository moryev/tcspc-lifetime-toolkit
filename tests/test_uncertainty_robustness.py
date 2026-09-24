import numpy as np
import pandas as pd

from tcspc_toolkit.uncertainty_robustness import (
    _finite_sample_conformal_quantile,
    _summarize_signed_residual_profile,
    build_week9_classical_uncertainty_scorecard,
)


def test_finite_sample_conformal_quantile_uses_order_statistic() -> None:
    scores = np.asarray(
        [
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.50,
        ],
        dtype=np.float64,
    )

    correction = (
        _finite_sample_conformal_quantile(
            scores,
            nominal_coverage=0.90,
        )
    )

    # n = 10:
    #
    # ceil((10 + 1) * 0.90)
    # = ceil(9.9)
    # = 10
    #
    # therefore the largest calibration score is selected.
    assert correction == 0.50


def test_conformal_quantile_rejects_negative_scores() -> None:
    scores = np.asarray(
        [
            0.0,
            -0.1,
            0.2,
        ],
        dtype=np.float64,
    )

    try:
        _finite_sample_conformal_quantile(
            scores,
            nominal_coverage=0.90,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Negative conformal scores must be rejected."
        )


def test_conformal_quantile_rejects_invalid_coverage() -> None:
    scores = np.asarray(
        [
            0.1,
            0.2,
            0.3,
        ],
        dtype=np.float64,
    )

    for coverage in (
        0.0,
        1.0,
        -0.1,
        1.1,
    ):
        try:
            _finite_sample_conformal_quantile(
                scores,
                nominal_coverage=coverage,
            )

        except ValueError:
            pass

        else:
            raise AssertionError(
                "Invalid nominal coverage must be rejected."
            )


def _make_classical_uncertainty_test_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "test_id": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
                "D",
                "D",
                "E",
                "E",
                "F",
                "F",
            ],

            "true_lifetime_ns": [
                1.0,
                2.0,
            ] * 6,

            "fitted_lifetime_ns": [
                1.05,
                1.95,
                1.10,
                1.90,
                1.04,
                1.96,
                1.15,
                1.85,
                1.06,
                1.94,
                1.30,
                2.30,
            ],

            "covariance_std_ns": [
                0.10,
                0.10,
            ] * 6,

            "covariance_lower_ns": [
                0.80,
                1.80,
            ] * 6,

            "covariance_upper_ns": [
                1.20,
                2.20,
            ] * 6,

            "bootstrap_std_ns": [
                0.12,
                0.12,
            ] * 6,

            "bootstrap_lower_ns": [
                0.75,
                1.75,
            ] * 6,

            "bootstrap_upper_ns": [
                1.25,
                2.25,
            ] * 6,

            "bootstrap_fit_failure_rate": [
                0.0,
                0.0,
            ] * 6,
        }
    )


def test_classical_uncertainty_scorecard_contains_both_methods_for_a_to_f(
) -> None:
    per_curve = (
        _make_classical_uncertainty_test_table()
    )

    scorecard = (
        build_week9_classical_uncertainty_scorecard(
            per_curve,
            nominal_coverage=0.90,
        )
    )

    assert scorecard.shape[0] == 12

    assert set(
        scorecard[
            "test_id"
        ]
    ) == {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    }

    assert set(
        scorecard[
            "method"
        ]
    ) == {
        "covariance",
        "parametric_bootstrap",
    }


def test_classical_uncertainty_scorecard_marks_test_f_primary_target(
) -> None:
    per_curve = (
        _make_classical_uncertainty_test_table()
    )

    scorecard = (
        build_week9_classical_uncertainty_scorecard(
            per_curve,
            nominal_coverage=0.90,
        )
    )

    test_f = scorecard.loc[
        scorecard[
            "test_id"
        ] == "F"
    ]

    assert set(
        test_f[
            "target_reference"
        ]
    ) == {
        "dominant_component_tau_1"
    }

    non_f = scorecard.loc[
        scorecard[
            "test_id"
        ] != "F"
    ]

    assert set(
        non_f[
            "target_reference"
        ]
    ) == {
        "monoexponential_lifetime"
    }


def test_classical_uncertainty_scorecard_exposes_error_to_uncertainty_ratio(
) -> None:
    per_curve = (
        _make_classical_uncertainty_test_table()
    )

    scorecard = (
        build_week9_classical_uncertainty_scorecard(
            per_curve,
            nominal_coverage=0.90,
        )
    )

    covariance_a = (
        scorecard.loc[
            (
                scorecard[
                    "test_id"
                ] == "A"
            )
            & (
                scorecard[
                    "method"
                ] == "covariance"
            )
        ]
        .iloc[0]
    )

    covariance_f = (
        scorecard.loc[
            (
                scorecard[
                    "test_id"
                ] == "F"
            )
            & (
                scorecard[
                    "method"
                ] == "covariance"
            )
        ]
        .iloc[0]
    )

    assert (
        covariance_f[
            "mean_abs_error_over_std"
        ]
        >
        covariance_a[
            "mean_abs_error_over_std"
        ]
    )


def test_signed_residual_profile_is_zero_for_balanced_residuals(
) -> None:
    residuals = np.asarray(
        [
            [
                1.0,
                -2.0,
                0.5,
            ],
            [
                -1.0,
                2.0,
                -0.5,
            ],
        ],
        dtype=np.float64,
    )

    (
        mean_profile,
        profile_rms,
        mean_absolute_profile,
        max_absolute_profile,
    ) = (
        _summarize_signed_residual_profile(
            residuals
        )
    )

    np.testing.assert_allclose(
        mean_profile,
        np.zeros(
            3,
            dtype=np.float64,
        ),
    )

    assert profile_rms == 0.0

    assert mean_absolute_profile == 0.0

    assert max_absolute_profile == 0.0


def test_signed_residual_profile_retains_systematic_structure(
) -> None:
    residuals = np.asarray(
        [
            [
                0.5,
                -1.0,
                2.0,
            ],
            [
                0.5,
                -1.0,
                2.0,
            ],
            [
                0.5,
                -1.0,
                2.0,
            ],
        ],
        dtype=np.float64,
    )

    (
        mean_profile,
        profile_rms,
        mean_absolute_profile,
        max_absolute_profile,
    ) = (
        _summarize_signed_residual_profile(
            residuals
        )
    )

    expected_profile = np.asarray(
        [
            0.5,
            -1.0,
            2.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        mean_profile,
        expected_profile,
    )

    expected_rms = float(
        np.sqrt(
            np.mean(
                expected_profile**2
            )
        )
    )

    assert np.isclose(
        profile_rms,
        expected_rms,
    )

    assert np.isclose(
        mean_absolute_profile,
        (
            0.5
            + 1.0
            + 2.0
        )
        / 3.0,
    )

    assert (
        max_absolute_profile
        == 2.0
    )


def test_signed_residual_profile_requires_finite_curve(
) -> None:
    residuals = np.asarray(
        [
            [
                np.nan,
                np.nan,
            ],
            [
                np.nan,
                np.nan,
            ],
        ],
        dtype=np.float64,
    )

    try:
        _summarize_signed_residual_profile(
            residuals
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "At least one finite residual curve must be required."
        )


