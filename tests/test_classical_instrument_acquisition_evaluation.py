import numpy as np
import pytest

from tcspc_toolkit.generalization import (
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    GeneralizationTestMeasurements,
    generate_generalization_test_suite,
)
from tcspc_toolkit.generalization_evaluation import (
    evaluate_classical_instrument_acquisition_benchmark,
)


def _subset_test(
    test: GeneralizationTestMeasurements,
    indices: np.ndarray,
) -> GeneralizationTestMeasurements:
    return GeneralizationTestMeasurements(
        test_id=test.test_id,
        time=test.time.copy(),
        X_histograms=(
            test.X_histograms[
                indices
            ].copy()
        ),
        y=(
            test.y[
                indices
            ].copy()
        ),
        metadata=(
            test.metadata.iloc[
                indices
            ]
            .reset_index(
                drop=True
            )
        ),
    )


@pytest.fixture(scope="module")
def classical_instrument_result():
    definition = (
        default_generalization_suite()
    )

    suite = (
        generate_generalization_test_suite(
            definition=definition
        )
    )

    # Covers both familiar IRF widths and both
    # temporal-shift directions while remaining small.
    indices = np.asarray(
        [
            0,
            1,
            6,
            7,
            12,
            13,
            18,
            19,
        ],
        dtype=np.int64,
    )

    tests = {
        test_id: _subset_test(
            suite.get_test(
                test_id
            ),
            indices,
        )
        for test_id in (
            "A",
            "C",
            "D",
            "E",
        )
    }

    return (
        evaluate_classical_instrument_acquisition_benchmark(
            tests=tests,
            irf_centre_ns=(
                definition.familiar.irf_centre_ns
            ),
            nominal_irf_fwhm_ns=0.40,
        )
    )


def test_classical_correct_irf_covers_a_c_d_e(
    classical_instrument_result,
) -> None:
    summary = (
        classical_instrument_result.summary
    )

    correct = summary.loc[
        summary["estimator"]
        == (
            "classical_reconvolution_correct_irf"
        )
    ]

    assert set(
        correct["test_id"]
    ) == {
        "A",
        "C",
        "D",
        "E",
    }


def test_nominal_irf_experiment_is_test_c_only(
    classical_instrument_result,
) -> None:
    summary = (
        classical_instrument_result.summary
    )

    nominal = summary.loc[
        summary["estimator"]
        == (
            "classical_reconvolution_nominal_irf"
        )
    ]

    assert set(
        nominal["test_id"]
    ) == {"C"}


def test_correct_irf_matches_true_width(
    classical_instrument_result,
) -> None:
    diagnostics = (
        classical_instrument_result.fit_diagnostics
    )

    correct = diagnostics.loc[
        diagnostics[
            "classical_irf_mode"
        ] == "correct_test_irf"
    ]

    np.testing.assert_allclose(
        correct[
            "irf_fwhm_error_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        0.0,
        atol=1e-12,
    )


def test_test_c_nominal_irf_is_deliberately_mismatched(
    classical_instrument_result,
) -> None:
    diagnostics = (
        classical_instrument_result.fit_diagnostics
    )

    nominal = diagnostics.loc[
        diagnostics[
            "classical_irf_mode"
        ] == "nominal_familiar_irf"
    ]

    np.testing.assert_allclose(
        nominal[
            "assumed_irf_fwhm_ns"
        ],
        0.40,
    )

    np.testing.assert_allclose(
        nominal[
            "irf_fwhm_ns"
        ],
        0.60,
    )

    np.testing.assert_allclose(
        nominal[
            "irf_fwhm_error_ns"
        ],
        -0.20,
    )


def test_temporal_misalignment_retains_shift_diagnostics(
    classical_instrument_result,
) -> None:
    diagnostics = (
        classical_instrument_result.fit_diagnostics
    )

    test_e = diagnostics.loc[
        (
            diagnostics["test_id"]
            == "E"
        )
        & (
            diagnostics[
                "classical_irf_mode"
            ]
            == "correct_test_irf"
        )
    ]

    assert set(
        np.round(
            test_e[
                "irf_shift_ns"
            ].to_numpy(
                dtype=np.float64
            ),
            decimals=2,
        )
    ) == {
        -0.15,
        0.15,
    }

    assert (
        "fitted_temporal_shift_ns"
        in test_e.columns
    )

    assert (
        "temporal_shift_error_ns"
        in test_e.columns
    )


def test_classical_failure_rates_are_reported(
    classical_instrument_result,
) -> None:
    assert np.all(
        np.isfinite(
            classical_instrument_result.summary[
                "classical_failure_rate"
            ]
        )
    )


def test_classical_degradation_uses_same_test_a_reference(
    classical_instrument_result,
) -> None:
    degradation = (
        classical_instrument_result.degradation
    )

    assert set(
        degradation[
            "reference_test_id"
        ]
    ) == {"A"}

    assert set(
        degradation[
            "reference_estimator"
        ]
    ) == {
        "classical_reconvolution_correct_irf"
    }

    assert len(
        degradation
    ) == 4


