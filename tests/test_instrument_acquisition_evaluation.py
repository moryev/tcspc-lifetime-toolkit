import numpy as np
import pytest

from tcspc_toolkit.config import (
    FeatureConfig,
)
from tcspc_toolkit.generalization import (
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    generate_generalization_test_suite,
)
from tcspc_toolkit.generalization_evaluation import (
    build_generalization_development_measurements,
    evaluate_instrument_acquisition_benchmark,
    fit_generalization_ml_estimators,
    prepare_generalization_data,
)


FEATURE_CONFIG = FeatureConfig(
    tail_start_ns=2.0,
    early_stop_ns=2.0,
    late_start_ns=3.0,
)


@pytest.fixture(scope="module")
def instrument_benchmark():
    definition = (
        default_generalization_suite()
    )

    suite = (
        generate_generalization_test_suite(
            definition=definition
        )
    )

    development = (
        build_generalization_development_measurements(
            definition=definition
        )
    )

    feature_config = FEATURE_CONFIG

    prepared = prepare_generalization_data(
        development_measurements=(
            development
        ),
        tests={
            "A": suite.get_test("A"),
            "C": suite.get_test("C"),
            "D": suite.get_test("D"),
            "E": suite.get_test("E"),
        },
        feature_config=feature_config,
    )

    fitted_estimators = (
        fit_generalization_ml_estimators(
            prepared
        )
    )

    return (
        evaluate_instrument_acquisition_benchmark(
            prepared=prepared,
            fitted_estimators=(
                fitted_estimators
            ),
        )
    )


def test_instrument_benchmark_contains_a_c_d_e(
    instrument_benchmark,
) -> None:
    assert set(
        instrument_benchmark.summary[
            "test_id"
        ]
    ) == {
        "A",
        "C",
        "D",
        "E",
    }


def test_instrument_benchmark_contains_all_ml_representations(
    instrument_benchmark,
) -> None:
    ml_rows = (
        instrument_benchmark.summary.loc[
            instrument_benchmark.summary[
                "estimator"
            ].isin(
                {
                    "ridge",
                    "random_forest",
                    "hist_gradient_boosting",
                }
            )
        ]
    )

    assert set(
        ml_rows["representation"]
    ) == {
        "engineered_features",
        "normalized_histogram",
        "pca_histogram",
    }


def test_instrument_benchmark_has_expected_summary_size(
    instrument_benchmark,
) -> None:
    assert len(
        instrument_benchmark.summary
    ) == 44

    assert len(
        instrument_benchmark.degradation
    ) == 33


def test_instrument_degradation_uses_test_a_reference(
    instrument_benchmark,
) -> None:
    degradation = (
        instrument_benchmark.degradation
    )

    assert set(
        degradation[
            "reference_test_id"
        ]
    ) == {"A"}

    assert set(
        degradation[
            "ood_test_id"
        ]
    ) == {
        "C",
        "D",
        "E",
    }


def test_nonclassical_predictions_are_finite(
    instrument_benchmark,
) -> None:
    assert np.all(
        np.isfinite(
            instrument_benchmark.predictions[
                "predicted_lifetime_ns"
            ]
        )
    )


def test_instrument_tests_preserve_lifetime_targets(
    instrument_benchmark,
) -> None:
    predictions = (
        instrument_benchmark.predictions
    )

    ridge_features = predictions.loc[
        (
            predictions["estimator"]
            == "ridge"
        )
        & (
            predictions["representation"]
            == "engineered_features"
        )
    ]

    targets = {}

    for test_id in (
        "A",
        "C",
        "D",
        "E",
    ):
        rows = ridge_features.loc[
            ridge_features[
                "test_id"
            ] == test_id
        ]

        targets[test_id] = (
            rows[
                "true_lifetime_ns"
            ].to_numpy()
        )

    for test_id in (
        "C",
        "D",
        "E",
    ):
        np.testing.assert_array_equal(
            targets["A"],
            targets[test_id],
        )


