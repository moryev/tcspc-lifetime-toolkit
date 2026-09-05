import numpy as np
import pytest
from numpy.typing import NDArray

from tcspc_toolkit.config import (
    FeatureConfig,
)
from tcspc_toolkit.generalization import (
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    GeneralizationTestMeasurements,
    generate_generalization_test_suite,
)
from tcspc_toolkit.generalization_evaluation import (
    build_generalization_development_measurements,
    evaluate_classical_instrument_acquisition_benchmark,
    evaluate_instrument_acquisition_benchmark,
    evaluate_model_mismatch_benchmark,
    fit_generalization_ml_estimators,
    prepare_generalization_data,
    evaluate_classical_model_mismatch_benchmark,
    build_day55_model_mismatch_report,
)
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
)


FEATURE_CONFIG = FeatureConfig(
    tail_start_ns=2.0,
    early_stop_ns=2.0,
    late_start_ns=3.0,
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


@pytest.fixture
def time_axis() -> NDArray[np.float64]:
    return np.linspace(0.0, 10.0, 1001, dtype=np.float64)


@pytest.fixture
def irf(
    time_axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    irf = generate_gaussian_irf(
        time=time_axis,
        centre=1.0,
        fwhm=0.2,
        amplitude=1.0,
    )

    irf = normalize_irf(
        time=time_axis,
        irf=irf,
    )

    return irf


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


@pytest.fixture(scope="module")
def model_mismatch_benchmark():
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

    prepared = prepare_generalization_data(
        development_measurements=(
            development
        ),
        tests={
            "A": suite.get_test("A"),
            "F": suite.get_test("F"),
        },
        feature_config=FEATURE_CONFIG,
    )

    fitted_estimators = (
        fit_generalization_ml_estimators(
            prepared
        )
    )

    return (
        evaluate_model_mismatch_benchmark(
            prepared=prepared,
            fitted_estimators=(
                fitted_estimators
            ),
        )
    )


@pytest.fixture(scope="module")
def classical_model_mismatch_result():
    definition = (
        default_generalization_suite()
    )

    suite = (
        generate_generalization_test_suite(
            definition=definition
        )
    )

    # Small but deliberately structured subset:
    #
    # - both weak and moderate Test-F contamination;
    # - both familiar IRF widths;
    # - both familiar temporal shifts;
    # - both photon-count levels.
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
            "F",
        )
    }

    return (
        evaluate_classical_model_mismatch_benchmark(
            tests=tests,
            irf_centre_ns=(
                definition
                .familiar
                .irf_centre_ns
            ),
        )
    )


@pytest.fixture(scope="module")
def day55_model_mismatch_report(
    model_mismatch_benchmark,
    classical_model_mismatch_result,
):
    return (
        build_day55_model_mismatch_report(
            nonclassical_result=(
                model_mismatch_benchmark
            ),
            classical_result=(
                classical_model_mismatch_result
            ),
        )
    )
