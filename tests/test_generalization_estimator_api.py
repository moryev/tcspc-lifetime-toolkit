"""Point-estimator extension tests without redefining the frozen protocols."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tcspc_toolkit.baselines import (
    estimate_lifetime_from_mean_arrival,
    predict_constant_mean_baseline,
)
from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.estimator_api import EstimatorSpec
from tcspc_toolkit.generalization import FINAL_ROBUSTNESS_TEST_IDS
from tcspc_toolkit.generalization_datasets import generate_generalization_test_suite
from tcspc_toolkit.generalization_evaluation import (
    GeneralizationABPreparedData,
    _evaluate_principal_nonclassical_estimators,
    _build_generalization_prediction_table,
    build_ab_comparison_table,
    build_generalization_development_measurements,
    build_mae_degradation_table,
    build_reference_mae_degradation_table,
    evaluate_generalization_suite_benchmark,
    evaluate_instrument_acquisition_benchmark,
    evaluate_model_mismatch_benchmark,
    evaluate_ml_representation_ab_benchmark,
    fit_generalization_ml_estimators,
    prepare_generalization_data,
    summarize_generalization_predictions,
)
from tcspc_toolkit.ml_models import (
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)


# The pre-Stage-2 execution order is an independent regression oracle, not a
# second production configuration. Do not derive it from the new specs.
LEGACY_FACTORIES = {
    "ridge": make_ridge_pipeline,
    "random_forest": make_random_forest_pipeline,
    "hist_gradient_boosting": make_hist_gradient_boosting_pipeline,
}
REPRESENTATIONS = ("engineered_features", "normalized_histogram", "pca_histogram")


@pytest.fixture(scope="module")
def prepared():
    suite = generate_generalization_test_suite()
    return prepare_generalization_data(
        development_measurements=build_generalization_development_measurements(),
        tests={test.test_id: test for test in suite.tests},
        feature_config=FeatureConfig(
            tail_start_ns=2.0, early_stop_ns=2.0, late_start_ns=3.0,
        ),
    )


def _test_inputs(prepared, test_id):
    return {
        "engineered_features": prepared.X_features[test_id],
        "normalized_histogram": prepared.X_normalized[test_id],
        "pca_histogram": prepared.X_pca[test_id],
    }


@pytest.fixture(scope="module")
def canonical_fits(prepared):
    development_inputs = {
        "engineered_features": prepared.development.X_features,
        "normalized_histogram": prepared.X_normalized_development,
        "pca_histogram": prepared.X_pca_development,
    }
    legacy = {}
    for name, factory in LEGACY_FACTORIES.items():
        legacy[name] = {}
        for representation, X in development_inputs.items():
            estimator = factory()
            estimator.fit(X, prepared.development.y)
            legacy[name][representation] = estimator
    return legacy, fit_generalization_ml_estimators(prepared)


def legacy_prediction_tables(prepared, fitted, test_ids):
    """The old ordered baseline/ML prediction loop, using unchanged builders."""
    tables = []
    for test_id in test_ids:
        test = prepared.tests[test_id]
        features = prepared.X_features[test_id]
        baseline_predictions = (
            ("constant_mean", "none", predict_constant_mean_baseline(
                y_train=prepared.development.y, n_predictions=test.y.size,
            )),
            ("mean_arrival_time", "engineered_features", estimate_lifetime_from_mean_arrival(
                mean_arrival_time_ns=features["mean_arrival_time_ns"].to_numpy(dtype=np.float64),
                peak_time_ns=features["peak_time_ns"].to_numpy(dtype=np.float64),
            )),
        )
        for name, representation, predictions in baseline_predictions:
            tables.append(_build_generalization_prediction_table(
                estimator_name=name, representation_name=representation,
                test=test, y_pred=predictions,
            ))
        inputs = _test_inputs(prepared, test_id)
        for name in LEGACY_FACTORIES:
            for representation in REPRESENTATIONS:
                tables.append(_build_generalization_prediction_table(
                    estimator_name=name, representation_name=representation,
                    test=test, y_pred=np.asarray(
                        fitted[name][representation].predict(inputs[representation]),
                        dtype=np.float64,
                    ),
                ))
    return pd.concat(tables, ignore_index=True)


@pytest.mark.parametrize(
    "evaluate, test_ids",
    [
        (evaluate_generalization_suite_benchmark, FINAL_ROBUSTNESS_TEST_IDS),
        (evaluate_instrument_acquisition_benchmark, ("A", "C", "D", "E")),
        (evaluate_model_mismatch_benchmark, ("A", "F")),
    ],
)
def test_canonical_predictions_and_tables_match_legacy_execution(
    prepared, canonical_fits, evaluate, test_ids,
):
    legacy, current = canonical_fits
    assert tuple(current) == tuple(LEGACY_FACTORIES)
    assert all(tuple(group) == REPRESENTATIONS for group in current.values())
    expected = legacy_prediction_tables(prepared, legacy, test_ids)
    result = evaluate(prepared=prepared, fitted_estimators=current)
    pd.testing.assert_frame_equal(result.predictions, expected, check_exact=True)
    summary = summarize_generalization_predictions(expected)
    pd.testing.assert_frame_equal(result.summary, summary, check_exact=True)
    pd.testing.assert_frame_equal(
        result.degradation,
        build_reference_mae_degradation_table(summary, reference_test_id="A"),
        check_exact=True,
    )


@pytest.mark.parametrize("principal", [True, False])
def test_ab_paths_match_independent_legacy_execution(prepared, canonical_fits, principal):
    legacy, current = canonical_fits
    ab = GeneralizationABPreparedData(
        development=prepared.development,
        test_a=prepared.tests["A"], test_b=prepared.tests["B"],
        X_features_a=prepared.X_features["A"], X_features_b=prepared.X_features["B"],
        X_normalized_development=prepared.X_normalized_development,
        X_normalized_a=prepared.X_normalized["A"], X_normalized_b=prepared.X_normalized["B"],
        X_pca_development=prepared.X_pca_development,
        X_pca_a=prepared.X_pca["A"], X_pca_b=prepared.X_pca["B"], pca=prepared.pca,
    )
    expected = legacy_prediction_tables(prepared, legacy, ("A", "B"))
    baselines = expected.estimator.isin(["constant_mean", "mean_arrival_time"])
    selected = (baselines | expected.representation.eq("engineered_features")) if principal else ~baselines
    expected = expected.loc[selected].reset_index(drop=True)
    # Frozen wrappers own selection/order, not the caller's dictionary order.
    reversed_fits = {name: dict(reversed(tuple(group.items())))
                     for name, group in reversed(tuple(current.items()))}
    if principal:
        actual = pd.concat(_evaluate_principal_nonclassical_estimators(
            prepared=ab, fitted_estimators=reversed_fits,
        ), ignore_index=True)
        summary = summarize_generalization_predictions(actual)
        degradation = build_mae_degradation_table(summary)
    else:
        result = evaluate_ml_representation_ab_benchmark(prepared=ab, fitted_estimators=reversed_fits)
        actual, summary, degradation = result.predictions, result.summary, result.degradation
    expected_summary = summarize_generalization_predictions(expected)
    expected_degradation = build_mae_degradation_table(expected_summary)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(summary, expected_summary, check_exact=True)
    pd.testing.assert_frame_equal(degradation, expected_degradation, check_exact=True)
    if not principal:
        pd.testing.assert_frame_equal(result.comparison, build_ab_comparison_table(
            summary=expected_summary, degradation=expected_degradation,
        ), check_exact=True)


def test_custom_estimator_selects_one_prepared_representation(prepared):
    fitted = fit_generalization_ml_estimators(
        prepared,
        estimator_specs=(EstimatorSpec("user/OLS-v2", LinearRegression, ("pca_histogram",)),),
    )
    assert tuple(fitted) == ("user/OLS-v2",)
    assert tuple(fitted["user/OLS-v2"]) == ("pca_histogram",)
    result = evaluate_generalization_suite_benchmark(
        prepared=prepared, fitted_estimators=fitted,
    )
    expected = LinearRegression().fit(prepared.X_pca_development, prepared.development.y)
    for test_id in FINAL_ROBUSTNESS_TEST_IDS:
        rows = result.predictions.loc[
            (result.predictions.estimator == "user/OLS-v2")
            & (result.predictions.test_id == test_id)
        ]
        np.testing.assert_array_equal(
            rows.predicted_lifetime_ns, expected.predict(prepared.X_pca[test_id]),
        )


@pytest.fixture
def custom_inputs(prepared):
    columns = ["mean_arrival_time_ns", "peak_time_ns"]
    development = {"arrival-pair": prepared.development.X_features[columns]}
    tests = {
        test_id: {"arrival-pair": prepared.X_features[test_id][columns]}
        for test_id in FINAL_ROBUSTNESS_TEST_IDS
    }
    specs = (EstimatorSpec("my linear model", LinearRegression, ("arrival-pair",)),)
    return specs, development, tests


def test_explicit_custom_representations_leave_preparation_and_baselines_unchanged(
    prepared, custom_inputs,
):
    specs, development, tests = custom_inputs
    original_features = prepared.development.X_features.copy(deep=True)
    original_components = prepared.pca.components_.copy()
    fitted = fit_generalization_ml_estimators(
        prepared, estimator_specs=specs,
        X_development_by_representation=development,
    )
    # This unselected override must not become the mean-arrival baseline input.
    for mapping in tests.values():
        mapping["engineered_features"] = object()
    result = evaluate_generalization_suite_benchmark(
        prepared=prepared, fitted_estimators=fitted,
        X_by_test_and_representation=tests,
    )
    pd.testing.assert_frame_equal(prepared.development.X_features, original_features)
    np.testing.assert_array_equal(prepared.pca.components_, original_components)
    assert set(result.summary.estimator) == {
        "my linear model", "constant_mean", "mean_arrival_time",
    }
    estimator = fitted["my linear model"]["arrival-pair"]
    assert estimator.feature_names_in_.tolist() == development["arrival-pair"].columns.tolist()
    for test_id in FINAL_ROBUSTNESS_TEST_IDS:
        rows = result.predictions.loc[result.predictions.test_id == test_id]
        ml_rows = rows.loc[rows.estimator == "my linear model"]
        assert set(ml_rows.representation) == {"arrival-pair"}
        np.testing.assert_array_equal(
            ml_rows.predicted_lifetime_ns, estimator.predict(tests[test_id]["arrival-pair"]),
        )
        features = prepared.X_features[test_id]
        np.testing.assert_array_equal(
            rows.loc[rows.estimator == "mean_arrival_time", "predicted_lifetime_ns"],
            estimate_lifetime_from_mean_arrival(
                mean_arrival_time_ns=features.mean_arrival_time_ns.to_numpy(dtype=np.float64),
                peak_time_ns=features.peak_time_ns.to_numpy(dtype=np.float64),
            ),
        )


@pytest.mark.parametrize("failure", ["missing", "row_count"])
def test_development_mapping_failures(prepared, custom_inputs, failure):
    specs, development, _ = custom_inputs
    if failure == "missing":
        development = {}
    else:
        development["arrival-pair"] = development["arrival-pair"].iloc[:-1]
    with pytest.raises(ValueError, match="Missing representation|row-aligned"):
        fit_generalization_ml_estimators(
            prepared, estimator_specs=specs,
            X_development_by_representation=development,
        )


@pytest.mark.parametrize("failure", ["missing_test", "missing_representation", "row_count"])
def test_test_mapping_failures(prepared, custom_inputs, failure):
    specs, development, tests = custom_inputs
    fitted = fit_generalization_ml_estimators(
        prepared, estimator_specs=specs,
        X_development_by_representation=development,
    )
    if failure == "missing_test":
        del tests["A"]
    elif failure == "missing_representation":
        tests["A"] = {}
    else:
        tests["A"]["arrival-pair"] = tests["A"]["arrival-pair"].iloc[:-1]
    with pytest.raises(ValueError, match="Missing representation|row-aligned"):
        evaluate_generalization_suite_benchmark(
            prepared=prepared, fitted_estimators=fitted,
            X_by_test_and_representation=tests,
        )


@pytest.mark.parametrize(
    "name", ["constant_mean", "mean_arrival_time", "classical_reconvolution_custom"],
)
def test_reporting_identity_collisions_are_rejected(prepared, name):
    with pytest.raises(ValueError, match="reserved by generalization reporting"):
        evaluate_generalization_suite_benchmark(
            prepared=prepared, fitted_estimators={name: {}},
        )
