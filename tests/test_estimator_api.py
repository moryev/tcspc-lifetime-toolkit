"""Extension contracts and small, non-scientific equivalence fixtures."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from tcspc_toolkit.estimator_api import (
    EstimatorSpec,
    fit_regressors,
    predict_regressors,
)

from tcspc_toolkit.ml_models import (
    make_canonical_ml_estimator_specs,
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)
from tcspc_toolkit.representations import (
    fit_pca_representation,
    normalize_histogram_batch,
    transform_pca_representation,
)


LEGACY_FACTORIES = {
    "ridge": make_ridge_pipeline,
    "random_forest": make_random_forest_pipeline,
    "hist_gradient_boosting": make_hist_gradient_boosting_pipeline,
}
CANONICAL_REPRESENTATIONS = (
    "engineered_features", "normalized_histogram", "pca_histogram",
)


@pytest.fixture
def comparison_data():
    """Deterministic plumbing data, not a replacement for the A-F suite.

    Both execution paths use these inputs under the installed dependencies.
    PCA sees training rows only; no benchmark seeds/configurations change.
    """
    x = np.linspace(0.2, 3.0, 60)
    train = np.arange(60) % 5 != 0
    features = pd.DataFrame(
        {"linear": x, "quadratic": x**2, "periodic": np.sin(x)}
    )
    histograms = 100 * np.exp(-np.arange(8)[None, :] / x[:, None]) + 1
    normalized = normalize_histogram_batch(histograms)
    pca = fit_pca_representation(normalized[train], n_components=3)
    X_train = {
        "engineered_features": features.loc[train],
        "normalized_histogram": normalized[train],
        "pca_histogram": transform_pca_representation(pca, normalized[train]),
    }
    X_test = {
        "engineered_features": features.loc[~train],
        "normalized_histogram": normalized[~train],
        "pca_histogram": transform_pca_representation(pca, normalized[~train]),
    }
    y_train = (0.8 + x + 0.1 * np.sin(3 * x))[train]
    return X_train, y_train, X_test


def pipeline_configuration(pipeline):
    return {
        step: {
            "class_name": type(model).__name__,
            "parameters": model.get_params(deep=False),
        }
        for step, model in pipeline.steps
    }


class MeanAdapter:
    """Only fit/predict, with no sklearn base, get_params, or clone hooks."""

    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, inputs, targets):
        self.fit_calls += 1
        self.training_input = inputs
        self.training_targets = targets
        self.mean = float(np.mean(targets))
        # Intentionally return None: only in-place fitting is required.

    def predict(self, inputs):
        self.predict_calls += 1
        self.prediction_input = inputs
        return [self.mean] * inputs.shape[0]


def test_noncanonical_estimators_and_custom_representations():
    features = pd.DataFrame({"custom feature": np.arange(6.0)})
    transformed = np.column_stack([np.arange(6.0), np.arange(6.0)**2])
    specs = (
        EstimatorSpec("user/OLS-v2", LinearRegression, ("my features",)),
        EstimatorSpec(
            "nearest neighbours", lambda: KNeighborsRegressor(n_neighbors=1),
            ("my features", "custom-transform"),
        ),
    )
    targets = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    representations = {
        "my features": features, "custom-transform": transformed,
        "unused": object(),  # Unselected values need not be valid inputs.
    }
    fitted = fit_regressors(
        estimator_specs=specs, X_train_by_representation=representations,
        y_train=targets,
    )
    predictions = predict_regressors(
        fitted_estimators=fitted, X_by_representation=representations,
    )
    assert list(predictions) == [spec.name for spec in specs]
    for spec in specs:
        assert tuple(fitted[spec.name]) == spec.representations
        assert tuple(predictions[spec.name]) == spec.representations
        for representation in spec.representations:
            result = predictions[spec.name][representation]
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float64
            np.testing.assert_allclose(result, targets)
    assert fitted["user/OLS-v2"]["my features"].feature_names_in_.tolist() == [
        "custom feature"
    ]


def test_structural_adapter_preserves_inputs_and_has_independent_instances():
    features = pd.DataFrame({"x": [1.0, 2.0]}, index=[9, 4])
    tensor = np.ones((2, 3, 4))  # No forced 2D sklearn-only representation rule.
    specs = (
        EstimatorSpec("custom", MeanAdapter, ("features", "tensor")),
        EstimatorSpec("another name", MeanAdapter, ("features",)),
    )
    inputs = {"features": features, "tensor": tensor}
    first = fit_regressors(
        estimator_specs=specs, X_train_by_representation=inputs, y_train=[1.0, 3.0],
    )
    second = fit_regressors(
        estimator_specs=specs, X_train_by_representation=inputs, y_train=[1.0, 3.0],
    )
    instances = [
        model
        for run in (first, second)
        for group in run.values()
        for model in group.values()
    ]
    assert len({id(model) for model in instances}) == 6
    test_inputs = {"features": features.iloc[:1], "tensor": tensor[:1]}
    predictions = predict_regressors(
        fitted_estimators=first, X_by_representation=test_inputs,
    )
    for spec in specs:
        for representation in spec.representations:
            adapter = first[spec.name][representation]
            assert adapter.training_input is inputs[representation]
            assert adapter.prediction_input is test_inputs[representation]
            assert adapter.fit_calls == adapter.predict_calls == 1
            assert adapter.training_targets.dtype == np.float64
            np.testing.assert_array_equal(
                predictions[spec.name][representation], [2.0],
            )


def test_spec_is_immutable():
    spec = EstimatorSpec("custom", MeanAdapter, ("input",))
    with pytest.raises(FrozenInstanceError):
        spec.name = "replacement"
    with pytest.raises(FrozenInstanceError):
        spec.representations += ("another",)


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"name": " "}, ValueError),
        ({"name": 1}, TypeError),
        ({"factory": None}, TypeError),
        ({"representations": []}, TypeError),
        ({"representations": ()}, ValueError),
        ({"representations": ("input", "input")}, ValueError),
        ({"representations": ("",)}, ValueError),
    ],
)
def test_invalid_specifications_fail(changes, error):
    kwargs = {"name": "custom", "factory": MeanAdapter, "representations": ("input",)}
    kwargs.update(changes)
    with pytest.raises(error):
        EstimatorSpec(**kwargs)


def unexpected_factory():
    pytest.fail("Invalid input must be rejected before factory execution.")


@pytest.mark.parametrize(
    "targets",
    [
        [], [[1.0], [2.0]], [1.0, np.nan], [1.0, np.inf],
        [0.0, 2.0], ["not numeric"],
    ],
)
def test_fit_rejects_invalid_lifetime_targets_before_calling_factory(targets):
    with pytest.raises(ValueError):
        fit_regressors(
            estimator_specs=(EstimatorSpec("custom", unexpected_factory, ("input",)),),
            X_train_by_representation={"input": np.ones((2, 1))}, y_train=targets,
        )


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"first": np.ones((2, 1))}, "Missing representation"),
        ({"first": np.ones((2, 1)), "second": np.ones((3, 1))}, "row-aligned"),
        ({"first": np.ones((2, 1)), "second": np.ones((0, 1))}, "at least one sample"),
        ({"first": np.ones((2, 1)), "second": np.array(1.0)}, "sample axis"),
        ({"first": np.ones((2, 1)), "second": [[1], [2]]}, "sample axis"),
    ],
)
def test_fit_validates_all_selected_representations_before_fitting(inputs, message):
    with pytest.raises(ValueError, match=message):
        fit_regressors(
            estimator_specs=(
                EstimatorSpec("custom", unexpected_factory, ("first", "second")),
            ),
            X_train_by_representation=inputs, y_train=[1.0, 2.0],
        )


@pytest.mark.parametrize("specs, error", [([], ValueError), ([object()], TypeError)])
def test_fit_requires_specifications(specs, error):
    with pytest.raises(error):
        fit_regressors(estimator_specs=specs, X_train_by_representation={}, y_train=[1.0])


def test_duplicate_estimator_names_fail():
    spec = EstimatorSpec("same", MeanAdapter, ("input",))
    with pytest.raises(ValueError, match="Estimator names must be unique"):
        fit_regressors(
            estimator_specs=(spec, spec),
            X_train_by_representation={"input": np.ones((2, 1))}, y_train=[1.0, 2.0],
        )


def test_factory_reusing_an_instance_fails_before_fitting():
    shared = MeanAdapter()
    with pytest.raises(ValueError, match="independent estimator"):
        fit_regressors(
            estimator_specs=(EstimatorSpec("custom", lambda: shared, ("a", "b")),),
            X_train_by_representation={"a": np.ones((2, 1)), "b": np.ones((2, 2))},
            y_train=[1.0, 2.0],
        )
    assert shared.fit_calls == 0


def test_factory_must_provide_fit_and_predict():
    class FitOnly:
        def fit(self, X, y):
            pass

    with pytest.raises(TypeError, match="fit and predict"):
        fit_regressors(
            estimator_specs=(EstimatorSpec("custom", FitOnly, ("input",)),),
            X_train_by_representation={"input": np.ones((2, 1))}, y_train=[1.0, 2.0],
        )


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"first": np.ones((2, 1))}, "Missing representation"),
        ({"first": np.ones((2, 1)), "second": np.ones((3, 1))}, "row-aligned"),
        ({"first": np.ones((0, 1)), "second": np.ones((0, 1))}, "at least one sample"),
    ],
)
def test_prediction_validates_rows_across_estimators_before_predicting(inputs, message):
    # Unfitted adapters would fail if predict were called before validation.
    fitted = {"one": {"first": MeanAdapter()}, "two": {"second": MeanAdapter()}}
    with pytest.raises(ValueError, match=message):
        predict_regressors(fitted_estimators=fitted, X_by_representation=inputs)
    assert all(
        model.predict_calls == 0
        for group in fitted.values()
        for model in group.values()
    )


@pytest.mark.parametrize("fitted", [{}, {"empty": {}}])
def test_prediction_rejects_empty_selections(fitted):
    with pytest.raises(ValueError):
        predict_regressors(fitted_estimators=fitted, X_by_representation={})


class PredictionAdapter:
    def __init__(self, prediction):
        self.prediction = prediction

    def fit(self, X, y):
        pytest.fail("Prediction must never fit a model.")

    def predict(self, X):
        return self.prediction


@pytest.mark.parametrize(
    "prediction, message",
    [
        (1.0, "one prediction per sample"),
        ([1.0], "one prediction per sample"),
        ([[1.0], [2.0]], "one prediction per sample"),
        ([1.0, np.nan], "finite"),
        ([1.0, np.inf], "finite"),
        (["not numeric", "values"], "numeric"),
    ],
)
def test_prediction_output_contract(prediction, message):
    with pytest.raises(RuntimeError, match=message):
        predict_regressors(
            fitted_estimators={"custom": {"input": PredictionAdapter(prediction)}},
            X_by_representation={"input": np.ones((2, 1))},
        )


def test_predictions_are_float_arrays_without_clipping():
    predictions = predict_regressors(
        fitted_estimators={"custom": {"input": PredictionAdapter((-2, 0, 3))}},
        X_by_representation={"input": np.ones((3, 1))},
    )
    result = predictions["custom"]["input"]
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [-2.0, 0.0, 3.0])


def test_canonical_specs_preserve_labels_configurations_and_seeds():
    specs = make_canonical_ml_estimator_specs()
    assert isinstance(specs, tuple)
    assert tuple(spec.name for spec in specs) == tuple(LEGACY_FACTORIES)
    for spec in specs:
        assert spec.representations == CANONICAL_REPRESENTATIONS
        estimator = spec.factory()
        assert (
            pipeline_configuration(estimator)
            == pipeline_configuration(LEGACY_FACTORIES[spec.name]())
        )
        if spec.name in ("random_forest", "hist_gradient_boosting"):
            assert estimator.named_steps["model"].random_state == 42


def test_generic_canonical_predictions_match_existing_factory_execution(comparison_data):
    X_train, y_train, X_test = comparison_data
    specs = make_canonical_ml_estimator_specs()
    fitted = fit_regressors(
        estimator_specs=specs,
        X_train_by_representation=X_train, y_train=y_train,
    )
    predictions = predict_regressors(fitted_estimators=fitted, X_by_representation=X_test)
    # Fitting one canonical instance must not leak learned state into another.
    for spec in specs:
        assert not hasattr(spec.factory().named_steps["model"], "n_features_in_")
    for name, factory in LEGACY_FACTORIES.items():
        for representation in CANONICAL_REPRESENTATIONS:
            legacy = factory()
            legacy.fit(X_train[representation], y_train)
            np.testing.assert_allclose(
                predictions[name][representation], legacy.predict(X_test[representation]),
                rtol=1e-12, atol=1e-12,
            )
