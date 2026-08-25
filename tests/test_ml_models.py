from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from tcspc_toolkit.ml_models import (
    DEFAULT_RANDOM_STATE,
    make_hist_gradient_boosting_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)


def test_ridge_pipeline_contains_scaler_and_model() -> None:
    pipeline = make_ridge_pipeline()

    assert list(
        pipeline.named_steps
    ) == [
        "scaler",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["scaler"],
        StandardScaler,
    )

    assert isinstance(
        pipeline.named_steps["model"],
        Ridge,
    )


def test_random_forest_pipeline_contains_model() -> None:
    pipeline = make_random_forest_pipeline()

    assert list(
        pipeline.named_steps
    ) == [
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["model"],
        RandomForestRegressor,
    )


def test_random_forest_uses_default_random_state() -> None:
    pipeline = make_random_forest_pipeline()

    model = pipeline.named_steps["model"]

    assert model.random_state == DEFAULT_RANDOM_STATE


def test_hist_gradient_boosting_pipeline_contains_model() -> None:
    pipeline = make_hist_gradient_boosting_pipeline()

    assert list(
        pipeline.named_steps
    ) == [
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["model"],
        HistGradientBoostingRegressor,
    )


def test_hist_gradient_boosting_uses_default_random_state() -> None:
    pipeline = make_hist_gradient_boosting_pipeline()

    model = pipeline.named_steps["model"]

    assert model.random_state == DEFAULT_RANDOM_STATE


def test_random_state_can_be_overridden() -> None:
    random_state = 123

    random_forest = make_random_forest_pipeline(
        random_state=random_state,
    )

    gradient_boosting = (
        make_hist_gradient_boosting_pipeline(
            random_state=random_state,
        )
    )

    assert (
        random_forest
        .named_steps["model"]
        .random_state
        == random_state
    )

    assert (
        gradient_boosting
        .named_steps["model"]
        .random_state
        == random_state
    )


