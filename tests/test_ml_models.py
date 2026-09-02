import numpy as np
import pytest

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
)

from tcspc_toolkit.ml_models import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    make_hist_gradient_boosting_pipeline,
    make_normalized_histogram_ridge_pipeline,
    make_pca_histogram_ridge_pipeline,
    make_random_forest_pipeline,
    make_ridge_pipeline,
)
from tcspc_toolkit.config import (
    CountNormalization,
)
from tcspc_toolkit.representations import (
    normalize_histogram_batch,
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


def test_normalized_histogram_ridge_pipeline_structure(
) -> None:
    pipeline = (
        make_normalized_histogram_ridge_pipeline()
    )

    assert list(
        pipeline.named_steps
    ) == [
        "normalize",
        "scaler",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["normalize"],
        FunctionTransformer,
    )

    assert isinstance(
        pipeline.named_steps["scaler"],
        StandardScaler,
    )

    assert isinstance(
        pipeline.named_steps["model"],
        Ridge,
    )


def test_pca_histogram_ridge_pipeline_structure(
) -> None:
    pipeline = (
        make_pca_histogram_ridge_pipeline(
            n_components=3,
        )
    )

    assert list(
        pipeline.named_steps
    ) == [
        "normalize",
        "pca",
        "scaler",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["normalize"],
        FunctionTransformer,
    )

    assert isinstance(
        pipeline.named_steps["pca"],
        PCA,
    )

    assert isinstance(
        pipeline.named_steps["scaler"],
        StandardScaler,
    )

    assert isinstance(
        pipeline.named_steps["model"],
        Ridge,
    )

    assert (
        pipeline.named_steps[
            "pca"
        ].n_components
        == 3
    )


def test_pca_histogram_pipeline_uses_default_components(
) -> None:
    pipeline = (
        make_pca_histogram_ridge_pipeline()
    )

    assert (
        pipeline.named_steps[
            "pca"
        ].n_components
        == DEFAULT_PCA_COMPONENTS
    )


@pytest.mark.parametrize(
    ("n_components", "exception_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (2.5, TypeError),
        (True, TypeError),
    ],
)
def test_pca_histogram_pipeline_rejects_invalid_components(
    n_components: object,
    exception_type: type[Exception],
) -> None:
    with pytest.raises(
        exception_type
    ):
        make_pca_histogram_ridge_pipeline(
            n_components=n_components,
        )


def test_normalized_histogram_pipeline_accepts_raw_histograms(
) -> None:
    X = np.asarray(
        [
            [10, 8, 5, 2],
            [20, 15, 8, 3],
            [12, 10, 7, 4],
            [30, 20, 10, 5],
            [8, 7, 5, 3],
            [25, 18, 9, 4],
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
        ],
        dtype=np.float64,
    )

    pipeline = (
        make_normalized_histogram_ridge_pipeline()
    )

    pipeline.fit(
        X,
        y,
    )

    predictions = pipeline.predict(
        X
    )

    assert predictions.shape == y.shape

    assert np.all(
        np.isfinite(predictions)
    )


def test_pca_histogram_pipeline_fits_pca_and_model(
) -> None:
    X = np.asarray(
        [
            [10, 8, 5, 2],
            [20, 15, 8, 3],
            [12, 10, 7, 4],
            [30, 20, 10, 5],
            [8, 7, 5, 3],
            [25, 18, 9, 4],
            [16, 12, 6, 2],
            [22, 17, 10, 6],
        ],
        dtype=np.float64,
    )

    y = np.linspace(
        1.0,
        4.0,
        X.shape[0],
    )

    pipeline = (
        make_pca_histogram_ridge_pipeline(
            n_components=2,
        )
    )

    pipeline.fit(
        X,
        y,
    )

    pca = pipeline.named_steps[
        "pca"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    assert hasattr(
        pca,
        "components_",
    )

    assert pca.components_.shape == (
        2,
        X.shape[1],
    )

    assert hasattr(
        model,
        "coef_",
    )


def test_pipeline_total_normalization_matches_representation_helper(
) -> None:
    X = np.asarray(
        [
            [10, 5, 2],
            [20, 10, 4],
            [8, 3, 1],
        ],
        dtype=np.float64,
    )

    pipeline = (
        make_normalized_histogram_ridge_pipeline()
    )

    normalizer = pipeline.named_steps[
        "normalize"
    ]

    transformed = normalizer.transform(
        X
    )

    expected = normalize_histogram_batch(
        X,
        mode=CountNormalization.TOTAL,
    )

    assert np.allclose(
        transformed,
        expected,
    )


