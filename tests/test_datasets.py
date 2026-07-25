import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.datasets import generate_dataset


@pytest.fixture
def time_axis() -> np.ndarray:
    return np.linspace(
        0.0,
        20.0,
        128,
        dtype=np.float64,
    )


def test_generate_dataset_returns_correct_shapes(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_dataset(
        n_curves=10,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    assert dataset.X.shape == (10, 128)
    assert dataset.y.shape == (10,)
    assert dataset.time.shape == (128,)
    assert len(dataset.metadata) == 10


def test_generate_dataset_is_reproducible(
    time_axis: np.ndarray,
) -> None:
    arguments = {
        "n_curves": 5,
        "time": time_axis,
        "lifetime_range": (1.0, 4.0),
        "amplitude_range": (100.0, 1_000.0),
        "background_range": (0.0, 10.0),
        "photon_count_range": (1_000, 10_000),
        "random_seed": 42,
    }

    first = generate_dataset(**arguments)
    second = generate_dataset(**arguments)

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    pd.testing.assert_frame_equal(
        first.metadata,
        second.metadata,
    )


def test_different_seeds_produce_different_curves(
    time_axis: np.ndarray,
) -> None:
    common_arguments = {
        "n_curves": 5,
        "time": time_axis,
        "lifetime_range": (1.0, 4.0),
        "amplitude_range": (100.0, 1_000.0),
        "background_range": (0.0, 10.0),
        "photon_count_range": (1_000, 10_000),
    }

    first = generate_dataset(
        **common_arguments,
        random_seed=42,
    )
    second = generate_dataset(
        **common_arguments,
        random_seed=43,
    )

    assert not np.array_equal(first.X, second.X)


def test_metadata_matches_lifetime_targets(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_dataset(
        n_curves=10,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    np.testing.assert_array_equal(
        dataset.y,
        dataset.metadata["lifetime_true"].to_numpy(),
    )


def test_expected_photon_count_matches_target(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_dataset(
        n_curves=20,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    np.testing.assert_allclose(
        dataset.metadata["photon_count_expected"],
        dataset.metadata["photon_count_target"],
        rtol=1e-12,
        atol=1e-8,
    )


def test_long_dataframe_has_one_row_per_time_bin(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_dataset(
        n_curves=10,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    long_dataframe = dataset.to_long_dataframe()

    assert len(long_dataframe) == 10 * len(time_axis)

    assert {
        "curve_id",
        "time_bin",
        "counts",
        "lifetime_true",
        "amplitude_true",
        "background_true",
        "photon_count",
    }.issubset(long_dataframe.columns)


def test_invalid_number_of_curves_is_rejected(
    time_axis: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_curves must be positive",
    ):
        generate_dataset(
            n_curves=0,
            time=time_axis,
            lifetime_range=(1.0, 4.0),
            amplitude_range=(100.0, 1_000.0),
            background_range=(0.0, 10.0),
            photon_count_range=(1_000, 10_000),
            random_seed=42,
        )


def test_negative_lifetime_range_is_rejected(
    time_axis: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="lifetime range",
    ):
        generate_dataset(
            n_curves=5,
            time=time_axis,
            lifetime_range=(-1.0, 4.0),
            amplitude_range=(100.0, 1_000.0),
            background_range=(0.0, 10.0),
            photon_count_range=(1_000, 10_000),
            random_seed=42,
        )
