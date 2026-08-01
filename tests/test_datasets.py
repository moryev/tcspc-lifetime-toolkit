import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.datasets import (
    generate_monoexponential_dataset,
    generate_grouped_monoexponential_dataset,
)


@pytest.fixture
def time_axis() -> np.ndarray:
    return np.linspace(
        0.0,
        20.0,
        128,
        dtype=np.float64,
    )


def test_get_targets_returns_lifetime_column(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_monoexponential_dataset(
        n_curves=5,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    targets = dataset.get_targets(
        "lifetime_true"
    )

    expected = dataset.metadata[
        ["lifetime_true"]
    ].to_numpy(dtype=np.float64)

    np.testing.assert_array_equal(
        targets,
        expected,
    )


def test_generate_dataset_returns_correct_shapes(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_monoexponential_dataset(
        n_curves=10,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    y = dataset.get_targets("lifetime_true").ravel()

    assert dataset.X.shape == (10, 128)
    assert y.shape == (10,)
    assert dataset.time.shape == (128,)
    assert len(dataset.metadata) == 10


def test_generate_monoexponential_dataset_is_reproducible(
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

    first = generate_monoexponential_dataset(**arguments)
    second = generate_monoexponential_dataset(**arguments)

    np.testing.assert_array_equal(
        first.time,
        second.time,
    )
    np.testing.assert_array_equal(
        first.X,
        second.X,
    )
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

    first = generate_monoexponential_dataset(
        **common_arguments,
        random_seed=42,
    )
    second = generate_monoexponential_dataset(
        **common_arguments,
        random_seed=43,
    )

    assert not np.array_equal(first.X, second.X)


def test_metadata_matches_lifetime_targets(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_monoexponential_dataset(
        n_curves=10,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    y = dataset.get_targets("lifetime_true").ravel()

    np.testing.assert_array_equal(
        y,
        dataset.metadata["lifetime_true"].to_numpy(),
    )


def test_expected_photon_count_matches_target(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_monoexponential_dataset(
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
    dataset = generate_monoexponential_dataset(
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
        generate_monoexponential_dataset(
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
        generate_monoexponential_dataset(
            n_curves=5,
            time=time_axis,
            lifetime_range=(-1.0, 4.0),
            amplitude_range=(100.0, 1_000.0),
            background_range=(0.0, 10.0),
            photon_count_range=(1_000, 10_000),
            random_seed=42,
        )


def test_generate_grouped_dataset_returns_correct_shapes(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    n_expected_curves = 4 * 3

    assert dataset.X.shape == (
        n_expected_curves,
        len(time_axis),
    )
    assert dataset.time.shape == (len(time_axis),)
    assert dataset.metadata.shape[0] == n_expected_curves


def test_grouped_dataset_contains_required_metadata_columns(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    expected_columns = {
        "curve_id",
        "parameter_group",
        "realization_id",
        "lifetime_true",
        "amplitude_true",
        "background_true",
        "photon_count_target",
        "photon_count_expected",
        "photon_count",
    }

    assert expected_columns.issubset(
        dataset.metadata.columns
    )


def test_grouped_dataset_has_expected_number_of_groups(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    parameter_groups = dataset.metadata[
        "parameter_group"
    ]

    assert parameter_groups.nunique() == 4

    np.testing.assert_array_equal(
        np.sort(parameter_groups.unique()),
        np.arange(4, dtype=np.int64),
    )


def test_each_parameter_group_has_expected_number_of_realizations(
    time_axis: np.ndarray,
) -> None:
    n_parameter_groups = 4
    n_realizations_per_group = 3

    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=n_parameter_groups,
        n_realizations_per_group=n_realizations_per_group,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    group_sizes = dataset.metadata.groupby(
        "parameter_group"
    ).size()

    assert len(group_sizes) == n_parameter_groups
    assert (
        group_sizes == n_realizations_per_group
    ).all()


def test_realization_ids_restart_within_each_group(
    time_axis: np.ndarray,
) -> None:
    n_realizations_per_group = 3

    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=n_realizations_per_group,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    expected_realization_ids = np.arange(
        n_realizations_per_group,
        dtype=np.int64,
    )

    for _, group_metadata in dataset.metadata.groupby(
        "parameter_group",
        sort=True,
    ):
        actual_realization_ids = group_metadata[
            "realization_id"
        ].to_numpy()

        np.testing.assert_array_equal(
            actual_realization_ids,
            expected_realization_ids,
        )


def test_curve_ids_are_unique_and_sequential(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    curve_ids = dataset.metadata[
        "curve_id"
    ].to_numpy()

    np.testing.assert_array_equal(
        curve_ids,
        np.arange(len(dataset.metadata), dtype=np.int64),
    )

    assert dataset.metadata["curve_id"].is_unique


def test_parameters_are_constant_within_each_group(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=5,
        n_realizations_per_group=4,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    parameter_columns = [
        "lifetime_true",
        "amplitude_true",
        "background_true",
        "photon_count_target",
        "photon_count_expected",
    ]

    unique_values_per_group = dataset.metadata.groupby(
        "parameter_group"
    )[parameter_columns].nunique()

    assert (
        unique_values_per_group == 1
    ).all().all()


def test_grouped_dataset_targets_match_metadata(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    targets = dataset.get_targets(
        "lifetime_true"
    ).ravel()

    expected = dataset.metadata[
        "lifetime_true"
    ].to_numpy(dtype=np.float64)

    np.testing.assert_array_equal(
        targets,
        expected,
    )


def test_grouped_dataset_is_reproducible(
    time_axis: np.ndarray,
) -> None:
    arguments = {
        "n_parameter_groups": 5,
        "n_realizations_per_group": 3,
        "time": time_axis,
        "lifetime_range": (1.0, 4.0),
        "amplitude_range": (100.0, 1_000.0),
        "background_range": (0.0, 10.0),
        "photon_count_range": (1_000, 10_000),
        "random_seed": 42,
    }

    first = generate_grouped_monoexponential_dataset(
        **arguments
    )
    second = generate_grouped_monoexponential_dataset(
        **arguments
    )

    np.testing.assert_array_equal(
        first.time,
        second.time,
    )
    np.testing.assert_array_equal(
        first.X,
        second.X,
    )
    pd.testing.assert_frame_equal(
        first.metadata,
        second.metadata,
    )


def test_grouped_dataset_different_seeds_produce_different_curves(
    time_axis: np.ndarray,
) -> None:
    common_arguments = {
        "n_parameter_groups": 5,
        "n_realizations_per_group": 3,
        "time": time_axis,
        "lifetime_range": (1.0, 4.0),
        "amplitude_range": (100.0, 1_000.0),
        "background_range": (0.0, 10.0),
        "photon_count_range": (1_000, 10_000),
    }

    first = generate_grouped_monoexponential_dataset(
        **common_arguments,
        random_seed=42,
    )
    second = generate_grouped_monoexponential_dataset(
        **common_arguments,
        random_seed=43,
    )

    assert not np.array_equal(
        first.X,
        second.X,
    )


def test_realizations_within_group_are_independently_sampled(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=3,
        n_realizations_per_group=5,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(10_000, 20_000),
        random_seed=42,
    )

    first_group_indices = dataset.metadata.index[
        dataset.metadata["parameter_group"] == 0
    ].to_numpy()

    first_group_curves = dataset.X[
        first_group_indices
    ]

    unique_curves = np.unique(
        first_group_curves,
        axis=0,
    )

    assert unique_curves.shape[0] > 1


def test_expected_photon_count_matches_target_for_grouped_dataset(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=10,
        n_realizations_per_group=3,
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


def test_measured_photon_count_matches_histogram_sum(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=5,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    histogram_sums = dataset.X.sum(axis=1)

    np.testing.assert_array_equal(
        dataset.metadata[
            "photon_count"
        ].to_numpy(),
        histogram_sums,
    )


def test_grouped_long_dataframe_has_one_row_per_time_bin(
    time_axis: np.ndarray,
) -> None:
    n_parameter_groups = 4
    n_realizations_per_group = 3

    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=n_parameter_groups,
        n_realizations_per_group=n_realizations_per_group,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    long_dataframe = dataset.to_long_dataframe()

    n_curves = (
        n_parameter_groups
        * n_realizations_per_group
    )

    assert len(long_dataframe) == (
        n_curves * len(time_axis)
    )

    expected_columns = {
        "curve_id",
        "time_bin",
        "counts",
        "parameter_group",
        "realization_id",
        "lifetime_true",
        "amplitude_true",
        "background_true",
        "photon_count_target",
        "photon_count_expected",
        "photon_count",
    }

    assert expected_columns.issubset(
        long_dataframe.columns
    )


def test_grouped_long_dataframe_preserves_group_metadata(
    time_axis: np.ndarray,
) -> None:
    dataset = generate_grouped_monoexponential_dataset(
        n_parameter_groups=4,
        n_realizations_per_group=3,
        time=time_axis,
        lifetime_range=(1.0, 4.0),
        amplitude_range=(100.0, 1_000.0),
        background_range=(0.0, 10.0),
        photon_count_range=(1_000, 10_000),
        random_seed=42,
    )

    long_dataframe = dataset.to_long_dataframe()

    rows_per_curve = long_dataframe.groupby(
        "curve_id"
    ).size()

    assert (
        rows_per_curve == len(time_axis)
    ).all()

    groups_per_curve = long_dataframe.groupby(
        "curve_id"
    )["parameter_group"].nunique()

    realization_ids_per_curve = long_dataframe.groupby(
        "curve_id"
    )["realization_id"].nunique()

    assert (groups_per_curve == 1).all()
    assert (realization_ids_per_curve == 1).all()


def test_non_positive_number_of_parameter_groups_is_rejected(
    time_axis: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_parameter_groups must be positive",
    ):
        generate_grouped_monoexponential_dataset(
            n_parameter_groups=0,
            n_realizations_per_group=3,
            time=time_axis,
            lifetime_range=(1.0, 4.0),
            amplitude_range=(100.0, 1_000.0),
            background_range=(0.0, 10.0),
            photon_count_range=(1_000, 10_000),
            random_seed=42,
        )


def test_non_positive_number_of_realizations_is_rejected(
    time_axis: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_realizations_per_group must be positive",
    ):
        generate_grouped_monoexponential_dataset(
            n_parameter_groups=5,
            n_realizations_per_group=0,
            time=time_axis,
            lifetime_range=(1.0, 4.0),
            amplitude_range=(100.0, 1_000.0),
            background_range=(0.0, 10.0),
            photon_count_range=(1_000, 10_000),
            random_seed=42,
        )


@pytest.mark.parametrize(
    (
        "n_parameter_groups",
        "n_realizations_per_group",
        "expected_message",
    ),
    [
        (
            0,
            3,
            "n_parameter_groups must be positive",
        ),
        (
            -1,
            3,
            "n_parameter_groups must be positive",
        ),
        (
            5,
            0,
            "n_realizations_per_group must be positive",
        ),
        (
            5,
            -1,
            "n_realizations_per_group must be positive",
        ),
    ],
)
def test_invalid_group_dimensions_are_rejected(
    time_axis: np.ndarray,
    n_parameter_groups: int,
    n_realizations_per_group: int,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        generate_grouped_monoexponential_dataset(
            n_parameter_groups=n_parameter_groups,
            n_realizations_per_group=(
                n_realizations_per_group
            ),
            time=time_axis,
            lifetime_range=(1.0, 4.0),
            amplitude_range=(100.0, 1_000.0),
            background_range=(0.0, 10.0),
            photon_count_range=(1_000, 10_000),
            random_seed=42,
        )
