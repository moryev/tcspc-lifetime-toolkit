from dataclasses import FrozenInstanceError, replace

import pytest

import numpy as np

from tcspc_toolkit.config import (
    CountNormalization,
    FeatureConfig,
    PreprocessingConfig,
    SimulationConfig,
    load_config,
    save_config,
)


def test_simulation_config_is_immutable() -> None:
    config = SimulationConfig(
        lifetime_ns=2.5,
        amplitude=10_000.0,
        background=5.0,
        n_bins=512,
        bin_width_ns=0.02,
        irf_fwhm_ns=0.3,
        irf_shift_ns=0.1,
    )

    with pytest.raises(FrozenInstanceError):
        config.lifetime_ns = 5.0


def test_replace_creates_modified_config() -> None:
    original = SimulationConfig(
        lifetime_ns=2.5,
        amplitude=10_000.0,
        background=5.0,
        n_bins=512,
        bin_width_ns=0.02,
        irf_fwhm_ns=0.3,
        irf_shift_ns=0.1,
    )

    modified = replace(
        original,
        lifetime_ns=5.0,
    )

    assert original.lifetime_ns == 2.5
    assert modified.lifetime_ns == 5.0


def test_simulation_config_json_round_trip(
    tmp_path,
) -> None:
    config = SimulationConfig(
        lifetime_ns=2.5,
        amplitude=10_000.0,
        background=5.0,
        n_bins=512,
        bin_width_ns=0.02,
        irf_fwhm_ns=0.3,
        irf_shift_ns=0.1,
    )

    path = tmp_path / "simulation.json"

    save_config(
        config=config,
        path=path,
    )

    loaded = load_config(
        path=path,
        config_type=SimulationConfig,
    )

    assert loaded == config


def test_preprocessing_config_json_round_trip(
    tmp_path,
) -> None:
    config = PreprocessingConfig(
        background_start_bin=0,
        background_stop_bin=50,
        crop_start_ns=0.5,
        crop_stop_ns=10.0,
        rebin_factor=2,
        normalization=CountNormalization.TOTAL,
    )

    path = tmp_path / "preprocessing.json"

    save_config(
        config=config,
        path=path,
    )

    loaded = load_config(
        path=path,
        config_type=PreprocessingConfig,
    )

    assert loaded == config


def test_feature_config_json_round_trip(
    tmp_path,
) -> None:
    config = FeatureConfig(
        tail_start_ns=4.0,
        early_stop_ns=2.0,
        late_start_ns=4.0,
        min_tail_points=5,
    )

    path = tmp_path / "features.json"

    save_config(
        config=config,
        path=path,
    )

    loaded = load_config(
        path=path,
        config_type=FeatureConfig,
    )

    assert loaded == config


@pytest.mark.parametrize(
    "field_name",
    [
        "tail_start_ns",
        "early_stop_ns",
        "late_start_ns",
    ],
)
def test_feature_config_rejects_non_finite_boundaries(
    field_name: str,
) -> None:
    values = {
        "tail_start_ns": 4.0,
        "early_stop_ns": 2.0,
        "late_start_ns": 4.0,
    }

    values[field_name] = np.nan

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        FeatureConfig(**values)


def test_feature_config_rejects_overlapping_regions() -> None:
    with pytest.raises(
        ValueError,
        match="early_stop_ns must be smaller",
    ):
        FeatureConfig(
            tail_start_ns=4.0,
            early_stop_ns=5.0,
            late_start_ns=4.0,
        )


def test_feature_config_rejects_too_few_tail_points() -> None:
    with pytest.raises(
        ValueError,
        match="at least 3",
    ):
        FeatureConfig(
            tail_start_ns=4.0,
            early_stop_ns=2.0,
            late_start_ns=4.0,
            min_tail_points=2,
        )


def test_feature_config_rejects_non_integer_tail_points() -> None:
    with pytest.raises(
        ValueError,
        match="must be an integer",
    ):
        FeatureConfig(
            tail_start_ns=4.0,
            early_stop_ns=2.0,
            late_start_ns=4.0,
            min_tail_points=3.5,
        )


