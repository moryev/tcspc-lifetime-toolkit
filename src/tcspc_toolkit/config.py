"""Configuration objects and serialization utilities for TCSPC workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path


class CountNormalization(Enum):
    TOTAL = "total"
    PEAK = "peak"


@dataclass(frozen=True)
class SimulationConfig:
    lifetime_ns: float
    amplitude: float
    background: float
    n_bins: int
    bin_width_ns: float
    irf_fwhm_ns: float
    irf_shift_ns: float


@dataclass(frozen=True)
class PreprocessingConfig:
    background_start_bin: int | None = None
    background_stop_bin: int | None = None
    crop_start_ns: float | None = None
    crop_stop_ns: float | None = None
    rebin_factor: int = 1
    normalization: CountNormalization | None = None


Config = SimulationConfig | PreprocessingConfig


def _config_to_dict(
    config: Config,
) -> dict[str, object]:
    """Convert a configuration object to JSON-compatible data."""
    data = asdict(config)

    if isinstance(config, PreprocessingConfig):
        if config.normalization is not None:
            data["normalization"] = config.normalization.value

    return data


def save_config(
    config: Config,
    path: str | Path,
) -> None:
    """Save a configuration object as JSON."""
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = _config_to_dict(config)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def load_config(
    path: str | Path,
    config_type: type[SimulationConfig] | type[PreprocessingConfig],
) -> Config:
    """Load a configuration object from JSON."""
    input_path = Path(path)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "configuration JSON must contain an object"
        )

    if config_type is PreprocessingConfig:
        normalization = data.get("normalization")

        if normalization is not None:
            data["normalization"] = CountNormalization(
                normalization
            )

    try:
        return config_type(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid configuration for {config_type.__name__}"
        ) from exc
