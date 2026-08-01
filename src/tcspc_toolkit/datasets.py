from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tcspc_toolkit.models import monoexponential_decay


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticDataset:
    """
    Collection of synthetic TCSPC decay curves.

    Attributes
    ----------
    time:
        Shared time axis with shape ``(n_time_bins,)``.
    X:
        Measured photon counts with shape
        ``(n_curves, n_time_bins)``.
    y:
        True lifetime for every curve with shape ``(n_curves,)``.
    metadata:
        One metadata row per generated curve.
    """

    time: FloatArray
    X: IntArray
    metadata: pd.DataFrame

    def get_targets(
            self,
            columns: str | list[str],
    ) -> FloatArray:
        selected = self.metadata[[columns] if isinstance(columns, str) else columns]
        return selected.to_numpy(dtype=np.float64)


    def to_long_dataframe(self) -> pd.DataFrame:
        """
        Convert the matrix representation into tidy long-table format.

        Returns
        -------
        pandas.DataFrame
            A table containing one row per time bin and curve.
        """
        n_curves, n_time_bins = self.X.shape

        curve_ids = np.repeat(
            np.arange(n_curves, dtype=np.int64),
            n_time_bins,
        )
        time_values = np.tile(self.time, n_curves)
        counts = self.X.reshape(-1)

        long_dataframe = pd.DataFrame(
            {
                "curve_id": curve_ids,
                "time_bin": time_values,
                "counts": counts,
            }
        )

        return long_dataframe.merge(
            self.metadata,
            on="curve_id",
            how="left",
            validate="many_to_one",
        )


def _validate_range(
    parameter_range: tuple[float, float],
    parameter_name: str,
    *,
    allow_zero: bool,
) -> None:
    """
    Validate a numerical sampling range.
    """
    lower, upper = parameter_range

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(
            f"{parameter_name} range must contain finite values."
        )

    minimum_allowed = 0.0 if allow_zero else np.nextafter(0.0, 1.0)

    if lower < minimum_allowed:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(
            f"{parameter_name} range must contain only "
            f"{comparison} values."
        )

    if upper < lower:
        raise ValueError(
            f"{parameter_name} range upper bound must be "
            f"greater than or equal to its lower bound."
        )


def _sample_uniform(
    rng: np.random.Generator,
    parameter_range: tuple[float, float],
    size: int,
) -> FloatArray:
    """
    Sample values uniformly, including support for fixed ranges.
    """
    lower, upper = parameter_range

    if lower == upper:
        return np.full(size, lower, dtype=np.float64)

    return rng.uniform(lower, upper, size=size)


def generate_monoexponential_dataset(
    *,
    n_curves: int,
    time: FloatArray,
    lifetime_range: tuple[float, float],
    amplitude_range: tuple[float, float],
    background_range: tuple[float, float],
    photon_count_range: tuple[int, int],
    random_seed: int | None = None,
) -> SyntheticDataset:
    """
    Generate synthetic Poisson-sampled mono-exponential TCSPC curves.

    Parameters
    ----------
    n_curves:
        Number of curves to generate.
    time:
        One-dimensional time axis shared by all curves.
    lifetime_range:
        Minimum and maximum lifetime.
    amplitude_range:
        Minimum and maximum provisional signal amplitude.
    background_range:
        Minimum and maximum provisional background level.
    photon_count_range:
        Minimum and maximum target expected photon count.
    random_seed:
        Seed used for reproducible parameter generation and Poisson
        sampling.

    Returns
    -------
    SyntheticDataset
        Dataset containing matrix and metadata representations.

    Notes
    -----
    The sampled amplitude and background are initially relative values.
    Both are multiplied by the same scale factor so that the total
    expected photon count matches the sampled target photon count.
    The scaled values are stored in the returned metadata.
    """
    if n_curves <= 0:
        raise ValueError("n_curves must be positive.")

    time = np.asarray(time, dtype=np.float64)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if time.size == 0:
        raise ValueError("time must contain at least one value.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    _validate_range(
        lifetime_range,
        "lifetime",
        allow_zero=False,
    )
    _validate_range(
        amplitude_range,
        "amplitude",
        allow_zero=False,
    )
    _validate_range(
        background_range,
        "background",
        allow_zero=True,
    )

    photon_count_min, photon_count_max = photon_count_range

    if photon_count_min <= 0:
        raise ValueError(
            "photon_count_range must contain positive values."
        )

    if photon_count_max < photon_count_min:
        raise ValueError(
            "photon_count_range upper bound must be greater than "
            "or equal to its lower bound."
        )

    rng = np.random.default_rng(random_seed)

    lifetimes = _sample_uniform(
        rng,
        lifetime_range,
        n_curves,
    )
    provisional_amplitudes = _sample_uniform(
        rng,
        amplitude_range,
        n_curves,
    )
    provisional_backgrounds = _sample_uniform(
        rng,
        background_range,
        n_curves,
    )

    target_photon_counts = rng.integers(
        photon_count_min,
        photon_count_max + 1,
        size=n_curves,
        dtype=np.int64,
    )

    X = np.empty(
        (n_curves, time.size),
        dtype=np.int64,
    )

    amplitudes_true = np.empty(n_curves, dtype=np.float64)
    backgrounds_true = np.empty(n_curves, dtype=np.float64)
    expected_photon_counts = np.empty(n_curves, dtype=np.float64)
    measured_photon_counts = np.empty(n_curves, dtype=np.int64)

    for curve_id in range(n_curves):
        provisional_expected_counts = monoexponential_decay(
            time=time,
            amplitude=provisional_amplitudes[curve_id],
            lifetime=lifetimes[curve_id],
            background=provisional_backgrounds[curve_id],
        )

        provisional_total = provisional_expected_counts.sum()

        if provisional_total <= 0:
            raise ValueError(
                "Generated expected curve has a non-positive total."
            )

        scale_factor = (
            target_photon_counts[curve_id] / provisional_total
        )

        amplitude_true = (
            provisional_amplitudes[curve_id] * scale_factor
        )
        background_true = (
            provisional_backgrounds[curve_id] * scale_factor
        )

        expected_counts = monoexponential_decay(
            time=time,
            amplitude=amplitude_true,
            lifetime=lifetimes[curve_id],
            background=background_true,
        )

        measured_counts = rng.poisson(expected_counts).astype(
            np.int64
        )

        X[curve_id] = measured_counts
        amplitudes_true[curve_id] = amplitude_true
        backgrounds_true[curve_id] = background_true
        expected_photon_counts[curve_id] = expected_counts.sum()
        measured_photon_counts[curve_id] = measured_counts.sum()

    metadata = pd.DataFrame(
        {
            "curve_id": np.arange(n_curves, dtype=np.int64),
            "lifetime_true": lifetimes,
            "amplitude_true": amplitudes_true,
            "background_true": backgrounds_true,
            "photon_count_target": target_photon_counts,
            "photon_count_expected": expected_photon_counts,
            "photon_count": measured_photon_counts,
        }
    )

    return SyntheticDataset(
        time=time,
        X=X,
        metadata=metadata,
    )


def generate_grouped_monoexponential_dataset(
    *,
    n_parameter_groups: int,
    n_realizations_per_group: int,
    time: FloatArray,
    lifetime_range: tuple[float, float],
    amplitude_range: tuple[float, float],
    background_range: tuple[float, float],
    photon_count_range: tuple[int, int],
    random_seed: int | None = None,
) -> SyntheticDataset:
    """
    Generate grouped Poisson-sampled mono-exponential TCSPC curves.

    Each parameter group represents one underlying physical parameter
    combination and one corresponding expected decay curve. Multiple
    independent Poisson realizations are sampled from that same expected
    curve.

    This structure is useful for studying data leakage and for performing
    group-aware train-test splitting.

    Parameters
    ----------
    n_parameter_groups:
        Number of independently sampled parameter combinations.
    n_realizations_per_group:
        Number of independent Poisson realizations generated from every
        parameter combination.
    time:
        One-dimensional time axis shared by all curves.
    lifetime_range:
        Minimum and maximum lifetime.
    amplitude_range:
        Minimum and maximum provisional signal amplitude.
    background_range:
        Minimum and maximum provisional background level.
    photon_count_range:
        Minimum and maximum target expected photon count.
    random_seed:
        Seed used for reproducible parameter generation and Poisson
        sampling.

    Returns
    -------
    SyntheticDataset
        Dataset containing measured count histograms and curve-level
        metadata.

    Notes
    -----
    Lifetime, amplitude, background, target photon count, and expected
    counts remain identical within each parameter group.

    Only the Poisson-sampled measured counts differ between realizations
    belonging to the same group.

    The sampled amplitude and background are initially relative values.
    Both are multiplied by the same scale factor so that the total
    expected photon count matches the sampled target photon count.
    """
    if n_parameter_groups <= 0:
        raise ValueError(
            "n_parameter_groups must be positive."
        )

    if n_realizations_per_group <= 0:
        raise ValueError(
            "n_realizations_per_group must be positive."
        )

    time = np.asarray(time, dtype=np.float64)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if time.size == 0:
        raise ValueError(
            "time must contain at least one value."
        )

    if not np.all(np.isfinite(time)):
        raise ValueError(
            "time must contain only finite values."
        )

    _validate_range(
        lifetime_range,
        "lifetime",
        allow_zero=False,
    )
    _validate_range(
        amplitude_range,
        "amplitude",
        allow_zero=False,
    )
    _validate_range(
        background_range,
        "background",
        allow_zero=True,
    )

    photon_count_min, photon_count_max = photon_count_range

    if photon_count_min <= 0:
        raise ValueError(
            "photon_count_range must contain positive values."
        )

    if photon_count_max < photon_count_min:
        raise ValueError(
            "photon_count_range upper bound must be greater than "
            "or equal to its lower bound."
        )

    rng = np.random.default_rng(random_seed)

    group_lifetimes = _sample_uniform(
        rng,
        lifetime_range,
        n_parameter_groups,
    )
    group_provisional_amplitudes = _sample_uniform(
        rng,
        amplitude_range,
        n_parameter_groups,
    )
    group_provisional_backgrounds = _sample_uniform(
        rng,
        background_range,
        n_parameter_groups,
    )

    group_target_photon_counts = rng.integers(
        photon_count_min,
        photon_count_max + 1,
        size=n_parameter_groups,
        dtype=np.int64,
    )

    n_curves = (
        n_parameter_groups
        * n_realizations_per_group
    )

    X = np.empty(
        (n_curves, time.size),
        dtype=np.int64,
    )

    curve_ids = np.arange(
        n_curves,
        dtype=np.int64,
    )
    parameter_groups = np.repeat(
        np.arange(
            n_parameter_groups,
            dtype=np.int64,
        ),
        n_realizations_per_group,
    )
    realization_ids = np.tile(
        np.arange(
            n_realizations_per_group,
            dtype=np.int64,
        ),
        n_parameter_groups,
    )

    lifetimes_true = np.empty(
        n_curves,
        dtype=np.float64,
    )
    amplitudes_true = np.empty(
        n_curves,
        dtype=np.float64,
    )
    backgrounds_true = np.empty(
        n_curves,
        dtype=np.float64,
    )
    target_photon_counts = np.empty(
        n_curves,
        dtype=np.int64,
    )
    expected_photon_counts = np.empty(
        n_curves,
        dtype=np.float64,
    )
    measured_photon_counts = np.empty(
        n_curves,
        dtype=np.int64,
    )

    curve_id = 0

    for parameter_group in range(n_parameter_groups):
        provisional_expected_counts = monoexponential_decay(
            time=time,
            amplitude=group_provisional_amplitudes[
                parameter_group
            ],
            lifetime=group_lifetimes[
                parameter_group
            ],
            background=group_provisional_backgrounds[
                parameter_group
            ],
        )

        provisional_total = (
            provisional_expected_counts.sum()
        )

        if provisional_total <= 0:
            raise ValueError(
                "Generated expected curve has a non-positive "
                "total."
            )

        scale_factor = (
            group_target_photon_counts[parameter_group]
            / provisional_total
        )

        amplitude_true = (
            group_provisional_amplitudes[
                parameter_group
            ]
            * scale_factor
        )
        background_true = (
            group_provisional_backgrounds[
                parameter_group
            ]
            * scale_factor
        )

        expected_counts = monoexponential_decay(
            time=time,
            amplitude=amplitude_true,
            lifetime=group_lifetimes[
                parameter_group
            ],
            background=background_true,
        )

        expected_total = expected_counts.sum()

        for _ in range(n_realizations_per_group):
            measured_counts = rng.poisson(
                expected_counts
            ).astype(np.int64)

            X[curve_id] = measured_counts

            lifetimes_true[curve_id] = (
                group_lifetimes[parameter_group]
            )
            amplitudes_true[curve_id] = (
                amplitude_true
            )
            backgrounds_true[curve_id] = (
                background_true
            )
            target_photon_counts[curve_id] = (
                group_target_photon_counts[
                    parameter_group
                ]
            )
            expected_photon_counts[curve_id] = (
                expected_total
            )
            measured_photon_counts[curve_id] = (
                measured_counts.sum()
            )

            curve_id += 1

    metadata = pd.DataFrame(
        {
            "curve_id": curve_ids,
            "parameter_group": parameter_groups,
            "realization_id": realization_ids,
            "lifetime_true": lifetimes_true,
            "amplitude_true": amplitudes_true,
            "background_true": backgrounds_true,
            "photon_count_target": target_photon_counts,
            "photon_count_expected": (
                expected_photon_counts
            ),
            "photon_count": measured_photon_counts,
        }
    )

    return SyntheticDataset(
        time=time,
        X=X,
        metadata=metadata,
    )