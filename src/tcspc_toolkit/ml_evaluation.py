from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.features import extract_feature_table
from tcspc_toolkit.simulation import (
    simulate_irf_convolved_histogram,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BenchmarkMeasurements:
    """Synthetic measurements used to construct benchmark inputs.

    Attributes
    ----------
    time:
        Shared TCSPC time axis.
    X_histograms:
        Raw Poisson-sampled histograms with shape
        ``(n_samples, n_time_bins)``.
    y:
        True fluorescence lifetime for every sample.
    metadata:
        Simulation nuisance variables and count diagnostics used
        for evaluation only.
    """

    time: FloatArray
    X_histograms: IntArray
    y: FloatArray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkDataset:
    """Common dataset used for lifetime-estimation benchmarking.

    Attributes
    ----------
    X_features:
        Engineered histogram features with shape
        ``(n_samples, n_features)``.
    X_histograms:
        Raw or explicitly preprocessed TCSPC histograms with shape
        ``(n_samples, n_time_bins)``.
    y:
        True fluorescence lifetime for each sample with shape
        ``(n_samples,)``.
    metadata:
        Simulation metadata used only for evaluation and
        stratification. One row corresponds to one sample.
    """

    X_features: pd.DataFrame
    X_histograms: IntArray
    y: FloatArray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkSplit:
    """Train/test split shared across all benchmark representations."""

    train_indices: IntArray
    test_indices: IntArray

    X_features_train: pd.DataFrame
    X_features_test: pd.DataFrame

    X_histograms_train: IntArray
    X_histograms_test: IntArray

    y_train: FloatArray
    y_test: FloatArray

    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame


@dataclass(frozen=True)
class RegressionMetrics:
    """Summary metrics for lifetime regression."""

    mae_ns: float
    median_absolute_error_ns: float
    mean_relative_error: float
    median_relative_error: float
    r2: float


def generate_benchmark_measurements(
    *,
    time: ArrayLike,
    lifetimes_ns: ArrayLike,
    signal_photon_counts: ArrayLike,
    background_levels: ArrayLike,
    irf_centre_ns: float,
    irf_fwhm_values_ns: ArrayLike,
    irf_shift_values_ns: ArrayLike,
    random_seed: int | None = 42,
) -> BenchmarkMeasurements:
    """Generate a structured IRF-convolved TCSPC benchmark.

    The benchmark contains every combination of lifetime, signal
    photon count, background level, IRF width, and IRF temporal shift.

    Parameters
    ----------
    time:
        Shared TCSPC time axis.
    lifetimes_ns:
        True fluorescence lifetimes included in the benchmark.
    signal_photon_counts:
        Target expected signal-photon counts.
    background_levels:
        Expected background counts per histogram bin.
    irf_centre_ns:
        Centre of the unshifted Gaussian IRF.
    irf_fwhm_values_ns:
        IRF full-width-at-half-maximum values.
    irf_shift_values_ns:
        Temporal IRF shifts.
    random_seed:
        Seed controlling Poisson sampling.

    Returns
    -------
    BenchmarkMeasurements
        Aligned raw histograms, lifetime targets, and simulation
        metadata.
    """
    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    lifetimes_array = np.asarray(
        lifetimes_ns,
        dtype=np.float64,
    )

    photon_counts_array = np.asarray(
        signal_photon_counts,
    )

    background_array = np.asarray(
        background_levels,
        dtype=np.float64,
    )

    irf_fwhm_array = np.asarray(
        irf_fwhm_values_ns,
        dtype=np.float64,
    )

    irf_shift_array = np.asarray(
        irf_shift_values_ns,
        dtype=np.float64,
    )

    parameter_arrays = {
        "lifetimes_ns": lifetimes_array,
        "signal_photon_counts": photon_counts_array,
        "background_levels": background_array,
        "irf_fwhm_values_ns": irf_fwhm_array,
        "irf_shift_values_ns": irf_shift_array,
    }

    for name, values in parameter_arrays.items():
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                f"{name} must contain at least one value."
            )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    if time_array.size < 2:
        raise ValueError(
            "time must contain at least two values."
        )

    if not np.issubdtype(
        photon_counts_array.dtype,
        np.integer,
    ):
        raise ValueError(
            "signal_photon_counts must contain integers."
        )

    photon_counts_array = photon_counts_array.astype(
        np.int64,
        copy=False,
    )

    n_samples = (
        lifetimes_array.size
        * photon_counts_array.size
        * background_array.size
        * irf_fwhm_array.size
        * irf_shift_array.size
    )

    X_histograms = np.empty(
        (
            n_samples,
            time_array.size,
        ),
        dtype=np.int64,
    )

    y = np.empty(
        n_samples,
        dtype=np.float64,
    )

    metadata_rows: list[
        dict[str, float | int]
    ] = []

    rng = np.random.default_rng(
        random_seed
    )

    parameter_grid = product(
        lifetimes_array,
        photon_counts_array,
        background_array,
        irf_fwhm_array,
        irf_shift_array,
    )

    for sample_id, parameters in enumerate(
        parameter_grid
    ):
        (
            lifetime_ns,
            signal_photon_count,
            background_per_bin,
            irf_fwhm_ns,
            irf_shift_ns,
        ) = parameters

        measured_counts, simulation_metadata = (
            simulate_irf_convolved_histogram(
                time=time_array,
                lifetime_ns=float(
                    lifetime_ns
                ),
                signal_photon_count=int(
                    signal_photon_count
                ),
                background_per_bin=float(
                    background_per_bin
                ),
                irf_centre_ns=irf_centre_ns,
                irf_fwhm_ns=float(
                    irf_fwhm_ns
                ),
                irf_shift_ns=float(
                    irf_shift_ns
                ),
                rng=rng,
            )
        )

        X_histograms[
            sample_id
        ] = measured_counts

        y[
            sample_id
        ] = float(
            lifetime_ns
        )

        metadata_rows.append(
            {
                "sample_id": sample_id,
                "signal_photon_count_target": int(
                    signal_photon_count
                ),
                "background_per_bin": float(
                    background_per_bin
                ),
                "irf_fwhm_ns": float(
                    irf_fwhm_ns
                ),
                "irf_shift_ns": float(
                    irf_shift_ns
                ),
                "expected_signal_counts": simulation_metadata[
                    "expected_signal_counts"
                ],
                "expected_background_counts": simulation_metadata[
                    "expected_background_counts"
                ],
                "expected_total_counts": simulation_metadata[
                    "expected_total_counts"
                ],
                "measured_total_counts": simulation_metadata[
                    "measured_total_counts"
                ],
            }
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    return BenchmarkMeasurements(
        time=time_array,
        X_histograms=X_histograms,
        y=y,
        metadata=metadata,
    )


def build_benchmark_dataset(
    measurements: BenchmarkMeasurements,
    *,
    feature_config: FeatureConfig,
) -> BenchmarkDataset:
    """Build aligned ML benchmark representations from measurements.

    Parameters
    ----------
    measurements:
        Simulated TCSPC measurements containing raw histograms,
        lifetime targets, metadata, and the shared time axis.
    feature_config:
        Configuration used to extract physically engineered
        histogram features.

    Returns
    -------
    BenchmarkDataset
        Aligned engineered features, raw histograms, lifetime
        targets, and evaluation metadata.

    Notes
    -----
    Engineered features are extracted directly from the same raw
    histograms stored in ``X_histograms``. Simulation metadata is
    kept separate and is not passed into feature extraction.
    """
    if measurements.time.ndim != 1:
        raise ValueError(
            "measurements.time must be one-dimensional."
        )

    if measurements.X_histograms.ndim != 2:
        raise ValueError(
            "measurements.X_histograms must be two-dimensional."
        )

    n_samples, n_time_bins = (
        measurements.X_histograms.shape
    )

    if n_time_bins != measurements.time.size:
        raise ValueError(
            "Histogram bin count must match the time axis."
        )

    if measurements.y.shape != (n_samples,):
        raise ValueError(
            "y must contain exactly one target per histogram."
        )

    if measurements.metadata.shape[0] != n_samples:
        raise ValueError(
            "metadata must contain exactly one row per histogram."
        )

    X_features = extract_feature_table(
        histograms=measurements.X_histograms,
        time=measurements.time,
        config=feature_config,
    )

    if X_features.shape[0] != n_samples:
        raise RuntimeError(
            "Feature extraction did not preserve sample count."
        )

    return BenchmarkDataset(
        X_features=X_features,
        X_histograms=measurements.X_histograms.copy(),
        y=measurements.y.copy(),
        metadata=measurements.metadata.copy(
            deep=True
        ),
    )


def split_benchmark_dataset(
    dataset: BenchmarkDataset,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BenchmarkSplit:
    """Split all benchmark representations using common sample indices.

    Parameters
    ----------
    dataset:
        Benchmark dataset containing aligned representations,
        targets, and metadata.
    test_size:
        Fraction of samples assigned to the test set.
    random_state:
        Seed controlling the reproducible train/test split.

    Returns
    -------
    BenchmarkSplit
        Train/test subsets with identical sample membership across
        histogram representations, engineered features, targets,
        and metadata.
    """
    n_samples = dataset.y.shape[0]

    if dataset.X_features.shape[0] != n_samples:
        raise ValueError(
            "X_features and y must contain the same number of samples."
        )

    if dataset.X_histograms.shape[0] != n_samples:
        raise ValueError(
            "X_histograms and y must contain the same number of samples."
        )

    if dataset.metadata.shape[0] != n_samples:
        raise ValueError(
            "metadata and y must contain the same number of samples."
        )

    sample_indices = np.arange(
        n_samples,
        dtype=np.int64,
    )

    train_indices, test_indices = train_test_split(
        sample_indices,
        test_size=test_size,
        random_state=random_state,
    )

    return BenchmarkSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        X_features_train=dataset.X_features.iloc[
            train_indices
        ].reset_index(drop=True),
        X_features_test=dataset.X_features.iloc[
            test_indices
        ].reset_index(drop=True),
        X_histograms_train=dataset.X_histograms[
            train_indices
        ],
        X_histograms_test=dataset.X_histograms[
            test_indices
        ],
        y_train=dataset.y[
            train_indices
        ],
        y_test=dataset.y[
            test_indices
        ],
        metadata_train=dataset.metadata.iloc[
            train_indices
        ].reset_index(drop=True),
        metadata_test=dataset.metadata.iloc[
            test_indices
        ].reset_index(drop=True),
    )


def _split_parameter_values(
    split: BenchmarkSplit,
) -> dict[
    str,
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
]:
    """Return train/test values for benchmark target and nuisance variables."""
    return {
        "lifetime_ns": (
            np.asarray(
                split.y_train,
                dtype=np.float64,
            ),
            np.asarray(
                split.y_test,
                dtype=np.float64,
            ),
        ),
        "signal_photon_count_target": (
            split.metadata_train[
                "signal_photon_count_target"
            ].to_numpy(dtype=np.float64),
            split.metadata_test[
                "signal_photon_count_target"
            ].to_numpy(dtype=np.float64),
        ),
        "background_per_bin": (
            split.metadata_train[
                "background_per_bin"
            ].to_numpy(dtype=np.float64),
            split.metadata_test[
                "background_per_bin"
            ].to_numpy(dtype=np.float64),
        ),
        "irf_fwhm_ns": (
            split.metadata_train[
                "irf_fwhm_ns"
            ].to_numpy(dtype=np.float64),
            split.metadata_test[
                "irf_fwhm_ns"
            ].to_numpy(dtype=np.float64),
        ),
        "irf_shift_ns": (
            split.metadata_train[
                "irf_shift_ns"
            ].to_numpy(dtype=np.float64),
            split.metadata_test[
                "irf_shift_ns"
            ].to_numpy(dtype=np.float64),
        ),
    }


def summarize_split_coverage(
    split: BenchmarkSplit,
) -> pd.DataFrame:
    """Summarize train/test coverage of benchmark parameter space.

    Parameters
    ----------
    split:
        Benchmark train/test split.

    Returns
    -------
    pandas.DataFrame
        One row per target or nuisance variable containing train/test
        ranges, means, numbers of unique values, and whether both
        subsets contain exactly the same parameter levels.

    Notes
    -----
    Matching minimum and maximum values alone do not guarantee that
    train and test cover the same parameter support. Therefore,
    ``same_support`` compares the complete set of unique values.
    """
    summary_rows: list[
        dict[str, float | int | bool | str]
    ] = []

    parameter_values = _split_parameter_values(
        split
    )

    for parameter_name, (
        train_values,
        test_values,
    ) in parameter_values.items():
        train_unique = np.unique(
            train_values
        )

        test_unique = np.unique(
            test_values
        )

        same_support = np.array_equal(
            train_unique,
            test_unique,
        )

        summary_rows.append(
            {
                "parameter": parameter_name,
                "train_min": float(
                    np.min(train_values)
                ),
                "train_max": float(
                    np.max(train_values)
                ),
                "test_min": float(
                    np.min(test_values)
                ),
                "test_max": float(
                    np.max(test_values)
                ),
                "train_mean": float(
                    np.mean(train_values)
                ),
                "test_mean": float(
                    np.mean(test_values)
                ),
                "train_n_unique": int(
                    train_unique.size
                ),
                "test_n_unique": int(
                    test_unique.size
                ),
                "same_support": bool(
                    same_support
                ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


def summarize_split_level_balance(
    split: BenchmarkSplit,
) -> pd.DataFrame:
    """Summarize train/test sample balance at each parameter level.

    Parameters
    ----------
    split:
        Benchmark train/test split.

    Returns
    -------
    pandas.DataFrame
        Long-format table containing one row per parameter level,
        with train/test counts and fractions.
    """
    balance_rows: list[
        dict[str, float | int | str]
    ] = []

    parameter_values = _split_parameter_values(
        split
    )

    for parameter_name, (
        train_values,
        test_values,
    ) in parameter_values.items():
        levels = np.union1d(
            train_values,
            test_values,
        )

        for level in levels:
            train_count = int(
                np.count_nonzero(
                    np.isclose(
                        train_values,
                        level,
                    )
                )
            )

            test_count = int(
                np.count_nonzero(
                    np.isclose(
                        test_values,
                        level,
                    )
                )
            )

            balance_rows.append(
                {
                    "parameter": parameter_name,
                    "value": float(level),
                    "train_count": train_count,
                    "test_count": test_count,
                    "train_fraction": (
                        train_count
                        / train_values.size
                    ),
                    "test_fraction": (
                        test_count
                        / test_values.size
                    ),
                }
            )

    return pd.DataFrame(
        balance_rows
    )


def evaluate_regression(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> RegressionMetrics:
    """Evaluate predicted lifetimes."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if np.any(y_true <= 0):
        raise ValueError(
            "True lifetimes must be strictly positive."
        )

    relative_errors = (
        np.abs(y_pred - y_true) / y_true
    )

    return RegressionMetrics(
        mae_ns=float(
            mean_absolute_error(y_true, y_pred)
        ),
        median_absolute_error_ns=float(
            median_absolute_error(y_true, y_pred)
        ),
        mean_relative_error=float(
            np.mean(relative_errors)
        ),
        median_relative_error=float(
            np.median(relative_errors)
        ),
        r2=float(
            r2_score(y_true, y_pred)
        ),
    )
