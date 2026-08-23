import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.irf import (
    generate_gaussian_irf,
    normalize_irf,
    shift_irf,
)
from tcspc_toolkit.models import (
    biexponential_decay,
    monoexponential_decay,
    multiexponential_decay,
)


def sample_photon_counts(
    expected_counts: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample photon counts using Poisson statistics."""
    if np.any(expected_counts < 0):
        raise ValueError("expected counts must be non-negative")

    return rng.poisson(expected_counts)


def simulate_irf_convolved_histogram(
    *,
    time: NDArray[np.float64],
    lifetime_ns: float,
    signal_photon_count: int,
    background_per_bin: float,
    irf_centre_ns: float,
    irf_fwhm_ns: float,
    irf_shift_ns: float,
    rng: np.random.Generator,
) -> tuple[
    NDArray[np.int64],
    dict[str, float],
]:
    """Simulate one IRF-convolved Poisson TCSPC histogram.

    Parameters
    ----------
    time:
        Shared time-bin coordinates.
    lifetime_ns:
        True mono-exponential fluorescence lifetime.
    signal_photon_count:
        Target expected number of signal photons within the
        measurement window.
    background_per_bin:
        Expected stationary background counts per histogram bin.
    irf_centre_ns:
        Temporal centre of the unshifted Gaussian IRF.
    irf_fwhm_ns:
        Gaussian IRF full width at half maximum.
    irf_shift_ns:
        Temporal shift applied to the IRF.
    rng:
        NumPy random-number generator used for Poisson sampling.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, float]]
        Poisson-sampled measured histogram and simulation metadata.

    Notes
    -----
    The stationary background is added after convolution because it
    should not itself be convolved with the instrument response.

    The convolved fluorescence signal is scaled so that its discrete
    sum equals ``signal_photon_count`` before background is added.
    """
    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    if not np.isfinite(lifetime_ns):
        raise ValueError(
            "lifetime_ns must be finite."
        )

    if lifetime_ns <= 0.0:
        raise ValueError(
            "lifetime_ns must be positive."
        )

    if (
        isinstance(signal_photon_count, (bool, np.bool_))
        or not isinstance(
            signal_photon_count,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "signal_photon_count must be an integer."
        )

    if signal_photon_count <= 0:
        raise ValueError(
            "signal_photon_count must be positive."
        )

    if not np.isfinite(background_per_bin):
        raise ValueError(
            "background_per_bin must be finite."
        )

    if background_per_bin < 0.0:
        raise ValueError(
            "background_per_bin must be non-negative."
        )

    ideal_decay = monoexponential_decay(
        time=time_array,
        amplitude=1.0,
        lifetime=lifetime_ns,
        background=0.0,
    )

    irf = generate_gaussian_irf(
        time=time_array,
        centre=irf_centre_ns,
        fwhm=irf_fwhm_ns,
    )

    irf = normalize_irf(
        time=time_array,
        irf=irf,
    )

    irf = shift_irf(
        time=time_array,
        irf=irf,
        shift=irf_shift_ns,
    )

    # Interpolation on a finite measurement window can slightly
    # change the integrated IRF area after shifting.
    irf = normalize_irf(
        time=time_array,
        irf=irf,
    )

    convolved_signal = convolve_decay_with_irf(
        time=time_array,
        decay=ideal_decay,
        irf=irf,
    )

    signal_sum = float(
        convolved_signal.sum()
    )

    if (
        not np.isfinite(signal_sum)
        or signal_sum <= 0.0
    ):
        raise ValueError(
            "Convolved signal must have a positive finite "
            "discrete sum."
        )

    expected_signal = (
        float(signal_photon_count)
        * convolved_signal
        / signal_sum
    )

    expected_counts = (
        expected_signal
        + background_per_bin
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    simulation_metadata = {
        "lifetime_true_ns": float(lifetime_ns),
        "signal_photon_count_target": float(
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
        "expected_signal_counts": float(
            expected_signal.sum()
        ),
        "expected_background_counts": float(
            background_per_bin
            * time_array.size
        ),
        "expected_total_counts": float(
            expected_counts.sum()
        ),
        "measured_total_counts": float(
            measured_counts.sum()
        ),
    }

    return (
        measured_counts,
        simulation_metadata,
    )


def simulate_monoexponential_decay(
    time: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate expected and Poisson-sampled mono-exponential decay curves."""
    rng = np.random.default_rng(random_seed)

    expected_counts = monoexponential_decay(
        time=time,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    return expected_counts, measured_counts


def simulate_biexponential_decay(
    time: NDArray[np.float64],
    amplitude_1: float,
    lifetime_1: float,
    amplitude_2: float,
    lifetime_2: float,
    background: float = 0.0,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate expected and Poisson-sampled bi-exponential decay curves.

    Parameters
    ----------
    time:
        Time axis.
    amplitude_1:
        Amplitude of the first exponential component.
    lifetime_1:
        Lifetime of the first exponential component.
    amplitude_2:
        Amplitude of the second exponential component.
    lifetime_2:
        Lifetime of the second exponential component.
    background:
        Constant background level.
    random_seed:
        Seed used for reproducible Poisson sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Expected photon counts and Poisson-sampled measured counts.
    """
    rng = np.random.default_rng(random_seed)

    expected_counts = biexponential_decay(
        time=time,
        amplitude_1=amplitude_1,
        lifetime_1=lifetime_1,
        amplitude_2=amplitude_2,
        lifetime_2=lifetime_2,
        background=background,
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    return expected_counts, measured_counts


def simulate_multiexponential_decay(
    time: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    lifetimes: NDArray[np.float64],
    background: float = 0.0,
    random_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate expected and Poisson-sampled multi-exponential decay curves.

    Parameters
    ----------
    time:
        One-dimensional time axis with shape ``(n_time_bins,)``.
    amplitudes:
        One-dimensional array of component amplitudes with shape
        ``(n_components,)``.
    lifetimes:
        One-dimensional array of component lifetimes with shape
        ``(n_components,)``.
    background:
        Constant background level.
    random_seed:
        Seed used for reproducible Poisson sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Expected photon counts and Poisson-sampled measured counts.
    """
    rng = np.random.default_rng(random_seed)

    expected_counts = multiexponential_decay(
        time=time,
        amplitudes=amplitudes,
        lifetimes=lifetimes,
        background=background,
    )

    measured_counts = sample_photon_counts(
        expected_counts=expected_counts,
        rng=rng,
    )

    return expected_counts, measured_counts