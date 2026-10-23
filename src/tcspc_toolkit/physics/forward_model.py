"""Deterministic expected counts for IRF-aware TCSPC reconvolution."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.physics.convolution import convolve_decay_with_irf
from tcspc_toolkit.irf.profiles import shift_irf
from tcspc_toolkit.physics.models import monoexponential_decay


def monoexponential_reconvolution_expected_counts(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float,
    temporal_shift: float,
) -> NDArray[np.float64]:
    """Return expected counts for the existing mono-exponential reconvolution model.

    ``irf`` is a non-negative, already normalized kernel on ``time``; its
    source and preparation are outside this function. ``temporal_shift`` uses
    the same time units as ``time`` and moves the IRF later when positive.
    Shifting does not renormalize the IRF. Convolution retains its bin-width
    factor and truncates to the supplied time window, then background is added.

    ``amplitude`` is the scale of the convolved unit-amplitude decay, not the
    total expected signal photons in the finite window.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    if lifetime <= 0:
        raise ValueError("lifetime must be positive")

    if background < 0:
        raise ValueError("background must be non-negative")

    decay = monoexponential_decay(
        time=time,
        amplitude=1.0,
        lifetime=lifetime,
        background=0.0,
    )

    shifted_irf = shift_irf(
        time=time,
        irf=irf,
        shift=temporal_shift,
    )

    convolved = convolve_decay_with_irf(
        time=time,
        decay=decay,
        irf=shifted_irf,
    )

    expected_counts = amplitude * convolved + background

    return expected_counts


def multiexponential_reconvolution_component_counts(
    time: ArrayLike,
    irf: ArrayLike,
    *,
    amplitudes: ArrayLike,
    lifetimes: ArrayLike,
    temporal_shift: float = 0.0,
) -> NDArray[np.float64]:
    """Return background-free component counts with shape ``(K, n_bins)``.

    Components retain the supplied order, including zero-amplitude and
    equal-lifetime components. Each row uses the existing mono-exponential
    forward model with one shared IRF shift. Positive shift moves the IRF
    later; shifting does not renormalize it. Time and lifetimes use the same
    units (normally ns). The supplied time origin is not changed.

    The IRF must already be prepared on the uniform time grid and normalized
    by area. Preparation, normalization and provenance remain caller-owned.
    The existing convolution bin-width factor and finite-window truncation
    apply separately to each component. ``amplitudes`` are pre-exponential
    scales, not expected photon totals or photon fractions.
    """
    time = np.asarray(time, dtype=np.float64)
    irf = np.asarray(irf, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    lifetimes = np.asarray(lifetimes, dtype=np.float64)

    if amplitudes.ndim != 1 or lifetimes.ndim != 1:
        raise ValueError("amplitudes and lifetimes must be one-dimensional")
    if amplitudes.size == 0:
        raise ValueError("at least one exponential component is required")
    if amplitudes.shape != lifetimes.shape:
        raise ValueError("amplitudes and lifetimes must have the same shape")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("amplitudes must be finite and non-negative")
    if not np.all(np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0):
        raise ValueError("lifetimes must be finite and positive")
    if not np.isfinite(temporal_shift):
        raise ValueError("temporal_shift must be finite")

    return np.stack(
        [
            monoexponential_reconvolution_expected_counts(
                time, irf, amplitude, lifetime, 0.0, temporal_shift
            )
            for amplitude, lifetime in zip(amplitudes, lifetimes, strict=True)
        ]
    )


def multiexponential_reconvolution_expected_counts(
    time: ArrayLike,
    irf: ArrayLike,
    *,
    amplitudes: ArrayLike,
    lifetimes: ArrayLike,
    background: float = 0.0,
    temporal_shift: float = 0.0,
) -> NDArray[np.float64]:
    """Return the summed component counts plus one constant background.

    ``background`` is the expected background count per bin, added once after
    convolution. All other conventions are those of
    :func:`multiexponential_reconvolution_component_counts`. For one component
    the result agrees exactly with the existing mono-exponential forward
    model; no fitter or noise sampling is performed here.
    """
    if not np.isfinite(background) or background < 0.0:
        raise ValueError("background must be finite and non-negative")
    components = multiexponential_reconvolution_component_counts(
        time,
        irf,
        amplitudes=amplitudes,
        lifetimes=lifetimes,
        temporal_shift=temporal_shift,
    )
    return components.sum(axis=0) + background
