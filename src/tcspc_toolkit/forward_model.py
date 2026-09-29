"""Deterministic expected counts for IRF-aware TCSPC reconvolution."""

import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.irf import shift_irf
from tcspc_toolkit.models import monoexponential_decay


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
