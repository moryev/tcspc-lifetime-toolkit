"""Instrument response function (IRF) generation and manipulation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import exponnorm


def generate_gaussian_irf(
    time: NDArray[np.float64],
    centre: float,
    fwhm: float,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a Gaussian instrument response function.

    Parameters
    ----------
    time
        One-dimensional array containing the time-bin coordinates.
    centre
        Temporal centre of the Gaussian IRF.
    fwhm
        Full width at half maximum of the Gaussian IRF.
        Must be finite and strictly positive.
    amplitude
        Peak amplitude of the Gaussian IRF. Must be finite and
        non-negative.

    Returns
    -------
    numpy.ndarray
        Gaussian IRF evaluated at the supplied time coordinates.
        The returned array has the same shape as ``time``.

    Raises
    ------
    ValueError
        If ``time`` is not one-dimensional, is empty, or contains
        non-finite values.
        If ``centre`` is not finite.
        If ``fwhm`` is not finite or is not strictly positive.
        If ``amplitude`` is not finite or is negative.

    Notes
    -----
    This function does not normalize the IRF. The ``amplitude`` parameter
    specifies its peak height, not its discrete sum or continuous area.
    """

    time_array = np.asarray(time, dtype=np.float64)

    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")

    if time_array.size == 0:
        raise ValueError("time must not be empty.")

    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values.")

    if not np.isfinite(centre):
        raise ValueError("centre must be finite.")

    if not np.isfinite(fwhm):
        raise ValueError("fwhm must be finite.")

    if fwhm <= 0.0:
        raise ValueError("fwhm must be greater than zero.")

    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite.")

    if amplitude < 0.0:
        raise ValueError("amplitude must be non-negative.")

    # Compute standard deviation for the given Gaussian IRF
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    exponent = -0.5 * ((time_array - centre) / sigma) ** 2

    return amplitude * np.exp(exponent)


def normalize_irf(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize an instrument response function to unit integrated area.

    The IRF is normalized according to

        integral IRF(t) dt = 1,

    where the integral is approximated numerically using the trapezoidal
    rule over the supplied time axis.

    Parameters
    ----------
    time
        One-dimensional, strictly increasing time axis.
    irf
        One-dimensional array containing non-negative IRF values.

    Returns
    -------
    NDArray[np.float64]
        A new IRF array with unit integrated area.

    Raises
    ------
    ValueError
        If either input is not one-dimensional, if their lengths differ,
        if they contain non-finite values, if the time axis is not strictly
        increasing, if the IRF contains negative values, or if the
        integrated IRF area is not positive.
    """
    time_array = np.asarray(time, dtype=np.float64)
    irf_array = np.asarray(irf, dtype=np.float64)

    if time_array.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if irf_array.ndim != 1:
        raise ValueError("irf must be one-dimensional.")

    if time_array.shape != irf_array.shape:
        raise ValueError("time and irf must have the same shape.")

    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(irf_array)):
        raise ValueError("irf must contain only finite values.")

    if np.any(irf_array < 0.0):
        raise ValueError("irf values must be non-negative.")

    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing.")

    area = np.trapezoid(irf_array, x=time_array)

    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("irf must have a positive integrated area.")

    return irf_array / area


from numpy.typing import NDArray
import numpy as np


def shift_irf(
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    shift: float,
) -> NDArray[np.float64]:
    """Shift an instrument response function along its time axis.

    The shifted IRF is defined as

        shifted_irf(t) = irf(t - shift)

    so that a positive shift moves the IRF toward later times and a
    negative shift moves it toward earlier times.

    Linear interpolation is used, which permits shifts that are not
    integer multiples of the time-bin width. Values outside the supplied
    time interval are replaced with zero.

    The shifted IRF is not renormalized. If part of the IRF moves outside
    the observed time window, its integrated area will therefore decrease.

    Parameters
    ----------
    time
        One-dimensional, strictly increasing time axis.
    irf
        One-dimensional instrument response function evaluated at `time`.
    shift
        Temporal shift in the same units as `time`.

    Returns
    -------
    NDArray[np.float64]
        Shifted IRF with the same shape as the input IRF.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional, have different lengths,
        contain non-finite values, the time axis is not strictly increasing,
        the IRF contains negative values, or the shift is not finite.
    """
    time = np.asarray(time, dtype=np.float64)
    irf = np.asarray(irf, dtype=np.float64)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if irf.ndim != 1:
        raise ValueError("irf must be one-dimensional.")

    if time.shape != irf.shape:
        raise ValueError("time and irf must have the same shape.")

    if time.size < 2:
        raise ValueError("time and irf must contain at least two values.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    if not np.all(np.isfinite(irf)):
        raise ValueError("irf must contain only finite values.")

    if not np.isfinite(shift):
        raise ValueError("shift must be finite.")

    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be strictly increasing.")

    if np.any(irf < 0.0):
        raise ValueError("irf values must be non-negative.")

    return np.interp(
        time - shift,
        time,
        irf,
        left=0.0,
        right=0.0,
    )


class IRFSourceKind(str, Enum):
    """Origin of a sampled IRF, independent of how it is later used."""

    SYNTHETIC_GAUSSIAN = "synthetic_gaussian"
    SYNTHETIC_EMG = "synthetic_emg"
    IMPORTED_SAMPLED = "imported_sampled"
    LEADING_EDGE_ESTIMATE = "leading_edge_estimate"


def _copy_irf_mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Follow the Issue-#6 carrier's copied, top-level read-only convention."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied = deepcopy(dict(values))
    if any(not isinstance(key, str) for key in copied):
        raise ValueError(f"{name} keys must be strings")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class IRFProfile:
    """A sampled source IRF in ns, retaining its original numerical scale.

    Source grids may be nonuniform. Arrays are copied and read-only; mappings
    are deep-copied and protected at the top level, as for ``SampledIRF``.
    Nested mapping values are not recursively frozen. Preparation history is
    recorded separately by ``prepare_irf``.
    """

    time_ns: NDArray[np.float64]
    values: NDArray[np.float64]
    source_kind: IRFSourceKind
    source_parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, IRFSourceKind):
            raise ValueError("source_kind must be an IRFSourceKind")
        try:
            time = np.asarray(self.time_ns, dtype=np.float64)
            values = np.asarray(self.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("IRF time and values must be numeric") from exc
        if time.ndim != 1 or time.size < 2:
            raise ValueError("IRF time must be one-dimensional with at least two values")
        if values.ndim != 1 or values.shape != time.shape:
            raise ValueError("IRF values must match the one-dimensional time grid")
        # Existing normalization validation also accepts strictly increasing
        # nonuniform grids. Discard its derived output: this is a source trace.
        normalize_irf(time=time, irf=values)
        time_copy = np.array(time, dtype=np.float64, copy=True)
        values_copy = np.array(values, dtype=np.float64, copy=True)
        time_copy.setflags(write=False)
        values_copy.setflags(write=False)
        object.__setattr__(self, "time_ns", time_copy)
        object.__setattr__(self, "values", values_copy)
        object.__setattr__(
            self,
            "source_parameters",
            _copy_irf_mapping(self.source_parameters, "source_parameters"),
        )
        object.__setattr__(self, "metadata", _copy_irf_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "provenance", _copy_irf_mapping(self.provenance, "provenance"))


def generate_emg_irf(
    time: NDArray[np.float64],
    gaussian_centre_ns: float,
    gaussian_fwhm_ns: float,
    tail_time_ns: float,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Sample a Gaussian convolved with a unit-area causal exponential.

    The Gaussian component has the given centre, FWHM, and peak amplitude.
    Its component centre/FWHM are generally not the EMG peak/full FWHM.
    The exponential tail time is in ns. This function does not normalize the
    sampled finite-window area; use ``prepare_irf`` for a forward kernel.
    """
    if not np.isfinite(tail_time_ns) or tail_time_ns < 0.0:
        raise ValueError("tail_time_ns must be finite and non-negative")
    if tail_time_ns == 0.0:
        return generate_gaussian_irf(time, gaussian_centre_ns, gaussian_fwhm_ns, amplitude)

    time_array = np.asarray(time, dtype=np.float64)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("time must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values")
    if not np.isfinite(gaussian_centre_ns):
        raise ValueError("gaussian_centre_ns must be finite")
    if not np.isfinite(gaussian_fwhm_ns) or gaussian_fwhm_ns <= 0.0:
        raise ValueError("gaussian_fwhm_ns must be finite and greater than zero")
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and non-negative")
    if amplitude == 0.0:
        return np.zeros_like(time_array)

    sigma_ns = gaussian_fwhm_ns / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    shape = tail_time_ns / sigma_ns
    if not np.isfinite(sigma_ns) or sigma_ns <= 0.0 or not np.isfinite(shape):
        raise ValueError("Gaussian width and tail time exceed numerical range")
    # SciPy also evaluates 1/K. A nonzero subnormal K can overflow that
    # reciprocal and silently produce zeros in the scaled-erfc calculation.
    with np.errstate(over="ignore", divide="ignore"):
        reciprocal_shape = np.divide(1.0, shape)
    if not np.isfinite(reciprocal_shape):
        raise ValueError("tail_time_ns is below numerical resolution relative to Gaussian width")

    log_scale = np.log(amplitude) + np.log(sigma_ns) + 0.5 * np.log(2.0 * np.pi)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_density = exponnorm.logpdf(
            time_array, K=shape, loc=gaussian_centre_ns, scale=sigma_ns
        )
        values = np.exp(log_density + log_scale)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("EMG evaluation produced invalid values")
    return values


def _generator_provenance(
    provenance: Mapping[str, Any] | None,
    generator: str,
) -> Mapping[str, Any]:
    """Record the generator without silently replacing caller provenance."""
    if provenance is None:
        copied = {}
    elif isinstance(provenance, Mapping):
        copied = dict(provenance)
    else:
        raise ValueError("provenance must be a mapping")
    if "generator" in copied and copied["generator"] != generator:
        raise ValueError("provenance generator conflicts with the IRF factory")
    return {**copied, "generator": generator}


def generate_gaussian_irf_profile(
    time_ns: ArrayLike,
    *,
    gaussian_centre_ns: float,
    gaussian_fwhm_ns: float,
    amplitude: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> IRFProfile:
    """Create a Gaussian sampled source with explicit ns parameters."""
    values = generate_gaussian_irf(
        time_ns, gaussian_centre_ns, gaussian_fwhm_ns, amplitude
    )
    return IRFProfile(
        time_ns=time_ns,
        values=values,
        source_kind=IRFSourceKind.SYNTHETIC_GAUSSIAN,
        source_parameters={
            "gaussian_centre_ns": gaussian_centre_ns,
            "gaussian_fwhm_ns": gaussian_fwhm_ns,
            "amplitude": amplitude,
        },
        metadata={} if metadata is None else metadata,
        provenance=_generator_provenance(provenance, "generate_gaussian_irf"),
    )


def generate_emg_irf_profile(
    time_ns: ArrayLike,
    *,
    gaussian_centre_ns: float,
    gaussian_fwhm_ns: float,
    tail_time_ns: float,
    amplitude: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> IRFProfile:
    """Create an EMG sampled source with explicit component parameters in ns."""
    values = generate_emg_irf(
        time_ns, gaussian_centre_ns, gaussian_fwhm_ns, tail_time_ns, amplitude
    )
    return IRFProfile(
        time_ns=time_ns,
        values=values,
        source_kind=IRFSourceKind.SYNTHETIC_EMG,
        source_parameters={
            "gaussian_centre_ns": gaussian_centre_ns,
            "gaussian_fwhm_ns": gaussian_fwhm_ns,
            "tail_time_ns": tail_time_ns,
            "amplitude": amplitude,
        },
        metadata={} if metadata is None else metadata,
        provenance=_generator_provenance(provenance, "generate_emg_irf"),
    )
