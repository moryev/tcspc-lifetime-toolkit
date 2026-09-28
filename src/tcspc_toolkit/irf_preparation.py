"""Explicit preparation of sampled IRF sources for forward-model grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.irf import IRFProfile, IRFSourceKind, normalize_irf
from tcspc_toolkit.measurements import SampledIRF
from tcspc_toolkit.preprocessing import time_axes_compatible, validate_time_axis


FloatArray = NDArray[np.float64]
ResamplingMethod = Literal["none", "linear"]


def irf_profile_from_sampled_irf(sampled_irf: SampledIRF) -> IRFProfile:
    """Snapshot an imported sampled trace without changing its scale or grid.

    Imported samples need not be independently measured physical responses.
    The original ``SampledIRF`` remains the Issue-#6 measurement carrier.
    """
    if not isinstance(sampled_irf, SampledIRF):
        raise TypeError("sampled_irf must be a SampledIRF")
    return IRFProfile(
        time_ns=sampled_irf.time_ns,
        values=sampled_irf.values,
        source_kind=IRFSourceKind.IMPORTED_SAMPLED,
        metadata=sampled_irf.metadata,
        provenance=sampled_irf.provenance,
    )


@dataclass(frozen=True)
class IRFPreparationDiagnostics:
    """Numerical preparation facts about the available sampled source trace.

    Support loss is geometric, relative only to the supplied source window.
    ``target_minus_retained_area`` exposes separate target-grid quadrature
    effects. The normalization factor multiplies target samples to unit area.
    No field certifies that unrecorded physical IRF tails were negligible.
    """

    source_area: float
    retained_source_area: float
    support_retained_fraction: float
    support_loss_fraction: float
    target_area_before_normalization: float
    target_area_after_normalization: float
    normalization_factor: float
    target_minus_retained_area: float
    source_spacing_min_ns: float
    source_spacing_max_ns: float
    target_spacing_ns: float
    source_fwhm_ns: float | None
    target_fwhm_ns: float | None
    target_bins_per_fwhm: float | None
    registration_offset_ns: float
    resampling_method: ResamplingMethod
    operations: tuple[str, ...]
    flags: tuple[str, ...]
    max_support_loss_fraction: float | None


@dataclass(frozen=True)
class PreparedIRF:
    """A derived unit-area kernel on a uniform ns forward-model grid."""

    source: IRFProfile
    time_ns: FloatArray
    kernel: FloatArray
    diagnostics: IRFPreparationDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.source, IRFProfile):
            raise ValueError("source must be an IRFProfile")
        if not isinstance(self.diagnostics, IRFPreparationDiagnostics):
            raise ValueError("diagnostics must be IRFPreparationDiagnostics")
        time = validate_time_axis(self.time_ns)
        kernel = np.asarray(self.kernel, dtype=np.float64)
        if kernel.ndim != 1 or kernel.shape != time.shape:
            raise ValueError("kernel must match the one-dimensional target grid")
        if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
            raise ValueError("kernel must contain finite nonnegative values")
        area = float(np.trapezoid(kernel, x=time))
        if not np.isfinite(area) or not np.isclose(area, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("kernel must have unit trapezoidal area")
        time_copy = np.array(time, dtype=np.float64, copy=True)
        kernel_copy = np.array(kernel, dtype=np.float64, copy=True)
        time_copy.setflags(write=False)
        kernel_copy.setflags(write=False)
        object.__setattr__(self, "time_ns", time_copy)
        object.__setattr__(self, "kernel", kernel_copy)


def _sampled_fwhm(time: FloatArray, values: FloatArray) -> float | None:
    """Interpolate the two half-height crossings around the principal peak."""
    peak_index = int(np.argmax(values))
    if peak_index == 0 or peak_index == values.size - 1:
        return None
    half_height = 0.5 * values[peak_index]
    above_half = values >= half_height
    above_indices = np.flatnonzero(above_half)
    if np.any(~above_half[above_indices[0] : above_indices[-1] + 1]):
        return None  # Multiple separated lobes make one FWHM ambiguous.
    left_below = np.flatnonzero(values[:peak_index] < half_height)
    right_below = np.flatnonzero(values[peak_index + 1 :] < half_height)
    if left_below.size == 0 or right_below.size == 0:
        return None
    left_index = int(left_below[-1])
    right_index = peak_index + 1 + int(right_below[0])
    left = time[left_index] + (
        (half_height - values[left_index])
        * (time[left_index + 1] - time[left_index])
        / (values[left_index + 1] - values[left_index])
    )
    right = time[right_index - 1] + (
        (half_height - values[right_index - 1])
        * (time[right_index] - time[right_index - 1])
        / (values[right_index] - values[right_index - 1])
    )
    width = float(right - left)
    return width if np.isfinite(width) and width > 0.0 else None


def _retained_piecewise_linear_area(
    registered_time: FloatArray,
    source_values: FloatArray,
    target_start: float,
    target_stop: float,
) -> float:
    """Integrate the registered linear source over the target acquisition window."""
    start = max(target_start, float(registered_time[0]))
    stop = min(target_stop, float(registered_time[-1]))
    if stop <= start:
        return 0.0
    inside = registered_time[(registered_time > start) & (registered_time < stop)]
    knots = np.concatenate(([start], inside, [stop]))
    values = np.interp(knots, registered_time, source_values)
    return float(np.trapezoid(values, x=knots))


def prepare_irf(
    profile: IRFProfile,
    target_time_ns: ArrayLike,
    *,
    resampling: ResamplingMethod = "none",
    registration_offset_ns: float = 0.0,
    max_support_loss_fraction: float | None = None,
) -> PreparedIRF:
    """Register, explicitly resample when requested, and normalize an IRF.

    A positive registration moves the source later. Linear resampling gives
    zero outside its recorded time domain. Source provenance and samples are
    never rewritten. Support diagnostics refer only to that recorded domain.
    """
    if not isinstance(profile, IRFProfile):
        raise TypeError("profile must be an IRFProfile")
    target = validate_time_axis(target_time_ns)
    if resampling not in ("none", "linear"):
        raise ValueError("resampling must be 'none' or 'linear'")
    try:
        offset = float(registration_offset_ns)
    except (TypeError, ValueError) as exc:
        raise ValueError("registration_offset_ns must be finite") from exc
    if not np.isfinite(offset):
        raise ValueError("registration_offset_ns must be finite")
    if max_support_loss_fraction is not None:
        try:
            loss_limit = float(max_support_loss_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_support_loss_fraction must be between 0 and 1") from exc
        if not np.isfinite(loss_limit) or not 0.0 <= loss_limit <= 1.0:
            raise ValueError("max_support_loss_fraction must be between 0 and 1")
    else:
        loss_limit = None

    registered_time = profile.time_ns + offset
    if not np.all(np.isfinite(registered_time)) or np.any(np.diff(registered_time) <= 0.0):
        raise ValueError("registration produces an invalid source time grid")
    try:
        compatible = time_axes_compatible(target, registered_time)
    except ValueError:
        compatible = False  # A genuinely nonuniform source needs resampling.
    if not compatible and resampling == "none":
        raise ValueError("IRF and target grids differ; request resampling='linear'")

    operations: list[str] = []
    if offset != 0.0:
        operations.append("registration")
    if compatible:
        target_values = np.array(profile.values, dtype=np.float64, copy=True)
        operations.append("compatible_grid_copy")
    else:
        target_values = np.interp(
            target, registered_time, profile.values, left=0.0, right=0.0
        )
        operations.append("linear_resampling")

    source_area = float(np.trapezoid(profile.values, x=profile.time_ns))
    retained_area = _retained_piecewise_linear_area(
        registered_time, profile.values, float(target[0]), float(target[-1])
    )
    if not np.isfinite(retained_area):
        raise ValueError("registered source support has nonfinite area")
    retained_fraction = float(np.clip(retained_area / source_area, 0.0, 1.0))
    lost_fraction = float(1.0 - retained_fraction)
    target_area = float(np.trapezoid(target_values, x=target))
    if not np.isfinite(target_area) or target_area <= 0.0:
        raise ValueError("resampled IRF must have a positive finite target area")
    if loss_limit is not None and lost_fraction > loss_limit:
        raise ValueError(
            f"support_loss_fraction {lost_fraction:.6g} exceeds "
            f"max_support_loss_fraction {loss_limit:.6g}"
        )
    normalization_factor = 1.0 / target_area
    if not np.isfinite(normalization_factor):
        raise ValueError("target IRF area is too small to normalize")
    kernel = normalize_irf(target, target_values)
    normalized_area = float(np.trapezoid(kernel, x=target))
    if not np.all(np.isfinite(kernel)) or not np.isfinite(normalized_area):
        raise ValueError("normalized IRF must be finite")
    operations.append("normalization")

    source_width = _sampled_fwhm(profile.time_ns, profile.values)
    target_width = _sampled_fwhm(target, target_values)
    target_spacing = float(np.diff(target)[0])
    flags: list[str] = []
    if lost_fraction > 0.0:
        flags.append("support_truncated")
    if source_width is None:
        flags.append("source_fwhm_unavailable")
    if target_width is None:
        flags.append("target_fwhm_unavailable")
    diagnostics = IRFPreparationDiagnostics(
        source_area=source_area,
        retained_source_area=retained_area,
        support_retained_fraction=retained_fraction,
        support_loss_fraction=lost_fraction,
        target_area_before_normalization=target_area,
        target_area_after_normalization=normalized_area,
        normalization_factor=normalization_factor,
        target_minus_retained_area=target_area - retained_area,
        source_spacing_min_ns=float(np.min(np.diff(profile.time_ns))),
        source_spacing_max_ns=float(np.max(np.diff(profile.time_ns))),
        target_spacing_ns=target_spacing,
        source_fwhm_ns=source_width,
        target_fwhm_ns=target_width,
        target_bins_per_fwhm=(
            None if target_width is None else target_width / target_spacing
        ),
        registration_offset_ns=offset,
        resampling_method=resampling,
        operations=tuple(operations),
        flags=tuple(flags),
        max_support_loss_fraction=loss_limit,
    )
    return PreparedIRF(source=profile, time_ns=target, kernel=kernel, diagnostics=diagnostics)
