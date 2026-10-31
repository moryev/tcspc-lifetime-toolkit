"""Private fixed-K records and coordinates; no fitting or public API policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.irf.preparation import PreparedIRF
from tcspc_toolkit.irf.profiles import IRFProfile
from tcspc_toolkit.preprocessing.histograms import (
    time_axes_compatible,
    validate_time_axis,
)


_FloatArray = NDArray[np.float64]
_TargetKind = Literal["raw_counts", "deterministic_expected_counts"]


def _readonly(values: ArrayLike) -> _FloatArray:
    if np.iscomplexobj(values):
        raise ValueError("values must be real")
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scalar(value: float, name: str, *, minimum: float | None = None) -> float:
    if np.iscomplexobj(value) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"invalid {name}")
    return result


def _bounds(values, name, *, positive=False):
    array = _readonly(values)
    if (
        array.shape != (2,)
        or not np.all(np.isfinite(array))
        or array[0] >= array[1]
        or (positive and array[0] <= 0)
    ):
        raise ValueError(f"{name} must be a finite increasing pair")
    if positive and np.log(array[0]) >= np.log(array[1]):
        raise ValueError("lifetime bounds are indistinguishable in log coordinates")
    return tuple(map(float, array))


@dataclass(frozen=True)
class _FixedKParameters:
    """Pre-exponential amplitudes, ns lifetimes/shift, background counts/bin."""

    amplitudes: tuple[float, ...]
    lifetimes_ns: tuple[float, ...]
    background: float = 0.0
    temporal_shift_ns: float = 0.0

    def __post_init__(self):
        a, tau = _readonly(self.amplitudes), _readonly(self.lifetimes_ns)
        if a.ndim != 1 or not a.size or tau.shape != a.shape:
            raise ValueError("nonempty matching amplitude/lifetime vectors required")
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError("amplitudes must be finite and nonnegative")
        if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError("lifetimes must be finite and positive")
        object.__setattr__(self, "amplitudes", tuple(map(float, a)))
        object.__setattr__(self, "lifetimes_ns", tuple(map(float, tau)))
        object.__setattr__(
            self, "background", _scalar(self.background, "background", minimum=0)
        )
        object.__setattr__(
            self, "temporal_shift_ns", _scalar(self.temporal_shift_ns, "shift")
        )

    @property
    def n_components(self):
        return len(self.amplitudes)


def _canonicalize_components(parameters: _FixedKParameters):
    order = np.lexsort((-np.asarray(parameters.amplitudes), parameters.lifetimes_ns))
    order.setflags(write=False)
    return _FixedKParameters(
        tuple(parameters.amplitudes[i] for i in order),
        tuple(parameters.lifetimes_ns[i] for i in order),
        parameters.background,
        parameters.temporal_shift_ns,
    ), order


@dataclass(frozen=True)
class _CoordinateMap:
    n_components: int
    lifetime_bounds_ns: tuple[float, float]
    temporal_shift_bounds_ns: tuple[float, float]
    amplitude_scale: float
    background_scale: float

    def __post_init__(self):
        object.__setattr__(
            self, "n_components", _positive_integer(self.n_components, "K")
        )
        object.__setattr__(
            self,
            "lifetime_bounds_ns",
            _bounds(self.lifetime_bounds_ns, "lifetimes", positive=True),
        )
        object.__setattr__(
            self,
            "temporal_shift_bounds_ns",
            _bounds(self.temporal_shift_bounds_ns, "shift"),
        )
        for name in ("amplitude_scale", "background_scale"):
            value = _scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError("scales must be positive")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.shift_scale) or self.shift_scale <= 0:
            raise ValueError("shift range is unrepresentable")

    @property
    def dimension(self):
        return 2 * self.n_components + 2

    @property
    def log_mid(self):
        lo, hi = self.lifetime_bounds_ns
        return 0.5 * np.log(lo) + 0.5 * np.log(hi)

    @property
    def shift_mid(self):
        lo, hi = self.temporal_shift_bounds_ns
        return 0.5 * lo + 0.5 * hi

    @property
    def shift_scale(self):
        lo, hi = self.temporal_shift_bounds_ns
        return 0.5 * hi - 0.5 * lo

    @property
    def lower(self):
        k = self.n_components
        return _readonly(
            [0.0] * k
            + [np.log(self.lifetime_bounds_ns[0]) - self.log_mid] * k
            + [0.0, -1.0]
        )

    @property
    def upper(self):
        k = self.n_components
        return _readonly(
            [np.inf] * k
            + [np.log(self.lifetime_bounds_ns[1]) - self.log_mid] * k
            + [np.inf, 1.0]
        )

    @property
    def names(self):
        k = self.n_components
        return tuple(
            [f"amplitude_{i + 1}" for i in range(k)]
            + [f"log_lifetime_{i + 1}" for i in range(k)]
            + ["background", "shift"]
        )

    def feasible(self, coordinates):
        x = np.asarray(coordinates)
        return bool(
            not np.iscomplexobj(x)
            and x.shape == (self.dimension,)
            and np.all(np.isfinite(x))
            and np.all(x >= self.lower)
            and np.all(x <= self.upper)
        )

    def encode(self, parameters: _FixedKParameters):
        if parameters.n_components != self.n_components:
            raise ValueError("component count mismatch")
        tau = np.asarray(parameters.lifetimes_ns)
        lo, hi = self.lifetime_bounds_ns
        slo, shi = self.temporal_shift_bounds_ns
        if (
            np.any(tau < lo)
            or np.any(tau > hi)
            or not slo <= parameters.temporal_shift_ns <= shi
        ):
            raise ValueError("parameters outside supplied bounds")
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.r_[
                np.asarray(parameters.amplitudes) / self.amplitude_scale,
                np.log(tau) - self.log_mid,
                parameters.background / self.background_scale,
                (parameters.temporal_shift_ns - self.shift_mid) / self.shift_scale,
            ]
        # Only endpoint roundoff in an already checked physical parameter is snapped.
        x[self.n_components : 2 * self.n_components] = np.clip(
            x[self.n_components : 2 * self.n_components],
            self.lower[self.n_components : 2 * self.n_components],
            self.upper[self.n_components : 2 * self.n_components],
        )
        x[-1] = np.clip(x[-1], -1, 1)
        if not self.feasible(x):
            raise ValueError("unrepresentable parameter coordinates")
        if np.any(
            (np.asarray(parameters.amplitudes) > 0) & (x[: self.n_components] == 0)
        ) or (parameters.background > 0 and x[-2] == 0):
            raise ValueError("positive linear parameter underflows in coordinates")
        return _readonly(x)

    def decode(self, coordinates):
        x = _readonly(coordinates)
        if not self.feasible(x):
            raise ValueError("infeasible coordinates")
        k = self.n_components
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            tau = np.clip(np.exp(self.log_mid + x[k : 2 * k]), *self.lifetime_bounds_ns)
            tau[x[k : 2 * k] == self.lower[k : 2 * k]] = self.lifetime_bounds_ns[0]
            tau[x[k : 2 * k] == self.upper[k : 2 * k]] = self.lifetime_bounds_ns[1]
            shift = float(
                np.clip(
                    self.shift_mid + self.shift_scale * x[-1],
                    *self.temporal_shift_bounds_ns,
                )
            )
            if x[-1] in (-1, 1):
                shift = self.temporal_shift_bounds_ns[int(x[-1] == 1)]
            amplitudes = self.amplitude_scale * x[:k]
            background = self.background_scale * x[-2]
            if np.any((x[:k] > 0) & (amplitudes == 0)) or (
                x[-2] > 0 and background == 0
            ):
                raise ValueError(
                    "positive coordinate underflows in physical parameters"
                )
            return _FixedKParameters(tuple(amplitudes), tuple(tau), background, shift)


@dataclass(frozen=True)
class _FixedKInputs:
    time_ns: _FloatArray
    targets: _FloatArray
    irf: PreparedIRF | _FloatArray
    target_kind: _TargetKind

    def __post_init__(self):
        time = _readonly(validate_time_axis(_readonly(self.time_ns)))
        y = _readonly(self.targets)
        if self.target_kind not in ("raw_counts", "deterministic_expected_counts"):
            raise ValueError("unknown target kind")
        if y.shape != time.shape or not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError("targets must be finite nonnegative values matching time")
        if self.target_kind == "raw_counts" and (
            np.any(y >= 2**53) or np.any(y != np.rint(y))
        ):
            raise ValueError("raw counts must be integer-valued and below 2**53")
        prepared = isinstance(self.irf, PreparedIRF)
        if prepared and not time_axes_compatible(time, self.irf.time_ns):
            raise ValueError("PreparedIRF grid mismatch")
        kernel = _readonly(self.irf.kernel if prepared else self.irf)
        if (
            kernel.shape != time.shape
            or not np.all(np.isfinite(kernel))
            or np.any(kernel < 0)
        ):
            raise ValueError("kernel must be finite, nonnegative and match time")
        area = float(np.trapezoid(kernel, x=time))
        if not np.isfinite(area) or not np.isclose(area, 1, rtol=1e-10, atol=1e-12):
            raise ValueError("kernel must already have unit trapezoidal area")
        if prepared:
            source = self.irf.source
            irf = PreparedIRF(
                IRFProfile(
                    source.time_ns,
                    source.values,
                    source.source_kind,
                    source_parameters=deepcopy(dict(source.source_parameters)),
                    metadata=deepcopy(dict(source.metadata)),
                    provenance=deepcopy(dict(source.provenance)),
                ),
                self.irf.time_ns,
                kernel,
                deepcopy(self.irf.diagnostics),
            )
        else:
            irf = kernel
        object.__setattr__(self, "time_ns", time)
        object.__setattr__(self, "targets", y)
        object.__setattr__(self, "irf", irf)

    @property
    def kernel(self):
        return self.irf.kernel if isinstance(self.irf, PreparedIRF) else self.irf


@dataclass(frozen=True, eq=False)
class _Evaluation:
    evaluation_id: int
    coordinates: _FloatArray
    parameters: _FixedKParameters | None
    expected_counts: _FloatArray | None
    component_counts: _FloatArray | None
    centered_nll: float | None
    status: str
    phase: str
    origin: str
    problem_id: object = None

    def __post_init__(self):
        for name in ("coordinates", "expected_counts", "component_counts"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))

    @property
    def finite(self):
        return (
            self.status == "finite"
            and self.centered_nll is not None
            and np.isfinite(self.centered_nll)
        )


@dataclass(frozen=True)
class _DerivativePolicy:
    """Explicit numerical controls, not a selected production policy."""

    relative_step: float
    relative_tolerance: float
    absolute_tolerance: float
    max_reductions: int

    def __post_init__(self):
        for name in ("relative_step", "relative_tolerance", "absolute_tolerance"):
            value = _scalar(getattr(self, name), name, minimum=0)
            if name == "relative_step" and value == 0:
                raise ValueError("step must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "max_reductions",
            _positive_integer(self.max_reductions, "max_reductions"),
        )


@dataclass(frozen=True)
class _CoordinateDerivative:
    value: float | None
    left: float | None
    right: float | None
    status: str
    step: float | None
    left_status: str | None = None
    right_status: str | None = None
    left_step: float | None = None
    right_step: float | None = None


@dataclass(frozen=True)
class _DerivativeEvaluation:
    base: _Evaluation
    derivatives: tuple[_CoordinateDerivative, ...]
    shift_region: str
    complete: bool

    @property
    def gradient(self):
        if not self.complete or any(d.value is None for d in self.derivatives):
            return None
        return _readonly([d.value for d in self.derivatives])
