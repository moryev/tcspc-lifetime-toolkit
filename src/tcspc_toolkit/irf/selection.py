"""Internal fixed-IRF selection shared by classical and Bayesian adapters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.irf.profiles import normalize_irf
from tcspc_toolkit.irf.preparation import PreparedIRF
from tcspc_toolkit.measurement.records import SampledIRF, TCSPCMeasurement
from tcspc_toolkit.preprocessing.histograms import time_axes_compatible


def _resolve_reconvolution_irf(
    measurement: TCSPCMeasurement,
    counts: NDArray[np.int64],
    prepared_irf: PreparedIRF | None,
) -> tuple[NDArray[np.float64], PreparedIRF | SampledIRF]:
    """Select a fixed kernel without changing the source IRF or its grid."""
    if prepared_irf is None and measurement.irf is None:
        raise InvalidMeasurementError(
            "a measured IRF on the histogram grid is required for experimental reconvolution"
        )
    if prepared_irf is not None:
        if not isinstance(prepared_irf, PreparedIRF):
            raise InvalidMeasurementError("prepared_irf must be a PreparedIRF")
        if not time_axes_compatible(measurement.time_ns, prepared_irf.time_ns):
            raise InvalidMeasurementError(
                "prepared_irf grid differs from measurement time grid; "
                "prepare the IRF explicitly on the measurement grid"
            )
        if (
            prepared_irf.kernel.shape != counts.shape
            or not np.all(np.isfinite(prepared_irf.kernel))
            or np.any(prepared_irf.kernel < 0.0)
        ):
            raise InvalidMeasurementError(
                "prepared_irf kernel must be finite, nonnegative, and unit-area"
            )
        kernel_area = float(np.trapezoid(prepared_irf.kernel, x=prepared_irf.time_ns))
        if not np.isfinite(kernel_area) or not np.isclose(
            kernel_area, 1.0, rtol=1e-10, atol=1e-12
        ):
            raise InvalidMeasurementError("prepared_irf kernel must have unit area")
        return prepared_irf.kernel, prepared_irf

    attached_irf = measurement.irf
    assert attached_irf is not None
    return normalize_irf(measurement.time_ns, attached_irf.values), attached_irf
