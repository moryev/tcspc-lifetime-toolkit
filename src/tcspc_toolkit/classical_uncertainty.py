"""Classical uncertainty methods for TCSPC lifetime estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tcspc_toolkit.fitting import (
    ReconvolutionFitResult,
    _reconvolution_model,
)
from tcspc_toolkit.uncertainty_evaluation import (
    UncertaintyMethodDefinition,
    UncertaintyOutputKind,
)


POISSON_LOCAL_COVARIANCE_METHOD_ID = "covariance"

DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP = 1e-4
DEFAULT_INFORMATION_CONDITION_LIMIT = 1e12


CLASSICAL_UNCERTAINTY_METHODS = {
    "covariance": UncertaintyMethodDefinition(
        method_id="covariance",
        output_kind=(
            UncertaintyOutputKind.PREDICTION_INTERVAL
        ),
        interpretation=(
            "Local fitted-parameter uncertainty derived from "
            "the expected Poisson Fisher information of the "
            "reconvolution model."
        ),
        calibration_scope=(
            "Meaningful only when the fitted physical model is "
            "adequate, the optimum is away from parameter bounds, "
            "and the local information matrix is sufficiently "
            "well conditioned."
        ),
    ),
    "parametric_bootstrap": UncertaintyMethodDefinition(
        method_id="parametric_bootstrap",
        output_kind=(
            UncertaintyOutputKind.PREDICTION_INTERVAL
        ),
        interpretation=(
            "Sampling variability obtained by resimulating "
            "measurements from the fitted model and refitting them."
        ),
        calibration_scope=(
            "Conditional on the fitted physical model; it does not "
            "capture model misspecification such as Test F."
        ),
    ),
    "repeated_poisson_realizations": UncertaintyMethodDefinition(
        method_id="repeated_poisson_realizations",
        output_kind=(
            UncertaintyOutputKind.EMPIRICAL_REFERENCE
        ),
        interpretation=(
            "Empirical estimator variability over repeated Poisson "
            "measurements generated from known physics."
        ),
        calibration_scope=(
            "Reference benchmark for repeated-measurement "
            "variability, not a per-curve uncertainty estimate "
            "available in deployment."
        ),
    ),
}


@dataclass(frozen=True)
class PoissonLocalCovarianceResult:
    """Local covariance diagnostics for one Poisson reconvolution fit."""

    covariance_matrix: NDArray[np.float64]

    amplitude_std: float
    lifetime_std: float
    background_std: float
    temporal_shift_std: float

    covariance_valid: bool
    condition_number: float
    information_rank: int
    boundary_hit: bool

    failure_reason: str | None


def _invalid_local_covariance_result(
    *,
    failure_reason: str,
    condition_number: float = np.nan,
    information_rank: int = 0,
    boundary_hit: bool = False,
) -> PoissonLocalCovarianceResult:
    """Construct a failure-aware covariance result."""

    return PoissonLocalCovarianceResult(
        covariance_matrix=np.full(
            (4, 4),
            np.nan,
            dtype=np.float64,
        ),
        amplitude_std=np.nan,
        lifetime_std=np.nan,
        background_std=np.nan,
        temporal_shift_std=np.nan,
        covariance_valid=False,
        condition_number=float(
            condition_number
        ),
        information_rank=int(
            information_rank
        ),
        boundary_hit=boundary_hit,
        failure_reason=failure_reason,
    )


def _poisson_fit_hits_boundary(
    *,
    parameters: NDArray[np.float64],
    temporal_shift_bounds: tuple[float, float],
    bin_width: float,
) -> bool:
    """Return whether the fitted solution lies effectively on a bound."""

    (
        amplitude,
        lifetime,
        background,
        temporal_shift,
    ) = parameters

    shift_lower, shift_upper = (
        temporal_shift_bounds
    )

    shift_tolerance = (
        0.5 * bin_width
    )

    amplitude_at_lower_bound = np.isclose(
        amplitude,
        0.0,
        rtol=0.0,
        atol=1e-10,
    )

    lifetime_at_lower_bound = np.isclose(
        lifetime,
        1e-12,
        rtol=0.0,
        atol=1e-10,
    )

    background_at_lower_bound = (
        background <= 1e-10
    )

    shift_at_lower_bound = np.isclose(
        temporal_shift,
        shift_lower,
        rtol=0.0,
        atol=shift_tolerance,
    )

    shift_at_upper_bound = np.isclose(
        temporal_shift,
        shift_upper,
        rtol=0.0,
        atol=shift_tolerance,
    )

    return bool(
        amplitude_at_lower_bound
        or lifetime_at_lower_bound
        or background_at_lower_bound
        or shift_at_lower_bound
        or shift_at_upper_bound
    )


def _parameter_scales(
    *,
    parameters: NDArray[np.float64],
    bin_width: float,
) -> NDArray[np.float64]:
    """Return physical scales used for dimensionless conditioning."""

    (
        amplitude,
        lifetime,
        background,
        temporal_shift,
    ) = parameters

    return np.asarray(
        [
            max(
                abs(amplitude),
                1.0,
            ),
            max(
                abs(lifetime),
                bin_width,
            ),
            max(
                abs(background),
                1.0,
            ),
            max(
                abs(temporal_shift),
                bin_width,
            ),
        ],
        dtype=np.float64,
    )


def _expected_count_jacobian(
    *,
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    parameters: NDArray[np.float64],
    parameter_scales: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    relative_step: float,
) -> NDArray[np.float64]:
    """Numerically differentiate expected counts with respect to parameters."""

    n_bins = time.size
    n_parameters = parameters.size

    jacobian = np.empty(
        (
            n_bins,
            n_parameters,
        ),
        dtype=np.float64,
    )

    def evaluate(
        candidate_parameters: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        (
            amplitude,
            lifetime,
            background,
            temporal_shift,
        ) = candidate_parameters

        return _reconvolution_model(
            time=time,
            irf=irf,
            amplitude=float(
                amplitude
            ),
            lifetime=float(
                lifetime
            ),
            background=float(
                background
            ),
            temporal_shift=float(
                temporal_shift
            ),
        )

    base_expected = evaluate(
        parameters
    )

    for parameter_index in range(
        n_parameters
    ):
        step = (
            relative_step
            * parameter_scales[
                parameter_index
            ]
        )

        minimum_step = (
            np.sqrt(
                np.finfo(
                    np.float64
                ).eps
            )
            * parameter_scales[
                parameter_index
            ]
        )

        step = max(
            step,
            minimum_step,
        )

        plus_parameters = (
            parameters.copy()
        )

        minus_parameters = (
            parameters.copy()
        )

        plus_parameters[
            parameter_index
        ] += step

        minus_parameters[
            parameter_index
        ] -= step

        can_step_plus = (
            plus_parameters[
                parameter_index
            ]
            < upper_bounds[
                parameter_index
            ]
        )

        can_step_minus = (
            minus_parameters[
                parameter_index
            ]
            > lower_bounds[
                parameter_index
            ]
        )

        if (
            can_step_plus
            and can_step_minus
        ):
            plus_expected = evaluate(
                plus_parameters
            )

            minus_expected = evaluate(
                minus_parameters
            )

            derivative = (
                plus_expected
                - minus_expected
            ) / (
                2.0 * step
            )

        elif can_step_plus:
            plus_expected = evaluate(
                plus_parameters
            )

            derivative = (
                plus_expected
                - base_expected
            ) / step

        elif can_step_minus:
            minus_expected = evaluate(
                minus_parameters
            )

            derivative = (
                base_expected
                - minus_expected
            ) / step

        else:
            derivative = np.full(
                n_bins,
                np.nan,
                dtype=np.float64,
            )

        jacobian[
            :,
            parameter_index,
        ] = derivative

    return jacobian


def estimate_poisson_reconvolution_local_covariance(
    *,
    time: ArrayLike,
    irf: ArrayLike,
    fit_result: ReconvolutionFitResult,
    temporal_shift_bounds: tuple[float, float],
    relative_step: float = (
        DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP
    ),
    condition_number_limit: float = (
        DEFAULT_INFORMATION_CONDITION_LIMIT
    ),
) -> PoissonLocalCovarianceResult:
    """Estimate local parameter covariance for a Poisson reconvolution fit.

    The approximation uses the expected Poisson Fisher information

        I(theta) = J.T @ diag(1 / mu) @ J,

    where J is the Jacobian of the expected TCSPC counts with respect
    to amplitude, lifetime, background, and temporal shift.

    The information matrix is evaluated in dimensionless scaled
    parameter coordinates for numerical conditioning. The resulting
    covariance is then transformed back to physical parameter units.

    No covariance is reported when the fit failed, lies on a parameter
    bound, produces invalid expected counts, or has singular / badly
    conditioned local information.
    """

    time_array = np.asarray(
        time,
        dtype=np.float64,
    )

    irf_array = np.asarray(
        irf,
        dtype=np.float64,
    )

    if time_array.ndim != 1:
        raise ValueError(
            "time must be one-dimensional."
        )

    if irf_array.ndim != 1:
        raise ValueError(
            "irf must be one-dimensional."
        )

    if time_array.shape != irf_array.shape:
        raise ValueError(
            "time and irf must have the same shape."
        )

    if time_array.size < 5:
        raise ValueError(
            "at least five time bins are required."
        )

    if not np.all(
        np.isfinite(
            time_array
        )
    ):
        raise ValueError(
            "time must contain only finite values."
        )

    if not np.all(
        np.isfinite(
            irf_array
        )
    ):
        raise ValueError(
            "irf must contain only finite values."
        )

    if np.any(
        irf_array < 0.0
    ):
        raise ValueError(
            "irf must contain only non-negative values."
        )

    time_differences = np.diff(
        time_array
    )

    if np.any(
        time_differences <= 0.0
    ):
        raise ValueError(
            "time must be strictly increasing."
        )

    if not np.allclose(
        time_differences,
        time_differences[0],
    ):
        raise ValueError(
            "time must be uniformly spaced."
        )

    if (
        not np.isfinite(
            relative_step
        )
        or relative_step <= 0.0
    ):
        raise ValueError(
            "relative_step must be positive and finite."
        )

    if (
        not np.isfinite(
            condition_number_limit
        )
        or condition_number_limit < 1.0
    ):
        raise ValueError(
            "condition_number_limit must be finite "
            "and at least 1."
        )

    shift_lower, shift_upper = (
        temporal_shift_bounds
    )

    if not (
        np.isfinite(
            shift_lower
        )
        and np.isfinite(
            shift_upper
        )
    ):
        raise ValueError(
            "temporal shift bounds must be finite."
        )

    if shift_lower >= shift_upper:
        raise ValueError(
            "lower temporal shift bound must be "
            "smaller than upper temporal shift bound."
        )

    if not fit_result.success:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "fit_unsuccessful"
                ),
            )
        )

    parameters = np.asarray(
        [
            fit_result.amplitude,
            fit_result.lifetime,
            fit_result.background,
            fit_result.temporal_shift,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            parameters
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_finite_parameters"
                ),
            )
        )

    if (
        fit_result.amplitude < 0.0
        or fit_result.lifetime <= 0.0
        or fit_result.background <= 0.0
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_physical_parameters"
                ),
            )
        )

    bin_width = float(
        time_differences[0]
    )

    boundary_hit = (
        _poisson_fit_hits_boundary(
            parameters=parameters,
            temporal_shift_bounds=(
                temporal_shift_bounds
            ),
            bin_width=bin_width,
        )
    )

    if boundary_hit:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "parameter_at_bound"
                ),
                boundary_hit=True,
            )
        )

    expected_counts = (
        _reconvolution_model(
            time=time_array,
            irf=irf_array,
            amplitude=fit_result.amplitude,
            lifetime=fit_result.lifetime,
            background=fit_result.background,
            temporal_shift=(
                fit_result.temporal_shift
            ),
        )
    )

    if (
        not np.all(
            np.isfinite(
                expected_counts
            )
        )
        or np.any(
            expected_counts <= 0.0
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "invalid_expected_counts"
                ),
            )
        )

    lower_bounds = np.asarray(
        [
            0.0,
            1e-12,
            1e-12,
            shift_lower,
        ],
        dtype=np.float64,
    )

    upper_bounds = np.asarray(
        [
            np.inf,
            np.inf,
            np.inf,
            shift_upper,
        ],
        dtype=np.float64,
    )

    parameter_scales = (
        _parameter_scales(
            parameters=parameters,
            bin_width=bin_width,
        )
    )

    jacobian = (
        _expected_count_jacobian(
            time=time_array,
            irf=irf_array,
            parameters=parameters,
            parameter_scales=(
                parameter_scales
            ),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            relative_step=relative_step,
        )
    )

    if not np.all(
        np.isfinite(
            jacobian
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_finite_jacobian"
                ),
            )
        )

    weighted_jacobian = (
        jacobian
        / np.sqrt(
            expected_counts
        )[:, None]
    )

    information_matrix = (
        weighted_jacobian.T
        @ weighted_jacobian
    )

    scaled_information_matrix = (
        parameter_scales[:, None]
        * information_matrix
        * parameter_scales[None, :]
    )

    information_rank = int(
        np.linalg.matrix_rank(
            scaled_information_matrix
        )
    )

    n_parameters = (
        parameters.size
    )

    if information_rank < n_parameters:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "singular_information_matrix"
                ),
                condition_number=np.inf,
                information_rank=(
                    information_rank
                ),
            )
        )

    condition_number = float(
        np.linalg.cond(
            scaled_information_matrix
        )
    )

    if not np.isfinite(
        condition_number
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "singular_information_matrix"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    if (
        condition_number
        > condition_number_limit
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "ill_conditioned_information_matrix"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    try:
        scaled_covariance = (
            np.linalg.inv(
                scaled_information_matrix
            )
        )

    except np.linalg.LinAlgError:
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "singular_information_matrix"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    covariance_matrix = (
        parameter_scales[:, None]
        * scaled_covariance
        * parameter_scales[None, :]
    )

    covariance_matrix = (
        0.5
        * (
            covariance_matrix
            + covariance_matrix.T
        )
    )

    if not np.all(
        np.isfinite(
            covariance_matrix
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_finite_covariance"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    parameter_variances = np.diag(
        covariance_matrix
    )

    if (
        not np.all(
            np.isfinite(
                parameter_variances
            )
        )
        or np.any(
            parameter_variances <= 0.0
        )
    ):
        return (
            _invalid_local_covariance_result(
                failure_reason=(
                    "non_positive_parameter_variance"
                ),
                condition_number=(
                    condition_number
                ),
                information_rank=(
                    information_rank
                ),
            )
        )

    parameter_standard_deviations = (
        np.sqrt(
            parameter_variances
        )
    )

    (
        amplitude_std,
        lifetime_std,
        background_std,
        temporal_shift_std,
    ) = parameter_standard_deviations

    return PoissonLocalCovarianceResult(
        covariance_matrix=(
            covariance_matrix
        ),
        amplitude_std=float(
            amplitude_std
        ),
        lifetime_std=float(
            lifetime_std
        ),
        background_std=float(
            background_std
        ),
        temporal_shift_std=float(
            temporal_shift_std
        ),
        covariance_valid=True,
        condition_number=(
            condition_number
        ),
        information_rank=(
            information_rank
        ),
        boundary_hit=False,
        failure_reason=None,
    )
