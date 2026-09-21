import numpy as np
from numpy.typing import NDArray

from tcspc_toolkit.classical_uncertainty import (
    estimate_poisson_reconvolution_local_covariance,
)
from tcspc_toolkit.fitting import (
    ReconvolutionFitResult,
    _reconvolution_model,
)


def _make_reconvolution_fit_result(
    *,
    time: NDArray[np.float64],
    irf: NDArray[np.float64],
    amplitude: float,
    lifetime: float,
    background: float,
    temporal_shift: float,
    success: bool = True,
) -> ReconvolutionFitResult:
    fitted_curve = _reconvolution_model(
        time=time,
        irf=irf,
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
        temporal_shift=temporal_shift,
    )

    return ReconvolutionFitResult(
        amplitude=amplitude,
        lifetime=lifetime,
        background=background,
        temporal_shift=temporal_shift,
        fitted_curve=fitted_curve,
        success=success,
    )


def test_poisson_local_covariance_is_valid_for_well_behaved_fit(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    fit_result = (
        _make_reconvolution_fit_result(
            time=time_axis,
            irf=irf,
            amplitude=100_000.0,
            lifetime=0.8,
            background=10.0,
            temporal_shift=0.1,
        )
    )

    result = (
        estimate_poisson_reconvolution_local_covariance(
            time=time_axis,
            irf=irf,
            fit_result=fit_result,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )
    )

    assert result.covariance_valid
    assert result.failure_reason is None

    assert (
        result.covariance_matrix.shape
        == (4, 4)
    )

    assert np.all(
        np.isfinite(
            result.covariance_matrix
        )
    )

    np.testing.assert_allclose(
        result.covariance_matrix,
        result.covariance_matrix.T,
        rtol=1e-10,
        atol=1e-12,
    )

    assert result.amplitude_std > 0.0
    assert result.lifetime_std > 0.0
    assert result.background_std > 0.0
    assert result.temporal_shift_std > 0.0

    assert np.isfinite(
        result.condition_number
    )

    assert (
        result.information_rank
        == 4
    )


def test_more_signal_photons_reduce_local_lifetime_uncertainty(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    low_count_fit = (
        _make_reconvolution_fit_result(
            time=time_axis,
            irf=irf,
            amplitude=10_000.0,
            lifetime=0.8,
            background=5.0,
            temporal_shift=0.1,
        )
    )

    high_count_fit = (
        _make_reconvolution_fit_result(
            time=time_axis,
            irf=irf,
            amplitude=1_000_000.0,
            lifetime=0.8,
            background=5.0,
            temporal_shift=0.1,
        )
    )

    low_count_result = (
        estimate_poisson_reconvolution_local_covariance(
            time=time_axis,
            irf=irf,
            fit_result=low_count_fit,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )
    )

    high_count_result = (
        estimate_poisson_reconvolution_local_covariance(
            time=time_axis,
            irf=irf,
            fit_result=high_count_fit,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )
    )

    assert (
        low_count_result.covariance_valid
    )

    assert (
        high_count_result.covariance_valid
    )

    assert (
        high_count_result.lifetime_std
        < low_count_result.lifetime_std
    )


def test_poisson_local_covariance_rejects_unsuccessful_fit(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    fit_result = (
        _make_reconvolution_fit_result(
            time=time_axis,
            irf=irf,
            amplitude=100_000.0,
            lifetime=0.8,
            background=10.0,
            temporal_shift=0.1,
            success=False,
        )
    )

    result = (
        estimate_poisson_reconvolution_local_covariance(
            time=time_axis,
            irf=irf,
            fit_result=fit_result,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )
    )

    assert not result.covariance_valid

    assert (
        result.failure_reason
        == "fit_unsuccessful"
    )

    assert np.isnan(
        result.lifetime_std
    )


def test_poisson_local_covariance_rejects_boundary_solution(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    fit_result = (
        _make_reconvolution_fit_result(
            time=time_axis,
            irf=irf,
            amplitude=100_000.0,
            lifetime=0.8,
            background=10.0,
            temporal_shift=0.5,
        )
    )

    result = (
        estimate_poisson_reconvolution_local_covariance(
            time=time_axis,
            irf=irf,
            fit_result=fit_result,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
        )
    )

    assert not result.covariance_valid
    assert result.boundary_hit

    assert (
        result.failure_reason
        == "parameter_at_bound"
    )

    assert np.isnan(
        result.lifetime_std
    )


def test_poisson_local_covariance_rejects_bad_conditioning(
    time_axis: NDArray[np.float64],
    irf: NDArray[np.float64],
) -> None:
    fit_result = (
        _make_reconvolution_fit_result(
            time=time_axis,
            irf=irf,
            amplitude=100_000.0,
            lifetime=0.8,
            background=10.0,
            temporal_shift=0.1,
        )
    )

    result = (
        estimate_poisson_reconvolution_local_covariance(
            time=time_axis,
            irf=irf,
            fit_result=fit_result,
            temporal_shift_bounds=(
                -0.5,
                0.5,
            ),
            condition_number_limit=1.0,
        )
    )

    assert not result.covariance_valid

    assert (
        result.failure_reason
        == "ill_conditioned_information_matrix"
    )

    assert np.isfinite(
        result.condition_number
    )


