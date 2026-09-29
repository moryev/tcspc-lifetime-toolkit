"""Deterministic Bayesian mathematics and canonical input boundaries."""

import math
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma, lognorm, uniform

from tcspc_toolkit.bayesian import (
    BayesianModelContext,
    BayesianPhysicalParameters,
    BayesianPriorConfig,
    BoundedUniformPrior,
    GammaPrior,
    IRFModelRelation,
    LogNormalPrior,
    log_likelihood,
    log_posterior,
    log_prior,
    log_transformation_jacobian,
    physical_to_sampling_coordinates,
    sampling_to_physical_parameters,
)
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.fitting import poisson_negative_log_likelihood
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import (
    IRFSourceKind,
    generate_emg_irf_profile,
    generate_gaussian_irf,
    generate_gaussian_irf_profile,
    normalize_irf,
)
from tcspc_toolkit.irf_estimation import estimate_irf_from_leading_edge
from tcspc_toolkit.irf_preparation import irf_profile_from_sampled_irf, prepare_irf
from tcspc_toolkit.measurements import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
)


def _prior(*, free_shift: bool) -> BayesianPriorConfig:
    return BayesianPriorConfig(
        amplitude=GammaPrior(shape=2.0, rate=0.02),
        lifetime_ns=LogNormalPrior(log_mean=math.log(1.5), log_std=0.5),
        background_per_bin=GammaPrior(shape=2.0, rate=0.5),
        temporal_shift_prior=(
            BoundedUniformPrior(lower_ns=-0.2, upper_ns=0.2)
            if free_shift else None
        ),
        fixed_temporal_shift_ns=None if free_shift else 0.0,
    )


def _raw_measurement(*, attached_irf: SampledIRF | None = None) -> TCSPCMeasurement:
    time = np.linspace(0.0, 8.0, 161)
    generating_irf = normalize_irf(
        time, generate_gaussian_irf(time, centre=1.0, fwhm=0.4)
    )
    expected = monoexponential_reconvolution_expected_counts(
        time, generating_irf, amplitude=50.0, lifetime=1.4,
        background=1.5, temporal_shift=0.0,
    )
    return TCSPCMeasurement(
        time_ns=time,
        values=np.rint(expected).astype(np.int64),
        data_kind=MeasurementDataKind.RAW_COUNTS,
        irf=attached_irf,
        sample_id="bayesian-deterministic-test",
        provenance={"origin": "controlled_test"},
    )


def _attached_irf(time: np.ndarray) -> SampledIRF:
    return SampledIRF(
        time_ns=time,
        values=7.0 * generate_gaussian_irf(time, centre=1.0, fwhm=0.4),
        metadata={"instrument": "test detector"},
        provenance={"origin": "imported trace"},
    )


def _context() -> BayesianModelContext:
    time = np.linspace(0.0, 8.0, 161)
    return BayesianModelContext(_raw_measurement(attached_irf=_attached_irf(time)))


def test_prior_densities_match_normalized_scipy_references() -> None:
    prior = _prior(free_shift=True)
    parameters = BayesianPhysicalParameters(120.0, 1.6, 2.0, 0.05)
    expected = (
        gamma.logpdf(120.0, a=2.0, scale=1.0 / 0.02)
        + lognorm.logpdf(1.6, s=0.5, scale=1.5)
        + gamma.logpdf(2.0, a=2.0, scale=1.0 / 0.5)
        + uniform.logpdf(0.05, loc=-0.2, scale=0.4)
    )

    assert math.isfinite(log_prior(parameters, prior))
    assert log_prior(parameters, prior) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "make_invalid",
    [
        lambda: GammaPrior(shape=0.0, rate=1.0),
        lambda: GammaPrior(shape=2.0, rate=-1.0),
        lambda: GammaPrior(shape=math.inf, rate=1.0),
        lambda: GammaPrior(shape=True, rate=1.0),
        lambda: LogNormalPrior(log_mean=math.nan, log_std=0.5),
        lambda: LogNormalPrior(log_mean=0.0, log_std=0.0),
        lambda: BoundedUniformPrior(lower_ns=0.2, upper_ns=0.2),
        lambda: BoundedUniformPrior(lower_ns=-math.inf, upper_ns=0.2),
        lambda: BayesianPriorConfig(
            GammaPrior(2.0, 1.0), LogNormalPrior(0.0, 1.0), GammaPrior(2.0, 1.0),
            temporal_shift_prior=None, fixed_temporal_shift_ns=None,
        ),
        lambda: BayesianPriorConfig(
            GammaPrior(2.0, 1.0), LogNormalPrior(0.0, 1.0), GammaPrior(2.0, 1.0),
            temporal_shift_prior=BoundedUniformPrior(-0.2, 0.2),
            fixed_temporal_shift_ns=0.0,
        ),
    ],
)
def test_invalid_prior_parameters_are_rejected(make_invalid) -> None:
    with pytest.raises(ValueError):
        make_invalid()


def test_prior_support_and_fixed_shift_are_explicit() -> None:
    free = _prior(free_shift=True)
    fixed = _prior(free_shift=False)
    inside = BayesianPhysicalParameters(10.0, 1.2, 0.8, 0.1)
    assert math.isfinite(log_prior(inside, free))
    assert log_prior(BayesianPhysicalParameters(10.0, 1.2, 0.8, 0.3), free) == -math.inf
    assert log_prior(BayesianPhysicalParameters(0.0, 1.2, 0.8, 0.1), free) == -math.inf
    assert log_prior(BayesianPhysicalParameters(10.0, 0.0, 0.8, 0.1), free) == -math.inf
    assert log_prior(BayesianPhysicalParameters(10.0, 1.2, -0.8, 0.1), free) == -math.inf
    assert log_prior(inside, fixed) == -math.inf
    assert math.isfinite(log_prior(BayesianPhysicalParameters(10.0, 1.2, 0.8, 0.0), fixed))


@pytest.mark.parametrize("free_shift", [False, True])
def test_coordinate_round_trip_and_positive_physical_parameters(free_shift: bool) -> None:
    prior = _prior(free_shift=free_shift)
    parameters = BayesianPhysicalParameters(10.0, 1.2, 0.8, 0.1 if free_shift else 0.0)
    coordinates = physical_to_sampling_coordinates(parameters, prior)
    restored = sampling_to_physical_parameters(coordinates, prior)

    assert coordinates.shape == (prior.n_sampling_coordinates,)
    assert restored.amplitude == pytest.approx(parameters.amplitude)
    assert restored.lifetime_ns == pytest.approx(parameters.lifetime_ns)
    assert restored.background_per_bin == pytest.approx(parameters.background_per_bin)
    assert restored.temporal_shift_ns == pytest.approx(parameters.temporal_shift_ns)
    assert restored.amplitude > 0.0 and restored.lifetime_ns > 0.0
    assert restored.background_per_bin > 0.0


@pytest.mark.parametrize("free_shift", [False, True])
def test_log_coordinate_jacobian_is_log_a_plus_log_tau_plus_log_b(
    free_shift: bool,
) -> None:
    prior = _prior(free_shift=free_shift)
    parameters = BayesianPhysicalParameters(10.0, 1.2, 0.8, 0.1 if free_shift else 0.0)
    coordinates = physical_to_sampling_coordinates(parameters, prior)

    assert log_transformation_jacobian(coordinates, prior) == pytest.approx(
        math.log(10.0 * 1.2 * 0.8)
    )
    if free_shift:
        shifted_coordinates = coordinates.copy()
        shifted_coordinates[3] = -0.1
        assert log_transformation_jacobian(shifted_coordinates, prior) == pytest.approx(
            log_transformation_jacobian(coordinates, prior)
        )


def test_likelihood_equals_existing_poisson_objective_on_shared_forward_counts() -> None:
    context = _context()
    parameters = BayesianPhysicalParameters(50.0, 1.4, 1.5, 0.025)
    expected_counts = monoexponential_reconvolution_expected_counts(
        context.time_ns, context.irf_kernel, amplitude=parameters.amplitude,
        lifetime=parameters.lifetime_ns, background=parameters.background_per_bin,
        temporal_shift=parameters.temporal_shift_ns,
    )
    reference = -poisson_negative_log_likelihood(context.counts, expected_counts)

    assert log_likelihood(parameters, context) == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize("free_shift", [False, True])
def test_posterior_is_prior_plus_likelihood_plus_jacobian(free_shift: bool) -> None:
    context = _context()
    prior = _prior(free_shift=free_shift)
    parameters = BayesianPhysicalParameters(50.0, 1.4, 1.5, 0.025 if free_shift else 0.0)
    coordinates = physical_to_sampling_coordinates(parameters, prior)
    expected = (
        log_prior(parameters, prior)
        + log_likelihood(parameters, context)
        + log_transformation_jacobian(coordinates, prior)
    )

    assert log_posterior(coordinates, context, prior) == pytest.approx(expected)


def test_invalid_proposals_return_negative_infinity_without_hiding_shape_errors() -> None:
    context = _context()
    prior = _prior(free_shift=True)
    ordinary = np.array([math.log(50.0), math.log(1.4), math.log(1.5), 0.0])
    for index, invalid_value in ((0, math.nan), (0, 1000.0), (1, -1000.0), (3, 1.0)):
        proposal = ordinary.copy()
        proposal[index] = invalid_value
        assert log_posterior(proposal, context, prior) == -math.inf

    assert log_likelihood(BayesianPhysicalParameters(-1.0, 1.4, 1.5, 0.0), context) == -math.inf
    with pytest.raises(ValueError, match="shape"):
        log_posterior(ordinary[:3], context, prior)
    with pytest.raises(ValueError, match="fixed shift"):
        physical_to_sampling_coordinates(
            BayesianPhysicalParameters(50.0, 1.4, 1.5, 0.1), _prior(free_shift=False)
        )


def test_processed_integer_like_intensity_cannot_enter_bayesian_poisson_context() -> None:
    time = np.linspace(0.0, 8.0, 161)
    prepared = prepare_irf(
        generate_gaussian_irf_profile(time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4),
        time,
    )
    processed = TCSPCMeasurement(
        time_ns=time,
        values=np.ones_like(time, dtype=np.float64),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    with pytest.raises(InvalidMeasurementError, match="raw photon counts"):
        BayesianModelContext(processed, prepared_irf=prepared)


def test_explicit_prepared_irf_takes_precedence_and_keeps_provenance() -> None:
    time = np.linspace(0.0, 8.0, 161)
    attached = _attached_irf(time)
    measurement = _raw_measurement(attached_irf=attached)
    source = generate_emg_irf_profile(
        time, gaussian_centre_ns=1.2, gaussian_fwhm_ns=0.4,
        tail_time_ns=0.35, provenance={"assumption": "deliberately_different"},
    )
    prepared = prepare_irf(source, time)
    context = BayesianModelContext(
        measurement,
        prepared_irf=prepared,
        irf_model_relation=IRFModelRelation.DELIBERATELY_MISSPECIFIED,
    )

    assert context.irf_selection == "explicit_prepared"
    assert context.irf_source is source
    assert context.irf_source_kind is IRFSourceKind.SYNTHETIC_EMG
    assert context.irf_model_relation is IRFModelRelation.DELIBERATELY_MISSPECIFIED
    assert context.prepared_irf.diagnostics is prepared.diagnostics
    assert context.irf_source.provenance["assumption"] == "deliberately_different"
    np.testing.assert_array_equal(context.irf_kernel, prepared.kernel)
    np.testing.assert_array_equal(attached.values, measurement.irf.values)
    assert not context.irf_kernel.flags.writeable


def test_attached_sampled_irf_fallback_normalizes_derived_copy() -> None:
    context = _context()
    attached = context.measurement.irf
    assert attached is not None
    original = attached.values.copy()

    assert context.irf_selection == "attached_sampled"
    assert context.irf_source is attached
    assert context.irf_source_kind is IRFSourceKind.IMPORTED_SAMPLED
    assert context.irf_source.provenance["origin"] == "imported trace"
    np.testing.assert_array_equal(
        context.irf_kernel, normalize_irf(context.time_ns, attached.values)
    )
    np.testing.assert_array_equal(attached.values, original)
    assert not context.irf_kernel.flags.writeable


def test_missing_and_incompatible_irfs_are_rejected_without_preparation() -> None:
    measurement = _raw_measurement()
    with pytest.raises(InvalidMeasurementError, match="measured IRF"):
        BayesianModelContext(measurement)

    source = generate_gaussian_irf_profile(
        measurement.time_ns, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4
    )
    mismatched = prepare_irf(
        source, measurement.time_ns + 0.01, resampling="linear"
    )
    with pytest.raises(InvalidMeasurementError, match="prepare the IRF explicitly"):
        BayesianModelContext(measurement, prepared_irf=mismatched)
    with pytest.raises(InvalidMeasurementError, match="must be a PreparedIRF"):
        BayesianModelContext(measurement, prepared_irf=object())


@pytest.mark.parametrize("source_kind", ["gaussian", "emg", "imported"])
def test_prepared_kernel_accepts_generic_sources_and_keeps_history(source_kind: str) -> None:
    measurement = _raw_measurement()
    time = measurement.time_ns
    if source_kind == "gaussian":
        source = generate_gaussian_irf_profile(
            time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4,
            provenance={"origin": "synthetic"},
        )
        expected_kind = IRFSourceKind.SYNTHETIC_GAUSSIAN
    elif source_kind == "emg":
        source = generate_emg_irf_profile(
            time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4,
            tail_time_ns=0.35, provenance={"origin": "synthetic"},
        )
        expected_kind = IRFSourceKind.SYNTHETIC_EMG
    else:
        sampled = SampledIRF(
            time_ns=time,
            values=generate_gaussian_irf(time, centre=1.0, fwhm=0.4),
            provenance={"origin": "empirical"},
        )
        source = irf_profile_from_sampled_irf(sampled)
        expected_kind = IRFSourceKind.IMPORTED_SAMPLED
    prepared = prepare_irf(source, time)
    model_relations = {
        "gaussian": IRFModelRelation.MATCHED,
        "emg": IRFModelRelation.DELIBERATELY_MISSPECIFIED,
        "imported": IRFModelRelation.UNSPECIFIED,
    }
    context = BayesianModelContext(
        measurement, prepared_irf=prepared,
        irf_model_relation=model_relations[source_kind],
    )

    assert context.irf_source_kind is expected_kind
    assert context.irf_source is source
    assert context.prepared_irf.diagnostics.operations == prepared.diagnostics.operations
    np.testing.assert_array_equal(context.irf_kernel, prepared.kernel)


def test_leading_edge_proxy_remains_a_fixed_labelled_assumption() -> None:
    time = np.arange(0.0, 5.05, 0.05)
    counts = np.rint(
        10.0 + 300.0 * np.exp(-0.5 * ((time - 2.2) / 0.3) ** 2)
    ).astype(np.int64)
    measurement = TCSPCMeasurement(
        time_ns=time, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id="proxy-test",
    )
    result = estimate_irf_from_leading_edge(
        measurement,
        background_window_ns=(0.0, 1.0),
        rising_edge_window_ns=(1.0, 3.0),
        smoothing_window_bins=9,
        polynomial_order=3,
    )
    assert result.profile is not None
    prepared = prepare_irf(result.profile, time)
    context = BayesianModelContext(measurement, prepared_irf=prepared)

    assert context.irf_source_kind is IRFSourceKind.LEADING_EDGE_ESTIMATE
    assert context.irf_source.provenance["derived_from"] == "fluorescence_histogram"
    assert context.irf_source.provenance["measurement_sample_id"] == "proxy-test"
    assert context.irf_model_relation is IRFModelRelation.UNSPECIFIED
    assert context.prepared_irf is prepared


def test_importing_base_package_and_deterministic_bayesian_module_needs_no_emcee() -> None:
    script = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'emcee' or name.startswith('emcee.'):
        raise ModuleNotFoundError('emcee deliberately blocked')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import tcspc_toolkit
from tcspc_toolkit.bayesian import GammaPrior
assert GammaPrior(2.0, 1.0).shape == 2.0
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_amplitude_slice_agrees_with_independent_physical_space_quadrature() -> None:
    time = np.linspace(0.0, 4.0, 9)
    attached = SampledIRF(time_ns=time, values=[0, 1, 0, 0, 0, 0, 0, 0, 0])
    measurement = TCSPCMeasurement(
        time_ns=time, values=[0, 1, 2, 1, 0, 1, 0, 0, 0],
        data_kind=MeasurementDataKind.RAW_COUNTS, irf=attached,
    )
    context = BayesianModelContext(measurement)
    prior = BayesianPriorConfig(
        amplitude=GammaPrior(2.0, 1.0),
        lifetime_ns=LogNormalPrior(0.0, 1.0),
        background_per_bin=GammaPrior(2.0, 1.0),
        temporal_shift_prior=None,
        fixed_temporal_shift_ns=0.0,
    )
    lifetime_ns = 1.0
    background = 0.6
    unit_signal = monoexponential_reconvolution_expected_counts(
        time, context.irf_kernel, amplitude=1.0, lifetime=lifetime_ns,
        background=0.0, temporal_shift=0.0,
    )

    def physical_integrand(amplitude: float) -> float:
        expected = amplitude * unit_signal + background
        reduced_poisson_log_likelihood = float(
            np.sum(measurement.values * np.log(expected) - expected)
        )
        return float(gamma.pdf(amplitude, a=2.0, scale=1.0)) * math.exp(
            reduced_poisson_log_likelihood
        )

    fixed_terms = (
        prior.lifetime_ns.log_density(lifetime_ns)
        + prior.background_per_bin.log_density(background)
        + math.log(lifetime_ns)
        + math.log(background)
    )

    def transformed_integrand(log_amplitude: float) -> float:
        coordinates = [log_amplitude, math.log(lifetime_ns), math.log(background)]
        return math.exp(log_posterior(coordinates, context, prior) - fixed_terms)

    physical_integral = quad(physical_integrand, 0.0, 30.0)[0]
    transformed_integral = quad(transformed_integrand, math.log(1e-12), math.log(30.0))[0]
    assert transformed_integral == pytest.approx(physical_integral, rel=1e-5)
