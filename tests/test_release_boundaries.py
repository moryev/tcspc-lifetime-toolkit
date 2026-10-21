"""Failure semantics at scientific boundaries, without new numerical policy."""

from dataclasses import replace

import numpy as np
import pytest

from tcspc_toolkit.estimation.bayesian.sampling import (
    BayesianSamplingConfig,
    _assess_sampling_diagnostics,
)
from tcspc_toolkit.estimation.fitting import poisson_negative_log_likelihood
from tcspc_toolkit.evaluation.diagnostics import (
    calculate_reduced_residuals,
    calculate_residuals,
)
from tcspc_toolkit.irf.preparation import prepare_irf
from tcspc_toolkit.irf.profiles import generate_gaussian_irf_profile


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["observed", "expected"])
def test_poisson_objective_rejects_invalid_counts(bad, side):
    inputs = {"observed": [1.0, 2.0], "expected": [1.0, 2.0]}
    inputs[side][1] = bad
    with pytest.raises(ValueError, match=side):
        poisson_negative_log_likelihood(**inputs)


@pytest.mark.parametrize(
    "kind", ["shape", "dimension", "negative", "nan", "infinite", "area", "zero"]
)
def test_prepared_irf_constructor_rejects_invalid_kernel(kind):
    time = np.linspace(0, 10, 101)
    prepared = prepare_irf(
        generate_gaussian_irf_profile(
            time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.5
        ),
        time,
    )
    kernel = prepared.kernel.copy()
    if kind == "shape":
        kernel = kernel[:-1]
    elif kind == "dimension":
        kernel = kernel[None, :]
    elif kind in ("negative", "nan", "infinite"):
        kernel[5] = {"negative": -1, "nan": np.nan, "infinite": np.inf}[kind]
    elif kind == "area":
        kernel *= 2
    else:
        kernel[:] = 0
    with pytest.raises(ValueError, match="kernel"):
        replace(prepared, kernel=kernel)
    # Failed construction must not mutate the already prepared/source objects.
    assert np.trapezoid(prepared.kernel, x=time) == pytest.approx(1)
    assert not prepared.kernel.flags.writeable


def test_autocorrelation_exception_is_an_unaccepted_diagnostic():
    from emcee.autocorr import AutocorrError

    transformed = np.random.default_rng(128).normal(size=(2, 100, 8, 3))
    physical = np.zeros((2, 100, 8, 4))
    physical[..., :3] = np.exp(transformed)
    config = BayesianSamplingConfig(
        random_seed=9,
        n_walkers=8,
        warmup_steps=10,
        production_steps=100,
        max_production_steps=100,
        extension_steps=10,
        n_ensembles=2,
    )

    def unavailable(chain):
        raise AutocorrError(np.ones(3), "Deliberate short-chain diagnostic failure")

    diagnostics = _assess_sampling_diagnostics(
        transformed,
        physical,
        np.full((2, 8), 0.45),
        config,
        extension_count=0,
        autocorrelation_estimator=unavailable,
    )
    assert not diagnostics.accepted
    assert "autocorrelation estimate unavailable" in diagnostics.failure_reasons
    assert np.isnan(diagnostics.autocorrelation_time_steps).all()
    assert np.isnan(diagnostics.approximate_effective_samples).all()


def test_residual_sign_and_poisson_floor_are_explicit():
    observed = np.array([0, 4, 1])
    fitted = np.array([0.0, 1.0, 4.0])
    np.testing.assert_array_equal(calculate_residuals(observed, fitted), [0, 3, -3])
    np.testing.assert_allclose(
        calculate_reduced_residuals(observed, fitted), [0, 3, -1.5]
    )
    np.testing.assert_allclose(
        calculate_reduced_residuals(observed, fitted, minimum_expected_count=4),
        [0, 1.5, -1.5],
    )


@pytest.mark.parametrize(
    "observed,fitted,message",
    [
        (np.ones((2, 1)), np.ones(2), "observed"),
        (np.ones(2), np.ones((2, 1)), "fitted"),
        (np.ones(2), np.ones(3), "same shape"),
    ],
)
@pytest.mark.parametrize("function", [calculate_residuals, calculate_reduced_residuals])
def test_residual_helpers_reject_incompatible_histograms(
    observed, fitted, message, function
):
    with pytest.raises(ValueError, match=message):
        function(observed, fitted)


@pytest.mark.parametrize("floor", [0, -1])
def test_pearson_residual_floor_must_be_positive(floor):
    with pytest.raises(ValueError, match="positive"):
        calculate_reduced_residuals(np.ones(2), np.ones(2), floor)
