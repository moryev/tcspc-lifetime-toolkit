"""Model-conditional Fisher covariance and Poisson bootstrap for one synthetic curve."""

import numpy as np

from tcspc_toolkit import (
    build_expected_counts_from_irf, fit_monoexponential_reconvolution,
    generate_gaussian_irf_profile, monoexponential_decay, prepare_irf, sample_photon_counts,
)
from tcspc_toolkit.estimation.classical import estimate_reconvolution_initial_guess
from tcspc_toolkit.uncertainty.classical import (
    estimate_parametric_poisson_bootstrap, estimate_poisson_reconvolution_local_covariance,
)


def run_example():
    time_ns = np.linspace(0.0, 20.0, 256)
    prepared = prepare_irf(generate_gaussian_irf_profile(
        time_ns, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.3,
    ), time_ns)
    expected = build_expected_counts_from_irf(
        monoexponential_decay(time_ns, 1.0, 2.5), prepared,
        signal_photon_count=20_000, background_per_bin=2.0,
    )
    counts = sample_photon_counts(expected, np.random.default_rng(1404))
    guess = estimate_reconvolution_initial_guess(time=time_ns, counts=counts, irf=prepared.kernel)
    shift_bounds = (-0.5, 0.5)
    fit = fit_monoexponential_reconvolution(
        time_ns, counts, prepared.kernel,
        (guess.amplitude, guess.lifetime_ns, guess.background, guess.temporal_shift_ns),
        temporal_shift_bounds=shift_bounds, objective="poisson",
    )
    if not fit.success:
        raise RuntimeError("Primary fit failed numerical validation")
    covariance = estimate_poisson_reconvolution_local_covariance(
        time=time_ns, irf=prepared.kernel, fit_result=fit, temporal_shift_bounds=shift_bounds,
    )
    bootstrap = estimate_parametric_poisson_bootstrap(
        time=time_ns, irf=prepared.kernel, fit_result=fit, temporal_shift_bounds=shift_bounds,
        rng=np.random.default_rng(1405), n_resamples=100, nominal_coverage=0.90,
    )
    print(f"Synthetic lifetime: 2.5000 ns; estimate: {fit.lifetime:.4f} ns")
    print(f"Local SD: {covariance.lifetime_std:.4f} ns; valid: {covariance.covariance_valid}")
    print(f"Bootstrap 90% percentile interval: [{bootstrap.lower_ns:.4f}, {bootstrap.upper_ns:.4f}] ns")
    print(f"Bootstrap valid: {bootstrap.bootstrap_valid}; successful refits: "
          f"{bootstrap.n_successful_fits}/{bootstrap.n_resamples}; boundary hits: {bootstrap.n_boundary_hits}")
    print(f"Failure reasons: covariance={covariance.failure_reason}; bootstrap={bootstrap.failure_reason}")
    print("One observation does not establish coverage. Both methods assume a mono-exponential model")
    print("and fixed IRF; bootstrap samples the fitted model and cannot repair physical model mismatch.")
    return fit, covariance, bootstrap


if __name__ == "__main__":
    run_example()
