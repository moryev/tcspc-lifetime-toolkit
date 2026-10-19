"""Real Bayesian Poisson inference; install .[bayesian], then allow about 1-2 minutes.

The budget is bounded; diagnostic acceptance is not guaranteed on every platform.
No diagnostic threshold is relaxed. This is one synthetic example, not calibration.
"""

import math
import numpy as np

from tcspc_toolkit import (
    MeasurementDataKind, TCSPCMeasurement, build_expected_counts_from_irf,
    generate_gaussian_irf_profile, monoexponential_decay, prepare_irf, sample_photon_counts,
)
from tcspc_toolkit.estimation.bayesian.posterior import (
    BayesianPriorConfig, GammaPrior, IRFModelRelation, LogNormalPrior,
    fit_bayesian_monoexponential_reconvolution,
)
from tcspc_toolkit.estimation.bayesian.sampling import BayesianSamplingConfig, BayesianSamplingStatus


def build_inputs():
    """Build explicit synthetic data, priors and budget without importing emcee."""
    time_ns = np.linspace(0.0, 16.0, 161)
    prepared = prepare_irf(generate_gaussian_irf_profile(
        time_ns, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.3,
    ), time_ns)
    expected = build_expected_counts_from_irf(
        monoexponential_decay(time_ns, 1.0, 2.5), prepared,
        signal_photon_count=20_000, background_per_bin=2.0,
    )
    measurement = TCSPCMeasurement(
        time_ns=time_ns, values=sample_photon_counts(expected, np.random.default_rng(1401)),
        data_kind=MeasurementDataKind.RAW_COUNTS, sample_id="bayesian-synthetic-example",
        provenance={"origin": "synthetic demonstration", "observation_seed": 1401},
    )
    # Explicit illustrative physical priors, not inferred from this observation.
    # Gamma uses shape/rate; amplitude is a reconvolution scale, not total photons.
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(shape=1.0, rate=0.0004),
        lifetime_ns=LogNormalPrior(log_mean=math.log(2.0), log_std=0.8),
        background_per_bin=GammaPrior(shape=1.2, rate=0.24),
        temporal_shift_prior=None, fixed_temporal_shift_ns=0.0,
    )
    config = BayesianSamplingConfig(
        random_seed=1402, n_walkers=16, n_ensembles=2, warmup_steps=250,
        production_steps=1000, max_production_steps=10_000, extension_steps=500,
        credible_interval_level=0.90,
    )  # All operational diagnostic thresholds retain their library defaults.
    return measurement, prepared, priors, config


def run_example():
    measurement, prepared, priors, config = build_inputs()
    print("Sampling a synthetic histogram with explicit priors, fixed IRF and fixed zero shift...", flush=True)
    run = fit_bayesian_monoexponential_reconvolution(
        measurement, prepared_irf=prepared, priors=priors, sampling_config=config,
        irf_model_relation=IRFModelRelation.MATCHED,
    )
    result = run.result
    print(f"Sampling status: {result.status.value}; elapsed: {result.runtime.total_seconds:.1f} s")
    print(f"Production steps per ensemble: {result.diagnostics.production_steps}")
    print(f"Diagnostic limitations: {result.diagnostics.failure_reasons or 'none reported'}")
    if result.status is not BayesianSamplingStatus.SUCCESS:
        print("PROVISIONAL: sampling diagnostics were not accepted; summaries below are not validated.")
    for summary in result.parameter_summaries:
        if summary.name == "lifetime_ns":
            print(f"Synthetic lifetime: 2.5000 ns; posterior median: {summary.median:.4f} ns")
            print(f"90% credible interval: [{summary.credible_lower:.4f}, {summary.credible_upper:.4f}] ns")
    print("Conditional on the stated prior, mono-exponential model, fixed IRF and alignment.")
    print("Accepted sampling does not establish physical-model correctness or interval calibration.")
    if result.status in {BayesianSamplingStatus.INITIALIZATION_FAILED, BayesianSamplingStatus.NUMERICAL_FAILURE}:
        raise RuntimeError(f"Inference failed: {result.failure_reason}")
    return run


if __name__ == "__main__":
    try:
        run_example()
    except ImportError as error:
        raise SystemExit(str(error)) from error
