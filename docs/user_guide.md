# User guide: choosing a scientifically appropriate workflow

Use the [example index](../examples/README.md) for complete runnable commands.
This guide explains choices and assumptions; the [API contract](api.md) defines
supported imports, and [scientific findings](scientific_findings.md) records evidence.

## Simulation and reconvolution

The [quick start](../examples/quickstart.py) separates deterministic physics from
photon sampling. On a uniform time grid in nanoseconds, its model is

```math
I_\tau(t)=\exp(-t/\tau),\qquad
\mu_i=A\,(h_{\Delta t}*I_\tau)(t_i)+B,\qquad
N_i\sim\mathop{\text{Poisson}}(\mu_i).
```

Here `h` is a unit-temporal-area IRF, `A` scales the convolved unit-amplitude
decay, `B` is expected detector background **per bin**, and a positive residual
shift moves the IRF later. Numerical convolution includes the bin-width factor
and retains the finite observation window. Add background **after** convolution.
For a Gaussian, FWHM equals `2 * sqrt(2 * log(2)) * sigma`.

There are two useful signal-scale conventions:

| Interface | Scale convention |
| --- | --- |
| `build_expected_counts_from_irf()` | Convolve an ideal decay, scale its finite-window sum to `signal_photon_count`, then add `background_per_bin`. |
| `physics.forward_model.monoexponential_reconvolution_expected_counts()` and the fitter | `amplitude` multiplies the convolved unit-amplitude decay; it is **not** total expected signal photons. |

The quick start uses the first convention for simulation and the second for
inference. The existing histogram-derived initialization helper computes a
compatible amplitude guess; generating parameters are not supplied to the fit.
`sample_photon_counts()` performs the separate Poisson draw with an explicit RNG.
The high-level `simulation.histograms.simulate_irf_convolved_histogram()` composes
the Gaussian workflow when a caller needs many synthetic observations.

`fit_monoexponential_reconvolution()` consumes arrays and an already-normalized
IRF on the same grid. Explicitly select `objective="poisson"`; the default is
least squares. It estimates amplitude, lifetime, background, and residual shift.
Classical fitting, local uncertainty, and Bayesian inference share the same
physical expected-count model. Fit success includes numerical checks but does
not certify global optimality or physical model correctness.

Keep raw observed counts for Poisson fitting. Background estimates may initialize
the fitted background; subtracting them from observations changes the count
model. Normalizing observed counts likewise invalidates their raw Poisson meaning.
Never infer raw-count semantics solely from an array's integer dtype.

Preserve enough time support to represent the IRF and decay tail. Cropping inputs
before convolution can introduce edge effects; it is different from evaluating
a fitting window after constructing the complete forward model. Shifted kernels
can lose support at finite-window boundaries. The shared forward model does not
renormalize the IRF after shifting.

## Measurements and IRF preparation

[experimental_csv.py](../examples/experimental_csv.py) imports a histogram whose
times are in picoseconds and an IRF whose times are in nanoseconds. Units are
declared, never inferred. `TCSPCMeasurement` stores internal nanoseconds, values,
data kind, metadata, and provenance. It has no intrinsic experimental truth.

The [committed CSVs](../data/examples/README.md) are **synthetic stand-ins**. Their
generator settings explain fixture provenance, not a trusted experimental
reference. For acquired measurements, supply your own acquisition-compatible IRF
and time calibration. Reference-based evaluation is a separate explicit operation
through `evaluate_estimate_against_reference()`.

Import uses named columns in a headered CSV, one histogram at a time. Declare
`raw_counts` for non-negative exact integer counts within `int64` range, or
`processed_intensity` for transformed values. Raw-Poisson methods call the
measurement's raw-count guard; processed measurements do not enter that path.
Imported histogram time grids must be uniform.

IRF source, preparation, and use have distinct meanings:

1. `SampledIRF` preserves an imported source trace. Conversion through
   `irf_profile_from_sampled_irf()` creates a source profile without transforming
   its values. Importing a trace does not prove it was independently measured.
2. `IRFProfile` records a Gaussian, EMG, imported sampled, or leading-edge-proxy
   origin. For an EMG, Gaussian-component center/FWHM differ from the full
   profile's peak/FWHM.
3. `prepare_irf()` creates a derived `PreparedIRF` kernel and diagnostics.
   `resampling="none"` requires matching registered grids. Choose `"linear"`
   explicitly for different grids; extrapolated samples are zero.
4. `fit_experimental_reconvolution(..., prepared_irf=...)` consumes an already
   prepared kernel. This explicit kernel takes precedence over an attached
   sampled IRF; the adapter performs no automatic preparation.

`registration_offset_ns` is a known coordinate correction; positive offsets move
the source later. It is distinct from the residual shift fitted during inference.
The experimental adapter currently initializes residual shift at zero, so its
bounds must include zero. The original attached, same-grid sampled-IRF path also
remains supported.

Inspect support-loss, grid-spacing, FWHM sampling, and normalization diagnostics.
Support loss describes the supplied source window, not unrecorded physical tails.
The original trace is preserved; normalization changes a derived kernel.
Preparation does not automatically infer zero time, smooth, or remove baseline.
The [design contract](design/master_design_document.md#generalized-irfs--issue-8)
defines these operations in detail.

Leading-edge IRF estimation is an explicit, approximate alternative requiring
specified windows and smoothing settings. A constructible, warning-free proxy
can be physically wrong. See [Notebook 16](../notebooks/16_generalized_irf_and_failure_awareness.ipynb)
and the [IRF findings](scientific_findings.md#post-week-9--issue-8-generalized-irfs-and-failure-awareness).

## Preprocessing and machine learning

There is no universal preprocessing pipeline. Validation, background estimation,
coordinate alignment, cropping, rebinning, and normalization are separate public
operations. Alignment changes coordinates without interpolating measured counts;
integer-factor rebinning preserves counts over a divisible selected region.
Normalization removes count scale, and background subtraction may legitimately
produce negative fluctuations in a derived representation.

[preprocessing_features.py](../examples/preprocessing_features.py) extracts raw
histogram features using explicit early, late, and tail windows. Feature extraction
does not silently subtract background, align, crop, rebin, or normalize the input.
Photon-arrival moments and tail descriptors therefore include acquisition effects;
they are not automatically corrected physical lifetimes.

The main ML representations retain different information:

| Representation | Interpretation |
| --- | --- |
| Engineered features | Count, moment, quantile, half-decay, tail, and early/late descriptors with an explicit feature schema. |
| TOTAL-normalized histograms | Temporal shape with absolute photon-count scale removed. |
| PCA-compressed histograms | Training-fitted variance directions; explained variance is not evidence of lifetime-predictive information. |

[ml_lifetime_prediction.py](../examples/ml_lifetime_prediction.py) uses the existing
TOTAL-normalization → training-fitted scaler → Ridge pipeline. Split before
fitting any learned transformation. Repeated noisy realizations of the same
physical condition require group-aware splits when assessing new-condition
generalization; this small example draws independent conditions instead.

The frozen A–F suite is external to training, representation fitting, selection,
tuning, and calibration. It is not the small example's test split. Experimental
ML adapters transform/predict using already-fitted development artifacts and a
compatible grid; they must not refit on an imported measurement. Synthetic
accuracy alone does not validate synthetic-to-real transfer.

For user-defined regressors, start with [custom_estimator.py](../examples/custom_estimator.py)
and the [extension contract](api.md#generic-estimator-extension-api). The caller
owns representations and positional row alignment; factories create fresh models.

## Uncertainty and model validity

| Output | What it describes |
| --- | --- |
| Local Poisson/Fisher covariance | Local statistical parameter uncertainty, conditional on the model and fixed IRF; may be invalid at boundaries or under poor conditioning. |
| Parametric Poisson bootstrap | Refit variability from draws of the fitted model; failed refits and interval availability must be reported. |
| Repeated-Poisson empirical variability | Sampling variability across independent observations of a fixed synthetic condition; a calibration reference, not a per-observation interval. |
| Quantile / conformal prediction intervals | Predictive intervals fitted/calibrated on separate development populations; shift can invalidate the intended coverage interpretation. |
| RF tree / training-bootstrap spread | Disagreement or sensitivity scores, not automatically nominal intervals or calibrated standard deviations. |
| Bayesian credible intervals | Posterior probability conditional on the likelihood, explicit prior, fixed IRF, and alignment assumptions. |

[uncertainty_example.py](../examples/uncertainty_example.py) reports a local SD and
a bootstrap percentile interval from one synthetic observation. One interval
cannot establish empirical coverage; a small bootstrap sample adds Monte Carlo
variation. Bootstrap cannot repair misspecification because it samples the same
fitted model.

**An estimator can correctly quantify statistical uncertainty while remaining
confidently wrong because its physical model is incomplete.** Narrow intervals,
small disagreement, low Poisson deviance, and posterior-predictive checks are not
universal model-validity tests. The [results guide](results.md) links evidence
without mixing physical generating parameters, experimental references, and
model-conditional pseudo-true projections.

## Bayesian inference

[bayesian_inference.py](../examples/bayesian_inference.py) performs actual sampling
with the optional `emcee` dependency. Its three inferred parameters are amplitude,
lifetime, and constant background. Residual shift is explicitly fixed at zero;
the IRF and alignment are known assumptions of this synthetic demonstration.
The library also supports a free residual shift with an explicit bounded prior.

The example specifies physical-space Gamma shape/rate priors for amplitude and
background and a log-normal lifetime prior. They are illustrative assumptions,
not universally suitable priors or values learned from the observation. Sampling
uses two independent ensembles and unchanged library diagnostic thresholds.

The default budget targets about 1–2 minutes. It can finish with
`insufficient_sampling`; any summaries are then labeled **provisional**, with
the diagnostic reasons printed. Successful numerical sampling still does not
prove the physical model or calibrate credible intervals. IRF-shape uncertainty
is not propagated. [Notebook 17](../notebooks/17_bayesian_poisson_inference_and_model_mismatch.ipynb)
adds posterior correlations and predictive checks, with
[saved-artifact prerequisites](reproducibility.md#committed-references-and-manifests).

## Persistence and limitations

Use [persistence_roundtrip.py](../examples/persistence_roundtrip.py) to create a
new SQLite file, record scientific context, close the writer, and retrieve facts
through read-only queries, parameterized SQL, and pandas. Its classical result is
computed; its Bayesian records are explicitly labeled fixtures with no MCMC.
The [persistence contract](persistence.md) is the authoritative schema-v1 reference.

Current limits include single-curve CSV import rather than vendor formats;
mono-exponential inverse fitting despite multi-component simulation; constant
background; no pile-up, dead-time, or afterpulsing model; no blind IRF recovery;
and no validated synthetic-to-real or arbitrary-shift uncertainty guarantee.
The toolkit is research software, not a validated replacement for experimental
analysis systems. Purcell sensing remains deferred until realistic inputs exist.
