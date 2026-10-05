# Design principles

1. Mathematical models belong in models.py.
2. Simulation workflows belong in simulation.py.
3. IRFs are generated independently from decay models.
4. Convolution is an independent operation.
5. Noise sampling is independent from signal generation.
6. Evaluation functions are organized by evaluated quantity, not by algorithm.
7. Public APIs use explicit names:
   monoexponential_decay()
   simulate_monoexponential_decay()
   fit_monoexponential_decay()
8. Convenience wrappers may compose lower-level functions.
9. Preprocessing is analysis-dependent. The toolkit should provide reusable transformations, not impose one universal pipeline. There is no single universally correct TCSPC preprocessing pipeline.
10. Imported experimental histograms preserve explicit time units, raw-count versus processed-intensity semantics, and source provenance. Internal time coordinates are in nanoseconds.
11. A sampled measured IRF retains its imported values and provenance separately from the decay histogram. Reconvolution normalizes a compatible IRF for forward modelling without changing the imported trace.
12. Experimental estimation does not imply known ground truth. Truth-based error metrics require an explicitly supplied trusted reference; synthetic generating parameters and experimental reference values have different meanings.

## Estimator extension boundary — Issue #2

The dependency direction is explicit (arrows mean "depends on"):

```text
benchmark orchestration
        ↓
canonical benchmark estimator specifications
        ↓
generic estimator execution
```

`estimator_api.py` defines the structural `RegressorProtocol`, immutable
`EstimatorSpec`, and generic fitting/prediction functions. It does not know
canonical estimator identities or representation names. `ml_models.py`
constructs canonical specifications from the existing factories, preserving
hyperparameters and seeds; generalization orchestration uses them by default.
User-defined specifications can also call generic execution directly.

Callers own prepared representations, feature meanings, positional sample
alignment, and train/calibration/test roles. Generic execution neither fits
representations nor computes benchmark metrics. The curated root API and
supported module-qualified workflows are documented in [the API contract](../api.md).
This boundary does not consolidate evaluation/report types (#11) or move
package files (#3).

## Generalized IRFs — Issue #8

IRF origin is independent of the numerical forward model:

```text
Gaussian / EMG generation ───────────────┐
SampledIRF → explicit source adapter ───┼→ IRFProfile
explicit leading-edge proxy ────────────┘       ↓
                                       prepare_irf(...)
                                              ↓
                                         PreparedIRF
                                              ↓
                             array-based convolution / reconvolution
```

### Source and import boundaries

`SampledIRF` remains the narrow Issue-#6 imported sampled-IRF carrier;
`TCSPCMeasurement.irf` retains that path and its compatible-grid rules.
`irf_profile_from_sampled_irf()` copies the original coordinates, values,
metadata, and provenance without numerical transformation. Its
`imported_sampled` classification does not establish independent physical
measurement of an IRF.

`IRFProfile` is the generalized sampled source representation in `irf.py`.
Its source kind distinguishes `synthetic_gaussian`, `synthetic_emg`,
`imported_sampled`, and `leading_edge_estimate`. It validates finite,
strictly increasing coordinates, finite nonnegative samples, and positive
finite trapezoidal area; nonuniform source grids are allowed. Arrays are
defensively copied and read-only; parameter/metadata/provenance mappings are
defensively copied following the measurement-carrier convention. Construction
does not normalize, shift, resample, smooth, or baseline-correct the source.

Gaussian and EMG profile factories record generation parameters. The EMG's
Gaussian-component centre and FWHM are not its final peak and full FWHM.
A positive tail time produces positive skew; exactly zero tail delegates to
the legacy Gaussian generator. Existing Gaussian simulation wrappers, seeds,
and operation order are unchanged.

### Explicit preparation

`prepare_irf()` in `irf_preparation.py` validates the source and uniform target
grid, applies caller-supplied registration to source coordinates, checks grid
compatibility or explicitly resamples, calculates diagnostics, and normalizes
the derived target-grid kernel with `normalize_irf()`.

- Positive `registration_offset_ns` moves the IRF later. Registration is a
  known coordinate offset, not inferred zero time.
- `resampling="none"` requires compatible registered grids; it never
  interpolates silently. `"linear"` performs deterministic linear
  interpolation, combining registration and resampling in one operation,
  with zero outside the available source domain.
- `PreparedIRF` keeps the source profile, read-only target coordinates and
  normalized kernel, and `IRFPreparationDiagnostics`. Transformation history
  belongs in diagnostics, not in rewritten source provenance.
- Geometric support loss integrates the registered piecewise-linear source
  over the target window, including partial boundary intervals. It is
  distinct from target-grid quadrature/interpolation area change. Diagnostics
  know only the available source trace, not physical tails outside it.
- Zero/nonfinite resulting area fails. Support loss is reported without a
  universal scientific tolerance; `max_support_loss_fraction` is an optional
  caller policy. Sampling/FWHM diagnostics do not certify physical adequacy.

Registration, resampling, and the fitted residual temporal shift are separate
operations. Preparation does not change optimizer initialization or fitted
shift semantics, and never calls automatic peak alignment.

### Forward-model boundaries

`build_expected_counts_from_irf()` in `simulation.py` unwraps a prepared
kernel for the existing array-based convolution, scales the finite-window
signal sum to `signal_photon_count`, and adds `background_per_bin`.
`sample_photon_counts()` remains a separate Poisson-sampling operation.

`fit_experimental_reconvolution(..., prepared_irf=...)` uses an explicitly
supplied prepared kernel in preference to an attached `SampledIRF`. The
target grid must already match the measurement; this adapter never prepares,
resamples, or registers an IRF. With no explicit prepared kernel, the original
Issue-#6 path normalizes a derived copy of the attached IRF as before.
Raw-count guards, original measurements, provenance, and result types remain
unchanged. `convolution.py` and `fitting.py` consume arrays, not IRF objects.

### Explicit approximate estimation and scientific evaluation

`estimate_irf_from_leading_edge()` in `irf_estimation.py` requires raw counts,
a uniform grid, explicit half-open background/rising-edge windows, and
explicit Savitzky–Golay settings. It preserves negative background-subtracted
fluctuations through smoothing, constructs a nonnegative pre-peak derivative
proxy, records discarded negative derivative area, and returns either a
normalized `leading_edge_estimate` profile or an explained numerical failure.

For mono-exponential fluorescence, `S'(t) = A h(t) - S(t)/tau`; the omitted
`S/tau` term is not estimated. A constructible proxy is not a recovered true
or measured IRF. The caller must explicitly estimate, prepare, and supply it
to a fitter. **No automatic leading-edge fallback occurs.**

`irf_evaluation.py` composes existing simulation, classical fitting, metrics,
ML, and Week-9 uncertainty APIs for paired shape-family mismatch and proxy
failure studies. It uses generic benchmark carriers, not frozen Week-8 A–F
definitions. Gaussian comparison kernels are matched to noiseless EMG peak
and full sampled FWHM, not fitted to fluorescence outcomes. ML fitting is
development-only. Classical uncertainty conditions on the assumed fixed
IRF; RF spread remains a disagreement score. Bi-exponential and finite-rise
controls have no unique mono-exponential target.

Notebook 16 is presentation and orchestration of these library APIs. Its
plain JSON manifest, `configs/issue8_irf_workflow.json`, fixes the audited
demonstration settings without adding a configuration framework. Detailed
results and fixture limitations belong in `docs/scientific_findings.md`.
Issue #2 stabilizes the API and extension contracts described above;
evaluation consolidation remains Issue #11.
