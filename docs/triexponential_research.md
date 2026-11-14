# Experimental K=3 reconvolution for research

K=3 fitting is experimental and intended for exploratory research. It fits three
nonnegative pre-exponential amplitudes, three bounded positive lifetimes, one
nonnegative background per bin and one shared bounded IRF shift. Returned
components are ordered by lifetime, with descending amplitude for exact ties.
Exact-zero amplitudes/background and equal lifetimes remain supported.

## Fit a measured histogram

The module-qualified measurement adapter follows the established K=2 convention.
Its `.fit` is the existing `MultiexponentialFitResult`; `.context` retains defensive
measurement and selected-IRF snapshots. Rejected fits and finite candidates retain
their numerical records.

```python
from tcspc_toolkit.estimation.triexponential import TriexponentialFitSettings
from tcspc_toolkit.irf.preparation import irf_profile_from_sampled_irf, prepare_irf
from tcspc_toolkit.measurement.triexponential import fit_experimental_triexponential_reconvolution
from tcspc_toolkit.measurement.io import load_sampled_irf_csv, load_tcspc_measurement_csv

measurement = load_tcspc_measurement_csv(
    "decay.csv", time_unit="ns", time_column="time_ns", data_kind="raw_counts",
)
sampled = load_sampled_irf_csv("irf.csv", time_unit="ns", time_column="time_ns")
prepared = prepare_irf(
    irf_profile_from_sampled_irf(sampled), measurement.time_ns,
    resampling="none", registration_offset_ns=0,
)
settings = TriexponentialFitSettings(
    max_seed_starts=16, reserved_recovery_starts=4,
    max_optimization_calls=20_000, wall_seconds=60,
    policy="compact", solver="trf",
)
result = fit_experimental_triexponential_reconvolution(
    measurement, prepared_irf=prepared, settings=settings,
    lifetime_bounds_ns=(0.1, 10), temporal_shift_bounds_ns=(-0.3, 0.3),
)
fit = result.fit
print(fit.parameters)
print(fit.numerical.accepted, fit.numerical.reasons)
print(fit.identifiability)
```

The bounds and settings above are explicit demonstration choices. Choose physical
bounds appropriate to the acquisition. The core fitter still requires explicit
settings. An explicit compatible `PreparedIRF` takes precedence. Alternatively,
attach a same-grid `SampledIRF` to `TCSPCMeasurement`; the established resolver
normalizes a defensive copy and records this in `result.context`. Different grids
require explicit preparation and a deliberate resampling/registration choice.

The adapter enforces `require_raw_counts()`. Supply raw histogram counts and their
declared time axis. Processed or normalized observations must retain their
processed-data designation and are rejected. Raw-count guards cannot discover a
false provenance declaration on an already transformed integer-valued array.

## Inspect the existing outputs

| Quantity | Existing field |
|---|---|
| Three lifetimes and amplitudes | `fit.parameters.lifetimes_ns`, `.amplitudes` |
| Background and shift | `fit.parameters.background`, `.temporal_shift_ns` |
| Expected histogram | `fit.expected_counts` |
| Background-free component predictions | `fit.component_counts`, shape `(3, n_bins)` |
| Amplitude, detected-window and ideal photon fractions | `fit.components` |
| Signed Poisson deviance residuals | `fit.statistics.deviance_residuals` |
| Likelihood, deviance, AIC and bin-based BIC | `fit.statistics` |
| Acceptance, fresh assessment and local validation | `fit.numerical` |
| Start termination, recovery, work and finite-best evidence | `fit.optimization` |
| Active bounds, zero amplitudes, equal lifetimes and conditioning availability | `fit.identifiability` |
| Model bounds, settings and prepared-IRF provenance | `fit.assumptions` |
| Measurement provenance and selected source/kernel history | `result.context` |

Check for `None` before using unavailable parameters, predictions or statistics.
Inspect `fit.optimization.best_observed_parameters` and
`best_observed_centered_nll` together with the returned-candidate assessment.
An exploratory best finite probe can have a different validation history.
The adapter preserves both records without promoting candidates.

Amplitudes are pre-exponential scales. Component predictions exclude background;
their sum plus the single background equals the expected histogram. Positive
temporal shift moves the IRF later. Finite-window truncation and bin-width scaling
follow the established forward model. AIC/BIC use eight nominal parameters, even
at singular solutions. Component fractions and local conditioning remain
model-conditional descriptive quantities.

## Reproducible two-case example

```bash
python examples/triexponential_reconvolution.py
python examples/triexponential_reconvolution.py --figure build/triexponential_research.png
```

The example uses 256 bins over approximately 16 ns, a compatible Gaussian IRF,
20,000 expected signal photons, 0.5 background counts/bin and a 0.043 ns shift.
Separated generating lifetimes are `(0.6, 2.0, 4.8)` ns. The difficult case has
coincident lifetimes `(1.5, 1.5, 4.8)` ns. Independent explicit seeds
`[37, 3, 20261113, case_index]` generate the two observations.

Each observation is fitted once using the explicit TRF configuration above.
The 60-second setting is a cooperative fitter deadline. This standalone example
has no external worker supervisor.
Optional plotting reuses those fits and shows observed counts, generating means,
returned expectations, components and signed deviance residuals. The example
prints rejected outcomes as well as accepted outcomes. Runtime and last numerical
digits can depend on the environment.

For coincident components, redistributing the first two amplitudes while keeping
their sum fixed produces the same expected curve to floating-point precision.
The example calculates and prints this alternative directly through the physical
forward model. It demonstrates an exact ambiguity independently of the numerical
fit's outcome. Generating parameters belong to these synthetic teaching inputs;
experimental measurements require separately supplied reference information.

## Scientific limitations

- Numerical acceptance checks the returned candidate under declared local tests.
  Global optimality remains unestablished.
- Similar reconstructed curves can correspond to substantially different
  lifetime/amplitude combinations. Weak, absent or overlapping components can be
  practically unidentifiable. Fitted lifetimes require physical interpretation
  beyond treating each component as a uniquely resolved population.
- Inspect residuals, parameter boundaries, conditioning and alternative plausible
  solutions before drawing physical conclusions. Full numerical rank is a local
  sensitivity property, and physical identifiability requires additional evidence.
- The independent audit produced six finite, freshly assessed and locally
  validated candidates. **Zero of six** satisfied the original strict
  all-starts-complete rule. All 144 starts were attempted; 64 reached evaluation
  limits and 12 stopped on unstable Jacobians. This historical verdict is retained.
- Deterministic close-component and sampled-IRF cases had small curve errors with
  largest lifetime errors around 28% and 97%. General independent optimizer
  qualification and production reliability certification remain outstanding.
- This workflow provides point estimates and diagnostics. Statistical intervals,
  when separately available for a model, require their own assumptions about
  noise, IRF knowledge and model adequacy. This example makes no interval claim.

The inspection, tests and finite operational runs support experimental research
use. The proposed scope amendment records the distinction between the delivered
workflow and deferred production/qualification work.
[Proposed Issue #37 amendment](design/issue37_research_scope_amendment.md)
