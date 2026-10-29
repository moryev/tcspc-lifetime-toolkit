# Public API and estimator extensions

For runnable tasks, start with the [example index](../examples/README.md) and
[user guide](user_guide.md). This page preserves the detailed API contract and
migration history; the [documentation index](README.md) maps the other guides.

The current package version is **0.8.0**. The ordered 66-name root surface,
reviewed canonical interfaces and supported symbol aliases are preserved.
Removed flat paths (and pickles naming their globals) remain unsupported; see
the [migration mapping](package_structure.md#removed-flat-module-mapping).
Release qualification does not introduce a v1.0 API-stability commitment.

This document defines the Issue-#2 public API contract for version 0.7.0:
a curated package-root convenience API, supported module-qualified workflows,
and a generic estimator extension boundary separate from frozen benchmark
configuration. Issue #11 has since added a module-qualified factual evaluation
layer beneath the existing scientific reports. Issue #3 has completed structural
relocation with estimator contracts/baselines, low-level physical primitives,
histogram preprocessing/features/representations, IRF profiles/preparation/
estimation, measurement records/I/O/adapters, generic histogram/dataset
simulation, core classical fitting/execution, Bayesian inference, ML
models/evaluation/cross-validation, historical evaluation diagnostics,
factual point/reference/attachment contracts, conditional analysis, plotting,
frozen Week-8 generalization protocol/data generation and execution/reporting,
uncertainty source/metric/split/provenance contracts and ML/classical methods,
the separate Week-7 mismatch/timing studies, Issue-8 IRF/Issue-4 Bayesian studies,
frozen Week-9 uncertainty orchestration/reporting, and SQLite persistence.

| Status | Surface |
| --- | --- |
| Generic extension API | `RegressorProtocol`, `EstimatorSpec`, `fit_regressors`, and `predict_regressors` are root exports; canonical `estimation.api` imports resolve to the same objects. The old flat `estimator_api` module is removed. |
| Stage-2 integration | Generalization ML fitting and shared final-test prediction consume that API; only the A-F suite entry point gains explicit custom test mappings. |
| Stage-3 naming | Scientific module-qualified names below are canonical; historical Day/Week names remain direct compatibility aliases. |
| Existing APIs | Current root exports and documented module-qualified workflows, preserved rather than globally restabilized in this pass. |
| Frozen configuration | `ml.models.make_canonical_ml_estimator_specs()` describes the established estimator/representation matrix, not a restriction on generic execution. |
| Evaluation architecture | Issue #11 adds module-qualified point/reference facts and interval/score attachments beneath legacy-compatible scientific reports; see [the evaluation contract](evaluation.md). These are not new root exports. |
| Package movement | Issue #3 places estimator contracts/baselines in `estimation` (Stage 1), physical primitives in `physics` (Stage 2A), histogram operations/features/representations in `preprocessing` (Stage 2B), IRF profiles/preparation/estimation in `irf` (Stage 3A), records/I/O/experimental adapters in `measurement` (Stage 3B), generic histogram/dataset generation in `simulation` (Stage 3C), core classical fitting/execution in `estimation` (Stage 4A), Bayesian inference in `estimation.bayesian` (Stage 4B), ML models/evaluation/cross-validation in `ml` (Stage 4C), historical diagnostics in `evaluation.diagnostics` (Stage 5A), factual contracts/builders/attachments in explicit `evaluation` submodules (Stage 5B), conditional analysis/plotting in `evaluation.conditional` and `evaluation.plots` (Stage 5C), and frozen Week-8 protocol/data generation in explicit `benchmarks` submodules (Stage 6A), and the intact generalization execution/reporting module in `benchmarks.generalization` (Stage 6B), and uncertainty source/metric/split/provenance contracts in `uncertainty.evaluation` (Stage 7A), and intact ML/classical uncertainty in `uncertainty.ml` and `uncertainty.classical` (Stage 7B), and Week-7 mismatch/timing studies in `benchmarks.mismatch` and `benchmarks.timing` (Stage 8A), and Issue-8 IRF/Issue-4 calibration and mismatch studies in `benchmarks.irf`, `benchmarks.bayesian_calibration`, and `benchmarks.bayesian_mismatch` (Stage 8B), and intact frozen Week-9 orchestration/reporting in `benchmarks.uncertainty` (Stage 8C), and the intact SQLite implementation in `persistence.sqlite` with the historical/public `persistence` package facade (Stage 9). |

The [package inventory](package_structure.md) records the current hierarchy and
historical relocation evidence. Issue #12 is complete, merged, and closed; its
[acceptance ledger](issue12_scientific_acceptance.md#stage-5-final-technical-acceptance)
is preserved.

The separate breaking cleanup removed **38 flat scientific module paths**.
Use the unchanged curated **66-name root API** or the canonical submodules below.
The reviewed `irf`, `preprocessing`, `simulation`, `evaluation`, and `persistence`
package APIs remain supported, as do the 23 documented symbol aliases.
See the [complete migration mapping](package_structure.md#removed-flat-module-mapping).

Ordinary historical Python pickles naming removed module globals may no longer
load. Canonical pickle roundtrips and retained package API lookups remain
supported. No legacy loader or dynamic module alias is provided. SQLite schema-v1
identity strings remain historical provenance labels, not import instructions.

## Persistence implementation and public package facade (Issue #3, Stage 9)

Canonical implementation: `tcspc_toolkit.persistence.sqlite`.
Historical/public compatibility: `tcspc_toolkit.persistence`.

The new package initializer directly re-exports all 62 supported names, including
`DuplicatePolicy` and the two Week-9 compatibility aliases. Old/new objects have
direct identity; the exact inventory is in the
[persistence contract](persistence.md#package-layout-and-compatibility-issue-3-stage-9).
Unlike the explicit-only benchmark/uncertainty namespaces, this initializer is
intentionally a public facade. It contains no SQL or database-opening logic.
Scientific adapter imports and pandas remain deferred in the single SQLite
implementation. Schema v1, its fingerprint, all seven historical stored type
strings, and the exact 66-name root API remain unchanged. No persistence names
are added to the root, and existing public-import examples remain supported.

## Curated root API and supported module-qualified workflows

The explicit `tcspc_toolkit.__all__` remains the curated package-root contract.
All pre-Issue-#2 root exports are preserved. Issue #2 Stage 4 added only the four generic
extension symbols and the missing `build_expected_counts_from_irf` entry.
Module-qualified workflows are also public where documented; absence from
the root does not make them private. Internal names beginning with `_` are
not extension points.

| Area | Current package-root names |
| --- | --- |
| Models and simulation | `monoexponential_decay`, `build_expected_counts_from_irf`, `sample_photon_counts`, `SyntheticDataset`, `generate_monoexponential_dataset` |
| IRF sources | `IRFSourceKind`, `IRFProfile`, `generate_gaussian_irf`, `generate_emg_irf`, `generate_gaussian_irf_profile`, `generate_emg_irf_profile`, `normalize_irf`, `shift_irf` |
| IRF preparation and estimation | `IRFPreparationDiagnostics`, `PreparedIRF`, `irf_profile_from_sampled_irf`, `prepare_irf`, `LeadingEdgeIRFDiagnostics`, `LeadingEdgeIRFResult`, `estimate_irf_from_leading_edge` |
| Classical inference | `convolve_decay_with_irf`, `LifetimeFitResult`, `ReconvolutionFitResult`, `fit_monoexponential_decay`, `fit_monoexponential_reconvolution` |
| Measurement boundary | `TimeUnit`, `MeasurementDataKind`, `SampledIRF`, `TCSPCMeasurement`, `convert_time_to_ns`, `load_sampled_irf_csv`, `load_tcspc_measurement_csv` |
| Experimental adapters | `ReferenceLifetimeEvaluation`, `fit_experimental_reconvolution`, `measurement_to_feature_table`, `measurement_to_histogram_batch`, `evaluate_estimate_against_reference` |
| Preprocessing | `align_to_irf`, `crop_time_window`, `detect_peak`, `estimate_background`, `normalize_counts`, `rebin_histogram`, `subtract_background`, `validate_histogram` |
| Configuration | `CountNormalization`, `FeatureConfig`, `PreprocessingConfig`, `SimulationConfig`, `load_config`, `save_config` |
| Features and representations | `FEATURE_NAMES`, `extract_feature_table`, `extract_features`, `cumulative_explained_variance`, `fit_pca_representation`, `normalize_histogram_batch`, `transform_pca_representation` |
| Generic estimator execution | `RegressorProtocol`, `EstimatorSpec`, `fit_regressors`, `predict_regressors` |
| Errors | `TCSPCError`, `InvalidHistogramError`, `InvalidMeasurementError`, `FeatureExtractionError` |

Importing the root does not initialize persistence, open a SQLite database,
or require the optional `emcee` sampler. Bayesian sampling remains an explicit
module-qualified operation requiring the `bayesian` extra; deterministic
Bayesian APIs also remain module-qualified.

Important supported module-qualified workflows include:

- `physics.models`, `physics.convolution`, and `physics.forward_model`: decay
  models, convolution, and the shared reconvolution expected-count model.
- `estimation.api` and `estimation.baselines`: generic estimator execution and
  baseline implementations.
- `estimation.fitting` and `estimation.classical`: fitting, initialization,
  per-curve execution, batch benchmarks, and fit diagnostics.
- `simulation.histograms` and `simulation.datasets`: expected counts, Poisson
  sampling, and ordinary/grouped synthetic datasets. The retained `simulation`
  package API exposes only its reviewed histogram surface.
- `irf.profiles`, `irf.preparation`, and `irf.estimation`: source profiles,
  explicit preparation, and leading-edge proxies. The retained `irf` package API
  exposes only its reviewed profile surface.
- `measurement.records`, `measurement.io`, and `measurement.adapters`:
  observations/provenance, explicit-unit CSV import, and experimental workflows.
- `preprocessing.histograms`, `preprocessing.features`, and
  `preprocessing.representations`: histogram operations, features, normalization,
  PCA, and count augmentation. The retained `preprocessing` package API exposes
  its reviewed histogram surface.
- `ml.models`, `ml.evaluation`, and `ml.cross_validation`: estimator factories,
  point evaluation, and development cross-validation.
- `benchmarks.generalization_protocol`, `benchmarks.generalization_datasets`,
  and `benchmarks.generalization`: frozen A–F definitions, paired datasets, and
  preparation/execution/reporting. Prepared-data carriers remain workflow-specific.
- `benchmarks.irf`, `benchmarks.bayesian_calibration`, `benchmarks.bayesian_mismatch`,
  `benchmarks.mismatch`, and `benchmarks.timing`: specialized scientific studies.
- `uncertainty.evaluation`: source interval/score carriers, metrics, and
  split/provenance policy; its A–F provenance guard retains the protocol dependency.
- `uncertainty.classical` and `uncertainty.ml`: model-conditional intervals,
  calibration, sensitivity, and disagreement-score workflows.
- `benchmarks.uncertainty`: frozen Week-9 reporting and its existing conformal workflow.
- `estimation.bayesian.posterior`, `estimation.bayesian.sampling`, and
  `estimation.bayesian.predictive`: priors, posterior/context, sampling, and
  predictive diagnostics. No Bayesian objects are root exports.
- `persistence` and `persistence.sqlite`: SQLite schema-v1 recording and read-only
  queries; see [persistence.md](persistence.md).
- `evaluation.results`, `evaluation.core`, and `evaluation.attachments`: factual
  point/reference results and separate uncertainty attachments. These are not
  root or diagnostics-package exports.
- `evaluation.conditional` and `evaluation.plots`: conditional analysis and
  plotting. The retained `evaluation` package API exposes seven diagnostics
  functions and does not aggregate these submodules.

This inventory does not promote every implementation helper to public API.
Private validation, prediction-table construction, SQL internals, and
study-specific orchestration are not generic extension hooks.

## Multi-exponential deterministic reconvolution (Issue #28, PR 1)

These additions are available through canonical module-qualified imports.
The root API and package facades are unchanged. This stage provides the
deterministic model and statistical primitives; bi-exponential fitting and
its scientific qualification belong to subsequent, separately reviewed stages.

```python
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
    multiexponential_reconvolution_expected_counts,
)
from tcspc_toolkit.evaluation.diagnostics import (
    PoissonFitStatistics,
    calculate_poisson_deviance,
    calculate_poisson_fit_statistics,
)
```

The two forward functions accept `time` and `irf` followed by keyword-only
`amplitudes`, `lifetimes`, and `temporal_shift=0.0`. The total-count function
also accepts `background=0.0`. Suitable array-like inputs are converted to
float64 arrays. Amplitudes and lifetimes are matching nonempty one-dimensional
arrays; amplitudes are finite and non-negative, lifetimes finite and positive.
Background is finite and non-negative; shift is finite.

`multiexponential_reconvolution_component_counts` returns a `(K, n_bins)`
array of background-free component expectations in **input order**. Equal
lifetimes and zero amplitudes are valid and are retained. The total-count
function returns their sum plus one constant background in counts per bin.
The K=1 result agrees exactly with the existing mono-exponential forward
model, whose implementation and estimator behavior are preserved.

Use an explicitly prepared, area-normalized kernel on the supplied uniform
grid, for example `prepared_irf.kernel` with `prepared_irf.time_ns`. Kernel
preparation and source provenance remain separate from the numerical array
input. No normalization, registration, resampling, or time-origin change is
performed here. Each component uses the existing mono-exponential model:
positive shift moves the IRF later, the shifted kernel is **not renormalized**,
convolution includes the bin width and truncates to the supplied window, and
background is added after convolution. Time, lifetimes, and shift use the same
units, conventionally nanoseconds. Periodic excitation and pile-up are not
included.

For components `c[k, i]` and pre-exponential parameters `A[k]`, distinguish:

| Quantity | Definition and interpretation |
| --- | --- |
| Component amplitude | `A[k]` in `sum(A[k] * exp(-time / tau[k]))`; it is a scale, not a finite-window photon budget. |
| Amplitude fraction | `A[k] / sum(A)` when total amplitude is positive. |
| Detected signal photon fraction | `sum(c[k]) / sum(c)` on this grid/window with this shifted IRF; excludes background. |
| Ideal infinite-window photon fraction | `A[k]*tau[k] / sum(A*tau)`, for the continuous untruncated ideal decay from time zero. |

These fractions generally differ. Their corresponding lifetime averages also
differ: `sum(A*tau)/sum(A)` is amplitude-weighted; `sum(A*tau**2)/sum(A*tau)`
uses ideal infinite-window photon weights; finite-window component weighting
uses the detected fractions above. None is automatically the observed mean
arrival time or a unique physical lifetime. Fractions and averages are
undefined for zero total signal; this stage returns curves and does not
silently invent such summaries.

### Poisson statistics and exact zero limits

`calculate_poisson_deviance(observed, expected)` returns
`2 * sum(mu - y + y*log(y/mu))` with stable evaluation near equality.
Inputs must be finite, non-negative, nonempty matching one-dimensional arrays.
It permits real-valued targets for **deterministic likelihood projections**;
this does not grant Poisson observation semantics to processed intensities.
Half the deviance is the centered negative log likelihood and can be used as
an optimization objective with the same minimizer as the reduced NLL.

No expectation floor is used. A zero observation and zero expectation
contribute zero; a zero observation contributes `2*mu`; positive observed
counts with zero expectation give infinite deviance. The near-equality series
and full-likelihood calculation avoid subtracting large, nearly equal terms.

For subnormal real-valued targets near the float64 underflow limit, intermediate
centered terms can round to zero even when the final deviance would be a
representable positive subnormal. This numerical limitation does not alter
the exact zero limits above.

`calculate_poisson_fit_statistics(observed, expected, *, n_parameters)` returns
a frozen `PoissonFitStatistics` record:

| Field | Meaning |
| --- | --- |
| `n_bins`, `n_parameters` | All observed bins, including zeros, and the caller-declared nominal parameter count. |
| `negative_log_likelihood` | Historical reduced NLL, `sum(mu - y*log(mu))`, omitting log-factorials. |
| `centered_negative_log_likelihood` | Reduced NLL relative to the saturated model; equals half the deviance. |
| `log_likelihood` | Full Poisson log likelihood, including log-factorials. |
| `deviance`, `deviance_residuals` | Total deviance and read-only signed per-bin square roots; positive residual means observed exceeds expected. |
| `aic` | `2*p - 2*log_likelihood`. |
| `bic_bins` | `p*log(n_bins) - 2*log_likelihood`; uses bins, not photon count, as sample size. |

This summary requires raw integer-valued counts below `2**53` for exact float64
integer representation. Integer-valued float arrays are accepted; fractional
counts are rejected. The caller is responsible for raw-count provenance:
numeric values cannot reveal earlier normalization or other processing.
Canonical measurement workflows must first use `require_raw_counts()`.

Parameter count is explicit (normally four for mono and six for bi
reconvolution with all parameters fitted). It is not reduced automatically
when components vanish or lifetimes coincide. Comparisons require identical
observations, binning/window, and likelihood conventions; a decay-model
comparison also holds the assumed IRF fixed. The regularity assumptions behind
information criteria can fail for weak components, equal lifetimes, or
boundary solutions. These are descriptive quantities, with no automatic model
choice, model-validity claim, or chi-square likelihood-ratio p-value.

The existing `estimation.fitting.poisson_negative_log_likelihood` and historical
`evaluation.diagnostics.calculate_poisson_deviance_residuals` retain their
numerical and validation behavior. In particular, the historical residual
helper still rejects zero expectations; the new statistics handle their
mathematical limits explicitly.

## Constrained bi-exponential Poisson reconvolution

The specialized PR-2 API is module-qualified; package initializers and the
ordered 66-name root API are unchanged:

```python
from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitResult,
    BiexponentialFitSettings,
    BiexponentialParameters,
    fit_biexponential_reconvolution,
)

result = fit_biexponential_reconvolution(
    time_ns,
    raw_counts,
    prepared_irf,  # Or an explicitly prepared, area-normalized array.
    lifetime_bounds_ns=(0.1, 8.0),
    temporal_shift_bounds_ns=(-0.3, 0.3),
)
```

Both bounds are mandatory, finite, and strictly increasing; the lifetime lower
bound is positive. The same lifetime bounds apply to both components. Counts
must be finite, non-negative, exactly integer-valued, and below `2**53`.
Integer-valued floating arrays are accepted. Passing arrays asserts raw-count
semantics; values cannot reveal prior processing. The canonical-measurement
adapter calls `require_raw_counts()` first. This function does not accept
processed intensities or deterministic fractional expectations as observations.

`PreparedIRF` inputs must have a compatible grid. Their supplied source metadata,
provenance, and preparation diagnostics are retained separately from the kernel.
Array inputs must match the grid shape and have finite, non-negative values and
unit trapezoidal area (`rtol=1e-10`, `atol=1e-12`). Preparation for an array is
caller-declared; its normalization is checked but its source is not inferred.
There is no silent resampling, alignment, normalization, or time-origin change.
Successful normalization or fitting does not establish experimental correctness.

The six parameters are two pre-exponential amplitudes, two lifetimes, one
background in counts/bin, and one shared temporal shift in ns. Amplitudes and
background have closed zero boundaries. Lifetime coordinates are logarithmic,
but the model permits exactly equal lifetimes and zero components. Positive
shift moves the IRF later without renormalizing lost support. The PR-1 forward
model supplies the same bin-width scaling and finite-window truncation; one
constant background is added after component convolution.

Optional `initial_guesses` is a sequence of `BiexponentialParameters` records.
Each record contains `amplitudes`, `lifetimes_ns`, `background`, and
`temporal_shift_ns`; guesses must respect the supplied bounds. They augment the
deterministic seed bank and never change constraints. Lifetimes are sorted only
in returned records, with corresponding amplitudes and component curves moved
together. Exactly equal lifetimes use descending amplitude as a secondary key;
no components are merged and no minimum lifetime gap is imposed.

`BiexponentialFitResult` groups the following:

| Field | Interpretation |
| --- | --- |
| `parameters` | Canonically ordered fitted component parameters and shared nuisance parameters; unavailable if no finite candidate exists. |
| `expected_counts`, `component_counts` | Read-only total expectation and `(2, n_bins)` background-free component curves. |
| `statistics` | PR-1 `PoissonFitStatistics`, with six nominal parameters even at degeneracies. |
| `components` | Amplitude fractions, component detected photon totals/fractions, ideal fractions proportional to `A*tau`, and explicitly weighted lifetime averages. |
| `optimization` | Every refinement's status and evaluated objective, selected candidate source, numerical checks, stopping reason, phase-specific evaluation counts/times and exhausted budgets. |
| `assumptions` | Bounds, resolved settings and numerical scales, raw-count assumption, IRF input kind, area and available source/preparation records. |
| `identifiability` | Separate structural, conditioning, boundary and multi-start warnings; weighted-Jacobian diagnostics and near-optimal parameter/contribution ranges. |

The amplitude-weighted lifetime uses weights `A/sum(A)`. The ideal-photon-weighted
lifetime uses `A*tau/sum(A*tau)`. The detected-signal-weighted lifetime uses
component photon sums over the actual acquisition window. These are named
summaries, not a unique physical truth or a mean photon arrival time. Fractions
and associated averages are unavailable when their denominator vanishes.

The objective is half the PR-1 stable Poisson deviance. No expectation floor is
used. The fitter retains the lowest finite evaluated candidate, including a
better validation probe when recovery cannot run. Optimizer termination is
reported separately from independent local numerical validation. The validation
record names its anchor parameters and the maximum descent found in its bounded
coordinate/coupled-probe neighborhood; a passing result is not proof of a global
minimum. The selected candidate can improve on that anchor within the declared
NLL tolerance. Its source distinguishes optimizer, initialization, validation,
and the analytical all-zero-observation solution.

Settings expose deterministic `economical`, `adaptive`, and `exhaustive` policies,
and separate hard budgets for initialization, optimization, validation, and
diagnostics. Optimizer finite-difference calls count against the optimization
budget. Start limits include recovery attempts. Incomplete numerical validation
is represented by `None`, never a pass; budget exhaustion is inspectable even
when a useful finite candidate is available. The selected default and its measured
cost are recorded in [the numerical development report](issue28_pr2_numerics.md).

The development correction for [Issue #33](issue33_stage2_correction.md) keeps
those numerical limits but changes invalid-trial handling. Nonfinite objectives
or unavailable gradients terminate the current L-BFGS-B invocation explicitly.
Linear amplitude/background derivatives use the exact Poisson score; shape
derivatives use bounded finite differences with finite one-sided fallbacks.
Safeguarded finite descent retains its finite probes. Exhaustive fits give every
finite seed one bounded invocation before local validation and required recovery;
optional per-start continuations cannot displace seeds. Economical and adaptive
policies retain their continuation priority and tier/early-stop rules. Every
solver invocation and model call is charged to the original work limits.
`finite_seed_candidates` and `unattempted_seed_candidates` describe the whole
finite bank, including seeds beyond an economical/adaptive early stop.
`optional_continuations_omitted` lists originating start indices whose additional
solver invocation was omitted under the exhaustive policy; their finite probes
remain eligible for selection. `recovery_unavailable_reason` separately reports
required validation recovery blocked by a limit (or pending continuations for
the other policies). Exactly 46 completed starts is not itself exhaustion.
Incomplete required work or failed/unavailable validation still prevents full
numerical acceptance, independently of optimizer success. The external
qualification assessment also rechecks physical feasibility, curves and objective
with one separately accounted forward call. See the
[Stage 2B scheduling record](issue33_stage2b_scheduling.md). The subsequent
[independent Stage 3 study](issue33_stage3_qualification_v1.md) passed primary
numerical acceptance at 99/100 and both component recovery/bias targets, but
failed its strict 100/100 independent agreement target at 99/100. A complete
46-seed bank can still leave no slot for required recovery after failed local
validation. The robustness correction is suitable for publication; Issue #33
remains open and the fitter is not fully qualified under that frozen gate.

Numerically invalid initialization trials are rejected without discarding other
finite candidates. `starts_screened` includes rejected trials;
`initialization_rejected_candidates` counts them, including templates that fail
before objective screening. Every attempted forward call is charged to the
initialization budget. `initialization_scaling_failed` separately records a
failed scale-reference evaluation, which retains the original count-based scale
and is not a screened candidate. With no finite candidate, parameter/curve fields
are unavailable and the stopping reason is `no_finite_candidate`. Caller input
and initial-guess validation remain strict.

The weighted Jacobian uses the recorded dimensionless coordinates and Poisson
weights `1/sqrt(mu)`. Its SVD rank tolerance includes measured finite-difference
error, so numerical noise does not falsely resolve exact degeneracy. Conditioning
is unavailable at nonregular zero-expectation bins, interpolation knots, unstable
derivatives, or exhausted diagnostic budgets. At a vanishing component its
lifetime is unconstrained even if a parameter value is returned.

Warning thresholds are configurable descriptive conventions: weak detected
fraction, lifetime similarity, proximity to bounds, and local conditioning. They
do not set numerical acceptance or establish scientific validity. Near-optimal
solutions are defined by a descriptive NLL window, not a confidence region.
Neither optimizer success, local full rank, agreement across starts, deviance,
AIC nor bin-based BIC certifies physical identifiability. There is no automatic
model selection, ordinary chi-square likelihood-ratio p-value, or component
confidence interval at the singular mono/bi boundary.

## Compatibility principles

- Canonical/root imports, reviewed package APIs, argument signatures, result
  fields, numerical behavior, and benchmark defaults remain supported. Stage-3
  symbol names remain direct aliases without wrappers or deprecation warnings.
  The 38 obsolete flat module paths are intentionally removed.
- Public renames use the reviewed old-to-new mapping and caller audit below.
  Use scientific terminology, not a blanket `Frozen...` prefix.
  Preserve Day/Week provenance where historically meaningful, and preserve
  A-F identifiers, protocol labels, seeds, and scientific reference outputs.
- Root exports are deliberate stability commitments, not an inventory of all
  implemented capabilities. Bayesian, CV, benchmark, evaluation, and persistence
  types remain supported through their documented module-qualified paths.
- Issue #3 retained flat imports through Issue #12. Their subsequent removal
  is a separate breaking change; it does not remove supported symbol aliases.
- Python naming changes must not reinterpret SQLite schema-v1 method/model
  identities, serialized provenance, or stored source-result-type strings.
- Keep the Issue-#1 typing rule: suitable normalized inputs use `ArrayLike`;
  genuine ndarray requirements, concrete ndarray returns, and stored arrays
  retain their intentional contracts. There is no new package-wide typing sweep.

## Boundaries and deferred work

### Issue-#3 structural migration contract

This subsection preserves the historical relocation sequence. Flat paths in its
mapping tables were available through Issue #12 and have since been removed.
The five package-initializer APIs remain supported. Historical checkpoints here
do not promise current flat-module imports or historical-flat pickle loading.

Package relocation preserves the exact 66-name root `__all__`, including its
current order. It changes implementation locations, not scientific behavior,
Issue-#11 evaluation semantics, frozen protocols/seeds, or schema-v1 identities.
Stage 0 established the structural guards; Stage 1 moved the generic estimator
API and baselines; Stage 2A moved models, convolution, and the forward model;
Stage 2B converted histogram preprocessing into a package and relocated features
and representations beneath it; Stage 3A similarly moved IRF profiles,
preparation, and estimation. Stage 3B moved measurement records, I/O, and
experimental adapters and extracted their shared fixed-IRF selection helper.
Stage 3C converted histogram simulation into a package and relocated ordinary
synthetic dataset generation beneath it.
Stage 4A moved core classical fitting/execution into `estimation.fitting` and
`estimation.classical`. Stage 4B moved Bayesian inference into
`estimation.bayesian.posterior`, `estimation.bayesian.sampling`, and
`estimation.bayesian.predictive`, without moving Bayesian studies, ML, or
uncertainty modules. Both estimation initializers remain minimal. The posterior
sampling entry point still imports sampling only when called; importing the
posterior, sampling, or predictive module does not require `emcee`. Executing
MCMC retains the existing optional-extra requirement.

Stage 4C moved ML factories, benchmark evaluation, and repeated CV into
`ml.models`, `ml.evaluation`, and `ml.cross_validation`. The `ml` initializer
does not aggregate these modules. Concrete sklearn factories remain separate
from generic `estimation.api`; factual dependencies now use
`evaluation.core` and `evaluation.results` after Stage 5B.
Frozen generalization and uncertainty orchestration remain in their existing
modules.

Stage 5A converted the historical `evaluation.py` module into a package.
Its canonical implementation is `tcspc_toolkit.evaluation.diagnostics`;
`tcspc_toolkit.evaluation` directly re-exports only the seven historical functions:
`generate_fitted_signal`, `calculate_residuals`, `calculate_reduced_residuals`,
`calculate_absolute_lifetime_error`, `calculate_relative_lifetime_error`,
`calculate_poisson_deviance_residuals`, and `calculate_lifetime_errors`.
The compatibility path remains supported through at least Issue #12.
The initializer does not aggregate factual results, attachments, conditional
analysis, plots, uncertainty, or benchmark reports. The existing diagnostics
dependency on `estimation.fitting.LifetimeFitResult` remains; parent-package
initialization costs must not be mistaken for imports owned by factual modules.
No root exports were added.

Stage 5B moved the Issue-#11 factual implementations into explicit submodules:

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.evaluation.results` | `tcspc_toolkit.evaluation_results` |
| `tcspc_toolkit.evaluation.core` | `tcspc_toolkit.evaluation_core` |
| `tcspc_toolkit.evaluation.attachments` | `tcspc_toolkit.evaluation_uncertainty` |

The results source has no package-internal dependency; core and attachments
depend only on results. They do not import diagnostics, estimators, uncertainty
algorithms, benchmarks, or persistence. Importing them still executes the
historical diagnostics package facade first; that inherited initialization cost
does not change their source-level dependency boundary. Field/reference/validity
semantics, caller-selected metric populations, and attachment identities are
unchanged. The `evaluation` initializer still exports only the seven diagnostics.

Stage 5C moved reusable conditional analysis and plotting without changing their
populations, reference selection, regime boundaries, table schemas, or visuals:

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.evaluation.conditional` | `tcspc_toolkit.conditional_evaluation` |
| `tcspc_toolkit.evaluation.plots` | `tcspc_toolkit.benchmark_plots` |

Conditional analysis retains both legacy-result adapters and explicit factual
paths; it does not select frozen A–F conditions. Plotting remains a separate
explicit import: importing `tcspc_toolkit.evaluation` does not load conditional
analysis or Matplotlib. Benchmark, uncertainty, and persistence modules moved in
the later Issue-3 stages described below.

Stage 6A creates the docstring-only `benchmarks` namespace and moves
the frozen Week-8 protocol/configuration and dataset-generation implementations.
Stage 6B relocates generalization preparation, execution, and reporting intact:

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.benchmarks.generalization_protocol` | `tcspc_toolkit.generalization` |
| `tcspc_toolkit.benchmarks.generalization_datasets` | `tcspc_toolkit.generalization_datasets` |
| `tcspc_toolkit.benchmarks.generalization` | `tcspc_toolkit.generalization_evaluation` |

The initializer imports no children. Protocol definitions, A–F ordering, seeds,
paired designs, RNG behavior, and development/external roles are unchanged.
The single `benchmarks.generalization` implementation retains all preparation,
execution, private helpers, result/report carriers, table compatibility policies,
and seven Day/Week aliases. Its historical facade re-exported the 61
supported names; implementation-owned monkeypatches target the canonical module.
PCA and estimators still fit only on development data and reuse fitted artifacts
on external A–F tests. None of these names is added to the 66-name root API.
Other benchmark studies, uncertainty implementations, and persistence remain
in their existing locations. The uncertainty provenance guard still depends
explicitly on the frozen protocol; ML uncertainty's protocol/data/prepared-data
annotations remain typing-only imports. Uncertainty robustness imports the
canonical prepared-data carrier eagerly; persistence imports `RobustnessMetrics`
locally, preserving schema v1 and stored identities. All three flat paths were retained through Issue #12 and are now removed.

Stages 7A–7B retain the docstring-only `uncertainty` namespace and place
three intact implementations in explicit submodules:

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.uncertainty.evaluation` | `tcspc_toolkit.uncertainty_evaluation` |
| `tcspc_toolkit.uncertainty.ml` | `tcspc_toolkit.ml_uncertainty` |
| `tcspc_toolkit.uncertainty.classical` | `tcspc_toolkit.classical_uncertainty` |

The 19 uncertainty-evaluation names remain available in the canonical module. Source carriers
such as `PredictionIntervalResult` and `UncertaintyScoreResult`, uncertainty
metrics, split/provenance policy, and `WEEK9_HYPOTHESES` stay together unchanged.
They remain separate from the generic point-linked `IntervalAttachment` and
`ScoreAttachment` contracts in `evaluation.attachments`.

The implementation retains its explicit `FINAL_ROBUSTNESS_TEST_IDS` dependency
on `benchmarks.generalization_protocol` for the A–F provenance restriction;
it does not import benchmark execution or datasets. The initializer imports
no children and adds no scientific imports beyond the existing root initialization
(which already loads scikit-learn). None of these names is root-exported.
Stage 7B preserves all 34 ML names (including three Week-9 aliases) and 18 classical
names by direct identity. Quantile, RF-spread, training-bootstrap, local covariance,
parametric-bootstrap, and repeated-Poisson implementations are not repartitioned.
Repeated-Poisson variability remains an empirical reference, not a per-point attachment.
ML's three benchmark references remain under `TYPE_CHECKING`; private projection
and reconstruction helpers stay private in the canonical implementation modules.
Patches of implementation-owned bindings must target those modules, not the facades.

At the Stage-7B checkpoint, `uncertainty_robustness.py` remained the frozen Week-9
study/report layer, with conformal intact. All eight changed persistence imports
remained function-local. Stage 8C below relocates that study without extraction.

Stage 8A moves the intact Week-7 mismatch and timing studies:

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.benchmarks.mismatch` | `tcspc_toolkit.mismatch_evaluation` |
| `tcspc_toolkit.benchmarks.timing` | `tcspc_toolkit.timing_evaluation` |

Mismatch retains its six public names and its paired mono-/bi-exponential design.
It is not frozen Week-8 Test F and imports no generalization protocol or execution
module. Paired ML factual references remain `generating_mono` and
`primary_component`; no pseudo-true reference is introduced.

Timing retains eight public names, including the `FloatArray` typing alias.
`repeated_batch` prediction timing excludes fitting and warmup measurements;
`per_curve_optimization` reuses existing reconvolution runtimes. Their units,
aggregation, ordering, and table schemas are unchanged.

Both historical paths directly re-export identical objects, with no new root
exports. The benchmark initializer remains docstring-only. Neither study has
production consumers or persistence-local imports. At the Stage-8A checkpoint,
IRF/Bayesian studies, uncertainty robustness, and persistence remained in place.

Stage 8B moves three intact scientific-study modules:

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.benchmarks.irf` | `tcspc_toolkit.irf_evaluation` |
| `tcspc_toolkit.benchmarks.bayesian_calibration` | `tcspc_toolkit.bayesian_evaluation` |
| `tcspc_toolkit.benchmarks.bayesian_mismatch` | `tcspc_toolkit.bayesian_mismatch_evaluation` |

`estimation.bayesian.*` remains the Bayesian estimator implementation;
`benchmarks.bayesian_*` owns the matched-model calibration/comparison and separate
model-mismatch studies. Priors, named streams, inference, PPC, and report schemas
are unchanged. Credible intervals remain conditional on the model, prior, and
fixed assumed IRF. The deterministic, prior-free `pseudo_true_mono` likelihood
projection remains separate from `generating_mono` and `primary_component`.

The IRF study retains source-profile, prepared-kernel, assumed-kernel, and
leading-edge-proxy distinctions. It has no direct frozen A–F dependency; existing
uncertainty dependencies can transitively load protocol configuration. Bayesian
mismatch eagerly imports the private `_match_gaussian_to_emg_profile` from
`benchmarks.irf` and `_unavailable_bootstrap`/`_unavailable_covariance` plus four
compact result carriers from `benchmarks.bayesian_calibration`. These helpers
remain private and are neither copied nor extracted.

Persistence's three calibration and four mismatch imports remain function-local.
Its schema-v1 source identity is still exactly
`tcspc_toolkit.bayesian_evaluation.Issue4Bayesian`, independent of the relocated
class's natural `__module__`. The four Issue-4 runner/audit scripts use canonical
study imports. The benchmark initializer is docstring-only, no root exports are
added, and importing any study does not require `emcee`. At the Stage-8B checkpoint,
persistence and uncertainty robustness remained flat; conformal had not been extracted.

The exact supported surfaces are below. `FloatArray`, `IntArray`, `Mechanism`,
`GeneratingModel`, and `ReferenceName` are existing typing aliases; there are no
additional historical callable aliases. Incidental imports and private helpers
are not added to the facades.

`benchmarks.irf` (18 names):

```text
FloatArray
IntArray
IRFShapeMismatchConfig
IRFShapeMismatchDataset
generate_irf_shape_mismatch_dataset
IRFClassicalPair
IRFClassicalEvaluation
evaluate_irf_shape_classical
summarize_irf_shape_classical
IRFConditionalUncertaintyEvaluation
evaluate_irf_shape_conditional_uncertainty
IRFMLTransferEvaluation
evaluate_gaussian_trained_irf_transfer
LeadingEdgeFailureConfig
LeadingEdgeRegimeResult
LeadingEdgeFailureEvaluation
evaluate_leading_edge_failure_regimes
summarize_leading_edge_regimes
```

`benchmarks.bayesian_calibration` (21 names):

```text
BayesianClassicalCondition
BayesianClassicalEvaluationConfig
Issue4RealizationSeeds
derive_issue4_seed
issue4_realization_seeds
build_issue4_expected_counts
sample_issue4_observation
Issue4ClassicalFit
Issue4Covariance
Issue4Bootstrap
Issue4Bayesian
BayesianClassicalRealizationResult
evaluate_issue4_realization
Issue4IntervalCoverage
Issue4EstimatorSpread
Issue4RuntimeSummary
Issue4UncertaintyScaleComparison
BayesianClassicalConditionSummary
BayesianCalibrationReport
summarize_issue4_condition
evaluate_issue4_condition
```

`benchmarks.bayesian_mismatch` (23 names):

```text
Mechanism
GeneratingModel
ReferenceName
MismatchAssumption
MismatchCondition
MismatchEvaluationConfig
MismatchWorkflow
build_mismatch_workflow
MismatchSeeds
mismatch_seeds
GeneratingCounts
build_mismatch_expected_counts
PseudoTrueReference
construct_pseudo_true_reference
MismatchReferenceComparison
MismatchPredictiveSummary
MismatchInferenceRecord
MismatchGroupSummary
MismatchReport
sample_mismatch_observation
evaluate_mismatch_inference
summarize_mismatch_records
evaluate_mismatch_workflow
```

- Issue #3 established direct flat re-exports through Issue #12. The subsequent
  cleanup removes those paths. Scientific names and direct Day/Week aliases remain
  supported in canonical modules.
- New subpackage initializers stay minimal: do not eagerly import an entire
  subtree. For existing names that become packages (`irf`, `simulation`,
  `preprocessing`, `evaluation`, `persistence`), the new `__init__.py` must also
  provide the reviewed old-path facade. Do not leave competing `name.py` and
  `name/` implementations whose import precedence obscures ownership.
- Canonical classes retain their authentic implementation `__module__`.
  Ordinary pickles naming removed flat globals may no longer load; canonical
  and retained-package lookups remain supported. Persisted `source_result_type`
  strings are historical schema-v1 provenance identities, not dynamically
  resolved import paths, and must not be rewritten.
- Import compatibility does not promise arbitrary module-global monkeypatch
  compatibility. Tests that replace an implementation dependency should patch
  its canonical implementation module after relocation, not a re-export shim.
- Generic estimator execution remains independent of sklearn/canonical model
  factories. Generic factual evaluation remains independent of estimators,
  frozen orchestration, Bayesian sampling, and persistence. Physical primitives
  stay below reporting; scientific computation must not depend on persistence.
  Root imports must neither require `emcee` nor initialize persistence/SQLite.

The structural guards in `test_package_structure.py`, the exact root contract
in `test_public_api.py`, canonical/removed report-pickle and alias tests, and persistence
source-identity tests form the structural closure gate. The import graph
distinguishes eager, function-local, and `TYPE_CHECKING` dependencies; fresh
processes also check real import order and side effects. The intentionally
deferred Bayesian sampling relationship is not an eager cycle.

The Issue-3 slices used these historical paths. Flat entries in the right
column are now removed; the five reviewed package entries remain supported.

| Canonical implementation | Historical Issue-3 import path |
| --- | --- |
| `tcspc_toolkit.evaluation.diagnostics` | `tcspc_toolkit.evaluation` (seven-function diagnostics facade) |
| `tcspc_toolkit.estimation.api` | `tcspc_toolkit.estimator_api` |
| `tcspc_toolkit.estimation.baselines` | `tcspc_toolkit.baselines` |
| `tcspc_toolkit.estimation.fitting` | `tcspc_toolkit.fitting` |
| `tcspc_toolkit.estimation.classical` | `tcspc_toolkit.classical_evaluation` |
| `tcspc_toolkit.physics.models` | `tcspc_toolkit.models` |
| `tcspc_toolkit.physics.convolution` | `tcspc_toolkit.convolution` |
| `tcspc_toolkit.physics.forward_model` | `tcspc_toolkit.forward_model` |
| `tcspc_toolkit.preprocessing.histograms` | `tcspc_toolkit.preprocessing` |
| `tcspc_toolkit.preprocessing.features` | `tcspc_toolkit.features` |
| `tcspc_toolkit.preprocessing.representations` | `tcspc_toolkit.representations` |
| `tcspc_toolkit.irf.profiles` | `tcspc_toolkit.irf` |
| `tcspc_toolkit.irf.preparation` | `tcspc_toolkit.irf_preparation` |
| `tcspc_toolkit.irf.estimation` | `tcspc_toolkit.irf_estimation` |
| `tcspc_toolkit.measurement.records` | `tcspc_toolkit.measurements` |
| `tcspc_toolkit.measurement.io` | `tcspc_toolkit.measurement_io` |
| `tcspc_toolkit.measurement.adapters` | `tcspc_toolkit.experimental` |
| `tcspc_toolkit.simulation.histograms` | `tcspc_toolkit.simulation` |
| `tcspc_toolkit.simulation.datasets` | `tcspc_toolkit.datasets` |

The `estimation` initializer imports nothing. The generic API has no sklearn or
package-internal dependency. Baselines retain the existing sklearn
`DummyRegressor` implementation; this move does not replace its arithmetic.
A normal dotted import still initializes the existing package root, which
already imports fitting and sklearn-backed representations. These inherited
root dependencies are not new dependencies of the `estimation` initializer.
Canonical/root identity, natural canonical `__module__`, removed-path failures,
and unchanged baseline/estimator behavior are tested. Responsibility path sets
must continue to reflect the canonical implementation boundaries.

The fitting and classical execution implementations do not broaden
`estimation/__init__.py`; request their modules explicitly. Fitting uses the
shared `physics.forward_model`. Classical execution uses
`evaluation.diagnostics` for Poisson-deviance diagnostics following the completed
Stage-5A migration. The generic factual
evaluation contracts do not depend on these estimators. Optimizers, objectives,
initialization, validity/failure policy, diagnostics, and timing are unchanged.
Schema-v1 source identities remain
`tcspc_toolkit.fitting.ReconvolutionFitResult` and
`tcspc_toolkit.classical_evaluation.ReconvolutionCurveResult`, despite executable
imports and natural `__module__` values moving to their canonical locations.
The stored strings remain valid provenance after the corresponding flat class paths are removed.

The `physics` initializer also imports nothing. Its implementations retain only
their existing NumPy/SciPy and low-level physical dependencies. The forward
model uses canonical `physics.models`/`physics.convolution` and
`irf.profiles.shift_irf`; classical and Bayesian inference remain consumers, not
dependencies. Root convenience exports remain exactly the reviewed 66 names;
the forward-model function is still module-qualified.

Unlike the import-free initializers above, `preprocessing/__init__.py` preserves
the former histogram module's supported API through direct re-exports from
`preprocessing.histograms`: `validate_time_axis`, `time_axes_compatible`,
`validate_histogram`, `estimate_background`, `subtract_background`, `detect_peak`,
`align_to_irf`, `crop_time_window`, `rebin_histogram`, and `normalize_counts`.
It also retains the historically imported `CountNormalization` alias, whose
definition remains in `config`. Private validators/tolerances, the logger, and
incidental numerical/type imports are not facade APIs.

The old `preprocessing.py` no longer competes with this package. Its initializer
does not import features or representations; callers request those submodules
explicitly. Normal root initialization already imports them for the established
root API, so initializer-isolation tests trace the actual importing module.
Canonical consumers use the implementation submodules, not the compatibility
facades. Histogram, feature, normalization, PCA, and augmentation algorithms are
unchanged. Measurement/IRF validation consumers still use the same histogram
helpers; their validation responsibility has not changed.

The `irf` module-to-package conversion follows the same narrow-facade policy.
`irf/__init__.py` directly re-exports only `IRFSourceKind`, `IRFProfile`,
`generate_gaussian_irf`, `generate_emg_irf`, `generate_gaussian_irf_profile`,
`generate_emg_irf_profile`, `normalize_irf`, and `shift_irf` from `irf.profiles`.
It does not import or expose preparation/estimation APIs; request those canonical
submodules explicitly. The old `irf.py`
no longer competes with the package. Root initialization already exposes the
reviewed preparation/estimation objects, independently of this narrow facade.

Measurement records depend only on IRF profiles; preparation consumes canonical
`measurement.records` and profiles. Keeping preparation out of the initializer
preserves that direction without a cycle. Source profiles, derived prepared
kernels, assumed inference kernels, and approximate leading-edge proxies remain
distinct, as do source provenance, preparation history, and inference assumptions.
Profile, preparation, and proxy algorithms are unchanged apart from imports.

The `measurement` initializer is docstring-only, not an aggregate API.
`measurement.records` retains observations and provenance without intrinsic
truth; raw counts and processed intensity remain distinct, with the same
Poisson-method guard. I/O still requires explicit units. Trusted references
remain separately and explicitly supplied to the higher-level adapters.

Stage 3B extracted the unchanged private `_resolve_reconvolution_irf` into
`irf.selection`. Both `measurement.adapters` and Bayesian model construction
use this internal fixed-assumption selector; Bayesian construction no longer
depends on the experimental adapter. It selects an explicit `PreparedIRF`
over an attached `SampledIRF`, validates the prepared grid/kernel, or normalizes
the attached trace. It does not prepare/resample implicitly, fit, or infer source
meaning. Its dependencies are IRF/measurement foundations, not adapters or
estimators. Selection is neither a new public/root API nor an export of
`irf/__init__.py`. The resolver body, including historical errors, is unchanged.

The `simulation` facade directly re-exports the former histogram module's seven
functions: `sample_photon_counts`, `build_expected_counts_from_irf`,
`simulate_irf_convolved_histogram`, `simulate_irf_convolved_biexponential_histogram`,
`simulate_monoexponential_decay`, `simulate_biexponential_decay`, and
`simulate_multiexponential_decay`. The old `simulation.py` no longer competes
with the package. The initializer imports only `simulation.histograms`, not
`simulation.datasets`; the root's existing dataset exports are a separate edge.
Flat `datasets` historically re-exported `SyntheticDataset`, both dataset generators,
and the existing array type aliases. Expected-count construction, Poisson
sampling, seed/Generator handling, and random-draw ordering are unchanged.
This is generic synthetic generation, not the frozen A–F protocol: the
generalization implementations remain separate under `benchmarks`, as described
in the Stage-6 inventory above.

### Stage-8C frozen uncertainty benchmark

| Canonical implementation | Historical flat path (now removed) |
| --- | --- |
| `tcspc_toolkit.benchmarks.uncertainty` | `tcspc_toolkit.uncertainty_robustness` |

The whole Week-9 orchestration/reporting module remains intact. It consumes reusable
`uncertainty.evaluation`, `uncertainty.ml`, and `uncertainty.classical`; it does not
replace their source carriers or generic factual attachments. Frozen A–F traversal,
calibration/external separation, scorecard populations, residual diagnostics, and
conformal calibration/external evaluation retain their existing policies. There is
no separate conformal module. Repeated-Poisson variability remains an empirical
reference, not a per-point attachment.

All 28 supported names below remain available from the canonical module.
They comprise six result/report classes, nine functions, two constants, and eleven
legacy aliases (including the historical seed alias); there are no public typing
aliases. The private `_Condition` TypeVar and helpers remain private. Tests that
inspect or patch private implementation bindings use the canonical module.
No production module consumes private robustness helpers.

```text
DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED
CONFORMALIZED_QUANTILE_METHOD_ID
ConformalizedQuantileCalibrationResult
conformalize_quantile_gradient_boosting
ConformalizedQuantileExternalResult
evaluate_frozen_conformalized_quantile
build_ml_interval_scorecard
build_ml_uncertainty_scorecard
MLUncertaintyRobustnessReport
build_ml_uncertainty_robustness_report
build_classical_uncertainty_scorecard
build_classical_conditional_uncertainty_scorecard
ClassicalUncertaintyRobustnessReport
evaluate_classical_uncertainty_robustness
ClassicalResidualMatrixResult
ResidualMismatchReport
evaluate_residual_mismatch_diagnostics
DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED
build_week9_interval_scorecard
build_week9_score_only_scorecard
Week9MLUncertaintyRobustnessReport
build_week9_ml_uncertainty_robustness_report
build_week9_classical_uncertainty_scorecard
build_week9_classical_conditional_scorecard
Week9ClassicalUncertaintyRobustnessReport
evaluate_week9_classical_uncertainty_robustness
Week9ResidualMismatchReport
evaluate_week9_residual_mismatch_diagnostics
```

The eleven legacy aliases are the final eleven entries above and retain their
existing canonical targets from the scientific-name mapping below. Canonical
class/function pickle lookup remains supported; the removed flat module no longer
resolves historical globals. No `__module__` metadata is fabricated.

Persistence is the only implementation consumer: `record_prediction_intervals`
and `record_conformal_intervals` import `CONFORMALIZED_QUANTILE_METHOD_ID` locally.
Only those import paths change. No persisted qualified
`tcspc_toolkit.uncertainty_robustness` string exists, and no mapping is introduced.
The benchmark has eager canonical dependencies on the generalization protocol,
datasets, and prepared-data implementation; it has no deferred or typing-only
imports, no persistence dependency, and no reverse consumers in reusable layers.
Both namespace initializers remain docstring-only; no root exports are added.

### Completed evaluation work and later gates

Stage 1 established a generic estimator execution interface and a separate
canonical specification factory. Stage 2 connects the generalization fitting
and shared final-test ML prediction boundary to it, as described below.
The dependency direction is canonical specification -> generic contract;
generic execution must not know Ridge/RF/HGB or canonical representation names.

- Issue #11: shared preparation, factual point/reference evaluation, and
  interval/score attachments now underlie the relevant generalization,
  conditional, mismatch, and Week-9 reporting paths. Specialized Bayesian and
  experimental reports retain their context; no universal report or
  representation framework was introduced. See [evaluation.md](evaluation.md).
- Issue #3: package restructuring is complete through Stage 10; see the
  [package inventory](package_structure.md). The later flat-module cleanup is separate.
- Issue #12: integration and scientific-reference acceptance is complete, merged,
  and closed; see its historical ledger. Small equivalence tests do not replace
  that scientific evidence.

## Regression strategy

`tests/test_estimator_api.py` compares canonical specification configurations
and predictions directly against the existing factories under the installed
dependencies. Both execution paths use the same deterministic inputs, with PCA
fitted only to training rows. Labels, representation choices, and explicit
RF/HGB seeds are asserted; existing model tests protect pipeline composition.
These checks are not replacements for the A-F or Week-9 scientific regressions.

The temporary Stage-0 JSON snapshot was removed during review: its extra check
was same-environment drift in the underlying factories themselves, but its toy
numerical outputs and captured unpinned sklearn defaults were not a durable
scientific reference. Tests no longer skip on dependency versions or require
snapshot regeneration after an upgrade. This establishes execution equivalence,
not identical numerical results across arbitrary dependency versions. Existing
scientific reference artifacts remain unchanged.

## Generic estimator extension API

Use the curated root imports for generic estimator execution:

```python
from tcspc_toolkit import (
    RegressorProtocol, EstimatorSpec, fit_regressors, predict_regressors,
)
```

The canonical module-qualified path is `tcspc_toolkit.estimation.api`, for
example `from tcspc_toolkit.estimation.api import EstimatorSpec`. The obsolete
`tcspc_toolkit.estimator_api` module is removed. Supported signatures are:

```python
class RegressorProtocol(Protocol):
    def fit(self, X: Any, y: NDArray[np.float64], /) -> object: ...
    def predict(self, X: Any, /) -> ArrayLike: ...

@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    factory: Callable[[], RegressorProtocol]
    representations: tuple[str, ...]

def fit_regressors(
    *,
    estimator_specs: Sequence[EstimatorSpec],
    X_train_by_representation: Mapping[str, Any],
    y_train: ArrayLike,
) -> dict[str, dict[str, RegressorProtocol]]: ...

def predict_regressors(
    *,
    fitted_estimators: Mapping[str, Mapping[str, RegressorProtocol]],
    X_by_representation: Mapping[str, Any],
) -> dict[str, dict[str, NDArray[np.float64]]]: ...
```

The interface is deliberately positional `fit(X, y)` / `predict(X)`, with no
required inheritance, `get_params`, sklearn cloning, or registry. `fit` updates
the instance; its return value is ignored, so a custom adapter may return `None`.
An adapter for another training library owns its training details internally.
This is not a promise to support every training system without an adapter.

Each specification selects a nonempty tuple of unique representation names;
the estimator-spec sequence must also be nonempty with unique estimator names.
Identifiers are nonblank strings, compared exactly and preserved verbatim,
not a closed enumeration. Blank/duplicate names and empty selections raise
`ValueError`; invalid identifier/container types raise `TypeError`.
Factories take no arguments and must return
fresh unfitted instances, with independent mutable model state. Reusing the
same instance within one fit call is rejected before fitting; freshness across
calls and independence of objects hidden inside adapters remain the factory
author's responsibility.

Representation values must be matrix-like: expose a nonempty, indexable `shape`
whose `shape[0]` is a positive Python/NumPy integer (not a boolean) row count.
No particular rank, feature width, dtype, or backend is imposed here; the
estimator validates those. Arrays, DataFrames, and backend-specific matrix/tensor
inputs pass through unchanged. `Any` preserves this backend flexibility, not
permission to pass arbitrary Python objects. Plain
nested lists lack `shape`; explicitly convert those representation inputs to
arrays. Unselected mapping entries are ignored.

Training targets accept array-like values and become a finite, positive,
one-dimensional float64 array of lifetimes in nanoseconds. Selected training
representations must match its row count. Selected prediction representations
must match each other's row count. **Alignment is positional**: these checks
cannot detect reordered samples or infer correspondence from DataFrame indices.
The caller owns matching row order, units, feature schema, and train/calibration/
test roles. Learn transforms on training data only, or include them inside the
factory-created pipeline. Existing repeated CV still uses its own fold-safe
cloning/representation rules; the new helper does not implement CV.

Predictions have the same nested estimator -> representation keys as fitted
models. They are concrete float64 arrays of shape `(n_samples,)`, with finite
values. Bad shape/nonfinite outputs raise `RuntimeError`, matching the existing
ML benchmark contract. Negative/zero predictions are preserved for evaluation,
not clipped. Prediction takes no targets and never fits, refits, scores, or
calibrates. Interval and multioutput predictions are outside this interface.

### Minimal extension example

This small example uses a noncanonical sklearn estimator, custom identifiers,
and caller-owned representations. It demonstrates the API, not a scientific
performance claim. A runnable version is available in
[`examples/custom_estimator.py`](../examples/custom_estimator.py): run
`python examples/custom_estimator.py` from an installed checkout.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from tcspc_toolkit import (
    EstimatorSpec, fit_regressors, predict_regressors,
)

specs = (EstimatorSpec("my-linear-model", LinearRegression, ("my-features",)),)
fitted = fit_regressors(
    estimator_specs=specs,
    X_train_by_representation={"my-features": np.array([[0.0], [1.0], [2.0]])},
    y_train=[1.0, 2.0, 3.0],
)
predictions = predict_regressors(
    fitted_estimators=fitted,
    X_by_representation={"my-features": np.array([[0.5], [1.5]])},
)
np.testing.assert_allclose(predictions["my-linear-model"]["my-features"], [1.5, 2.5])
```

### Canonical specifications are separate configuration

Canonical benchmark configuration is deliberately module-qualified, not a
root convenience export. User-defined specifications can use the generic
execution API directly without importing canonical experiment definitions.

`tcspc_toolkit.ml.models.make_canonical_ml_estimator_specs()` returns an ordered
tuple with these definitions, all selecting
`("engineered_features", "normalized_histogram", "pca_histogram")`:

| Identifier | Existing zero-argument factory | Unchanged pipeline |
| --- | --- | --- |
| `ridge` | `make_ridge_pipeline` | `StandardScaler` -> `Ridge` |
| `random_forest` | `make_random_forest_pipeline` | `RandomForestRegressor(random_state=42)` |
| `hist_gradient_boosting` | `make_hist_gradient_boosting_pipeline` | `HistGradientBoostingRegressor(random_state=42)` |

All other sklearn defaults remain as before; this pass does not pin or tune
hyperparameters. The factory does not prepare features, normalize histograms,
or fit PCA. It describes the existing three-by-three point-estimator matrix,
not quantile/uncertainty estimators or every existing benchmark variant.
Generic execution imports none of this configuration. A/B, timing, uncertainty,
persistence, and historical report-building paths remain unchanged. The shared
generalization ML execution path is integrated in Stage 2, not consolidated
with those other paths.

## Generalization point-estimator integration (Stage 2)

The existing `tcspc_toolkit.benchmarks.generalization` entry points now accept:

```python
def fit_generalization_ml_estimators(
    prepared: GeneralizationABPreparedData | GeneralizationPreparedData,
    *,
    estimator_specs: Sequence[EstimatorSpec] | None = None,
    X_development_by_representation: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, RegressorProtocol]]: ...

def evaluate_generalization_suite_benchmark(
    *,
    prepared: GeneralizationPreparedData,
    fitted_estimators: Mapping[str, Mapping[str, RegressorProtocol]],
    X_by_test_and_representation: Mapping[str, Mapping[str, Any]] | None = None,
) -> GeneralizationSuiteBenchmarkResult: ...
```

The fitting defaults come from `make_canonical_ml_estimator_specs()` and the
three existing development matrices in `prepared`. With no new arguments,
canonical keys, order, predictions, seeds, and benchmark tables are preserved.
Both existing prepared-data carriers remain supported, not merged.

Custom specs may select any subset of the canonical representations without
supplying another mapping. For custom representations, provide the matrices
explicitly. A supplied mapping **replaces** the canonical mapping: omitted
required keys fail rather than falling back silently. Test mappings are nested
as `{test_id: {representation_name: matrix}}`, using uppercase A-F test IDs.
Every selected representation must be present for every evaluated test.

For example, given an existing `GeneralizationPreparedData` named `prepared`:

```python
from sklearn.linear_model import LinearRegression
from tcspc_toolkit.estimation.api import EstimatorSpec
from tcspc_toolkit.benchmarks.generalization import (
    fit_generalization_ml_estimators, evaluate_generalization_suite_benchmark,
)

columns = ["mean_arrival_time_ns", "peak_time_ns"]
fitted = fit_generalization_ml_estimators(
    prepared,
    estimator_specs=(EstimatorSpec("my-ols", LinearRegression, ("arrival-pair",)),),
    X_development_by_representation={
        "arrival-pair": prepared.development.X_features[columns],
    },
)
result = evaluate_generalization_suite_benchmark(
    prepared=prepared,
    fitted_estimators=fitted,
    X_by_test_and_representation={
        test_id: {"arrival-pair": prepared.X_features[test_id][columns]}
        for test_id in prepared.tests
    },
)
```

This selects already-prepared columns, not a newly fitted representation.
Matrices pass through unchanged and `prepared` is not mutated. Development
rows must match `prepared.development.y`; test rows must match the corresponding
prepared observation batch. Sample ordering and matching feature meanings remain
the caller's responsibility. No training, PCA fitting, or calibration happens
during prediction.

The shared internal ML loop calls `predict_regressors` and iterates its nested
mapping; it knows no Ridge/RF/HGB identities or mandatory representation set.
Baseline calculations still use the original prepared data, even if custom ML
inputs override a canonical representation key. Predictions still feed the
existing prediction/summary/degradation builders, with unchanged table schemas.

The instrument/acquisition and A/F mismatch entry points also use that shared
loop but gain no new arguments: they retain their fixed test selections and
canonical prepared matrices. A/B execution and historical report builders keep
their existing canonical selections. Custom estimator results are not a claim
to reproduce the canonical benchmark and need not satisfy those report builders.

Issue #11 Stage 4 supplies explicit families to internal generalization summaries,
so ML names starting with `classical_reconvolution` are no longer reserved there.
Reusing `constant_mean` or `mean_arrival_time` for ML in the same result conflicts
with the actual baseline identities. The standalone table-only
`summarize_generalization_predictions` retains historical prefix inference for
compatibility; specialized frozen reports retain canonical selections. See the
[evaluation contract](evaluation.md) for the factual/projection boundary.

`tests/test_generalization_estimator_api.py` checks default prediction, summary,
and degradation tables against the former ordered execution loop, plus custom
estimator/representation selection and integration failures. It adds no numeric
snapshot, new frozen protocol, representation framework, or reporting object.


## Scientific API names (Stage 3)

Use these canonical names from the modules listed below; none is newly
exported at the package root. Each legacy name is the **same object** as its
canonical counterpart, not a subclass or forwarding wrapper.

### `benchmarks.generalization`

| Legacy compatibility name | Canonical name |
| --- | --- |
| `Day53ABReport` | `PhotonCountShiftReport` |
| `build_day53_ab_report` | `build_photon_count_shift_report` |
| `Day55ModelMismatchReport` | `DecayModelMismatchReport` |
| `build_day55_model_mismatch_report` | `build_decay_model_mismatch_report` |
| `build_day55_severity_comparison` | `build_model_mismatch_severity_comparison` |
| `Week8RobustnessReport` | `GeneralizationRobustnessReport` |
| `build_week8_robustness_report` | `build_generalization_robustness_report` |

### `uncertainty.ml`

| Legacy compatibility name | Canonical name |
| --- | --- |
| `evaluate_week9_quantile_robustness_conditions` | `evaluate_quantile_interval_robustness` |
| `evaluate_week9_paired_quantile_response` | `evaluate_paired_quantile_interval_response` |
| `evaluate_week9_paired_ml_uncertainty_response` | `evaluate_paired_uncertainty_score_response` |

### `benchmarks.uncertainty`

| Legacy compatibility name | Canonical name |
| --- | --- |
| `DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED` | `DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED` |
| `build_week9_interval_scorecard` | `build_ml_interval_scorecard` |
| `build_week9_score_only_scorecard` | `build_ml_uncertainty_scorecard` |
| `Week9MLUncertaintyRobustnessReport` | `MLUncertaintyRobustnessReport` |
| `build_week9_ml_uncertainty_robustness_report` | `build_ml_uncertainty_robustness_report` |
| `build_week9_classical_uncertainty_scorecard` | `build_classical_uncertainty_scorecard` |
| `build_week9_classical_conditional_scorecard` | `build_classical_conditional_uncertainty_scorecard` |
| `Week9ClassicalUncertaintyRobustnessReport` | `ClassicalUncertaintyRobustnessReport` |
| `evaluate_week9_classical_uncertainty_robustness` | `evaluate_classical_uncertainty_robustness` |
| `Week9ResidualMismatchReport` | `ResidualMismatchReport` |
| `evaluate_week9_residual_mismatch_diagnostics` | `evaluate_residual_mismatch_diagnostics` |

### `persistence`

| Legacy compatibility name | Canonical name |
| --- | --- |
| `record_week9_interval_scorecard_row` | `record_ml_interval_scorecard_row` |
| `record_week9_score_only_scorecard_row` | `record_ml_uncertainty_scorecard_row` |

These are naming changes, not newly generic scientific protocols:

- Photon-count shift reporting still consumes the established A/B results.
- Decay-model mismatch reporting still compares mono-exponential reference
  Test A with bi-exponential Test F, including its severity and residual
  diagnostics. The primary component and signal-photon-weighted references
  remain distinct.
- Generalization robustness synthesis still selects the canonical principal
  estimators across A-F. It is not the generic estimator execution interface.
- Quantile-interval robustness evaluates the five B/D/F conditional regimes;
  paired response functions use the existing matched Test-A references.
- ML interval and uncertainty-score scorecards retain their different meanings.
  Classical covariance/bootstrap and residual reports keep their established
  model-conditional interpretation. Renaming does not broaden calibration claims.
- The classical robustness bootstrap seed remains exactly `62_001`.
- Persistence aliases write the same schema-v1 rows, metric/method identities,
  scope hashes, source-result identities, and duplicate behavior.

Report field names, constructors, and frozen dataclass behavior are unchanged.
New objects use the canonical class name in introspection, repr, and new pickle
references. Alias-qualified globals remain resolvable in canonical modules;
globals naming removed flat modules may no longer load. Tests cover canonical
roundtrips, supported aliases, and deliberate historical-flat failures. This does
not promise older installations can read newer canonical-name pickles. No custom
serialization layer is introduced.

### Retained history and deferred boundaries

The Day/Week audit classifies the remaining occurrences as follows:

| Category | Retained surface and reason |
| --- | --- |
| Frozen protocol/reproducibility identity | `WEEK9_HYPOTHESES` is the historical study hypothesis set, not a generic API configuration. Protocol strings such as `week8-day55-v2`, A-F IDs, stored method/model/source identities, random streams, and numerical seeds stay unchanged. |
| Historical documentation/notebook narrative | Week/Day prose, scientific findings, historical test and artifact filenames, notebook-local report variables, and saved-output/cache labels remain valid provenance. Notebook 13-14 executable API imports/calls use canonical names without regenerating outputs. |
| Specialized/internal scientific boundaries | A/B prepared/result carriers and legacy-compatible reports remain over the Issue-#11 factual layer. Private `_week9_scorecard_row` and `_validate_week9_scorecard_scope` validate the existing scorecard format; they are not public extension points. Explicit `Issue4*` study records remain unchanged. |
| Supported symbol aliases | All 23 aliases above remain supported in canonical modules or the retained persistence package API, without warnings. This does not restore removed flat module paths. |

Issue #11 has since implemented the factual evaluation/reporting layer described
in [evaluation.md](evaluation.md). Issue #3 has completed physical package movement;
neither change was part of the Issue-#2 naming pass documented above.

## Experimental bi-exponential reconvolution (Issue #28 PR 3)

The following APIs are module-qualified in `measurement.adapters`; root and
subpackage exports remain unchanged:

```python
from tcspc_toolkit.measurement.adapters import (
    fit_experimental_biexponential_reconvolution,
    compare_experimental_reconvolution,
)

result = fit_experimental_biexponential_reconvolution(
    measurement, prepared_irf=prepared,
    lifetime_bounds_ns=(0.1, 8.0), temporal_shift_bounds_ns=(-0.3, 0.3),
)
```

Both consume `TCSPCMeasurement.require_raw_counts()`. Integer-looking processed
intensities are rejected. No observation normalization, background subtraction,
resampling, registration or zero-time inference occurs. `initial_guesses` and
`settings` forward to the bi-exponential fitter, including the Issue-33 numerical
safeguards; finite lifetime/shift bounds remain explicit.
`ExperimentalBiexponentialResult.fit` retains its structured
result. `context` snapshots the measurement and selected source, copies read-only
arrays and deep-copies metadata (protected at the top level, following existing
record semantics).

An explicit compatible `PreparedIRF` takes precedence and retains its source
parameters, metadata, provenance and preparation diagnostics. Otherwise the
existing resolver normalizes a copy of the attached same-grid `SampledIRF`;
context records that selection, areas and operation. It does not fabricate
`IRFPreparationDiagnostics`. The low-level fitter still labels this kernel as
a declared normalized array. Source identity, numerical preparation facts and
experimental correctness are different claims.

`compare_experimental_reconvolution(...)` additionally accepts the existing
mono `background_fraction` (default .1), requires at least five bins and shift
bounds including zero, and returns `ExperimentalReconvolutionComparison`:
shared context, original mono result, original bi result, reconstructed read-only
mono expectation, actual bounds, and `ReconvolutionStatisticsComparison`.
Unavailable curves remain `None`; finite failed fits retain descriptive statistics
alongside their original validity flags.

The pure `evaluation.reconvolution.compare_reconvolution_statistics(y, mono_mu,
bi_mu)` recomputes the PR-1 full Poisson likelihood (including factorial terms),
deviance, AIC and bin-based BIC. Use nominal p=4/6 even at zero-component/equal-tau
boundaries, and n=all bins including zeros. All differences are **bi minus mono**.
Invalid arrays raise; missing curves or nonfinite likelihoods produce explained
unavailable differences, with no expectation floor. The unchanged mono bounds
are A>=0, tau>=1e-12 with no upper bound, B>=1e-12; bi has A,B>=0 and the supplied
finite tau interval. Shared shift bounds do not make the feasible sets identical.
Mono tau outside bi bounds is explicitly flagged. No automatic model order,
ordinary chi-square likelihood-ratio p-value or physical-validity certificate
is returned.

See the [executable example](../examples/biexponential_reconvolution.py) and
[completed qualification record](issue28_pr3_qualification.md). Primary C00
numerical-acceptance, component-median-error and bias targets passed. Independent
optimizer agreement failed in two designated cases (119/121 passed); both were
rejected by local validation. That historical outcome remains immutable. The
subsequent [Issue-33 study](issue33_stage3_qualification_v1.md) uses new observations:
99/100 primary numerical acceptance and passing component recovery/bias, but a
failed strict independent-agreement target of 99/100 against 100/100.
The numerical failure contract retains the best finite candidate with explicit
diagnostics; it does not promise global optimality or physical identifiability.
