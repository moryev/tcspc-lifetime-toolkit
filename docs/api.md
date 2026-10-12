# Public API and estimator extensions

This document defines the Issue-#2 public API contract for version 0.7.0:
a curated package-root convenience API, supported module-qualified workflows,
and a generic estimator extension boundary separate from frozen benchmark
configuration. Issue #11 has since added a module-qualified factual evaluation
layer beneath the existing scientific reports. Issue #3 has begun structural
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
| Generic extension API | `RegressorProtocol`, `EstimatorSpec`, `fit_regressors`, and `predict_regressors` are root exports; canonical `estimation.api` and compatibility `estimator_api` imports resolve to the same objects. |
| Stage-2 integration | Generalization ML fitting and shared final-test prediction consume that API; only the A-F suite entry point gains explicit custom test mappings. |
| Stage-3 naming | Scientific module-qualified names below are canonical; historical Day/Week names remain direct compatibility aliases. |
| Existing APIs | Current root exports and documented module-qualified workflows, preserved rather than globally restabilized in this pass. |
| Frozen configuration | `ml.models.make_canonical_ml_estimator_specs()` describes the established estimator/representation matrix, not a restriction on generic execution. |
| Evaluation architecture | Issue #11 adds module-qualified point/reference facts and interval/score attachments beneath legacy-compatible scientific reports; see [the evaluation contract](evaluation.md). These are not new root exports. |
| Package movement | Issue #3 places estimator contracts/baselines in `estimation` (Stage 1), physical primitives in `physics` (Stage 2A), histogram operations/features/representations in `preprocessing` (Stage 2B), IRF profiles/preparation/estimation in `irf` (Stage 3A), records/I/O/experimental adapters in `measurement` (Stage 3B), generic histogram/dataset generation in `simulation` (Stage 3C), core classical fitting/execution in `estimation` (Stage 4A), Bayesian inference in `estimation.bayesian` (Stage 4B), ML models/evaluation/cross-validation in `ml` (Stage 4C), historical diagnostics in `evaluation.diagnostics` (Stage 5A), factual contracts/builders/attachments in explicit `evaluation` submodules (Stage 5B), conditional analysis/plotting in `evaluation.conditional` and `evaluation.plots` (Stage 5C), and frozen Week-8 protocol/data generation in explicit `benchmarks` submodules (Stage 6A), and the intact generalization execution/reporting module in `benchmarks.generalization` (Stage 6B), and uncertainty source/metric/split/provenance contracts in `uncertainty.evaluation` (Stage 7A), and intact ML/classical uncertainty in `uncertainty.ml` and `uncertainty.classical` (Stage 7B), and Week-7 mismatch/timing studies in `benchmarks.mismatch` and `benchmarks.timing` (Stage 8A), and Issue-8 IRF/Issue-4 calibration and mismatch studies in `benchmarks.irf`, `benchmarks.bayesian_calibration`, and `benchmarks.bayesian_mismatch` (Stage 8B), and intact frozen Week-9 orchestration/reporting in `benchmarks.uncertainty` (Stage 8C), and the intact SQLite implementation in `persistence.sqlite` with the historical/public `persistence` package facade (Stage 9). |

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
All pre-Issue-#2 root exports are preserved. Stage 4 adds only the four generic
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

- `physics.models`, `physics.convolution`, and `physics.forward_model`:
  decay models, numerical convolution, and the shared deterministic
  reconvolution expected-count model. Flat `models`, `convolution`, and
  `forward_model` imports remain direct-object compatibility paths.
- `estimation.api` and `estimation.baselines`: canonical generic estimator
  execution and baseline implementations. `estimator_api` and `baselines` remain
  direct-object compatibility paths through at least Issue #12.
- `estimation.fitting` and `estimation.classical`: classical fitting,
  initialization, per-curve execution, batch benchmarks, and fit diagnostics.
  Flat `fitting` and `classical_evaluation` remain direct-object compatibility
  paths through at least Issue #12; these are not generic factual evaluation.
- `simulation.histograms` and `simulation.datasets`: deterministic expected
  counts, Poisson sampling, and ordinary/grouped synthetic datasets. Historical
  `simulation` and `datasets` imports remain direct-object compatibility paths;
  the `simulation` facade does not aggregate dataset or frozen benchmark APIs.
- `irf.profiles`, `irf.preparation`, and `irf.estimation`: source profiles,
  explicit prepared kernels, and failure-aware leading-edge proxies. Historical
  `irf`, `irf_preparation`, and `irf_estimation` imports remain supported;
  the `irf` package facade exposes only the former profile-level API.
- `measurement.records`, `measurement.io`, and `measurement.adapters`:
  observation/provenance carriers, explicit-unit CSV import, and higher-level
  experimental fitting/features/reference comparison. Flat `measurements`,
  `measurement_io`, and `experimental` remain direct-object compatibility paths.
- `preprocessing.histograms`, `preprocessing.features`, and
  `preprocessing.representations`: histogram/time-grid operations, engineered
  features, normalization, PCA, and count augmentation. The historical
  `preprocessing`, `features`, and `representations` paths remain supported.
- `ml.models`, `ml.evaluation`, `ml.cross_validation`, and `preprocessing.representations`:
  estimator factories (including `make_canonical_ml_estimator_specs()`),
  point-estimate evaluation, repeated CV and its configuration/result types,
  and transforms. Flat `ml_models`, `ml_evaluation`, and `cross_validation`
  remain direct-object compatibility paths through at least Issue #12.
- `benchmarks.generalization_protocol` and `benchmarks.generalization_datasets`:
  frozen Week-8 A–F definitions and paired observation generation. Flat
  `generalization` and `generalization_datasets` remain direct-object
  compatibility paths through at least Issue #12.
- `benchmarks.generalization` (historical `generalization_evaluation`),
  `benchmarks.irf`, `benchmarks.bayesian_calibration`, `benchmarks.bayesian_mismatch`,
  `benchmarks.mismatch`, and `benchmarks.timing`: scientific protocols and
  specialized evaluation/reporting paths and report/result classes, not a
  unified generic interface. Prepared-data carriers remain workflow-specific,
  not package-root abstractions.
- `uncertainty.evaluation`: source interval/score carriers, uncertainty metrics,
  and development/calibration split and provenance policy. The historical
  `uncertainty_evaluation` path remains a direct-object compatibility facade.
  Its existing A–F provenance guard depends on frozen benchmark protocol configuration.
- `uncertainty.classical` and `uncertainty.ml`: distinct model-conditional
  intervals, calibration, sensitivity, and disagreement-score workflows.
  Historical `classical_uncertainty` and `ml_uncertainty` imports remain
  direct-object compatibility facades.
- `benchmarks.uncertainty`: the frozen Week-9 study/report layer, including its
  existing conformal workflow. Historical `uncertainty_robustness` imports remain
  a direct-object compatibility facade.
- `estimation.bayesian.posterior`, `estimation.bayesian.sampling`, and
  `estimation.bayesian.predictive`: explicit priors, posterior/context,
  sampling, and predictive diagnostics. Flat `bayesian`, `bayesian_sampling`,
  and `bayesian_predictive` remain direct-object compatibility paths through
  at least Issue #12. No Bayesian objects are root exports. Study orchestration
  remains in `bayesian_evaluation` and `bayesian_mismatch_evaluation`.
- `persistence`: optional SQLite schema-v1 storage, scientific recording
  adapters, and read-only queries; see [persistence.md](persistence.md).
- `evaluation.results`, `evaluation.core`, and `evaluation.attachments`:
  factual point/reference contracts, explicit-population primitives, and separate
  interval/score attachments. Flat `evaluation_results`, `evaluation_core`,
  and `evaluation_uncertainty` remain direct-object compatibility paths through
  at least Issue #12. These objects are not root or diagnostics-facade exports.
- `evaluation.conditional` and `evaluation.plots`: reusable TCSPC conditional
  analysis and diagnostic-table plotting. Flat `conditional_evaluation` and
  `benchmark_plots` remain direct-object compatibility paths through at least
  Issue #12. Neither module is aggregated by the diagnostics facade.

This inventory does not promote every implementation helper to public API.
Private validation, prediction-table construction, SQL internals, and
study-specific orchestration are not generic extension hooks.

## Compatibility principles

- Existing imports, argument signatures, result fields, numerical behavior,
  and benchmark defaults remain compatible. Stage-3 names use direct aliases,
  with no wrappers or runtime deprecation warnings. Old names remain supported
  at least through Issue #12; removal requires a future explicit decision.
- Public renames use the reviewed old-to-new mapping and caller audit below.
  Use scientific terminology, not a blanket `Frozen...` prefix.
  Preserve Day/Week provenance where historically meaningful, and preserve
  A-F identifiers, protocol labels, seeds, and scientific reference outputs.
- Root exports are deliberate stability commitments, not an inventory of all
  implemented capabilities. Bayesian, CV, benchmark, evaluation, and persistence
  types remain supported through their documented module-qualified paths.
- Issue #3 should preserve approved stable imports when implementations move;
  compatibility shims must be deliberate, not created speculatively now.
- Python naming changes must not reinterpret SQLite schema-v1 method/model
  identities, serialized provenance, or stored source-result-type strings.
- Keep the Issue-#1 typing rule: suitable normalized inputs use `ArrayLike`;
  genuine ndarray requirements, concrete ndarray returns, and stored arrays
  retain their intentional contracts. There is no new package-wide typing sweep.

## Boundaries and deferred work

### Issue-#3 structural migration contract

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

| Canonical implementation | Historical direct-export facade |
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

| Canonical implementation | Historical direct-export facade |
| --- | --- |
| `tcspc_toolkit.evaluation.conditional` | `tcspc_toolkit.conditional_evaluation` |
| `tcspc_toolkit.evaluation.plots` | `tcspc_toolkit.benchmark_plots` |

Conditional analysis retains both legacy-result adapters and explicit factual
paths; it does not select frozen A–F conditions. Plotting remains a separate
explicit import: importing `tcspc_toolkit.evaluation` does not load conditional
analysis or Matplotlib. Benchmark, uncertainty, and persistence modules have
not moved in these stages.

Stage 6A creates the docstring-only `benchmarks` namespace and moves
the frozen Week-8 protocol/configuration and dataset-generation implementations.
Stage 6B relocates generalization preparation, execution, and reporting intact:

| Canonical implementation | Historical direct-export facade |
| --- | --- |
| `tcspc_toolkit.benchmarks.generalization_protocol` | `tcspc_toolkit.generalization` |
| `tcspc_toolkit.benchmarks.generalization_datasets` | `tcspc_toolkit.generalization_datasets` |
| `tcspc_toolkit.benchmarks.generalization` | `tcspc_toolkit.generalization_evaluation` |

The initializer imports no children. Protocol definitions, A–F ordering, seeds,
paired designs, RNG behavior, and development/external roles are unchanged.
The single `benchmarks.generalization` implementation retains all preparation,
execution, private helpers, result/report carriers, table compatibility policies,
and seven Day/Week aliases. Its historical facade directly re-exports the 61
supported names; implementation-owned monkeypatches target the canonical module.
PCA and estimators still fit only on development data and reuse fitted artifacts
on external A–F tests. None of these names is added to the 66-name root API.
Other benchmark studies, uncertainty implementations, and persistence remain
in their existing locations. The uncertainty provenance guard still depends
explicitly on the frozen protocol; ML uncertainty's protocol/data/prepared-data
annotations remain typing-only imports. Uncertainty robustness imports the
canonical prepared-data carrier eagerly; persistence imports `RobustnessMetrics`
locally, preserving schema v1 and stored identities. All three flat paths remain supported
through at least Issue #12.

Stages 7A–7B retain the docstring-only `uncertainty` namespace and place
three intact implementations in explicit submodules:

| Canonical implementation | Historical direct-export facade |
| --- | --- |
| `tcspc_toolkit.uncertainty.evaluation` | `tcspc_toolkit.uncertainty_evaluation` |
| `tcspc_toolkit.uncertainty.ml` | `tcspc_toolkit.ml_uncertainty` |
| `tcspc_toolkit.uncertainty.classical` | `tcspc_toolkit.classical_uncertainty` |

The 19 uncertainty-evaluation names retain their identity through the facade. Source carriers
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

| Canonical implementation | Historical direct-export facade |
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

| Canonical implementation | Historical direct-export facade |
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

- Established flat module imports remain compatibility paths through at least
  Issue #12. Removal requires a separate explicit compatibility decision.
  Canonical implementations will live in subpackages; old paths must re-export
  identical objects, not wrappers or parallel implementations. Issue-#2
  scientific names and direct Day/Week aliases retain the same policy.
- New subpackage initializers stay minimal: do not eagerly import an entire
  subtree. For existing names that become packages (`irf`, `simulation`,
  `preprocessing`, `evaluation`, `persistence`), the new `__init__.py` must also
  provide the reviewed old-path facade. Do not leave competing `name.py` and
  `name/` implementations whose import precedence obscures ownership.
- Existing serialized class references must resolve through old paths.
  Canonical classes may acquire their new implementation `__module__`; do not
  fake historical metadata. Persisted `source_result_type` strings are instead
  historical schema-v1 provenance identities, not dynamically resolved import
  paths, and must not be mechanically rewritten during moves.
- Import compatibility does not promise arbitrary module-global monkeypatch
  compatibility. Tests that replace an implementation dependency should patch
  its canonical implementation module after relocation, not a re-export shim.
- Generic estimator execution remains independent of sklearn/canonical model
  factories. Generic factual evaluation remains independent of estimators,
  frozen orchestration, Bayesian sampling, and persistence. Physical primitives
  stay below reporting; scientific computation must not depend on persistence.
  Root imports must neither require `emcee` nor initialize persistence/SQLite.

The structural guards in `test_package_structure.py`, the exact root contract
in `test_public_api.py`, legacy report-pickle/alias tests, and persistence
source-identity tests are the incremental migration gate. The import graph
distinguishes eager, function-local, and `TYPE_CHECKING` dependencies; fresh
processes also check real import order and side effects. The intentionally
deferred Bayesian sampling relationship is not an eager cycle.

The implemented slices have these paths:

| Canonical implementation | Compatibility import |
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
Direct old/new/root identity, natural canonical `__module__`, historical path
resolution, and unchanged baseline/estimator behavior are tested. Future
moves must extend these cases and update responsibility path sets deliberately.

The fitting and classical execution implementations do not broaden
`estimation/__init__.py`; request their modules explicitly. Fitting uses the
shared `physics.forward_model`. Classical execution retains its existing
`evaluation.py` dependency for Poisson-deviance diagnostics; that legacy
diagnostic module remains flat until its own migration. The generic factual
evaluation contracts do not depend on these estimators. Optimizers, objectives,
initialization, validity/failure policy, diagnostics, and timing are unchanged.
Schema-v1 source identities remain
`tcspc_toolkit.fitting.ReconvolutionFitResult` and
`tcspc_toolkit.classical_evaluation.ReconvolutionCurveResult`, despite executable
imports and natural `__module__` values moving to their canonical locations.
Historical class paths still resolve through the direct-reexport facades.

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
submodules (or their flat compatibility facades) explicitly. The old `irf.py`
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
Flat `datasets` directly re-exports `SyntheticDataset`, both dataset generators,
and the existing array type aliases. Expected-count construction, Poisson
sampling, seed/Generator handling, and random-draw ordering are unchanged.
This is generic synthetic generation, not the frozen A–F protocol: the
`generalization*` modules remain separate and have not moved in this stage.

### Stage-8C frozen uncertainty benchmark

| Canonical implementation | Historical direct-export facade |
| --- | --- |
| `tcspc_toolkit.benchmarks.uncertainty` | `tcspc_toolkit.uncertainty_robustness` |

The whole Week-9 orchestration/reporting module remains intact. It consumes reusable
`uncertainty.evaluation`, `uncertainty.ml`, and `uncertainty.classical`; it does not
replace their source carriers or generic factual attachments. Frozen A–F traversal,
calibration/external separation, scorecard populations, residual diagnostics, and
conformal calibration/external evaluation retain their existing policies. There is
no separate conformal module. Repeated-Poisson variability remains an empirical
reference, not a per-point attachment.

All 28 supported names below are direct identical objects through both paths.
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
existing canonical targets from the scientific-name mapping below. Historical
class/function pickle lookup resolves through the facade without changing
`__module__`.

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
- Issue #3: incrementally relocate remaining implementations; only the
  `estimation`, `physics`, and `preprocessing` modules listed above have moved so far.
- Issue #12: perform full post-refactor integration and scientific-reference
  verification; the small equivalence tests below do not replace it.

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
example `from tcspc_toolkit.estimation.api import EstimatorSpec`. Imports from
`tcspc_toolkit.estimator_api` remain supported through at least Issue #12 and
resolve to the same objects. Their signatures are:

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

### `ml_uncertainty`

| Legacy compatibility name | Canonical name |
| --- | --- |
| `evaluate_week9_quantile_robustness_conditions` | `evaluate_quantile_interval_robustness` |
| `evaluate_week9_paired_quantile_response` | `evaluate_paired_quantile_interval_response` |
| `evaluate_week9_paired_ml_uncertainty_response` | `evaluate_paired_uncertainty_score_response` |

### `benchmarks.uncertainty` (compatibility: `uncertainty_robustness`)

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
references. Old module-qualified pickle references remain resolvable through
the aliases; compatibility tests cover legacy report payloads and new round
trips. This does not promise that older toolkit installations can read newly
written canonical-name pickles. No custom serialization layer is introduced.

### Retained history and deferred boundaries

The Day/Week audit classifies the remaining occurrences as follows:

| Category | Retained surface and reason |
| --- | --- |
| Frozen protocol/reproducibility identity | `WEEK9_HYPOTHESES` is the historical study hypothesis set, not a generic API configuration. Protocol strings such as `week8-day55-v2`, A-F IDs, stored method/model/source identities, random streams, and numerical seeds stay unchanged. |
| Historical documentation/notebook narrative | Week/Day prose, scientific findings, historical test and artifact filenames, notebook-local report variables, and saved-output/cache labels remain valid provenance. Notebook 13-14 executable API imports/calls use canonical names without regenerating outputs. |
| Specialized/internal scientific boundaries | A/B prepared/result carriers and legacy-compatible reports remain over the Issue-#11 factual layer. Private `_week9_scorecard_row` and `_validate_week9_scorecard_scope` validate the existing scorecard format; they are not public extension points. Explicit `Issue4*` study records remain unchanged. |
| Legacy public imports | All 23 aliases above remain supported at least through Issue #12, without warnings. Removal is a separate future compatibility decision. |

Issue #11 has since implemented the factual evaluation/reporting layer described
in [evaluation.md](evaluation.md). Issue #3 will address physical package movement;
neither change was part of the Issue-#2 naming pass documented above.
