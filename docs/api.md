# Public API and estimator extensions

This document defines the Issue-#2 public API contract for version 0.7.0:
a curated package-root convenience API, supported module-qualified workflows,
and a generic estimator extension boundary separate from frozen benchmark
configuration. Evaluation consolidation (#11) and package movement (#3)
remain separate work, not implemented interfaces.

| Status | Surface |
| --- | --- |
| Generic extension API | `RegressorProtocol`, `EstimatorSpec`, `fit_regressors`, and `predict_regressors` are root exports; their `estimator_api` imports resolve to the same objects. |
| Stage-2 integration | Generalization ML fitting and shared final-test prediction consume that API; only the A-F suite entry point gains explicit custom test mappings. |
| Stage-3 naming | Scientific module-qualified names below are canonical; historical Day/Week names remain direct compatibility aliases. |
| Existing APIs | Current root exports and documented module-qualified workflows, preserved rather than globally restabilized in this pass. |
| Frozen configuration | `ml_models.make_canonical_ml_estimator_specs()` describes the established estimator/representation matrix, not a restriction on generic execution. |
| Deferred architecture | Issue #11 evaluation consolidation and Issue #3 physical package movement are plans, not established interfaces. |

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

Important module-qualified workflows remain in their existing flat modules:

- `ml_models`, `ml_evaluation`, `cross_validation`, and `representations`:
  estimator factories (including `make_canonical_ml_estimator_specs()`),
  point-estimate evaluation, repeated CV and its configuration/result types,
  and transforms.
- `generalization`, `generalization_datasets`, `generalization_evaluation`,
  `classical_evaluation`, `conditional_evaluation`, `irf_evaluation`,
  `mismatch_evaluation`, and `timing_evaluation`: scientific protocols and
  specialized evaluation/reporting paths and report/result classes, not a
  unified generic interface. Prepared-data carriers remain workflow-specific,
  not package-root abstractions.
- `classical_uncertainty`, `ml_uncertainty`, `uncertainty_evaluation`, and
  `uncertainty_robustness`: distinct model-conditional intervals, calibration,
  sensitivity, and disagreement-score workflows.
- `bayesian`, `bayesian_sampling`, `bayesian_predictive`, `bayesian_evaluation`,
  and `bayesian_mismatch_evaluation`: explicit priors, inference, diagnostics,
  and study-specific comparisons. Their configuration/result types are not
  newly promoted to the root.
- `persistence`: optional SQLite schema-v1 storage, scientific recording
  adapters, and read-only queries; see [persistence.md](persistence.md).

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

Stage 1 established a generic estimator execution interface and a separate
canonical specification factory. Stage 2 connects the generalization fitting
and shared final-test ML prediction boundary to it, as described below.
The dependency direction is canonical specification -> generic contract;
generic execution must not know Ridge/RF/HGB or canonical representation names.

- Issue #11: consolidate A/B, A/C/D/E, and A-F evaluation paths, duplicated
  prediction/summary/degradation machinery, overlapping result/report types,
  and reporting integration across classical/ML/uncertainty/Bayesian/experimental
  workflows. Stage 1 introduces no unified report or representation framework.
- Issue #3: move files and introduce subpackages. No relocation occurs here.
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

Imports from `tcspc_toolkit.estimator_api` remain supported and resolve to the
same canonical objects. Their signatures are:

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

`tcspc_toolkit.ml_models.make_canonical_ml_estimator_specs()` returns an ordered
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

The existing `tcspc_toolkit.generalization_evaluation` entry points now accept:

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
from tcspc_toolkit.estimator_api import EstimatorSpec
from tcspc_toolkit.generalization_evaluation import (
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

Existing reporting distinguishes baselines and classical methods by name.
At this boundary, `constant_mean`, `mean_arrival_time`, and names starting with
`classical_reconvolution` are therefore reserved, unlike the unrestricted generic
Stage-1 estimator API. Broader method-identity/reporting design remains Issue #11.

`tests/test_generalization_estimator_api.py` checks default prediction, summary,
and degradation tables against the former ordered execution loop, plus custom
estimator/representation selection and integration failures. It adds no numeric
snapshot, new frozen protocol, representation framework, or reporting object.


## Scientific API names (Stage 3)

Use these canonical names from their existing flat modules; none is newly
exported at the package root. Each legacy name is the **same object** as its
canonical counterpart, not a subclass or forwarding wrapper.

### `generalization_evaluation`

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

### `uncertainty_robustness`

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
| Deferred/internal scientific boundaries | A/B preparation/result carriers and report consolidation remain Issue #11. Private `_week9_scorecard_row` and `_validate_week9_scorecard_scope` still validate the existing scorecard format; they are not public extension points. Explicit `Issue4*` study records remain unchanged. |
| Legacy public imports | All 23 aliases above remain supported at least through Issue #12, without warnings. Removal is a separate future compatibility decision. |

Issue #11 will address evaluation/report consolidation; Issue #3 will address
physical package movement. Neither is implemented by these naming changes.
