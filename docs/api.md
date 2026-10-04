# Public API and estimator extensions

This inventory accompanies Issue #2, Stages 0-2, starting from version 0.7.0
at `74b4cd0b14d52348bd5f0136fb059f41af4821f0`. It is not a declaration that
API stabilization or the later evaluation/package refactors are complete.

| Status | Surface |
| --- | --- |
| New supported Issue-#2 API | The four module-qualified symbols in `estimator_api` documented below; root promotion is not part of Stage 1. |
| Stage-2 integration | Generalization ML fitting and shared final-test prediction consume that API; only the A-F suite entry point gains explicit custom test mappings. |
| Existing APIs | Current root exports and documented module-qualified workflows, preserved rather than globally restabilized in this pass. |
| Frozen configuration | `ml_models.make_canonical_ml_estimator_specs()` describes the established estimator/representation matrix, not a restriction on generic execution. |
| Deferred architecture | Issue #11 evaluation consolidation and Issue #3 physical package movement are plans, not established interfaces. |

## Existing public API inventory

The explicit `tcspc_toolkit.__all__` remains the curated package-root contract.
Stage 1 does not change root imports or exports. Module-qualified workflows
are also public where documented; absence from the root does not make them
private. Internal names beginning with `_` are not extension points.

| Area | Current package-root names |
| --- | --- |
| Models and simulation | `monoexponential_decay`, `sample_photon_counts`, `SyntheticDataset`, `generate_monoexponential_dataset` |
| IRF sources | `IRFSourceKind`, `IRFProfile`, `generate_gaussian_irf`, `generate_emg_irf`, `generate_gaussian_irf_profile`, `generate_emg_irf_profile`, `normalize_irf`, `shift_irf` |
| IRF preparation and estimation | `IRFPreparationDiagnostics`, `PreparedIRF`, `irf_profile_from_sampled_irf`, `prepare_irf`, `LeadingEdgeIRFDiagnostics`, `LeadingEdgeIRFResult`, `estimate_irf_from_leading_edge` |
| Classical inference | `convolve_decay_with_irf`, `LifetimeFitResult`, `ReconvolutionFitResult`, `fit_monoexponential_decay`, `fit_monoexponential_reconvolution` |
| Measurement boundary | `TimeUnit`, `MeasurementDataKind`, `SampledIRF`, `TCSPCMeasurement`, `convert_time_to_ns`, `load_sampled_irf_csv`, `load_tcspc_measurement_csv` |
| Experimental adapters | `ReferenceLifetimeEvaluation`, `fit_experimental_reconvolution`, `measurement_to_feature_table`, `measurement_to_histogram_batch`, `evaluate_estimate_against_reference` |
| Preprocessing | `align_to_irf`, `crop_time_window`, `detect_peak`, `estimate_background`, `normalize_counts`, `rebin_histogram`, `subtract_background`, `validate_histogram` |
| Configuration | `CountNormalization`, `FeatureConfig`, `PreprocessingConfig`, `SimulationConfig`, `load_config`, `save_config` |
| Features and representations | `FEATURE_NAMES`, `extract_feature_table`, `extract_features`, `cumulative_explained_variance`, `fit_pca_representation`, `normalize_histogram_batch`, `transform_pca_representation` |
| Errors | `TCSPCError`, `InvalidHistogramError`, `InvalidMeasurementError`, `FeatureExtractionError` |

`build_expected_counts_from_irf` is currently imported at the root but omitted
from `__all__`. This known discrepancy is deliberately reserved for the later
public-export review; prefer its documented `tcspc_toolkit.simulation` path.

Important module-qualified workflows remain in their existing flat modules:

- `ml_models`, `ml_evaluation`, `cross_validation`, and `representations`:
  estimator factories, point-estimate evaluation, repeated CV, and transforms.
- `generalization`, `generalization_datasets`, `generalization_evaluation`,
  `classical_evaluation`, `conditional_evaluation`, `irf_evaluation`,
  `mismatch_evaluation`, and `timing_evaluation`: scientific protocols and
  specialized evaluation/reporting paths, not a unified generic interface.
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

- Existing imports, call signatures, return types, numerical behavior, and
  benchmark defaults remain unchanged in this additive pass. There are no
  compatibility aliases or deprecation warnings to add yet.
- Any later public rename needs a reviewed old-to-new mapping and caller
  audit. Use scientific terminology, not a blanket `Frozen...` prefix.
  Preserve Day/Week provenance where historically meaningful, and preserve
  A-F identifiers, protocol labels, seeds, and scientific reference outputs.
- Root exports will be reviewed together in Stage 4, not expanded automatically
  to every Bayesian, CV, classical-benchmark, or evaluation result object.
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

- Later Issue-#2 stages: separately approve scientific renames and root exports.
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

## Supported Issue-#2 extension API (Stage 1)

Import the new interface from `tcspc_toolkit.estimator_api`, not the package
root. Its four public symbols are:

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
performance claim. Behavioral tests cover the same extension contract.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from tcspc_toolkit.estimator_api import (
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
