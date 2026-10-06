# Evaluation architecture: Issue #11, Stages 0–5

Stages 0–1 added module-qualified factual result contracts and small metric
primitives. Stage 2 shares representation-preparation mechanics beneath the
existing A/B and multi-test carriers. Stage 3 routes nonclassical generalization
point evaluation through those facts and projects back to unchanged legacy tables.
Stage 4 adapts classical generalization fits into the same point/reference facts
and supplies explicit method families to internal summaries. Stage 5 adds explicit
method/reference paths for conditional diagnostics, retaining a compatibility-only
path for historical callers lacking those semantics. Uncertainty integration,
report consolidation and notebook migration remain later reviewed stages.
The 66-name root API is unchanged.

## Responsibilities and existing inventory

| Current location | Carriers / entry points | Responsibility and consolidation boundary |
|---|---|---|
| `ml_evaluation` | `BenchmarkMeasurements`, `BenchmarkDataset`, `BenchmarkSplit`, `HistogramRepresentations`; dataset/split/representation builders | Synthetic benchmark inputs, aligned training/test membership, development-only learned representations. Preserve indices, feature definitions, normalization and fitted PCA. |
| `generalization`, `generalization_datasets` | `GeneralizationSuiteDefinition`, protocol/numerics/domain objects, `GeneralizationTestMeasurements`, `GeneralizationTestSuite` | Frozen A–F definitions, seeds, metadata and generated observations. These are not arbitrary evaluation-batch types. |
| `generalization_evaluation` | `GeneralizationABPreparedData`, `GeneralizationPreparedData`; both preparation functions | Shared feature/normalization/PCA mechanics beneath unchanged carriers. A/B additionally validates identities and paired targets. Learned artifacts stay outside the factual result core. |
| `ml_evaluation`, `mismatch_evaluation` | `RegressionBenchmarkResult`, `RegressionMetrics`; ordinary and mismatch result builders | Array predictions and reference-based metrics; overlapping validation/error construction with differing exception contracts. |
| `generalization_evaluation` | Principal/representation A/B, instrument/acquisition, model-mismatch and full-suite benchmark results | Overlapping prediction/summary/degradation tables. Specialized results additionally retain comparisons, fit diagnostics or Test-F severity/reference diagnostics. |
| `conditional_evaluation` | Prediction-diagnostic builders; conditional and standard-regime summaries | Group-specific error/failure reporting. Legacy columns and invalid-row visibility differ from generalization. |
| `classical_evaluation` | `ReconvolutionCurveResult`, `ReconvolutionBenchmarkResult`, `ReconvolutionBenchmarkSummary` | Numerical fit facts, initialization, optimizer/numerical validity, boundaries, NLL/deviance, failures and timings. Keep fit details separate from point projections. |
| `cross_validation` | `RepeatedCVConfig`, `RepeatedCVBenchmarkResult` | Development-only fold-local fitting/cloning, repeated-fold metrics. Not final-test orchestration. |
| `mismatch_evaluation`, `timing_evaluation` | Paired estimator/classical mismatch results; `InferenceTimingResult` | Paired reference/shift comparisons and named timing scopes. Batch prediction and per-curve optimization timing are not interchangeable. |
| `ml_uncertainty` | Quantile/score calibration, external, robustness and paired result families | Fitted uncertainty artifacts, calibration separation and frozen conditional evaluation. Shared traversal is a later candidate, not shared scientific interpretation. |
| `uncertainty_evaluation` | Interval/score results, method definitions, development split, interval/quantile/score/selective metrics | Already separates intervals, heuristic scores and empirical reference variability. Retain that distinction. |
| `uncertainty_robustness` | Conformal calibration/external results; `MLUncertaintyRobustnessReport`, `ClassicalUncertaintyRobustnessReport`, `ClassicalResidualMatrixResult`, `ResidualMismatchReport` | Separate interval/score scorecards, conditional classical uncertainty, residual profiles and paired diagnostics. |
| `classical_uncertainty` | Local covariance, parametric bootstrap and repeated-Poisson result types | Per-fit conditional uncertainty versus repeated-observation empirical variability; retain separate types. |
| `generalization_evaluation` reports | `PhotonCountShiftReport`, `InstrumentAcquisitionDiagnostics`, `DecayModelMismatchReport`, `GeneralizationRobustnessReport` | Scientifically specialized frozen composition over tables, not candidates for one universal report. |
| `irf_evaluation` | Shape-mismatch dataset/pairs, classical/conditional/ML-transfer results, leading-edge regime results | Paired observations with explicit assumed IRFs, source/proxy diagnostics and sometimes no unique mono-exponential target. |
| `experimental` | Feature/histogram adapters, `ReferenceLifetimeEvaluation` | Estimation without intrinsic truth; trusted-reference comparison only when explicitly supplied. No representation refitting. |
| Bayesian modules | `BayesianReconvolutionResult`, `BayesianInferenceRun`, predictive results, `BayesianCalibrationReport`, `MismatchReport`, `Issue4*` study records | Posterior/prior/fixed-IRF semantics, accepted/rejected sampling, PPC, primary versus pseudo-true comparisons. Adapt reporting later; do not replace inference or scientific records. |
| `persistence` | Point, uncertainty, metric and scientific recording adapters | External consumer of existing concrete types and identities. Schema v1, source-result identities and duplicate semantics remain unchanged. |

The reusable estimator boundary remains `estimator_api`: structural `fit`/`predict`,
factory-created state and caller-prepared matrices. Canonical specifications remain
in `ml_models`. Neither new evaluation module imports those canonical definitions.

## Shared preparation mechanics (Stage 2)

```text
frozen measurement/configuration carriers
                  ↓
shared representation-preparation mechanics
                  ↓
unchanged A/B and multi-test prepared-data carriers
                  ↓
baseline/ML factual evaluation and legacy table projection (Stage 3)
```

`prepare_generalization_ab_data` and `prepare_generalization_data` delegate to
the private `_prepare_generalization_representations` helper in
`generalization_evaluation`. It reuses `GeneralizationPreparedData` internally;
no new public carrier, preparation framework or `PreparedEvaluationData` exists.

The common mechanics build the development dataset, extract ordered test features,
check feature schemas, TOTAL-normalize each histogram, fit one full-SVD PCA on
normalized development histograms only, and transform development and each final
test with that same artifact. Test targets and metadata never enter PCA fitting.
Feature configuration, PCA defaults, float64 representations and insertion order
are unchanged. Feature tables keep their positional RangeIndex; metadata keeps
its original index. The development dataset retains its existing copies of raw
histograms, targets and metadata; final-test carriers remain the original objects.
Preparation does not mutate either source. The A/B projection only assigns matrix
references, without extra matrix copies. Both public carriers still expose `pca`.

The A/B wrapper retains exact A/B identities, equal time axes and paired lifetime
targets, including its existing schema-error messages. Pair-ID/protocol-metadata
validation belongs to `GeneralizationTestSuite`; preparation neither weakens that
validation nor adds a new membership policy. Multi-test dictionary keys retain
strip/uppercase normalization, uniqueness/key-match checks and caller ordering.
`GeneralizationTestMeasurements` and `GeneralizationTestSuite` remain frozen A–F
carriers; arbitrary evaluation IDs still belong to `EvaluationBatch`.

No `EvaluationBatch` views were needed for preparation deduplication; Stage 3
constructs them at the execution boundary below. PCA is not hidden in a batch.
Test-F targets and descriptive weighted lifetimes remain untouched; preparation
invents no new reference semantics.
Portable preparation tests were run before and after the refactor against direct
TOTAL division, unchanged single-histogram features and an independently fitted
full-SVD PCA. Separate changes to test counts or targets/metadata leave the fitted
development state unchanged. Stage-0 legacy table expectations remain in use.

## Nonclassical point execution and compatibility projection (Stage 3)

```text
prepared representations
          ↓
baseline / ML execution
          ↓
PointEvaluationResult
          ↓
legacy generalization table projection
          ↓
existing summaries / frozen reports
```

Private adapters in `generalization_evaluation` now share this path for principal
A/B, representation A/B, A/C/D/E instrument/acquisition, A/F model mismatch and
the full A–F suite. Public signatures and result dataclasses are unchanged.
Principal A/B still selects the three canonical engineered-feature estimators;
representation A/B still selects their three canonical representations. Frozen
wrappers own those selections and ordering, not the factual executor.

`_build_generalization_evaluation_batch` views already-prepared matrices without
copying or refitting them. It snapshots metadata and uses its `sample_id` column,
or positional row numbers for legacy carriers without that column. A–E expose
their existing targets as `generating_mono`; F exposes its existing primary-component
target as `primary_component`. These kinds also provide distinct reference IDs.
No weighted-mixture or pseudo-true reference is created. Metadata indices do not
define sample identity, and source metadata/targets remain unchanged.

`_evaluate_nonclassical_batch` accepts arbitrary named batches and selected
representations, invokes `predict_regressors`, and combines baseline and ML facts
in one `PointEvaluationResult`. Baseline algorithms are supplied from the unchanged
development-mean and mean-arrival functions. Families are explicit: `baseline`
for those two methods, `ml` for regressors. Constant mean has no canonical
representation (`None`); mean arrival uses `engineered_features`. No family is
inferred from a name. An ML `classical_reconvolution_custom` remains ML and can
coexist with a distinct actual classical method ID. Canonical identity validation
rejects duplicate points or conflicting reuse of one method ID across families.

Custom ML representations affect only selected regressor inputs; unselected
mapping entries are ignored. Baselines retain their original prepared inputs.
Ordinary finite baseline/ML estimates are valid, including zero and negative
values. No clipping occurs. Shape/finiteness checks for ML outputs belong to
`predict_regressors`; selected matrix row counts are also checked against the
batch before execution. A/B now uses that same prediction validation contract.
Generic execution also supports no-reference batches, without inventing targets.

`_project_generalization_predictions` explicitly selects the frozen reference,
preserves metadata column placement and point order, and resets the legacy table
index. It projects `method_id` to `estimator`, absent representation to `"none"`,
and `lifetime_estimate_ns` to `predicted_lifetime_ns`; `test_id` remains the frozen
metadata column. `true_lifetime_ns` retains the legacy target semantics, including
F's primary component. Signed/absolute errors come from canonical comparisons,
but invalid-row errors are masked to NaN in this view only. Finite invalid values
and their errors remain inspectable in canonical facts.

Stage 4 removes prefix inference from internal generalization summaries as described
below. Valid-only errors, attempted/valid counts, reference A and near-zero safeguards
retain their policies. This does not make the whole reporting layer generic.

## Classical point adaptation and explicit-family summaries (Stage 4)

```text
baseline / ML execution ───┐
                          ├─> PointEvaluationResult
classical fit diagnostics ┘             ↓
                           compatibility table projections
                                       ↓
                           existing summaries / frozen reports
```

`_build_classical_point_evaluation` accepts an already prepared `EvaluationBatch`,
existing row-aligned classical diagnostics, and the declared method ID. It supplies
`MethodDescriptor(method_id, "classical", "raw_histogram")`, copies the scalar
`fitted_lifetime_ns` into canonical lifetime facts, and uses **the original
`valid_fit` decision and `failure_reason`**. It neither fits nor recomputes acceptance,
positivity, boundary or optimizer rules. Sample IDs/order and row counts are checked.
Classical batches need no representation matrix; their references remain
`generating_mono` for A–E and `primary_component` for F.

A finite rejected fit retains its lifetime and finite reference errors in canonical
facts. A failed nonfinite fit retains its nonfinite lifetime, invalid state and
failure reason, with unavailable numerical errors. Nothing is clipped or repaired.
Distinct IRF/model variants use the existing distinct method IDs; combining duplicate
point identities is rejected. There is no name-prefix inference in the adapter.

Principal A/B and the A/C/D/E, A/F and A–F classical wrappers now consume this adapter
and the Stage-3 legacy projection. A/B retains its basic prediction schema and original
`ReconvolutionBenchmarkResult` objects. The other three paths retain the four appended
columns, in order: `classical_irf_mode`, `assumed_irf_fwhm_ns`,
`fitted_temporal_shift_ns`, `temporal_shift_error_ns`. As before, failure reasons and
the full `valid_fit`/parameter/optimizer/boundary/NLL/deviance/runtime/initialization
diagnostics live in `per_curve` / `fit_diagnostics`, not additional prediction columns.
Diagnostics remain authoritative and separate; the core does not duplicate them.
Legacy invalid-row error masking, metadata placement and index reset are unchanged.

Per-width IRF grouping, condition-specific versus nominal Test-C fits, temporal-shift
diagnostics, A/F pairing, weighted-mixture descriptive diagnostics and severity
comparisons remain unchanged. No weighted or pseudo-true reference is introduced.
Classical degradation still uses its existing denominator policies, including the
instrument comparison's strictly positive correct-IRF Test-A reference rather than
silently adopting the generic near-zero threshold.

`_summarize_generalization_predictions` shares the unchanged legacy calculation,
but requires an explicit method-family mapping. All migrated internal wrappers
declare families from their execution roles: baseline, regressor, or classical fit.
Consequently a custom ML `classical_reconvolution_custom` now works through the
full-suite wrapper and has no classical failure rate. Reusing an actual baseline
method ID for an ML method in the same result still fails canonical identity checks;
that is a conflicting identity, not a reserved-prefix rule.

The public `summarize_generalization_predictions(predictions)` signature is unchanged.
This standalone table-only compatibility wrapper lacks descriptors and still infers
classical failure-rate applicability from `classical_reconvolution...`. Re-summarizing
an arbitrary custom method using that legacy wrapper can therefore differ from its
explicit-family internal summary. Specialized frozen reports still retain their
canonical method selections; this stage does not generalize those report contracts.

Independent controlled-fit expectations were run before and after migration for
all four paths, including diagnostic tables, mixed/all-failed groups and table order.
Real-fit regressions separately cover inference and full-suite projection. Conditional
evaluation still retains finite-invalid errors and its metadata index. Uncertainty,
persistence, report/result dataclasses, notebooks and root exports are unchanged.

## Conditional point adaptation and presentation policy (Stage 5)

```text
explicit method + reference              ambiguous historical inputs
             ↓                                       ↓
   PointEvaluationResult                  compatibility-only conditional path
             │                                       ↓
     ┌───────┴───────────┐                unchanged historical diagnostic table
     ↓                   ↓
generalization      conditional projection
projection          retain finite invalid errors
mask invalid errors      ↓
     ↓              valid-only grouped summaries
frozen summaries/reports
```

`build_prediction_diagnostics` and `build_ml_prediction_diagnostics` accept the
optional keyword-only pair `method: MethodDescriptor` and `reference:
LifetimeReference`. Supplying just one raises: neither family nor reference kind
can be recovered safely from a historical estimator name or numeric target vector.
With both supplied, the descriptor's method ID determines the displayed
`estimator_name`; its explicit family and representation ID remain in canonical
facts without changing the legacy column schema. This supports explicit ML and
baseline methods, including arbitrary or misleading names, without name dispatch.

The reference must be fully available and exactly equal to the existing target
vector in positional order. Callers declare `generating_mono` for ordinary mono
simulations and `primary_component` when the target is a bi-exponential component;
the adapter does not guess either. The existing `LifetimeReference` carries kind,
identity and any scope rather than introducing another reference vocabulary.
Partial/no-reference evaluation remains supported by the factual core, but is not
the contract of these legacy, fully targeted conditional tables.

For example, given an existing split and regression result:

```python
from tcspc_toolkit.conditional_evaluation import build_ml_prediction_diagnostics
from tcspc_toolkit.evaluation_results import LifetimeReference, MethodDescriptor

diagnostics = build_ml_prediction_diagnostics(
    split=split,
    result=result,
    method=MethodDescriptor("user_regressor", "ml", "engineered_features"),
    reference=LifetimeReference(
        "simulation_target", "generating_mono", split.y_test,
        [True] * len(split.y_test),
    ),
)
```

Omitting both semantic arguments preserves the compatibility-only calculation.
`RegressionBenchmarkResult` has no family or reference-kind fields and is also
used for baseline and mismatch outputs in Notebook 12. Those unchanged callers
therefore do **not** create falsely labelled canonical facts. Name/prefix inference
is not a migration strategy. Migrating callers to explicit semantics remains a
later reviewed task; no notebooks or result constructors changed in Stage 5.

`build_classical_prediction_diagnostics` accepts only an optional `reference`:
the adapter already knows family `classical`, retains its supplied/default method
ID, and declares representation `raw_histogram`. When the reference is supplied,
the original `valid_fit`, lifetime and failure reason enter the same factual
builder. Without it, historical calls remain table-only. Existing `per_curve`
columns, including errors, optimizer/IRF diagnostics, runtime and failure reasons,
remain authoritative and unchanged. Canonical scalar estimates/validity supply
the compatibility aliases; no absent diagnostic columns are manufactured.

The private adapter uses one invocation-local `conditional` evaluation ID and
ordered positional sample IDs. It does not treat a pandas index or arbitrary
metadata label as identity. Metadata stays alongside facts in the projection;
duplicate/non-default indexes, repeated metadata sample labels, row ordering and
existing column positions survive unchanged. These local facts are not a new
cross-batch public identity contract. No representations are fitted or copied.

Finite invalid estimates retain finite canonical errors with
`is_valid_comparison=False`. The conditional projection displays these errors;
the generalization projection masks them. **Error visibility is not metric
eligibility.** Conditional summaries still select only `valid_estimate` rows,
count failures over all attempts, preserve all-invalid NaNs and keep the existing
regime boundaries/order. Signed error remains estimate minus reference; relative
error remains absolute error divided by reference.

The compatibility-only array path still accepts empty inputs and even an explicit
validity mask labelling a nonfinite estimate valid (the grouped summary rejects
unusable selected errors). The explicit factual path instead enforces the core's
nonempty batch and finite-valid-estimate rules. An invalid infinite estimate is
retained canonically with NaN comparison errors; the array-table projection alone
restores historical signed-infinite/absolute-infinite/relative-infinite display.
Classical tables keep their existing NaN errors for nonfinite fits. None of these
presentation policies changes core validation or summary eligibility.

## Implemented factual contracts

### Identity and batches

`evaluation_results.MethodDescriptor(method_id, family, representation_id=None)`
is immutable and deliberately small. Families are `baseline`, `ml`, `classical`,
and `bayesian`, matching persistence vocabulary without depending on persistence.
Names are nonblank strings preserved verbatim. No case folding or prefix inference
occurs: `classical_reconvolution_custom` can be an ML method. Different model or
point-summary variants must receive distinct caller-chosen method IDs. This is an
identity contract within a result collection, not model configuration hashing.
One method ID has one family throughout that collection, including across
representations and evaluation batches.

`EvaluationBatch(evaluation_id, sample_ids, representations=None, metadata=None,
references=None)` holds a nonempty ordered batch; omitted mappings become empty.
IDs are arbitrary nonblank strings;
sample IDs may also be integers (not booleans). Sample IDs must be unique within
the batch; a sample label can recur in another evaluation. No A–F meaning is
inferred from a label. `representation_id=None` means not applicable.

Representations may be empty. When supplied they expose the same `shape[0]` as
the sample count. NumPy arrays, DataFrames and backend-specific matrices pass
through unchanged. Feature content and positional ordering remain caller-owned;
row-count validation cannot detect reordered data. No simulation, transformation,
fitting or train/calibration/test policy is executed here.

Metadata is row-aligned and kept out of result columns. Its pandas index does not
define sample identity. If it contains `sample_id` or `evaluation_id`, those
columns must agree with the explicit identities. Sample IDs are stored as a tuple.
Mapping shells are copied and read-only; matrix values pass through unchanged.

Batch metadata and canonical result tables are private snapshots: constructor
inputs are copied, and the public DataFrame properties return defensive copies.
An ordinary cell edit or row deletion on a returned table cannot invalidate the
stored snapshot. To change facts, construct a new carrier and validate again.
Canonical tables contain validated scalar values. Arbitrary nested user objects
inside metadata cells are not recursively copied by pandas and remain caller-owned.
This is an ownership contract, not a claim that frozen dataclasses make pandas
or all nested user objects deeply immutable. Private backing attributes are internal.

### References

`LifetimeReference(reference_id, kind, values_ns, available, *,
reference_version=None, assumption_label=None)` normalizes array-like inputs to
copied, read-only NumPy arrays. Availability is an explicit boolean vector.
Available lifetimes must be finite and positive. Unavailable numeric placeholders
are ignored when constructing comparisons.
Ordinary writes to `values_ns` and `available` are rejected by NumPy. Deliberately
overriding the write protection or mutating array structure is unsupported.

Kinds are `generating_mono`, `primary_component`, `trusted_experimental` and
`pseudo_true_mono`. The last requires a nonblank caller-owned `assumption_label`
to identify its model-conditional scope; ordinary references may omit the label
or supply a nonblank one. It describes runtime scope, **not a SQLite primary or
foreign key**, and does not replace full scientific provenance. Future adapters
must map and verify the actual domain context. Versions are optional, never invented.
A changed kind/version/assumption label needs a distinct reference ID throughout
a combined result, not just within one evaluation batch.

No provenance dictionary, weighted-mixture truth kind or implied physical truth
is introduced. In particular, the Test-F photon-weighted component lifetime stays
a descriptive mismatch diagnostic, not a generic physical reference.

### Point and comparison tables

`PointEvaluationResult(points, reference_comparisons)` copies and validates two
DataFrames. It contains no metrics, degradation tables or scientific diagnostics.
The exact canonical column order is:

```text
points:
  evaluation_id, sample_id, method_id, method_family, representation_id,
  lifetime_estimate_ns, is_valid, failure_reason

reference_comparisons:
  evaluation_id, sample_id, method_id, representation_id,
  reference_id, reference_kind, reference_version, assumption_label,
  reference_available, reference_value_ns,
  error_ns, absolute_error_ns, relative_error, is_valid_comparison
```

The point key is `(evaluation_id, sample_id, method_id, representation_id)`.
`method_family` is deliberately not part of the key: conflicting reuse of a method
ID is rejected, not interpreted as a different method. Representation distinguishes
variants, but cannot change the method's family. Comparison identity adds
`reference_id`, whose kind/version/scope must agree throughout the result.
Reference values and availability must agree for each evaluation/sample/reference
across methods. Values can differ between observations, as lifetimes naturally do;
they cannot silently depend on which estimator is being compared. Repeated
identities, orphan comparisons or inconsistent computed errors/validity are
rejected. Multiple references add comparison rows, never duplicate point rows.

Boolean masks must actually be boolean, not truthy integers/strings. Numeric
columns normalize to float64. Nullable string labels normalize pandas missing
values to object-column `None`, including absent representation keys. Required
identifiers cannot be missing. Additional diagnostic columns belong elsewhere,
not in these exact schemas. `points` and `reference_comparisons` return editable
copies; reconstruct a result to validate and adopt edits.

**Finite estimate != scientifically valid estimate.** The caller supplies
`is_valid`. A valid estimate must be finite and cannot have a failure reason.
An invalid result may retain any numeric output, including finite values, NaN or
infinity, and an optional failure reason. Zero/negative estimates are not clipped
or treated as automatically invalid: method-specific adapters own that judgment.

Signed error is estimate minus reference; relative error is absolute error divided
by reference. Errors remain visible for finite invalid estimates. Unavailable
references and nonfinite estimates produce NaN errors. Comparison validity also
requires point validity, reference availability and finite calculated errors.

**Absence of reference != zero error.** A no-reference batch produces points and
an empty, schema-preserving comparison table. A partially available named reference
produces an explicit row per sample, with `reference_available=False` and NaN
reference/errors where absent. This retains attempted and available counts.

## Implemented builders and derived primitives

All are module-qualified from `evaluation_core`:

- `build_point_evaluation(*, batch, method, lifetime_estimates_ns, is_valid,
  failure_reasons=None)` builds one method/batch result without execution or scoring.
  Point order is sample order; comparison order is reference order then sample order.
  A representation label may describe a source matrix no longer resident in memory.
- `combine_point_evaluations(results)` concatenates in caller order and revalidates
  identities/facts. It does not sort, deduplicate or overwrite. The sequence is nonempty.
- `calculate_reference_errors(estimates, references, *, reference_available)` returns
  signed, absolute and absolute-relative ndarray errors; it does not apply scientific
  validity as a visibility filter.
- `summarize_point_errors(errors_ns, *, is_valid, reference_available, eligible)`
  requires an explicit contributor mask. It returns `n_attempted`, `n_valid`,
  `n_available`, `n_contributing`, MAE, RMSE, bias, median AE and p90/p95 AE.
  Percentiles use linear interpolation. `eligible` may deliberately include finite
  invalid estimates, but cannot include unavailable references or nonfinite errors.
  Nothing is silently intersected/dropped. All metrics use the selected population;
  an empty selection yields NaN metrics. Rates/coverage are not inferred. Evaluate
  one reference population at a time, not pooled repeated comparisons.
- `calculate_degradation_ratio(*, reference_error, shifted_error,
  minimum_reference_error)` requires an explicit nonnegative finite denominator
  threshold. It returns shifted/reference, or NaN for a denominator at/below that
  threshold, negative/nonfinite errors or a nonfinite quotient. Zero shifted error
  over a usable reference returns zero. It does not select reference populations.

These kernels do not define one historical denominator convention. An
attempted-population metric that propagates a missing value to NaN must remain
an explicit policy in a later wrapper, not silently become finite-only MAE.

The three summary masks have independent meanings: `is_valid` is a scientific or
method-validity fact; `reference_available` is a reference fact; `eligible` is
the caller-selected metric population. Contributors must satisfy the requested
metric's numerical requirements (here, available references and finite errors).
Unusable selected inputs raise. Selecting a finite but scientifically invalid
estimate does not relabel it as valid or change the independent valid count.
The generic degradation function implements only its documented explicit policy;
historical exception/NaN/infinity/threshold variants belong to later compatibility
projections, not extra modes of this primitive.

## Baseline policies that are deliberately not equivalent

| Existing behavior | Preservation requirement for later migration |
|---|---|
| Generalization prediction builder masks errors on invalid rows; conditional builder retains errors for finite invalid rows and preserves the metadata index | Shared raw facts can support both projections; do not silently change either view. |
| Regression metrics include relative error and sklearn R²; robustness metrics include bias and tail percentiles | Share mathematics, retain existing applicability/validation and public result fields. |
| Classical benchmark errors use valid fits; failures use all attempts; runtime uses finite timing records | Do not filter failed rows out of attempted counts or pool timing scopes. |
| Some classical uncertainty scorecard point metrics use finite fitted lifetimes, not `valid_fit` | Keep that explicit eligibility policy; do not equate finiteness with fit acceptance. |
| ML quantile scorecard MAE uses attempted predictions and can be NaN when one prediction is nonfinite; interval metrics use valid intervals | Preserve both denominators. `test_persistence_uncertainty` explicitly checks these stored facts. |
| Standard generalization ratios use a `1e-12` safeguard; scalar helper raises, table helpers return NaN | Keep the legacy policies; the new ratio primitive does not replace them in this checkpoint. |
| Classical instrument comparisons use correct-IRF A as the reference even across method IDs, with a positive-denominator check | Share comparison mechanics later, not automatic same-method/reference selection. |
| Week-7 mismatch ML ratio can be infinity at zero reference MAE; classical counterpart uses NaN | Do not silently unify these legacy outputs. |
| CV summarizes fold metrics with sample SD (`ddof=1`) | Not pooled sample errors or final-test uncertainty; fold-local sklearn cloning stays separate. |
| Test F scores against primary component tau_1; weighted lifetime is descriptive; Bayesian pseudo-true projection is prior-free and assumption-conditional | Three scientific concepts, not interchangeable ground truth. |
| Interval coverage uses valid intervals; quantile pinball/crossing uses finite triplets; score metrics use valid score/estimate pairs | Keep masks and counts separate. Never repair quantile crossings silently. |
| RF tree spread uses population SD, training-bootstrap spread uses sample SD; tail/retention selection has established rounding and stable ordering | Do not relabel either as a nominal interval or change ranking populations. |
| Bayesian study coverage/inclusion distinguishes valid-interval fractions from successful-and-covering fractions over all attempts | Preserve validity, reference kind and denominators; PPC tail probabilities are descriptive, not universal model-validity p-values. |

Independent small legacy expectations in `test_evaluation_legacy_contracts.py`
lock A/B and A/C/B point, summary and degradation tables, sample ordering and the
finite-invalid visibility distinction. They do not call production builders to
generate expected values and are not dependency-version snapshots. Existing
estimator-equivalence tests separately protect canonical fitted predictions.

## Separate scientific outputs and later reviewed work

Classical fit parameters, optimizer facts, IRF assumptions, residuals, posterior
summaries/PPC and scientific provenance remain in existing domain objects.
Intervals, heuristic scores, local covariance and empirical repeated-Poisson
variability remain distinct. No uncertainty attachment class exists at this stage.

Later Issue-#11 stages will review remaining conditional caller migration,
uncertainty integration, method-specific result adapters and frozen report
consolidation. No combined
development/training/final-test owner (`PreparedEvaluationData`) or broad result
containing policy-dependent summaries/degradation is established here. Existing
public constructors and Issue-#2 legacy aliases remain supported through #12.

Persistence is an external consumer: reference/method vocabulary aligns, but
scope/version/assumption labels are not substituted for database keys. Existing
schema, source identities, uncertainty-method restrictions and duplicate policies
are unchanged. Representability will be tested through reviewed adapters later.

Issue #3 owns physical package movement; Issue #12 owns the post-reorganization
clean-environment, reference-workflow and cross-feature integration gate. Neither
is claimed complete by this additive checkpoint.
