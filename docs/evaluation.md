# Evaluation architecture: Issue #11, Stage 0–1

This checkpoint adds module-qualified factual result contracts and small metric
primitives. **No existing benchmark execution path has migrated.** Preparation,
multi-test orchestration, classical/uncertainty integration and notebook migration
remain later reviewed stages. The 66-name root API is unchanged.

## Responsibilities and existing inventory

| Current location | Carriers / entry points | Responsibility and consolidation boundary |
|---|---|---|
| `ml_evaluation` | `BenchmarkMeasurements`, `BenchmarkDataset`, `BenchmarkSplit`, `HistogramRepresentations`; dataset/split/representation builders | Synthetic benchmark inputs, aligned training/test membership, development-only learned representations. Preserve indices, feature definitions, normalization and fitted PCA. |
| `generalization`, `generalization_datasets` | `GeneralizationSuiteDefinition`, protocol/numerics/domain objects, `GeneralizationTestMeasurements`, `GeneralizationTestSuite` | Frozen A–F definitions, seeds, metadata and generated observations. These are not arbitrary evaluation-batch types. |
| `generalization_evaluation` | `GeneralizationABPreparedData`, `GeneralizationPreparedData`; both preparation functions | Duplicate feature/normalization/PCA mechanics. A/B additionally validates identities and paired targets. Learned artifacts stay outside the new factual result core. |
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

Later Issue-#11 stages will review preparation sharing, multi-test execution,
method-specific result adapters and thin frozen report projections. No combined
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
