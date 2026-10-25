# Package architecture and post-Issue-12 import cleanup

Issues #3 and #12 are complete, merged, and closed. The separate breaking cleanup
removes the 38 flat scientific compatibility modules retained through Issue #12.
Use the curated 66-name root API or canonical hierarchy. The five reviewed public
package APIs and supported Day/Week symbol aliases remain unchanged.

Current first-party Python consumers, all 17 notebooks' active code, examples,
CLI paths, and recommended documentation imports use supported paths. Negative
tests prove the removed modules are absent; fixed historical deny-lists continue
to guard imports even after their source files disappear. Notebook files and
scientific references are unchanged.

The Issue-3 inventory, dependency graph and checkpoint results below are explicitly
historical. Issue-12 installation, scientific and integration evidence remains in
the [acceptance ledger](issue12_scientific_acceptance.md#stage-5-final-technical-acceptance).

## Canonical tree and ownership

```text
tcspc_toolkit/
├── physics/          __init__.py, models.py, convolution.py, forward_model.py
├── irf/              __init__.py, profiles.py, preparation.py, estimation.py, selection.py
├── measurement/      __init__.py, records.py, io.py, adapters.py
├── simulation/       __init__.py, histograms.py, datasets.py
├── preprocessing/    __init__.py, histograms.py, features.py, representations.py
├── estimation/       __init__.py, api.py, baselines.py, fitting.py, classical.py
│   └── bayesian/     __init__.py, posterior.py, sampling.py, predictive.py
├── ml/               __init__.py, models.py, evaluation.py, cross_validation.py
├── evaluation/       __init__.py, results.py, core.py, attachments.py,
│                     diagnostics.py, conditional.py, plots.py
├── uncertainty/      __init__.py, evaluation.py, ml.py, classical.py
├── benchmarks/       __init__.py, generalization_protocol.py, generalization_datasets.py,
│                     generalization.py, mismatch.py, timing.py, irf.py,
│                     bayesian_calibration.py, bayesian_mismatch.py, uncertainty.py
├── persistence/      __init__.py, sqlite.py
└── __init__.py, __main__.py, cli.py, config.py, exceptions.py
```

Reusable physical, measurement, estimator, factual-evaluation and uncertainty
machinery lives outside `benchmarks`. Study-level orchestration is in
`benchmarks`: frozen Week-8 generalization, the separate Week-7 mono/bi mismatch
study, timing, Issue-8 IRF studies, Issue-4 matched Bayesian calibration and
Bayesian mismatch, and frozen Week-9 uncertainty reporting. The existing
conformal workflow remains inside `benchmarks.uncertainty`; it was not extracted.
`estimation.bayesian.*` owns inference, not those study protocols.

## Current production inventory

There are **62 production Python files**: 45 canonical implementation files,
five retained public package initializers (`irf`, `preprocessing`, `simulation`,
`evaluation`, `persistence`), and 12 root/support files (five root files and seven
docstring-only namespace initializers). No obsolete flat facade remains.

Issue #28 PR 2 adds `estimation/multiexponential.py` for constrained
bi-exponential Poisson fitting. It reuses the shared physical forward model and
Poisson diagnostics and adds no initializer or root exports. The historical
Issue-3 inventory below remains unchanged.

The ordered root API is exactly 66 names, with direct canonical object identity.
Supported symbol aliases, canonical serialization and retained package API lookup
remain covered. Removed flat module imports and their historical pickle globals
are covered by explicit negative contracts.

## Historical Issue-3 complete production inventory

The following 99-file inventory records Issue #3 Stage 10 on
`refactor/issue-3-package-structure`, following Stage-9 commit
`0d8f77381dd03b7044a9b431442e13fb5fb9bd9c`. It is preserved as relocation history,
not the current file list. Every Python file at that checkpoint was assigned once:

- **A — canonical implementation:** 44 files in 11 package families.
- **B — historical compatibility facade:** 43 files (38 root files and five package initializers).
- **C — root/support:** 12 files (five genuine root support files and seven docstring-only namespace initializers).
- **D — unexpected substantial root implementation:** zero.

There were **99 production Python files**, compared with 48 at the pre-move
Stage-0 checkpoint `93efa73a`. The increase of 51 is accounted for by the 43
retained facades, seven namespace-only initializers, and the canonical shared
IRF-selection module established in the earlier reviewed stages. It does not
represent copied scientific implementations. All substantive root scientific
implementations have moved into subpackages.

Paths are relative to `src/tcspc_toolkit/`; target names omit `tcspc_toolkit.`.
Line counts are informational snapshots from Issue #3; they include documentation
and imports and are not current size or performance acceptance thresholds.
For **every B row**, the count of scientific/executable body statements after
excluding documentation, direct imports and `__all__` declarations is **zero**.
There are no functions, classes, wrappers, SQL definitions or numerical bodies
in those facades.

| File | Class | Lines | Direct canonical target for B |
| --- | --- | ---: | --- |
| `__init__.py` | C | 190 | — |
| `__main__.py` | C | 5 | — |
| `baselines.py` | B | 17 | `estimation.baselines` |
| `bayesian.py` | B | 38 | `estimation.bayesian.posterior` |
| `bayesian_evaluation.py` | B | 49 | `benchmarks.bayesian_calibration` |
| `bayesian_mismatch_evaluation.py` | B | 53 | `benchmarks.bayesian_mismatch` |
| `bayesian_predictive.py` | B | 32 | `estimation.bayesian.predictive` |
| `bayesian_sampling.py` | B | 38 | `estimation.bayesian.sampling` |
| `benchmark_plots.py` | B | 19 | `evaluation.plots` |
| `benchmarks/__init__.py` | C | 1 | — |
| `benchmarks/bayesian_calibration.py` | A | 742 | — |
| `benchmarks/bayesian_mismatch.py` | A | 1050 | — |
| `benchmarks/generalization.py` | A | 6188 | — |
| `benchmarks/generalization_datasets.py` | A | 1059 | — |
| `benchmarks/generalization_protocol.py` | A | 958 | — |
| `benchmarks/irf.py` | A | 1044 | — |
| `benchmarks/mismatch.py` | A | 683 | — |
| `benchmarks/timing.py` | A | 577 | — |
| `benchmarks/uncertainty.py` | A | 2787 | — |
| `classical_evaluation.py` | B | 21 | `estimation.classical` |
| `classical_uncertainty.py` | B | 43 | `uncertainty.classical` |
| `cli.py` | C | 257 | — |
| `conditional_evaluation.py` | B | 25 | `evaluation.conditional` |
| `config.py` | C | 157 | — |
| `convolution.py` | B | 9 | `physics.convolution` |
| `cross_validation.py` | B | 30 | `ml.cross_validation` |
| `datasets.py` | B | 17 | `simulation.datasets` |
| `estimation/__init__.py` | C | 4 | — |
| `estimation/api.py` | A | 255 | — |
| `estimation/baselines.py` | A | 197 | — |
| `estimation/bayesian/__init__.py` | C | 1 | — |
| `estimation/bayesian/posterior.py` | A | 418 | — |
| `estimation/bayesian/predictive.py` | A | 498 | — |
| `estimation/bayesian/sampling.py` | A | 734 | — |
| `estimation/classical.py` | A | 903 | — |
| `estimation/fitting.py` | A | 674 | — |
| `estimator_api.py` | B | 13 | `estimation.api` |
| `evaluation/__init__.py` | B | 25 | `evaluation.diagnostics` |
| `evaluation/attachments.py` | A | 138 | — |
| `evaluation/conditional.py` | A | 845 | — |
| `evaluation/core.py` | A | 173 | — |
| `evaluation/diagnostics.py` | A | 269 | — |
| `evaluation/plots.py` | A | 531 | — |
| `evaluation/results.py` | A | 329 | — |
| `evaluation_core.py` | B | 14 | `evaluation.core` |
| `evaluation_results.py` | B | 13 | `evaluation.results` |
| `evaluation_uncertainty.py` | B | 11 | `evaluation.attachments` |
| `exceptions.py` | C | 17 | — |
| `experimental.py` | B | 21 | `measurement.adapters` |
| `features.py` | B | 26 | `preprocessing.features` |
| `fitting.py` | B | 17 | `estimation.fitting` |
| `forward_model.py` | B | 9 | `physics.forward_model` |
| `generalization.py` | B | 35 | `benchmarks.generalization_protocol` |
| `generalization_datasets.py` | B | 31 | `benchmarks.generalization_datasets` |
| `generalization_evaluation.py` | B | 129 | `benchmarks.generalization` |
| `irf/__init__.py` | B | 27 | `irf.profiles` |
| `irf/estimation.py` | A | 331 | — |
| `irf/preparation.py` | A | 261 | — |
| `irf/profiles.py` | A | 433 | — |
| `irf/selection.py` | A | 50 | — |
| `irf_estimation.py` | B | 21 | `irf.estimation` |
| `irf_evaluation.py` | B | 43 | `benchmarks.irf` |
| `irf_preparation.py` | B | 23 | `irf.preparation` |
| `measurement/__init__.py` | C | 1 | — |
| `measurement/adapters.py` | A | 165 | — |
| `measurement/io.py` | A | 195 | — |
| `measurement/records.py` | A | 197 | — |
| `measurement_io.py` | B | 15 | `measurement.io` |
| `measurements.py` | B | 25 | `measurement.records` |
| `mismatch_evaluation.py` | B | 19 | `benchmarks.mismatch` |
| `ml/__init__.py` | C | 1 | — |
| `ml/cross_validation.py` | A | 608 | — |
| `ml/evaluation.py` | A | 1438 | — |
| `ml/models.py` | A | 337 | — |
| `ml_evaluation.py` | B | 50 | `ml.evaluation` |
| `ml_models.py` | B | 29 | `ml.models` |
| `ml_uncertainty.py` | B | 75 | `uncertainty.ml` |
| `models.py` | B | 13 | `physics.models` |
| `persistence/__init__.py` | B | 135 | `persistence.sqlite` |
| `persistence/sqlite.py` | A | 5559 | — |
| `physics/__init__.py` | C | 4 | — |
| `physics/convolution.py` | A | 115 | — |
| `physics/forward_model.py` | A | 60 | — |
| `physics/models.py` | A | 180 | — |
| `preprocessing/__init__.py` | B | 32 | `preprocessing.histograms` |
| `preprocessing/features.py` | A | 775 | — |
| `preprocessing/histograms.py` | A | 710 | — |
| `preprocessing/representations.py` | A | 438 | — |
| `representations.py` | B | 20 | `preprocessing.representations` |
| `simulation/__init__.py` | B | 21 | `simulation.histograms` |
| `simulation/datasets.py` | A | 608 | — |
| `simulation/histograms.py` | A | 637 | — |
| `timing_evaluation.py` | B | 23 | `benchmarks.timing` |
| `uncertainty/__init__.py` | C | 1 | — |
| `uncertainty/classical.py` | A | 2082 | — |
| `uncertainty/evaluation.py` | A | 660 | — |
| `uncertainty/ml.py` | A | 3728 | — |
| `uncertainty_evaluation.py` | B | 45 | `uncertainty.evaluation` |
| `uncertainty_robustness.py` | B | 66 | `benchmarks.uncertainty` |

## API and initializer contracts

At the Issue-3 checkpoint, the ordered root `__all__` contained 66 names. Root bindings and all
524 facade-export bindings (including repeated names across historical paths)
resolved directly to canonical objects. Historical import/pickle lookup,
public constants/typing aliases, and Day/Week aliases were retained through
Issue #12. The separate post-Issue-12 cleanup removes the 38 flat module paths
while preserving supported symbols and the reviewed package APIs.

Docstring-only namespaces are `physics`, `measurement`, `estimation`,
`estimation.bayesian`, `ml`, `uncertainty`, and `benchmarks`.
Intentional historical package facades are `irf` (profiles), `preprocessing`
(histogram operations), `simulation` (histograms), `evaluation` (seven diagnostics
functions only), and `persistence` (the 62-name SQLite public API).
None is an aggregate convenience API for its whole subtree.

## Historical Issue-3 dependency audit

Arrows below mean **imports/depends on**, not data flow. The source/AST audit
uses the parent-aware graph in `tests/test_package_structure.py`; cross-package
imports include reached initializers, but do not re-run an already-active parent.

- 99 modules; **242 direct eager internal edges**, **380 parent-aware eager edges**.
- All 99 strongly connected components are singletons; **zero eager cycles**.
- **20 distinct deferred internal edges** at 56 import sites: the Bayesian
  posterior-to-sampling dispatch, plus 55 persistence-local scientific imports
  covering 19 target modules. Both pandas imports in persistence are also local.
- **Three typing-only edges**, all from `uncertainty.ml` to the canonical
  generalization protocol/data/execution stack.
- **Zero canonical/support implementation imports of historical facades**.
- **Zero scientific execution imports of persistence**.

The conceptual layering is not a rigid rank for every source file. Deliberate
existing relationships remain:

- `physics.forward_model → irf.profiles` reuses the shared IRF shift operation.
- `measurement.adapters` composes fitting, features and diagnostics; measurement
  records and CSV I/O remain their lower-level boundaries.
- `estimation.classical → evaluation.diagnostics` shares fit diagnostics;
  diagnostics consumes fitting, not the classical batch execution module.
- `ml.evaluation → evaluation.core/results` builds generic factual results.
  `evaluation.conditional` consumes classical/ML results for reporting; factual
  `evaluation.results/core/attachments` remain estimator-independent.
- `uncertainty.evaluation → benchmarks.generalization_protocol` preserves the
  frozen provenance/leakage guard. It is configuration, not benchmark execution.
- `uncertainty.ml` has no eager/deferred benchmark execution imports; its three
  benchmark edges remain under `TYPE_CHECKING`.
- The posterior's sampling dispatch and persistence scientific adapters remain
  function-local. The historical persistence package facade eagerly re-exports
  only `persistence.sqlite`.

There are no `physics → benchmarks`, `simulation → uncertainty`,
`estimation.api → ml`, or factual-result-to-estimator imports.

### Complete parent-aware eager adjacency list

Names omit `tcspc_toolkit.`; `<root>` denotes the root package.
`[]` means no internal eager edge. Facade-to-canonical edges are intentional.

```text
<root> -> [config, estimation, estimation.api, estimation.fitting, exceptions, irf, irf.estimation, irf.preparation, irf.profiles, measurement, measurement.adapters, measurement.io, measurement.records, physics, physics.convolution, physics.models, preprocessing, preprocessing.features, preprocessing.histograms, preprocessing.representations, simulation, simulation.datasets, simulation.histograms]
__main__ -> [cli]
baselines -> [estimation, estimation.baselines]
bayesian -> [estimation, estimation.bayesian, estimation.bayesian.posterior]
bayesian_evaluation -> [benchmarks, benchmarks.bayesian_calibration]
bayesian_mismatch_evaluation -> [benchmarks, benchmarks.bayesian_mismatch]
bayesian_predictive -> [estimation, estimation.bayesian, estimation.bayesian.predictive]
bayesian_sampling -> [estimation, estimation.bayesian, estimation.bayesian.sampling]
benchmark_plots -> [evaluation, evaluation.plots]
benchmarks -> []
benchmarks.bayesian_calibration -> [estimation, estimation.bayesian, estimation.bayesian.posterior, estimation.bayesian.sampling, estimation.classical, irf, irf.preparation, measurement, measurement.records, physics, physics.forward_model, simulation, simulation.histograms, uncertainty, uncertainty.classical]
benchmarks.bayesian_mismatch -> [benchmarks.bayesian_calibration, benchmarks.irf, estimation, estimation.bayesian, estimation.bayesian.posterior, estimation.bayesian.predictive, estimation.bayesian.sampling, estimation.classical, estimation.fitting, irf, irf.preparation, irf.profiles, measurement, measurement.records, physics, physics.forward_model, simulation, simulation.histograms, uncertainty, uncertainty.classical]
benchmarks.generalization -> [benchmarks.generalization_datasets, benchmarks.generalization_protocol, config, estimation, estimation.api, estimation.baselines, estimation.classical, evaluation, evaluation.core, evaluation.results, irf, irf.profiles, ml, ml.evaluation, ml.models, preprocessing, preprocessing.features, preprocessing.representations]
benchmarks.generalization_datasets -> [benchmarks.generalization_protocol, simulation, simulation.histograms]
benchmarks.generalization_protocol -> []
benchmarks.irf -> [config, estimation, estimation.classical, evaluation, evaluation.diagnostics, irf, irf.estimation, irf.preparation, irf.profiles, measurement, measurement.adapters, measurement.records, ml, ml.evaluation, ml.models, physics, physics.models, preprocessing, preprocessing.features, preprocessing.histograms, simulation, simulation.histograms, uncertainty, uncertainty.classical, uncertainty.ml]
benchmarks.mismatch -> [config, estimation, estimation.classical, evaluation, evaluation.results, ml, ml.evaluation, ml.models, preprocessing, preprocessing.features, simulation, simulation.histograms]
benchmarks.timing -> [estimation, estimation.baselines, estimation.classical, ml, ml.evaluation, ml.models]
benchmarks.uncertainty -> [benchmarks.generalization, benchmarks.generalization_datasets, benchmarks.generalization_protocol, estimation, estimation.classical, evaluation, evaluation.attachments, evaluation.core, evaluation.diagnostics, evaluation.results, irf, irf.profiles, ml, ml.evaluation, uncertainty, uncertainty.classical, uncertainty.evaluation, uncertainty.ml]
classical_evaluation -> [estimation, estimation.classical]
classical_uncertainty -> [uncertainty, uncertainty.classical]
cli -> [estimation, estimation.fitting, measurement, measurement.io, measurement.records, simulation, simulation.histograms]
conditional_evaluation -> [evaluation, evaluation.conditional]
config -> []
convolution -> [physics, physics.convolution]
cross_validation -> [ml, ml.cross_validation]
datasets -> [simulation, simulation.datasets]
estimation -> []
estimation.api -> []
estimation.baselines -> []
estimation.bayesian -> []
estimation.bayesian.posterior -> [estimation.fitting, exceptions, irf, irf.preparation, irf.profiles, irf.selection, measurement, measurement.records, physics, physics.forward_model]
estimation.bayesian.predictive -> [estimation.bayesian.posterior, estimation.bayesian.sampling, evaluation, evaluation.diagnostics, physics, physics.forward_model, simulation, simulation.histograms]
estimation.bayesian.sampling -> [estimation.bayesian.posterior, irf, irf.preparation, irf.profiles, measurement, measurement.records, physics, physics.forward_model]
estimation.classical -> [estimation.fitting, evaluation, evaluation.diagnostics, physics, physics.convolution, physics.models, preprocessing, preprocessing.histograms]
estimation.fitting -> [physics, physics.forward_model, physics.models]
estimator_api -> [estimation, estimation.api]
evaluation -> [evaluation.diagnostics]
evaluation_core -> [evaluation, evaluation.core]
evaluation_results -> [evaluation, evaluation.results]
evaluation_uncertainty -> [evaluation, evaluation.attachments]
evaluation.attachments -> [evaluation.results]
evaluation.conditional -> [estimation, estimation.classical, evaluation.core, evaluation.diagnostics, evaluation.results, ml, ml.evaluation]
evaluation.core -> [evaluation.results]
evaluation.diagnostics -> [estimation, estimation.fitting, physics, physics.models]
evaluation.plots -> []
evaluation.results -> []
exceptions -> []
experimental -> [measurement, measurement.adapters]
features -> [preprocessing, preprocessing.features]
fitting -> [estimation, estimation.fitting]
forward_model -> [physics, physics.forward_model]
generalization -> [benchmarks, benchmarks.generalization_protocol]
generalization_datasets -> [benchmarks, benchmarks.generalization_datasets]
generalization_evaluation -> [benchmarks, benchmarks.generalization]
irf -> [irf.profiles]
irf_estimation -> [irf, irf.estimation]
irf_evaluation -> [benchmarks, benchmarks.irf]
irf_preparation -> [irf, irf.preparation]
irf.estimation -> [irf.preparation, irf.profiles, measurement, measurement.records, preprocessing, preprocessing.histograms]
irf.preparation -> [irf.profiles, measurement, measurement.records, preprocessing, preprocessing.histograms]
irf.profiles -> []
irf.selection -> [exceptions, irf.preparation, irf.profiles, measurement, measurement.records, preprocessing, preprocessing.histograms]
measurement -> []
measurement_io -> [measurement, measurement.io]
measurement.adapters -> [config, estimation, estimation.classical, evaluation, evaluation.diagnostics, exceptions, irf, irf.preparation, irf.selection, measurement.records, preprocessing, preprocessing.features, preprocessing.histograms]
measurement.io -> [exceptions, measurement.records]
measurement.records -> [exceptions, irf, irf.profiles, preprocessing, preprocessing.histograms]
measurements -> [measurement, measurement.records]
mismatch_evaluation -> [benchmarks, benchmarks.mismatch]
ml -> []
ml_evaluation -> [ml, ml.evaluation]
ml_models -> [ml, ml.models]
ml_uncertainty -> [uncertainty, uncertainty.ml]
ml.cross_validation -> [ml.evaluation]
ml.evaluation -> [config, estimation, estimation.baselines, evaluation, evaluation.core, evaluation.results, ml.models, preprocessing, preprocessing.features, preprocessing.representations, simulation, simulation.histograms]
ml.models -> [config, estimation, estimation.api, preprocessing, preprocessing.representations]
models -> [physics, physics.models]
persistence -> [persistence.sqlite]
persistence.sqlite -> []
physics -> []
physics.convolution -> []
physics.forward_model -> [irf, irf.profiles, physics.convolution, physics.models]
physics.models -> []
preprocessing -> [preprocessing.histograms]
preprocessing.features -> [config, exceptions, preprocessing.histograms]
preprocessing.histograms -> [config, exceptions]
preprocessing.representations -> [config, preprocessing.histograms]
representations -> [preprocessing, preprocessing.representations]
simulation -> [simulation.histograms]
simulation.datasets -> [physics, physics.models]
simulation.histograms -> [irf, irf.preparation, irf.profiles, physics, physics.convolution, physics.models]
timing_evaluation -> [benchmarks, benchmarks.timing]
uncertainty -> []
uncertainty_evaluation -> [uncertainty, uncertainty.evaluation]
uncertainty_robustness -> [benchmarks, benchmarks.uncertainty]
uncertainty.classical -> [estimation, estimation.classical, estimation.fitting, evaluation, evaluation.attachments, evaluation.results, irf, irf.profiles, physics, physics.forward_model, simulation, simulation.histograms, uncertainty.evaluation]
uncertainty.evaluation -> [benchmarks, benchmarks.generalization_protocol]
uncertainty.ml -> [evaluation, evaluation.attachments, evaluation.core, evaluation.results, ml, ml.evaluation, ml.models, uncertainty.evaluation]
```

### Deferred and typing-only adjacency lists

These are direct source edges, without eager parent expansion.

```text
deferred:
  estimation.bayesian.posterior -> estimation.bayesian.sampling
  persistence.sqlite -> benchmarks.bayesian_calibration
  persistence.sqlite -> benchmarks.bayesian_mismatch
  persistence.sqlite -> benchmarks.generalization
  persistence.sqlite -> benchmarks.generalization_datasets
  persistence.sqlite -> benchmarks.uncertainty
  persistence.sqlite -> estimation.bayesian.posterior
  persistence.sqlite -> estimation.bayesian.predictive
  persistence.sqlite -> estimation.bayesian.sampling
  persistence.sqlite -> estimation.classical
  persistence.sqlite -> estimation.fitting
  persistence.sqlite -> irf.estimation
  persistence.sqlite -> irf.preparation
  persistence.sqlite -> irf.profiles
  persistence.sqlite -> measurement.records
  persistence.sqlite -> ml.evaluation
  persistence.sqlite -> preprocessing.histograms
  persistence.sqlite -> uncertainty.classical
  persistence.sqlite -> uncertainty.evaluation
  persistence.sqlite -> uncertainty.ml
typing:
  uncertainty.ml -> benchmarks.generalization
  uncertainty.ml -> benchmarks.generalization_datasets
  uncertainty.ml -> benchmarks.generalization_protocol
```

## Historical Issue-3 repository-wide reconciliation

At that checkpoint, historical imports were classified by use. Issue-12 Stage 1
subsequently migrated ordinary consumers and active notebook code, as noted above:

| Area | Classification and action |
| --- | --- |
| Tests | A: deliberate historical API/compatibility regression. Public historical calls remain; implementation-private access and patch seams use canonical modules. |
| Developer scripts | C: nine stale imports corrected in two scripts, with import-only normalized-text/AST equivalence. The other two scripts already use canonical study imports. |
| Examples | B: supported user-facing API. Both examples are retained; `persistence_roundtrip.py` deliberately uses the public persistence facade and supported scientific paths. |
| Documentation | B for historical import examples and migration history. Current architecture/status guidance is reconciled; historical scientific evidence is not rewritten. |
| Notebooks | D: harmless historical tutorial imports, preserved verbatim without execution or output/metadata changes. Exact notebook import inventory follows. |

The five substitutions in `audit_issue4_stage5_classical_fits.py` are
`classical_evaluation → estimation.classical`, `fitting → estimation.fitting`,
`forward_model → physics.forward_model`, `irf → irf.profiles`, and
`irf_preparation → irf.preparation`. This also restores the script's private
`_poisson_coordinate_descent_check` import at its actual implementation.

The four substitutions in `run_issue4_bayesian_benchmarks.py` are
`bayesian → estimation.bayesian.posterior`,
`bayesian_sampling → estimation.bayesian.sampling`, `irf → irf.profiles`,
and `irf_preparation → irf.preparation`. Script definitions are import-checked;
no benchmark, correction, saved reference, or scientific output is regenerated.

The monkeypatch ownership audit found no stale implementation-owned facade
target. `test_persistence_example.py` intentionally patches the public
`persistence.connect_database` binding called by the example; that consumer seam
is preserved. Canonical `irf.estimation.savgol_filter` is not a patch of the
historical `irf` facade.

The initial Stage-10 audit found that `correction_source_sha256` still hashed
`bayesian_mismatch_evaluation.py`, then a compatibility facade. A subsequently
approved Stage-10 correction changed only that lookup to the canonical scientific
implementation, `src/tcspc_toolkit/benchmarks/bayesian_mismatch.py`. This is the
current provenance source, protected by a structural regression test. No saved
correction artifact was regenerated, and no persisted schema-v1 identity changed.

### Historical Issue-3 notebook import inventory

These paths describe the Issue-3 checkpoint, before Issue-12 Stage 1B migrated
active code-cell imports. The current 17 notebooks have zero active imports of
the 38 removable flat facades. The `tcspc_toolkit.` prefix is omitted below.
At the earlier checkpoint, all 17 notebooks were read without execution, including
source-only reads of 949 cells omitted by the capped initial previews; those later cells contained
no additional package imports.

| Notebook | Historical imports at the Issue-3 checkpoint |
| --- | --- |
| `01_tcspc_simulation.ipynb` | `simulation` |
| `02_classical_lifetime_fitting.ipynb` | `evaluation`, `fitting` |
| `03_synthetic_dataset_analysis.ipynb` | `datasets` |
| `04_first_ml_baseline.ipynb` | `datasets` |
| `05_data_leakage_and_grouped_evaluation.ipynb` | `models`, `simulation` |
| `06_grouped_dataset_api_workflow.ipynb` | `datasets` |
| `07_irf_convolution_and_realistic_simulation.ipynb` | `convolution`, `irf`, `models`, `simulation` |
| `08_naive_vs_reconvolution_fitting.ipynb` | `convolution`, `fitting`, `irf`, `models`, `simulation` |
| `09_poisson_reconvolution_fitting_and_validation.ipynb` | `convolution`, `evaluation`, `fitting`, `irf`, `models`, `simulation` |
| `10_preprocessing_tcspc_histograms.ipynb` | `convolution`, `fitting`, `irf`, `models`, `preprocessing`, `simulation` |
| `11_feature_engineering.ipynb` | `convolution`, `features`, `irf`, `models`, `simulation`, `representations` |
| `12_ml_benchmarking.ipynb` | `benchmark_plots`, `classical_evaluation`, `conditional_evaluation`, `irf`, `mismatch_evaluation`, `ml_evaluation`, `ml_models`, `timing_evaluation`, `representations` |
| `13_generalization_and_robustness.ipynb` | `cross_validation`, `generalization`, `generalization_datasets`, `generalization_evaluation`, `irf`, `ml_evaluation`, `ml_models` |
| `14_uncertainty_and_failure_awareness.ipynb` | `classical_uncertainty`, `generalization`, `generalization_datasets`, `generalization_evaluation`, `ml_uncertainty`, `uncertainty_evaluation`, `uncertainty_robustness` |
| `15_experimental_tcspc_workflow.ipynb` | `experimental`, `features`, `measurement_io`, `ml_models`, `simulation` |
| `16_generalized_irf_and_failure_awareness.ipynb` | `experimental`, `irf`, `irf_preparation`, `irf_estimation`, `irf_evaluation`, `measurement_io`, `models`, `simulation` |
| `17_bayesian_poisson_inference_and_model_mismatch.ipynb` | `bayesian`, `bayesian_predictive`, `bayesian_sampling`, `forward_model`, `irf`, `irf_preparation`, `measurements`, `simulation` |

## Scientific and storage boundaries

Stage 10 changes **no production source**, numerical expectation, manifest,
seed, RNG policy, or notebook. The existing regression contracts preserve:

- raw photon counts versus processed intensities and the raw-Poisson guard;
- source IRF, prepared kernel, fixed assumed inference IRF and estimated proxy;
- `generating_mono`, `primary_component`, `trusted_experimental` and
  `pseudo_true_mono` reference identities;
- Bayesian intervals conditional on model/prior/fixed IRF and deterministic,
  prior-free, model-conditional pseudo-true likelihood projections;
- nominal intervals versus heuristic scores, RF spread, ML bootstrap,
  classical covariance/bootstrap, empirical repeated-Poisson variability,
  conformal calibration, and Bayesian credible intervals;
- development training/calibration versus untouched external A–F observations,
  frozen condition ordering/seeds, and Week-7 mismatch versus Week-8 Test F.

Persistence remains schema **v1**, fingerprint
`1edaa036c1e9ffe00927e87d7d14aad89e995c11a0a68172f6cc68842e833cf9`.
All seven literal historical stored identities remain verbatim; see the
[complete persistence inventory](persistence.md#package-layout-and-compatibility-issue-3-stage-9).
Canonical executable imports do not authorize replacing those strings.

Fresh-process guards preserve blocked-`emcee` behavior, deferred persistence
pandas/scientific imports, explicit plotting/study imports, and all initializer
boundaries. Root initialization still has its inherited sklearn/pandas cost;
Issue #3 does not claim or implement root-import optimization.

## Historical Issue-3 validation and acceptance boundary

The committed baseline passed 188 structural/API tests and all 270 persistence
tests. Five additional global closure guards cover the complete inventory,
facade/root identity and historical pickle lookup, metadata authenticity,
developer-script imports, and monkeypatch ownership. Full repository regression
passed all 77 test files using the configured Python 3.12.0 project interpreter
in these bounded groups:

| Group | Files | Passed | Warnings |
| --- | ---: | ---: | ---: |
| Structure, public API, aliases, estimator API, CLI/config/examples | 7 | 247 | 1 |
| Physics, simulation, preprocessing and representations | 12 | 359 | 0 |
| Measurement and IRF | 6 | 108 | 1 |
| Factual evaluation, attachments, diagnostics and reports | 10 | 157 | 3 |
| Generalization protocol/data/preparation/execution | 5 | 132 | 1 |
| Frozen generalization studies and synthesis | 7 | 54 | 1 |
| ML, classical estimators, mismatch and timing | 9 | 183 | 1 |
| Uncertainty and robustness/conformal/residual reports | 8 | 104 | 1 |
| Bayesian inference, predictive checks and studies | 6 | 106 | 0 |
| Complete persistence suite | 7 | 270 | 0 |
| **Total** | **77** | **1,720** | **9** |

There were **zero failures and zero skips**. The nine warnings are the known
seven Windows joblib physical-core lookup fallbacks and two finite-invalid
mismatch subtraction warnings; no new warning category appeared. Scientific
expected values were not changed. `git diff --check` passes.

The closure diff contains documentation, five structural tests, and nine
import-only substitutions in two scripts. Production source, notebooks,
examples, configurations, and saved scientific outputs are unchanged.

Issue #3's physical architecture requires no further package migration.
At Issue #3 closure, clean-install matrices, wheel/sdist acceptance,
scientific-reference reruns, cross-feature fresh-environment workflows and
performance acceptance remained for Issue #12. Its subsequent evidence is in the
[acceptance ledger](issue12_scientific_acceptance.md#stage-5-final-technical-acceptance).
Later release qualification remains separate.

<a id="post-issue-12-compatibility-cleanup-handoff"></a>

## Post-Issue-12 compatibility removal record

All **38 obsolete flat scientific modules** have been removed in a separate
breaking cleanup after Issue #12. Their absence is intentional: old imports fail
with `ModuleNotFoundError`. No runtime aliases or legacy unpickler replace them.
The five public package APIs are retained, including `tcspc_toolkit.persistence`.

The fixed manifest in the structure tests and the historical acceptance blocker
retain all 38 names independently of source files. Compatibility tests now check
actual absence and rejected historical pickle globals while retaining canonical
object identity, serialization, report fields, aliases and architectural boundaries.

Ordinary historical Python pickles whose globals name a removed module may no
longer load. Canonical pickle roundtrips and supported package-global lookups
remain available. Schema v1 and its seven stored historical identity strings are
unchanged: those strings are opaque provenance labels, not dynamic import paths.
No database migration, scientific reference regeneration or algorithm change is
part of this cleanup.

### Removed flat module mapping

Use a curated root export where documented; otherwise use the canonical module
below. Supported symbol names, including Day/Week aliases, are unchanged.

| Removed file under `src/tcspc_toolkit/` | Supported canonical module |
| --- | --- |
| `baselines.py` | `tcspc_toolkit.estimation.baselines` |
| `bayesian.py` | `tcspc_toolkit.estimation.bayesian.posterior` |
| `bayesian_evaluation.py` | `tcspc_toolkit.benchmarks.bayesian_calibration` |
| `bayesian_mismatch_evaluation.py` | `tcspc_toolkit.benchmarks.bayesian_mismatch` |
| `bayesian_predictive.py` | `tcspc_toolkit.estimation.bayesian.predictive` |
| `bayesian_sampling.py` | `tcspc_toolkit.estimation.bayesian.sampling` |
| `benchmark_plots.py` | `tcspc_toolkit.evaluation.plots` |
| `classical_evaluation.py` | `tcspc_toolkit.estimation.classical` |
| `classical_uncertainty.py` | `tcspc_toolkit.uncertainty.classical` |
| `conditional_evaluation.py` | `tcspc_toolkit.evaluation.conditional` |
| `convolution.py` | `tcspc_toolkit.physics.convolution` |
| `cross_validation.py` | `tcspc_toolkit.ml.cross_validation` |
| `datasets.py` | `tcspc_toolkit.simulation.datasets` |
| `estimator_api.py` | `tcspc_toolkit.estimation.api` |
| `evaluation_core.py` | `tcspc_toolkit.evaluation.core` |
| `evaluation_results.py` | `tcspc_toolkit.evaluation.results` |
| `evaluation_uncertainty.py` | `tcspc_toolkit.evaluation.attachments` |
| `experimental.py` | `tcspc_toolkit.measurement.adapters` |
| `features.py` | `tcspc_toolkit.preprocessing.features` |
| `fitting.py` | `tcspc_toolkit.estimation.fitting` |
| `forward_model.py` | `tcspc_toolkit.physics.forward_model` |
| `generalization.py` | `tcspc_toolkit.benchmarks.generalization_protocol` |
| `generalization_datasets.py` | `tcspc_toolkit.benchmarks.generalization_datasets` |
| `generalization_evaluation.py` | `tcspc_toolkit.benchmarks.generalization` |
| `irf_estimation.py` | `tcspc_toolkit.irf.estimation` |
| `irf_evaluation.py` | `tcspc_toolkit.benchmarks.irf` |
| `irf_preparation.py` | `tcspc_toolkit.irf.preparation` |
| `measurement_io.py` | `tcspc_toolkit.measurement.io` |
| `measurements.py` | `tcspc_toolkit.measurement.records` |
| `mismatch_evaluation.py` | `tcspc_toolkit.benchmarks.mismatch` |
| `ml_evaluation.py` | `tcspc_toolkit.ml.evaluation` |
| `ml_models.py` | `tcspc_toolkit.ml.models` |
| `ml_uncertainty.py` | `tcspc_toolkit.uncertainty.ml` |
| `models.py` | `tcspc_toolkit.physics.models` |
| `representations.py` | `tcspc_toolkit.preprocessing.representations` |
| `timing_evaluation.py` | `tcspc_toolkit.benchmarks.timing` |
| `uncertainty_evaluation.py` | `tcspc_toolkit.uncertainty.evaluation` |
| `uncertainty_robustness.py` | `tcspc_toolkit.benchmarks.uncertainty` |
