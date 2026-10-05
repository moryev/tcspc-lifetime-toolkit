# AGENTS.md

## Purpose

This file provides repository-level instructions for Codex and other repository-aware coding agents working on the **TCSPC Lifetime Toolkit**.

Treat it as an operational map, not as a complete project history. For detailed scientific findings, implementation history, and roadmap rationale, read the repository documentation and the relevant GitHub issues.

## Project scope

The TCSPC Lifetime Toolkit is a scientific Python package for simulating, fitting, and evaluating time-correlated single-photon-counting (TCSPC) decay data.

The toolkit combines:

- physical decay models;
- instrument-response-function (IRF) modelling, generalized sampled/source profiles, preparation, and convolution;
- Poisson photon-counting simulation;
- classical lifetime fitting and reconvolution;
- canonical experimental-measurement import and representation;
- preprocessing and physically interpretable feature extraction;
- machine-learning lifetime estimation;
- controlled robustness/generalization benchmarks;
- uncertainty calibration and failure-awareness analysis;
- Bayesian Poisson reconvolution, posterior sampling, posterior summaries, and posterior-predictive diagnostics;
- controlled decay-model and IRF-model mismatch evaluation across classical and Bayesian inference.

The current package version is **v0.7.0**. Weeks 8–9 of the scientific roadmap are complete. Post-v0.7.0 Issues #6, #8, #4, #9, #1, and #2 have been implemented, merged, and closed. Issue #11 (evaluation-architecture consolidation) is now the active integration target, followed by Issue #3 (package restructuring) and Issue #12 (full integration/regression verification).

## Repository map

- `src/tcspc_toolkit/` — package implementation.
- `tests/` — automated scientific and regression tests.
- `notebooks/` — reproducible scientific demonstrations and analyses.
- `docs/scientific_findings.md` — accumulated scientific findings, including the Week 8–9 conclusions.
- `docs/design/master_design_document.md` — current design principles.
- `configs/` — committed configuration files used for reproducible workflows, including the Issue-8 and Issue-4 manifests.
- `scripts/` — focused reproducibility/audit utilities; keep scientific algorithms in the package when they are reusable.
- `data/` — project data area; do not add large or restricted experimental datasets without explicit approval.
- `README.md` — public project overview, installation instructions, and current capabilities.
- `CHANGELOG.md` — release history.
- `CITATION.cff` — citation metadata.

## Environment and canonical commands

The package supports **Python >= 3.11**.

Create and activate a virtual environment, then install the project in editable mode with development and optional Bayesian dependencies for the full test suite:

```bash
python -m pip install -e ".[dev,bayesian]"
```

Run the full test suite with:

```bash
python -m pytest
```

Run a targeted test file during development with:

```bash
python -m pytest tests/test_<relevant_module>.py
```

The project currently does not require Ruff, Black, mypy, or other lint/type-check commands as mandatory validation steps. Those belong to the later Week 12 CI/release work unless an active issue explicitly introduces them earlier.

## Source-of-truth priority

Before changing code:

1. Read this file.
2. Read the active GitHub issue in full.
3. Read Issue #5 for the post-Week-9 development sequence when roadmap context matters.
4. Inspect the relevant source files and tests.
5. Consult `README.md`, `docs/scientific_findings.md`, and `docs/design/master_design_document.md` as needed.

For task-specific requirements, the **active GitHub issue is authoritative**.

Do not silently resolve contradictions between an issue, documentation, tests, and current behavior. Identify the conflict and preserve existing scientific behavior unless the task explicitly requires changing it.

## Current design principles

Preserve the scientific separation expressed in `docs/design/master_design_document.md`, even if files are reorganized later:

- mathematical models are separate from simulation workflows;
- IRFs are generated/represented independently from decay models;
- convolution is an independent operation;
- the shared mono-exponential reconvolution expected-count model lives in `forward_model.py` and is reused by classical fitting, uncertainty, and Bayesian inference rather than reimplemented per estimator;
- noise sampling is independent from deterministic signal generation;
- evaluation functions are organized by the quantity being evaluated, not merely by estimator family;
- public APIs use explicit scientific names;
- convenience wrappers may compose lower-level functions rather than duplicate their logic;
- preprocessing is analysis-dependent: there is no universally correct TCSPC preprocessing pipeline.

Issue #2 completed the public-API stabilization phase. The package now has a curated root API, a generic estimator execution contract independent of canonical benchmark estimator identities, and scientific/domain-oriented public names with legacy compatibility aliases. Issue #11 is now the active phase for deliberate evaluation-architecture consolidation. Issue #3 then performs the physical package/subpackage reorganization. During Issue #11, consolidate evaluation structures and method identity without prematurely performing the Issue-#3 file/package movement.

## Scientific guardrails

### Photon-counting semantics

Raw TCSPC histogram counts are modeled with Poisson statistics where the raw-count assumption applies.

Do not silently apply Poisson likelihoods or Poisson-specific uncertainty formulas to data that have already been normalized, background-subtracted into non-count values, rescaled, or otherwise transformed in ways that invalidate the count model.

Keep raw measurements conceptually distinct from derived representations.

### IRF physics

The measured TCSPC signal is not generally the ideal fluorescence decay. IRF convolution and detector background are core parts of the physical measurement model.

Do not replace reconvolution with a simpler decay fit when the task requires IRF-aware inference.

Keep IRF origin separate from IRF use. Synthetic, imported/measured, estimated-proxy, and deliberately misspecified IRF sources now feed the same downstream preparation/forward-model machinery through explicit source provenance and prepared kernels. Do not collapse source identity, preparation history, and the final kernel into one ambiguous object.

### Ground truth versus experimental reference values

Synthetic datasets can contain known generating parameters such as `lifetime_true_ns`.

Experimental data usually do not have intrinsic ground truth.

For experimental workflows:

- report estimates, diagnostics, and uncertainty without inventing truth;
- compute MAE, bias, coverage, or other truth-based metrics only when a trusted reference value is explicitly supplied;
- distinguish a calibration/reference value from a simulated generating parameter.

### Canonical experimental-measurement boundary

Issue #6 established the first canonical experimental-data path. Preserve these semantics in later work:

- `TCSPCMeasurement` is the canonical carrier for one imported experimental histogram.
- Imported time units are explicit and converted to internal nanoseconds; do not infer units from values or column names.
- `MeasurementDataKind.RAW_COUNTS` and `MeasurementDataKind.PROCESSED_INTENSITY` are scientifically distinct. Raw Poisson-count methods must pass through the raw-count guard rather than infer semantics from dtype alone.
- `SampledIRF` remains the deliberately narrow carrier for an imported sampled/measured source trace.
- `IRFProfile` represents a generalized source profile with explicit `IRFSourceKind`, metadata, and provenance.
- `PreparedIRF` is the derived unit-area kernel on the forward-model grid, with explicit `IRFPreparationDiagnostics`.
- The imported/source IRF trace keeps its own values, metadata, and provenance. Registration, resampling, and normalization create derived objects rather than mutating the source.
- The original Issue-6 same-grid path remains backward compatible. When grids differ, use the explicit Issue-8 preparation path; never resample, realign, baseline-correct, smooth, or infer zero time silently.
- Experimental lifetime estimates are separate from optional reference-based evaluation. A trusted reference is supplied explicitly and is not stored as intrinsic measurement ground truth.
- Experimental ML adapters may transform/predict with already-fitted development artifacts, but must not fit or refit representations/models on the experimental measurement.

### Generalized IRF contract

Issue #8 is complete. Preserve its distinction among source provenance, preparation operations, assumed kernels, and failure-aware proxies:

- Gaussian and EMG synthetic profiles, imported sampled profiles, and leading-edge-derived proxies are scientifically different sources even if they ultimately produce arrays on the same target grid.
- Registration and resampling are explicit operations with diagnostics; support loss and sampling adequacy must remain inspectable.
- A leading-edge proxy is opt-in and approximate. Numerical constructibility or a warning-free proxy does not establish physical correctness.
- A deliberately misspecified IRF must remain identifiable as an assumption used for inference; do not rewrite it as if it were the generating/measured IRF.
- Model-mismatch studies may compare matched and misspecified IRFs on the same observation. Preserve that pairing and provenance.

### Frozen Week-8 robustness protocol

The final A–F suite is a frozen external generalization/robustness benchmark.

Do not use Tests A–F for:

- estimator training;
- representation fitting;
- hyperparameter selection;
- uncertainty training;
- uncertainty calibration;
- conformal calibration.

Preserve the intended roles of development/training data, held-out calibration data, and untouched external A–F evaluation data.

Do not change the frozen A–F definitions merely to improve reported results. Any intentional protocol change must be explicit, justified, tested, and documented.

### Week-9 uncertainty semantics

Do not collapse all uncertainty outputs into one concept.

The repository intentionally distinguishes among:

- nominal prediction/confidence intervals;
- heuristic uncertainty or disagreement scores;
- bootstrap sensitivity;
- local Poisson/Fisher covariance;
- repeated-Poisson empirical sampling variability;
- conformal calibration;
- Bayesian posterior/credible-interval uncertainty conditional on an explicit prior, mono-exponential model, and fixed assumed IRF;
- uncertainty under distribution shift;
- physical model misspecification.

Random-Forest tree spread and training-data bootstrap spread are not automatically nominal prediction intervals.

Classical covariance, parametric Poisson bootstrap, and Bayesian posterior uncertainty all quantify statistical uncertainty conditional on an assumed model. Bayesian credible intervals additionally depend on the stated prior and fixed assumed IRF; they are not automatic model-form uncertainty intervals.

The central Week-9 result is:

> An estimator can correctly quantify statistical uncertainty while remaining confidently wrong because its physical model is incomplete.

Therefore do not interpret narrow intervals, low ensemble spread, good scalar Poisson deviance, or current aggregate residual diagnostics as universal evidence that the physical model is valid.


### Bayesian inference semantics

Issue #4 is complete. Preserve the following contracts established by the final implementation and scientific evaluation:

- Bayesian inference consumes canonical raw-count measurements and the shared mono-exponential reconvolution forward model; do not create a separate Bayesian-only physical forward model.
- Priors are explicit scientific assumptions. Keep prior policy/configuration identifiable in comparisons and persistence.
- The assumed IRF is fixed within a Bayesian inference run. Posterior uncertainty does not automatically propagate uncertainty in IRF shape or provenance.
- Sampling diagnostics assess numerical exploration of the assumed posterior. Successful `emcee` diagnostics do not establish that the decay model or IRF is physically correct.
- Prior/posterior predictive checks are descriptive model-conditional diagnostics. Their tail probabilities must not be presented as universally calibrated model-validity p-values.
- Under decay-model mismatch, a bi-exponential generating process has no unique mono-exponential physical truth. Keep generating component parameters distinct from the deterministic pseudo-true mono-exponential projection.
- A pseudo-true lifetime is a deterministic, prior-free, model-conditional likelihood projection used to separate within-model estimation error from physical model discrepancy. Never label it as the physical "true lifetime".
- The completed Stage-5 matched-model and Stage-6 mismatch scientific records are frozen evidence. Do not silently rewrite those generated outputs when changing later code; use explicit audit/correction artifacts when scientifically necessary.
- The Stage-6.5 corrected pseudo-true references and their documented reanalysis are authoritative for Issue-4 mismatch interpretation.

## Reproducibility rules

- Preserve explicit random seeds where benchmark reproducibility depends on them.
- Preserve the named deterministic random streams used by Issue-4 observation, bootstrap, Bayesian, and posterior-predictive workflows unless an explicit task changes the protocol.
- Do not replace frozen benchmark seeds, regimes, scientific manifests, or saved reference results without an explicit task requirement.
- Keep train/calibration/test roles separate.
- Prefer committed configuration and library code over hidden notebook state.
- Record new scientific assumptions in code/docstrings/tests and, when they affect interpretation, in project documentation.
- Preserve provenance for experimental measurements, IRFs, model configurations, and persisted benchmark results as those features are added.

## Notebook rules

Notebooks are scientific demonstrations, not the primary home for algorithms.

- Reuse committed package code wherever reasonably possible.
- Do not implement substantial scientific algorithms only inside notebook cells.
- If a notebook needs reusable logic, add it to the library with tests first.
- Keep notebooks reproducible from a fresh kernel.
- Use explicit seeds/configuration where stochastic behavior is shown.
- Clearly distinguish published/physical assumptions, synthetic demonstration choices, inferred quantities, calibration-dependent quantities, and limitations.

## Coding and change discipline

Before editing:

- inspect the current implementation and related tests;
- search for existing abstractions before creating a new parallel helper, dataclass, metric, or validation path;
- understand whether the requested change affects public behavior, benchmark reproducibility, or scientific interpretation.

While editing:

- make the smallest coherent change that satisfies the active issue;
- avoid unrelated refactors;
- preserve backwards behavior unless the issue explicitly changes it;
- prefer explicit scientific names over generic aliases;
- preserve array shape/unit conventions and document any new unit conversion;
- add or update tests for meaningful behavior changes;
- test physical/numerical behavior, not only implementation details.

For substantial changes, propose an implementation plan before modifying multiple modules.

## Validation workflow

During implementation:

1. Run the most relevant targeted tests.
2. Add regression tests for new scientific behavior or corrected failures.
3. Run broader affected test groups.
4. Before concluding an issue, run:

```bash
python -m pytest
```

If the full suite cannot be run, say exactly what was run and why the full validation was not completed.

Before finishing, review the diff for:

- accidental public-API changes;
- leakage between training/calibration/frozen evaluation data;
- altered benchmark seeds/configurations;
- changed numerical/scientific behavior outside the issue scope;
- duplicated library logic inside notebooks;
- undocumented assumptions.

## Git and issue workflow

For substantive implementation work, prefer a dedicated feature branch and a focused pull request rather than mixing unrelated changes.

Repository-aware coding agents should normally stop after implementation, validation, and final diff review. Do not commit, push, merge, delete branches, or close GitHub issues unless the user explicitly requests that action. The default workflow is for the user to review and perform Git history operations manually.

The PR/commit description should summarize:

- what changed;
- why it changed;
- relevant scientific assumptions;
- tests run;
- any remaining limitations or follow-up work.

Do not close an issue solely because code was written; confirm its acceptance criteria and validation requirements.

## Post-Week-9 roadmap

Issue #5 is the orchestration issue and should be consulted for the full rationale.

Current integration status:

1. **#10** — repository/Codex transition — **complete**;
2. **#6** — experimental TCSPC data ingestion, processing, and evaluation — **complete and closed**;
3. **#8** — generalized IRF models, sampled-IRF preparation, and leading-edge/failure-awareness evaluation — **complete and closed**;
4. **#4** — Bayesian Poisson inference — **complete and closed**, including matched-model calibration, model-mismatch evaluation, numerical audits, documentation, and Notebook 17;
5. **#9** — SQLite persistence for experiments and benchmark results — **complete, merged, and closed**;
6. **#1** — standardize array input type annotations — **complete, merged, and closed**;
7. **#2** — API stabilization and package hardening — **complete, merged, and closed**;
8. **#11** — consolidate Week 7–9 evaluation architecture — **current active integration target**;
9. **#3** — reorganize `tcspc_toolkit` into coherent subpackages;
10. **#12** — full integration and regression verification;
11. **#13** — Week 10 Purcell-enhanced TCSPC sensing demonstration;
12. **#14** — Week 11 documentation and user experience;
13. **#15** — Week 12 continuous integration and release.

The completed Issue-#1 typing pass established the current array-input convention: public APIs that normalize suitable inputs through `np.asarray(...)` advertise `ArrayLike`, while APIs that genuinely require NumPy-array-specific behavior retain `NDArray[...]`. Do not reopen that distinction opportunistically during later refactors.

Issue #2 established the current API-stability boundary:

- `tcspc_toolkit.__all__` is a curated high-level convenience contract; specialized advanced APIs may remain supported through module-qualified imports.
- The generic estimator extension layer is `estimator_api.py`: `RegressorProtocol`, `EstimatorSpec`, `fit_regressors()`, and `predict_regressors()`.
- Generic estimator execution must not know canonical estimator identities. Canonical Ridge, Random-Forest, and HistGradientBoosting specifications are supplied separately by `ml_models.py`.
- Caller-supplied representations remain caller-owned matrix-like inputs with positional row alignment; the generic execution layer does not fit representations or infer sample identity.
- Public roadmap-specific Day/Week names replaced during Issue #2 now have scientific/domain-oriented canonical names. The historical names remain compatibility aliases through at least Issue #12; frozen protocol strings, persisted identities, seeds, and study provenance remain unchanged.
- Do not broaden the root export surface opportunistically during #11/#3; preserve the reviewed public import contract unless an active issue explicitly changes it.

Preserve the completed generalized-IRF, Bayesian, persistence, experimental-measurement, uncertainty, API-stability, and frozen-benchmark scientific contracts during the remaining architectural work.

## Codex working protocol

For each substantial issue:

1. Read this `AGENTS.md`.
2. Read the active issue and Issue #5.
3. Inspect relevant code, tests, and documentation.
4. Summarize the current behavior and constraints.
5. Propose an implementation plan before broad/multi-file changes.
6. Implement incrementally.
7. Run targeted tests throughout.
8. Run the full relevant/full repository test suite before completion.
9. Review the final diff against the issue scope and scientific guardrails.
10. Summarize changes, tests, scientific implications, and remaining limitations.
11. Stop with a tested, reviewed, uncommitted diff unless the user explicitly requests Git history operations.

## Current integration orientation

Issues #6, #8, #4, #9, #1, and #2 are complete. Do not reopen or redesign their established scientific or public-API contracts merely because later architectural work consumes their outputs.

Issue #9 established schema-v1 persistence, scientific recording adapters, read-only query/DataFrame helpers, and an end-to-end SQL aggregation example. Continue to preserve these persistence boundaries:

- keep persistence optional and separate from numerical computation;
- preserve the distinction between measurements, generating truth, trusted experimental references, assumed models/IRFs, and pseudo-true projections;
- preserve estimator/method identity, configuration, random seeds, validity states, and provenance needed to reproduce a stored result;
- store scientifically queryable scalar fields relationally where useful;
- do not hide all benchmark variables in opaque JSON solely for convenience;
- do not store large histograms, posterior chains, or model binaries as SQLite BLOBs by default;
- use parameterized SQL, foreign-key enforcement, schema-version metadata, and explicit transaction/duplicate semantics;
- do not let evaluation consolidation change established persistence semantics merely for naming or table-unification convenience.

Issue #1 established truthful public array-input contracts without changing numerical behavior. Preserve concrete NumPy-array return and stored-field contracts where they remain appropriate.

Issue #2 established the public API and estimator-extension boundary:

- preserve the curated root API and reviewed `__all__` contract;
- preserve the generic estimator execution layer as independent from canonical benchmark configuration;
- preserve user-supplied estimator/representation support without estimator-name dispatch or a scikit-learn inheritance/cloning requirement;
- preserve scientific canonical names and the legacy Day/Week compatibility aliases introduced for existing callers;
- preserve frozen benchmark defaults, stored method identities, and protocol strings despite Python-level naming cleanup.

Issue #11 is now the active architectural task. Its purpose is to consolidate the Week-7–9 evaluation architecture while preserving scientific distinctions and the Issue-#2 public contracts.

For Issue #11:

- inventory the current A/B, A/C/D/E, A/F, and A–F evaluation entry points, result carriers, prediction builders, summary/degradation builders, and specialized report layers before changing them;
- reduce duplicated prediction, summary, degradation, and comparison machinery where scientifically equivalent;
- separate generic evaluation infrastructure from frozen benchmark configuration without changing the frozen A–F definitions or development/calibration/evaluation roles;
- integrate classical, ML, interval uncertainty, heuristic uncertainty scores, Bayesian outputs, and experimental/reference evaluation without conflating their distinct semantics;
- replace estimator-name conventions used to infer method family or diagnostics with explicit method/result metadata where appropriate;
- preserve the standalone generic estimator API from Issue #2; do not reintroduce hard-coded estimator identities into generic execution;
- preserve current public names and compatibility aliases unless consolidation requires a deliberate, reviewed compatibility change;
- do not physically reorganize modules into the Issue-#3 target package structure during this issue;
- do not alter physical models, likelihoods, IRF semantics, uncertainty definitions, persistence schema v1, random seeds, or saved scientific reference artifacts merely to simplify evaluation code.

Known Issue-#11 technical debt inherited from Issue #2 includes:

- separate `GeneralizationABPreparedData` and `GeneralizationPreparedData` carriers;
- separate principal-A/B, representation-A/B, instrument/acquisition, model-mismatch, and full-suite result/report paths;
- duplicated nonclassical/classical prediction-table and degradation/comparison helpers;
- reporting logic that currently reserves `constant_mean`, `mean_arrival_time`, and the `classical_reconvolution...` prefix to infer method semantics;
- canonical estimator/representation/test assumptions embedded in some specialized reports even though the lower-level estimator execution boundary is now generic.

Consult Issues #11, #5, #3, and #12 together when deciding architectural boundaries, but implement only the active issue.
