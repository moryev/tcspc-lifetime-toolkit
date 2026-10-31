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

The latest published GitHub release is **v0.8.0** (October 22, 2026; PyPI deferred). v0.7.0 is the preceding historical release. Do not change the published tag or release metadata merely because development continues; further version bumps and releases require separate maintainer authorization. Weeks 8–9 of the scientific roadmap are complete. Post-v0.7.0 Issues #6, #8, #4, #9, #1, #2, #11, and #3 have been implemented, merged, and closed. Issue #12 (full integration/regression verification) is complete, merged, and closed. The [acceptance ledger](docs/issue12_scientific_acceptance.md#stage-5-final-technical-acceptance) records the evidence, validation status and limitations. The separate post-Issue-12 cleanup removed the 38 obsolete flat scientific module paths.

**Issue #14 (Week 11 documentation/user experience) is complete, merged as [PR #23](https://github.com/moryev/tcspc-lifetime-toolkit/pull/23), and closed.** Its short README, task-oriented documentation, eight executable user examples, and quick-start figure are the accepted user-facing baseline. **Issue #15 CI and release qualification completed, with the merged-main reference qualification in [run #39717483724](https://github.com/moryev/tcspc-lifetime-toolkit/actions/runs/39717483724); it is closed and v0.8.0 was published on GitHub on October 22, 2026.** **#13 Purcell-enhanced TCSPC sensing is deferred / experiment-dependent** and is not a release blocker. Resume #13 only when there are sufficiently concrete experiment-specific or photonic-simulation inputs to make the application scientifically grounded. Do not invent an arbitrary synthetic sensing response merely to satisfy the old calendar ordering. Do not repeat completed scientific reference workflows or reopen package restructuring without a concrete defect and explicit task scope.

**Post-v0.8.0 research status (October 30, 2026):** Issue #28 PRs #29/#31/#34 and Issue #33 PR #35 are merged. General fixed-K deterministic reconvolution is available, but production Poisson reconvolution fitting remains mono- or bi-exponential only. The original #28 independent-audit criterion failed at 119/121; the corrected #33 Stage 3 strict criterion failed at 99/100. [Issue #33](https://github.com/moryev/tcspc-lifetime-toolkit/issues/33) is **open and deferred** for future recovery-capacity/qualification work. [Issue #37](https://github.com/moryev/tcspc-lifetime-toolkit/issues/37) defines the reviewed three-stage K=3 roadmap. The maintainer authorized PR 1's private fixed-K numerical foundations only; implementation is awaiting review, with no public K=3 fitter or new scientific qualification.

## Repository map

- `src/tcspc_toolkit/` — canonical implementations in `physics`, `irf`, `measurement`, `simulation`, `preprocessing`, `estimation`, `ml`, `evaluation`, `uncertainty`, `benchmarks`, and `persistence`; only five genuine support files remain at package root.
- `docs/package_structure.md` — final module inventory, facade mappings, dependency graph, and Issue-#3 closure boundaries.
- `tests/` — automated scientific and regression tests.
- `notebooks/` — reproducible scientific demonstrations and analyses.
- `examples/` — eight supported, user-facing executable workflows, including the reproducible Poisson reconvolution quick start; the full Bayesian example is not a routine test-suite smoke test.
- `docs/README.md`, `docs/user_guide.md`, `docs/cli.md`, `docs/results.md`, and `docs/reproducibility.md` — Issue-#14 navigation, task guidance, qualified scientific summaries, and execution prerequisites.
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

Issue #15 introduces Ruff 0.17.1 (`E4,E7,E9,F`) and the Ruff formatter with an
unchanged-debt hash baseline. Run `python -m ruff check src tests examples scripts`
and `python scripts/check_format.py`. Never refresh baseline hashes to bypass
formatting failures; new/modified Python files must be formatted.
See [release readiness](docs/release_readiness.md) for coverage and artifact commands.
The Ubuntu 24.04 Python 3.11/3.12 matrix, 80% combined branch-coverage floor,
artifact installs, and pinned Windows Week-7 reference regeneration passed in
[GitHub run #39717483724](https://github.com/moryev/tcspc-lifetime-toolkit/actions/runs/39717483724).
Coverage was 82.58%; both Linux suites passed 1,927 tests each (with one
intentionally skipped reference-only check). Do not claim this qualifies every
Python/platform combination or establishes experimental validity. Future tags,
releases, or distribution publications still need separate maintainer approval.

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
- the shared mono-exponential reconvolution expected-count model lives in `physics.forward_model` and is reused by classical fitting, uncertainty, and Bayesian inference rather than reimplemented per estimator;
- Issue #28 PR 1 added general deterministic K-component reconvolution expected-count/component functions that compose the unchanged mono model; this is a forward model, **not** an arbitrary-K optimizer;
- Issue #28 PR 2 added the constrained bi-exponential Poisson fitter in `estimation.multiexponential`, reusing those primitives with grouped numerical and descriptive identifiability diagnostics; PR 3 added raw-measurement integration and independent qualification with a preserved failed audit target;
- Issue #33 PR #35 added objective/gradient safeguards and exhaustive seed-first scheduling, with an independently documented remaining recovery-capacity limitation; do not describe the bi-exponential fitter as fully scientifically qualified;
- noise sampling is independent from deterministic signal generation;
- evaluation functions are organized by the quantity being evaluated, not merely by estimator family;
- public APIs use explicit scientific names;
- convenience wrappers may compose lower-level functions rather than duplicate their logic;
- preprocessing is analysis-dependent: there is no universally correct TCSPC preprocessing pipeline.

Issue #2 completed the public-API stabilization phase. The package now has a curated root API, a generic estimator execution contract independent of canonical benchmark estimator identities, and scientific/domain-oriented public names with legacy compatibility aliases. Issue #11 completed the deliberate evaluation-architecture consolidation: generic point/reference facts, explicit method identity, separate interval/score uncertainty attachments, and shared robustness/reporting orchestration now sit beneath specialized scientific diagnostics and compatibility views. Issue #3 has completed physical package/subpackage reorganization. Preserve the clarified Issue-#11 responsibilities and semantics during subsequent development; do not reopen scientific or metric-policy decisions merely to obtain a cleaner directory tree.

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
8. **#11** — consolidate Week 7–9 evaluation architecture — **complete, merged, and closed**;
9. **#3** — reorganize `tcspc_toolkit` into coherent subpackages — **complete, merged, and closed**;
10. **#12** — full integration and regression verification — **complete, merged, and closed**; see the acceptance ledger;
11. **#14** — Week 11 documentation and user experience — **complete, merged in PR #23, and closed**;
12. **#15** — Week 12 continuous integration and release preparation — **complete and closed; v0.8.0 published**;
13. **#13** — Week 10 Purcell-enhanced TCSPC sensing demonstration — **deferred / experiment-dependent; not a release blocker**;
14. **#28** — general K-component forward reconvolution and bi-exponential Poisson fitting — **PRs #29/#31/#34 merged; Issue #28 remains open and the failed historical 119/121 audit target is preserved**.
15. **#33** — bi-exponential optimizer safeguards and seed-first scheduling — **PR #35 merged; strict new Stage 3 agreement failed at 99/100 against 100/100. Issue #33 is open but deferred for future recovery-capacity/qualification work**.
16. **#37 — fixed-K multi-exponential Poisson fitting, beginning with K=3** — reviewed roadmap; PR 1 private foundations authorized and awaiting implementation review. PR 2 optimizer and PR 3 integration/qualification remain separate review gates; this is not a continuation of the old #28 PR 3 qualification.

The completed Issue-#1 typing pass established the current array-input convention: public APIs that normalize suitable inputs through `np.asarray(...)` advertise `ArrayLike`, while APIs that genuinely require NumPy-array-specific behavior retain `NDArray[...]`. Do not reopen that distinction opportunistically during later refactors.

Issue #2 established the current API-stability boundary:

- `tcspc_toolkit.__all__` is a curated high-level convenience contract; specialized advanced APIs may remain supported through module-qualified imports.
- The canonical generic estimator extension layer is `estimation.api` (the obsolete flat `estimator_api` module has been removed): `RegressorProtocol`, `EstimatorSpec`, `fit_regressors()`, and `predict_regressors()`.
- Generic estimator execution must not know canonical estimator identities. Canonical Ridge, Random-Forest, and HistGradientBoosting specifications are supplied separately by `ml.models`.
- Caller-supplied representations remain caller-owned matrix-like inputs with positional row alignment; the generic execution layer does not fit representations or infer sample identity.
- Public roadmap-specific Day/Week names replaced during Issue #2 now have scientific/domain-oriented canonical names. The historical symbol names remain direct aliases in their canonical modules; frozen protocol strings, persisted identities, seeds, and study provenance remain unchanged.
- Do not broaden the root export surface opportunistically for future K>=3 fitting; preserve the reviewed public import contract unless a new active issue explicitly changes it.

Preserve the completed generalized-IRF, Bayesian, persistence, experimental-measurement, uncertainty, API-stability, and frozen-benchmark scientific contracts in future K>=3 work. Do not restart Issue #12-style integration work unless a concrete regression is found.

## Completed v0.8.0 release qualification (historical)

Issue #15 is closed. Its Ubuntu Python 3.11/3.12 tests, Ruff/formatting, 80% combined
statement/branch-coverage floor, artifact installation checks, and separately
qualified pinned Windows Week-7 reference were accepted for v0.8.0. The GitHub
Release was published on October 22, 2026; PyPI is deferred. Preserve the
release-readiness gates, deterministic reference evidence, citation metadata,
and reviewed import/SQLite contracts. Do not re-run expensive reference
regeneration by default or alter existing release artifacts. No future version,
tag, or release publication is authorized merely by passing CI.

## Issue #28 — staged multi-exponential reconvolution

**PR 1 is complete and merged** as [PR #29](https://github.com/moryev/tcspc-lifetime-toolkit/pull/29)
(squash commit `f888a515ef3a15753762a4a91e4790e015205b58`). The new
module-qualified APIs in `physics.forward_model` provide background-free
component counts and general-K total expectations. The new
`evaluation.diagnostics` APIs provide stable Poisson deviance and a structured
raw-count statistics record (full likelihood, centered NLL, signed residuals,
AIC, and explicitly bin-based BIC). They retain the historical mono model,
existing reduced NLL/residual behavior, curated root API, frozen benchmark
results, and schema v1. PR 1 qualification reported 2,123 passing local tests,
1 intentional skip, and 82.73% combined coverage; consult GitHub Actions for
current CI state. Read [docs/api.md](docs/api.md) for the new scientific contract.

**PR 2 is complete and merged** as [PR #31](https://github.com/moryev/tcspc-lifetime-toolkit/pull/31)
(merge commit `96e88f16c7fcb032dc2c80b6f7e58c436569c530`). It adds the
module-qualified constrained bi-exponential Poisson fitter in
`estimation.multiexponential`, with explicit finite positive lifetime bounds,
finite shift bounds, PreparedIRF or normalized-array inputs, canonical component
ordering, separate numerical failure reporting, and descriptive identifiability
diagnostics. Reuse the unchanged PR-1 forward model and centered objective;
preserve positive-shift convention, IRF preparation/provenance, bin-width scaling,
finite-window truncation, pre-exponential amplitude semantics, and one background.
The root 66-name API, historical mono fitting, frozen references, schema v1,
and release metadata remain unchanged.

PR 2's predeclared development comparison selected the **exhaustive** deterministic
multi-start default with **46 starts** and **56,000 optimization objective calls**;
initialization, validation, and diagnostics have separate hard work limits.
The original stream-2 held-out evaluation was locked before use. After a narrow
initialization-overflow correction, an October 25 *post-correction regression
replay* matched all 20 historical observations exactly in every non-timing
field; it is not a fresh, untouched held-out evaluation. Preserve the original
source-hash locks, configurations and qualification evidence. The merged PR
reported 2,224 passing tests, one intentional skip, 83.40% combined coverage;
[post-merge CI](https://github.com/moryev/tcspc-lifetime-toolkit/actions/runs/40182315771)
passed. Read [docs/api.md](docs/api.md) and
[the PR-2 numerical record](docs/issue28_pr2_numerics.md) before extension.

Numerical convergence, local Jacobian rank, near-optimal-start agreement, and
warning-free diagnostics do **not** certify physical identifiability or accurate
component recovery. The reported warning-free 10.18% secondary-lifetime error
on one noisy observation is a retained negative result, not a threshold-tuning
opportunity. Do not alter the frozen PR-2 optimization policy or warning
thresholds based on later qualification observations.

**PR 3 is complete and merged as PR #34.** The historical study contains 1,550
observations, 1,850 paired outcomes and 205 audits; the designated independent
agreement target failed at 119/121. Preserve its 66-file lock, raw evidence,
saved observations and scientific conclusions. The following integration and
qualification principles describe that completed scope, not authorization to
repeat the study.
Read [Issue #28](https://github.com/moryev/tcspc-lifetime-toolkit/issues/28)
and audit existing `measurement.adapters`, `irf.selection`,
`TCSPCMeasurement.require_raw_counts()`, `PreparedIRF`, the mono adapter,
and PR-2 fit/result contracts before proposing changes. Design an explicit
canonical raw-measurement bi-exponential adapter that resolves and preserves
the assumed IRF and its provenance without silently preprocessing observations
or altering either fitter. On the *same* histogram, grid/window, and assumed
IRF, permit descriptive paired mono/bi Poisson statistics with coherent
likelihood and nominal-parameter conventions; do not use an ordinary chi-square
likelihood-ratio p-value for the singular mono/bi boundary.

Predeclare the **independent repeated-Poisson qualification** (conditions,
seeds, grids, replicate counts, denominators, fixed numerical settings,
metrics, and audit procedure) before running it. Preserve Issue #28's fixed
512-bin high-count baseline and qualification targets (at least 95% numerical
acceptance, median component-lifetime relative error below 5%, relative bias
below 2%), and report every miss without relaxing thresholds or filtering
unfavorable cases. Include lower counts, weak/zero components, close/equal
lifetimes, short windows, background, shifts, IRF-resolution limits, Gaussian,
EMG, sampled and explicitly approximate proxy IRFs, plus matched/misspecified
IRF comparisons. Distinguish generating truth, pseudo-true mono projections,
and optional trusted experimental references. Assess runtime and plan the
separately invoked study around the measured approximately 13-second default
fit; keep CI deterministic and lightweight. Never tune fitter defaults or
warnings on the qualification data, overwrite frozen Week-7/8/9 or PR-2 evidence,
or present warning absence as a calibrated recovery guarantee.

The original Issue #28 scope deferred Bayesian/ML/CLI extensions, arbitrary-K
fitting, automatic model-order selection, component intervals, persistence schema
changes, benchmark regeneration, new release publication, and Issue #13. A new,
separately scoped K>=3 fitting issue may supersede the **arbitrary-K fitting
exclusion only for its explicitly approved scope**; the historical #28 evidence
and scientific contracts remain unchanged. Use `Refs #28` for remaining #28
work; implementation alone does not authorize issue closure.
A separate maintainer approval is required for further numerical changes,
scientific execution, releases or Git history operations beyond an explicit task.

**Issue #33 — merged and deferred:** [PR #35](https://github.com/moryev/tcspc-lifetime-toolkit/pull/35)
is merged; Stage 1 is preserved at `3bff378f`, Stage 2/2B at
`5be88edda17ad6612a8447dca0e11a6543502b85`, and Stage 3 at `55530dfc`. Nonfinite objectives
and failed derivatives cannot fabricate stationary success; finite candidates
are retained, and exhaustive scheduling completes required seeds first. The
46-start and evaluation limits, physical model and Poisson semantics remain.
All fourteen saved development regressions passed. They are not untouched
qualification observations.

The new [Stage 3 qualification](docs/issue33_stage3_qualification_v1.md) completed
200 independent observations, 210 production fits and 133 complete audits under
its unchanged execution lock. Primary numerical acceptance passed at 99/100,
as did both component recovery and bias targets. Strict independent agreement
failed at 99/100 against 100/100. Four fits were numerically rejected; new C00
r050 completed all 46 seeds, detected further descent and correctly reported
required recovery unavailable. Optional continuation omission is distinct from
incomplete required work. Agreement never proves global optimality or physical
identifiability. The robustness improvement was published through merged PR #35,
**not** fully qualified. [Issue #33](https://github.com/moryev/tcspc-lifetime-toolkit/issues/33)
remains **open but deferred** for future investigation of r050 and three true-mono
recovery-capacity failures; it is **not a blocker** for separately planned K>=3
fitting. Do not rerun or tune on qualification outcomes without separate
authorization. Preserve both historical failed audit criteria (119/121 and
99/100). Future work should use `Refs #33`, not `Closes #33`, until actually resolved.

## Issue #37 — fixed-K multi-exponential fitting

- **Current numerical baseline:** `physics.forward_model` already computes general K-component reconvolved expectations with a shared shift and one post-convolution background. Production inference remains mono-exponential or **bi-exponential**; `fit_biexponential_reconvolution` is explicitly K=2. Neither general-K Poisson fitting nor K=3 fitting nor automatic model-order selection is implemented.
- **Authorized current milestone — PR 1 only:** isolated private records/coordinates, counted objective infrastructure, safeguarded derivatives, canonicalization, candidate validation and structural diagnostics, with deterministic tests. Read [the numerical contract and PR-2 handoff](docs/design/issue37_fixed_k_foundations.md). No public fitter, expensive experiment, commit, push or PR creation is authorized by this milestone.
- **Preferred initial scientific scope:** fixed **K=3** (eight nominal parameters: three nonnegative amplitudes, three positive bounded lifetimes, one nonnegative background and one bounded shared shift); design reusable internals for later fixed K>3 without promising arbitrary-K production support now. Preserve exact zero components, equal lifetimes, input/returned component ordering semantics and explicit singularities.
- **Scientific requirements:** constrain and scale parameters, design deterministic yet computationally bounded starts, retain feasible finite best candidates and correctly charged evaluations, distinguish optimizer convergence from identifiability and physical correctness, and treat component permutations, weak components, close lifetimes, IRF-resolution limits and nonidentifiability explicitly. No automatic model-order choice or invalid ordinary chi-square LRT across singular component boundaries.
- **Compatibility first:** preserve existing mono/bi APIs and their qualified/frozen numerical behavior, historical PR-1/2/3 and #33 evidence, IRF preparation/provenance, raw-count Poisson guard, exact-zero/background and shared-shift conventions, 66-name root API and SQLite schema v1. Do not rewrite the bi-exponential fitter merely to generalize it without an approved design and regression evidence.
- **Independent review stages:** (1) authorized private foundations and numerical contracts; (2) separately authorized focused K=3 fitter, measured policy selection and independent optimizer development benchmarks; (3) separately authorized measurement integration and frozen scientific qualification. Compact/balanced/exhaustive policies, TRF/Powell audit strategy/work limits and a proposed 100/100 audit target remain development candidates, not frozen production decisions.
- [Issue #33](https://github.com/moryev/tcspc-lifetime-toolkit/issues/33) and Issue #13 remain deferred; don't conflate their outstanding work with the proposed K=3 project or assume that general-K deterministic forward modelling implies a qualified general-K optimizer.

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

Issues #6, #8, #4, #9, #1, #2, #11, #3, #12, #14, and #15 are complete and closed. Do not reopen or redesign their established scientific, evaluation, package-structure, public-API, or integration contracts in future multi-exponential development unless a concrete defect is discovered and the active issue explicitly requires it.

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

Issue #11 completed the Week-7–9 evaluation-architecture consolidation. Preserve its established boundaries during integration verification:

- generic factual evaluation uses explicit method identity, optional lifetime references, canonical point results, and separate interval/score uncertainty attachments;
- frozen A–F and Week-9 scientific configuration remains separate from generic evaluation machinery;
- classical fit diagnostics, Bayesian model/prior/IRF context, experimental provenance, repeated-Poisson empirical variability, and other method-specific scientific information remain specialized rather than being forced into universal result objects;
- conditional finite-invalid presentation, mismatch/degradation policies, and other documented metric-population differences are intentional compatibility/scientific boundaries;
- legacy table-only compatibility paths may retain historical naming behavior, but new internal architecture must not infer scientific method family from estimator-name prefixes.

Issue #3 package restructuring is complete, merged, and closed after Stage 10. All 44 substantive
implementation files are inside coherent subpackages. The post-Issue-12 cleanup
removed the 38 obsolete flat scientific modules. Five reviewed public package
initializers, five genuine root support files and seven minimal namespace
initializers remain: 61 production Python files in total.

That completed restructuring inventory is historical. Issue #28 PR 2 added one
canonical implementation; PR 3 added two more. Issue #37 PR 1 adds three private
foundation modules. The current inventory is 50 substantive implementations
and 67 production Python files, with unchanged
initializer APIs.

The final Issue-#3 regression baseline on merged `main` was **1,721 passed, 0 failed, 0 skipped**, with two known finite-invalid mismatch warnings. This is a historical pre-Issue-#12 baseline. The Issue-#14 local validation reported **1,877 passed** with two established mismatch warnings and one Windows CPU-detection fallback warning; these are recorded local results, not proof of CI portability.

For future work:

- Use canonical submodule imports inside implementations and developer scripts.
- First-party tests, examples, notebook code cells, and current documentation must use the curated root API or canonical submodules. Do not recreate the 38 removed flat modules. Keep their fixed historical deny-list in the removal/readiness guards; do not infer it from surviving files. Reviewed package-initializer APIs remain supported. Issue-12 Stage 1B notebook import migration was source-only.
- Preserve supported direct Day/Week symbol aliases in canonical modules. Removed flat paths and ordinary pickles naming their globals are intentionally unsupported. Do not add legacy loaders, `sys.modules` aliases, dynamic import shims, or fake `__module__` without a separate design decision.
- Keep the exact ordered 66-name root API. Do not broaden package initializers into convenience APIs.
- Keep `benchmarks`, `uncertainty`, `measurement`, `physics`, `ml`, `estimation`, and `estimation.bayesian` initializers docstring-only. Historical `irf`, `preprocessing`, `simulation`, `evaluation`, and `persistence` initializers re-export only their reviewed surfaces.
- Preserve numerical/scientific behavior, frozen A–F and Week-9 protocols, random seeds, schema v1, and historical stored identities. Canonical imports do not authorize rewriting persisted Python-path strings.
- Preserve the uncertainty provenance dependency on frozen generalization protocol configuration; ML uncertainty's other benchmark relationships remain typing-only.
- Keep scientific execution independent of persistence; persistence adapters retain deferred scientific imports.
- Do not extract conformal workflows or repartition settled modules merely for directory symmetry.
- Keep Issue #12 separate: clean-install/package acceptance, reference-workflow reruns, cross-feature integration, and performance/release qualification are its scope, not completed by Issue #3's repository regression run.

Consult Issues #3, #5, #11, #12, #14, #15, #28 and deferred #33 for history, current sequencing, and acceptance boundaries.
Issue #12's completed technical acceptance is recorded in the [historical acceptance ledger](docs/issue12_scientific_acceptance.md#stage-5-final-technical-acceptance). The [post-Issue-12 removal record](docs/package_structure.md#post-issue-12-compatibility-cleanup-handoff) documents the separate breaking cleanup. Preserve schema v1 and all seven historical persistence identity strings verbatim: these are opaque provenance labels, never import instructions.

Current task orientation:
- v0.8.0 and #14/#15 are complete; do not reopen their release gates without a concrete regression;
- #28 PRs 1–3 are merged; #28 remains open and its original failed 119/121 audit criterion and evidence are immutable;
- #33 PR #35 is merged; **#33 remains open but deferred**, with the new failed 99/100 strict audit target and required-recovery-capacity limitation explicitly retained;
- **current authorized scope:** Issue #37 PR 1 private fixed-K numerical foundations only; stop for review before PR 2 fitting orchestration or PR 3 measurement integration/scientific qualification;
- preserve PR 1/2 physics/statistics, existing bi-exponential numerical contracts, the ordered 66-name root API, and all historical evidence;
- leave #13 deferred until realistic experiment-specific or photonic-simulation inputs are available;
- do not start package movement, new scientific qualification, or automatic publishing/tagging without explicit authorization.
