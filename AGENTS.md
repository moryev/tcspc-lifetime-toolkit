# AGENTS.md

## Purpose

This file provides repository-level instructions for Codex and other repository-aware coding agents working on the **TCSPC Lifetime Toolkit**.

Treat it as an operational map, not as a complete project history. For detailed scientific findings, implementation history, and roadmap rationale, read the repository documentation and the relevant GitHub issues.

## Project scope

The TCSPC Lifetime Toolkit is a scientific Python package for simulating, fitting, and evaluating time-correlated single-photon-counting (TCSPC) decay data.

The toolkit combines:

- physical decay models;
- instrument-response-function (IRF) modelling and convolution;
- Poisson photon-counting simulation;
- classical lifetime fitting and reconvolution;
- canonical experimental-measurement import and representation;
- preprocessing and physically interpretable feature extraction;
- machine-learning lifetime estimation;
- controlled robustness/generalization benchmarks;
- uncertainty calibration and failure-awareness analysis.

The current release is **v0.7.0**. Weeks 8–9 of the scientific roadmap are complete. The repository is now in the post-Week-9 integration phase described in GitHub Issue #5.

## Repository map

- `src/tcspc_toolkit/` — package implementation.
- `tests/` — automated scientific and regression tests.
- `notebooks/` — reproducible scientific demonstrations and analyses.
- `docs/scientific_findings.md` — accumulated scientific findings, including the Week 8–9 conclusions.
- `docs/design/master_design_document.md` — current design principles.
- `configs/` — committed configuration files used for reproducible workflows.
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
- noise sampling is independent from deterministic signal generation;
- evaluation functions are organized by the quantity being evaluated, not merely by estimator family;
- public APIs use explicit scientific names;
- convenience wrappers may compose lower-level functions rather than duplicate their logic;
- preprocessing is analysis-dependent: there is no universally correct TCSPC preprocessing pipeline.

Issues #2 and #3 will later stabilize the public API and reorganize the package. Until then, do not perform broad architectural cleanup opportunistically while implementing another issue.

## Scientific guardrails

### Photon-counting semantics

Raw TCSPC histogram counts are modeled with Poisson statistics where the raw-count assumption applies.

Do not silently apply Poisson likelihoods or Poisson-specific uncertainty formulas to data that have already been normalized, background-subtracted into non-count values, rescaled, or otherwise transformed in ways that invalidate the count model.

Keep raw measurements conceptually distinct from derived representations.

### IRF physics

The measured TCSPC signal is not generally the ideal fluorescence decay. IRF convolution and detector background are core parts of the physical measurement model.

Do not replace reconvolution with a simpler decay fit when the task requires IRF-aware inference.

Keep IRF origin separate from IRF use: synthetic, measured, estimated, and deliberately misspecified IRFs should eventually be able to feed the same downstream forward-model machinery.

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
- `SampledIRF` is currently a deliberately narrow carrier for an imported sampled/measured IRF, not the final generalized IRF abstraction.
- The imported IRF trace keeps its own values, metadata, and provenance. Reconvolution normalizes a derived copy rather than mutating the imported trace.
- Issue #6 accepts only numerically compatible measurement/IRF grids. Do not silently resample, realign, baseline-correct, smooth, or infer zero time.
- Experimental lifetime estimates are separate from optional reference-based evaluation. A trusted reference is supplied explicitly and is not stored as intrinsic measurement ground truth.
- Experimental ML adapters may transform/predict with already-fitted development artifacts, but must not fit or refit representations/models on the experimental measurement.

Issue #8 owns generalized IRF sources, deterministic resampling/alignment policy, leading-edge IRF estimation, and broader IRF diagnostics. Build those capabilities on the canonical measurement path above rather than creating a second experimental-data model.

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
- uncertainty under distribution shift;
- physical model misspecification.

Random-Forest tree spread and training-data bootstrap spread are not automatically nominal prediction intervals.

Classical covariance and parametric Poisson bootstrap quantify statistical uncertainty conditional on the assumed forward model.

The central Week-9 result is:

> An estimator can correctly quantify statistical uncertainty while remaining confidently wrong because its physical model is incomplete.

Therefore do not interpret narrow intervals, low ensemble spread, good scalar Poisson deviance, or current aggregate residual diagnostics as universal evidence that the physical model is valid.

## Reproducibility rules

- Preserve explicit random seeds where benchmark reproducibility depends on them.
- Do not replace frozen benchmark seeds or regimes without an explicit task requirement.
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
2. **#6** — experimental TCSPC data ingestion, processing, and evaluation — **complete**;
3. **#8** — generalized IRF models, sampled-IRF preparation, and leading-edge/failure-awareness evaluation — **implemented and verified; commit/closure review pending**;
4. **#4** — Bayesian Poisson inference — **implemented, including scientific evaluation and Notebook 17; closure review pending**;
5. **#9** — SQLite persistence for experiments and benchmark results;
6. **#1** — standardize array input type annotations;
7. **#2** — API stabilization and package hardening;
8. **#11** — consolidate Week 7–9 evaluation architecture;
9. **#3** — reorganize `tcspc_toolkit` into coherent subpackages;
10. **#12** — full integration and regression verification;
11. **#13** — Week 10 Purcell-enhanced TCSPC sensing demonstration;
12. **#14** — Week 11 documentation and user experience;
13. **#15** — Week 12 continuous integration and release.

**Issue #4: Bayesian Poisson inference** has completed implementation, scientific evaluation, numerical audits, documentation, and Notebook 17; issue closure requires review.

Preserve the generalized-IRF and Bayesian contracts when later API/package work is explicitly requested.

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

## Bayesian integration orientation

Issue #8's implementation and Notebook 16 establish the generalized IRF
workflow. When reviewing Issue #4, read this file, Issues #5
and #4, the Issue-#6 measurement boundary, and the committed Issue-#8 source,
preparation, estimation, and evaluation APIs. Consult Notebooks 16–17 and the
design/scientific-findings documentation for the distinction between source
provenance, preparation history, fixed kernels, approximate proxies, and
model-conditional uncertainty. Begin with read-only orientation and a scoped
plan; do not treat roadmap ordering as authorization to implement the next
issue automatically.
