# Reproducibility and execution guide

Start with [installation](../README.md#installation) and the
[eight executable examples](../examples/README.md). This guide locates the
reproducible evidence and explains execution cost; scientific interpretations
live in [results](results.md) and [scientific findings](scientific_findings.md).

## Environment and quick validation

Python **>= 3.11** is declared in [pyproject.toml](../pyproject.toml).
The base install includes NumPy, SciPy, pandas, Matplotlib and scikit-learn.
The `dev` extra adds pytest/Jupyter and pinned quality/build tools;
`bayesian` adds `emcee>=3.1.6,<4`.
SQLite needs no extra dependency. Editable installation from a checkout gives
access to examples, notebooks and committed input files as well as the package.

For complete development validation, from an activated environment at the root:

```bash
python -m pip install -e ".[dev,bayesian]"
python -m pytest
```

For focused documentation/CLI checks:

```bash
python -m pytest tests/test_documentation_examples.py tests/test_cli.py tests/test_package_structure.py
```

The example tests execute the README's unchanged Python block, check physical
fit output and raw-count semantics, exercise standalone commands, and check
local links/import contracts. Bayesian configuration/dispatch/optional-dependency
checks are lightweight. Full sampling of the Bayesian **example** is a separate
validation, not mandatory work in every test-suite invocation:

```bash
python examples/bayesian_inference.py
```

The broader suite retains its existing Bayesian tests and optional-dependency
behavior. Install both extras when reporting a full-suite result. The declared
Python range is not evidence that every Python/dependency combination has been
tested. Issue-12 clean-install acceptance was on Python 3.12. Issue #15 qualified
Ubuntu 24.04 Python 3.11/3.12 and pinned Windows exact regeneration in
[GitHub Actions run #39717483724](https://github.com/moryev/tcspc-lifetime-toolkit/actions/runs/39717483724).
See [release readiness](release_readiness.md) for identical local/CI quality,
coverage and artifact-install commands, dependency floors and remaining gates.

Run the full repository suite in an editable **Git checkout**: provenance and
source-architecture guards intentionally inspect Git/source state. An installed
wheel is qualified separately outside the checkout; its success does not require
shipping tests or notebooks as runtime package data. The sdist includes the
redistributable workflows but is not itself a Git checkout.

## Seeds, inputs and outputs

| Workflow | Reproducibility choice | Output |
| --- | --- | --- |
| Quick start | Observation seed 42; explicit grid, IRF and photon budget | Console; optional PNG |
| CSV/features | Committed synthetic files; declared input units | Console |
| ML example | Simulation seed 1403; split seed 42; fitted training-only pipeline | Console |
| Uncertainty example | Observation seed 1404; bootstrap seed 1405; 100 refits | Console |
| Bayesian example | Observation seed 1401; sampler seed 1402; explicit priors/budget | Console |
| Custom estimator | Deterministic toy arrays | Console |
| Persistence | Explicit synthetic inputs and provenance; fresh database path | SQLite and queried tables |

The [fixture README](../data/examples/README.md) describes all bundled CSVs.
Their generator settings are provenance, not intrinsic experimental truth.
In-memory examples require no external or private datasets. Persistence timestamps
and generated run identities need not make database bytes identical across runs.

For benchmark work, retain committed seeds and the named independent Issue-4
observation/bootstrap/Bayesian/posterior-predictive streams. Never reuse frozen
A–F data for training, representation fitting, hyperparameter selection, or
calibration. Numeric comparisons use the recorded tolerance/population policy;
wall-clock times and floating-point last digits are not portable goldens.
For the frozen Week-7 sample #770, see the [bitwise versus numerical
reproducibility investigation](release_readiness.md#week-7-bitwise-and-numerical-reproducibility):
the original raw-count realization is fixed for fitting, while portable
simulation checks compare the deterministic expected-count model.

## Regenerate the quick-start figure

```bash
python examples/quickstart.py --figure docs/images/quickstart.png
```

This performs the same seeded simulation and fit as the README, then plots its
generating expectation, observed counts, fitted curve and signed Poisson-deviance
residuals. It uses Matplotlib's non-interactive backend. No additional optimizer
run, benchmark regeneration, or scientific-reference modification occurs.
The figure is an educational matched-model example, not a coverage study.

## Committed references and manifests

The compact [data/reference](../data/reference) JSON records preserve current
accepted Week-7/8, Week-9 ML, and Week-9 classical summaries and their provenance.
Read them directly to inspect results without running studies. Their exact paths
and metric populations are linked in [results](results.md#which-evidence-is-current).
The [Issue-12 acceptance ledger](issue12_scientific_acceptance.md) owns the
reference policy, full commands, runtime evidence, audit decisions and limitations.
Its historical anchors and reference outputs are preserved.

| Configuration | Consumer / purpose |
| --- | --- |
| [example_simulation.json](../configs/example_simulation.json) | Small `SimulationConfig` example; not a CLI `--config` input |
| [issue8_irf_workflow.json](../configs/issue8_irf_workflow.json) | Notebook 16's generalized-IRF demonstration manifest |
| [issue4_bayesian_workflow.json](../configs/issue4_bayesian_workflow.json) | Matched-model smoke/scientific profile manifest |
| [issue4_bayesian_mismatch_workflow.json](../configs/issue4_bayesian_mismatch_workflow.json) | Paired decay/IRF mismatch smoke/scientific profile manifest |

The Issue-4 runner scripts expose committed library profiles; consult their
`--help` rather than assuming every JSON file is a CLI input:

```bash
python scripts/run_issue4_bayesian_benchmarks.py --help
python scripts/run_issue4_bayesian_mismatch_benchmarks.py --help
python scripts/run_issue12_scientific_reproducibility.py --help
python scripts/run_issue12_integration.py --help
```

The Issue-12 scientific runner compares regenerated results with a selected
reference; choosing a reference is an explicit acceptance decision. Use the
documented post-correction policy, not the oldest reference merely because it
is available. Broad scientific profiles may take many minutes to hours.
They are not needed to use the examples or to update documentation.

Issue-4 generated chains/tables/audits live under ignored `data/generated/`
locations. They are distinct from committed compact reference JSONs. The
Stage-5/6 scientific records and Stage-6.5 corrected pseudo-true references are
historical evidence: do not overwrite them while experimenting. Use separate
output directories for new runs, preserve configurations/seeds, and label any
new correction explicitly. The correction/audit scripts are specialized tools,
not routine installation steps. See the
[Bayesian findings and correction record](scientific_findings.md#post-week-9--issue-4-bayesian-poisson-inference-and-model-conditional-uncertainty).

## Notebook navigation and prerequisites

Install `dev`, launch `jupyter lab`, open the desired notebook, and run from a
fresh kernel using the installed environment. Notebook titles describe their
scientific scope; later numbers are not required to use earlier API workflows.

| Notebooks | Purpose / prerequisite |
| --- | --- |
| [01](../notebooks/01_tcspc_simulation.ipynb), [02](../notebooks/02_classical_lifetime_fitting.ipynb) | Ideal simulation and baseline fitting; 02 consumes the CSV generated by 01 |
| [03](../notebooks/03_synthetic_dataset_analysis.ipynb), [04](../notebooks/04_first_ml_baseline.ipynb) | Synthetic datasets and first ML baseline; 04 consumes 03's generated dataset |
| [05](../notebooks/05_data_leakage_and_grouped_evaluation.ipynb), [06](../notebooks/06_grouped_dataset_api_workflow.ipynb) | Leakage and grouped evaluation |
| [07](../notebooks/07_irf_convolution_and_realistic_simulation.ipynb), [08](../notebooks/08_naive_vs_reconvolution_fitting.ipynb), [09](../notebooks/09_poisson_reconvolution_fitting_and_validation.ipynb) | IRF convolution, reconvolution and Poisson fitting |
| [10](../notebooks/10_preprocessing_tcspc_histograms.ipynb), [11](../notebooks/11_feature_engineering.ipynb) | Preprocessing choices and physical features |
| [12](../notebooks/12_ml_benchmarking.ipynb), [13](../notebooks/13_generalization_and_robustness.ipynb) | Development comparisons and frozen A–F external evaluation; larger workloads |
| [14](../notebooks/14_uncertainty_and_failure_awareness.ipynb) | Calibration/failure-awareness studies; bootstrap-heavy, potentially lengthy |
| [15](../notebooks/15_experimental_tcspc_workflow.ipynb) | Canonical experimental-style path using bundled synthetic CSVs |
| [16](../notebooks/16_generalized_irf_and_failure_awareness.ipynb) | Generalized IRF preparation, mismatch and approximate leading-edge controls |
| [17](../notebooks/17_bayesian_poisson_inference_and_model_mismatch.ipynb) | Bayesian scientific-record analysis; needs generated Stage-5/6 results and Stage-6.5 correction artifacts |

Notebook 17 is not self-contained in a clean checkout. Its initial cells specify
the required artifact paths and fail clearly if they are absent. No public
artifact download archive is currently linked by this repository. Use the small
[Bayesian script](../examples/bayesian_inference.py) for standalone actual
inference, or follow the existing scientific workflow to reproduce full records.
Do not substitute hand-built posterior summaries for missing sampled evidence.

## Recording a reproducible new analysis

Record the code revision, Python/dependency versions, input units/data kind,
measurement and IRF provenance, explicit preparation operations, model/prior
assumptions, seeds, and fit/sampling validity. Separate generating parameters,
trusted experimental references and deterministic pseudo-true projections.
The [SQLite contract](persistence.md) provides optional recording/query adapters
and scalar scientific fields without making persistence part of computation.

For publication, cite the [CITATION.cff](../CITATION.cff) release metadata and also
identify the checkout revision when using unreleased work. Current package
metadata remains 0.7.0; no release is created by the documentation update.
