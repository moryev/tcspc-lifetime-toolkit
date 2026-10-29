# Documentation

Start with the [five-minute quick start](../README.md#five-minute-quick-start)
and the [executable example index](../examples/README.md). The examples require
only an installed checkout and the inputs already committed to this repository.

## Choose a workflow

| I want to… | Start here | Detailed reference |
| --- | --- | --- |
| Simulate and fit a realistic histogram | [Quick start](../examples/quickstart.py) | [Forward model and parameter conventions](user_guide.md#simulation-and-reconvolution) |
| Analyze a CSV histogram and sampled IRF | [CSV example](../examples/experimental_csv.py) | [Measurements and IRF preparation](user_guide.md#measurements-and-irf-preparation) |
| Prepare features or train an estimator | [Features and ML examples](../examples/README.md) | [Preprocessing and ML](user_guide.md#preprocessing-and-machine-learning) |
| Interpret statistical uncertainty | [Uncertainty example](../examples/uncertainty_example.py) | [Uncertainty meanings](user_guide.md#uncertainty-and-model-validity) |
| Sample a Bayesian posterior | [Bayesian example](../examples/bayesian_inference.py) | [Bayesian assumptions](user_guide.md#bayesian-inference) |
| Use a custom regressor | [Custom estimator](../examples/custom_estimator.py) | [Estimator extension API](api.md#generic-estimator-extension-api) |
| Store and query scientific results | [SQLite example](../examples/persistence_roundtrip.py) | [Persistence contract](persistence.md) |
| Use the command line | [CLI guide](cli.md) | `tcspc --help`, `tcspc simulate --help`, `tcspc fit --help` |
| Inspect established scientific results | [Results summary](results.md) | [Scientific findings](scientific_findings.md) |
| Reproduce a workflow or run tests | [Reproducibility guide](reproducibility.md) | [Issue-12 acceptance ledger](issue12_scientific_acceptance.md#stage-5-final-technical-acceptance) |
| Run quality/build checks or review release gates | [Release readiness](release_readiness.md) | [CI workflow](../.github/workflows/ci.yml) |

## Reference documents

- [Issue-33 independent qualification](issue33_stage3_qualification_v1.md): numerical
  safeguards, passing primary recovery targets, failed strict audit agreement
  (99/100 against 100/100), and the remaining recovery-capacity limitation.
- [Public API](api.md): curated root exports, supported canonical modules,
  generic estimator extension, and retained symbol aliases.
- [Package structure](package_structure.md): ownership, dependencies, and the
  migration mapping for the 38 removed flat modules.
- [Evaluation contracts](evaluation.md): point/reference facts, separate interval
  and score attachments, and intentionally different metric populations.
- [SQLite persistence](persistence.md): schema v1, identities, transactions,
  recording adapters, and read-only queries.
- [Design principles](design/master_design_document.md): scientific boundaries
  underlying implementation choices.

The [notebook catalog](reproducibility.md#notebook-navigation-and-prerequisites) separates introductory
workflows, expensive studies, and notebooks requiring saved artifacts.
Historical notebook outputs and acceptance checkpoints remain evidence of their
original runs. Use the [current reference policy](results.md#which-evidence-is-current)
when comparing numerical results.
