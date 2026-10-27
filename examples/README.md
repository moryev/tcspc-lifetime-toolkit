# Executable examples

Install the checkout as described in the [README](../README.md#installation),
then run these commands from its root. Each script uses committed library code
and small synthetic inputs. No untracked measurement files are needed.
Outputs below are **educational demonstrations**, not benchmark results;
see the separate [results guide](../docs/results.md) for accepted evidence.

## Choose a workflow

| Script | Purpose | Dependencies / inputs | Typical runtime |
| --- | --- | --- | --- |
| [quickstart.py](quickstart.py) | Simulate and fit an IRF-convolved Poisson histogram | Base install; generated synthetic data | Seconds |
| [experimental_csv.py](experimental_csv.py) | Import separate decay/IRF CSVs and reconvolve | Base; committed synthetic CSVs | Seconds |
| [biexponential_reconvolution.py](biexponential_reconvolution.py) | Descriptive paired mono/bi raw-measurement fits | Base; committed synthetic CSVs; exhaustive bi policy | Tens of seconds |
| [preprocessing_features.py](preprocessing_features.py) | Validate counts and extract physical features | Base; committed synthetic decay CSV | Seconds |
| [ml_lifetime_prediction.py](ml_lifetime_prediction.py) | Train and evaluate a small Ridge pipeline | Base; generated synthetic data | Seconds |
| [uncertainty_example.py](uncertainty_example.py) | Compare local covariance and Poisson bootstrap | Base; generated synthetic data, 100 refits | Usually under 30 seconds |
| [bayesian_inference.py](bayesian_inference.py) | Run genuine Bayesian Poisson reconvolution | Optional `bayesian` extra; generated synthetic data | About 1–2 minutes; hardware-dependent |
| [custom_estimator.py](custom_estimator.py) | Extend the generic estimator contract | Base; illustrative feature matrix | Seconds |
| [persistence_roundtrip.py](persistence_roundtrip.py) | Record, query and aggregate SQLite results | Base; new output path in an existing directory | Seconds |

Except for persistence and an explicitly requested figure, scripts print results
without writing files. Input paths are relative to the scripts, not the caller's
working directory. Random seeds are explicit; numerical last digits and runtimes
may vary across dependency versions.

## 1. Physical quick start

```bash
python examples/quickstart.py
python examples/quickstart.py --figure docs/images/quickstart.png
```

The complete chain is a 2.5 ns mono-exponential decay, a unit-area Gaussian IRF
(centre 1 ns, FWHM 0.3 ns), convolution, 100,000 expected signal photons in the
finite window, 2 background counts/bin, Poisson observation seed 42, and a
Poisson reconvolution fit. Background and a bounded temporal shift are fitted.
Expected output: lifetime about **2.5017 ns**, background **2.155 counts/bin**,
shift **0.00031 ns**, numerical validation `True`, deviance **543.328 / 512 bins**.
The second command regenerates the README figure from this same computed fit.
Neither the small error nor the deviance proves general model validity.

Read the [forward-model conventions](../docs/user_guide.md#simulation-and-reconvolution)
and explore [Notebook 09](../notebooks/09_poisson_reconvolution_fitting_and_validation.ipynb).

## 2. Experimental-style CSV import

```bash
python examples/experimental_csv.py
```

Loads `data/examples/experimental_style_decay.csv` in declared **ps** and
`experimental_style_irf.csv` in declared **ns**. Both are **synthetic fixtures**,
not acquired measurements. The source IRF is preserved; explicit preparation
uses the same physical grid, no resampling and zero registration shift.
Expected output: 81 bins, sampled-source provenance, preparation diagnostics,
a valid fit near **3.0021 ns**, and deviance near **76.643**.
No intrinsic experimental truth, reference error, or coverage is reported.

See [fixture provenance](../data/examples/README.md),
[measurement and IRF handling](../docs/user_guide.md#measurements-and-irf-preparation),
and [Notebook 15](../notebooks/15_experimental_tcspc_workflow.ipynb).

## 3. Preprocessing and features

```bash
python examples/preprocessing_features.py
```

Validates the synthetic CSV as raw counts, extracts counts, arrival-time,
quantile and tail-slope features with explicit windows, and creates a separate
total-normalized representation. Expected output includes raw total **7552**
and normalized sum **1.000000**. Raw observations remain available for inference.
The features include IRF/background effects; they are not unbiased lifetime
estimates. Normalization does not preserve Poisson-count semantics.

See [preprocessing choices](../docs/user_guide.md#preprocessing-and-machine-learning) and
[Notebook 11](../notebooks/11_feature_engineering.ipynb).

## 4. Leakage-safe ML lifetime prediction

```bash
python examples/ml_lifetime_prediction.py
```

Generates 96 independent mono-exponential observations (lifetimes 1–4 ns,
fixed Gaussian IRF, 20,000 signal photons, background 2 counts/bin).
Simulation seed is 1403; split seed is 42. The 72/24 split occurs before fitting
the normalized-histogram scaler/Ridge pipeline. Expected held-out MAE is about
**0.0242 ns**, compared with **0.6155 ns** for the training-mean baseline.
This is an easy matched-distribution illustration, not external generalization
evidence. It does not train, tune, or evaluate on frozen A–F data.
Repeated observations of shared physical parameters would require grouped splits.

See [ML workflow guidance](../docs/user_guide.md#preprocessing-and-machine-learning),
[Notebook 05](../notebooks/05_data_leakage_and_grouped_evaluation.ipynb), and
[Notebook 12](../notebooks/12_ml_benchmarking.ipynb) for broader evaluation.

## 5. Statistical uncertainty

```bash
python examples/uncertainty_example.py
```

Fits one synthetic 2.5 ns observation with matched IRF, then computes local
Poisson/Fisher covariance and a 100-refit parametric Poisson bootstrap.
Observation seed is 1404; bootstrap seed is 1405.
Expected estimate is **2.4725 ns**, local lifetime SD about **0.0224 ns**, and
bootstrap 90% interval about **[2.4257, 2.5005] ns**, with 100 valid refits.
The script prints validity and boundary/failure information. One observation
does not estimate coverage; bootstrap Monte Carlo error remains appreciable.
Both outputs condition on the assumed decay/IRF model and omit model-form error.

See [uncertainty interpretation](../docs/user_guide.md#uncertainty-and-model-validity)
and [Notebook 14](../notebooks/14_uncertainty_and_failure_awareness.ipynb).

## 6. Genuine Bayesian inference

```bash
python -m pip install -e ".[bayesian]"
python examples/bayesian_inference.py
```

This calls the real `emcee` inference path on 161 raw-count bins with a 2.5 ns
generating lifetime and a fixed Gaussian IRF/alignment. Priors are explicit:
Gamma(shape 1, rate 0.0004) for reconvolution amplitude, log-normal(log-mean
log(2), log-SD 0.8) for lifetime in ns, and Gamma(shape 1.2, rate 0.24) for
background counts/bin. These are illustrative assumptions, not fitted priors.

The configuration uses 16 walkers in each of two ensembles, 250 warm-up steps,
1,000 initial production steps, and extensions of 500 up to 10,000 steps.
All sampling-diagnostic thresholds retain their library defaults.
Observation/sampler seeds are 1401/1402. A validated run gives posterior median
about **2.4938 ns**, 90% credible interval **[2.4570, 2.5308] ns**.
Expect approximately 1–2 minutes on the development machine, not a timing promise.

Status, actual production steps and diagnostic failures are printed.
Budget exhaustion produces explicitly **provisional** summaries, not an accepted
posterior claim. Successful sampling assesses exploration of the assumed
posterior; it does not establish physical correctness or calibrated coverage.
The ordinary test suite checks configuration, dispatch and missing-extra behavior
without running this full example. Validate full inference separately with the
command above. No hand-constructed posterior is used by this script.

See [Bayesian interpretation](../docs/user_guide.md#bayesian-inference) and
[Notebook 17](../notebooks/17_bayesian_poisson_inference_and_model_mismatch.ipynb)
(the notebook additionally needs the scientific artifacts described in the
[reproducibility guide](../docs/reproducibility.md)).

## 7. Custom estimator extension

```bash
python examples/custom_estimator.py
```

The preserved example supplies a small user-owned representation and estimator
through `EstimatorSpec`, `fit_regressors()` and `predict_regressors()`.
Expected output: predictions **[1.5, 2.5]**. Its toy feature matrix demonstrates
the generic contract; it is not a physical TCSPC simulation or accuracy study.
See the [API contract](../docs/api.md#generic-estimator-extension-api).

## 8. SQLite persistence roundtrip

Choose a **new** database filename; its parent directory must exist:

```bash
python examples/persistence_roundtrip.py --output data/generated/roundtrip.sqlite
```

The preserved script refuses to overwrite an existing database. It records a
synthetic observation, its IRF, fit results, uncertainty and provenance, then
prints queried tables and an SQL aggregation. Classical fits/covariance are
computed; Bayesian posterior/sampler/PPC summaries are
**illustrative persistence fixtures**, not results of sampling. This example
does not require `emcee` and must not be used as Bayesian scientific evidence.

SQLite support is in Python's standard library. Schema v1 stores scalar facts
and provenance; histograms and posterior chains are not embedded as large BLOBs.
See the [persistence contract](../docs/persistence.md) for query semantics,
duplicates, references and aggregation validity policies.

## 9. Experimental mono/bi comparison

```bash
python examples/biexponential_reconvolution.py
```

Uses the same explicitly prepared synthetic CSV fixtures and original raw bins
for both models. Prints parameter estimates, local numerical validation, budget
status, identifiability diagnostics and descriptive AIC/bin-based BIC differences.
The mono lifetime is unbounded above; the bi bounds are [.1,8] ns. No trusted
experimental reference, physical model-order decision or likelihood-ratio p-value
is inferred. This example is separate from the not-yet-run
[PR-3 qualification study](../docs/issue28_pr3_qualification.md).
