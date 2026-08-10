# TCSPC Lifetime Toolkit

A scientific Python toolkit for simulating, fitting, and evaluating time-correlated single-photon counting decay curves.

The project is being developed as an end-to-end framework for classical and machine-learning-based fluorescence-lifetime estimation. Its long-term goal is to support realistic TCSPC data processing, benchmarking of lifetime estimators, and lifetime-based sensing demonstrations.

## Scientific motivation

Time-correlated single-photon counting (TCSPC) is widely used to measure fluorescence and excited-state lifetimes. A measured TCSPC histogram contains photon counts distributed over time bins following an excitation event.

Even for a simple mono-exponential decay, the measured histogram differs from the underlying expected decay because photon detection is a stochastic process. For ideal photon-counting measurements, the observed count in each time bin can be modelled as a Poisson-distributed random variable.

This project separates the analysis into reusable stages:

1. define a physical decay model;
2. calculate the expected photon counts;
3. generate a Poisson-sampled measurement;
4. estimate model parameters by nonlinear fitting;
5. reconstruct the fitted signal;
6. evaluate lifetime errors and fit residuals.
7. generate synthetic datasets for machine-learning experiments;
8. train and evaluate baseline data-driven lifetime estimators;
9. compare ordinary and group-aware evaluation strategies to identify potential data leakage.

The current implementation provides both a transparent classical fitting baseline and preliminary machine-learning workflows 
based on normalized TCSPC histograms. These workflows include simple regression models, grouped synthetic datasets, 
and leakage-aware evaluation. The current implementation also supports Gaussian instrument-response-function modelling and 
numerical convolution. Future releases will extend these capabilities with reconvolution fitting, experimental IRF support, 
and more advanced detector models.

## Current functionality

The current version supports:

* mono-, bi-, and multi-exponential decay modelling;
* constant background counts;
* Gaussian instrument response function generation;
* IRF normalization to unit temporal area;
* non-integer temporal IRF shifting;
* numerical convolution of fluorescence decays with IRFs;
* IRF-convolved TCSPC simulation with Poisson photon-count sampling;
* Poisson sampling of photon-count histograms;
* reproducible simulations using a random seed;
* generation of independent and grouped synthetic mono-exponential datasets;
* structured storage of simulated curves and metadata;
* metadata-based selection of machine-learning targets;
* nonlinear least-squares lifetime fitting;
* covariance-based parameter standard errors;
* fitted-signal reconstruction;
* raw residual calculation;
* Poisson-scaled Pearson residuals;
* absolute and relative lifetime errors;
* baseline machine-learning lifetime estimation using normalized TCSPC histograms;
* random and group-aware train-test evaluation;
* data-leakage analysis for repeated noisy realizations;
* CSV export of simulated and evaluated data;
* Jupyter notebooks demonstrating the classical, machine-learning, and grouped-evaluation workflows.

## Current scientific assumptions

The current simulation workflow represents TCSPC measurements as a sequence of physical and numerical stages:

1. generate an ideal fluorescence decay;
2. generate and normalize an instrument response function (IRF);
3. convolve the ideal decay with the IRF;
4. add the detector background;
5. sample the resulting expected counts using Poisson statistics.

For a mono-exponential fluorescence decay, the ideal signal is represented by

```math
I(t) = A \exp\left(-\frac{t}{\tau}\right).
```

where:

* $A$ is the decay amplitude;
* $\tau$ is the fluorescence lifetime.

The instrument response function describes the temporal broadening introduced by the measurement system. 
In the current implementation, the IRF is modelled as a Gaussian function,

```math
\mathrm{IRF}(t)
=
C \exp\left[
-\frac{(t-t_0)^2}{2\sigma^2}
\right].
```

where:

* $t_0$ is the IRF centre;
* $\sigma$ determines the IRF width;
* $C$ is the Gaussian amplitude.

The Gaussian width is specified through the full width at half maximum (FWHM),

```math
\mathrm{FWHM}
=
2\sqrt{2\ln 2}\,\sigma.
```

Before convolution, the IRF is normalized to unit temporal area,

```math
\int \mathrm{IRF}(t)\,dt = 1.
```

The instrument-broadened fluorescence signal is then calculated by convolution,

```math
[\mathrm{IRF} * I](t)
=
\int \mathrm{IRF}(t-t') I(t')\,dt'.
```

Numerically, the convolution is evaluated on a uniform time grid. The discrete convolution therefore includes the time-bin width $\Delta t$ so that the numerical sum approximates the continuous convolution integral.

A constant detector background $B$ is added after convolution, giving the expected TCSPC signal

```math
\lambda(t)
=
[\mathrm{IRF} * I](t) + B.
```

For each time bin $i$, the measured photon count is then sampled according to

```math
N_i
\sim
\mathrm{Poisson}(\lambda_i).
```

where $\lambda_i$ is the expected photon count in that time bin.

The resulting forward model can therefore be summarized as

```math
I(t)
\longrightarrow
\mathrm{IRF}(t)
\longrightarrow
[\mathrm{IRF} * I](t)
\longrightarrow
\lambda(t)
\longrightarrow
N_i.
```

**Version 0.2 introduces Gaussian instrument-response modelling and IRF-convolved TCSPC simulation. 
The current implementation assumes a uniform time grid and a time-invariant Gaussian IRF. The current IRF-convolved workflow 
is demonstrated for mono-exponential fluorescence decay. Experimental IRF loading, more advanced detector effects such as 
pile-up and afterpulsing, and full reconvolution fitting are not yet included.**

## Installation

### Requirements

* Python 3.11 or newer
* pip
* Git

### Clone the repository

Clone the repository and enter the project directory:

```bash
git clone https://github.com/moryev/tcspc-lifetime-toolkit.git
cd tcspc-lifetime-toolkit
```

### Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install the package

Install the project in editable mode:

```bash
python -m pip install -e .
```

Editable installation allows changes made inside `src/tcspc_toolkit/` to become available without reinstalling the package after every edit.

### Install the project together with the development dependencies

There are project-specific optional dependencies (such as Jupyter and pytest). If you want to install the project in editable mode together with these optional dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Minimal example

The following example generates an idealized TCSPC decay with Poisson photon-counting statistics:

```python
import numpy as np

from tcspc_toolkit.simulation import simulate_monoexponential_decay

time_ns = np.linspace(
    start=0.0,
    stop=20.0,
    num=512,
)

expected_counts, measured_counts = simulate_monoexponential_decay(
    time=time_ns,
    amplitude=10_000.0,
    lifetime=2.5,
    background=5.0,
    random_seed=42,
)

print("First expected counts:")
print(expected_counts[:5])

print("First measured counts:")
print(measured_counts[:5])
```

The fitting and evaluation workflow is:

```python
import numpy as np

from tcspc_toolkit.evaluation import (
    calculate_absolute_lifetime_error,
    calculate_reduced_residuals,
    calculate_relative_lifetime_error,
    calculate_residuals,
    generate_fitted_signal,
)
from tcspc_toolkit.fitting import fit_monoexponential_decay

true_lifetime_ns = 2.5

background_guess = float(
    np.median(measured_counts[-50:])
)

amplitude_guess = float(
    measured_counts.max() - background_guess
)

fit_result = fit_monoexponential_decay(
    time=time_ns,
    counts=measured_counts,
    initial_guess=(
        amplitude_guess,
        2.0,
        background_guess,
    ),
)

fitted_counts = generate_fitted_signal(
    time=time_ns,
    fit_result=fit_result,
)

raw_residuals = calculate_residuals(
    observed=measured_counts,
    fitted=fitted_counts,
)

reduced_residuals = calculate_reduced_residuals(
    observed=measured_counts,
    fitted=fitted_counts,
)

absolute_error_ns = calculate_absolute_lifetime_error(
    true_lifetime=true_lifetime_ns,
    estimated_lifetime=fit_result.lifetime,
)

relative_error = calculate_relative_lifetime_error(
    true_lifetime=true_lifetime_ns,
    estimated_lifetime=fit_result.lifetime,
)

print(
    f"Estimated lifetime: "
    f"{fit_result.lifetime:.4f} ± "
    f"{fit_result.lifetime_std:.4f} ns"
)

print(f"Absolute error: {absolute_error_ns:.4f} ns")
print(f"Relative error: {relative_error * 100:.2f}%")
```

## Notebooks

The repository currently includes the following notebook workflows:

### `01_tcspc_simulation.ipynb`

Demonstrates:

* construction of a time axis;
* calculation of an expected mono-exponential decay;
* Poisson sampling of photon counts;
* visualization on linear and logarithmic scales;
* creation of a pandas DataFrame;
* CSV export of synthetic data.

### `02_classical_lifetime_fitting.ipynb`

Demonstrates:

* loading simulated TCSPC data;
* selection of initial fitting parameters;
* nonlinear least-squares fitting;
* lifetime and uncertainty estimation;
* fitted-signal reconstruction;
* absolute and relative lifetime errors;
* raw and Poisson-scaled residual analysis;
* export of detailed and fit-level results.

### `03_synthetic_dataset_analysis.ipynb`

Demonstrates:

* generation of multiple synthetic TCSPC decay curves;
* creation of machine-learning matrices and lifetime targets;
* inspection of curve-level metadata;
* conversion from matrix to long-table representation;
* validation of photon-count and metadata consistency;
* visualization of selected decay curves;
* analysis of lifetime and photon-count distributions;
* comparison of target, expected, and measured photon counts;
* grouping and averaging of curves by lifetime range;
* normalization of decay curves for shape comparison;
* CSV export of long-format data;
* compressed NumPy export of the machine-learning dataset.

### `04_first_ml_baseline.ipynb`

Demonstrates:

* loading the compressed synthetic machine-learning dataset;
* inspection of feature-matrix, target, and metadata shapes;
* normalization of each TCSPC decay curve by its total photon count;
* separation of the dataset into training and test subsets;
* construction of a mean-prediction regression baseline;
* training of Ridge regression, Random Forest, and Gradient Boosting models;
* prediction of fluorescence lifetimes for previously unseen decay curves;
* evaluation using mean absolute error, median absolute error, relative error, and $R^2$;
* comparison of model performance against the mean baseline;
* analysis of prediction error as a function of photon count;
* analysis of prediction error as a function of true lifetime;
* visualization of predicted versus true lifetimes;
* collection of model-level metrics in a summary table.

### `05_data_leakage_and_grouped_evaluation.ipynb`

Demonstrates:

* generation of repeated Poisson realizations from shared simulation parameter sets;
* assignment of curve, parameter-group, and realization identifiers;
* construction of a naive random row-level train–test split;
* construction of a parameter-group-aware train–test split;
* verification of row-level and parameter-group overlap between training and test subsets;
* analysis of how repeated realizations are distributed across a naive random split;
* training of mean-prediction, Ridge, Random Forest, and Gradient Boosting models under both splitting protocols;
* comparison of model performance using mean absolute error, median absolute error, relative error, and $R^2$;
* quantification of the performance difference between random and group-aware evaluation;
* analysis of signed prediction residuals and systematic prediction bias;
* analysis of prediction error as a function of true lifetime;
* analysis of prediction error as a function of measured total photon count;
* identification of lifetime and photon-count regions with the largest prediction errors;
* discussion of interpolation performance, simulation-group leakage, and scientifically defensible test-set construction;
* documentation of current limitations and future evaluation scenarios involving new noise regimes, model mismatch, and unfamiliar instrument response functions.

### `06_grouped_dataset_api_workflow.ipynb`

Demonstrates:

* generation of grouped mono-exponential TCSPC datasets through the reusable generate_grouped_monoexponential_dataset() API;
* inspection of the returned SyntheticDataset, including the time axis, measured histogram matrix, and metadata table;
* verification of dataset dimensions, parameter-group counts, realization counts, and within-group parameter consistency;
* visualization of independent Poisson realizations generated from one shared physical parameter set;
* normalization of measured histograms by their total photon counts;
* explicit selection of the lifetime target through dataset.get_targets();
* extraction of parameter-group identifiers from dataset metadata;
* construction of an ordinary row-level random train–test split;
* construction of a parameter-group-aware train–test split using GroupShuffleSplit;
* quantification of parameter-group overlap under both splitting protocols;
* training of a mean-prediction baseline and a Random Forest regressor;
* comparison of model performance using mean absolute error, median absolute error, mean relative error, and $R^2$;
* visualization of mean absolute error under random and group-aware evaluation;
* comparison of true and predicted lifetimes for the Random Forest model;
* demonstration of how the package-level grouped dataset API supports reproducible and leakage-aware ML evaluation;
* discussion of why group-aware splitting is preferable when several noisy realizations share the same underlying simulation parameters.

### `07_irf_convolution_and_realistic_simulation.ipynb`

Demonstrates:

* construction of a uniform TCSPC time axis and inspection of the corresponding time-bin width;
* generation of Gaussian instrument response functions through `generate_gaussian_irf()`;
* normalization of IRFs through `normalize_irf()` so that their area is equal to one;
* comparison of raw and normalized IRFs and verification that normalization preserves the IRF shape while removing dependence on the original Gaussian amplitude;
* temporal shifting of normalized IRFs through `shift_irf()` using positive and negative time offsets;
* visualization of the IRF shift convention and inspection of shifted peak positions;
* generation of an ideal mono-exponential fluorescence decay through `monoexponential_decay()` with zero detector background before convolution;
* numerical convolution of the ideal fluorescence signal with the IRF through `convolve_decay_with_irf()`;
* comparison of ideal and IRF-convolved fluorescence decays on linear and logarithmic scales;
* explicit addition of detector background after convolution according to the physical model $`\lambda(t) = [\mathrm{IRF} * I](t) + B`$;
* demonstration of why detector background should not be included in the fluorescence signal before convolution;
* Poisson sampling of the expected IRF-convolved signal through `sample_photon_counts()` using a reproducible NumPy random-number generator;
* verification of Poisson-sampling reproducibility using independent generators initialized with the same random seed;
* comparison of expected counts and Poisson-sampled TCSPC histograms;
* demonstration of the complete compositional simulation workflow from fluorescence model to convolution, detector background, and photon-counting noise;
* investigation of how increasing IRF FWHM distorts the leading edge, peak position, and temporal shape of a short-lifetime fluorescence decay;
* comparison of IRF-convolved signals for several fluorescence lifetimes at a fixed IRF width;
* analysis of the relationship between fluorescence lifetime and IRF FWHM, including cases where the lifetime is shorter than, comparable to, or much longer than the instrument response;
* peak-normalized comparison of ideal and convolved decay curves using matched colors for each fluorescence lifetime;
* logarithmic visualization of lifetime-dependent decay tails while excluding floating-point-scale numerical convolution artifacts below a relative signal threshold;
* investigation of how temporal IRF misalignment changes the position and shape of the convolved decay;
* analysis of boundary and truncation effects when shifted IRFs extend beyond the finite observation window;
* inspection of IRF-area loss caused by truncation and its effect on the resulting convolved signal;
* investigation of time-bin resolution and how insufficient temporal sampling affects narrow IRFs and numerical convolution;
* integration checks confirming that IRF generation, normalization, convolution, background addition, and Poisson sampling work together consistently;
* discussion of the main numerical and physical limitations of the current realistic TCSPC simulation workflow and its extension toward reconvolution fitting and more complex decay models.

08_naive_vs_reconvolution_fitting.ipynb

Demonstrates:

* construction of a uniform TCSPC time axis for lifetime-fitting experiments and inspection of the corresponding time-bin width;
* generation and normalization of a fixed Gaussian instrument response function through `generate_gaussian_irf()` and `normalize_irf()`;
* verification that the normalized IRF has unit integrated area before it is used in convolution and reconvolution fitting;
* definition of representative fluorescence lifetimes spanning regimes where the lifetime is much longer than, comparable to, and shorter than the IRF FWHM;
* generation of ideal mono-exponential fluorescence decays through `monoexponential_decay()`;
* numerical convolution of the ideal fluorescence signal with the fixed IRF through `convolve_decay_with_irf()`;
* adjustment of the decay amplitude so that different lifetime conditions contain approximately the same total number of expected signal photons;
* explicit addition of detector background after convolution according to the physical measurement model $\mu(t) = A[\mathrm{IRF} * I_\tau](t) + B$;
* generation of Poisson-sampled TCSPC histograms through `sample_photon_counts()` using reproducible NumPy random-number generators;
* comparison of the same simulated TCSPC histogram with two competing lifetime-fitting approaches: a naive mono-exponential fit and an IRF-aware reconvolution fit;
* restriction of the naive exponential fit to the measured decay region beginning at the histogram peak so that the simple model is not trivially penalized by the IRF-generated leading edge;
* construction of data-driven initial guesses for amplitude, lifetime, and background without using the known true lifetime;
* use of the naive-fit result as a practical initialization for the reconvolution fit while keeping the known IRF shape and width fixed;
* reconvolution fitting through `fit_monoexponential_reconvolution()` with simultaneous estimation of amplitude, fluorescence lifetime, detector background, and temporal IRF shift;
* visual comparison of measured photon counts, the true expected reconvolved signal, the naive fitted decay, and the reconvolution fitted curve;
* calculation of signed relative lifetime errors through $(\tau_{\mathrm{fit}}-\tau_{\mathrm{true}})/\tau_{\mathrm{true}}$ to distinguish lifetime overestimation from underestimation;
* comparison of recovered lifetimes for representative cases including $\tau \gg \mathrm{FWHM}_{\mathrm{IRF}}$, $\tau \approx 3\,\mathrm{FWHM}_{\mathrm{IRF}}$, $\tau \approx \mathrm{FWHM}_{\mathrm{IRF}}$, and $\tau < \mathrm{FWHM}_{\mathrm{IRF}}$;
* systematic lifetime sweep over a broad range of $\tau_{\mathrm{true}}/\mathrm{FWHM}_{\mathrm{IRF}}$ values to quantify when neglecting the IRF becomes scientifically significant;
* visualization of relative lifetime bias as a function of $\tau_{\mathrm{true}}/\mathrm{FWHM}_{\mathrm{IRF}}$ on a logarithmic horizontal axis;
* explicit marking of the physically important transition $\tau_{\mathrm{true}} = \mathrm{FWHM}_{\mathrm{IRF}}$ in the lifetime-bias figure;
* demonstration that naive exponential fitting increasingly overestimates short lifetimes as the fluorescence lifetime approaches and falls below the IRF width;
* demonstration that reconvolution fitting substantially suppresses the systematic lifetime bias introduced by ignoring the instrument response;
* repeated Poisson simulation of each lifetime condition across multiple independent realizations to separate systematic model bias from statistical photon-counting variability;
* calculation of median relative lifetime error and 16th–84th percentile intervals for both fitting approaches;
* comparison of statistical spread between naive and reconvolution lifetime estimates across the full lifetime-to-IRF-width range;
* verification of numerical fit success rates for both fitting methods across all simulated lifetime regimes and Poisson realizations;
* demonstration that successful optimizer convergence alone is not sufficient for physical correctness, since the naive model can converge reliably while remaining systematically biased;
* analysis of the distinction between model-induced bias and increasing parameter uncertainty when fluorescence lifetimes become shorter than the instrument response;
* presentation of a controlled scientific benchmark showing why reconvolution becomes necessary when $\tau_{\mathrm{true}}$ approaches $\mathrm{FWHM}_{\mathrm{IRF}}$;
* discussion of the remaining limitation that both fitting approaches still use least-squares objectives despite the underlying Poisson photon-counting statistics, motivating the next development step toward Poisson-aware fitting.

## Repository structure

```text
tcspc-lifetime-toolkit/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── examples/
│   │   └── ideal_tcspc_decay.csv
│   └── generated/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_tcspc_simulation.ipynb
│   ├── 02_classical_lifetime_fitting.ipynb
│   ├── 03_synthetic_dataset_analysis.ipynb
│   ├── 04_first_ml_baseline.ipynb
│   ├── 05_data_leakage_and_grouped_evaluation.ipynb
│   ├── 06_grouped_dataset_api_workflow.ipynb
│   ├── 07_irf_convolution_and_realistic_simulation.ipynb
│   └── 08_naive_vs_reconvolution_fitting.ipynb
│
├── src/
│   └── tcspc_toolkit/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── convolution.py
│       ├── datasets.py
│       ├── evaluation.py
│       ├── fitting.py
│       ├── irf.py
│       ├── ml_evaluation.py
│       ├── models.py
│       └── simulation.py
│
└── tests/
    ├── conftest.py
    ├── test_convolution.py
    ├── test_datasets.py
    ├── test_fitting.py
    ├── test_irf.py
    ├── test_ml_evaluation.py
    ├── test_models.py
    └── test_simulation.py
```

The modules currently have the following responsibilities:

* `__init__.py`: package initialization and definition of the public package interface;
* `__main__.py`: package entry point for python -m tcspc_toolkit;
* `cli.py`: command-line tools for simulating and fitting TCSPC data;
* `convolution.py`: numerical convolution and temporal-grid alignment of ideal decay curves with instrument-response functions, including time-bin scaling and measurement-window truncation;
* `datasets.py`: synthetic datasets generation for the consequent ML baseline;
* `evaluation.py`: fitted signals, residuals, and lifetime-error metrics;
* `fitting.py`: nonlinear parameter estimation and structured fit results;
* `irf.py`: generation and manipulation of instrument response functions, including Gaussian IRF construction, normalization, temporal shifting, and related validation;
* `ml_evaluation.py`: regression metrics and diagnostic analyses for machine-learning lifetime predictions;
* `models.py`: mathematical decay models;
* `simulation.py`: expected-curve generation and Poisson sampling;
* `data/examples/`: small example datasets tracked by Git;
* `data/generated/`: generated outputs that are not normally tracked by Git;
* `notebooks/`: documented analysis workflows;
* `tests/`: automated verification of physical, numerical, and package behaviour;
* `pyproject.toml`: package metadata, dependencies, build configuration, and command-line entry points.

## Current limitations

The current implementation is intentionally simplified.

It does not yet include:

* Poisson maximum-likelihood fitting;
* weighted least-squares fitting;
* pile-up effects;
* detector dead time;
* afterpulsing;
* time-dependent background;
* experimental file-format import;
* calibrated confidence or prediction intervals.

The covariance-based standard errors returned by the current least-squares fit should therefore be interpreted as preliminary local uncertainty estimates.

## Development roadmap

Planned development stages include:

1. additional synthetic IRF models, automatic and experimental/measured IRF support;
2. reconvolution fitting;
3. Poisson-likelihood lifetime estimation;
4. automated preprocessing and initial guesses;
5. physically interpretable feature extraction;
6. machine-learning lifetime estimation;
7. benchmarking classical and data-driven methods;
8. robustness studies under model mismatch; 
9. a Purcell-enhanced lifetime-sensing demonstration;
10. support for fitting user-provided experimental TCSPC data;
11. tools for preparing experimental and synthetic datasets for machine-learning applications;
12. addition of deep learning models (e.g., CNNs, autoencoders) trained for photon-efficient neural inference and reconstruction from ultra-low photon counts (sparse data);
13. a graphical user interface.

## Reproducibility

Synthetic photon-count data are generated using NumPy random-number generators. Supplying a fixed `random_seed` makes a simulation reproducible.

For example:

```python
expected_1, measured_1 = simulate_monoexponential_decay(
    time=time_ns,
    amplitude=10_000.0,
    lifetime=2.5,
    background=5.0,
    random_seed=42,
)

expected_2, measured_2 = simulate_monoexponential_decay(
    time=time_ns,
    amplitude=10_000.0,
    lifetime=2.5,
    background=5.0,
    random_seed=42,
)

assert np.array_equal(measured_1, measured_2)
```

## Project status

This repository is under active development.

The present codebase is an educational and scientific-software prototype. It is not yet intended as a validated replacement for established experimental TCSPC-analysis software.

## Citation

If you use this toolkit in scientific work, please cite:

> Morozov Y., *TCSPC Lifetime Toolkit*, version 0.2.1,
> https://github.com/moryev/tcspc-lifetime-toolkit

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

## Contact

If you are interested in the project, have questions, or would like to contribute, please feel free to contact me at:

**Yevhenii Morozov**  
Email: [morozov.ye.m@gmail.com](mailto:morozov.ye.m@gmail.com)
