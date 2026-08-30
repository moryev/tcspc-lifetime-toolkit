# TCSPC Lifetime Toolkit

A scientific Python toolkit for simulating, fitting, and evaluating time-correlated single-photon counting decay curves.

The project is being developed as an end-to-end framework for classical and machine-learning-based fluorescence-lifetime estimation. Its long-term goal is to support realistic TCSPC simulation and data processing, reproducible benchmarking of lifetime estimators, and lifetime-based sensing demonstrations.

## Scientific motivation

Time-correlated single-photon counting (TCSPC) is widely used to measure fluorescence and excited-state lifetimes. A measured TCSPC histogram contains photon counts distributed over time bins following an excitation event.

The measured histogram is generally not identical to the underlying fluorescence decay. The finite temporal response of the measurement system broadens the signal through the instrument response function (IRF), while photon detection introduces stochastic counting fluctuations. For an ideal photon-counting measurement, the observed count in each time bin can be modelled as a Poisson-distributed random variable.

Accurate lifetime estimation therefore requires separating the underlying fluorescence dynamics from both instrumental broadening and photon-counting statistics. This becomes particularly important when the fluorescence lifetime is comparable to or shorter than the IRF width, or when only a limited number of photons are available.

This project separates TCSPC analysis into reusable physical, statistical, and data-analysis stages:

1. define a physical fluorescence-decay model;
2. define and normalize the instrument response function;
3. construct the expected TCSPC histogram through numerical convolution and detector-background addition;
4. generate Poisson-sampled photon-count measurements;
5. validate measured histograms and construct analysis-dependent derived representations where appropriate;
6. estimate physical parameters using ordinary or IRF-aware reconvolution fitting;
7. evaluate fitted models using lifetime errors, raw residuals, and Poisson-aware residual diagnostics;
8. generate controlled synthetic datasets for machine-learning experiments;
9. construct reproducible train/test splits shared across estimator and representation benchmarks;
10. evaluate statistical, physics-inspired, linear, and nonlinear lifetime estimators;
11. compare engineered features, normalized histograms, and PCA-compressed representations;
12. benchmark classical reconvolution and ML under controlled lifetime, photon-count, background, IRF-width, and IRF-shift regimes;
13. introduce controlled model mismatch and quantify estimator degradation;
14. benchmark inference runtime and throughput alongside predictive accuracy.

The current implementation provides a reproducible framework for both
classical and data-driven TCSPC lifetime estimation.

Classical inference includes ordinary mono-exponential least-squares fitting,
IRF-aware least-squares reconvolution, and Poisson maximum-likelihood
reconvolution with simultaneous estimation of amplitude, lifetime, detector
background, and temporal IRF shift.

The machine-learning workflow now includes physically inspired baselines,
scikit-learn regression pipelines, controlled representation benchmarks,
conditional evaluation across physical operating regimes, comparison with
classical reconvolution, controlled model-mismatch experiments, and inference
timing.

Synthetic benchmark datasets can vary fluorescence lifetime, signal photon
count, detector background, IRF width, and IRF temporal shift while retaining
explicit ground-truth metadata and reproducible train/test splits.

The central scientific objective is not simply to demonstrate that machine
learning can predict fluorescence lifetime, but to determine under which
photon-count, background, IRF, and model-mismatch conditions data-driven
estimation outperforms, matches, or complements physically explicit fitting.

## Current functionality

The current version supports:

* mono-, bi-, and multi-exponential decay modelling;
* constant detector background counts;
* Gaussian instrument response function generation;
* IRF normalization to unit temporal area;
* non-integer temporal IRF shifting;
* numerical convolution of fluorescence decays with IRFs;
* IRF-convolved TCSPC simulation with Poisson photon-count sampling;
* reproducible simulations using NumPy random-number generators;
* generation of independent and grouped synthetic mono-exponential datasets;
* structured storage of simulated curves and metadata;
* validation of raw TCSPC histograms, including dimensionality, finite values, non-negative integer-like photon counts, and approximately uniform time bins;
* explicit background estimation from user-selected histogram regions;
* background subtraction for derived visualization and analysis representations while preserving negative statistical fluctuations;
* discrete peak detection for raw photon-count histograms;
* temporal-coordinate alignment to the IRF peak without modifying or interpolating measured photon counts;
* time-window cropping using physical time coordinates;
* photon-count-preserving temporal rebinning by integer factors;
* total-count and peak-count normalization for visualization and machine-learning representations;
* physically interpretable feature extraction from raw TCSPC histograms, including intensity, photon-arrival moments, quantile timing, half-decay timing, tail, and early/late descriptors;
* stable engineered-feature schemas and batch construction of pandas feature tables;
* batch construction of total- or peak-normalized histogram representations;
* PCA compression of normalized TCSPC histograms;
* leakage-safe PCA workflows with fitting restricted to training data and separate transformation of training and test samples;
* explained-variance analysis for PCA representations;
* immutable simulation, preprocessing, and feature configuration dataclasses;
* reproducible factorial benchmark generation with controlled variation of fluorescence lifetime, signal photon count, detector background, IRF width, and IRF temporal shift;
* shared train-test splits preserving aligned engineered-feature, raw-histogram, target, and metadata representations;
* train/test support and level-balance diagnostics for controlled benchmark variables;
* constant-mean regression baselines;
* physics-inspired mean-arrival-time lifetime estimation;
* reusable scikit-learn pipelines for Ridge regression, Random Forest regression, and Histogram Gradient Boosting regression;
* unified regression evaluation using MAE, median absolute error, RMSE, relative error, and $R^2$;
* controlled representation benchmarking using the same samples, targets, and estimator family across engineered features, TOTAL-normalized histograms, and PCA-compressed histograms;
* photon-count ablation experiments that restore measured total counts to normalized histogram representations;
* batch classical reconvolution benchmarking with histogram-derived initial guesses;
* per-curve reconvolution diagnostics including fitted lifetime, optimizer success, fit validity, parameter-boundary hits, Poisson negative log-likelihood, Poisson deviance, and optimizer runtime;
* aggregate classical-fit summaries including fit success/failure rate, MAE, median absolute error, RMSE, and runtime statistics;
* standardized per-sample diagnostics shared by ML and classical estimators;
* reusable benchmark plots for true-versus-predicted lifetime, signed and absolute error distributions, error versus physical conditions, and paired estimator comparison;
* conditional performance analysis across lifetime, photon-count, background, IRF-width, and IRF-misalignment regimes;
* regime-level summaries including MAE, median absolute error, bias, 90th- and 95th-percentile absolute errors, and failure rate;
* matched weakly bi-exponential model-mismatch simulation while preserving primary lifetime and nuisance conditions sample-by-sample;
* direct comparison of ML distribution shift and classical forward-model mismatch;
* mismatch summaries reporting in-distribution and mismatch MAE, MAE degradation, bias changes, and failure rates;
* repeated batch inference timing for mean-arrival and machine-learning estimators;
* reuse of recorded per-curve reconvolution optimization runtimes for computational-cost comparison;
* estimator-throughput and accuracy-versus-runtime benchmarking.
* JSON serialization and loading of configuration objects using `pathlib`;
* metadata-based selection of machine-learning targets;
* nonlinear least-squares mono-exponential lifetime fitting;
* IRF-aware mono-exponential reconvolution fitting;
* simultaneous reconvolution estimation of amplitude, lifetime, detector background, and temporal IRF shift;
* selectable least-squares and Poisson maximum-likelihood reconvolution objectives;
* Poisson negative log-likelihood evaluation;
* covariance-based parameter standard errors for ordinary least-squares mono-exponential fitting;
* fitted-signal reconstruction;
* raw residual calculation;
* Poisson-scaled Pearson residuals;
* Poisson deviance residuals;
* absolute and relative lifetime errors;
* baseline machine-learning lifetime estimation using normalized TCSPC histograms;
* random and group-aware train-test evaluation;
* data-leakage analysis for repeated noisy realizations;
* CSV export of simulated and evaluated data;
* Jupyter notebooks demonstrating classical fitting, realistic TCSPC simulation,
  reconvolution fitting, Poisson-aware validation, preprocessing, feature
  engineering, leakage-safe representations, classical-versus-ML benchmarking,
  conditional performance analysis, controlled model mismatch, and inference
  timing.

## Current scientific assumptions

The current realistic TCSPC workflow represents a measurement as a sequence of physical, instrumental, and statistical stages:

1. generate an ideal fluorescence decay;
2. generate and normalize an instrument response function (IRF);
3. optionally shift the IRF in time;
4. convolve the ideal fluorescence decay with the IRF;
5. scale the fluorescence contribution by its amplitude;
6. add the detector background;
7. sample the resulting expected counts using Poisson statistics.

For a mono-exponential fluorescence decay, the unit-amplitude decay shape is represented by

```math
I_\tau(t)
=
\exp\left(
-\frac{t}{\tau}
\right),
```

where $\tau$ is the fluorescence lifetime.

The instrument response function describes the temporal broadening introduced by the measurement system. In the current implementation, the IRF is modelled as a Gaussian function,

```math
\mathrm{IRF}(t)
=
C
\exp\left[
-\frac{(t-t_0)^2}{2\sigma^2}
\right],
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
\int
\mathrm{IRF}(t)\,dt
=
1.
```

A temporal shift $\Delta t$ can be applied to describe the relative alignment between the fluorescence decay and the instrument response,

```math
\mathrm{IRF}_{\Delta t}(t)
=
\mathrm{IRF}(t-\Delta t).
```

The instrument-broadened fluorescence signal is then calculated through convolution,

```math
[\mathrm{IRF}_{\Delta t} * I_\tau](t)
=
\int
\mathrm{IRF}_{\Delta t}(t-t')
I_\tau(t')
\,dt'.
```

Numerically, the convolution is evaluated on a uniform time grid. The discrete convolution includes the time-bin width $\Delta t_{\mathrm{bin}}$ so that the numerical sum approximates the continuous convolution integral.

The expected TCSPC photon-count curve is

```math
\mu(t)
=
A
[\mathrm{IRF}_{\Delta t} * I_\tau](t)
+
B,
```

where:

* $A$ is the fluorescence-signal amplitude;
* $\tau$ is the fluorescence lifetime;
* $B$ is the constant detector background;
* $\Delta t$ is the relative temporal shift of the IRF.

For each time bin $i$, the measured photon count is sampled according to

```math
N_i
\sim
\mathrm{Poisson}(\mu_i),
```

where $\mu_i$ is the expected photon count in that time bin.

The resulting forward model can therefore be summarized as

```math
I_\tau(t)
\longrightarrow
\mathrm{IRF}_{\Delta t}(t)
\longrightarrow
[\mathrm{IRF}_{\Delta t} * I_\tau](t)
\longrightarrow
\mu(t)
\longrightarrow
N_i.
```

Reconvolution fitting evaluates this forward model repeatedly while varying the fitted physical parameters. In the current mono-exponential implementation, the fitted parameter vector is

```math
\theta
=
(A,\tau,B,\Delta t).
```

Least-squares reconvolution estimates these parameters by minimizing ordinary residual errors between the measured histogram and the reconvolved model.

For Poisson maximum-likelihood reconvolution, the fitted parameters are estimated by minimizing the reduced Poisson negative log-likelihood,

```math
-\log L
=
\sum_i
\left[
\mu_i
-
N_i\log(\mu_i)
\right],
```

up to an additive term that does not depend on the fitted parameters.

Poisson deviance residuals are additionally available for statistically scaled model diagnostics across histogram regions with strongly different photon-count levels.

## Preprocessing philosophy

TCSPC preprocessing is analysis-dependent. The toolkit therefore provides small, composable preprocessing functions rather than enforcing a single universal preprocessing pipeline.

The current preprocessing utilities include:

- `validate_histogram()` for validating raw measured TCSPC histograms;
- `estimate_background()` for estimating a stationary background level from an explicitly selected bin interval;
- `subtract_background()` for constructing background-corrected derived representations;
- `detect_peak()` for locating the discrete maximum of a raw photon-count histogram;
- `align_to_irf()` for redefining the temporal origin relative to the IRF peak without modifying the measured counts;
- `crop_time_window()` for selecting a physical time interval;
- `rebin_histogram()` for combining neighboring bins while preserving the total photon count;
- `normalize_counts()` for total-count or peak-count normalization.

These operations are deliberately not combined into a single hard-coded `preprocess()` routine because different downstream analyses require different statistical treatment.

### Poisson reconvolution fitting

For Poisson maximum-likelihood fitting, the measured photon counts are retained as raw counts:

```text
raw photon counts
        ↓
validate histogram
        ↓
background estimate ──→ initial background guess
        ↓
IRF information ──────→ temporal-reference / shift information
        ↓
numerical initialization
        ↓
Poisson reconvolution fit
```

The detector background is included directly in the expected-count model,

```math
\mu(t)
=
A
[\mathrm{IRF}_{\Delta t} * I_\tau](t)
+
B,
```

rather than being subtracted from the observed counts before fitting.

Similarly, normalization is not applied to the raw observations before Poisson-likelihood fitting because normalization removes the absolute photon-count scale on which the count likelihood is defined.

The current numerical implementation can use a least-squares reconvolution result as an initialization for subsequent Poisson maximum-likelihood refinement. Least squares in this workflow acts as a numerical initialization step; the final statistical objective remains the Poisson likelihood.

### Visualization and machine-learning representations

For visualization, exploratory analysis, or machine-learning input preparation, a different sequence may be appropriate:

```text
raw photon counts
        ↓
validate histogram
        ↓
temporal alignment
        ↓
crop
        ↓
rebin
        ↓
normalize
        ↓
analysis- or ML-ready representation
```

Cropping changes the observation window, rebinning trades temporal resolution for increased counts per bin, and normalization removes absolute count-scale information. These transformations can therefore be useful for representation-oriented analyses without being statistically neutral.

Rebinning preserves the total number of photons within the rebinned region,

```math
\sum_i N_i^{(\mathrm{rebinned})}
=
\sum_i N_i^{(\mathrm{original})},
```

provided the selected number of bins is exactly divisible by the rebinning factor.

Cropping before reconvolution should be distinguished from selecting a fitting window after constructing the forward model. Premature truncation of the IRF or fluorescence signal can introduce convolution edge effects.

The guiding design principle is therefore:


> **Preprocessing choices should follow the scientific question and statistical model
> rather than being imposed by the software architecture.**


## Machine-learning representations

The toolkit supports three complementary representations of TCSPC
measurements for machine-learning analysis.

### Engineered physical features

`extract_features()` and `extract_feature_table()` convert raw measured
histograms into a stable set of physically interpretable descriptors,
including photon-count, temporal-moment, quantile, half-decay, tail, and
early/late features.

Feature extraction does not silently subtract background, normalize, crop,
align, or rebin the histogram. Any required preprocessing must be performed
explicitly so that the scientific meaning of each feature remains traceable.

### Normalized TCSPC histograms

`normalize_histogram_batch()` preserves the complete binned temporal shape
while normalizing each histogram independently.

Total-count normalization removes absolute photon-count information:

```math
x_i
=
\frac{N_i}{\sum_j N_j}.
```

Absolute photon counts and normalized decay shape therefore represent
different physical information.

### PCA-compressed histograms

`fit_pca_representation()` and `transform_pca_representation()` provide a
lower-dimensional representation of normalized histogram bins.

PCA captures directions of large histogram variance but does not use lifetime
labels and is not itself a lifetime estimator. Components explaining large
variance are therefore not necessarily those carrying the most lifetime
information.

PCA and any other learned preprocessing transformation must be fitted using
training data only. Test data must not contribute to fitted PCA components,
scalers, feature-selection rules, or other learned transformations.

The appropriate preprocessing depends on the intended analysis. Feature
engineering and machine-learning representations may use normalization or
derived temporal descriptors, whereas Poisson reconvolution fitting should
generally retain the original photon counts and background statistics.

## Current implementation status
**Version 0.6 extends the toolkit from machine-learning representation
construction to a reproducible classical-versus-data-driven TCSPC lifetime
benchmarking framework.
The toolkit now supports controlled factorial benchmark datasets spanning
fluorescence lifetime, signal photon count, detector background, Gaussian IRF
width, and temporal IRF shift. A common reproducible train/test split can be
reused across statistical baselines, the physics-inspired mean-arrival-time
estimator, Ridge regression, Random Forest regression, Histogram Gradient
Boosting regression, representation benchmarks, and classical
mono-exponential reconvolution fitting.
Machine-learning inputs can be represented as physically engineered features,
TOTAL-normalized histogram bins, or leakage-safe PCA-compressed histograms.
Controlled representation benchmarks isolate the effect of representation
while keeping samples, targets, and estimator families fixed, and photon-count
ablation experiments test the information removed by TOTAL normalization.
Classical reconvolution benchmarking now records fitted lifetime, optimizer
success and validity, parameter-boundary hits, Poisson likelihood/deviance
diagnostics, and per-curve optimization runtime. ML and classical results are
converted into a common diagnostic representation so that performance can be
analysed conditionally across lifetime, photon-count, background, IRF-width,
and IRF-misalignment regimes using MAE, median absolute error, bias,
upper-tail error quantiles, and failure rate.
Version 0.6 also introduces controlled model-mismatch evaluation through
matched weakly bi-exponential test curves. ML estimators remain trained on
mono-exponential data while classical reconvolution continues to use a
mono-exponential forward model, allowing distribution shift and physical
model misspecification to be compared directly.
Estimator-only computational benchmarking reports repeated batch inference
time and throughput for the physics-inspired and ML estimators together with
the recorded optimization cost of classical reconvolution.
The present implementation still assumes a uniform time grid and Gaussian IRF
model. Classical reconvolution currently treats the IRF shape and width as
fixed during an individual fit, although benchmark datasets can contain
controlled IRF-width variation. Experimental IRF loading, fitted IRF width,
multi-exponential reconvolution fitting, detector effects such as pile-up,
dead time and afterpulsing, calibrated uncertainty intervals, experimental
file-format import, and deep-learning estimators remain future extensions.**

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

### `08_naive_vs_reconvolution_fitting.ipynb`

Demonstrates:

* construction of a uniform TCSPC time axis for lifetime-fitting experiments and inspection of the corresponding time-bin width;
* generation and normalization of a fixed Gaussian instrument response function through `generate_gaussian_irf()` and `normalize_irf()`;
* verification that the normalized IRF has unit integrated area before it is used in convolution and reconvolution fitting;
* definition of representative fluorescence lifetimes spanning regimes where the lifetime is much longer than, comparable to, and shorter than the IRF FWHM;
* generation of ideal mono-exponential fluorescence decays through `monoexponential_decay()`;
* numerical convolution of the ideal fluorescence signal with the fixed IRF through `convolve_decay_with_irf()`;
* adjustment of the decay amplitude so that different lifetime conditions contain approximately the same total number of expected signal photons;
* explicit addition of detector background after convolution according to the physical measurement model $`\mu(t) = A[\mathrm{IRF} * I_\tau](t) + B`$;
* generation of Poisson-sampled TCSPC histograms through `sample_photon_counts()` using reproducible NumPy random-number generators;
* comparison of the same simulated TCSPC histogram with two competing lifetime-fitting approaches: a naive mono-exponential fit and an IRF-aware reconvolution fit;
* restriction of the naive exponential fit to the measured decay region beginning at the histogram peak so that the simple model is not trivially penalized by the IRF-generated leading edge;
* construction of data-driven initial guesses for amplitude, lifetime, and background without using the known true lifetime;
* use of the naive-fit result as a practical initialization for the reconvolution fit while keeping the known IRF shape and width fixed;
* reconvolution fitting through `fit_monoexponential_reconvolution()` with simultaneous estimation of amplitude, fluorescence lifetime, detector background, and temporal IRF shift;
* visual comparison of measured photon counts, the true expected reconvolved signal, the naive fitted decay, and the reconvolution fitted curve;
* calculation of signed relative lifetime errors through $`(\tau_{\mathrm{fit}}-\tau_{\mathrm{true}})/\tau_{\mathrm{true}}`$ to distinguish lifetime overestimation from underestimation;
* comparison of recovered lifetimes for representative cases including $`\tau \gg \mathrm{FWHM}_{\mathrm{IRF}}`$, $`\tau \approx 3\,\mathrm{FWHM}_{\mathrm{IRF}}`$, $`\tau \approx \mathrm{FWHM}_{\mathrm{IRF}}`$, and $`\tau < \mathrm{FWHM}_{\mathrm{IRF}}`$;
* systematic lifetime sweep over a broad range of $`\tau_{\mathrm{true}}/\mathrm{FWHM}_{\mathrm{IRF}}`$ values to quantify when neglecting the IRF becomes scientifically significant;
* visualization of relative lifetime bias as a function of $`\tau_{\mathrm{true}}/\mathrm{FWHM}_{\mathrm{IRF}}`$ on a logarithmic horizontal axis;
* explicit marking of the physically important transition $`\tau_{\mathrm{true}} = \mathrm{FWHM}_{\mathrm{IRF}}`$ in the lifetime-bias figure;
* demonstration that naive exponential fitting increasingly overestimates short lifetimes as the fluorescence lifetime approaches and falls below the IRF width;
* demonstration that reconvolution fitting substantially suppresses the systematic lifetime bias introduced by ignoring the instrument response;
* repeated Poisson simulation of each lifetime condition across multiple independent realizations to separate systematic model bias from statistical photon-counting variability;
* calculation of median relative lifetime error and 16th–84th percentile intervals for both fitting approaches;
* comparison of statistical spread between naive and reconvolution lifetime estimates across the full lifetime-to-IRF-width range;
* verification of numerical fit success rates for both fitting methods across all simulated lifetime regimes and Poisson realizations;
* demonstration that successful optimizer convergence alone is not sufficient for physical correctness, since the naive model can converge reliably while remaining systematically biased;
* analysis of the distinction between model-induced bias and increasing parameter uncertainty when fluorescence lifetimes become shorter than the instrument response;
* presentation of a controlled scientific benchmark showing why reconvolution becomes necessary when $`\tau_{\mathrm{true}}`$ approaches $`\mathrm{FWHM}_{\mathrm{IRF}}`$;
* discussion of the remaining limitation that both fitting approaches still use least-squares objectives despite the underlying Poisson photon-counting statistics, motivating the next development step toward Poisson-aware fitting.

### `09_poisson_reconvolution_fitting_and_validation.ipynb`

Demonstrates:

* construction of a reproducible synthetic TCSPC measurement using a normalized Gaussian instrument response function, mono-exponential fluorescence decay, temporal IRF shift, detector background, and Poisson photon-count sampling;
* explicit construction of the physical forward model $`\mu(t) = A[\mathrm{IRF}_{\Delta t} * I_\tau](t) + B`$ with detector background added after convolution;
* validation of the imposed temporal IRF shift through comparison of reference and shifted IRF peak positions;
* generation of a challenging short-lifetime TCSPC histogram in a regime where the fluorescence lifetime is comparable to the IRF width;
* comparison of three lifetime-estimation approaches applied to the same measured histogram: naive unconvolved least squares, least-squares reconvolution, and Poisson maximum-likelihood reconvolution;
* restriction of the naive exponential fit to the post-peak decay region while reconvolution methods fit the complete TCSPC histogram;
* simultaneous reconvolution estimation of fluorescence amplitude, lifetime, detector background, and temporal IRF shift while keeping the IRF shape and width fixed;
* use of the least-squares reconvolution solution as a practical initialization for Poisson maximum-likelihood refinement;
* tabulated comparison of true and recovered physical parameters and signed relative lifetime errors;
* visualization of measured photon counts, the true expected TCSPC curve, naive exponential fit, least-squares reconvolution fit, and Poisson-MLE reconvolution fit on linear and logarithmic scales;
* calculation and visualization of raw residuals for all fitting approaches;
* calculation of Poisson deviance residuals through `calculate_poisson_deviance_residuals()` to account for the count-dependent statistical scale of photon-counting data;
* demonstration that the naive exponential model leaves strong systematic residual structure while reconvolution produces approximately unstructured residual fluctuations around zero;
* direct comparison of Poisson negative log-likelihood values for least-squares and Poisson-MLE reconvolution solutions using `poisson_negative_log_likelihood()`;
* demonstration that least-squares and Poisson-MLE reconvolution give nearly identical parameter estimates in a high-count regime while the Poisson solution achieves the lower Poisson negative log-likelihood;
* systematic comparison of naive least squares, least-squares reconvolution, and Poisson-MLE reconvolution across a broad range of $`\tau_{\mathrm{true}}/\mathrm{FWHM}_{\mathrm{IRF}}`$ values;
* demonstration that ignoring the IRF produces rapidly increasing lifetime bias when the fluorescence lifetime approaches or falls below the IRF width;
* demonstration that both reconvolution approaches substantially suppress the systematic model bias across the lifetime-to-IRF-width sweep;
* Monte Carlo investigation of lifetime recovery at the challenging condition $`\tau/\mathrm{FWHM}_{\mathrm{IRF}} = 0.5`$ over expected fluorescence signal levels from 100 to 100,000 photons;
* scaling of detector background with signal photon count so that the approximate signal-to-background ratio remains fixed during the Monte Carlo photon-count sweep;
* repeated Poisson simulation and reconvolution fitting to separate estimator bias from statistical lifetime uncertainty;
* comparison of least-squares and Poisson-MLE reconvolution using lifetime-error bias, median absolute error, RMSE, standard deviation, and error-distribution boxplots;
* demonstration that Poisson maximum likelihood provides lower lifetime-error RMSE in low- and intermediate-count regimes while least-squares and Poisson fitting converge toward similar performance at high photon counts;
* demonstration that successful numerical optimization does not guarantee precise parameter recovery when photon statistics contain insufficient information;
* analysis of the distinction between systematic forward-model error, statistical estimator efficiency, and the fundamental information loss associated with finite IRF width and limited photon counts;
* validation of a complete TCSPC inference workflow from physical decay modelling and IRF reconvolution to Poisson sampling, parameter estimation, and Poisson-aware residual diagnostics.

### `10_preprocessing_tcspc_histograms.ipynb`

Demonstrates:

* generation of a realistic IRF-convolved, Poisson-sampled TCSPC measurement;
* validation of raw photon-count histograms and inspection of bin width and total photon count;
* explicit background estimation from a known background-dominated time region;
* comparison of the estimated and true detector-background levels;
* background subtraction and interpretation of negative background-corrected bins as statistical fluctuations;
* preservation of the original raw histogram for Poisson-likelihood fitting;
* peak detection using a simulated measured IRF photon-count histogram;
* alignment of the temporal coordinate to the continuous IRF peak without modifying measured counts;
* selection of scientifically meaningful time windows using physical time coordinates;
* comparison of temporal rebinning factors and numerical verification of photon-count conservation;
* analysis of the trade-off between temporal resolution and relative Poisson fluctuations;
* comparison of total-count and peak-count normalization;
* demonstration that normalization is useful for shape-based representations but removes the absolute count scale required by the present Poisson likelihood;
* construction of a raw-count statistical fitting workflow using background information for initialization;
* least-squares initialization followed by Poisson maximum-likelihood reconvolution refinement;
* construction of a separate crop–rebin–normalize workflow for machine-learning representations;
* demonstration that scientifically appropriate preprocessing depends on the downstream analysis rather than on a single universal pipeline.

### `11_feature_engineering.ipynb`

Demonstrates:

* generation of a structured IRF-convolved TCSPC dataset with controlled variation of fluorescence lifetime, signal-photon count, detector background, IRF width, and IRF temporal shift;
* extraction of the complete stable engineered-feature schema from raw photon-count histograms;
* physical interpretation of intensity, photon-arrival moment, quantile, half-decay, tail, and early/late features;
* exploratory analysis of feature–lifetime relationships across varying nuisance conditions;
* controlled investigation of feature sensitivity to photon count, background, IRF width, and IRF temporal shift;
* construction of total-normalized histogram representations;
* explicit demonstration that total normalization removes pure absolute count scaling;
* construction of a common stratified train–test split shared by all representations;
* leakage-safe PCA fitting using training histograms only;
* analysis of per-component and cumulative explained variance;
* visualization of the first PCA component vectors;
* visualization of normalized histograms in PC1–PC2 space colored by true lifetime;
* construction of aligned engineered-feature, normalized-histogram, and PCA-compressed input matrices;
* comparison of the scientific advantages and limitations of all three machine-learning representations.

### `12_ml_benchmarking.ipynb`

Demonstrates:

* construction of a reproducible 810-curve factorial TCSPC benchmark spanning fluorescence lifetime, signal photon count, detector background, IRF width, and IRF temporal shift;
* construction of a fixed 80/20 train-test split with explicit support and parameter-balance checks;
* adaptation of engineered-feature windows to remain valid across a broad 0.5–6 ns lifetime range;
* evaluation of constant-mean and physics-inspired mean-arrival-time baselines;
* training and evaluation of Ridge, Random Forest, and Histogram Gradient Boosting regressors using the same engineered-feature representation;
* comparison of engineered features, TOTAL-normalized 400-bin histograms, and 10-component PCA representations under controlled Ridge and nonlinear-model benchmarks;
* analysis of PCA explained variance and the distinction between total histogram variance and lifetime-predictive information;
* photon-count ablation by restoring measured total counts to normalized and PCA representations;
* Poisson reconvolution benchmarking using a nominal Gaussian IRF with fitted amplitude, lifetime, background, and temporal shift;
* direct aggregate and paired comparison of classical reconvolution with engineered-feature ML;
* conditional analysis across short/medium/long lifetime, low/medium/high photon count, low/medium/high background, narrow/medium/broad IRF width, and small/medium/large IRF-misalignment regimes;
* reporting of MAE, median absolute error, bias, 90th- and 95th-percentile absolute errors, failure rates, and classical parameter-boundary diagnostics;
* demonstration that the preferred estimator changes across physical operating regimes rather than following a universal global ordering;
* generation of matched weakly bi-exponential test curves with a 10% slow secondary component while preserving the original primary lifetime and nuisance conditions;
* comparison of ML distribution shift with classical mono-exponential forward-model mismatch;
* analysis of mismatch-induced MAE degradation, lifetime bias, fit validity, and per-curve error changes;
* repeated batch inference timing for mean-arrival, Ridge, Random Forest, and Histogram Gradient Boosting estimators;
* comparison with recorded per-curve classical reconvolution optimization runtimes;
* analysis of estimator throughput and the accuracy-versus-computational-cost trade-off;
* final synthesis of when data-driven TCSPC lifetime estimation outperforms, matches, or complements classical reconvolution;
* explicit discussion of benchmark scope, interpolation limits, model mismatch, and future experimental validation requirements.

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
│   ├── 08_naive_vs_reconvolution_fitting.ipynb
│   ├── 09_poisson_reconvolution_fitting_and_validation.ipynb
│   ├── 10_preprocessing_tcspc_histograms.ipynb
│   ├── 11_feature_engineering.ipynb
│   └── 12_ml_benchmarking.ipynb
│
├── src/
│   └── tcspc_toolkit/
│       ├── __init__.py
│       ├── __main__.py
│       ├── baselines.py
│       ├── benchmark_plots.py
│       ├── classical_evaluation.py
│       ├── cli.py
│       ├── conditional_evaluation.py
│       ├── config.py
│       ├── convolution.py
│       ├── datasets.py
│       ├── evaluation.py
│       ├── exceptions.py
│       ├── features.py
│       ├── fitting.py
│       ├── irf.py
│       ├── mismatch_evaluation.py
│       ├── ml_evaluation.py
│       ├── ml_models.py
│       ├── models.py
│       ├── preprocessing.py
│       ├── representations.py
│       ├── simulation.py
│       └── timing_evaluation.py
│
└── tests/
    ├── conftest.py
    ├── test_baselines.py
    ├── test_benchmark_plots.py
    ├── test_classical_evaluation.py
    ├── test_conditional_evaluation.py
    ├── test_config.py
    ├── test_convolution.py
    ├── test_datasets.py
    ├── test_evaluation.py
    ├── test_feature_integration.py
    ├── test_features.py
    ├── test_fitting.py
    ├── test_irf.py
    ├── test_mismatch_evaluation.py
    ├── test_ml_evaluation.py
    ├── test_ml_models.py
    ├── test_models.py
    ├── test_preprocessing.py
    ├── test_preprocessing_integration.py
    ├── test_representations.py
    ├── test_simulation.py
    └── test_timing_evaluation.py
```

The modules currently have the following responsibilities:

* `__init__.py`: package initialization and definition of the public package interface;
* `__main__.py`: package entry point for python -m tcspc_toolkit;
* `baselines.py`: statistical and physics-inspired lifetime-estimation baselines, including constant-mean and mean-arrival-time estimators;
* `benchmark_plots.py`: reusable visualization utilities for prediction accuracy, error distributions, physical-condition dependence, and paired estimator comparisons;
* `classical_evaluation.py`: batch mono-exponential reconvolution benchmarking, histogram-derived initialization, fit-validity and boundary diagnostics, Poisson fit statistics, error metrics, and runtime summaries;
* `cli.py`: command-line tools for simulating and fitting TCSPC data;
* `conditional_evaluation.py`: standardized ML/classical prediction diagnostics, numeric regime assignment, and conditional performance summaries across benchmark operating conditions;
* `config.py`: immutable configuration dataclasses, normalization-mode definitions, and JSON serialization/loading utilities for reproducible simulation and preprocessing workflows;
* `convolution.py`: numerical convolution and temporal-grid alignment of ideal decay curves with instrument-response functions, including time-bin scaling and measurement-window truncation;
* `datasets.py`: synthetic datasets generation for the consequent ML baseline;
* `evaluation.py`: fitted signals, residuals, and lifetime-error metrics;
* `exceptions.py`: package-specific exception hierarchy for representing domain-level TCSPC validation and processing errors;
* `features.py`: extraction of physically interpretable TCSPC histogram features, including photon-count descriptors, photon-arrival moments, quantile times, half-decay timing, tail characteristics, and early/late count relationships; also defines the stable engineered-feature schema and batch feature-table construction;
* `fitting.py`: nonlinear parameter estimation and structured fit results;
* `irf.py`: generation and manipulation of instrument response functions, including Gaussian IRF construction, normalization, temporal shifting, and related validation;
* `mismatch_evaluation.py`: matched bi-exponential model-mismatch dataset construction and in-distribution versus mismatch evaluation for ML and classical estimators;
* `ml_evaluation.py`: reproducible benchmark-dataset construction and splitting, regression metrics, baseline and estimator evaluation, histogram/PCA representation construction, representation benchmarks, photon-count ablation, and split-coverage diagnostics;
* `ml_models.py`: reusable scikit-learn pipelines for Ridge, Random Forest, and Histogram Gradient Boosting lifetime regression;
* `models.py`: mathematical decay models;
* `preprocessing.py`: composable preprocessing utilities for raw TCSPC histograms, including histogram validation, background estimation and subtraction, peak detection, IRF-relative temporal alignment, time-window cropping, photon-count-preserving rebinning, and analysis-dependent count normalization;
* `representations.py`: construction of machine-learning representations from TCSPC histograms, including batch histogram normalization, leakage-safe PCA fitting and transformation, and cumulative explained-variance analysis;
* `simulation.py`: expected-curve generation and Poisson sampling;
* `timing_evaluation.py`: repeated batch inference timing, reconvolution-runtime summaries, throughput calculation, and computational-cost comparison;
* `data/examples/`: small example datasets tracked by Git;
* `data/generated/`: generated outputs that are not normally tracked by Git;
* `notebooks/`: documented analysis workflows;
* `tests/`: automated verification of physical, numerical, and package behaviour;
* `pyproject.toml`: package metadata, dependencies, build configuration, and command-line entry points.

## Current limitations

The current implementation is intentionally simplified.

It does not yet include:

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

1. additional synthetic IRF models and experimental/measured IRF loading and calibration;
2. automated and noise-aware preprocessing and initial-guess strategies;
3. expanded out-of-distribution benchmarks, including unseen lifetime ranges, leave-one-IRF-out evaluation, and cross-instrument tests;
4. additional model-mismatch scenarios including asymmetric IRFs, structured background, pile-up, dead time, and afterpulsing;
5. bi- and multi-exponential reconvolution fitting and corresponding classical benchmarks;
6. calibrated uncertainty estimation and prediction intervals for classical and machine-learning estimators;
7. a Purcell-enhanced lifetime-sensing demonstration;
8. support for fitting user-provided experimental TCSPC data and standard experimental file formats;
9. synthetic-to-real validation using experimental reference measurements;
10. deep-learning models such as MLPs, 1D CNNs, autoencoders, and photon-efficient neural estimators;
11. interoperability with established TCSPC/FLIM analysis libraries where scientifically useful;
12. profiling-driven CPU/GPU acceleration for large-scale inference;
13. a graphical or interactive benchmark interface;
14. an agentic AI assistant built on top of the validated scientific toolkit, using LLM tool/function calling to orchestrate deterministic preprocessing, fitting, benchmarking, and reporting functions.

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

> Morozov Y., *TCSPC Lifetime Toolkit*, version 0.6.0,
> https://github.com/moryev/tcspc-lifetime-toolkit

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

## Contact

If you are interested in the project, have questions, or would like to contribute, please feel free to contact me at:

**Yevhenii Morozov**  
Email: [morozov.ye.m@gmail.com](mailto:morozov.ye.m@gmail.com)
