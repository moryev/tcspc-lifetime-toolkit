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

The current implementation provides a transparent classical baseline that will later be compared with data-driven lifetime estimators.

## Current functionality

The current version supports:

* mono-exponential decay modelling;
* constant background counts;
* Poisson sampling of photon-count histograms;
* reproducible simulations using a random seed;
* nonlinear least-squares lifetime fitting;
* covariance-based parameter standard errors;
* fitted-signal reconstruction;
* raw residual calculation;
* Poisson-scaled Pearson residuals;
* absolute and relative lifetime errors;
* CSV export of simulated and evaluated data;
* Jupyter notebooks demonstrating the complete workflow.

## Current scientific assumptions

The expected decay is represented by

$$
I(t) = A \exp\left(-\frac{t}{\tau}\right) + B,
$$

where:

* (A) is the decay amplitude;
* ($\tau$) is the lifetime;
* (B) is a constant background level.

For each time bin ($i$), the measured photon count is sampled according to

$$
N_i \sim \mathrm{Poisson}(\lambda_i),
$$

where ($\lambda_i$) is the expected count predicted by the decay model.

The initial fitting baseline uses unweighted nonlinear least squares. Poisson-scaled residuals are calculated as

$$
r_i =
\frac{N_i-\widehat{N}_i}
{\sqrt{\widehat{N}_i}},
$$

where ($N_i$) is the measured count and ($\widehat{N}_i$) is the fitted count.

**Version 0.1 initially treats TCSPC curves as Poisson-sampled mono-exponential decays without IRF convolution. Instrument-response modelling is added in the next development stage.**

## Installation

### Requirements

* Python 3.11 or newer
* pip
* Git

### Clone the repository

Clone the repository and enter the project directory:

```bash
git clone https://github.com/YevhM/tcspc-lifetime-toolkit.git
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

from tcspc_toolkit.simulation import simulate_ideal_decay

time_ns = np.linspace(
    start=0.0,
    stop=20.0,
    num=512,
)

expected_counts, measured_counts = simulate_ideal_decay(
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

## Repository structure

```text
tcspc-lifetime-toolkit/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── examples/
│       └── ideal_tcspc_decay.csv
│   └── generated/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_tcspc_simulation.ipynb
│   └── 02_classical_lifetime_fitting.ipynb
│
├── src/
│   └── tcspc_toolkit/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── models.py
│       ├── simulation.py
│       ├── fitting.py
│       └── evaluation.py
│
└── tests/
    ├── test_models.py
    ├── test_simulation.py
    └── test_fitting.py
```

The modules currently have the following responsibilities:

* `models.py`: mathematical decay models;
* `simulation.py`: expected-curve generation and Poisson sampling;
* `fitting.py`: nonlinear parameter estimation and structured fit results;
* `evaluation.py`: fitted signals, residuals, and lifetime-error metrics;
* `cli.py`: command-line tools for simulating and fitting TCSPC data;
* `__main__.py`: package entry point for python -m tcspc_toolkit;
* `data/examples/`: small example datasets tracked by Git;
* `data/generated/`: generated outputs that are not normally tracked by Git;
* `notebooks/`: documented simulation and fitting workflows;
* `tests/`: automated verification of physical, numerical, and package behaviour.
* `pyproject.toml`: package metadata, dependencies, build configuration, and command-line entry points.

## Current limitations

The current implementation is intentionally simplified.

It does not yet include:

* an instrument response function;
* IRF convolution or reconvolution fitting;
* temporal alignment between the IRF and decay;
* Poisson maximum-likelihood fitting;
* weighted least-squares fitting;
* bi-exponential or multi-exponential decays;
* pile-up effects;
* detector dead time;
* afterpulsing;
* time-dependent background;
* experimental file-format import;
* automated initial-parameter estimation;
* machine-learning lifetime estimation;
* calibrated confidence or prediction intervals.

The covariance-based standard errors returned by the current least-squares fit should therefore be interpreted as preliminary local uncertainty estimates.

## Development roadmap

Planned development stages include:

1. Gaussian and measured IRF support;
2. IRF convolution and reconvolution fitting;
3. Poisson-likelihood lifetime estimation;
4. automated preprocessing and initial guesses;
5. generation of large synthetic TCSPC datasets;
6. physically interpretable feature extraction;
7. machine-learning lifetime estimation;
8. benchmarking classical and data-driven methods;
9. robustness studies under model mismatch;
10. a Purcell-enhanced lifetime-sensing demonstration.

## Reproducibility

Synthetic photon-count data are generated using NumPy random-number generators. Supplying a fixed `random_seed` makes a simulation reproducible.

For example:

```python
expected_1, measured_1 = simulate_ideal_decay(
    time=time_ns,
    amplitude=10_000.0,
    lifetime=2.5,
    background=5.0,
    random_seed=42,
)

expected_2, measured_2 = simulate_ideal_decay(
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

