# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-07-30

### Added

- mono-exponential TCSPC decay model;
- Poisson photon-count simulation;
- CSV input and output;
- SciPy-based lifetime fitting;
- covariance-based parameter uncertainties;
- residual and lifetime-error evaluation;
- command-line simulation and fitting;
- automated physical and numerical tests;
- synthetic dataset generator;
- preliminary machine-learning baselines;
- leakage-aware evaluation workflows;
- installation and usage documentation.

### Limitations

- no IRF convolution;
- no reconvolution fitting;
- mono-exponential decay model only;
- simplified detector model;
- preliminary ML evaluation on synthetic data.


## [0.1.1] - 2026-08-02

### Added

* bi-exponential and multi-exponential decay models;
* Poisson-sampled bi-exponential and multi-exponential simulations;
* grouped mono-exponential synthetic dataset generation;
* repeated Poisson realizations for shared physical parameter groups;
* `parameter_group` and `realization_id` metadata for leakage-aware evaluation;
* metadata-based machine-learning target selection through `SyntheticDataset.get_targets()`;
* expanded tests for grouped datasets, multi-component decay models, and simulation reproducibility.

### Changed

* generalized `SyntheticDataset` so that machine-learning targets are selected from metadata instead of being stored as a fixed target array;
* renamed the mono-exponential dataset generator to `generate_monoexponential_dataset()`;
* clarified the separation between deterministic decay models and stochastic simulation workflows;
* extended the public API for future bi-exponential, multi-exponential, and grouped-data workflows;
* updated documentation to reflect the implemented machine-learning and leakage-aware evaluation functionality.

### Notes

* Notebook 05 retains its original standalone grouped-generation workflow;
* the new grouped dataset generator is tested in Notebook 06 and available for future notebooks and package-based evaluation workflows;
* IRF generation, convolution, and reconvolution fitting remain planned for the next feature release.

### Limitations

* no IRF convolution;
* no reconvolution fitting;
* fitting currently focuses on the mono-exponential model;
* grouped dataset generation currently supports mono-exponential decays;
* simplified detector model;
* machine-learning evaluation remains preliminary and is based on synthetic data.


## [0.2.0] - 2026-08-08

### Added

* Gaussian instrument response function generation;
* IRF normalization to unit temporal area;
* non-integer temporal IRF shifting through interpolation;
* numerical convolution of fluorescence decays with instrument response functions;
* explicit time-bin-width correction in numerical convolution;
* convolution output alignment with the original TCSPC time axis;
* automated physical and numerical tests for IRF normalization, convolution dimensions, narrow-IRF behaviour, broad-IRF distortion, temporal shifting, boundary truncation, and time-grid consistency;
* realistic IRF-convolved TCSPC simulation workflow;
* Notebook 07 demonstrating Gaussian IRF generation, normalization, shifting, convolution, background addition, and Poisson sampling.

### Changed

* introduced `convolution.py` to separate numerical convolution from instrument-response-function generation and manipulation;
* extended the package architecture to support instrument-response-aware TCSPC workflows;
* clarified the separation between deterministic decay models, instrument-response modelling, convolution, and stochastic photon-count sampling;
* updated the public API to expose Gaussian IRF generation, normalization, temporal shifting, and decay–IRF convolution;
* updated the README with the new module structure, capabilities, workflow examples, and scientific limitations;
* updated the documented TCSPC forward model from ideal decay simulation toward IRF-convolved expected photon-count curves.

### Notes

* IRF-convolved simulations are constructed explicitly from decay generation, IRF generation, normalization, optional temporal shifting, numerical convolution, background addition, and Poisson sampling;
* convolution is implemented as a separate operation rather than through a dedicated `simulate_irf_convolved_decay()` convenience function;
* IRF shifting preserves boundary truncation and does not automatically renormalize signal area lost outside the observed time window;
* Notebook 07 provides the reference workflow for the new IRF-convolved simulation functionality.

### Limitations

* no experimental IRF loading;
* no full reconvolution fitting;
* Gaussian, time-invariant IRFs only;
* convolution currently assumes a uniform time grid;
* fitting currently focuses on the mono-exponential model;
* grouped dataset generation currently supports mono-exponential decays;
* detector effects such as pile-up and afterpulsing are not modelled;
* machine-learning evaluation remains preliminary and is based on synthetic data.
