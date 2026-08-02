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
