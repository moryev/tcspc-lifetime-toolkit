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


## [0.2.1] - 2026-08-08

### Fixed

- corrected GitHub rendering of mathematical expressions in the README;
- replaced problematic display-math formatting with GitHub-compatible fenced math blocks.

### Notes

- no scientific or API behaviour changed in this patch release.


## [0.3.0] - 2026-08-13

### Added

* mono-exponential IRF reconvolution fitting with simultaneous estimation of amplitude, fluorescence lifetime, detector background, and temporal IRF shift;
* least-squares reconvolution fitting through `fit_monoexponential_reconvolution()`;
* Poisson negative log-likelihood through `poisson_negative_log_likelihood()` for photon-counting model evaluation and optimization;
* Poisson maximum-likelihood reconvolution fitting using the same physical forward model as least-squares reconvolution;
* selectable reconvolution fitting objectives through `objective="least_squares"` and `objective="poisson"`;
* structured `ReconvolutionFitResult` output containing fitted physical parameters, fitted TCSPC curve, and optimizer success status;
* Poisson deviance residuals through `calculate_poisson_deviance_residuals()` for statistically scaled residual diagnostics;
* automated tests for reconvolution-model behaviour, temporal shifting, parameter recovery, Poisson likelihood evaluation, Poisson reconvolution fitting, low-background numerical stability, and Poisson deviance residuals;
* Notebook 08 comparing naive unconvolved lifetime fitting with least-squares reconvolution across different lifetime-to-IRF-width regimes;
* Notebook 09 integrating naive least squares, least-squares reconvolution, Poisson-MLE reconvolution, raw and deviance residual diagnostics, likelihood comparison, lifetime/IRF-width sweeps, and Monte Carlo photon-count validation.

### Changed

* extended the fitting workflow from ideal mono-exponential least-squares fitting to direct IRF-aware reconvolution fitting;
* extended `fit_monoexponential_reconvolution()` with selectable least-squares and Poisson optimization objectives;
* extended the TCSPC analysis workflow to treat photon-counting statistics explicitly during parameter estimation and residual evaluation;
* kept IRF generation and manipulation in `irf.py`, numerical convolution in `convolution.py`, and parameter estimation in `fitting.py` to preserve separation of physical and numerical responsibilities;
* retained the IRF shape and width as fixed known quantities during reconvolution fitting while allowing temporal alignment to be fitted;
* updated scientific validation from single-curve parameter recovery toward systematic lifetime/IRF-width and photon-count benchmarking;
* strengthened Poisson parameter-recovery tests using scientifically meaningful relative lifetime and temporal-shift tolerances;
* updated the README and notebook documentation to reflect the implemented reconvolution and Poisson-likelihood workflows.

### Fixed

* improved numerical robustness of Poisson reconvolution fitting near zero detector background by using a strictly positive lower background bound for the Poisson optimizer;
* prevented finite-difference gradient evaluation from encountering invalid infinite-objective differences when low-count fits approach zero expected photon counts.

### Notes

* the largest improvement over naive fitting comes from using the correct IRF-aware forward model when the fluorescence lifetime becomes comparable to the IRF width;
* least-squares and Poisson-MLE reconvolution can produce nearly identical estimates in high-count measurements;
* Monte Carlo validation demonstrates an increasing statistical advantage for Poisson maximum likelihood as photon counts decrease;
* Poisson-MLE fitting is initialized effectively from a least-squares reconvolution solution in the validation workflow;
* successful optimizer convergence should not be interpreted as guaranteed parameter precision in extremely photon-limited measurements;
* Notebook 09 demonstrates stable reconvolution fitting down to very low synthetic photon counts while also showing the increasing statistical uncertainty in that regime.

### Limitations

* reconvolution fitting currently supports mono-exponential fluorescence decay only;
* the IRF shape and width are treated as known and fixed during fitting;
* experimental or measured IRF import and calibration are not yet implemented;
* IRF FWHM is not fitted because of potential identifiability with short fluorescence lifetimes;
* bi-exponential and multi-exponential reconvolution fitting are not yet implemented;
* weighted least-squares fitting is not yet implemented;
* Poisson optimization currently uses bounded numerical optimization without custom analytical gradients or parameter transformations;
* detector effects such as pile-up, dead time, and afterpulsing are not modelled;
* confidence intervals for reconvolution and Poisson-MLE parameters are not yet calibrated;
* machine-learning evaluation remains based primarily on synthetic data.