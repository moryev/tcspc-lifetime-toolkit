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


## [0.4.0] - 2026-08-17

### Added

* raw TCSPC histogram validation through `validate_histogram()`, including checks for one-dimensional arrays, matching lengths, finite values, non-negative integer-like photon counts, strictly increasing time coordinates, and approximately uniform time bins;
* stationary background estimation from explicitly selected histogram regions through `estimate_background()`;
* background subtraction through `subtract_background()` while preserving negative statistical fluctuations and leaving the input histogram unchanged;
* discrete peak detection for raw photon-count histograms through `detect_peak()`;
* IRF-relative temporal-coordinate alignment through `align_to_irf()` without modifying or interpolating measured photon counts;
* physical time-window selection through `crop_time_window()` using half-open interval semantics;
* photon-count-preserving histogram rebinning through `rebin_histogram()` for integer rebinning factors;
* total-count and peak-count normalization through `normalize_counts()` and the `CountNormalization` enum;
* package-specific histogram-validation errors through `InvalidHistogramError`;
* immutable `SimulationConfig` and `PreprocessingConfig` dataclasses for reproducible workflow configuration;
* JSON configuration serialization and loading through `save_config()` and `load_config()` using `pathlib`;
* logging for selected preprocessing operations such as background estimation and peak detection;
* integration tests covering IRF-relative cropping, composed machine-learning preprocessing, and use of preprocessing-derived background estimates to initialize raw-count Poisson fitting;
* Notebook 10 demonstrating validation, background estimation and subtraction, peak detection, temporal alignment, cropping, rebinning, normalization, and analysis-dependent preprocessing workflows.

### Changed

* extended the package architecture with dedicated `preprocessing.py`, `config.py`, and `exceptions.py` responsibilities;
* changed the preprocessing design from a potential single fixed pipeline toward a toolbox of small, composable scientific transformations;
* established separate preprocessing strategies for statistical inference and representation-oriented workflows;
* retained raw photon-count histograms for Poisson maximum-likelihood reconvolution instead of applying obligatory background subtraction or normalization;
* established background estimates as useful fitting-initialization information rather than replacements for the background parameter in the Poisson forward model;
* established temporal IRF alignment as a coordinate transformation rather than a shift or interpolation of measured photon counts;
* established cropping as distinct from selecting a fitting region after construction of the full reconvolution model, avoiding unnecessary convolution-edge effects;
* established photon-count conservation as a physical invariant of histogram rebinning;
* documented total and peak normalization as representation-oriented operations suitable for visualization, shape comparison, and machine-learning preparation rather than raw-count Poisson likelihood fitting;
* used least-squares reconvolution as a practical numerical initialization step before Poisson maximum-likelihood refinement in the reference preprocessing notebook workflow;
* expanded the package-level public API to expose the intended preprocessing and configuration functionality while keeping internal validation helpers private;
* updated the README with the preprocessing philosophy, separate statistical and machine-learning workflows, configuration utilities, package structure, and Notebook 10 documentation.

### Notes

* TCSPC preprocessing is intentionally analysis-dependent; the toolkit does not define a universal `preprocess()` routine or a monolithic preprocessing-pipeline class;
* raw photon counts remain the statistically appropriate observations for the current Poisson reconvolution fitter;
* background-subtracted values may legitimately become negative because individual Poisson observations fluctuate around the estimated mean background;
* rebinning preserves the total photon count within the rebinned region while trading temporal resolution for increased counts per bin and lower relative Poisson fluctuations;
* cropping may change the total photon count because it changes the selected observation window;
* normalization removes absolute photon-count information and therefore changes the statistical interpretation of the data;
* least-squares initialization improves numerical robustness of the current Poisson reconvolution workflow without changing the final Poisson likelihood objective;
* Notebook 10 provides the reference demonstration of two distinct workflows: raw-count statistical fitting and crop–rebin–normalize machine-learning representation.

### Limitations

* background-only regions must currently be selected explicitly; automatic background-region detection is not implemented;
* peak detection currently uses a simple discrete maximum without smoothing, interpolation, or noise-aware estimation;
* measured photon-count histograms are not fractionally shifted or interpolated during temporal alignment;
* rebinning currently requires an integer factor that exactly divides the number of selected histogram bins;
* automatic IRF rebinning for reconvolution is not implemented;
* cropping before convolution is not intended as a substitute for applying a fitting window to a model constructed on the full time grid;
* filtering, smoothing, denoising, baseline-polynomial fitting, and general interpolation are not yet implemented;
* preprocessing configuration currently uses lightweight dataclasses and JSON rather than higher-level configuration frameworks;
* Poisson reconvolution optimization remains sensitive to numerical initialization and currently benefits from least-squares warm-starting;
* experimental TCSPC and measured-IRF import workflows remain planned for later development stages.


## [0.5.0] - 2026-08-22

### Added

* engineered TCSPC feature extraction through `extract_features()`;
* photon-arrival mean, variance, and skewness features;
* cumulative photon-arrival quantiles at 10, 25, 50, 75, and 90 percent;
* post-peak half-decay timing;
* configurable log-tail slope, integrated tail fraction, and early/late count-ratio features;
* stable engineered-feature schema through `FEATURE_NAMES`;
* batch feature extraction through `extract_feature_table()`;
* batch histogram normalization through `normalize_histogram_batch()`;
* PCA fitting and transformation utilities for compressed histogram representations;
* cumulative PCA explained-variance analysis;
* tests covering feature calculations, schema stability, normalization invariance,
  PCA determinism, dimensionality, and train/test separation.

### Changed

* established a clear separation between measurement-derived ML features,
  prediction targets, and synthetic-data metadata;
* established three ML-ready TCSPC representations: engineered features,
  normalized histogram bins, and PCA-compressed histograms;
* established total-count normalization as the principal normalized-histogram
  representation while retaining peak normalization as an alternative;
* made PCA fitting explicitly train-only to prevent information leakage;
* moved batch histogram representation construction into `representations.py`;
* removed the legacy `ml_evaluation.normalize_histograms()` helper in favor of
  `representations.normalize_histogram_batch()`;
* restricted `ml_evaluation.py` to machine-learning evaluation responsibilities.

### Notes

* engineered features are calculated from histogram measurements rather than
  from simulation-generation metadata;
* simulation metadata remains suitable for labels, stratification, diagnostics,
  and controlled experiments but is not implicitly used as an ML input;
* total-count normalization intentionally removes absolute photon-count
  information;
* PCA components are data-dependent features and must therefore be fitted only
  on training data before transforming validation or test histograms.


## [0.6.0] - 2026-08-30

### Added

* reproducible factorial TCSPC benchmark generation with controlled variation of fluorescence lifetime, signal photon count, detector background, IRF width, and IRF temporal shift;
* shared benchmark train-test splits preserving aligned engineered-feature matrices, raw histograms, lifetime targets, and simulation metadata;
* train/test support and parameter-level balance diagnostics for controlled benchmark variables;
* constant-mean regression baseline;
* physics-inspired mean-arrival-time lifetime estimator;
* reusable scikit-learn pipelines for Ridge regression, Random Forest regression, and Histogram Gradient Boosting regression;
* unified regression benchmarking with MAE, median absolute error, RMSE, mean and median relative error, and $R^2$;
* controlled representation benchmarking across engineered physical features, TOTAL-normalized histogram bins, and PCA-compressed histograms using common samples, targets, and estimator families;
* photon-count ablation experiments that restore measured total photon counts to normalized and PCA histogram representations;
* classical reconvolution benchmarking through `classical_evaluation.py`;
* histogram-derived initial guesses for batch reconvolution fitting;
* per-curve classical-fit diagnostics including fitted lifetime, optimizer success, fit validity, parameter-boundary hits, Poisson negative log-likelihood, Poisson deviance, runtime, and failure information;
* aggregate reconvolution summaries including success/failure rates, MAE, median absolute error, RMSE, and runtime statistics;
* standardized per-sample prediction diagnostics shared by machine-learning and classical estimators;
* reusable benchmarking plots for true-versus-predicted lifetime, signed and absolute lifetime-error distributions, error versus physical conditions, and paired estimator comparison;
* conditional performance analysis across lifetime, photon-count, background, IRF-width, and IRF-misalignment regimes;
* regime-level summaries including sample counts, failure rates, MAE, median absolute error, signed bias, and 90th- and 95th-percentile absolute errors;
* controlled weakly bi-exponential TCSPC simulation for model-mismatch experiments;
* matched bi-exponential mismatch datasets preserving primary lifetime, photon-count target, background, IRF width, and IRF shift sample-by-sample;
* direct comparison of machine-learning distribution shift with classical mono-exponential forward-model mismatch;
* mismatch summaries reporting in-distribution and mismatch MAE, absolute and relative MAE degradation, bias changes, and failure rates;
* repeated batch inference timing for mean-arrival, Ridge, Random Forest, and Histogram Gradient Boosting estimators;
* reuse of recorded per-curve reconvolution optimization runtimes for computational-cost benchmarking;
* estimator throughput and accuracy-versus-runtime comparison;
* `baselines.py` for statistical and physics-inspired lifetime-estimation baselines;
* `ml_models.py` for reusable scikit-learn lifetime-regression pipelines;
* `classical_evaluation.py` for batch reconvolution benchmarking;
* `conditional_evaluation.py` for standardized diagnostics and physical-regime analysis;
* `benchmark_plots.py` for reusable benchmark visualization;
* `mismatch_evaluation.py` for controlled model-mismatch benchmarking;
* `timing_evaluation.py` for inference-runtime and throughput evaluation;
* Notebook 12 (`12_ml_benchmarking.ipynb`) integrating the complete Week 7 classical-versus-machine-learning benchmarking workflow.

### Changed

* extended `ml_evaluation.py` from basic regression evaluation to reproducible benchmark-dataset construction, shared train-test splitting, estimator benchmarking, representation comparison, photon-count ablation, and split diagnostics;
* extended the synthetic TCSPC workflow from feature and representation construction to full estimator benchmarking under controlled physical nuisance variation;
* established a common evaluation framework in which statistical baselines, physics-inspired estimators, machine-learning regressors, and classical reconvolution can be compared on the same test samples;
* established engineered physical features as the primary representation for the Week 7 classical-versus-ML benchmark while retaining normalized histograms and PCA representations for controlled comparison;
* extended benchmark evaluation beyond aggregate metrics to condition-dependent accuracy, bias, upper-tail error, fitting reliability, and computational cost;
* established explicit short/medium/long, low/medium/high, narrow/medium/broad, and IRF-misalignment regimes for scientifically interpretable conditional evaluation;
* extended classical reconvolution evaluation from individual fitting workflows to batch benchmarking with retained fit failures and parameter-boundary diagnostics;
* established controlled model mismatch as a first-class benchmark dimension rather than treating optimizer convergence as sufficient evidence of physical correctness;
* established estimator-only inference timing as a separate benchmark from model training and feature construction;
* updated the README to describe the Version 0.6 benchmarking architecture, Week 7 functionality, Notebook 12, revised package structure, updated limitations, and future development roadmap.

### Notes

* Notebook 12 uses a reproducible 810-curve factorial benchmark spanning six fluorescence lifetimes, three photon-count levels, three background levels, three IRF widths, and five signed IRF shifts;
* the final benchmark uses a fixed 80/20 train-test split shared across estimator and representation comparisons;
* the nonlinear engineered-feature models provide the strongest aggregate in-distribution accuracy in the current synthetic benchmark, but estimator ranking changes across physical operating regimes;
* classical reconvolution remains competitive or superior in selected conditions, including photon-limited measurements and large temporal IRF misalignment;
* the conditional benchmark demonstrates that classical and data-driven estimators should be treated as complementary rather than universally ordered;
* PCA preserves a large fraction of normalized-histogram variance but does not necessarily preserve the most lifetime-predictive information;
* TOTAL normalization intentionally removes absolute photon-count scale, and the photon-count ablation benchmark quantifies the effect of restoring measured total counts;
* the controlled mismatch benchmark introduces a 10% slow secondary component with `tau_secondary = 2 * tau_primary` while retaining the primary lifetime as the prediction target;
* machine-learning estimators are not retrained for the mismatch dataset, and classical reconvolution continues to use a mono-exponential forward model;
* successful numerical reconvolution fitting under model mismatch is explicitly distinguished from recovery of an unbiased physical lifetime;
* inference timing reports core estimator cost rather than complete raw-histogram-to-lifetime latency;
* machine-learning timing excludes model training and engineered-feature extraction, while reconvolution timing reuses the recorded numerical-optimization runtime for each curve.

### Limitations

* the final Week 7 benchmark remains based on synthetic TCSPC measurements;
* train and test subsets share the same discrete lifetime and nuisance-parameter support, so the present ML results primarily characterize interpolation and robustness across familiar physical levels rather than extrapolation to unseen lifetime ranges;
* the current synthetic IRF remains Gaussian;
* experimental or measured IRF import and calibration are not yet implemented;
* classical reconvolution uses a fixed IRF shape and width during an individual fit while temporal IRF shift is fitted;
* bi-exponential and multi-exponential reconvolution fitting are not yet implemented;
* the current model-mismatch benchmark covers one controlled weakly bi-exponential perturbation and does not yet include asymmetric IRFs, structured backgrounds, detector pile-up, dead time, or afterpulsing;
* experimental TCSPC file-format import and synthetic-to-real validation remain future work;
* calibrated uncertainty estimates and prediction intervals for the benchmarked estimators are not yet implemented;
* deep-learning lifetime estimators are not yet included;
* the reported inference benchmark measures estimator-only computational cost rather than complete preprocessing and feature-construction latency.


## [0.7.0] - 2026-09-26

Version 0.7 extends the toolkit from in-distribution estimator benchmarking to
controlled external generalization, robustness, uncertainty calibration, and
failure-awareness.

The release combines the Week 8 and Week 9 development stages. Week 8 asks
whether estimator performance survives controlled distribution shift and
physical model mismatch. Week 9 asks whether the corresponding uncertainty
estimates recognize when the estimator becomes unreliable.

### Week 8 — Generalization and robustness

#### Added

* frozen Week-8 generalization protocol separating a familiar development
  domain from untouched final robustness Tests A–F;
* explicit familiar-domain definitions for fluorescence lifetime, signal
  photon count, detector background, IRF width, and temporal IRF shift;
* reproducible paired A–F robustness-suite generation with fixed test-specific
  random seeds and retained sample provenance;
* matched `pair_id` structure enabling sample-by-sample comparison between
  familiar Test A and shifted or misspecified conditions;
* Test A as the familiar-distribution external reference;
* Test B for low- and high-photon-count out-of-distribution evaluation;
* Test C for broadened-IRF evaluation;
* Test D for elevated detector-background evaluation;
* Test E for increased temporal IRF-misalignment evaluation;
* Test F for controlled weak and moderate bi-exponential model mismatch;
* dominant-component and signal-photon-weighted lifetime references for
  interpreting bi-exponential Test F;
* reproducible repeated K-fold cross-validation infrastructure for
  development-set performance-stability analysis;
* canonical repeated-CV configuration using five folds, five repeats, and an
  explicit random seed;
* long-form fold-level CV results containing repeat, fold, training-set size,
  validation-set size, MAE, median absolute error, RMSE, signed bias, and
  $R^2$;
* aggregate repeated-CV summaries reporting mean and sample standard deviation
  across repeated train-validation partitions;
* estimator cloning within every CV fold to prevent fitted-state reuse;
* leakage-safe engineered-feature, TOTAL-normalized-histogram, and
  PCA-compressed representation evaluation;
* leakage-safe TOTAL-normalized-histogram Ridge pipeline operating directly on
  raw TCSPC histograms;
* leakage-safe TOTAL-normalized-histogram → PCA → StandardScaler → Ridge
  pipeline with PCA fitted independently inside every CV training fold;
* multi-estimator repeated-CV benchmarking using identical deterministic
  train-validation partitions;
* frozen external evaluation of Ridge, Random Forest, Histogram Gradient
  Boosting, mean-arrival estimation, and classical Poisson reconvolution;
* photon-count-regime diagnostics separating low- and high-photon Test-B
  conditions;
* paired familiar-versus-OOD diagnostics for photon-count shifts;
* controlled instrument/acquisition robustness evaluation across Tests C–E;
* explicit comparison of correctly specified and deliberately misspecified
  IRFs under Test C;
* paired IRF-width diagnostics for broadened-response measurements;
* temporal-shift recovery diagnostics for Test E;
* classical reconvolution robustness evaluation using per-curve physical IRF
  information;
* controlled comparison of mono-exponential estimators against weak and
  moderate bi-exponential Test-F measurements;
* paired Test-A versus Test-F model-mismatch diagnostics;
* Test-F severity summaries for weak and moderate secondary-component
  contamination;
* classical Poisson goodness-of-fit summaries under model mismatch;
* unified A–F prediction tables and MAE-degradation scorecards for the
  principal estimators;
* final Week-8 robustness synthesis combining machine-learning and classical
  estimator results;
* Notebook 13 (`13_generalization_and_robustness.ipynb`) integrating the
  complete Week-8 generalization and robustness workflow;
* automated tests covering protocol definitions, paired A–F generation,
  repeated cross-validation, leakage prevention, external-test isolation,
  photon-count OOD evaluation, instrument/acquisition robustness,
  model-mismatch evaluation, paired diagnostics, severity analysis, and final
  robustness synthesis.

#### Changed

* extended the evaluation architecture from a single in-distribution
  train/test benchmark to an explicit development-versus-external-test
  protocol;
* established repeated cross-validation as a development-only operation,
  separate from the final untouched A–F robustness suite;
* established Tests A–F as final external evaluation data that cannot enter
  training, preprocessing fitting, feature selection, hyperparameter
  selection, or model selection;
* required learned representation parameters, including PCA, to be fitted
  exclusively on development data before final-test transformation;
* established raw development histograms rather than globally precomputed PCA
  features as the correct input for cross-validated PCA pipelines;
* established identical repeated-CV partitions across estimators for fair
  stability comparison;
* established mean ± sample-standard-deviation reporting across repeated
  development partitions as the primary CV stability summary;
* extended generalization evaluation beyond aggregate error to paired
  degradation, nuisance-specific diagnostics, fit validity, parameter
  recovery, and goodness-of-fit;
* established controlled physical distribution shift and forward-model
  misspecification as distinct robustness questions;
* established explicit per-curve IRF information for classical Tests C–E so
  that acquisition degradation can be distinguished from deliberate IRF
  misspecification;
* established the dominant-component lifetime `tau_1` as the primary frozen
  Test-F scoring reference while retaining the signal-photon-weighted
  lifetime as an additional descriptive reference;
* distinguished numerical fit success from scientific model validity under
  deliberate bi-exponential mismatch.

#### Notes

* the canonical repeated-CV protocol performs 25 evaluations per estimator
  using 5 folds × 5 repeats;
* strong development-set cross-validation does not guarantee external
  robustness;
* engineered-feature Ridge provides particularly strong repeated-CV
  performance, while Random Forest is the strongest current ML estimator
  across most final A–F robustness conditions;
* low photon count and elevated detector background are the dominant current
  weaknesses of the engineered-feature ML estimators;
* IRF broadening is comparatively benign when the broadened IRF is correctly
  represented, while explicit IRF misspecification is substantially more
  consequential;
* classical Poisson reconvolution remains the lowest-MAE principal estimator
  across the current A–F suite and benefits strongly when relevant nuisance
  parameters are represented explicitly in the forward model;
* under Test F, mono-exponential classical reconvolution can remain
  numerically successful while developing systematic lifetime bias;
* smaller ML degradation under Test F does not establish correct
  bi-exponential modelling because the ML estimators were trained only on
  mono-exponential data;
* robustness is estimator-, representation-, nuisance-, and mismatch-specific
  rather than universal;
* Notebook 13 provides the reproducible synthesis of the complete Week-8
  robustness stage.

### Week 9 — Uncertainty and failure-awareness

#### Added

* reusable quantile-regression pipeline construction using
  `HistGradientBoostingRegressor` with explicit quantile loss;
* three-model 0.05/0.50/0.95 quantile lifetime estimator producing nominal
  90% prediction intervals;
* dedicated Week-9 uncertainty-training and held-out calibration split;
* prediction-interval evaluation including empirical coverage, coverage
  error, mean and median interval width, interval score, and interval failure
  rate;
* quantile-specific evaluation including lower, median, and upper pinball loss
  and explicit quantile-crossing diagnostics;
* frozen quantile-estimator evaluation on external robustness conditions
  without refitting or recalibration;
* paired Test-A uncertainty-response diagnostics for photon-count, background,
  and model-mismatch shifts;
* Random-Forest tree-disagreement uncertainty scoring based on per-tree
  lifetime predictions;
* reusable non-parametric training-data bootstrap prediction spread,
  demonstrated with Ridge regression;
* uncertainty-score evaluation using error correlation and
  low-/high-uncertainty error subsets;
* paired A-to-shift uncertainty-score diagnostics retaining the frozen Week-8
  `pair_id` structure;
* local Poisson/Fisher covariance estimation for reconvolution parameters
  using the expected-count Jacobian;
* parameter scaling for Fisher-information calculation and explicit matrix
  conditioning diagnostics;
* explicit covariance failure handling for unsuccessful fits,
  parameter-boundary solutions, rank deficiency, and ill-conditioned local
  information;
* parametric Poisson-bootstrap uncertainty for reconvolution lifetime
  estimates;
* bootstrap lifetime distributions, percentile intervals, standard
  deviations, and retained refit/boundary failures;
* repeated-Poisson empirical calibration experiments in which the physical
  TCSPC condition is fixed and only the independent Poisson realization
  changes;
* empirical comparison of lifetime sampling variability with local covariance
  and parametric-bootstrap uncertainty;
* split-conformal calibration of the direct quantile prediction intervals
  using only the held-out development calibration subset;
* complete frozen A–F ML uncertainty scorecards for direct quantile,
  conformalized quantile, Random-Forest disagreement, and Ridge
  training-bootstrap methods;
* complete frozen A–F classical uncertainty scorecards for local covariance
  and parametric Poisson bootstrap;
* conditional classical uncertainty scorecards for low/high photon count,
  elevated background, and weak/moderate bi-exponential model mismatch;
* normalized error-to-uncertainty diagnostics for identifying overconfident
  predictions;
* signed Poisson-deviance residual matrices and aggregate residual-profile
  diagnostics for familiar versus model-mismatch conditions;
* paired Test-F-minus-Test-A residual-profile analysis;
* Notebook 14 (`14_uncertainty_and_failure_awareness.ipynb`) integrating the
  complete Week-9 uncertainty-calibration and failure-awareness workflow;
* automated tests covering uncertainty splitting, interval metrics, quantile
  crossing, RF spread, ML-bootstrap reproducibility, local covariance
  validity and conditioning, parametric-bootstrap behaviour,
  repeated-Poisson calibration, A–F identity preservation, conditional
  regime preservation, conformal calibration, classical uncertainty
  scorecards, physical high-count repeated-Poisson behaviour, and
  residual-structure diagnostics.

#### Changed

* reformulated the principal Poisson reconvolution optimizer in scaled
  dimensionless parameter coordinates to improve numerical conditioning
  while preserving the same physical model, likelihood, public fitting API,
  and physical parameter bounds;
* established uncertainty training, uncertainty calibration, and final A–F
  robustness evaluation as strictly separate data roles;
* established prediction intervals, uncertainty scores, and empirical
  repeated-measurement variability as distinct uncertainty output types;
* established repeated-Poisson experiments as the empirical calibration
  reference for synthetic classical lifetime uncertainty;
* extended final robustness evaluation from point-estimate accuracy to
  uncertainty coverage, sharpness, normalized error, and failure-awareness;
* established Test F as a deliberate evaluation of uncertainty under physical
  forward-model misspecification rather than ordinary statistical noise;
* retained the dominant-component lifetime `tau_1` as the frozen primary
  Test-F scoring reference while explicitly recognizing that a
  bi-exponential decay has no unique mono-exponential true lifetime;
* extended project documentation from robustness analysis to uncertainty
  calibration, conformal evaluation, classical uncertainty, and
  model-mismatch failure-awareness.

#### Notes

* direct quantile gradient boosting provides conservative but broad prediction
  intervals on the current small development dataset; interval width reacts
  to some acquisition changes but provides weak recognition of controlled
  bi-exponential model mismatch;
* Random-Forest tree disagreement is the strongest current sample-level ML
  error-ranking uncertainty score, while Ridge training-bootstrap spread is
  more sensitive to several acquisition-distribution shifts than to
  sample-level prediction error;
* under correctly specified mono-exponential conditions, local
  Poisson/Fisher covariance and parametric Poisson bootstrap reproduce
  empirical repeated-measurement variability reasonably well and respond
  physically to changes in photon count and detector background;
* uncertainty calibration exposed a numerical-conditioning problem in the
  Poisson reconvolution optimizer; scaled optimization removed the resulting
  high-photon systematic bias and restored physically consistent uncertainty
  behaviour;
* the principal Week-9 failure occurs under controlled bi-exponential model
  mismatch: classical lifetime error increases substantially while
  covariance and bootstrap uncertainty increase only modestly, causing
  nominal 90% interval coverage to collapse;
* parametric bootstrap does not resolve physical model misspecification
  because its synthetic measurements are sampled from the same fitted
  mono-exponential forward model;
* split conformalization adds zero correction in the present experiment
  because every held-out calibration target already lies inside the original
  direct-quantile interval;
* scalar Poisson deviance remains almost insensitive to Test-F mismatch, and
  aggregate signed-residual profiles provide only modest additional
  discrimination without a clear severity-dependent temporal signature;
* the central Week-9 conclusion is that an estimator can correctly quantify
  statistical uncertainty under its assumed physical model while remaining
  confidently wrong when that physical model is incomplete;
* Notebook 14 provides the reproducible synthesis of the complete Week-9
  uncertainty and failure-awareness stage.

### Release scope

Version 0.7 remains a synthetic-benchmark release. Experimental TCSPC import,
measured-IRF workflows, synthetic-to-real validation, general bi- and
multi-exponential inverse fitting, Bayesian inference, detector pile-up,
dead-time and afterpulsing models, and broader physical model-selection
diagnostics remain future development stages.

The principal release-level conclusion is that predictive accuracy,
robustness, and uncertainty calibration are separate requirements. Strong
in-distribution performance does not guarantee robustness under distribution
shift, and well-calibrated statistical uncertainty does not guarantee
awareness of physical model misspecification.
