"""Curated package-root exports and import-time optional boundaries."""

import importlib
import pickle
import subprocess
import sys
from types import FunctionType

import pytest

import tcspc_toolkit as toolkit
from tcspc_toolkit.estimation import api as estimator_api
from tcspc_toolkit.simulation import build_expected_counts_from_irf


# Reviewed Issue-#2 contract: independent of runtime export discovery. Package
# relocation must preserve both membership and the existing declaration order.
EXPECTED_ROOT_EXPORTS = (
    "monoexponential_decay",
    "build_expected_counts_from_irf",
    "sample_photon_counts",
    "SyntheticDataset",
    "generate_monoexponential_dataset",
    "generate_gaussian_irf",
    "generate_emg_irf",
    "generate_gaussian_irf_profile",
    "generate_emg_irf_profile",
    "IRFSourceKind",
    "IRFProfile",
    "IRFPreparationDiagnostics",
    "PreparedIRF",
    "irf_profile_from_sampled_irf",
    "prepare_irf",
    "LeadingEdgeIRFDiagnostics",
    "LeadingEdgeIRFResult",
    "estimate_irf_from_leading_edge",
    "normalize_irf",
    "shift_irf",
    "convolve_decay_with_irf",
    "LifetimeFitResult",
    "ReconvolutionFitResult",
    "fit_monoexponential_decay",
    "fit_monoexponential_reconvolution",
    "TCSPCError",
    "InvalidHistogramError",
    "InvalidMeasurementError",
    "FeatureExtractionError",
    "TimeUnit",
    "MeasurementDataKind",
    "SampledIRF",
    "TCSPCMeasurement",
    "convert_time_to_ns",
    "load_sampled_irf_csv",
    "load_tcspc_measurement_csv",
    "ReferenceLifetimeEvaluation",
    "fit_experimental_reconvolution",
    "measurement_to_feature_table",
    "measurement_to_histogram_batch",
    "evaluate_estimate_against_reference",
    "align_to_irf",
    "crop_time_window",
    "detect_peak",
    "estimate_background",
    "normalize_counts",
    "rebin_histogram",
    "subtract_background",
    "validate_histogram",
    "CountNormalization",
    "FeatureConfig",
    "PreprocessingConfig",
    "SimulationConfig",
    "load_config",
    "save_config",
    "FEATURE_NAMES",
    "extract_feature_table",
    "extract_features",
    "cumulative_explained_variance",
    "fit_pca_representation",
    "normalize_histogram_batch",
    "transform_pca_representation",
    "RegressorProtocol",
    "EstimatorSpec",
    "fit_regressors",
    "predict_regressors",
)


def assert_same_export(public_module, canonical_module, name):
    """Supported root/package exports retain exact canonical object identity."""
    public = importlib.import_module(public_module)
    canonical = importlib.import_module(canonical_module)
    assert getattr(public, name) is getattr(canonical, name)


def assert_surface_after_removal(historical_module, canonical, names):
    """Retain symbols and current serialization while rejecting removed globals."""
    retained_packages = {
        "tcspc_toolkit.irf", "tcspc_toolkit.preprocessing",
        "tcspc_toolkit.simulation", "tcspc_toolkit.evaluation",
        "tcspc_toolkit.persistence",
    }
    if historical_module in retained_packages:
        public = importlib.import_module(historical_module)
        assert tuple(public.__all__) == names
        for name in names:
            assert getattr(public, name) is getattr(canonical, name)
    else:
        assert importlib.util.find_spec(historical_module) is None
        with pytest.raises(ModuleNotFoundError) as error:
            importlib.import_module(historical_module)
        assert error.value.name == historical_module
        for name in names:
            historical_global = f"c{historical_module}\n{name}\n.".encode()
            with pytest.raises(ModuleNotFoundError) as error:
                pickle.loads(historical_global)
            assert error.value.name == historical_module
    for name in names:
        obj = getattr(canonical, name)
        if isinstance(obj, (type, FunctionType)):
            assert pickle.loads(pickle.dumps(obj)) is obj


def test_all_exports_are_unique_and_accessible():
    assert tuple(toolkit.__all__) == EXPECTED_ROOT_EXPORTS
    assert len(toolkit.__all__) == len(set(toolkit.__all__)) == 66
    for name in toolkit.__all__:
        assert getattr(toolkit, name) is not None, name


@pytest.mark.parametrize("name", [
    "RegressorProtocol", "EstimatorSpec", "fit_regressors", "predict_regressors",
])
def test_extension_exports_are_canonical_objects(name):
    assert name in toolkit.__all__
    assert_same_export("tcspc_toolkit", estimator_api.__name__, name)


@pytest.mark.parametrize("old_module, canonical_module, names", [
    ("tcspc_toolkit.estimator_api", "tcspc_toolkit.estimation.api", (
        "RegressorProtocol", "EstimatorSpec", "fit_regressors", "predict_regressors",
    )),
    ("tcspc_toolkit.baselines", "tcspc_toolkit.estimation.baselines", (
        "FloatArray", "predict_constant_mean_baseline", "estimate_lifetime_from_mean_arrival",
    )),
    ("tcspc_toolkit.models", "tcspc_toolkit.physics.models", (
        "monoexponential_decay", "biexponential_decay", "multiexponential_decay",
    )),
    ("tcspc_toolkit.convolution", "tcspc_toolkit.physics.convolution", (
        "convolve_decay_with_irf",
    )),
    ("tcspc_toolkit.forward_model", "tcspc_toolkit.physics.forward_model", (
        "monoexponential_reconvolution_expected_counts",
    )),
])
def test_canonical_foundations_survive_removed_module_paths(old_module, canonical_module, names):
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    for name in names:
        if name != "FloatArray":
            assert getattr(canonical, name).__module__ == canonical_module


@pytest.mark.parametrize("module, name", [
    ("models", "monoexponential_decay"),
    ("convolution", "convolve_decay_with_irf"),
])
def test_physical_root_exports_are_canonical_objects(module, name):
    assert_same_export("tcspc_toolkit", f"tcspc_toolkit.physics.{module}", name)


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.preprocessing", "histograms", (
        "CountNormalization", "validate_time_axis", "time_axes_compatible",
        "validate_histogram", "estimate_background", "subtract_background",
        "detect_peak", "align_to_irf", "crop_time_window", "rebin_histogram",
        "normalize_counts",
    )),
    ("tcspc_toolkit.features", "features", (
        "FEATURE_NAMES", "extract_features", "extract_feature_table",
        "quantile_arrival_time", "half_decay_time", "tail_log_slope",
        "integrated_tail_fraction", "early_late_count_ratio",
    )),
    ("tcspc_toolkit.representations", "representations", (
        "normalize_histogram_batch", "fit_pca_representation",
        "transform_pca_representation", "cumulative_explained_variance",
        "augment_with_total_counts",
    )),
])
def test_preprocessing_canonical_surfaces_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.preprocessing.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    for name in names:
        if name in EXPECTED_ROOT_EXPORTS:
            assert_same_export("tcspc_toolkit", canonical_module, name)
        if name not in {"FEATURE_NAMES", "CountNormalization"}:
            assert getattr(canonical, name).__module__ == canonical_module
    # This enum was historically imported from preprocessing by real callers;
    # it remains the config-owned object, not a second definition.
    if module == "histograms":
        assert_same_export(old_module, "tcspc_toolkit.config", "CountNormalization")


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.irf", "profiles", (
        "IRFSourceKind", "IRFProfile", "generate_gaussian_irf", "generate_emg_irf",
        "generate_gaussian_irf_profile", "generate_emg_irf_profile", "normalize_irf", "shift_irf",
    )),
    ("tcspc_toolkit.irf_preparation", "preparation", (
        "FloatArray", "ResamplingMethod", "IRFPreparationDiagnostics", "PreparedIRF",
        "irf_profile_from_sampled_irf", "prepare_irf",
    )),
    ("tcspc_toolkit.irf_estimation", "estimation", (
        "TimeWindow", "BinRange", "LeadingEdgeIRFDiagnostics", "LeadingEdgeIRFResult",
        "estimate_irf_from_leading_edge",
    )),
])
def test_irf_canonical_surfaces_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.irf.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    for name in names:
        if name in EXPECTED_ROOT_EXPORTS:
            assert_same_export("tcspc_toolkit", canonical_module, name)
        if name not in {"FloatArray", "ResamplingMethod", "TimeWindow", "BinRange"}:
            assert getattr(canonical, name).__module__ == canonical_module


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.measurements", "records", (
        "FloatArray", "IntArray", "TimeUnit", "MeasurementDataKind", "SampledIRF",
        "TCSPCMeasurement", "convert_time_to_ns",
    )),
    ("tcspc_toolkit.measurement_io", "io", (
        "load_sampled_irf_csv", "load_tcspc_measurement_csv",
    )),
    ("tcspc_toolkit.experimental", "adapters", (
        "ReferenceLifetimeEvaluation", "fit_experimental_reconvolution",
        "measurement_to_feature_table", "measurement_to_histogram_batch",
        "evaluate_estimate_against_reference",
    )),
])
def test_measurement_canonical_surfaces_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.measurement.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    for name in names:
        if name in EXPECTED_ROOT_EXPORTS:
            assert_same_export("tcspc_toolkit", canonical_module, name)
        if name not in {"FloatArray", "IntArray"}:
            assert getattr(canonical, name).__module__ == canonical_module


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.simulation", "histograms", (
        "sample_photon_counts", "build_expected_counts_from_irf",
        "simulate_irf_convolved_histogram", "simulate_irf_convolved_biexponential_histogram",
        "simulate_monoexponential_decay", "simulate_biexponential_decay",
        "simulate_multiexponential_decay",
    )),
    ("tcspc_toolkit.datasets", "datasets", (
        "FloatArray", "IntArray", "SyntheticDataset", "generate_monoexponential_dataset",
        "generate_grouped_monoexponential_dataset",
    )),
])
def test_simulation_canonical_surfaces_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.simulation.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    for name in names:
        if name in EXPECTED_ROOT_EXPORTS:
            assert_same_export("tcspc_toolkit", canonical_module, name)
        if name not in {"FloatArray", "IntArray"}:
            assert getattr(canonical, name).__module__ == canonical_module


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.fitting", "fitting", (
        "LifetimeFitResult", "ReconvolutionFitResult", "fit_monoexponential_decay",
        "poisson_negative_log_likelihood", "fit_monoexponential_reconvolution",
    )),
    ("tcspc_toolkit.classical_evaluation", "classical", (
        "ReconvolutionInitialGuess", "ReconvolutionCurveResult",
        "ReconvolutionBenchmarkSummary", "ReconvolutionBenchmarkResult",
        "estimate_reconvolution_initial_guess", "fit_single_reconvolution_curve",
        "evaluate_reconvolution_benchmark",
    )),
])
def test_classical_canonical_surfaces_preserve_supported_objects_and_removed_paths(old_module, module, names):
    canonical_module = f"tcspc_toolkit.estimation.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    for name in names:
        assert getattr(canonical, name).__module__ == canonical_module
        if name in EXPECTED_ROOT_EXPORTS:
            assert_same_export("tcspc_toolkit", canonical_module, name)


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.bayesian", "posterior", (
        "GammaPrior",
        "LogNormalPrior",
        "BoundedUniformPrior",
        "BayesianPriorConfig",
        "BayesianPhysicalParameters",
        "IRFModelRelation",
        "BayesianModelContext",
        "sampling_to_physical_parameters",
        "physical_to_sampling_coordinates",
        "log_transformation_jacobian",
        "log_prior",
        "log_likelihood",
        "log_posterior",
        "fit_bayesian_monoexponential_reconvolution",
    )),
    ("tcspc_toolkit.bayesian_sampling", "sampling", (
        "PHYSICAL_PARAMETER_NAMES",
        "SAMPLING_COORDINATE_NAMES",
        "PARAMETER_UNITS",
        "BayesianSamplingConfig",
        "BayesianSamplingStatus",
        "BayesianPosteriorSamples",
        "BayesianParameterSummary",
        "BayesianSamplingDiagnostics",
        "BayesianRuntime",
        "BayesianModelContextSummary",
        "BayesianReconvolutionResult",
        "BayesianInferenceRun",
        "BayesianInitializationError",
        "fit_bayesian_monoexponential_reconvolution",
    )),
    ("tcspc_toolkit.bayesian_predictive", "predictive", (
        "PHYSICAL_PARAMETER_ORDER",
        "BayesianPredictiveBand",
        "summarize_predictive_band",
        "BayesianPredictiveDiscrepancy",
        "BayesianPredictiveWindowDiscrepancy",
        "BayesianPosteriorPredictiveDiagnostics",
        "calculate_bayesian_posterior_predictive_diagnostics",
        "BayesianPriorPredictiveResult",
        "BayesianPosteriorPredictiveResult",
        "sample_bayesian_prior_predictive",
        "sample_bayesian_posterior_predictive",
    )),
])
def test_bayesian_canonical_surfaces_preserve_supported_objects_and_removed_paths(old_module, module, names):
    canonical_module = f"tcspc_toolkit.estimation.bayesian.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        if not name.isupper():
            assert getattr(canonical, name).__module__ == canonical_module


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.ml_models", "models", (
        "DEFAULT_RANDOM_STATE",
        "DEFAULT_PCA_COMPONENTS",
        "make_ridge_pipeline",
        "make_random_forest_pipeline",
        "make_hist_gradient_boosting_pipeline",
        "make_canonical_ml_estimator_specs",
        "make_quantile_hist_gradient_boosting_pipeline",
        "make_normalized_histogram_ridge_pipeline",
        "make_pca_histogram_ridge_pipeline",
    )),
    ("tcspc_toolkit.ml_evaluation", "evaluation", (
        "FloatArray",
        "IntArray",
        "BenchmarkMeasurements",
        "BenchmarkDataset",
        "BenchmarkSplit",
        "RegressionMetrics",
        "RegressionBenchmarkResult",
        "HistogramRepresentations",
        "generate_benchmark_measurements",
        "build_benchmark_dataset",
        "split_benchmark_dataset",
        "build_histogram_representations",
        "summarize_split_coverage",
        "summarize_split_level_balance",
        "evaluate_regression",
        "evaluate_regressor",
        "evaluate_ridge_representation_benchmark",
        "evaluate_photon_count_ablation",
        "evaluate_nonlinear_representation_benchmark",
        "evaluate_baselines",
    )),
    ("tcspc_toolkit.cross_validation", "cross_validation", (
        "DEFAULT_CV_N_SPLITS",
        "DEFAULT_CV_N_REPEATS",
        "DEFAULT_CV_RANDOM_STATE",
        "RepeatedCVConfig",
        "RepeatedCVBenchmarkResult",
        "make_repeated_kfold",
        "evaluate_regressor_repeated_cv",
        "CV_METRIC_COLUMNS",
        "summarize_repeated_cv",
        "evaluate_estimators_repeated_cv",
    )),
])
def test_ml_canonical_surfaces_preserve_supported_objects_and_removed_paths(old_module, module, names):
    canonical_module = f"tcspc_toolkit.ml.{module}"
    canonical = importlib.import_module(canonical_module)
    assert_surface_after_removal(old_module, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        if not name.isupper() and name not in {"FloatArray", "IntArray"}:
            assert getattr(canonical, name).__module__ == canonical_module


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.evaluation_results", "results", (
        "MethodDescriptor", "LifetimeReference", "EvaluationBatch", "PointEvaluationResult",
    )),
    ("tcspc_toolkit.evaluation_core", "core", (
        "build_point_evaluation", "combine_point_evaluations", "calculate_reference_errors",
        "summarize_point_errors", "calculate_degradation_ratio",
    )),
    ("tcspc_toolkit.evaluation_uncertainty", "attachments", (
        "IntervalAttachment", "ScoreAttachment",
    )),
])
def test_factual_canonical_surfaces_preserve_supported_objects_without_expanding_root(old_module, module, names):
    canonical_module = f"tcspc_toolkit.evaluation.{module}"
    canonical = importlib.import_module(canonical_module)
    diagnostics_facade = importlib.import_module("tcspc_toolkit.evaluation")
    assert_surface_after_removal(old_module, canonical, names)
    assert tuple(canonical.__all__) == names
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        assert getattr(canonical, name).__module__ == canonical_module
        assert not hasattr(diagnostics_facade, name)
        assert not hasattr(toolkit, name)


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.conditional_evaluation", "conditional", (
        "build_prediction_diagnostics", "build_ml_prediction_diagnostics",
        "build_classical_prediction_diagnostics", "summarize_conditional_performance",
        "assign_numeric_regimes", "add_benchmark_regimes", "summarize_standard_regimes",
    )),
    ("tcspc_toolkit.benchmark_plots", "plots", (
        "plot_true_vs_predicted", "plot_error_histogram", "plot_error_vs_condition",
        "plot_paired_absolute_error_comparison",
    )),
])
def test_analysis_canonical_surfaces_preserve_supported_functions_without_expanding_root(old_module, module, names):
    canonical_module = f"tcspc_toolkit.evaluation.{module}"
    canonical = importlib.import_module(canonical_module)
    diagnostics_facade = importlib.import_module("tcspc_toolkit.evaluation")
    assert_surface_after_removal(old_module, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        assert getattr(canonical, name).__module__ == canonical_module
        assert not hasattr(diagnostics_facade, name)
        assert not hasattr(toolkit, name)


@pytest.mark.parametrize("old_module, module, names", [
    ("tcspc_toolkit.generalization", "generalization_protocol", (
        "FINAL_ROBUSTNESS_TEST_IDS",
        "DEVELOPMENT_ONLY_OPERATIONS",
        "GeneralizationShift",
        "GeneralizationRegime",
        "GeneralizationProtocol",
        "default_generalization_protocol",
        "FamiliarSimulationDomain",
        "DEFAULT_GENERALIZATION_SUITE_VERSION",
        "DEFAULT_DEVELOPMENT_REFERENCE",
        "default_familiar_simulation_domain",
        "GeneralizationNumerics",
        "GeneralizationSuiteDefinition",
        "default_generalization_numerics",
        "default_generalization_suite",
    )),
    ("tcspc_toolkit.generalization_datasets", "generalization_datasets", (
        "FloatArray",
        "IntArray",
        "PAIR_VARIANTS_PER_CONDITION",
        "GeneralizationTestMeasurements",
        "GeneralizationTestSuite",
        "build_generalization_time_axis",
        "build_paired_parameter_design",
        "build_test_parameter_table",
        "build_test_f_parameter_table",
        "generate_generalization_test",
        "generate_generalization_test_suite",
        "build_generalization_suite_manifest",
    )),
    ("tcspc_toolkit.generalization_evaluation", "generalization", (
        "DEFAULT_DEVELOPMENT_RANDOM_SEED",
        "MIN_REFERENCE_MAE_NS",
        "RobustnessMetrics",
        "RobustnessDegradation",
        "calculate_robustness_metrics",
        "calculate_mae_degradation",
        "build_generalization_development_measurements",
        "GeneralizationABPreparedData",
        "GeneralizationPreparedData",
        "prepare_generalization_ab_data",
        "prepare_generalization_data",
        "fit_generalization_ml_estimators",
        "PrincipalABBenchmarkResult",
        "summarize_generalization_predictions",
        "build_mae_degradation_table",
        "build_reference_mae_degradation_table",
        "evaluate_principal_ab_benchmark",
        "RepresentationABBenchmarkResult",
        "build_ab_comparison_table",
        "evaluate_ml_representation_ab_benchmark",
        "add_test_b_photon_count_regime",
        "summarize_test_b_photon_count_ood",
        "PhotonCountShiftReport",
        "build_photon_count_shift_report",
        "build_generalization_plot_diagnostics",
        "InstrumentAcquisitionBenchmarkResult",
        "ModelMismatchBenchmarkResult",
        "build_test_f_reference_diagnostics",
        "summarize_test_f_model_mismatch",
        "evaluate_model_mismatch_benchmark",
        "evaluate_instrument_acquisition_benchmark",
        "ClassicalInstrumentAcquisitionBenchmarkResult",
        "evaluate_classical_instrument_acquisition_benchmark",
        "build_instrument_representation_comparison",
        "build_test_c_paired_irf_diagnostics",
        "summarize_test_c_paired_irf_diagnostics",
        "summarize_test_e_temporal_shift_recovery",
        "InstrumentAcquisitionDiagnostics",
        "build_instrument_acquisition_diagnostics",
        "ClassicalModelMismatchBenchmarkResult",
        "build_classical_af_paired_diagnostics",
        "summarize_classical_test_f_model_mismatch",
        "evaluate_classical_model_mismatch_benchmark",
        "build_af_comparison_table",
        "build_model_mismatch_severity_comparison",
        "build_classical_model_mismatch_gof_summary",
        "DecayModelMismatchReport",
        "build_decay_model_mismatch_report",
        "GeneralizationSuiteBenchmarkResult",
        "evaluate_generalization_suite_benchmark",
        "ClassicalGeneralizationSuiteBenchmarkResult",
        "evaluate_classical_generalization_suite_benchmark",
        "GeneralizationRobustnessReport",
        "build_generalization_robustness_report",
        "Day53ABReport",
        "build_day53_ab_report",
        "Day55ModelMismatchReport",
        "build_day55_model_mismatch_report",
        "build_day55_severity_comparison",
        "Week8RobustnessReport",
        "build_week8_robustness_report",
    )),
])
def test_benchmark_canonical_surfaces_preserve_supported_objects_without_expanding_root(old_module, module, names):
    canonical_module = f"tcspc_toolkit.benchmarks.{module}"
    canonical = importlib.import_module(canonical_module)
    namespace = importlib.import_module("tcspc_toolkit.benchmarks")
    assert_surface_after_removal(old_module, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        if not name.isupper() and name not in {"FloatArray", "IntArray"}:
            assert getattr(canonical, name).__module__ == canonical_module
        assert not hasattr(namespace, name)
        assert not hasattr(toolkit, name)


def test_generalization_canonical_and_removed_pickle_globals():
    canonical = importlib.import_module("tcspc_toolkit.benchmarks.generalization")
    assert_surface_after_removal(
        "tcspc_toolkit.generalization_evaluation", canonical,
        ("PhotonCountShiftReport", "Day53ABReport",
         "GeneralizationRobustnessReport", "Week8RobustnessReport"),
    )


def test_uncertainty_evaluation_canonical_surface_preserves_public_objects_and_removed_paths():
    names = (
        "DEFAULT_UNCERTAINTY_CALIBRATION_FRACTION",
        "DEFAULT_UNCERTAINTY_SPLIT_SEED",
        "WEEK9_HYPOTHESES",
        "UncertaintyOutputKind",
        "UncertaintyDataRole",
        "UncertaintyMethodDefinition",
        "UncertaintyDataProvenance",
        "UncertaintyDevelopmentSplit",
        "make_uncertainty_development_split",
        "PredictionIntervalResult",
        "UncertaintyScoreResult",
        "IntervalEvaluationMetrics",
        "QuantileIntervalEvaluationMetrics",
        "UncertaintyScoreMetrics",
        "SelectivePredictionMetrics",
        "evaluate_prediction_intervals",
        "evaluate_quantile_prediction_intervals",
        "evaluate_uncertainty_scores",
        "evaluate_selective_prediction",
    )
    canonical = importlib.import_module("tcspc_toolkit.uncertainty.evaluation")
    namespace = importlib.import_module("tcspc_toolkit.uncertainty")
    assert_surface_after_removal("tcspc_toolkit.uncertainty_evaluation", canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        obj = getattr(canonical, name)
        assert not hasattr(namespace, name)
        assert not hasattr(toolkit, name)
        if callable(obj):
            assert obj.__module__ == canonical.__name__
            assert pickle.loads(pickle.dumps(obj)) is obj
    # Source uncertainty outputs remain distinct from point-linked attachments.
    attachments = importlib.import_module("tcspc_toolkit.evaluation.attachments")
    assert canonical.PredictionIntervalResult is not attachments.IntervalAttachment
    assert canonical.UncertaintyScoreResult is not attachments.ScoreAttachment
    assert not hasattr(canonical, "IntervalAttachment")
    assert not hasattr(canonical, "ScoreAttachment")


@pytest.mark.parametrize("historical, canonical_name, names, aliases", [
    ("tcspc_toolkit.ml_uncertainty", "tcspc_toolkit.uncertainty.ml", (
        "QUANTILE_GRADIENT_BOOSTING_METHOD_ID",
        "DEFAULT_LOWER_QUANTILE",
        "DEFAULT_MEDIAN_QUANTILE",
        "DEFAULT_UPPER_QUANTILE",
        "DEFAULT_QUANTILE_NOMINAL_COVERAGE",
        "RANDOM_FOREST_TREE_SPREAD_METHOD_ID",
        "ML_TRAINING_BOOTSTRAP_METHOD_ID",
        "DEFAULT_BOOTSTRAP_REPLICATES",
        "project_ml_prediction_intervals",
        "project_ml_uncertainty_scores",
        "QuantileGradientBoostingIntervalEstimator",
        "RandomForestTreeSpreadEstimator",
        "BootstrapPredictionSpreadEstimator",
        "MLUncertaintyScoreCalibrationResult",
        "fit_and_evaluate_random_forest_tree_spread",
        "fit_and_evaluate_ridge_bootstrap_spread",
        "QuantileGradientBoostingCalibrationResult",
        "QuantileGradientBoostingExternalResult",
        "QuantileGradientBoostingRobustnessResult",
        "QuantileGradientBoostingPairedConditionResult",
        "QuantileGradientBoostingPairedRobustnessResult",
        "fit_and_evaluate_quantile_gradient_boosting",
        "evaluate_frozen_quantile_gradient_boosting",
        "evaluate_quantile_interval_robustness",
        "evaluate_paired_quantile_interval_response",
        "MLUncertaintyScoreExternalResult",
        "MLUncertaintyScorePairedConditionResult",
        "MLUncertaintyScorePairedRobustnessResult",
        "evaluate_paired_uncertainty_score_response",
        "evaluate_frozen_ml_uncertainty_score",
        "ML_UNCERTAINTY_METHODS",
        "evaluate_week9_quantile_robustness_conditions",
        "evaluate_week9_paired_quantile_response",
        "evaluate_week9_paired_ml_uncertainty_response",
    ), (
        ("evaluate_week9_quantile_robustness_conditions", "evaluate_quantile_interval_robustness"),
        ("evaluate_week9_paired_quantile_response", "evaluate_paired_quantile_interval_response"),
        ("evaluate_week9_paired_ml_uncertainty_response", "evaluate_paired_uncertainty_score_response"),
    )),
    ("tcspc_toolkit.classical_uncertainty", "tcspc_toolkit.uncertainty.classical", (
        "POISSON_LOCAL_COVARIANCE_METHOD_ID",
        "DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP",
        "DEFAULT_INFORMATION_CONDITION_LIMIT",
        "PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID",
        "REPEATED_POISSON_REFERENCE_METHOD_ID",
        "DEFAULT_CLASSICAL_BOOTSTRAP_REPLICATES",
        "DEFAULT_CLASSICAL_NOMINAL_COVERAGE",
        "CLASSICAL_LOCAL_COVARIANCE_INTERVAL_KIND",
        "CLASSICAL_PARAMETRIC_BOOTSTRAP_INTERVAL_KIND",
        "CLASSICAL_UNCERTAINTY_METHODS",
        "PoissonLocalCovarianceResult",
        "ParametricPoissonBootstrapResult",
        "project_poisson_local_covariance_interval",
        "project_parametric_poisson_bootstrap_interval",
        "estimate_parametric_poisson_bootstrap",
        "RepeatedPoissonUncertaintyResult",
        "evaluate_repeated_poisson_uncertainty",
        "estimate_poisson_reconvolution_local_covariance",
    ), ()),
])
def test_uncertainty_method_canonical_surfaces_preserve_public_surface_and_canonical_pickle(
    historical, canonical_name, names, aliases,
):
    canonical = importlib.import_module(canonical_name)
    namespace = importlib.import_module("tcspc_toolkit.uncertainty")
    assert_surface_after_removal(historical, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        obj = getattr(canonical, name)
        assert not hasattr(namespace, name)
        assert not hasattr(toolkit, name)
        if callable(obj):
            assert obj.__module__ == canonical_name
            assert pickle.loads(pickle.dumps(obj)) is obj
    for alias, name in aliases:
        assert getattr(canonical, alias) is getattr(canonical, name)


@pytest.mark.parametrize("historical, canonical_name, names", [
    ("tcspc_toolkit.mismatch_evaluation", "tcspc_toolkit.benchmarks.mismatch", (
        "EstimatorMismatchResult",
        "ClassicalMismatchResult",
        "generate_matched_biexponential_mismatch_dataset",
        "evaluate_ml_mismatch_benchmark",
        "evaluate_classical_mismatch_benchmark",
        "summarize_mismatch_benchmark",
    )),
    ("tcspc_toolkit.timing_evaluation", "tcspc_toolkit.benchmarks.timing", (
        "FloatArray",
        "InferenceTimingResult",
        "benchmark_fitted_regressor_runtime",
        "benchmark_mean_arrival_runtime",
        "benchmark_ml_inference_runtime",
        "summarize_reconvolution_runtime",
        "benchmark_inference_runtime",
        "summarize_inference_timing",
    )),
])
def test_week7_study_canonical_surfaces_preserve_public_surface_and_removed_paths(
    historical, canonical_name, names,
):
    canonical = importlib.import_module(canonical_name)
    namespace = importlib.import_module("tcspc_toolkit.benchmarks")
    assert_surface_after_removal(historical, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        obj = getattr(canonical, name)
        assert not hasattr(namespace, name)
        assert not hasattr(toolkit, name)
        if name != "FloatArray":
            assert obj.__module__ == canonical_name
            assert pickle.loads(pickle.dumps(obj)) is obj
    if "FloatArray" in names:
        # Preserve the existing numpy typing alias, not a fabricated local type.
        from numpy import float64
        from numpy.typing import NDArray
        assert canonical.FloatArray == NDArray[float64]


@pytest.mark.parametrize("historical, canonical_name, names", [
    ("tcspc_toolkit.irf_evaluation", "tcspc_toolkit.benchmarks.irf", (
        "FloatArray",
        "IntArray",
        "IRFShapeMismatchConfig",
        "IRFShapeMismatchDataset",
        "generate_irf_shape_mismatch_dataset",
        "IRFClassicalPair",
        "IRFClassicalEvaluation",
        "evaluate_irf_shape_classical",
        "summarize_irf_shape_classical",
        "IRFConditionalUncertaintyEvaluation",
        "evaluate_irf_shape_conditional_uncertainty",
        "IRFMLTransferEvaluation",
        "evaluate_gaussian_trained_irf_transfer",
        "LeadingEdgeFailureConfig",
        "LeadingEdgeRegimeResult",
        "LeadingEdgeFailureEvaluation",
        "evaluate_leading_edge_failure_regimes",
        "summarize_leading_edge_regimes",
    )),
    ("tcspc_toolkit.bayesian_evaluation", "tcspc_toolkit.benchmarks.bayesian_calibration", (
        "BayesianClassicalCondition",
        "BayesianClassicalEvaluationConfig",
        "Issue4RealizationSeeds",
        "derive_issue4_seed",
        "issue4_realization_seeds",
        "build_issue4_expected_counts",
        "sample_issue4_observation",
        "Issue4ClassicalFit",
        "Issue4Covariance",
        "Issue4Bootstrap",
        "Issue4Bayesian",
        "BayesianClassicalRealizationResult",
        "evaluate_issue4_realization",
        "Issue4IntervalCoverage",
        "Issue4EstimatorSpread",
        "Issue4RuntimeSummary",
        "Issue4UncertaintyScaleComparison",
        "BayesianClassicalConditionSummary",
        "BayesianCalibrationReport",
        "summarize_issue4_condition",
        "evaluate_issue4_condition",
    )),
    ("tcspc_toolkit.bayesian_mismatch_evaluation", "tcspc_toolkit.benchmarks.bayesian_mismatch", (
        "Mechanism",
        "GeneratingModel",
        "ReferenceName",
        "MismatchAssumption",
        "MismatchCondition",
        "MismatchEvaluationConfig",
        "MismatchWorkflow",
        "build_mismatch_workflow",
        "MismatchSeeds",
        "mismatch_seeds",
        "GeneratingCounts",
        "build_mismatch_expected_counts",
        "PseudoTrueReference",
        "construct_pseudo_true_reference",
        "MismatchReferenceComparison",
        "MismatchPredictiveSummary",
        "MismatchInferenceRecord",
        "MismatchGroupSummary",
        "MismatchReport",
        "sample_mismatch_observation",
        "evaluate_mismatch_inference",
        "summarize_mismatch_records",
        "evaluate_mismatch_workflow",
    )),
])
def test_irf_bayesian_study_canonical_surfaces_preserve_surface_and_removed_paths(
    historical, canonical_name, names,
):
    canonical = importlib.import_module(canonical_name)
    namespace = importlib.import_module("tcspc_toolkit.benchmarks")
    typing_aliases = {"FloatArray", "IntArray", "Mechanism", "GeneratingModel", "ReferenceName"}
    assert_surface_after_removal(historical, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        obj = getattr(canonical, name)
        assert not hasattr(namespace, name)
        assert not hasattr(toolkit, name)
        if name not in typing_aliases:
            assert obj.__module__ == canonical_name
            assert pickle.loads(pickle.dumps(obj)) is obj



def test_week9_benchmark_canonical_surface_preserves_public_alias_and_pickle_identity():
    names = (
        "DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED",
        "CONFORMALIZED_QUANTILE_METHOD_ID",
        "ConformalizedQuantileCalibrationResult",
        "conformalize_quantile_gradient_boosting",
        "ConformalizedQuantileExternalResult",
        "evaluate_frozen_conformalized_quantile",
        "build_ml_interval_scorecard",
        "build_ml_uncertainty_scorecard",
        "MLUncertaintyRobustnessReport",
        "build_ml_uncertainty_robustness_report",
        "build_classical_uncertainty_scorecard",
        "build_classical_conditional_uncertainty_scorecard",
        "ClassicalUncertaintyRobustnessReport",
        "evaluate_classical_uncertainty_robustness",
        "ClassicalResidualMatrixResult",
        "ResidualMismatchReport",
        "evaluate_residual_mismatch_diagnostics",
        "DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED",
        "build_week9_interval_scorecard",
        "build_week9_score_only_scorecard",
        "Week9MLUncertaintyRobustnessReport",
        "build_week9_ml_uncertainty_robustness_report",
        "build_week9_classical_uncertainty_scorecard",
        "build_week9_classical_conditional_scorecard",
        "Week9ClassicalUncertaintyRobustnessReport",
        "evaluate_week9_classical_uncertainty_robustness",
        "Week9ResidualMismatchReport",
        "evaluate_week9_residual_mismatch_diagnostics",
    )
    aliases = (
        ("DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED", "DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED"),
        ("build_week9_interval_scorecard", "build_ml_interval_scorecard"),
        ("build_week9_score_only_scorecard", "build_ml_uncertainty_scorecard"),
        ("Week9MLUncertaintyRobustnessReport", "MLUncertaintyRobustnessReport"),
        ("build_week9_ml_uncertainty_robustness_report", "build_ml_uncertainty_robustness_report"),
        ("build_week9_classical_uncertainty_scorecard", "build_classical_uncertainty_scorecard"),
        ("build_week9_classical_conditional_scorecard", "build_classical_conditional_uncertainty_scorecard"),
        ("Week9ClassicalUncertaintyRobustnessReport", "ClassicalUncertaintyRobustnessReport"),
        ("evaluate_week9_classical_uncertainty_robustness", "evaluate_classical_uncertainty_robustness"),
        ("Week9ResidualMismatchReport", "ResidualMismatchReport"),
        ("evaluate_week9_residual_mismatch_diagnostics", "evaluate_residual_mismatch_diagnostics"),
    )
    historical = "tcspc_toolkit.uncertainty_robustness"
    canonical_name = "tcspc_toolkit.benchmarks.uncertainty"
    canonical = importlib.import_module(canonical_name)
    namespace = importlib.import_module("tcspc_toolkit.benchmarks")
    assert_surface_after_removal(historical, canonical, names)
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        obj = getattr(canonical, name)
        assert not hasattr(namespace, name)
        assert not hasattr(toolkit, name)
        if callable(obj):
            assert obj.__module__ == canonical_name
            assert pickle.loads(pickle.dumps(obj)) is obj
    for alias, name in aliases:
        assert getattr(canonical, alias) is getattr(canonical, name)
    # Test-only private consumers migrate; they do not become compatibility API.
    for private in (
        "_finite_sample_conformal_quantile", "_summarize_signed_residual_profile",
        "_evaluate_classical_uncertainty_test", "_report_interval_rows", "_report_score_rows",
    ):
        assert callable(getattr(canonical, private))


def test_expected_counts_builder_is_in_the_root_contract():
    assert "build_expected_counts_from_irf" in toolkit.__all__
    assert toolkit.build_expected_counts_from_irf is build_expected_counts_from_irf


def test_fresh_root_import_needs_no_sampler_or_persistence(tmp_path):
    # Exercise real import resolution in a fresh interpreter, blocking only the
    # optional sampler. An audit hook detects actual SQLite connection attempts.
    script = """
import importlib.abc
import sys

attempts = []
class NoSampler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'emcee' or fullname.startswith('emcee.'):
            attempts.append(fullname)
            raise ModuleNotFoundError('optional sampler unavailable', name=fullname)
        return None

def forbid_database_connection(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('root import must not initialize persistence')

sys.meta_path.insert(0, NoSampler())
sys.addaudithook(forbid_database_connection)
assert 'tcspc_toolkit' not in sys.modules
import tcspc_toolkit
from tcspc_toolkit import EstimatorSpec, fit_regressors, predict_regressors
assert all(hasattr(tcspc_toolkit, name) for name in tcspc_toolkit.__all__)
assert 'tcspc_toolkit.persistence' not in sys.modules
assert 'emcee' not in sys.modules
assert attempts == []
# Confirm the sampler really is unavailable in this process, even if installed.
try:
    import emcee
except ModuleNotFoundError as error:
    assert error.name == 'emcee'
else:
    raise AssertionError('sampler blocker was not exercised')
assert attempts == ['emcee']
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=tmp_path,
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []
