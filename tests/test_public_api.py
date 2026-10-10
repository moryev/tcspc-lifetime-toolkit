"""Curated package-root exports and import-time optional boundaries."""

import importlib
import pickle
import subprocess
import sys

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


def assert_same_export(old_module, canonical_module, name):
    """Reuse for old-flat -> new-canonical cases as each module actually moves.

    Compatibility exports are the same objects, not forwarding wrappers. Stage
    0 established the root/flat pattern; migrated cases now name their actual
    canonical modules explicitly.
    """
    old = importlib.import_module(old_module)
    canonical = importlib.import_module(canonical_module)
    assert getattr(old, name) is getattr(canonical, name)


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
def test_relocated_exports_keep_direct_identity_and_historical_lookup(old_module, canonical_module, names):
    canonical = importlib.import_module(canonical_module)
    for name in names:
        assert_same_export(old_module, canonical_module, name)
        if name != "FloatArray":
            obj = getattr(canonical, name)
            assert obj.__module__ == canonical_module
            # A trusted protocol-0 GLOBAL lookup exercises the historical path
            # without pretending that new objects retain old __module__ values.
            assert pickle.loads(f"c{old_module}\n{name}\n.".encode()) is obj


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
def test_preprocessing_facades_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.preprocessing.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    for name in names:
        assert_same_export(old_module, canonical_module, name)
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
def test_irf_facades_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.irf.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    for name in names:
        assert_same_export(old_module, canonical_module, name)
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
def test_measurement_facades_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.measurement.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    for name in names:
        assert_same_export(old_module, canonical_module, name)
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
def test_simulation_facades_preserve_supported_objects_and_root_identity(old_module, module, names):
    canonical_module = f"tcspc_toolkit.simulation.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    for name in names:
        assert_same_export(old_module, canonical_module, name)
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
def test_classical_facades_preserve_supported_objects_and_historical_lookup(old_module, module, names):
    canonical_module = f"tcspc_toolkit.estimation.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    for name in names:
        assert_same_export(old_module, canonical_module, name)
        assert getattr(canonical, name).__module__ == canonical_module
        if name in EXPECTED_ROOT_EXPORTS:
            assert_same_export("tcspc_toolkit", canonical_module, name)
        if name in {"ReconvolutionFitResult", "ReconvolutionCurveResult"}:
            # Historical GLOBAL references still resolve, while newly pickled
            # objects naturally use the canonical implementation module.
            assert pickle.loads(f"c{old_module}\n{name}\n.".encode()) is getattr(canonical, name)


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
def test_bayesian_facades_preserve_supported_objects_and_historical_lookup(old_module, module, names):
    canonical_module = f"tcspc_toolkit.estimation.bayesian.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        assert_same_export(old_module, canonical_module, name)
        if not name.isupper():
            assert getattr(canonical, name).__module__ == canonical_module
    if module == "sampling":
        # Persisted historical source identity remains resolvable without
        # forcing new objects to pretend to be defined in the old module.
        assert pickle.loads(
            b"ctcspc_toolkit.bayesian_sampling\nBayesianReconvolutionResult\n."
        ) is canonical.BayesianReconvolutionResult


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
def test_ml_facades_preserve_supported_objects_and_historical_lookup(old_module, module, names):
    canonical_module = f"tcspc_toolkit.ml.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    assert tuple(facade.__all__) == names
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        assert_same_export(old_module, canonical_module, name)
        if not name.isupper() and name not in {"FloatArray", "IntArray"}:
            assert getattr(canonical, name).__module__ == canonical_module
    if module == "evaluation":
        assert pickle.loads(
            b"ctcspc_toolkit.ml_evaluation\nRegressionBenchmarkResult\n."
        ) is canonical.RegressionBenchmarkResult


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
def test_factual_facades_preserve_supported_objects_without_expanding_root(old_module, module, names):
    canonical_module = f"tcspc_toolkit.evaluation.{module}"
    canonical = importlib.import_module(canonical_module)
    facade = importlib.import_module(old_module)
    diagnostics_facade = importlib.import_module("tcspc_toolkit.evaluation")
    assert tuple(canonical.__all__) == tuple(facade.__all__) == names
    assert not set(names).intersection(EXPECTED_ROOT_EXPORTS)
    for name in names:
        assert_same_export(old_module, canonical_module, name)
        assert getattr(canonical, name).__module__ == canonical_module
        assert not hasattr(diagnostics_facade, name)
        assert not hasattr(toolkit, name)


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
