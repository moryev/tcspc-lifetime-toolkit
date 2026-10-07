"""Curated package-root exports and import-time optional boundaries."""

import importlib
import subprocess
import sys

import pytest

import tcspc_toolkit as toolkit
from tcspc_toolkit import estimator_api
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
    0 exercises existing root/flat paths; it invents no future modules.
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
