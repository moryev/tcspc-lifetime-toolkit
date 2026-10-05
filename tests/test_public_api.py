"""Curated package-root exports and import-time optional boundaries."""

import subprocess
import sys

import pytest

import tcspc_toolkit as toolkit
from tcspc_toolkit import estimator_api
from tcspc_toolkit.simulation import build_expected_counts_from_irf


def test_all_exports_are_unique_and_accessible():
    assert len(toolkit.__all__) == len(set(toolkit.__all__))
    for name in toolkit.__all__:
        assert getattr(toolkit, name) is not None, name


@pytest.mark.parametrize("name", [
    "RegressorProtocol", "EstimatorSpec", "fit_regressors", "predict_regressors",
])
def test_extension_exports_are_canonical_objects(name):
    assert name in toolkit.__all__
    assert getattr(toolkit, name) is getattr(estimator_api, name)


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
