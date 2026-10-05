"""Portable preparation expectations, verified before the shared-helper refactor.

The oracle uses unchanged single-histogram features, direct TOTAL division and
sklearn's explicit full-SVD PCA, not either preparation function or its helper.
No dependency-version-specific fitted values are stored.
"""

from copy import deepcopy
from dataclasses import fields, replace
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.features import extract_features
from tcspc_toolkit.generalization import default_generalization_suite
from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
from tcspc_toolkit import generalization_evaluation as evaluation
from tcspc_toolkit.ml_evaluation import BenchmarkMeasurements


CONFIG = FeatureConfig(tail_start_ns=2.0, early_stop_ns=2.0, late_start_ns=3.0)
FEATURE_COLUMNS = (
    "total_counts", "peak_height", "peak_time_ns", "mean_arrival_time_ns",
    "arrival_time_variance_ns2", "arrival_time_skewness", "t10_ns", "t25_ns",
    "t50_ns", "t75_ns", "t90_ns", "half_decay_time_ns", "tail_log_slope_per_ns",
    "integrated_tail_fraction", "early_late_count_ratio",
)
AB_FIELDS = (
    "development", "test_a", "test_b", "X_features_a", "X_features_b",
    "X_normalized_development", "X_normalized_a", "X_normalized_b",
    "X_pca_development", "X_pca_a", "X_pca_b", "pca",
)
MULTI_FIELDS = (
    "development", "tests", "X_features", "X_normalized_development",
    "X_normalized", "X_pca_development", "X_pca", "pca",
)


@pytest.fixture
def measurements():
    time = np.arange(16, dtype=np.float64) / 2
    lifetimes = np.linspace(0.8, 2.0, 12)
    counts = np.maximum(np.rint(120 * np.exp(-time / lifetimes[:, None])), 1).astype(np.int64)
    development = BenchmarkMeasurements(
        time, counts, lifetimes,
        pd.DataFrame({"sample_id": np.arange(12)[::-1], "role": "development"}, index=np.arange(100, 112)),
    )
    definition = default_generalization_suite()
    tests = {}
    targets = np.array([1.7, 0.9, 2.2])
    for offset, test_id in enumerate("ABCDEF"):
        histograms = np.maximum(np.rint((60 + offset * 20) * np.exp(-time / targets[:, None])), 1).astype(np.int64)
        metadata = pd.DataFrame({
            "sample_id": [7, 3, 11], "pair_id": [7, 3, 11], "test_id": test_id,
            "suite_version": definition.suite_version,
            "development_reference": definition.development_reference,
            "random_seed": definition.seed_for(test_id),
        }, index=[90, 30, 70])
        if test_id == "F":
            metadata["tau_1_ns"] = targets
            metadata["tau_2_ns"] = targets * 2
            metadata["photon_weighted_lifetime_ns"] = targets * 1.4
        tests[test_id] = GeneralizationTestMeasurements(test_id, time.copy(), histograms, targets.copy(), metadata)
    return development, tests


def _prepare(kind, development, tests, n_components=2):
    options = dict(development_measurements=development, feature_config=CONFIG)
    if n_components is not None:
        options["n_pca_components"] = n_components
    if kind == "ab":
        return evaluation.prepare_generalization_ab_data(test_a=tests["A"], test_b=tests["B"], **options)
    return evaluation.prepare_generalization_data(tests=tests, **options)


def _test_views(prepared):
    if isinstance(prepared, evaluation.GeneralizationABPreparedData):
        return {
            "A": (prepared.test_a, prepared.X_features_a, prepared.X_normalized_a, prepared.X_pca_a),
            "B": (prepared.test_b, prepared.X_features_b, prepared.X_normalized_b, prepared.X_pca_b),
        }
    return {key: (test, prepared.X_features[key], prepared.X_normalized[key], prepared.X_pca[key])
            for key, test in prepared.tests.items()}


def _expected_features(measurement):
    return pd.concat([
        extract_features(measurement.time, row, CONFIG) for row in measurement.X_histograms
    ], ignore_index=True)


def _normalized(measurement):
    counts = measurement.X_histograms.astype(np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def _assert_pca_equal(actual, expected):
    assert actual.get_params() == expected.get_params()
    for name in ("mean_", "components_", "explained_variance_", "explained_variance_ratio_", "singular_values_"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), strict=True)
    for name in ("n_samples_", "n_features_in_", "n_components_", "noise_variance_"):
        assert getattr(actual, name) == getattr(expected, name)


@pytest.mark.parametrize("kind", ["ab", "multi"])
@pytest.mark.parametrize("n_components", [2, None], ids=["two-components", "legacy-default"])
def test_preparation_matches_independent_outputs_and_preserves_sources(measurements, kind, n_components):
    development, tests = measurements
    before = deepcopy(measurements)
    for measurement in (development, *tests.values()):
        for array in (measurement.time, measurement.X_histograms, measurement.y):
            array.setflags(write=False)
    # Multi-test preparation preserves caller order and normalizes keys; it does
    # not sort arbitrary subsets into suite order. The frozen suite is separate.
    supplied = {f" {key.lower()} ": tests[key] for key in "FACBED"} if kind == "multi" else tests
    prepared = _prepare(kind, development, supplied, n_components)
    expected_type = evaluation.GeneralizationABPreparedData if kind == "ab" else evaluation.GeneralizationPreparedData
    expected_fields = AB_FIELDS if kind == "ab" else MULTI_FIELDS
    assert type(prepared) is expected_type
    assert tuple(field.name for field in fields(prepared)) == expected_fields
    reconstructed = expected_type(*(getattr(prepared, name) for name in expected_fields))
    assert reconstructed.pca is prepared.pca
    assert reconstructed.development is prepared.development

    pd.testing.assert_frame_equal(prepared.development.X_features, _expected_features(development), check_exact=True)
    assert tuple(prepared.development.X_features.columns) == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(prepared.development.metadata, development.metadata, check_exact=True)
    np.testing.assert_array_equal(prepared.development.X_histograms, development.X_histograms, strict=True)
    np.testing.assert_array_equal(prepared.development.y, development.y, strict=True)
    assert not np.shares_memory(prepared.development.X_histograms, development.X_histograms)
    assert not np.shares_memory(prepared.development.y, development.y)
    expected_normalized = _normalized(development)
    np.testing.assert_array_equal(prepared.X_normalized_development, expected_normalized, strict=True)
    expected_pca = PCA(n_components=10 if n_components is None else n_components, svd_solver="full").fit(expected_normalized)
    _assert_pca_equal(prepared.pca, expected_pca)
    np.testing.assert_array_equal(prepared.X_pca_development, expected_pca.transform(expected_normalized), strict=True)

    views = _test_views(prepared)
    assert tuple(views) == tuple("AB" if kind == "ab" else "FACBED")
    if kind == "multi":
        assert tuple(prepared.X_features) == tuple(prepared.X_normalized) == tuple(prepared.X_pca) == tuple(views)
    for key, (test, features, normalized, transformed) in views.items():
        assert test is tests[key]
        assert tuple(features.columns) == FEATURE_COLUMNS
        pd.testing.assert_frame_equal(features, _expected_features(test), check_exact=True)
        np.testing.assert_array_equal(normalized, _normalized(test), strict=True)
        np.testing.assert_array_equal(transformed, expected_pca.transform(_normalized(test)), strict=True)
        # The public artifact is the same development fit used for every test.
        np.testing.assert_array_equal(transformed, prepared.pca.transform(normalized), strict=True)
    for original, saved in zip((development, *tests.values()), (before[0], *before[1].values())):
        for name in ("time", "X_histograms", "y"):
            np.testing.assert_array_equal(getattr(original, name), getattr(saved, name), strict=True)
        pd.testing.assert_frame_equal(original.metadata, saved.metadata, check_exact=True)


@pytest.mark.parametrize("kind", ["ab", "multi"])
@pytest.mark.parametrize("change", ["histograms", "targets-and-metadata"])
def test_final_test_changes_cannot_affect_development_pca(measurements, kind, change):
    development, tests = measurements
    original = _prepare(kind, development, tests)
    changed = deepcopy(tests)
    for test in changed.values():
        if change == "histograms":
            test.X_histograms[:] = test.X_histograms * 3 + np.arange(test.time.size)
        else:
            test.y[:] *= 3
            test.metadata["sample_id"] += 1000
            test.metadata["total_counts"] = -999
            test.metadata["mean_arrival_time_ns"] = -999
    updated = _prepare(kind, development, changed)
    _assert_pca_equal(updated.pca, original.pca)
    np.testing.assert_array_equal(updated.X_pca_development, original.X_pca_development, strict=True)
    pd.testing.assert_frame_equal(updated.development.X_features, original.development.X_features, check_exact=True)
    if change == "targets-and-metadata":
        for key, (_, features, normalized, transformed) in _test_views(updated).items():
            previous = _test_views(original)[key]
            pd.testing.assert_frame_equal(features, previous[1], check_exact=True)
            np.testing.assert_array_equal(normalized, previous[2], strict=True)
            np.testing.assert_array_equal(transformed, previous[3], strict=True)
    else:
        assert not np.array_equal(updated.X_pca_a if kind == "ab" else updated.X_pca["A"],
                                  original.X_pca_a if kind == "ab" else original.X_pca["A"])


@pytest.mark.parametrize("invalid, message", [
    ("identity-a", "test_a must contain robustness Test A."),
    ("identity-b", "test_b must contain robustness Test B."),
    ("time-a", "Development data and Test A must share the same time axis."),
    ("time-b", "Development data and Test B must share the same time axis."),
    ("pairing", "Tests A and B must preserve paired lifetime targets."),
])
def test_ab_specific_guards_remain_explicit(measurements, invalid, message):
    development, tests = measurements
    if invalid.startswith("identity"):
        tests[invalid[-1].upper()] = tests["C"]
    elif invalid.startswith("time"):
        key = invalid[-1].upper()
        tests[key] = replace(tests[key], time=tests[key].time + 0.1)
    else:
        tests["B"] = replace(tests["B"], y=tests["B"].y[::-1].copy())
    with pytest.raises(ValueError, match=re.escape(message)):
        _prepare("ab", development, tests)


@pytest.mark.parametrize("invalid, message", [
    ("empty", "tests must contain at least one generalization test."),
    ("duplicate", "Generalization test IDs must be unique."),
    ("key", "Dictionary key and test.test_id must match."),
    ("time", "Development data and all final tests must share the same time axis."),
])
def test_multi_test_guards_remain_unchanged(measurements, invalid, message):
    development, tests = measurements
    if invalid == "empty":
        tests = {}
    elif invalid == "duplicate":
        tests = {"A": tests["A"], " a ": tests["A"]}
    elif invalid == "key":
        tests = {"B": tests["A"]}
    else:
        tests = {"A": replace(tests["A"], time=tests["A"].time + 0.1)}
    with pytest.raises(ValueError, match=re.escape(message)):
        _prepare("multi", development, tests)


@pytest.mark.parametrize("kind, test_id, message", [
    ("ab", "A", "Development and Test-A engineered feature schemas do not match."),
    ("ab", "B", "Development and Test-B engineered feature schemas do not match."),
    ("multi", "C", "Development and final-test feature schemas do not match."),
])
def test_schema_checks_preserve_the_legacy_diagnostics(measurements, monkeypatch, kind, test_id, message):
    development, tests = measurements
    original = evaluation.extract_feature_table

    def mismatched(**kwargs):
        table = original(**kwargs)
        if kwargs["histograms"] is tests[test_id].X_histograms:
            table = table.rename(columns={"total_counts": "changed"})
        return table

    monkeypatch.setattr(evaluation, "extract_feature_table", mismatched)
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _prepare(kind, development, tests)
