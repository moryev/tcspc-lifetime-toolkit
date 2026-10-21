"""Explicit scientific acceptance; never executes notebooks or updates references.

Run with the project interpreter, e.g. ``python scripts/run_issue12_scientific_reproducibility.py
--all --reference issue12-post-correction``. Reports go to a new system temporary directory. This is deliberately
outside ordinary pytest: full classical traversal can take several minutes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
import hashlib
from html.parser import HTMLParser
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VERSION = "issue12-stage3b2-v1"
REFERENCE_PATH = ROOT / "data/reference/issue12_week7_week8_post_correction.json"
REFERENCE_ID = "issue12-week7-week8-post-correction"
WEEK9_REFERENCE_PATH = ROOT / "data/reference/issue12_week9_ml_uncertainty_post_correction.json"
WEEK9_REFERENCE_ID = "issue12-week9-ml-uncertainty-post-correction"
CLASSICAL_REFERENCE_PATH = ROOT / "data/reference/issue12_week9_classical_uncertainty_post_correction.json"
CLASSICAL_REFERENCE_ID = "issue12-week9-classical-uncertainty-post-correction"
CLASSICAL_PROFILE = "issue12-week9-classical-full-v1"
DAY61_CONDITIONS = (
    ("1k photons / low background", 2., 1000, .5),
    ("10k photons / low background", 2., 10000, .5),
    ("100k photons / low background", 2., 100000, .5),
    ("1k photons / elevated background", 2., 1000, 5.),
    ("10k photons / elevated background", 2., 10000, 5.),
    ("100k photons / elevated background", 2., 100000, 5.),
    ("1 ns lifetime", 1., 10000, .5),
    ("4 ns lifetime", 4., 10000, .5),
)
METHODS = ("ridge", "random_forest", "hist_gradient_boosting")
REPRESENTATIONS = ("engineered_features", "normalized_histogram", "pca_histogram")
TEST_IDS = tuple("ABCDEF")
NOTEBOOKS = {
    7: "12_ml_benchmarking.ipynb",
    8: "13_generalization_and_robustness.ipynb",
    9: "14_uncertainty_and_failure_awareness.ipynb",
}


class AcceptanceFailure(RuntimeError):
    """A discrepancy stops the affected workflow without changing its oracle."""


def require(condition, message):
    if not condition:
        raise AcceptanceFailure(message)


REMOVED_FLAT_MODULE_NAMES = (
    "tcspc_toolkit.baselines",
    "tcspc_toolkit.bayesian",
    "tcspc_toolkit.bayesian_evaluation",
    "tcspc_toolkit.bayesian_mismatch_evaluation",
    "tcspc_toolkit.bayesian_predictive",
    "tcspc_toolkit.bayesian_sampling",
    "tcspc_toolkit.benchmark_plots",
    "tcspc_toolkit.classical_evaluation",
    "tcspc_toolkit.classical_uncertainty",
    "tcspc_toolkit.conditional_evaluation",
    "tcspc_toolkit.convolution",
    "tcspc_toolkit.cross_validation",
    "tcspc_toolkit.datasets",
    "tcspc_toolkit.estimator_api",
    "tcspc_toolkit.evaluation_core",
    "tcspc_toolkit.evaluation_results",
    "tcspc_toolkit.evaluation_uncertainty",
    "tcspc_toolkit.experimental",
    "tcspc_toolkit.features",
    "tcspc_toolkit.fitting",
    "tcspc_toolkit.forward_model",
    "tcspc_toolkit.generalization",
    "tcspc_toolkit.generalization_datasets",
    "tcspc_toolkit.generalization_evaluation",
    "tcspc_toolkit.irf_estimation",
    "tcspc_toolkit.irf_evaluation",
    "tcspc_toolkit.irf_preparation",
    "tcspc_toolkit.measurement_io",
    "tcspc_toolkit.measurements",
    "tcspc_toolkit.mismatch_evaluation",
    "tcspc_toolkit.ml_evaluation",
    "tcspc_toolkit.ml_models",
    "tcspc_toolkit.ml_uncertainty",
    "tcspc_toolkit.models",
    "tcspc_toolkit.representations",
    "tcspc_toolkit.timing_evaluation",
    "tcspc_toolkit.uncertainty_evaluation",
    "tcspc_toolkit.uncertainty_robustness",
)


def flat_facades():
    """Remember the 38 removed paths, independently of installed source files."""
    return set(REMOVED_FLAT_MODULE_NAMES)


class BlockFlatFacades(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.names = flat_facades()

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names:
            raise ImportError(f"Scientific acceptance imported a flat facade: {fullname}")
        return None


class BodyRows(HTMLParser):
    """Read only saved table body cells; no notebook source is evaluated."""
    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_body = False
        self.cell = None
        self.row = []

    def handle_starttag(self, tag, attrs):
        if tag == "tbody":
            self.in_body = True
        elif self.in_body and tag == "tr":
            self.row = []
        elif self.in_body and tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == "tr" and self.in_body:
            self.rows.append(self.row)
        elif tag == "tbody":
            self.in_body = False


def reference_rows(week, cell_id, table_index=None):
    path = ROOT / "notebooks" / NOTEBOOKS[week]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == cell_id)
    tables = [o["data"]["text/html"] for o in cell.get("outputs", [])
              if "text/html" in o.get("data", {})]
    if table_index is None:
        require(len(tables) == 1, f"Expected one saved reference table at {path.name}:{cell_id}")
        table_index = 0
    require(0 <= table_index < len(tables), f"Missing saved table: {cell_id}/{table_index}")
    parser = BodyRows()
    parser.feed("".join(tables[table_index]))
    require(bool(parser.rows), f"Empty reference table: {cell_id}")
    return parser.rows


def compare(report, label, observed, expected, decimals):
    """Half a displayed decimal unit, not a version-independent bitwise oracle."""
    import math

    tolerance = 0.5 * 10 ** (-decimals)
    difference = abs(float(observed) - float(expected))
    passed = math.isfinite(float(observed)) and difference <= tolerance + 1e-14
    report.setdefault("comparisons", []).append({
        "label": label, "expected": float(expected),
        "observed": float(observed) if math.isfinite(float(observed)) else None,
        "absolute_difference": difference if math.isfinite(difference) else None,
        "relative_difference": difference / abs(float(expected)) if expected and math.isfinite(difference) else None,
        "absolute_tolerance": tolerance, "status": "PASS" if passed else "FAIL",
    })


def stop_on_discrepancy(report, phase):
    failed = [r for r in report.get("comparisons", []) if r["status"] == "FAIL"]
    if failed:
        report["first_discrepancy_phase"] = phase
        raise AcceptanceFailure(f"{phase}: {len(failed)} reference comparisons failed; see JSON")


def records(frame):
    # pandas encodes missing/nonfinite diagnostic cells as JSON null.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def fingerprint(measurement):
    digest = hashlib.sha256()
    for array in (measurement.time, measurement.X_histograms, measurement.y):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    digest.update(measurement.metadata.to_json(orient="split", double_precision=15).encode())
    return digest.hexdigest()


def same_measurements(first, second):
    import numpy as np
    import pandas as pd

    for name in ("time", "X_histograms", "y"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), strict=True)
    pd.testing.assert_frame_equal(first.metadata, second.metadata, check_exact=True)


def frame_fingerprint(frame):
    """Exact deterministic values/order, excluding runtime only at the caller."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode())
    for name in frame:
        values = frame[name].to_numpy()
        digest.update(str(values.dtype).encode())
        if values.dtype.kind in "biufc":
            digest.update(values.tobytes())
        else:
            digest.update(json.dumps(values.tolist(), default=str).encode())
    return digest.hexdigest()


def population(frame):
    valid = frame["valid_fit"].to_numpy(dtype=bool)
    failures = frame.loc[~valid, ["sample_id", "failure_reason"]]
    return dict(n_total=len(frame), n_valid=int(valid.sum()), n_failed=int((~valid).sum()),
                failures=records(failures), metric_population="valid fits only")


def paired_error_change(first, second):
    """Explicit complete-pair reporting, never a ratio of unpaired populations."""
    import numpy as np
    np.testing.assert_array_equal(first["sample_id"], second["sample_id"])
    valid = first["valid_fit"].to_numpy(bool) & second["valid_fit"].to_numpy(bool)
    change = (np.abs(second["error_ns"].to_numpy()[valid])
              - np.abs(first["error_ns"].to_numpy()[valid]))
    require(len(change) > 0 and np.isfinite(change).all(), "No finite complete pairs")
    return dict(n_complete_pairs=len(change), mean_absolute_error_change_ns=float(change.mean()),
                median_absolute_error_change_ns=float(np.median(change)),
                n_worse=int(sum(change > 0)), n_better=int(sum(change < 0)),
                n_equal=int(sum(change == 0)))


def compact_scorecard(summary):
    """Select canonical summary metrics; attach counts and descriptive A ratios."""
    rows = []
    for row in records(summary):
        baseline = summary.loc[(summary.estimator == row["estimator"])
                               & (summary.representation == row["representation"])
                               & (summary.test_id == "A"), "mae_ns"].item()
        rows.append({k: row[k] for k in ("estimator", "representation", "test_id", "mae_ns", "bias_ns")}
                    | dict(n_total=row["n_total_samples"], n_valid=row["n_valid_predictions"],
                           n_failed=row["n_total_samples"] - row["n_valid_predictions"],
                           failure_rate=1 - row["n_valid_predictions"] / row["n_total_samples"],
                           mae_ratio_over_A=row["mae_ns"] / baseline))
    return rows


def read_current_reference():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    require(reference["reference_id"] == REFERENCE_ID and reference["reference_version"] == 1,
            "Unsupported current reference")
    require(set(reference["weeks"]) == {"week7", "week8"}, "Incomplete reference")
    return reference


def compare_current(observed, expected, tolerances, path=""):
    """Strict schema/order/identities; named metric tolerances, no default slack."""
    import math
    failures = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            return [f"{path}: keys differ"]
        for key in expected:
            failures.extend(compare_current(observed[key], expected[key], tolerances, f"{path}/{key}"))
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return [f"{path}: length differs"]
        for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
            failures.extend(compare_current(actual, wanted, tolerances, f"{path}/{index}"))
    elif isinstance(expected, float) and path.rsplit("/", 1)[-1] in tolerances:
        allowance = tolerances[path.rsplit("/", 1)[-1]]
        if (not isinstance(observed, (float, int)) or not math.isfinite(observed)
                or abs(observed - expected) > allowance):
            failures.append(f"{path}: expected {expected}, observed {observed}, atol {allowance}")
    elif type(observed) is not type(expected) or observed != expected:
        failures.append(f"{path}: expected {expected!r}, observed {observed!r} (exact)")
    return failures


def verify_repeat(report, previous):
    require(report["versions"] == previous["versions"], "Exact rerun requires the same environment")
    for week, result in report["workflows"].items():
        prior = previous["workflows"][week]
        require(result["science"] == prior["science"], f"{week}: nondeterministic scientific metrics/identity")
        require(result["repeatability"] == prior["repeatability"],
                f"{week}: nondeterministic arrays, metadata, predictions or classical parameters/NLL")
    report["deterministic_rerun"] = "PASS"


def week7(report):
    import numpy as np
    import pandas as pd
    from tcspc_toolkit.config import FeatureConfig
    from tcspc_toolkit.irf.profiles import generate_gaussian_irf, normalize_irf
    from tcspc_toolkit.ml.evaluation import (
        generate_benchmark_measurements, build_benchmark_dataset, split_benchmark_dataset,
    )
    from tcspc_toolkit.benchmarks.mismatch import (
        generate_matched_biexponential_mismatch_dataset, evaluate_ml_mismatch_benchmark,
        evaluate_classical_mismatch_benchmark, summarize_mismatch_benchmark,
    )
    from tcspc_toolkit.benchmarks.timing import (
        benchmark_inference_runtime, summarize_inference_timing,
    )

    report["protocol"] = {
        "source": NOTEBOOKS[7], "reference_cell": "83354e7feeb8b834",
        "simulation_seed": 42, "split_seed": 42, "mismatch_seed": 42,
        "secondary_fraction": .10, "secondary_lifetime_factor": 2.,
        "references": ["generating_mono", "primary_component"],
        "representation": "engineered_features", "method_ids": list(METHODS),
    }
    time = np.arange(0., 20., .05, dtype=np.float64)
    options = dict(time=time, lifetimes_ns=[.5, 1., 2., 3., 4.5, 6.],
                   signal_photon_counts=[1000, 10000, 100000], background_levels=[0., .5, 2.],
                   irf_centre_ns=1., irf_fwhm_values_ns=[.15, .30, .60],
                   irf_shift_values_ns=[-.20, -.05, 0., .05, .20], random_seed=42)
    measured = generate_benchmark_measurements(**options)
    same_measurements(measured, generate_benchmark_measurements(**options))
    require(measured.X_histograms.shape == (810, 400), "Week-7 geometry")
    config = FeatureConfig(tail_start_ns=2., early_stop_ns=2., late_start_ns=3.)
    split = split_benchmark_dataset(build_benchmark_dataset(measured, feature_config=config),
                                    test_size=.20, random_state=42)
    require((len(split.y_train), len(split.y_test)) == (648, 162), "Week-7 split")
    require(not set(split.train_indices) & set(split.test_indices), "Train/test overlap")
    options = dict(time=time, reference_split=split, feature_config=config, irf_centre_ns=1.,
                   secondary_fraction=.10, secondary_lifetime_factor=2., random_seed=42)
    mismatch = generate_matched_biexponential_mismatch_dataset(**options)
    repeated = generate_matched_biexponential_mismatch_dataset(**options)
    np.testing.assert_array_equal(mismatch.X_histograms, repeated.X_histograms)
    pd.testing.assert_frame_equal(mismatch.metadata, repeated.metadata, check_exact=True)
    np.testing.assert_array_equal(mismatch.y, split.y_test)
    for column in ("signal_photon_count_target", "background_per_bin", "irf_fwhm_ns", "irf_shift_ns"):
        np.testing.assert_array_equal(mismatch.metadata[column], split.metadata_test[column])
    report.update(dataset_status="PASS", geometry=[810, 400], split=[648, 162],
                  data_sha256=fingerprint(measured), deterministic_data="exact repeat",
                  mismatch_counts_sha256=hashlib.sha256(mismatch.X_histograms.tobytes()).hexdigest())
    print("Week 7: deterministic data PASS; fitting paired ML", flush=True)
    ml = evaluate_ml_mismatch_benchmark(reference_split=split, mismatch_dataset=mismatch)
    repeat_ml = evaluate_ml_mismatch_benchmark(reference_split=split, mismatch_dataset=mismatch)
    require(tuple(ml) == METHODS, "Week-7 method ordering")
    for method in ml:
        for condition in ("in_distribution", "mismatch"):
            np.testing.assert_array_equal(getattr(ml[method], condition).y_pred,
                                          getattr(repeat_ml[method], condition).y_pred)
    table = summarize_mismatch_benchmark(y_true=split.y_test, ml_results=ml).set_index("estimator")
    report["ml_metrics"] = records(table.reset_index())
    report["deterministic_ml"] = "exact repeat of fitted predictions"
    report["repeatability"] = {
        f"{method}/{condition}": hashlib.sha256(getattr(result, condition).y_pred.tobytes()).hexdigest()
        for method, result in ml.items() for condition in ("in_distribution", "mismatch")
    }
    report["repeatability"]["metadata"] = frame_fingerprint(measured.metadata)
    report["repeatability"]["mismatch_metadata"] = frame_fingerprint(mismatch.metadata)
    expected = reference_rows(7, "83354e7feeb8b834")
    columns = list(table.columns)
    require([r[1] for r in expected] == [*METHODS, "reconvolution"], "Week-7 reference ordering")
    for row in expected[:-1]:
        for column, value in zip(columns, row[2:], strict=True):
            compare(report, f"{row[1]}/{column}", table.loc[row[1], column], float(value), 6)
    stop_on_discrepancy(report, "Week-7 ML/reporting")
    report["ml_invariance"] = dict(status="PASS", comparisons=len(report["comparisons"]))
    print("Week 7: ML reference PASS; fitting 324 classical curves", flush=True)
    irf = normalize_irf(time=time, irf=generate_gaussian_irf(time=time, centre=1., fwhm=.30))
    classical = evaluate_classical_mismatch_benchmark(
        time=time, reference_split=split, mismatch_dataset=mismatch, irf=irf,
        temporal_shift_bounds=(-.5, .5), objective="poisson")
    table = summarize_mismatch_benchmark(y_true=split.y_test, ml_results=ml,
                                         classical_result=classical).set_index("estimator")
    report["paired_metrics"] = records(table.reset_index())
    for column, value in zip(columns, expected[-1][2:], strict=True):
        compare(report, f"reconvolution/{column}", table.loc["reconvolution", column], float(value), 6)
    report["classical_failures"] = {
        label: records(result.per_curve.groupby("failure_reason", dropna=False).size().reset_index(name="count"))
        for label, result in (("matched", classical.in_distribution), ("mismatch", classical.mismatch))
    }
    report["historical_notebook_comparison"] = report.pop("comparisons")
    row = table.loc["reconvolution"]
    report["science"] = {
        "protocol": report["protocol"], "dataset_sha256": report["data_sha256"],
        "mismatch_counts_sha256": report["mismatch_counts_sha256"],
        "n_development": 810, "n_train": 648, "n_test": 162, "n_bins": 400,
        "matched": population(classical.in_distribution.per_curve) | {
            "mae_ns": float(row.in_distribution_mae_ns), "bias_ns": float(row.in_distribution_bias_ns)},
        "mismatch": population(classical.mismatch.per_curve) | {
            "mae_ns": float(row.mismatch_mae_ns), "bias_ns": float(row.mismatch_bias_ns)},
        "paired": paired_error_change(classical.in_distribution.per_curve, classical.mismatch.per_curve),
        "mae_ratio": float(row.mae_ratio), "ml_metrics": report["ml_metrics"],
    }
    for label, result in (("matched", classical.in_distribution), ("mismatch", classical.mismatch)):
        report["repeatability"][label + "_classical"] = frame_fingerprint(result.per_curve.drop(columns="runtime_ms"))
    timing = benchmark_inference_runtime(split=split, reconvolution_result=classical.in_distribution,
                                        n_repeats=200, warmup_runs=10)
    require(tuple(timing) == ("mean_arrival_time", *METHODS, "reconvolution"), "Timing ordering")
    timing_table = summarize_inference_timing(timing)
    require(np.isfinite(timing_table["median_time_per_curve_ms"]).all(), "Nonfinite timing")
    require((timing_table["median_time_per_curve_ms"] >= 0).all(), "Negative timing")
    require(set(timing_table["timing_mode"]) == {"repeated_batch", "per_curve_optimization"}, "Timing scopes")
    for name, result in timing.items():
        batch = name != "reconvolution"
        require(result.n_curves_per_call == (162 if batch else 1), "Timing batch/fit scope")
        require(result.n_measurements == (200 if batch else 162), "Timing repetitions")
        require(result.warmup_runs == (10 if batch else 0), "Timing warmup")
        require(np.isfinite(result.runtime_per_curve_ms).all()
                and (result.runtime_per_curve_ms >= 0).all(), "Invalid raw timing")
    np.testing.assert_array_equal(timing["reconvolution"].runtime_per_curve_ms,
                                  classical.in_distribution.per_curve.runtime_ms.to_numpy())
    report["timing_structure"] = "PASS"
    report["timing_method_order"] = list(timing)
    report["timing"] = records(timing_table)


def week8(report):
    import numpy as np
    import pandas as pd
    from tcspc_toolkit.config import FeatureConfig
    from tcspc_toolkit.benchmarks.generalization_protocol import default_generalization_suite
    from tcspc_toolkit.benchmarks.generalization_datasets import (
        generate_generalization_test_suite, build_generalization_suite_manifest,
    )
    from tcspc_toolkit.benchmarks.generalization import (
        build_generalization_development_measurements, prepare_generalization_data,
        fit_generalization_ml_estimators, evaluate_generalization_suite_benchmark,
        evaluate_classical_generalization_suite_benchmark, build_generalization_robustness_report,
        summarize_test_f_model_mismatch, evaluate_classical_instrument_acquisition_benchmark,
    )

    definition = default_generalization_suite()
    require(definition.suite_version == "week8-day55-v2", "Frozen protocol version")
    require(tuple(definition.seed_for(t) for t in TEST_IDS) == tuple(range(50001, 50007)), "External seeds")
    suite = generate_generalization_test_suite(definition)
    repeated = generate_generalization_test_suite(definition)
    require(tuple(t.test_id for t in suite.tests) == TEST_IDS, "A-F ordering")
    tests = {t.test_id: t for t in suite.tests}
    for test, repeat in zip(suite.tests, repeated.tests, strict=True):
        same_measurements(test, repeat)
        require(test.X_histograms.shape == (192, 400), "A-F geometry")
        np.testing.assert_array_equal(test.metadata["pair_id"], np.arange(192))
        np.testing.assert_array_equal(test.y, tests["A"].y)
        np.testing.assert_array_equal(test.time, np.arange(0., 20., .05))
    development = build_generalization_development_measurements(definition=definition, random_seed=42)
    same_measurements(development, build_generalization_development_measurements(definition=definition, random_seed=42))
    require(development.X_histograms.shape == (64, 400), "Development geometry")
    f = tests["F"]
    require(set(f.metadata["secondary_fraction"]) == {.05, .15}, "F fractions")
    np.testing.assert_array_equal(f.y, f.metadata["primary_lifetime_ns"])
    np.testing.assert_array_equal(f.metadata["secondary_lifetime_ns"], 2 * f.y)
    report.update(protocol=definition.suite_version, development_seed=42,
                  manifest=records(build_generalization_suite_manifest(suite)), dataset_status="PASS",
                  development_sha256=fingerprint(development), deterministic_data="exact repeat",
                  external_sha256={t: fingerprint(test) for t, test in tests.items()},
                  method_ids=list(METHODS), representations=list(REPRESENTATIONS))
    prepared = prepare_generalization_data(development_measurements=development, tests=tests,
        feature_config=FeatureConfig(tail_start_ns=2., early_stop_ns=2., late_start_ns=3.), n_pca_components=10)
    require(prepared.pca.n_samples_ == 64, "PCA must fit development only")
    np.testing.assert_allclose(prepared.pca.mean_, prepared.X_normalized_development.mean(axis=0), rtol=0, atol=0)
    fitted = fit_generalization_ml_estimators(prepared)
    require(tuple(fitted) == METHODS, "Week-8 method ordering")
    require(all(tuple(v) == REPRESENTATIONS for v in fitted.values()), "Representation ordering")
    print("Week 8: data/preparation PASS; evaluating all 1,152 observations with 9 ML fits", flush=True)
    nonclassical = evaluate_generalization_suite_benchmark(prepared=prepared, fitted_estimators=fitted)
    repeat_result = evaluate_generalization_suite_benchmark(
        prepared=prepared, fitted_estimators=fit_generalization_ml_estimators(prepared))
    pd.testing.assert_frame_equal(nonclassical.predictions, repeat_result.predictions, check_exact=True)
    require(len(nonclassical.summary) == 66, "11 nonclassical methods/representations per test")
    report["ml_metrics"] = records(nonclassical.summary)
    report["deterministic_ml"] = "exact repeat of refitted predictions"
    report["repeatability"] = {
        "ml_predictions": frame_fingerprint(nonclassical.predictions),
        "development_metadata": frame_fingerprint(development.metadata),
        **{test_id + "_metadata": frame_fingerprint(test.metadata) for test_id, test in tests.items()},
    }
    expected_mae = reference_rows(8, "e802c319fc9ab6ad")
    expected_degradation = reference_rows(8, "93c3d55a39cf7b40")
    require([r[0] for r in expected_mae] == ["constant_mean", "mean_arrival_time", *METHODS, "classical_reconvolution"], "Scorecard method order")
    for row, ratios in zip(expected_mae[:-1], expected_degradation[:-1], strict=True):
        method = row[0]
        representation = "none" if method == "constant_mean" else "engineered_features"  # noqa: F841 - used by pandas query below
        selection = nonclassical.summary.query("estimator == @method and representation == @representation").set_index("test_id")
        for test_id, value in zip(TEST_IDS, row[1:], strict=True):
            compare(report, f"{method}/{test_id}/mae_ns", selection.loc[test_id, "mae_ns"], float(value), 4)
        for test_id, value in zip(TEST_IDS[1:], ratios[1:], strict=True):
            compare(report, f"{method}/{test_id}/degradation", selection.loc[test_id, "mae_ns"] / selection.loc["A", "mae_ns"], float(value), 3)
    stop_on_discrepancy(report, "Week-8 ML/reporting")
    report["ml_invariance"] = dict(status="PASS", comparisons=len(report["comparisons"]))
    print("Week 8: ML reference PASS; full 1,152-curve classical traversal", flush=True)
    classical = evaluate_classical_generalization_suite_benchmark(
        tests=tests, irf_centre_ns=1., temporal_shift_bounds=(-.5, .5), objective="poisson")
    synthesis = build_generalization_robustness_report(nonclassical_result=nonclassical, classical_result=classical)
    report["scorecard"] = records(synthesis.principal_summary)
    report["classical_failure_diagnostics"] = records(classical.fit_diagnostics.groupby(
        ["test_id", "failure_reason"], dropna=False, sort=False).size().reset_index(name="count"))
    report["F_severity"] = records(summarize_test_f_model_mismatch(pd.concat(
        [nonclassical.predictions, classical.predictions], ignore_index=True)))
    for test_id, value in zip(TEST_IDS, expected_mae[-1][1:], strict=True):
        compare(report, f"classical/{test_id}/mae_ns", synthesis.mae_matrix.loc["classical_reconvolution", test_id], float(value), 4)
    for test_id, value in zip(TEST_IDS[1:], expected_degradation[-1][1:], strict=True):
        compare(report, f"classical/{test_id}/degradation", synthesis.degradation_matrix.loc["classical_reconvolution", f"{test_id}_over_A"], float(value), 3)
    for test_id, value in zip(TEST_IDS, reference_rows(8, "f5e40edbc2b8ea2b")[0][1:], strict=True):
        compare(report, f"classical/{test_id}/failure_rate", synthesis.classical_failure_rate_matrix.loc["classical_reconvolution", test_id], float(value), 4)
    report["historical_notebook_comparison"] = report.pop("comparisons")
    report["repeatability"]["classical"] = frame_fingerprint(classical.fit_diagnostics.drop(columns="runtime_ms"))
    scorecard = compact_scorecard(pd.concat([nonclassical.summary, classical.summary], ignore_index=True))
    require(len(scorecard) == 72, "Complete scorecard must contain 12 methods/representations per test")
    populations = {t: population(classical.fit_diagnostics.query("test_id == @t")) for t in TEST_IDS}
    report["science"] = {
        "protocol_id": definition.suite_version, "test_order": list(TEST_IDS),
        "development_seed": 42, "n_development": 64, "n_bins": 400,
        "external_seeds": {t: definition.seed_for(t) for t in TEST_IDS},
        "external_counts": {t: len(test.y) for t, test in tests.items()},
        "development_sha256": fingerprint(development), "external_sha256": report["external_sha256"],
        "manifest": report["manifest"], "methods": list(METHODS), "representations": list(REPRESENTATIONS),
        "scorecard": scorecard, "classical_populations": populations,
    }
    # Public orchestration reuses the same Test-C observations for both IRF assumptions.
    before = fingerprint(tests["C"])
    instrument = evaluate_classical_instrument_acquisition_benchmark(tests=tests, irf_centre_ns=1., nominal_irf_fwhm_ns=.4)
    require(before == fingerprint(tests["C"]), "Test-C observations mutated")
    report["C_IRF_comparison"] = records(instrument.test_c_irf_comparison)
    diagnostic = instrument.fit_diagnostics.query("test_id == 'C'")
    correct = diagnostic.query("classical_irf_mode == 'correct_test_irf'").reset_index(drop=True)
    wrong = diagnostic.query("classical_irf_mode == 'nominal_familiar_irf'").reset_index(drop=True)
    for frame in (correct, wrong):
        np.testing.assert_array_equal(frame.sample_id, tests["C"].metadata.sample_id)
        np.testing.assert_array_equal(frame.pair_id, tests["C"].metadata.pair_id)
    np.testing.assert_array_equal(correct.irf_fwhm_ns, wrong.irf_fwhm_ns)
    np.testing.assert_array_equal(correct.assumed_irf_fwhm_ns, tests["C"].metadata.irf_fwhm_ns)
    np.testing.assert_array_equal(wrong.assumed_irf_fwhm_ns, np.full(192, .4))
    parameters = ["fitted_lifetime_ns", "fitted_amplitude", "fitted_background", "fitted_temporal_shift_ns", "poisson_nll"]
    np.testing.assert_array_equal(correct[parameters], classical.fit_diagnostics.query("test_id == 'C'")[parameters])
    report["repeatability"]["C_correct_and_misspecified"] = frame_fingerprint(diagnostic.drop(columns="runtime_ms"))
    comparison = report["C_IRF_comparison"][0]
    report["science"]["test_c_irf"] = {
        "same_observation_sha256": before, "generating_fwhm_ns": sorted(correct.irf_fwhm_ns.unique().tolist()),
        "misspecified_fwhm_ns": .4,
        "correct": population(correct) | dict(mae_ns=comparison["mae_correct_irf_ns"], bias_ns=comparison["bias_correct_irf_ns"]),
        "misspecified": population(wrong) | dict(mae_ns=comparison["mae_nominal_irf_ns"], bias_ns=comparison["bias_nominal_irf_ns"]),
        "mae_ratio": comparison["nominal_to_correct_mae_ratio"],
        "paired": paired_error_change(correct, wrong),
    }
    report["test_c_pairing"] = "PASS"
    severity_counts = f.metadata.model_mismatch_severity.value_counts().to_dict()
    require(severity_counts == {"weak": 96, "moderate": 96}, "Test-F severity populations")
    np.testing.assert_array_equal(f.metadata.signal_photon_weighted_lifetime_ns,
                                  (1 - f.metadata.secondary_fraction) * f.y
                                  + f.metadata.secondary_fraction * f.metadata.secondary_lifetime_ns)
    severity_rows = []
    for row in report["F_severity"]:
        severity_rows.append({k: row[k] for k in (
            "estimator", "representation", "model_mismatch_severity", "secondary_fraction",
            "mae_to_primary_ns", "bias_to_primary_ns", "mae_to_signal_photon_weighted_ns", "bias_to_signal_photon_weighted_ns")}
            | dict(n_total=row["n_total_samples"], n_valid=row["n_valid_predictions"],
                   n_failed=row["n_total_samples"] - row["n_valid_predictions"]))
    require(len(severity_rows) == 24, "Two F severities for every method/representation")
    report["science"]["test_f"] = {
        "reference": "primary_component", "descriptive_reference": "signal_photon_weighted",
        "interpretation": "Neither reference is a unique mono-exponential physical truth; no pseudo-true projection.",
        "severity_counts": severity_counts, "metrics": severity_rows,
        "classical_populations": {
            severity: population(classical.fit_diagnostics.query("test_id == 'F' and model_mismatch_severity == @severity"))
            for severity in ("weak", "moderate")},
    }
    report["test_f_reference_semantics"] = "PASS"


def read_week9_reference():
    reference = json.loads(WEEK9_REFERENCE_PATH.read_text(encoding="utf-8"))
    require(reference["reference_id"] == WEEK9_REFERENCE_ID and reference["reference_version"] == 1,
            "Unsupported Week-9 ML reference")
    require(set(reference["weeks"]) == {"week9_ml"}, "Unexpected Week-9 reference scope")
    return reference


def uncertainty_population(source, sample_ids):
    """Count attempted source rows, independently of point or reference validity."""
    import numpy as np
    from tcspc_toolkit.uncertainty.evaluation import PredictionIntervalResult

    interval = isinstance(source, PredictionIntervalResult)
    mask = source.valid_interval_mask if interval else source.valid_score_mask
    require(len(mask) == len(sample_ids), "Uncertainty sample identity alignment")
    return dict(n_attempted=len(mask), n_valid=int(mask.sum()), n_failed=int((~mask).sum()),
                failure_rate=float((~mask).mean()), failures=[
                    dict(sample_id=sample_ids[i], reason="invalid_interval" if interval else "invalid_score")
                    for i in np.flatnonzero(~mask)],
                metric_population="valid source rows; failures divided by all attempted rows")


@contextmanager
def audit_week9_training(development, split):
    """Observe every real pipeline fit, including the exact 200 bootstrap draws.

    This delegates unchanged to sklearn; no scientific algorithm is replaced.
    A changed training population/order fails before fitting the offending data.
    """
    import numpy as np
    from sklearn.pipeline import Pipeline

    X = development.X_features.to_numpy(dtype=np.float64)[list(split.training_indices)]
    y = development.y[list(split.training_indices)]
    rng = np.random.default_rng(42)
    draws = [rng.integers(0, 48, size=48) for _ in range(200)]
    calls = []
    original = Pipeline.fit

    def checked(self, actual_X, actual_y, *args, **kwargs):
        index = len(calls)
        require(index < 205, "Unexpected uncertainty training/refitting call")
        indices = np.arange(48) if index < 5 else draws[index - 5]
        np.testing.assert_array_equal(actual_X, X[indices], strict=True)
        np.testing.assert_array_equal(actual_y, y[indices], strict=True)
        calls.append(type(self.named_steps["model"]).__name__)
        return original(self, actual_X, actual_y, *args, **kwargs)

    with patch.object(Pipeline, "fit", checked):
        yield calls
    require(calls == ["HistGradientBoostingRegressor"] * 3 + ["RandomForestRegressor"] + ["Ridge"] * 201,
            "Unexpected model fitting sequence/budget")


@contextmanager
def frozen_week9_models():
    """Fail loudly if calibration/external evaluation tries to refit anything."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from tcspc_toolkit.uncertainty.ml import (
        QuantileGradientBoostingIntervalEstimator, RandomForestTreeSpreadEstimator,
        BootstrapPredictionSpreadEstimator,
    )
    with ExitStack() as stack:
        for cls in (Pipeline, PCA, QuantileGradientBoostingIntervalEstimator,
                    RandomForestTreeSpreadEstimator, BootstrapPredictionSpreadEstimator):
            stack.enter_context(patch.object(cls, "fit", side_effect=AcceptanceFailure(
                "Frozen Week-9 evaluation attempted model/representation fitting")))
        yield


def week9_ml(report):
    """Day-58/59/62 ML plus A/F point-fit residuals; no classical uncertainty run."""
    import numpy as np
    import pandas as pd
    from tcspc_toolkit.config import FeatureConfig
    from tcspc_toolkit.benchmarks.generalization_protocol import default_generalization_suite
    from tcspc_toolkit.benchmarks.generalization_datasets import generate_generalization_test_suite
    from tcspc_toolkit.benchmarks.generalization import (
        build_generalization_development_measurements, prepare_generalization_data,
    )
    from tcspc_toolkit.uncertainty.evaluation import (
        make_uncertainty_development_split, PredictionIntervalResult,
        evaluate_prediction_intervals, evaluate_uncertainty_scores,
        evaluate_quantile_prediction_intervals, UncertaintyDataProvenance, UncertaintyDataRole,
    )
    from tcspc_toolkit.uncertainty.ml import (
        fit_and_evaluate_quantile_gradient_boosting, fit_and_evaluate_random_forest_tree_spread,
        fit_and_evaluate_ridge_bootstrap_spread, evaluate_frozen_quantile_gradient_boosting,
        evaluate_frozen_ml_uncertainty_score, evaluate_paired_quantile_interval_response,
        evaluate_paired_uncertainty_score_response, project_ml_prediction_intervals,
        project_ml_uncertainty_scores,
    )
    from tcspc_toolkit.benchmarks.uncertainty import (
        conformalize_quantile_gradient_boosting, evaluate_frozen_conformalized_quantile,
        build_ml_uncertainty_robustness_report, evaluate_residual_mismatch_diagnostics,
    )
    from tcspc_toolkit.evaluation.core import build_point_evaluation, summarize_point_errors
    from tcspc_toolkit.evaluation.results import EvaluationBatch, MethodDescriptor, LifetimeReference
    from tcspc_toolkit.evaluation.attachments import IntervalAttachment, ScoreAttachment

    definition = default_generalization_suite()
    require(definition.suite_version == "week8-day55-v2", "Week-9 frozen external protocol")
    require([definition.seed_for(t) for t in TEST_IDS] == list(range(50001, 50007)), "External seeds")
    features = FeatureConfig(tail_start_ns=2., early_stop_ns=2., late_start_ns=3.)

    def prepare():
        development = build_generalization_development_measurements(definition=definition, random_seed=42)
        suite = generate_generalization_test_suite(definition)
        tests = {test.test_id: test for test in suite.tests}
        prepared = prepare_generalization_data(
            development_measurements=development, tests=tests, feature_config=features)
        return development, prepared

    development, prepared = prepare()
    repeated_development, repeated = prepare()
    same_measurements(development, repeated_development)
    require(development.X_histograms.shape == (64, 400), "Development geometry")
    require(tuple(prepared.tests) == TEST_IDS, "External ordering")
    pd.testing.assert_frame_equal(prepared.development.X_features, repeated.development.X_features, check_exact=True)
    for name in ("X_normalized_development", "X_pca_development"):
        np.testing.assert_array_equal(getattr(prepared, name), getattr(repeated, name), strict=True)
    for test_id, test in prepared.tests.items():
        same_measurements(test, repeated.tests[test_id])
        require(test.X_histograms.shape == (192, 400), "External geometry")
        pd.testing.assert_frame_equal(prepared.X_features[test_id], repeated.X_features[test_id], check_exact=True)
        for name in ("X_normalized", "X_pca"):
            np.testing.assert_array_equal(getattr(prepared, name)[test_id], getattr(repeated, name)[test_id], strict=True)
        UncertaintyDataProvenance(test_id, UncertaintyDataRole.EXTERNAL_EVALUATION)
    require(prepared.pca.n_samples_ == 64, "Shared preparation PCA is development-only (unused by Week-9 methods)")
    split = make_uncertainty_development_split(64)
    require(split == make_uncertainty_development_split(64), "Split repeatability")
    require((split.random_seed, split.calibration_fraction) == (57, .25), "Split policy")
    require(len(split.training_indices) == 48 and len(split.calibration_indices) == 16, "48/16 split")
    require(set(split.training_indices).isdisjoint(split.calibration_indices), "Calibration leakage")
    require(set(split.training_indices) | set(split.calibration_indices) == set(range(64)), "Split partition")
    ids = prepared.development.metadata.sample_id.tolist()
    calibration_ids = [ids[i] for i in split.calibration_indices]
    science = report["science"] = dict(
        protocol_id=definition.suite_version, development_seed=42, n_development=64,
        split_seed=57, n_training=48, n_calibration=16,
        training_ids=[ids[i] for i in split.training_indices], calibration_ids=calibration_ids,
        training_indices=list(split.training_indices), calibration_indices=list(split.calibration_indices),
        test_order=list(TEST_IDS), external_seeds={t: definition.seed_for(t) for t in TEST_IDS},
        external_counts={t: len(prepared.tests[t].y) for t in TEST_IDS},
        input_fingerprints={"development": fingerprint(development),
                            **{t: fingerprint(prepared.tests[t]) for t in TEST_IDS}},
        feature_config=asdict(features), representation="engineered_features",
        preparation_policy="Stateless features; shared 64-development PCA unused by uncertainty; learned Ridge scaling uses 48 training rows or their bootstrap resamples.",
    )
    week8_reference = read_current_reference()["weeks"]["week8"]
    require(science["input_fingerprints"] == {
        "development": week8_reference["development_sha256"],
        **week8_reference["external_sha256"]}, "Inputs differ from the accepted Week-8 reference")
    digests = report["repeatability"] = {
        "development_features": frame_fingerprint(prepared.development.X_features),
        "development_metadata": frame_fingerprint(development.metadata),
        "development_pca": frame_fingerprint(pd.DataFrame(prepared.X_pca_development)),
        **{t + "_features": frame_fingerprint(prepared.X_features[t]) for t in TEST_IDS},
    }
    print("Week 9 ML: exact data/split/preparation PASS; fitting only 48 training observations", flush=True)
    with audit_week9_training(prepared.development, split) as fit_calls:
        quantile = fit_and_evaluate_quantile_gradient_boosting(prepared.development, split=split)
        rf = fit_and_evaluate_random_forest_tree_spread(prepared.development, split=split)
        bootstrap = fit_and_evaluate_ridge_bootstrap_spread(prepared.development, split=split)
    report["training_fit_calls"] = len(fit_calls)
    models = quantile.fitted_estimator
    science["estimator_policy"] = dict(
        random_state=42, quantiles=[.05, .50, .95], nominal_level=.95 - .05,
        quantile_models=[model.named_steps["model"].get_params(deep=False)
                         for model in (models.lower_model, models.median_model, models.upper_model)],
        random_forest=rf.fitted_estimator.pipeline.named_steps["model"].get_params(deep=False),
        ridge=bootstrap.fitted_estimator.reference_estimator_.named_steps["model"].get_params(deep=False),
        bootstrap_replicates=bootstrap.fitted_estimator.n_bootstrap,
        bootstrap_policy="48 row draws with replacement per fit; default_rng(42); reference fit supplies point estimate",
        rf_score="tree prediction population std (ddof=0)",
        bootstrap_score="refitted-model prediction sample std (ddof=1)", tail_fraction=.20,
    )
    require(bootstrap.fitted_estimator.n_bootstrap == 200, "Bootstrap budget")
    science["calibration"] = []
    science["external"] = []
    science["test_f_severity"] = []
    point_ids = {
        "quantile_gradient_boosting": "quantile_gradient_boosting_median",
        "conformalized_quantile_gradient_boosting": "quantile_gradient_boosting_median",
        "random_forest_tree_spread": "random_forest_tree_mean",
        "ml_training_bootstrap": "training_bootstrap_reference",
    }

    def collect(source, y, sample_ids, evaluation_id):
        interval = isinstance(source, PredictionIntervalResult)
        kind = ("conformal_prediction" if source.method_id.startswith("conformalized_")
                else "quantile_prediction") if interval else "score"
        method = MethodDescriptor(point_ids[source.method_id], "ml", "engineered_features")
        reference_kind = "primary_component" if evaluation_id.startswith("F") else "generating_mono"
        reference = LifetimeReference(reference_kind, reference_kind, y, np.ones(len(y), dtype=bool))
        batch = EvaluationBatch(evaluation_id, sample_ids, references={reference_kind: reference})
        points = build_point_evaluation(batch=batch, method=method,
            lifetime_estimates_ns=source.prediction, is_valid=np.isfinite(source.prediction))
        if interval:
            attachment = project_ml_prediction_intervals(batch=batch, points=points,
                point_method=method, intervals=source, interval_kind=kind)
            require(isinstance(attachment, IntervalAttachment), "Intervals remain factual attachments")
            rows = attachment.intervals
            require(rows.interval_kind.eq(kind).all() and rows.nominal_level.eq(.95 - .05).all(),
                    "Interval identity/kind/nominal level")
            metrics = asdict(evaluate_prediction_intervals(y, source))
            if kind == "quantile_prediction":
                metrics.update(asdict(evaluate_quantile_prediction_intervals(y, source,
                    lower_quantile=.05, median_quantile=.50, upper_quantile=.95)))
        else:
            attachment = project_ml_uncertainty_scores(batch=batch, points=points,
                point_method=method, scores=source)
            require(isinstance(attachment, ScoreAttachment), "Spread stays a score attachment")
            rows = attachment.scores
            require("nominal_level" not in rows, "Scores must not claim coverage")
            metrics = asdict(evaluate_uncertainty_scores(y, source))
            metrics["mean_uncertainty_score"] = float(np.mean(source.uncertainty_score[source.valid_score_mask]))
        require(rows.sample_id.tolist() == sample_ids and rows.method_id.eq(method.method_id).all()
                and rows.uncertainty_method_id.eq(source.method_id).all(), "Attachment identity/order")
        digests[evaluation_id + "/" + source.method_id] = frame_fingerprint(rows)
        digests[evaluation_id + "/" + source.method_id + "/points"] = frame_fingerprint(points.points)
        comparisons = points.reference_comparisons
        point_metrics = summarize_point_errors(comparisons.error_ns,
            is_valid=points.points.is_valid, reference_available=comparisons.reference_available,
            eligible=comparisons.is_valid_comparison)
        return dict(method_id=source.method_id, point_method_id=method.method_id,
                    output_kind=kind, nominal_level=source.nominal_coverage if interval else None,
                    evaluation_id=evaluation_id, reference_kind=reference_kind,
                    population=uncertainty_population(source, sample_ids), metrics=metrics,
                    point_metrics=point_metrics)

    with frozen_week9_models():
        conformal = conformalize_quantile_gradient_boosting(quantile, prepared.development)
        science["conformal"] = dict(correction_ns=conformal.correction_ns,
            n_calibration_scores=conformal.n_calibration_scores,
            finite_sample_rank=min(conformal.n_calibration_scores,
                int(np.ceil((conformal.n_calibration_scores + 1) * (.95 - .05)))),
            rule="max(lower-y, y-upper, 0); rank=min(n, ceil((n+1)*nominal_level))",
            calibration_source="development calibration only", nominal_level=.95 - .05)
        digests["conformity_scores"] = frame_fingerprint(pd.DataFrame({"score": conformal.calibration_scores_ns}))
        cal_y = prepared.development.y[list(split.calibration_indices)]
        expected_scores = np.maximum.reduce([
            quantile.calibration_intervals.lower - cal_y,
            cal_y - quantile.calibration_intervals.upper, np.zeros(len(cal_y))])
        np.testing.assert_array_equal(conformal.calibration_scores_ns, expected_scores)
        require(conformal.n_calibration_scores == 16, "Conformal calibration population")
        for source in (quantile.calibration_intervals, conformal.calibration_intervals,
                       rf.calibration_scores, bootstrap.calibration_scores):
            science["calibration"].append(collect(source, cal_y, calibration_ids, "uncertainty_calibration"))
        for t, test in prepared.tests.items():
            X = prepared.X_features[t]
            raw = evaluate_frozen_quantile_gradient_boosting(quantile, X, test.y, condition_id=t)
            corrected = evaluate_frozen_conformalized_quantile(conformal, X, test.y, test_id=t)
            sources = [raw.intervals, corrected.intervals]
            sources.extend(evaluate_frozen_ml_uncertainty_score(c, X, test.y, condition_id=t).scores
                           for c in (rf, bootstrap))
            for source in sources:
                science["external"].append(collect(source, test.y, test.metadata.sample_id.tolist(), t))
                if t == "F":
                    for severity in ("weak", "moderate"):
                        mask = test.metadata.model_mismatch_severity.to_numpy() == severity
                        fields = ("prediction", "lower", "upper") if isinstance(source, PredictionIntervalResult) else (
                            "prediction", "uncertainty_score")
                        subset = replace(source, **{name: getattr(source, name)[mask] for name in fields})
                        science["test_f_severity"].append(collect(subset, test.y[mask],
                            test.metadata.sample_id[mask].tolist(), "F_" + severity))
        score_calibrations = {"Random-Forest tree spread": rf, "Ridge training bootstrap": bootstrap}
        robustness = build_ml_uncertainty_robustness_report(prepared=prepared,
            quantile_calibration=quantile, conformal_calibration=conformal, score_calibrations=score_calibrations)
        science["interval_scorecard"] = records(robustness.interval_scorecard)
        science["score_scorecard"] = records(robustness.score_only_scorecard)
        paired = {"quantile_gradient_boosting": evaluate_paired_quantile_interval_response(
            quantile, prepared, definition=definition)}
        paired.update({c.calibration_scores.method_id: evaluate_paired_uncertainty_score_response(
            c, prepared, definition=definition) for c in (rf, bootstrap)})
        science["paired_response"] = {method: records(result.summary) for method, result in paired.items()}
        for method, result in paired.items():
            require(list(result.condition_results) == ["B_low_photon", "B_high_photon", "D_high_background",
                    "F_weak_mismatch", "F_moderate_mismatch"], "Frozen conditional traversal")
            require(result.summary.n_pairs.tolist() == [128, 64, 192, 96, 96], "Conditional counts")
            for condition, item in result.condition_results.items():
                digests[method + "/paired/" + condition] = frame_fingerprint(item.diagnostics)
    report["no_external_refitting"] = "PASS"
    f = prepared.tests["F"]
    np.testing.assert_array_equal(f.y, f.metadata.primary_lifetime_ns)
    np.testing.assert_array_equal(f.metadata.signal_photon_weighted_lifetime_ns,
        (1 - f.metadata.secondary_fraction) * f.y
        + f.metadata.secondary_fraction * f.metadata.secondary_lifetime_ns)
    science["test_f"] = dict(severity_counts={s: int(sum(f.metadata.model_mismatch_severity == s))
        for s in ("weak", "moderate")}, reference="primary_component",
        descriptive_reference="signal_photon_weighted_lifetime_ns; not a unique mono-exponential truth",
        weighted_reference_fingerprint=frame_fingerprint(f.metadata[["sample_id", "signal_photon_weighted_lifetime_ns"]]))
    require(science["test_f"]["severity_counts"] == {"weak": 96, "moderate": 96}, "F severity")

    # Saved Notebook-14 tables are evidence, never executed. Any unexplained
    # difference stops candidate collection before a current reference is made.
    historical = {}
    cal_values = [quantile.interval_metrics.empirical_coverage,
                  quantile.interval_metrics.mean_interval_width,
                  rf.score_metrics.spearman_error_correlation,
                  bootstrap.score_metrics.spearman_error_correlation, conformal.correction_ns]
    for row, actual in zip(reference_rows(9, "910a6d01bbf11f44"), cal_values, strict=True):
        compare(historical, "calibration/" + row[-2], actual, float(row[-1]), 4)
    for index, frame in enumerate((robustness.interval_scorecard, robustness.score_only_scorecard)):
        saved = reference_rows(9, "339769e99c73f7e3", index)
        for row, actual in zip(saved, records(frame), strict=True):
            values = list(actual.values())
            require(row[1:4] == values[:3], "Historical report identity/order")
            for column, value, expected in zip(list(actual)[3:], values[3:], row[4:], strict=True):
                compare(historical, f"{values[0]}/{values[1]}/{column}", value, float(expected), 4)
    report["historical_notebook_comparison"] = historical.get("comparisons", [])
    stop_on_discrepancy(historical, "Week-9 ML/calibration historical comparison")
    print("Week 9 ML: calibration and A-F historical tables MATCH; A/F point-fit residuals only", flush=True)
    residual = evaluate_residual_mismatch_diagnostics(prepared=prepared, definition=definition)
    science["residual_summary"] = records(residual.summary)
    science["residual_populations"] = [dict(condition_id=row["condition_id"],
        n_attempted=row["n_curves"], n_valid=row["n_valid_fits"],
        n_failed=row["n_curves"] - row["n_valid_fits"])
        for row in science["residual_summary"]]
    digests["signed_residual_profiles"] = frame_fingerprint(residual.profile_table)
    digests["paired_signed_residual_profiles"] = frame_fingerprint(residual.paired_profile_table)
    for row, actual in zip(reference_rows(9, "7dde16d16f42f0e6"), records(residual.summary), strict=True):
        require(row[1] == actual["condition_id"], "Residual condition ordering")
        for column, value, expected in zip(list(actual)[1:], list(actual.values())[1:], row[2:], strict=True):
            if column in ("n_curves", "n_valid_fits"):
                require(value == int(expected), "Residual population changed")
            else:
                compare(historical, f"residual/{row[1]}/{column}", value, float(expected), 4)
    report["historical_notebook_comparison"] = historical.get("comparisons", [])
    stop_on_discrepancy(historical, "Week-9 signed-residual historical comparison")
    report["residual_negative_result"] = "Historical scalar and signed-profile summaries reproduced; no general mismatch-discrimination claim."


def scientific_json(value):
    """Lossless finite scalar conversion; explicit null for unusable diagnostics."""
    import math
    import numpy as np
    if isinstance(value, dict):
        return {str(k): scientific_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [scientific_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def record_digest(value):
    return hashlib.sha256(json.dumps(scientific_json(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_digest(*arrays):
    import numpy as np
    digest = hashlib.sha256()
    for value in arrays:
        value = np.asarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def optimizer_call_sites():
    """Observe the three reviewed call sites, never infer recovery from fit quality.

    AST line ranges survive formatting changes. A changed optimizer call structure
    deliberately requires review of this instrumentation, not a silent guess.
    """
    import ast
    from tcspc_toolkit.estimation import fitting
    labels = {"optimization_result": "primary", "continuation": "continuation",
              "candidate": "half_bin_fallback"}
    sites = {}
    for node in ast.walk(ast.parse(Path(fitting.__file__).read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "minimize"):
            require(len(node.targets) == 1 and isinstance(node.targets[0], ast.Name), "Optimizer assignment")
            label = labels[node.targets[0].id]
            sites.update({line: label for line in range(node.lineno, node.end_lineno + 1)})
    require(set(sites.values()) == set(labels.values()), "Review optimizer instrumentation")
    return sites


class ClassicalObservationAudit:
    """Read-only observers around canonical calls; no draws, starts or fits added.

    All RNG consumption and iteration remain owned by production orchestrators.
    Bootstrap refit records are streamed into a digest, not retained as large arrays.
    """
    def __init__(self, identities, progress_label):
        self.identities = iter(identities)
        self.rows = []
        self.in_bootstrap = False
        self.current = None
        self.calls = []
        self.sites = optimizer_call_sites()
        self.label = progress_label
        self.started = perf_counter()
        self.timings = dict(primary_seconds=0., covariance_seconds=0., bootstrap_seconds=0.)

    def minimize(self, original, *args, **kwargs):
        # The wrapper's caller is the production fitter; only read its line number.
        label = self.sites.get(sys._getframe(2).f_lineno)
        require(label is not None, "Unreviewed optimizer call site")
        self.calls.append(label)
        return original(*args, **kwargs)

    def fit(self, original, **kwargs):
        started = perf_counter()
        self.calls = []
        result = original(**kwargs)
        row = asdict(result)
        row.pop("runtime_ms")
        row = scientific_json(row)
        row["optimizer_calls"] = {label: self.calls.count(label)
            for label in ("primary", "continuation", "half_bin_fallback")}
        require(row["optimizer_calls"]["primary"] == 1, "Primary optimizer was not observed exactly once")
        require(bool(row["recovery_attempted"]) == bool(sum(self.calls.count(k)
            for k in ("continuation", "half_bin_fallback"))), "Recovery observer disagrees with fitter")
        inputs = array_digest(kwargs["time"], kwargs["counts"], kwargs["irf"])
        if self.in_bootstrap:
            self.refit_digest.update(record_digest(dict(inputs=inputs, result=row)).encode())
            for label in self.refit_work:
                self.refit_work[label] += int(row["optimizer_calls"][label] > 0)
            if not result.valid_fit:
                self.refit_failures.append(dict(refit_index=self.refit_index,
                    failure_reason=result.failure_reason))
            self.refit_index += 1
        else:
            identity = next(self.identities)
            self.current = dict(**identity, input_fingerprint=inputs, primary=row,
                                covariance=None, bootstrap=None)
            self.rows.append(self.current)
            self.timings["primary_seconds"] += perf_counter() - started
            elapsed = perf_counter() - self.started
            if self.label == "Day 62" and elapsed > 4 * 3600:
                require(elapsed * 1152 / len(self.rows) <= 6 * 3600,
                        "Day-62 runtime exceeds four hours and projects beyond six; stop for review")
            if len(self.rows) == 1 or len(self.rows) % 5 == 0:
                print(f"{self.label}: primary {len(self.rows)} {identity['condition_id']}/"
                      f"{identity['sample_id']} ({perf_counter()-self.started:.1f}s)", flush=True)
        return result

    def covariance(self, original, **kwargs):
        started = perf_counter()
        result = original(**kwargs)
        self.timings["covariance_seconds"] += perf_counter() - started
        self.current["covariance"] = scientific_json(asdict(result))
        return result

    def bootstrap(self, original, **kwargs):
        started = perf_counter()
        self.in_bootstrap = True
        self.refit_digest = hashlib.sha256()
        self.refit_index, self.refit_failures = 0, []
        self.refit_work = dict(continuation=0, half_bin_fallback=0)
        before = record_digest(kwargs["rng"].bit_generator.state)
        try:
            result = original(**kwargs)
        finally:
            self.in_bootstrap = False
        self.timings["bootstrap_seconds"] += perf_counter() - started
        require(self.refit_index == result.n_resamples, "Missing bootstrap refit observations")
        require(len(self.refit_failures) == result.n_failed_fits, "Bootstrap failure population mismatch")
        values = asdict(result)
        samples = values.pop("lifetime_samples_ns")
        values.update(refit_fingerprint=self.refit_digest.hexdigest(),
            sample_fingerprint=array_digest(samples), refit_failures=self.refit_failures,
            recovery_fits=self.refit_work, rng_before=before,
            rng_after=record_digest(kwargs["rng"].bit_generator.state))
        self.current["bootstrap"] = scientific_json(values)
        return result

    @contextmanager
    def observe(self):
        from tcspc_toolkit.estimation import fitting
        from tcspc_toolkit.uncertainty import classical
        from tcspc_toolkit.benchmarks import uncertainty
        original_minimize = fitting.minimize
        original_fit = classical.fit_single_reconvolution_curve
        original_covariance = classical.estimate_poisson_reconvolution_local_covariance
        original_bootstrap = classical.estimate_parametric_poisson_bootstrap
        with ExitStack() as stack:
            stack.enter_context(patch.object(fitting, "minimize",
                lambda *a, **k: self.minimize(original_minimize, *a, **k)))
            for owner in (classical, uncertainty):
                stack.enter_context(patch.object(owner, "fit_single_reconvolution_curve",
                    lambda **k: self.fit(original_fit, **k)))
                stack.enter_context(patch.object(owner, "estimate_poisson_reconvolution_local_covariance",
                    lambda **k: self.covariance(original_covariance, **k)))
                stack.enter_context(patch.object(owner, "estimate_parametric_poisson_bootstrap",
                    lambda **k: self.bootstrap(original_bootstrap, **k)))
            yield self


def wilson_interval(covered, n):
    """Descriptive 95% binomial interval, not a regression acceptance band."""
    from statistics import NormalDist
    if n == 0:
        return None
    z = NormalDist().inv_cdf(.975)
    p = covered / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * (p*(1-p)/n + z*z/(4*n*n))**.5 / (1 + z*z/n)
    return [centre-radius, centre+radius]


def classical_detail_records(rows):
    """Include derived Gaussian endpoints in temporary records, without mutation.

    The scientific fingerprint already covers the fitted lifetime and covariance
    diagnostics defining these endpoints. Day-62 projection endpoints are retained.
    """
    import copy
    from statistics import NormalDist
    output = copy.deepcopy(rows)
    quantile = NormalDist().inv_cdf(.95)
    for row in output:
        covariance = row["covariance"]
        if covariance is not None and "lower_ns" not in covariance:
            valid = covariance["covariance_valid"]
            lifetime = row["primary"]["fitted_lifetime_ns"]
            width = quantile * covariance["lifetime_std"] if valid else None
            covariance.update(lower_ns=lifetime-width if valid else None,
                              upper_ns=lifetime+width if valid else None)
    return output


def classical_population(rows):
    """Supplement canonical metrics with explicit availability/refit populations."""
    import numpy as np
    from statistics import NormalDist
    from tcspc_toolkit.uncertainty.evaluation import PredictionIntervalResult, evaluate_prediction_intervals
    primary_failures = [dict(sample_id=r["sample_id"], failure_reason=r["primary"]["failure_reason"])
                        for r in rows if not r["primary"]["valid_fit"]]
    prediction = np.array([r["primary"]["fitted_lifetime_ns"] for r in rows], dtype=float)
    truth = np.array([r["reference_ns"] for r in rows])
    strict = np.array([r["primary"]["valid_fit"] for r in rows])
    result = dict(n_total=len(rows), primary=dict(attempted=len(rows), valid=int(strict.sum()),
        failed=len(primary_failures), failures=primary_failures,
        strict_primary_fit_failure_rate=len(primary_failures)/len(rows),
        legacy_finite_prediction_failure_rate=float(np.mean(~np.isfinite(prediction)))),
        input_fingerprint=record_digest([r["input_fingerprint"] for r in rows]),
        result_fingerprint=record_digest(rows))
    finite = np.isfinite(prediction)
    result["point_error"] = dict(metric_population="finite predictions, including finite-invalid (legacy Day-62)",
        mae_ns=float(np.mean(np.abs(prediction[finite]-truth[finite]))),
        bias_ns=float(np.mean(prediction[finite]-truth[finite])))
    for method in ("covariance", "bootstrap"):
        lower, upper, scales, failures = [], [], [], []
        for row in rows:
            source = row[method]
            valid = source is not None and source[method + "_valid"]
            if not valid:
                failures.append(dict(sample_id=row["sample_id"], failure_reason=(
                    source["failure_reason"] if source else "primary_invalid:" + row["primary"]["failure_reason"])))
            scale = source["lifetime_std" if method == "covariance" else "bootstrap_std_ns"] if valid else np.nan
            if method == "covariance":
                displacement = NormalDist().inv_cdf(.95) * scale
                estimate = row["primary"]["fitted_lifetime_ns"]
                lo, hi = (estimate-displacement, estimate+displacement) if valid else (np.nan, np.nan)
                if source is not None and "lower_ns" in source:
                    require((not valid) or (lo == source["lower_ns"] and hi == source["upper_ns"]),
                            "Observed covariance bounds disagree with canonical projection")
            else:
                lo, hi = (source["lower_ns"], source["upper_ns"]) if valid else (np.nan, np.nan)
            lower.append(lo)
            upper.append(hi)
            scales.append(scale)
        source = PredictionIntervalResult(prediction=np.where(strict, prediction, np.nan),
            lower=np.array(lower), upper=np.array(upper), nominal_coverage=.90,
            method_id="covariance" if method == "covariance" else "parametric_bootstrap")
        metrics = evaluate_prediction_intervals(truth, source)
        valid = np.isfinite(lower) & np.isfinite(upper) & strict
        covered = int(np.sum(valid & (truth >= lower) & (truth <= upper)))
        count = int(valid.sum())
        require(count == len(rows)-len(failures), "Interval observer validity mismatch")
        summary = dict(attempted=len(rows), evaluated=sum(r[method] is not None for r in rows),
            valid=count, failed=len(failures), failure_rate=len(failures)/len(rows), failures=failures,
            coverage=metrics.empirical_coverage, covered=covered, coverage_denominator=count,
            coverage_wilson_95=wilson_interval(covered, count),
            mean_width_ns=metrics.mean_interval_width, median_width_ns=metrics.median_interval_width,
            mean_std_ns=float(np.nanmean(scales)) if count else None)
        if method == "covariance":
            summary["boundary_failures"] = sum(r[method] is not None and r[method]["boundary_hit"] for r in rows)
            summary["conditioning_failures"] = [f for f in failures
                if "information" in (f["failure_reason"] or "")]
        else:
            sources = [r[method] for r in rows if r[method] is not None]
            summary.update(requested_refits=sum(s["n_resamples"] for s in sources),
                valid_refits=sum(s["n_successful_fits"] for s in sources),
                failed_refits=sum(s["n_failed_fits"] for s in sources),
                boundary_hits=sum(s["n_boundary_hits"] for s in sources),
                minimum_valid_refits=min((s["n_successful_fits"] for s in sources), default=0),
                maximum_refit_failure_rate=max((s["fit_failure_rate"] for s in sources), default=None),
                high_refit_failure_observations=[dict(sample_id=r["sample_id"],
                    requested=r[method]["n_resamples"], valid=r[method]["n_successful_fits"],
                    failed=r[method]["n_failed_fits"]) for r in rows
                    if r[method] is not None and r[method]["fit_failure_rate"] >= .10])
            summary["aggregate_refit_failure_rate"] = (summary["failed_refits"]/summary["requested_refits"]
                if summary["requested_refits"] else None)
        result[method] = summary
    result["recovery"] = {
        "primary": {k: sum(r["primary"]["optimizer_calls"][k] > 0 for r in rows)
                    for k in ("continuation", "half_bin_fallback")},
        "bootstrap": {k: sum(r["bootstrap"]["recovery_fits"][k] for r in rows if r["bootstrap"] is not None)
                      for k in ("continuation", "half_bin_fallback")}}
    return scientific_json(result)


def classical_protocol(day):
    common = dict(profile_id=CLASSICAL_PROFILE, nominal_coverage=.90,
        temporal_shift_bounds=[-.5, .5], background_fraction=.10,
        methods={"covariance": "prediction_interval", "parametric_bootstrap": "prediction_interval",
                 "repeated_poisson_realizations": "empirical_reference"})
    if day == 61:
        return common | dict(n_repeated=50, n_bootstrap=100,
            rng_policy="one sequential RNG per condition: primary, bootstrap, next primary",
            conditions=[dict(condition_id=f"day61-{i}", label=label, true_lifetime_ns=tau,
                signal_photon_count=photons, background_per_bin=background, seed=61000+i,
                irf_centre_ns=1., irf_fwhm_ns=.25, irf_shift_ns=.05)
                for i, (label, tau, photons, background) in enumerate(DAY61_CONDITIONS, 1)])
    require(day == 62, "Unknown classical day")
    return common | dict(protocol_id="week8-day55-v2", test_order=list(TEST_IDS),
        n_per_test=192, n_bootstrap=200, bootstrap_seed=62001,
        external_seeds=dict(zip(TEST_IDS, range(50001, 50007))),
        rng_policy="one sequential bootstrap RNG across A-F; invalid primaries consume no bootstrap draws",
        test_f_reference="primary_component", weighted_reference="descriptive only, not pseudo_true_mono")


def week9_classical_day61(report):
    import numpy as np
    from tcspc_toolkit.uncertainty.classical import evaluate_repeated_poisson_uncertainty
    protocol = classical_protocol(61)
    science = report["science"] = dict(protocol=protocol, conditions=[])
    report["repeatability"] = {}
    report["timing"] = {}
    details = []
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        rng = np.random.default_rng(condition["seed"])
        audit = ClassicalObservationAudit([dict(condition_id=cid, sample_id=i,
            reference_ns=condition["true_lifetime_ns"]) for i in range(50)], "Day 61")
        with audit.observe():
            result = evaluate_repeated_poisson_uncertainty(time=np.arange(0., 20., .05),
                **{k: condition[k] for k in ("true_lifetime_ns", "signal_photon_count", "background_per_bin",
                    "irf_centre_ns", "irf_fwhm_ns", "irf_shift_ns")},
                temporal_shift_bounds=(-.5, .5), n_repeats=50, n_bootstrap_resamples=100,
                rng=rng, nominal_coverage=.90)
        require(len(audit.rows) == 50, "Day-61 realization count")
        summary = classical_population(audit.rows)
        summary.update(condition_id=cid, empirical=dict(bias_ns=result.empirical_bias_ns,
            std_ns=result.empirical_std_ns, rmse_ns=result.empirical_rmse_ns),
            final_rng_fingerprint=record_digest(rng.bit_generator.state))
        for method in ("covariance", "bootstrap"):
            summary[method]["std_ratio"] = getattr(result, method + "_to_empirical_std_ratio")
            require(summary[method]["coverage"] == getattr(result, method + "_metrics").empirical_coverage,
                    "Observer/canonical coverage disagree")
            require(summary[method]["mean_std_ns"] == getattr(result, "mean_" + method + "_std_ns"),
                    "Observer/canonical uncertainty scale disagree")
        science["conditions"].append(scientific_json(summary))
        report["repeatability"][cid] = summary["result_fingerprint"]
        report["timing"][cid] = audit.timings
        details.extend(audit.rows)
        print(f"Day 61 {cid}: complete, primary {summary['primary']['valid']}/50, "
              f"bootstrap {summary['bootstrap']['valid_refits']}/{summary['bootstrap']['requested_refits']}", flush=True)
    report["details"] = details  # main writes these separately, never into the reference
    report["historical_notebook_comparison"] = []


def week9_classical_day62(report):
    import numpy as np
    from tcspc_toolkit.config import FeatureConfig
    from tcspc_toolkit.benchmarks.generalization_protocol import default_generalization_suite
    from tcspc_toolkit.benchmarks.generalization_datasets import generate_generalization_test_suite
    from tcspc_toolkit.benchmarks.generalization import (
        build_generalization_development_measurements, prepare_generalization_data)
    from tcspc_toolkit.benchmarks.uncertainty import evaluate_classical_uncertainty_robustness
    protocol = classical_protocol(62)
    definition = default_generalization_suite()
    require(definition.suite_version == protocol["protocol_id"], "Frozen external protocol")
    tests = {t.test_id: t for t in generate_generalization_test_suite(definition).tests}
    require(list(tests) == list(TEST_IDS), "External order")
    identities = []
    for tid, test in tests.items():
        require(len(test.y) == 192 and definition.seed_for(tid) == protocol["external_seeds"][tid], "External identity")
        for i, target in enumerate(test.y):
            identities.append(dict(condition_id=tid, sample_id=int(test.metadata.iloc[i].sample_id), reference_ns=float(target)))
    np.testing.assert_array_equal(tests["F"].y, tests["F"].metadata.primary_lifetime_ns)
    previous = read_current_reference()["weeks"]["week8"]
    for tid, test in tests.items():
        require(fingerprint(test) == previous["external_sha256"][tid], "Accepted Week-8 inputs differ")
    prepared = prepare_generalization_data(
        development_measurements=build_generalization_development_measurements(definition=definition, random_seed=42),
        tests=tests, feature_config=FeatureConfig(tail_start_ns=2., early_stop_ns=2., late_start_ns=3.), n_pca_components=10)
    audit = ClassicalObservationAudit(identities, "Day 62")
    report["details"] = audit.rows
    with audit.observe():
        result = evaluate_classical_uncertainty_robustness(prepared=prepared, definition=definition,
            n_bootstrap_resamples=200, nominal_coverage=.90, bootstrap_random_seed=62001)
    require(len(audit.rows) == 1152, "Full external traversal incomplete")
    for row, canonical in zip(audit.rows, result.per_curve.itertuples(index=False), strict=True):
        require(row["condition_id"] == canonical.test_id and row["sample_id"] == canonical.sample_id,
                "Observed traversal and canonical table differ")
        if row["covariance"] is not None:
            row["covariance"].update(scientific_json(dict(lower_ns=canonical.covariance_lower_ns,
                upper_ns=canonical.covariance_upper_ns)))
    bootstrap_rows = [r["bootstrap"] for r in audit.rows if r["bootstrap"] is not None]
    require(all(a["rng_after"] == b["rng_before"] for a, b in zip(bootstrap_rows, bootstrap_rows[1:])),
            "Global A-F sequential bootstrap RNG was interrupted")
    science = report["science"] = dict(protocol=protocol, conditions=[],
        external_sha256={tid: fingerprint(t) for tid, t in tests.items()},
        legacy_scorecard=records(result.scorecard), legacy_conditional_scorecard=records(result.conditional_scorecard))
    for tid in TEST_IDS:
        rows = [r for r in audit.rows if r["condition_id"] == tid]
        summary = classical_population(rows)
        summary["condition_id"] = tid
        science["conditions"].append(summary)
        canonical = result.per_curve.loc[result.per_curve.test_id == tid]
        require(summary["primary"]["valid"] == int(canonical.classical_fit_valid.sum()), "Strict primary population")
        for method in ("covariance", "bootstrap"):
            require(summary[method]["valid"] == int(canonical[method+"_valid"].sum()), "Interval population")
    science["test_f_severity"] = []
    for severity in ("weak", "moderate"):
        ids = set(tests["F"].metadata.loc[tests["F"].metadata.model_mismatch_severity == severity, "sample_id"])
        require(len(ids) == 96, "Test-F severity count")
        rows = [r for r in audit.rows if r["condition_id"] == "F" and r["sample_id"] in ids]
        science["test_f_severity"].append(dict(severity=severity, **classical_population(rows)))
    report["repeatability"] = dict(records=record_digest(audit.rows), per_curve=frame_fingerprint(result.per_curve))
    report["timing"] = audit.timings
    report["details"] = audit.rows
    report["historical_notebook_comparison"] = []


def read_classical_reference():
    reference = json.loads(CLASSICAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    require(reference["reference_id"] == CLASSICAL_REFERENCE_ID and reference["reference_version"] == 1,
            "Unsupported classical reference")
    require(reference["profile_id"] == CLASSICAL_PROFILE, "Unknown classical profile")
    require(all(day in reference for day in ("day61", "day62")), "Incomplete classical reference")
    for day in (61, 62):
        require(reference[f"day{day}"]["protocol"] == classical_protocol(day),
                f"Classical Day-{day} reference protocol differs")
    require(reference["provenance"]["independent_reruns"]["status"] == "PASS",
            "Classical reference requires independent exact agreement")
    return reference


def classical_historical_comparison(day, science):
    """Keep historical sources separate; differences never rewrite either oracle."""
    comparisons = {}
    findings = (ROOT / "docs/scientific_findings.md").read_text(encoding="utf-8")
    if day == 61:
        table = findings.split("| Condition                         | Empirical std (ns)", 1)[1]
        doc_rows = [line.split("|")[1:-1] for line in table.splitlines()[2:10]]
        saved = reference_rows(9, "517e9d4b7bfe797e")
        require(len(saved) == len(doc_rows) == len(science["conditions"]) == 8, "Historical Day-61 rows")
        columns = ("empirical_std", "covariance_std", "covariance_ratio", "bootstrap_std", "bootstrap_ratio",
                   "covariance_coverage", "bootstrap_coverage")
        for current, notebook, documentation in zip(science["conditions"], saved, doc_rows, strict=True):
            values = [current["empirical"]["std_ns"], current["covariance"]["mean_std_ns"],
                current["covariance"]["std_ratio"], current["bootstrap"]["mean_std_ns"],
                current["bootstrap"]["std_ratio"], current["covariance"]["coverage"], current["bootstrap"]["coverage"]]
            for source, expected, precision in (("Notebook14", notebook[2:], [4]*7),
                    ("scientific_findings", documentation[1:], [4, 4, 3, 4, 3, 2, 2])):
                for name, actual, old, decimals in zip(columns, values, expected, precision, strict=True):
                    compare(comparisons, f"{source}/{current['condition_id']}/{name}", actual, float(old), decimals)
    else:
        for cell, key, identity in (("a9df22789358b5cc", "legacy_scorecard", "test_id"),
                                   ("fef57a13bfb661b8", "legacy_conditional_scorecard", "condition_id")):
            columns = ["mae_ns", "empirical_coverage", "mean_width_ns", "mean_uncertainty_std_ns", "mean_abs_error_over_std"]
            if identity == "test_id":
                columns.append("fit_failure_rate")
            for old, current in zip(reference_rows(9, cell), science[key], strict=True):
                require(old[1:3] == [current["method"], current[identity]], "Historical classical row identity")
                for name, value in zip(columns, old[3:], strict=True):
                    compare(comparisons, f"Notebook14/{current[identity]}/{current['method']}/{name}",
                            current[name], float(value), 4)
        table = findings.split("| Test | Covariance coverage | Bootstrap coverage | Covariance mean width (ns)", 1)[1]
        for line, current in zip(table.splitlines()[2:8], science["conditions"], strict=True):
            old = [v.strip() for v in line.split("|")[1:-1]]
            require(old[0] == current["condition_id"], "Historical Day-62 ordering")
            for (method, metric), value in zip((("covariance", "coverage"), ("bootstrap", "coverage"),
                    ("covariance", "mean_width_ns"), ("bootstrap", "mean_width_ns")), old[1:], strict=True):
                compare(comparisons, f"scientific_findings/{old[0]}/{method}/{metric}", current[method][metric], float(value), 3)
    return comparisons["comparisons"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--week7", action="store_true")
    modes.add_argument("--week8", action="store_true")
    modes.add_argument("--all", action="store_true", help="Week 7 and Week 8 only")
    modes.add_argument("--week9-ml", action="store_true", help="Week-9 ML/conformal and A/F residuals; no classical uncertainty")
    modes.add_argument("--week9-classical-day61", action="store_true", help="Full 8 x 50 x 100 sequential-RNG calibration")
    modes.add_argument("--week9-classical-day62", action="store_true", help="Full A-F x 192 x 200 sequential-RNG robustness")
    modes.add_argument("--week9-classical", action="store_true", help="Both full classical protocols; intentionally expensive")
    reference_mode = parser.add_mutually_exclusive_group(required=True)
    reference_mode.add_argument("--reference", choices=["issue12-post-correction", "issue12-week9-ml", "issue12-week9-classical"],
                                help="Compare to the reviewed current JSON, not historical notebook tables")
    reference_mode.add_argument("--collect-candidate", action="store_true",
                                help="Collect temporary evidence only; never writes or approves a reference")
    parser.add_argument("--repeat-report", type=Path,
                        help="Compare exact deterministic fingerprints/metrics with an independent prior run")
    parser.add_argument("--compare-report", type=Path,
                        help="Read-only reference comparison of completed evidence; no scientific traversal")
    args = parser.parse_args(argv)
    if args.week9_classical and args.collect_candidate:
        parser.error("Collect Day 61 and Day 62 separately; verify Day-61 repeats before starting Day 62")
    classical_mode = args.week9_classical or args.week9_classical_day61 or args.week9_classical_day62
    expected_reference = ("issue12-week9-classical" if classical_mode else
                          "issue12-week9-ml" if args.week9_ml else "issue12-post-correction")
    if args.reference and args.reference != expected_reference:
        parser.error("Reference must match the explicitly selected workflow")
    if args.compare_report and not args.reference:
        parser.error("--compare-report requires --reference")
    reference_path = (CLASSICAL_REFERENCE_PATH if classical_mode else
                      WEEK9_REFERENCE_PATH if args.week9_ml else REFERENCE_PATH)
    loader = read_classical_reference if classical_mode else read_week9_reference if args.week9_ml else read_current_reference
    reference = loader() if args.reference else None
    blocker = BlockFlatFacades()
    require(not blocker.names.intersection(sys.modules), "Start acceptance in a fresh interpreter")
    sys.meta_path.insert(0, blocker)
    output = Path(tempfile.mkdtemp(prefix="tcspc-issue12-stage3a-")) / "acceptance.json"
    report = {
        "acceptance_version": VERSION,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {"Python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn")}},
        "facade_blocker_count": len(blocker.names), "workflows": {},
        "reference_selection": args.reference or "candidate-collection-not-acceptance",
        "reference": ({"path": str(reference_path.relative_to(ROOT)),
                       "id": reference["reference_id"], "version": reference["reference_version"],
                       "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                       "comparison_tolerances": reference["comparison_tolerances"]} if reference else None),
        "deterministic_rerun": "NOT REQUESTED",
        "references": {str(w): {"path": "notebooks/" + name,
            "sha256": hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest()}
            for w, name in NOTEBOOKS.items()},
    }
    print(f"Report: {output}", flush=True)
    try:
        for week, run in ((7, week7), (8, week8), ("9_ml", week9_ml),
                          ("9_classical_day61", week9_classical_day61), ("9_classical_day62", week9_classical_day62)):
            if not ((args.all and week in (7, 8)) or getattr(args, f"week{week}")
                    or (args.week9_classical and week in ("9_classical_day61", "9_classical_day62"))):
                continue
            result = report["workflows"][f"week{week}"] = {}
            started = perf_counter()
            try:
                if args.compare_report:
                    prior = json.loads(args.compare_report.read_text(encoding="utf-8"))
                    require(prior["status"] != "FAIL" and prior["facade_blocker_count"] == 38,
                            "Cannot accept failed/unblocked evidence")
                    if classical_mode:
                        require(prior["deterministic_rerun"] == "PASS", "Classical evidence needs independent exact agreement")
                    result.update(prior["workflows"][f"week{week}"])
                    result["scientific_elapsed_seconds"] = result["elapsed_seconds"]
                    if classical_mode and "details_path" in result:
                        # Copy/enrich reporting only; never overwrite original evidence.
                        result["details"] = json.loads(Path(result["details_path"]).read_text(encoding="utf-8"))
                    report["evidence_source"] = dict(path=str(args.compare_report),
                        sha256=hashlib.sha256(args.compare_report.read_bytes()).hexdigest(),
                        git_commit=prior["git_commit"], versions=prior["versions"],
                        deterministic_rerun=prior["deterministic_rerun"])
                else:
                    run(result)
                if week in ("9_classical_day61", "9_classical_day62"):
                    result["historical_notebook_comparison"] = classical_historical_comparison(
                        61 if week.endswith("61") else 62, result["science"])
                historical = result["historical_notebook_comparison"]
                result["historical_difference_summary"] = {
                    "role": "informational; historical notebook outputs are not current acceptance oracles",
                    "differences": sum(r["status"] == "FAIL" for r in historical),
                    "comparisons": len(historical),
                }
                if reference:
                    expected = (reference[week.removeprefix("9_classical_")] if classical_mode
                                else reference["weeks"][f"week{week}"])
                    differences = compare_current(result["science"], expected,
                                                  reference["comparison_tolerances"][f"week{week}"])
                    result["current_reference_acceptance"] = dict(
                        status="FAIL" if differences else "PASS", discrepancies=differences)
                    require(not differences, f"Week {week}: current reference differs: {differences[:3]}")
                else:
                    result["current_reference_acceptance"] = dict(status="NOT RUN — candidate collection")
                result["status"] = "PASS" if reference else "CANDIDATE"
            except Exception as exc:
                result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                print(result["error"], flush=True)
                # Stop this workflow. The other independent week may still be audited.
            finally:
                if "details" in result:
                    detail_path = output.with_name(f"week{week}_details.json")
                    detail_path.write_text(json.dumps(classical_detail_records(result.pop("details")),
                        indent=2, allow_nan=False), encoding="utf-8")
                    result["details_path"] = str(detail_path)
                result["elapsed_seconds"] = perf_counter() - started
                report["status"] = ("FAIL" if any(r["status"] == "FAIL" for r in report["workflows"].values())
                                    else "PASS" if reference else "CANDIDATE")
                output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                print(f"Week {week}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
        if args.repeat_report and report["status"] != "FAIL":
            try:
                verify_repeat(report, json.loads(args.repeat_report.read_text(encoding="utf-8")))
            except (AcceptanceFailure, KeyError) as exc:
                report.update(status="FAIL", deterministic_rerun="FAIL", repeat_error=str(exc))
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Independent deterministic rerun: {report['deterministic_rerun']}", flush=True)
    finally:
        sys.meta_path.remove(blocker)
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
