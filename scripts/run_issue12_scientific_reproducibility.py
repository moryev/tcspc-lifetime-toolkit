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
VERSION = "issue12-stage3b1-v1"
REFERENCE_PATH = ROOT / "data/reference/issue12_week7_week8_post_correction.json"
REFERENCE_ID = "issue12-week7-week8-post-correction"
WEEK9_REFERENCE_PATH = ROOT / "data/reference/issue12_week9_ml_uncertainty_post_correction.json"
WEEK9_REFERENCE_ID = "issue12-week9-ml-uncertainty-post-correction"
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


def flat_facades():
    """The reviewed 38 root .py facades, not historical package initializers."""
    support = {"__init__", "__main__", "cli", "config", "exceptions"}
    names = {p.stem for p in (ROOT / "src/tcspc_toolkit").glob("*.py")} - support
    require(len(names) == 38, "Review the flat-facade inventory before running acceptance")
    return {"tcspc_toolkit." + name for name in names}


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
        representation = "none" if method == "constant_mean" else "engineered_features"
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--week7", action="store_true")
    modes.add_argument("--week8", action="store_true")
    modes.add_argument("--all", action="store_true", help="Week 7 and Week 8 only")
    modes.add_argument("--week9-ml", action="store_true", help="Week-9 ML/conformal and A/F residuals; no classical uncertainty")
    reference_mode = parser.add_mutually_exclusive_group(required=True)
    reference_mode.add_argument("--reference", choices=["issue12-post-correction", "issue12-week9-ml"],
                                help="Compare to the reviewed current JSON, not historical notebook tables")
    reference_mode.add_argument("--collect-candidate", action="store_true",
                                help="Collect temporary evidence only; never writes or approves a reference")
    parser.add_argument("--repeat-report", type=Path,
                        help="Compare exact deterministic fingerprints/metrics with an independent prior run")
    args = parser.parse_args(argv)
    if args.reference and (args.week9_ml != (args.reference == "issue12-week9-ml")):
        parser.error("Select issue12-week9-ml only with --week9-ml; Week-7/8 references are unchanged")
    reference_path = WEEK9_REFERENCE_PATH if args.week9_ml else REFERENCE_PATH
    reference = (read_week9_reference() if args.week9_ml else read_current_reference()) if args.reference else None
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
        for week, run in ((7, week7), (8, week8), ("9_ml", week9_ml)):
            if not ((args.all and week in (7, 8)) or getattr(args, f"week{week}")):
                continue
            result = report["workflows"][f"week{week}"] = {}
            started = perf_counter()
            try:
                run(result)
                historical = result["historical_notebook_comparison"]
                result["historical_difference_summary"] = {
                    "role": "informational; historical notebook outputs are not current acceptance oracles",
                    "differences": sum(r["status"] == "FAIL" for r in historical),
                    "comparisons": len(historical),
                }
                if reference:
                    differences = compare_current(result["science"], reference["weeks"][f"week{week}"],
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
