"""Explicit Week-7/8 acceptance; never executes notebooks or updates references.

Run with the project interpreter, e.g. ``python scripts/run_issue12_scientific_reproducibility.py
--all``. Reports go to a new system temporary directory. This is deliberately
outside ordinary pytest: full classical traversal can take several minutes.
"""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
VERSION = "issue12-stage3a-v1"
METHODS = ("ridge", "random_forest", "hist_gradient_boosting")
REPRESENTATIONS = ("engineered_features", "normalized_histogram", "pca_histogram")
TEST_IDS = tuple("ABCDEF")
NOTEBOOKS = {
    7: "12_ml_benchmarking.ipynb",
    8: "13_generalization_and_robustness.ipynb",
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


def reference_rows(week, cell_id):
    path = ROOT / "notebooks" / NOTEBOOKS[week]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == cell_id)
    tables = [o["data"]["text/html"] for o in cell.get("outputs", [])
              if "text/html" in o.get("data", {})]
    require(len(tables) == 1, f"Expected one saved reference table at {path.name}:{cell_id}")
    parser = BodyRows()
    parser.feed("".join(tables[0]))
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
    expected = reference_rows(7, "83354e7feeb8b834")
    columns = list(table.columns)
    require([r[1] for r in expected] == [*METHODS, "reconvolution"], "Week-7 reference ordering")
    for row in expected[:-1]:
        for column, value in zip(columns, row[2:], strict=True):
            compare(report, f"{row[1]}/{column}", table.loc[row[1], column], float(value), 6)
    stop_on_discrepancy(report, "Week-7 ML/reporting")
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
    stop_on_discrepancy(report, "Week-7 classical fitting/reporting")
    timing = benchmark_inference_runtime(split=split, reconvolution_result=classical.in_distribution,
                                        n_repeats=200, warmup_runs=10)
    require(tuple(timing) == ("mean_arrival_time", *METHODS, "reconvolution"), "Timing ordering")
    timing_table = summarize_inference_timing(timing)
    require(np.isfinite(timing_table["median_time_per_curve_ms"]).all(), "Nonfinite timing")
    require((timing_table["median_time_per_curve_ms"] >= 0).all(), "Negative timing")
    require(set(timing_table["timing_mode"]) == {"repeated_batch", "per_curve_optimization"}, "Timing scopes")
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
    stop_on_discrepancy(report, "Week-8 classical fitting/reporting")
    # Public orchestration reuses the same Test-C observations for both IRF assumptions.
    before = fingerprint(tests["C"])
    instrument = evaluate_classical_instrument_acquisition_benchmark(tests=tests, irf_centre_ns=1., nominal_irf_fwhm_ns=.4)
    require(before == fingerprint(tests["C"]), "Test-C observations mutated")
    report["C_IRF_comparison"] = records(instrument.test_c_irf_comparison)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--week7", action="store_true")
    modes.add_argument("--week8", action="store_true")
    modes.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
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
        "references": {str(w): {"path": "notebooks/" + name,
            "sha256": hashlib.sha256((ROOT / "notebooks" / name).read_bytes()).hexdigest()}
            for w, name in NOTEBOOKS.items()},
    }
    print(f"Report: {output}", flush=True)
    try:
        for week, run in ((7, week7), (8, week8)):
            if not (args.all or getattr(args, f"week{week}")):
                continue
            result = report["workflows"][f"week{week}"] = {}
            started = perf_counter()
            try:
                run(result)
                result["status"] = "PASS"
            except Exception as exc:
                result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                print(result["error"], flush=True)
                # Stop this workflow. The other independent week may still be audited.
            finally:
                result["elapsed_seconds"] = perf_counter() - started
                report["status"] = "PASS" if all(r["status"] == "PASS" for r in report["workflows"].values()) else "FAIL"
                output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                print(f"Week {week}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    finally:
        sys.meta_path.remove(blocker)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
