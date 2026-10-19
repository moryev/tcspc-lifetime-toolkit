"""Lightweight harness contracts; no full scientific reruns in normal pytest."""

import ast
import importlib.util
import json
import copy
import re
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scientific_acceptance", ROOT / "scripts/run_issue12_scientific_reproducibility.py"
)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def test_blocker_matches_reviewed_retirement_manifest():
    tree = ast.parse((ROOT / "tests/test_package_structure.py").read_text(encoding="utf-8"))
    registry = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "REMOVED_FLAT_FACADES" for t in node.targets))
    packages = {"irf", "preprocessing", "simulation", "evaluation", "persistence"}
    assert acceptance.flat_facades() == {"tcspc_toolkit." + name for name in registry}
    assert len(registry) == 38
    assert all(not (ROOT / "src/tcspc_toolkit" / f"{name}.py").exists() for name in registry)
    blocker = acceptance.BlockFlatFacades()
    for name in blocker.names:
        with pytest.raises(ImportError, match="flat facade"):
            blocker.find_spec(name)
    for name in packages | {"config", "benchmarks.generalization", "estimation.classical"}:
        assert blocker.find_spec("tcspc_toolkit." + name) is None


def test_classical_full_profile_matches_historical_notebook_source():
    from tcspc_toolkit.uncertainty.classical import CLASSICAL_UNCERTAINTY_METHODS
    notebook = json.loads((ROOT / "notebooks/14_uncertainty_and_failure_awareness.ipynb").read_text(encoding="utf-8"))
    source = "".join(next(c["source"] for c in notebook["cells"] if c.get("id") == "517e9d4b7bfe797e"))
    tree = ast.parse(source)
    conditions = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "repeated_poisson_conditions" for t in n.targets))
    assert tuple(conditions) == acceptance.DAY61_CONDITIONS
    day61, day62 = acceptance.classical_protocol(61), acceptance.classical_protocol(62)
    assert day61["profile_id"] == day62["profile_id"] == "issue12-week9-classical-full-v1"
    assert (day61["n_repeated"], day61["n_bootstrap"], day62["n_per_test"], day62["n_bootstrap"]) == (50, 100, 192, 200)
    assert [c["seed"] for c in day61["conditions"]] == list(range(61001, 61009))
    assert day62["bootstrap_seed"] == 62001
    assert day62["test_order"] == list("ABCDEF")
    assert day62["external_seeds"] == dict(zip("ABCDEF", range(50001, 50007)))
    assert day61["nominal_coverage"] == day62["nominal_coverage"] == .90
    assert day62["test_f_reference"] == "primary_component"
    assert day61["methods"] == {"covariance": "prediction_interval", "parametric_bootstrap": "prediction_interval",
                                 "repeated_poisson_realizations": "empirical_reference"}
    assert day61["methods"] == {name: method.output_kind.value for name, method in CLASSICAL_UNCERTAINTY_METHODS.items()}


def test_classical_observers_preserve_rng_and_canonical_results():
    """Six cheap fits, not the full scientific profile, prove observation is passive."""
    import numpy as np
    from tcspc_toolkit.uncertainty.classical import evaluate_repeated_poisson_uncertainty
    options = dict(time=np.arange(0., 20., .05), true_lifetime_ns=2., signal_photon_count=1000,
        background_per_bin=.5, irf_centre_ns=1., irf_fwhm_ns=.25, irf_shift_ns=.05,
        temporal_shift_bounds=(-.5, .5), n_repeats=2, n_bootstrap_resamples=2, nominal_coverage=.90)
    plain_rng, observed_rng = np.random.default_rng(61001), np.random.default_rng(61001)
    plain = evaluate_repeated_poisson_uncertainty(**options, rng=plain_rng)
    audit = acceptance.ClassicalObservationAudit([dict(condition_id="test", sample_id=i, reference_ns=2.)
                                                for i in range(2)], "unit")
    with audit.observe():
        observed = evaluate_repeated_poisson_uncertainty(**options, rng=observed_rng)
    assert acceptance.record_digest(acceptance.asdict(plain)) == acceptance.record_digest(acceptance.asdict(observed))
    assert plain_rng.bit_generator.state == observed_rng.bit_generator.state
    summary = acceptance.classical_population(audit.rows)
    assert summary["primary"]["valid"] == 2
    assert summary["bootstrap"]["requested_refits"] == summary["bootstrap"]["valid_refits"] == 4
    assert summary["covariance"]["coverage"] == observed.covariance_metrics.empirical_coverage
    assert summary["bootstrap"]["coverage"] == observed.bootstrap_metrics.empirical_coverage
    assert all(len(r["bootstrap"]["refit_fingerprint"]) == 64 for r in audit.rows)
    detailed = acceptance.classical_detail_records(audit.rows)
    assert "lower_ns" not in audit.rows[0]["covariance"]
    np.testing.assert_array_equal([r["covariance"]["lower_ns"] for r in detailed], observed.covariance_intervals.lower)
    np.testing.assert_array_equal([r["covariance"]["upper_ns"] for r in detailed], observed.covariance_intervals.upper)
    assert audit.rows[0]["bootstrap"]["rng_after"] != audit.rows[1]["bootstrap"]["rng_before"]
    assert set(acceptance.optimizer_call_sites().values()) == {"primary", "continuation", "half_bin_fallback"}
    # Reporting must expose partial refit failure, not invent a stricter policy.
    partial = copy.deepcopy(audit.rows[:1])
    partial[0]["bootstrap"].update(n_resamples=100, n_successful_fits=2, n_failed_fits=98, fit_failure_rate=.98)
    partial_summary = acceptance.classical_population(partial)["bootstrap"]
    assert partial_summary["valid"] == partial_summary["coverage_denominator"] == 1
    assert partial_summary["requested_refits"] == 100
    assert partial_summary["valid_refits"] == partial_summary["minimum_valid_refits"] == 2
    assert partial_summary["failed_refits"] == 98
    assert partial_summary["aggregate_refit_failure_rate"] == .98
    assert partial_summary["high_refit_failure_observations"] == [dict(sample_id=0, requested=100, valid=2, failed=98)]
    # A finite-invalid primary is not silently counted as a nonfinite prediction.
    rows = copy.deepcopy(audit.rows)
    rows[0]["primary"].update(valid_fit=False, failure_reason="optimizer_unsuccessful")
    rows[0].update(covariance=None, bootstrap=None)
    reduced = acceptance.classical_population(rows)
    assert reduced["primary"]["strict_primary_fit_failure_rate"] == .5
    assert reduced["primary"]["legacy_finite_prediction_failure_rate"] == 0.
    for method in ("covariance", "bootstrap"):
        assert reduced[method]["attempted"] == 2
        assert reduced[method]["evaluated"] == reduced[method]["valid"] == reduced[method]["coverage_denominator"] == 1
        assert reduced[method]["failed"] == 1
    assert reduced["bootstrap"]["requested_refits"] == 2
    rows[1]["primary"].update(valid_fit=False, failure_reason="optimizer_unsuccessful")
    rows[1].update(covariance=None, bootstrap=None)
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        unavailable = acceptance.classical_population(rows)
    assert not captured
    for method in ("covariance", "bootstrap"):
        assert unavailable[method]["coverage"] is None
        assert unavailable[method]["mean_std_ns"] is None
        assert unavailable[method]["attempted"] == unavailable[method]["failed"] == 2
        assert unavailable[method]["valid"] == unavailable[method]["coverage_denominator"] == 0


def test_wilson_is_descriptive_not_an_acceptance_tolerance():
    assert acceptance.wilson_interval(0, 0) is None
    lo, hi = acceptance.wilson_interval(45, 50)
    assert .78 < lo < .80 and .95 < hi < .97
    assert acceptance.compare_current({"covered": 44}, {"covered": 45}, {})


def test_two_historical_day61_tables_remain_distinct_informational_sources():
    rows = acceptance.reference_rows(9, "517e9d4b7bfe797e")
    conditions = []
    for i, row in enumerate(rows, 1):
        empirical, cov, cov_ratio, boot, boot_ratio, cov_coverage, boot_coverage = map(float, row[2:])
        conditions.append(dict(condition_id=f"day61-{i}", empirical=dict(std_ns=empirical),
            covariance=dict(mean_std_ns=cov, std_ratio=cov_ratio, coverage=cov_coverage),
            bootstrap=dict(mean_std_ns=boot, std_ratio=boot_ratio, coverage=boot_coverage)))
    comparisons = acceptance.classical_historical_comparison(61, dict(conditions=conditions))
    assert len(comparisons) == 112
    assert all(c["status"] == "PASS" for c in comparisons if c["label"].startswith("Notebook14/"))
    assert any(c["status"] == "FAIL" for c in comparisons if c["label"].startswith("scientific_findings/"))


def test_historical_day62_comparison_reads_saved_tables_without_scientific_execution():
    science = {}
    for cell, key, identity in (("a9df22789358b5cc", "legacy_scorecard", "test_id"),
                               ("fef57a13bfb661b8", "legacy_conditional_scorecard", "condition_id")):
        columns = ["mae_ns", "empirical_coverage", "mean_width_ns", "mean_uncertainty_std_ns", "mean_abs_error_over_std"]
        if identity == "test_id":
            columns.append("fit_failure_rate")
        science[key] = [dict(method=row[1], **{identity: row[2]}, **dict(zip(columns, map(float, row[3:]), strict=True)))
                        for row in acceptance.reference_rows(9, cell)]
    science["conditions"] = []
    for tid in "ABCDEF":
        rows = [r for r in science["legacy_scorecard"] if r["test_id"] == tid]
        science["conditions"].append(dict(condition_id=tid, **{
            method: dict(coverage=row["empirical_coverage"], mean_width_ns=row["mean_width_ns"])
            for method, row in zip(("covariance", "bootstrap"), rows, strict=True)}))
    comparisons = acceptance.classical_historical_comparison(62, science)
    assert all(c["status"] == "PASS" for c in comparisons if c["label"].startswith("Notebook14/"))
    assert sum(c["label"].startswith("scientific_findings/") for c in comparisons) == 24


@pytest.mark.parametrize("repeated", ["PASS", "NOT REQUESTED"])
def test_classical_saved_evidence_comparison_never_runs_a_third_traversal(tmp_path, monkeypatch, repeated):
    reference = dict(reference_id=acceptance.CLASSICAL_REFERENCE_ID, reference_version=1,
        comparison_tolerances={"week9_classical_day61": {}}, day61={"n_total": 400})
    # Reuse an existing read-only path for metadata hashing; the loader is mocked.
    monkeypatch.setattr(acceptance, "CLASSICAL_REFERENCE_PATH", acceptance.REFERENCE_PATH)
    monkeypatch.setattr(acceptance, "read_classical_reference", lambda: reference)
    monkeypatch.setattr(acceptance, "flat_facades", lambda: set())
    monkeypatch.setattr(acceptance, "classical_historical_comparison", lambda *args: [])
    monkeypatch.setattr(acceptance, "week9_classical_day61", lambda report: pytest.fail("third scientific traversal"))
    monkeypatch.setattr(acceptance.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    evidence = tmp_path / "candidate.json"
    details = tmp_path / "original-details.json"
    details.write_text("[]")
    evidence.write_text(json.dumps(dict(status="CANDIDATE", facade_blocker_count=38,
        git_commit="scientific-commit", versions={}, deterministic_rerun=repeated,
        workflows={"week9_classical_day61": dict(science={"n_total": 400}, elapsed_seconds=123.,
            historical_notebook_comparison=[], repeatability={"records": "exact-digest"}, details_path=str(details))})))
    before = evidence.read_bytes()
    code = acceptance.main(["--week9-classical-day61", "--reference", "issue12-week9-classical",
                            "--compare-report", str(evidence)])
    assert code == (0 if repeated == "PASS" else 1)
    assert evidence.read_bytes() == before
    assert details.read_bytes() == b"[]"
    report = json.loads((tmp_path / "acceptance.json").read_text())
    if repeated == "PASS":
        assert report["workflows"]["week9_classical_day61"]["current_reference_acceptance"]["status"] == "PASS"
        assert report["workflows"]["week9_classical_day61"]["scientific_elapsed_seconds"] == 123.
        copied = Path(report["workflows"]["week9_classical_day61"]["details_path"])
        assert copied != details and json.loads(copied.read_text()) == []


def test_combined_candidate_cannot_start_day62_before_day61_repeat_review():
    with pytest.raises(SystemExit) as error:
        acceptance.main(["--week9-classical", "--collect-candidate"])
    assert error.value.code == 2


def test_classical_current_reference_identity_provenance_and_rng_policy():
    reference = acceptance.read_classical_reference()
    assert acceptance.CLASSICAL_REFERENCE_PATH.name == "issue12_week9_classical_uncertainty_post_correction.json"
    assert reference["reference_version"] == 1
    assert reference["profile_id"] == "issue12-week9-classical-full-v1"
    assert reference["scientific_source_commit"] == "b21e3411f85f68a4db8cd304277f0e65acd5e27d"
    assert {"reference_id", "purpose", "environment", "provenance", "comparison_policy",
            "day61", "day62", "historical_context"} <= reference.keys()
    assert set(reference["environment"]) == {"Python", "numpy", "scipy", "pandas", "scikit-learn"}
    provenance = reference["provenance"]
    assert len(provenance["commits"]) == 6
    assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in provenance["commits"].values())
    assert provenance["independent_reruns"]["status"] == "PASS"
    for day, count in ((61, 400), (62, 1152)):
        evidence = provenance["independent_reruns"]["days"][f"day{day}"]
        assert evidence["n_records"] == count
        assert evidence["facade_blocker_count"] == 38
        assert len(evidence["report_sha256"]) == len(evidence["detailed_file_sha256"]) == 2
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in evidence["report_sha256"])
        assert reference[f"day{day}"]["protocol"] == acceptance.classical_protocol(day)
    assert "primary measurement, bootstrap draws, next primary" in provenance["sequential_rng"]["day61"]
    assert "62001" in provenance["sequential_rng"]["day62"]
    assert "Invalid primary fits consume no bootstrap draws" in provenance["sequential_rng"]["day62"]
    assert reference["historical_context"]["day61"]["notebook_comparisons"] == 56
    assert reference["historical_context"]["day61"]["findings_comparisons"] == 56
    assert reference["historical_context"]["day62"]["comparisons"] == 146
    assert "informational" in reference["historical_context"]["policy"]


@pytest.mark.parametrize("day", [61, 62])
def test_classical_reference_population_schema_and_coverage_denominators(day):
    reference = acceptance.read_classical_reference()
    science = reference[f"day{day}"]
    ids = [f"day61-{i}" for i in range(1, 9)] if day == 61 else list("ABCDEF")
    assert [c["condition_id"] for c in science["conditions"]] == ids
    count, refits = (50, 100) if day == 61 else (192, 200)
    for row in science["conditions"]:
        assert row["n_total"] == row["primary"]["attempted"] == count
        assert row["primary"]["valid"] + row["primary"]["failed"] == count
        assert len(row["primary"]["failures"]) == row["primary"]["failed"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["input_fingerprint"])
        assert re.fullmatch(r"[0-9a-f]{64}", row["result_fingerprint"])
        for method in ("covariance", "bootstrap"):
            interval = row[method]
            assert interval["attempted"] == count
            assert interval["evaluated"] == row["primary"]["valid"]
            assert interval["valid"] + interval["failed"] == count
            assert len(interval["failures"]) == interval["failed"]
            assert interval["coverage_denominator"] == interval["valid"]
            assert interval["coverage"] == interval["covered"] / interval["valid"]
            assert interval["failure_rate"] == interval["failed"] / count
            assert {"mean_std_ns", "mean_width_ns", "median_width_ns"} <= interval.keys()
        boot = row["bootstrap"]
        assert boot["requested_refits"] == row["primary"]["valid"] * refits
        assert boot["requested_refits"] == boot["valid_refits"] + boot["failed_refits"]
        assert boot["aggregate_refit_failure_rate"] == boot["failed_refits"] / boot["requested_refits"]
    totals = [sum(c[k]["valid"] for c in science["conditions"]) for k in ("primary", "covariance", "bootstrap")]
    assert totals == ([400, 400, 400] if day == 61 else [1151, 1150, 1151])
    assert sum(c["bootstrap"]["failed_refits"] for c in science["conditions"]) == (1 if day == 61 else 96)


def test_classical_reference_retains_strict_failure_and_test_f_semantics():
    science = acceptance.read_classical_reference()["day62"]
    assert science["protocol"]["test_f_reference"] == "primary_component"
    assert science["protocol"]["weighted_reference"] == "descriptive only, not pseudo_true_mono"
    c = science["conditions"][2]
    assert c["primary"]["failures"] == [{"sample_id": 69, "failure_reason": "optimizer_unsuccessful"}]
    assert c["primary"]["strict_primary_fit_failure_rate"] == 1 / 192
    assert c["primary"]["legacy_finite_prediction_failure_rate"] == 0.
    f = science["conditions"][-1]
    assert f["covariance"]["failures"] == [{"sample_id": 165, "failure_reason": "parameter_at_bound"}]
    assert [r["severity"] for r in science["test_f_severity"]] == ["weak", "moderate"]
    weak, moderate = science["test_f_severity"]
    assert weak["n_total"] == moderate["n_total"] == 96
    assert weak["covariance"]["failures"] == f["covariance"]["failures"]
    assert (weak["covariance"]["covered"], weak["covariance"]["valid"]) == (66, 95)
    assert (moderate["covariance"]["covered"], moderate["covariance"]["valid"]) == (33, 96)
    assert (weak["bootstrap"]["covered"], moderate["bootstrap"]["covered"]) == (63, 32)


@pytest.mark.parametrize("day", [61, 62])
def test_classical_comparator_uses_named_precision_but_exact_populations_and_fingerprints(day):
    reference = acceptance.read_classical_reference()
    expected = reference[f"day{day}"]
    tolerances = reference["comparison_tolerances"][f"week9_classical_day{day}"]
    assert set(tolerances.values()) == {5e-7, 5e-5, 1e-12}
    assert not {"nominal_coverage", "covered", "valid", "failed", "seed", "bootstrap_seed"} & tolerances.keys()
    assert acceptance.compare_current(expected, expected, tolerances) == []
    actual = copy.deepcopy(expected)
    actual["conditions"][0]["covariance"]["mean_std_ns"] += 2e-5
    assert acceptance.compare_current(actual, expected, tolerances) == []
    actual["conditions"][0]["covariance"]["mean_std_ns"] += 5e-5
    assert acceptance.compare_current(actual, expected, tolerances)
    for section, key, value in (("primary", "valid", 0), ("covariance", "covered", 0)):
        actual = copy.deepcopy(expected)
        actual["conditions"][0][section][key] = value
        assert acceptance.compare_current(actual, expected, tolerances)
    actual = copy.deepcopy(expected)
    actual["conditions"][0]["result_fingerprint"] = "0" * 64
    assert acceptance.compare_current(actual, expected, tolerances)


@pytest.mark.parametrize("day", [61, 62])
def test_real_classical_reference_offline_acceptance_is_read_only(day, tmp_path, monkeypatch):
    reference = acceptance.read_classical_reference()
    original_reference = acceptance.CLASSICAL_REFERENCE_PATH.read_bytes()
    key = f"week9_classical_day{day}"
    monkeypatch.setattr(acceptance, "flat_facades", lambda: set())
    for name in ("week9_classical_day61", "week9_classical_day62"):
        monkeypatch.setattr(acceptance, name, lambda report: pytest.fail("scientific traversal forbidden"))
    monkeypatch.setattr(acceptance, "classical_historical_comparison", lambda *args: [])
    monkeypatch.setattr(acceptance.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    evidence = tmp_path / "completed.json"
    evidence.write_text(json.dumps(dict(status="CANDIDATE", facade_blocker_count=38,
        git_commit=reference["scientific_source_commit"], versions=reference["environment"],
        deterministic_rerun="PASS", workflows={key: dict(science=reference[f"day{day}"],
        elapsed_seconds=123., historical_notebook_comparison=[], repeatability={})})))
    before = evidence.read_bytes()
    assert acceptance.main([f"--week9-classical-day{day}", "--reference", "issue12-week9-classical",
                            "--compare-report", str(evidence)]) == 0
    assert evidence.read_bytes() == before
    assert acceptance.CLASSICAL_REFERENCE_PATH.read_bytes() == original_reference
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["status"] == report["workflows"][key]["current_reference_acceptance"]["status"] == "PASS"


def test_external_observer_matches_canonical_reporting_and_sequential_stream():
    from types import SimpleNamespace
    from dataclasses import replace
    from tcspc_toolkit.benchmarks.generalization_protocol import default_generalization_suite
    from tcspc_toolkit.benchmarks.generalization_datasets import generate_generalization_test_suite
    from tcspc_toolkit.benchmarks.uncertainty import evaluate_classical_uncertainty_robustness
    definition = default_generalization_suite()
    # Also retain B-high/F-moderate so canonical conditional reporting is defined.
    # Eight observations and two refits each, never the full scientific profile.
    tests = {}
    for t in generate_generalization_test_suite(definition).tests:
        indices = [0]
        if t.test_id == "B":
            indices.append(int(t.metadata.index[t.metadata.signal_photon_count_target.isin(
                definition.numerics.test_b_high_photon_counts)][0]))
        if t.test_id == "F":
            indices.append(int(t.metadata.index[t.metadata.model_mismatch_severity == "moderate"][0]))
        tests[t.test_id] = replace(t, X_histograms=t.X_histograms[indices], y=t.y[indices],
            metadata=t.metadata.iloc[indices].reset_index(drop=True))
    audit = acceptance.ClassicalObservationAudit([dict(condition_id=t.test_id, sample_id=int(t.metadata.iloc[i].sample_id),
        reference_ns=float(y)) for t in tests.values() for i, y in enumerate(t.y)], "unit-external")
    with audit.observe():
        result = evaluate_classical_uncertainty_robustness(prepared=SimpleNamespace(tests=tests),
            definition=definition, n_bootstrap_resamples=2, nominal_coverage=.90, bootstrap_random_seed=62001)
    assert len(audit.rows) == 8
    assert result.per_curve.classical_fit_valid.all()
    bootstraps = [r["bootstrap"] for r in audit.rows]
    assert all(a["rng_after"] == b["rng_before"] for a, b in zip(bootstraps, bootstraps[1:]))
    for row, canonical in zip(audit.rows, result.per_curve.itertuples(index=False), strict=True):
        row["covariance"].update(lower_ns=canonical.covariance_lower_ns, upper_ns=canonical.covariance_upper_ns)
        summary = acceptance.classical_population([row])
        assert summary["covariance"]["valid"] == int(canonical.covariance_valid)
        assert summary["bootstrap"]["valid"] == int(canonical.bootstrap_valid)
    assert next(r for r in audit.rows if r["condition_id"] == "C")["primary"]["optimizer_calls"]["half_bin_fallback"] == 2


@pytest.mark.parametrize("week,cell,rows,width", [
    (7, "83354e7feeb8b834", 4, 10),
    (8, "e802c319fc9ab6ad", 6, 7),
    (8, "93c3d55a39cf7b40", 6, 6),
    (8, "f5e40edbc2b8ea2b", 1, 7),
])
def test_committed_reference_table_is_readable_without_execution(week, cell, rows, width):
    result = acceptance.reference_rows(week, cell)
    assert len(result) == rows
    assert {len(row) for row in result} == {width}


def test_html_reader_ignores_headers_and_preserves_body_labels():
    reader = acceptance.BodyRows()
    reader.feed('<table><thead><tr><th>ignore</th></tr></thead><tbody>'
                '<tr><th>method</th><td><b>0.2500</b></td></tr></tbody></table>')
    assert reader.rows == [["method", "0.2500"]]


@pytest.mark.parametrize("decimals", [3, 4, 6])
def test_comparison_uses_only_half_of_displayed_unit(decimals):
    unit = 10 ** -decimals
    report = {}
    acceptance.compare(report, "inside", 1 + .49 * unit, 1, decimals)
    acceptance.compare(report, "outside", 1 + .51 * unit, 1, decimals)
    assert [r["status"] for r in report["comparisons"]] == ["PASS", "FAIL"]
    with pytest.raises(acceptance.AcceptanceFailure, match="1 reference comparisons failed"):
        acceptance.stop_on_discrepancy(report, "test phase")
    assert report["first_discrepancy_phase"] == "test phase"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_discrepancies_are_failures_and_json_safe(value):
    report = {}
    acceptance.compare(report, "nonfinite", value, 1., 4)
    assert report["comparisons"][0]["status"] == "FAIL"
    json.dumps(report, allow_nan=False)


def test_missing_failure_reason_is_a_json_null_not_nan_dictionary_key():
    frame = pd.DataFrame({"failure_reason": [None, "poisson_numerical_validation_failed", None]})
    counts = frame.groupby("failure_reason", dropna=False).size().reset_index(name="count")
    result = acceptance.records(counts)
    json.dumps(result, allow_nan=False)
    assert {r["count"] for r in result} == {1, 2}
    assert next(r for r in result if r["count"] == 2)["failure_reason"] is None


def test_discrepancy_writes_fail_report_and_does_not_continue_workflow(tmp_path, monkeypatch):
    def fail(report):
        acceptance.compare(report, "scientific metric", 2., 1., 4)
        acceptance.stop_on_discrepancy(report, "fitting")
        pytest.fail("workflow continued after discrepancy")

    monkeypatch.setattr(acceptance, "week7", fail)
    monkeypatch.setattr(acceptance, "flat_facades", lambda: {"tcspc_toolkit.nonexistent_test_facade"})
    monkeypatch.setattr(acceptance.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(acceptance.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    assert acceptance.main(["--week7", "--collect-candidate"]) == 1
    result = json.loads((tmp_path / "acceptance.json").read_text())
    assert result["status"] == "FAIL"
    assert result["workflows"]["week7"]["first_discrepancy_phase"] == "fitting"
    assert "week8" not in result["workflows"]


def test_current_reference_schema_protocol_provenance_and_populations():
    reference = acceptance.read_current_reference()
    assert acceptance.REFERENCE_PATH == ROOT / "data/reference/issue12_week7_week8_post_correction.json"
    assert reference["reference_id"] == "issue12-week7-week8-post-correction"
    assert reference["reference_version"] == 1
    assert reference["purpose"]
    assert re.fullmatch(r"[0-9a-f]{40}", reference["source_commit"])
    for key in ("historical_week7_notebook_publication_commit", "historical_week8_notebook_publication_commit",
                "poisson_scaling_commit", "validation_recovery_commit", "issue12_fallback_fix_commit"):
        assert re.fullmatch(r"[0-9a-f]{40}", reference["historical_provenance"][key])
    assert set(reference["environment_used_for_reference"]) == {"Python", "numpy", "scipy", "pandas", "scikit-learn"}
    w7, w8 = reference["weeks"]["week7"], reference["weeks"]["week8"]
    assert (w7["n_development"], w7["n_train"], w7["n_test"], w7["n_bins"]) == (810, 648, 162, 400)
    assert [w7["protocol"][key] for key in ("simulation_seed", "split_seed", "mismatch_seed")] == [42]*3
    assert w7["protocol"]["references"] == ["generating_mono", "primary_component"]
    assert w8["protocol_id"] == "week8-day55-v2"
    assert w8["test_order"] == list("ABCDEF")
    assert (w8["development_seed"], w8["n_development"], w8["n_bins"]) == (42, 64, 400)
    assert w8["external_seeds"] == dict(zip("ABCDEF", range(50001, 50007)))
    assert w8["external_counts"] == dict.fromkeys("ABCDEF", 192)
    assert len(w8["scorecard"]) == 72
    assert w8["methods"] == list(acceptance.METHODS)
    assert w8["representations"] == list(acceptance.REPRESENTATIONS)
    assert w8["test_f"]["severity_counts"] == {"weak": 96, "moderate": 96}
    assert w8["test_f"]["reference"] == "primary_component"
    assert w8["test_f"]["descriptive_reference"] == "signal_photon_weighted"
    populations = [w7["matched"], w7["mismatch"], *w8["classical_populations"].values(),
                   w8["test_c_irf"]["correct"], w8["test_c_irf"]["misspecified"],
                   *w8["test_f"]["classical_populations"].values()]
    for population in populations:
        assert population["n_total"] == population["n_valid"] + population["n_failed"]
        assert population["n_failed"] == len(population["failures"])
        assert population["metric_population"] == "valid fits only"
    assert w8["classical_populations"]["C"]["failures"] == [
        {"sample_id": 69, "failure_reason": "optimizer_unsuccessful"}]
    for row in w8["scorecard"] + w8["test_f"]["metrics"]:
        assert row["n_total"] == row["n_valid"] + row["n_failed"]


def test_reference_tolerances_preserve_display_precision_and_exact_identity():
    ref = acceptance.read_current_reference()
    policy = ref["comparison_tolerances"]
    assert set(policy["week7"].values()) == {5e-7}
    assert policy["week8"]["mae_ns"] == 5e-5
    assert policy["week8"]["mae_ratio_over_A"] == 5e-4
    assert set(policy["week8"].values()) == {5e-5, 5e-4}
    assert not {"n_valid", "sample_id", "secondary_fraction", "seed"} & set(policy["week8"])
    for week, expected in ref["weeks"].items():
        assert acceptance.compare_current(expected, expected, policy[week]) == []
        def check_numeric_fields(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    if name in policy[week]:
                        assert isinstance(item, float), (week, name, item)
                    if name.startswith("n_") and isinstance(item, (int, float)):
                        assert type(item) is int, (week, name, item)
                    check_numeric_fields(item)
            elif isinstance(value, list):
                for item in value:
                    check_numeric_fields(item)
        check_numeric_fields(expected)


@pytest.mark.parametrize("observed,expected,passes", [
    ({"mae_ns": 1.000049}, {"mae_ns": 1.0}, True),
    ({"mae_ns": 1.000051}, {"mae_ns": 1.0}, False),
    ({"mae_ns": float("nan")}, {"mae_ns": 1.0}, False),
    ({"n_valid": 191.0}, {"n_valid": 191}, False),
    ({"n_valid": 192}, {"n_valid": 191}, False),
    ({"failures": [{"sample_id": 68}]}, {"failures": [{"sample_id": 69}]}, False),
    ({"ids": ["B", "A"]}, {"ids": ["A", "B"]}, False),
    ({"mae_ns": 1.0, "extra": 1}, {"mae_ns": 1.0}, False),
    ({"seed": 42.000001}, {"seed": 42.0}, False),
])
def test_current_comparator_is_not_a_blanket_float_tolerance(observed, expected, passes):
    assert (not acceptance.compare_current(observed, expected, {"mae_ns": 5e-5})) == passes


def test_independent_repeat_checks_fingerprints_and_not_machine_timing():
    prior = dict(versions={"numpy": "example"}, workflows={"week7": dict(
        science={"n_valid": 162}, repeatability={"classical": "exact-byte-digest"}, timing=1)})
    current = copy.deepcopy(prior)
    current["workflows"]["week7"]["timing"] = 999
    acceptance.verify_repeat(current, prior)
    assert current["deterministic_rerun"] == "PASS"
    current["workflows"]["week7"]["repeatability"]["classical"] = "different-parameters"
    with pytest.raises(acceptance.AcceptanceFailure, match="nondeterministic"):
        acceptance.verify_repeat(current, prior)


def test_population_and_paired_metrics_do_not_hide_missing_pairs():
    first = pd.DataFrame(dict(sample_id=[1, 2, 3], valid_fit=[True, False, True],
                              error_ns=[1., 100., -2.], failure_reason=[None, "optimizer_unsuccessful", None]))
    second = first.copy()
    second["valid_fit"] = [True, True, False]
    second["error_ns"] = [2., 0., 0.]
    assert acceptance.population(first)["n_failed"] == 1
    paired = acceptance.paired_error_change(first, second)
    assert paired == dict(n_complete_pairs=1, mean_absolute_error_change_ns=1.,
                          median_absolute_error_change_ns=1., n_worse=1, n_better=0, n_equal=0)


@pytest.mark.parametrize("mode,observed_count,expected_status", [
    (["--collect-candidate"], 162, "CANDIDATE"),
    (["--reference", "issue12-post-correction"], 162, "PASS"),
    (["--reference", "issue12-post-correction"], 161, "FAIL"),
])
def test_historical_differences_are_informational_and_candidate_is_not_acceptance(
    tmp_path, monkeypatch, mode, observed_count, expected_status,
):
    def run(report):
        report.update(science={"n_valid": observed_count}, historical_notebook_comparison=[{"status": "FAIL"}])

    monkeypatch.setattr(acceptance, "week7", run)
    monkeypatch.setattr(acceptance, "flat_facades", lambda: set())
    monkeypatch.setattr(acceptance.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(acceptance, "read_current_reference", lambda: dict(
        reference_id=acceptance.REFERENCE_ID, reference_version=1,
        comparison_tolerances={"week7": {}}, weeks={"week7": {"n_valid": 162}}))
    assert acceptance.main(["--week7", *mode]) == (1 if expected_status == "FAIL" else 0)
    report = json.loads((tmp_path / "acceptance.json").read_text())
    assert report["status"] == expected_status
    assert report["workflows"]["week7"]["historical_difference_summary"]["differences"] == 1
    if expected_status == "FAIL":
        assert report["workflows"]["week7"]["current_reference_acceptance"]["discrepancies"]


def test_reference_selection_is_explicit():
    with pytest.raises(SystemExit) as error:
        acceptance.main(["--week7"])
    assert error.value.code == 2


@pytest.mark.parametrize("args", [
    ["--week9-ml", "--reference", "issue12-post-correction"],
    ["--week8", "--reference", "issue12-week9-ml"],
])
def test_week9_reference_cannot_silently_replace_week7_week8(args):
    with pytest.raises(SystemExit) as error:
        acceptance.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("cell,index,count", [
    ("910a6d01bbf11f44", None, 5),
    ("339769e99c73f7e3", 0, 12),
    ("339769e99c73f7e3", 1, 12),
    ("7dde16d16f42f0e6", None, 3),
])
def test_week9_saved_tables_are_read_without_execution(cell, index, count):
    assert len(acceptance.reference_rows(9, cell, index)) == count


def test_week9_reference_schema_split_kinds_populations_and_provenance():
    from tcspc_toolkit.uncertainty.evaluation import make_uncertainty_development_split

    ref = acceptance.read_week9_reference()
    assert acceptance.WEEK9_REFERENCE_PATH == ROOT / "data/reference/issue12_week9_ml_uncertainty_post_correction.json"
    assert re.fullmatch(r"[0-9a-f]{40}", ref["source_commit"])
    for key, value in ref["historical_provenance"].items():
        if key.endswith("_commit"):
            assert re.fullmatch(r"[0-9a-f]{40}", value)
    assert ref["independent_rerun"]["status"] == "PASS"
    w9 = ref["weeks"]["week9_ml"]
    split = make_uncertainty_development_split(64)
    assert (w9["development_seed"], w9["split_seed"]) == (42, 57)
    assert (w9["n_development"], w9["n_training"], w9["n_calibration"]) == (64, 48, 16)
    assert w9["training_indices"] == list(split.training_indices)
    assert w9["calibration_indices"] == list(split.calibration_indices)
    assert set(w9["training_ids"]).isdisjoint(w9["calibration_ids"])
    w8 = acceptance.read_current_reference()["weeks"]["week8"]
    assert w9["protocol_id"] == w8["protocol_id"] == "week8-day55-v2"
    assert w9["test_order"] == list("ABCDEF")
    assert w9["external_counts"] == dict.fromkeys("ABCDEF", 192)
    assert w9["external_seeds"] == dict(zip("ABCDEF", range(50001, 50007)))
    assert w9["input_fingerprints"] == {"development": w8["development_sha256"], **w8["external_sha256"]}
    assert w9["conformal"]["finite_sample_rank"] == w9["conformal"]["n_calibration_scores"] == 16
    assert w9["estimator_policy"]["bootstrap_replicates"] == 200
    assert w9["estimator_policy"]["random_state"] == 42
    methods = {"quantile_gradient_boosting", "conformalized_quantile_gradient_boosting",
               "random_forest_tree_spread", "ml_training_bootstrap"}
    assert {r["method_id"] for r in w9["calibration"]} == methods
    assert len(w9["external"]) == 24
    assert len(w9["test_f_severity"]) == 8
    for group, count in (("calibration", 16), ("external", 192), ("test_f_severity", 96)):
        for row in w9[group]:
            population = row["population"]
            assert population["n_attempted"] == count == population["n_valid"] + population["n_failed"]
            assert len(population["failures"]) == population["n_failed"]
            assert row["point_metrics"]["n_attempted"] == count
            if row["output_kind"] == "score":
                assert row["nominal_level"] is None
                assert "empirical_coverage" not in row["metrics"]
            else:
                assert row["output_kind"] in {"quantile_prediction", "conformal_prediction"}
                assert row["nominal_level"] == .95 - .05
                assert row["metrics"]["n_valid_intervals"] == population["n_valid"]
    assert w9["test_f"]["severity_counts"] == {"weak": 96, "moderate": 96}
    assert w9["test_f"]["reference"] == "primary_component"
    assert len(w9["residual_summary"]) == 3
    tolerance = ref["comparison_tolerances"]["week9_ml"]
    assert set(tolerance.values()) == {5e-5}
    assert not {"nominal_level", "nominal_coverage", "tail_fraction", "random_state", "n_valid"} & tolerance.keys()
    assert acceptance.compare_current(w9, w9, tolerance) == []


@pytest.mark.parametrize("role", ["uncertainty_training", "uncertainty_calibration"])
@pytest.mark.parametrize("test_id", list("ABCDEF"))
def test_week9_external_provenance_rejects_training_and_calibration(test_id, role):
    from tcspc_toolkit.uncertainty.evaluation import UncertaintyDataProvenance, UncertaintyDataRole
    with pytest.raises(ValueError, match="only for external"):
        UncertaintyDataProvenance(test_id, UncertaintyDataRole(role))


def test_training_audit_observes_exact_training_rows_and_bootstrap_draws(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from tcspc_toolkit.uncertainty.evaluation import make_uncertainty_development_split

    development = SimpleNamespace(X_features=pd.DataFrame(np.arange(128).reshape(64, 2)), y=np.arange(64.))
    split = make_uncertainty_development_split(64)
    X = development.X_features.to_numpy(dtype=float)[list(split.training_indices)]
    y = development.y[list(split.training_indices)]
    monkeypatch.setattr(Pipeline, "fit", lambda self, *args, **kwargs: self)
    with acceptance.audit_week9_training(development, split) as calls:
        for model in [HistGradientBoostingRegressor() for _ in range(3)] + [RandomForestRegressor(), Ridge()]:
            Pipeline([("model", model)]).fit(X, y)
        rng = np.random.default_rng(42)
        for _ in range(200):
            indices = rng.integers(0, 48, 48)
            Pipeline([("model", Ridge())]).fit(X[indices], y[indices])
    assert len(calls) == 205
    with pytest.raises(AssertionError):
        with acceptance.audit_week9_training(development, split):
            Pipeline([("model", Ridge())]).fit(X + 1, y)


def test_calibration_external_guard_rejects_refitting():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    with acceptance.frozen_week9_models():
        with pytest.raises(acceptance.AcceptanceFailure, match="attempted model/representation fitting"):
            Pipeline([("model", Ridge())]).fit([[1.]], [1.])


@pytest.mark.parametrize("interval", [False, True])
def test_uncertainty_population_preserves_failed_ids(interval):
    import numpy as np
    from tcspc_toolkit.uncertainty.evaluation import PredictionIntervalResult, UncertaintyScoreResult
    if interval:
        source = PredictionIntervalResult(np.array([1., 2.]), np.array([.5, 3.]), np.array([2., 2.]), .9, "test")
    else:
        source = UncertaintyScoreResult(np.array([1., 2.]), np.array([.1, np.nan]), "test")
    population = acceptance.uncertainty_population(source, [11, 12])
    assert population["n_attempted"] == 2
    assert population["n_valid"] == population["n_failed"] == 1
    assert population["failure_rate"] == .5
    assert population["failures"] == [{"sample_id": 12, "reason": "invalid_interval" if interval else "invalid_score"}]
