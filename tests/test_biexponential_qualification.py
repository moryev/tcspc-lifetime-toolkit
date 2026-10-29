"""Protocol, physical generation, audit independence and resume safeguards.

All sampled observations here are stream 2; no scientific qualification draws.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.special import gammaln

from tcspc_toolkit.benchmarks import multiexponential as study
from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitSettings,
    fit_biexponential_reconvolution,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return json.loads((ROOT / "configs/issue28_pr3_qualification.json").read_text())


@pytest.fixture
def runner():
    return runpy.run_path(str(ROOT / "scripts/qualify_biexponential_reconvolution.py"))


def test_corrected_totals_and_frozen_policy(protocol):
    assert study.validate_protocol(protocol) == dict(
        conditions=30,
        observations=1550,
        comparisons=1850,
        production_fits=3700,
        scheduled_audits=205,
    )
    assert protocol["fit_settings"] == asdict(BiexponentialFitSettings())
    assert sum(len(c["arms"]) - 1 for c in protocol["conditions"]) == 6
    assert protocol["conditions"][0]["replicates"] == 100
    modified = deepcopy(protocol)
    modified["conditions"][25]["arms"].append("extra")
    with pytest.raises(ValueError, match="totals"):
        study.validate_protocol(modified)


@pytest.mark.parametrize("stream", [0, 1])
def test_scientific_streams_blocked_before_any_rng_construction(
    protocol, stream, monkeypatch
):
    protocol["execution_status"] = "awaiting_prequalification_review"

    def forbidden(*args):
        pytest.fail("scientific RNG was constructed before approval")

    monkeypatch.setattr(np.random, "PCG64", forbidden)
    with pytest.raises(PermissionError, match="approval"):
        study.observation_rng(protocol, stream, 0, 0)


def test_repeatable_independent_nonqualification_stream(protocol):
    case = study.build_generating_case(protocol, 3)
    one = study.sample_case(protocol, case, stream=2, replicate=0)
    two = study.sample_case(protocol, case, stream=2, replicate=0)
    other = study.sample_case(protocol, case, stream=2, replicate=1)
    np.testing.assert_array_equal(one.values, two.values)
    assert not np.array_equal(one.values, other.values)
    assert one.provenance["seed_entropy"] == [28, 3, 20261025, 1, 2, 3, 0]


@pytest.mark.parametrize("condition_id", range(30))
def test_declared_generating_conditions_against_direct_convolution(
    protocol, condition_id
):
    case = study.build_generating_case(protocol, condition_id)
    c = case.configuration
    time = case.irf.time_ns
    shifted = np.interp(time - c["shift_ns"], time, case.irf.kernel, left=0, right=0)
    direct = np.array(
        [
            a
            * np.convolve(np.exp(-time / tau), shifted)[: time.size]
            * (time[1] - time[0])
            for a, tau in zip(case.amplitudes, c["lifetimes_ns"], strict=True)
        ]
    )
    np.testing.assert_allclose(case.component_counts, direct, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        case.expected_counts,
        direct.sum(axis=0) + c["background"],
        rtol=1e-12,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        direct.sum(axis=1),
        np.array(c["detected_fractions"]) * c["signal_photons"],
        rtol=1e-12,
        atol=1e-10,
    )
    assert case.expected_counts.size == (103 if condition_id == 21 else 512)
    if condition_id == 27:
        assert case.irf.source.provenance["experimentally_measured"] is False
        assert case.irf.diagnostics.resampling_method == "linear"


def test_proxy_is_fixed_independent_and_retains_diagnostics(protocol):
    prepared, diagnostics, calibration = study.build_calibration_proxy(
        protocol, stream=2
    )
    assert calibration.provenance["seed_entropy"][-3:] == [2, 28, 1000000]
    assert diagnostics.status == "proxy_constructed"
    assert prepared.source.source_kind.value == "leading_edge_estimate"
    assert prepared.source.provenance
    assert prepared.diagnostics.resampling_method == "none"
    case = study.build_generating_case(protocol, 28)
    evaluation = study.sample_case(protocol, case, stream=2, replicate=0)
    assert not np.array_equal(calibration.values, evaluation.values)


@pytest.fixture
def zero_fit(protocol):
    case = study.build_generating_case(protocol, 0)
    result = fit_biexponential_reconvolution(
        case.irf.time_ns,
        np.zeros(512, dtype=int),
        case.irf,
        lifetime_bounds_ns=(0.1, 8),
        temporal_shift_bounds_ns=(-0.3, 0.3),
    )
    return case, result


def test_diagnostic_budget_exhaustion_is_not_numerical_failure(protocol, zero_fit):
    case, result = zero_fit
    result = replace(
        result,
        optimization=replace(result.optimization, exhausted_budgets=("diagnostics",)),
    )
    assessment = study.assess_biexponential_fit(
        result, case.irf.time_ns, np.zeros(512), case.irf.kernel, protocol["acceptance"]
    )
    assert assessment.accepted
    assert not assessment.diagnostic_budget_complete
    assert assessment.external_forward_evaluations == 1
    assert not assessment.jacobian_available
    assert assessment.diagnostic_unavailable_reason


@pytest.mark.parametrize(
    "budget", ["initialization", "optimization", "validation", "starts"]
)
def test_required_stage_budgets_do_invalidate(protocol, zero_fit, budget):
    case, result = zero_fit
    result = replace(
        result, optimization=replace(result.optimization, exhausted_budgets=(budget,))
    )
    a = study.assess_biexponential_fit(
        result, case.irf.time_ns, np.zeros(512), case.irf.kernel, protocol["acceptance"]
    )
    assert not a.accepted
    assert "required_stage_budget_exhausted" in a.reasons


def test_misleading_success_and_cached_objective_do_not_pass(protocol, zero_fit):
    case, result = zero_fit
    bad = replace(
        result,
        optimization=replace(
            result.optimization,
            optimizer_reported_success=True,
            numerical_validation_passed=False,
        ),
        statistics=replace(result.statistics, centered_negative_log_likelihood=100),
    )
    a = study.assess_biexponential_fit(
        bad, case.irf.time_ns, np.zeros(512), case.irf.kernel, protocol["acceptance"]
    )
    assert not a.accepted
    assert a.objective_difference == 100
    assert "local_validation_failed_or_unavailable" in a.reasons
    assert "objective_recheck_failed" in a.reasons


def test_accepted_vanishing_components_are_unavailable_not_filtered_fits(zero_fit):
    case, result = zero_fit
    recovery = study.component_recovery(case, result, True)
    assert recovery["truth_components_available"]
    assert recovery["fitted_component_available"] == [False, False]
    assert recovery["accepted_component_relative_errors"] == [None, None]


def test_denominators_include_failures_and_unavailable_components(protocol):
    errors = [[0.01, 0.02], [0.03, None], [None, None], [None, None]]
    rows = [
        dict(
            assessment=dict(accepted=i < 3, diagnostic_budget_complete=i != 1),
            recovery=dict(accepted_component_relative_errors=e),
        )
        for i, e in enumerate(errors)
    ]
    protocol["acceptance"]["bootstrap_replicates"] = 20
    result = study.summarize_recovery(rows, protocol, 0, 0)
    assert result["n_attempted"] == 4
    assert result["n_accepted"] == 3
    assert [c["n_contributing"] for c in result["components"]] == [2, 1]
    assert [c["n_unavailable_among_accepted"] for c in result["components"]] == [1, 2]
    assert result["joint_accepted_both_within_5_percent_fraction"] == 0.25
    assert not result["baseline_targets_met"]


def test_independent_audit_recovers_deterministic_curve_without_shared_model(
    protocol, monkeypatch
):
    time = np.linspace(0, 20, 128)
    kernel = np.exp(-0.5 * ((time - 1) / 0.18) ** 2)
    kernel /= np.trapezoid(kernel, x=time)
    shifted = np.interp(time - 0.075, time, kernel, left=0, right=0)
    y = 2 + sum(
        a * np.convolve(np.exp(-time / tau), shifted)[: time.size] * (time[1] - time[0])
        for a, tau in [(6000, 0.8), (2000, 3.2)]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("independent solver used production forward model")

    monkeypatch.setattr(
        study, "multiexponential_reconvolution_component_counts", forbidden
    )
    audit = study.independent_biexponential_audit(
        time, y, kernel, (0.1, 8), (-0.3, 0.3), protocol["audit"]
    )
    assert audit["complete"]
    assert audit["centered_nll"] < 1e-6
    assert audit["forward_calls"] <= 100000
    np.testing.assert_allclose(
        audit["parameters"]["lifetimes_ns"], [0.8, 3.2], rtol=1e-4
    )
    # Full likelihood at the recovered curve has the same centered-loss offset.
    p = audit["parameters"]
    shift = np.interp(time - p["temporal_shift_ns"], time, kernel, left=0, right=0)
    fitted = p["background"] + sum(
        a * np.convolve(np.exp(-time / t), shift)[: time.size] * (time[1] - time[0])
        for a, t in zip(p["amplitudes"], p["lifetimes_ns"], strict=True)
    )
    ll = np.sum(y * np.log(fitted) - fitted - gammaln(y + 1))
    saturated = np.sum(y * np.log(y) - y - gammaln(y + 1))
    assert saturated - ll == pytest.approx(audit["centered_nll"], abs=1e-8)


def test_audit_hard_limit_counts_derivative_predictions(protocol):
    time = np.linspace(0, 10, 32)
    kernel = np.ones(32) / 10
    settings = {**protocol["audit"], "max_forward_calls": 7}
    result = study.independent_biexponential_audit(
        time, np.ones(32), kernel, (0.1, 8), (-0.3, 0.3), settings
    )
    assert result["forward_calls"] == 7
    assert result["budget_exhausted"]
    assert not result["complete"]
    assert result["parameters"] is not None


def test_checkpoint_resume_tamper_lock_and_completed_failure(tmp_path, runner):
    checkpoint = runner["Checkpoints"]
    lock = {"source": "abc", "settings": {"starts": 46}}
    store = checkpoint(tmp_path, lock)
    failed = {"status": "timeout", "limit_seconds": 300}
    store.put("case", failed)
    resumed = checkpoint(tmp_path, deepcopy(lock))
    assert resumed.get("case") == failed
    resumed.put("case", failed)
    with pytest.raises(ValueError, match="overwrite"):
        resumed.put("case", {"status": "completed"})
    with pytest.raises(ValueError, match="lock differs"):
        checkpoint(tmp_path, {"source": "changed"})
    path = tmp_path / "case.json"
    data = json.loads(path.read_text())
    data["payload"]["status"] = "completed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupt"):
        resumed.get("case")


def test_observation_saved_before_fit_and_not_redrawn(
    tmp_path, runner, protocol, monkeypatch
):
    store = runner["Checkpoints"](tmp_path, {"test": "nonqualification"})
    key, original = runner["observation_checkpoint"](store, protocol, 0, 0, 2)
    assert original["provenance"]["stream"] == 2

    def forbidden(*args, **kwargs):
        pytest.fail("resume attempted a fresh draw")

    monkeypatch.setitem(
        runner["observation_checkpoint"].__globals__, "sample_case", forbidden
    )
    other_key, resumed = runner["observation_checkpoint"](store, protocol, 0, 0, 2)
    assert other_key == key
    assert resumed == original
    assert not list(tmp_path.glob("production*.json"))


def test_study_gate_precedes_proxy_or_observation_generation(
    tmp_path, runner, protocol
):
    protocol["execution_status"] = "awaiting_prequalification_review"
    store = runner["Checkpoints"](tmp_path, {"test": True})
    with pytest.raises(PermissionError, match="STOP"):
        runner["execute_study"](store, protocol)
    assert [p.name for p in tmp_path.iterdir()] == ["lock.json"]


def test_proxy_checkpoint_roundtrip_and_source_lock(tmp_path, runner, protocol):
    # Issue 33 must refuse to resume the historical qualification with a changed
    # fitter. Exercise checkpoint transport using the saved lock, never refresh
    # the old source hash to match development code.
    with pytest.raises(ValueError, match="PR-2 source lock changed"):
        runner["source_lock"](protocol)
    lock = json.loads(
        (
            ROOT / "docs/validation/issue28_pr3_execution_lock_2026-10-26.json"
        ).read_text()
    )["lock"]
    assert (
        lock["sources_lf_sha256"]["src/tcspc_toolkit/estimation/multiexponential.py"]
        == protocol["frozen_pr2_lf_sha256"]
    )
    store = runner["Checkpoints"](tmp_path, lock)
    record = runner["proxy_checkpoint"](store, protocol, 2)
    case = study.build_generating_case(protocol, 28)
    restored = runner["resolve_arm"](protocol, case, "calibration_proxy", 2, record)
    np.testing.assert_array_equal(restored.kernel, record["kernel"])
    assert restored.source.provenance == record["source"]["provenance"]


def test_incomplete_study_cannot_publish_filtered_denominators(
    tmp_path, runner, protocol
):
    store = runner["Checkpoints"](tmp_path, {"test": True})
    with pytest.raises(ValueError, match="incomplete study"):
        runner["summarize_study"](store, protocol)


def test_timeout_checkpoint_is_not_retried(tmp_path, runner, protocol, monkeypatch):
    import subprocess

    store = runner["Checkpoints"](tmp_path, {"test": True})
    task = dict(
        kind="audit", stream=2, condition_id=0, replicate=0, arm_index=0, arm="matched"
    )
    calls = []

    def timeout(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    first = runner["execute_task"](store, protocol, task)
    second = runner["execute_task"](store, protocol, task)
    assert first == second
    assert first["status"] == "timeout"
    assert calls == [300]
    assert store.get("runtime-audit-s2-c00-r000-a0")["wall_seconds"] >= 0


def test_programming_error_stops_and_remains_recorded(
    tmp_path, runner, protocol, monkeypatch
):
    import subprocess
    from types import SimpleNamespace

    store = runner["Checkpoints"](tmp_path, {"test": True})
    task = dict(
        kind="audit", stream=2, condition_id=0, replicate=0, arm_index=0, arm="matched"
    )
    calls = []

    def failed(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(returncode=1, stdout="", stderr="programming defect")

    monkeypatch.setattr(subprocess, "run", failed)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            runner["execute_task"](store, protocol, task)
    assert len(calls) == 1
    assert store.get("audit-s2-c00-r000-a0")["stderr"] == "programming defect"


def test_projection_dispatch_preserves_existing_projector_contract(
    protocol, runner, monkeypatch
):
    from tcspc_toolkit.benchmarks import bayesian_mismatch

    calls = []

    def projector(condition, assumption, expected, bounds):
        calls.append((condition, assumption, expected, bounds))
        return {"test": "dispatch only"}

    monkeypatch.setattr(bayesian_mismatch, "construct_pseudo_true_reference", projector)
    for condition_id, arm in [(0, "matched"), (29, "matched"), (29, "gaussian_0.3")]:
        case = study.build_generating_case(protocol, condition_id)
        result = runner["projection_task"](protocol, case, arm)
        assert result["status"] == "completed"
        assert calls[-1][1].assumption_id == arm
        np.testing.assert_array_equal(calls[-1][2], case.expected_counts)
        assert calls[-1][3] == (-0.3, 0.3)
    unsupported = study.build_generating_case(protocol, 8)
    assert (
        runner["projection_task"](protocol, unsupported, "matched")["status"]
        == "unavailable"
    )
    assert len(calls) == 3


def test_executable_example_outside_checkout(tmp_path):
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples/biexponential_reconvolution.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Bi minus mono AIC" in completed.stdout
    assert "no experimental truth" in completed.stdout
    assert "do not establish physical correctness" in completed.stdout
    assert not list(tmp_path.iterdir())


def test_compact_reporting_preserves_failure_and_unavailable_denominators():
    reporting = runpy.run_path(
        str(ROOT / "scripts/report_biexponential_qualification.py")
    )
    completed = dict(
        status="completed",
        recovery={"all_finite_component_relative_errors": [0.1, 0.01]},
        warning_free_error_above_5_percent=True,
        identifiability={},
        optimization={"exhausted_budgets": [], "objective_evaluations": 100},
        mono={"valid_fit": True},
        assessment={
            "accepted": True,
            "jacobian_available": True,
            "diagnostic_unavailable_reason": None,
        },
        curve_metrics={"total_relative_l2": 0.02},
        statistics={"delta_aic": -10, "delta_bic_bins": 2, "delta_deviance": -14},
        production_seconds=1,
    )
    undefined = deepcopy(completed)
    undefined["recovery"]["all_finite_component_relative_errors"] = [None, None]
    undefined["warning_free_error_above_5_percent"] = False
    undefined["curve_metrics"] = None
    undefined["assessment"] = {
        "jacobian_available": False,
        "diagnostic_unavailable_reason": "structural degeneracy",
    }
    report = reporting["condition_details"](
        [completed, undefined, {"status": "timeout"}]
    )
    assert report["execution_status_counts"] == {"completed": 2, "timeout": 1}
    assert report["n_warning_free_completed"] == 2
    assert report["n_warning_free_with_component_errors"] == 1
    assert report["n_warning_free_any_component_error_above_5_percent"] == 1
    assert report["n_accepted_warning_free_with_component_errors"] == 1
    assert report["n_accepted_warning_free_any_component_error_above_5_percent"] == 1
    assert report["curve_and_nuisance"]["total_relative_l2"]["n"] == 1
    assert report["n_jacobian_available"] == 1
    assert report["jacobian_unavailable_reasons"] == {"structural degeneracy": 1}


def test_compact_attempt_does_not_fabricate_timeout_parameters(protocol):
    reporting = runpy.run_path(
        str(ROOT / "scripts/report_biexponential_qualification.py")
    )
    row = reporting["compact_attempt"](
        protocol["conditions"][0], 0, {"status": "timeout"}, "hash"
    )
    assert row["accepted"] is False
    assert row["status"] == "timeout"
    assert row["bi_parameters_lifetimes_ns_1"] is None
    assert row["bi_parameters_amplitudes_2"] is None
    assert row["recovery_accepted_component_relative_errors_1"] is None


def test_audit_reporting_retains_incomplete_and_failed_target_attempts(protocol):
    reporting = runpy.run_path(
        str(ROOT / "scripts/report_biexponential_qualification.py")
    )
    summaries = [
        dict(
            condition_id=0,
            arm="matched",
            scheduled_audits=[
                dict(complete=True, absolute_nll_difference=0.001),
                dict(
                    complete=False,
                    absolute_nll_difference=0.0,
                    exhausted_budgets=["start_nfev"],
                ),
                dict(status="timeout"),
                dict(complete=True, absolute_nll_difference=2.0),
            ],
        ),
        dict(
            condition_id=7,
            arm="matched",
            scheduled_audits=[dict(complete=True, absolute_nll_difference=5.0)],
        ),
    ]
    summary = reporting["audit_summary"](summaries, protocol["audit"])
    assert summary["n_scheduled"] == 5
    assert summary["n_targeted"] == 4
    assert summary["n_targeted_complete_and_agree"] == 1
    assert summary["n_targeted_best_objective_agrees"] == 2
    assert summary["absolute_gap"]["n"] == 4
    assert summary["exhausted_budget_counts"] == {"start_nfev": 1}
    assert summary["target_verdict"] == "FAIL"


def test_completed_evidence_accounts_for_every_arm_and_preserves_failures(protocol):
    directory = ROOT / "docs/validation"
    report = json.loads((directory / "issue28_pr3_results_2026-10-27.json").read_text())
    with (directory / "issue28_pr3_results_2026-10-27.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(int(r["condition_id"]), r["arm"], int(r["replicate"])) for r in rows]
    expected = {
        (c["id"], arm, replicate)
        for c in protocol["conditions"]
        for arm in c["arms"]
        for replicate in range(c["replicates"])
    }
    assert len(keys) == len(set(keys)) == 1850
    assert set(keys) == expected
    assert sum(r["accepted"] == "True" for r in rows) == 1844
    for condition in report["conditions"]:
        subset = [
            r
            for r in rows
            if int(r["condition_id"]) == condition["condition_id"]
            and r["arm"] == condition["arm"]
        ]
        assert len(subset) == condition["n_attempted"]
        assert sum(r["accepted"] == "True" for r in subset) == condition["n_accepted"]
    baseline = report["conditions"][0]
    assert baseline["n_accepted"] == 98 and baseline["baseline_targets_met"]
    assert report["audits"]["n_targeted"] == 121
    assert report["audits"]["n_targeted_complete_and_agree"] == 119
    assert report["audits"]["target_verdict"] == "FAIL"
    failures = {
        row["replicate"]: row
        for row in report["numerical_failures"]
        if row["condition_id"] == 0
    }
    assert set(failures) == {11, 58}
    assert failures[11]["start_kinds"] == {"seed": 45, "validation_recovery": 1}
    assert failures[58]["start_kinds"] == {"seed": 46}
    assert failures[58]["successful_starts_with_nonfinite_endpoint"] == [23]
    assert all(not r["assessment"]["accepted"] for r in failures.values())


def test_completed_evidence_preserves_historical_identity():
    directory = ROOT / "docs/validation"
    report = json.loads((directory / "issue28_pr3_results_2026-10-27.json").read_text())
    execution = json.loads(
        (directory / "issue28_pr3_execution_lock_2026-10-26.json").read_text()
    )
    assert report["execution_lock"] == execution["lock"]
    assert len(execution["lock"]["sources_lf_sha256"]) == 66
    assert not any(
        k.startswith("docs/") for k in execution["lock"]["sources_lf_sha256"]
    )
    csv_bytes = (directory / "issue28_pr3_results_2026-10-27.csv").read_bytes()
    # Git may change checkout line endings; the generated CSV uses CRLF.
    csv_bytes = csv_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert (
        hashlib.sha256(csv_bytes).hexdigest()
        == report["integrity"]["attempt_csv_sha256"]
    )
    assert report["integrity"]["sorted_checkpoint_inventory_sha256"] == (
        "d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14"
    )
