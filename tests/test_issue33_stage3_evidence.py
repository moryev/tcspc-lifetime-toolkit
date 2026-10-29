"""Read-only consistency of published evidence; no fitting or qualification draws."""

from collections import Counter
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qualify_biexponential_reconvolution import digest, source_digest
from qualify_issue33_stage3 import observation_key, task_key

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "docs/validation" / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def evidence():
    return {
        name: read(f"issue33_stage3_{name}_v1.json")
        for name in ("execution_lock", "results", "diagnostics", "integrity")
    }


def test_frozen_source_and_execution_identity(evidence):
    lock = evidence["execution_lock"]
    expected = "7abb3188578458b57f9372572a8d5f0e9567c30864c1bf9cfbc3460f554be73a"
    assert digest(lock) == expected
    for name, value in lock["source_lf_sha256"].items():
        assert source_digest(ROOT / name) == value, name
    for name in ("results", "diagnostics", "integrity"):
        assert evidence[name]["execution_lock_sha256"] == expected
    assert lock["protocol"]["root_entropy"] == [33, 3, 20261029, 1]
    assert lock["protocol"]["fit_settings"]["max_starts"] == 46
    # The original denied record is review history; approval is a separate record.
    assert not read("issue33_stage3_execution_authorization_v1.json")[
        "scientific_execution_authorized"
    ]
    approval = read("issue33_stage3_execution_approval_2026-10-29.json")
    assert approval["scientific_execution_authorized"]
    assert approval["lock_sha256"] == expected


def test_complete_identity_inventory_and_terminal_accounting(evidence):
    summary = evidence["results"]
    tasks = evidence["execution_lock"]["populations"]["qualification"]
    expected = {task_key(t) for t in tasks}
    actual = [task_key(r["task"]) for r in summary["accounting"]]
    assert len(actual) == len(set(actual)) == len(expected) == 543
    assert set(actual) == expected
    assert (
        summary["scheduled_counts"]
        == summary["terminal_counts"]
        == dict(observation=200, production=210, audit=133)
    )
    assert summary["statuses"] == {"completed": 543}
    assert summary["all_scheduled_tasks_terminal"]
    inventory = summary["evidence_inventory"]
    assert len(inventory) == 2173
    for key in expected:
        for prefix in ("", "started-", "transport-", "timing-"):
            assert prefix + key + ".json" in inventory
    assert digest(inventory) == summary["evidence_sha256"]
    for name in ("diagnostics", "integrity"):
        assert evidence[name]["evidence_sha256"] == summary["evidence_sha256"]


def test_primary_successes_do_not_erase_failed_strict_audit_gate(evidence):
    summary = evidence["results"]
    gates = summary["primary_gates"]
    assert gates["numerical_acceptance"]
    assert gates["median_absolute_relative_errors"] == [True, True]
    assert gates["absolute_mean_signed_relative_biases"] == [True, True]
    assert gates["complete_untampered_population"]
    assert not gates["complete_usable_audit_agreement_100_of_100"]
    assert not gates["overall_pass"]
    primary = next(r for r in summary["recovery"] if r["condition_id"] == 0)
    assert primary["n_scheduled"] == primary["n_attempted"] == 100
    assert primary["n_accepted"] == 99
    assert [c["n_contributing"] for c in primary["components"]] == [99, 99]
    assert [c["n_unavailable_among_accepted"] for c in primary["components"]] == [0, 0]


def test_all_audits_complete_with_disagreement_direction_preserved(evidence):
    summary = evidence["results"]
    assert summary["complete_audits"] == 133
    assert summary["unavailable_audit_comparisons"] == 0
    assert summary["substantive_audit_disagreements"] == 4
    disagreements = {}
    for row in summary["audit_comparisons"]:
        assert row["audit_complete"] and row["comparison_available"]
        if row["classification"] == "disagreement":
            t = row["task"]
            disagreements[t["condition_id"], t["replicate"]] = row[
                "production_minus_audit_centered_nll"
            ]
    assert disagreements == pytest.approx(
        {
            (0, 50): 8.503872612156272,
            (5, 4): 68.67913048378404,
            (5, 9): 42.07243220686797,
            (10, 4): -0.4644208861714674,
        }
    )
    assert summary["audit_tally"]["populations"]["primary"] == dict(
        agreement=99,
        disagreement=1,
        audit_incomplete=0,
        production_unavailable=0,
        not_run=0,
    )


def test_required_recovery_failures_remain_explicit(evidence):
    rejected = evidence["diagnostics"]["numerically_rejected_fits"]
    assert {(r["task"]["condition_id"], r["task"]["replicate"]) for r in rejected} == {
        (0, 50),
        (5, 1),
        (5, 4),
        (5, 9),
    }
    for row in rejected:
        o = row["optimization"]
        assert not row["assessment"]["accepted"]
        assert o["numerical_validation_passed"] is False
        assert o["max_detectable_descent_nll"] > 0.01
        assert o["unattempted_seed_candidates"] == 0
        assert o["exhausted_budgets"] == ["starts"]
        assert o["recovery_unavailable_reason"] == "start_budget_exhausted"
        assert o["stopping_reason"] == "recovery_start_budget_exhausted"
    r050 = next(r for r in rejected if r["task"]["condition_id"] == 0)
    assert r050["assessment"]["recomputed_centered_nll"] == pytest.approx(245.763159)
    assert r050["optimization"]["finite_seed_candidates"] == 46
    assert sum(r050["start_messages"].values()) == 46
    assert r050["optimization"]["objective_evaluations"] == 9266


def test_seed_recovery_optional_omission_and_call_totals(evidence):
    group = evidence["diagnostics"]["all_production"]
    assert group["numerical_accepted"] == group["local_validation_passes"] == 206
    assert group["start_kinds"] == {"seed": 9521, "validation_recovery": 6}
    assert group["starts"]["total"] == 9527
    assert group["starts"]["maximum"] == 46
    assert group["unattempted_required_seeds"]["total"] == 0
    assert group["optional_continuations_omitted"]["total"] == 6504
    assert group["fits_with_optional_omission"] == 203
    assert group["exhausted_budgets"] == {"starts": 4}
    calls = Counter()
    for record in evidence["results"]["accounting"]:
        calls.update(record["calls"])
    assert dict(calls) == evidence["results"]["known_forward_calls_by_stage"]
    assert sum(calls.values()) == 2774514
    assert calls["external_assessment"] == 210
    assert calls["shared_audit_recheck"] == 133
    assert evidence["results"]["terminal_tasks_with_unknown_call_counts"] == 0


def test_c26_pairing_and_warning_free_mismatch_remain_visible(evidence):
    tasks = evidence["execution_lock"]["populations"]["qualification"]
    pairs = evidence["diagnostics"]["c26_paired_observations"]
    assert len(pairs) == len({r["observation_sha256"] for r in pairs}) == 10
    for pair in pairs:
        arms = [
            t
            for t in tasks
            if t["kind"] == "production"
            and t["condition_id"] == 26
            and t["replicate"] == pair["replicate"]
        ]
        assert len(arms) == 2 and len({observation_key(t) for t in arms}) == 1
        assert pair["same_observation_identity"]
        assert all(pair["numerical_acceptance"].values())
    mismatched = next(
        r
        for r in evidence["diagnostics"]["condition_arms"]
        if r["arm"] == "gaussian_0.3"
    )
    assert mismatched["numerical_accepted"] == mismatched["warning_free_fits"] == 10
    recovery = next(
        r for r in evidence["results"]["recovery"] if r["arm"] == "gaussian_0.3"
    )
    assert recovery["components"][0]["median_absolute_relative_error"] > 0.39


def test_stage1_and_stage2b_regression_evidence_is_preserved():
    prior = read("issue33_stage2b_results_2026-10-28.json")
    assert prior["summary"]["all"]["cases"] == 14
    assert prior["summary"]["all"]["full_numerical_acceptance"] == 14
    for name, sha in prior["integrity"]["stage1_files_lf_sha256"].items():
        assert source_digest(ROOT / name.replace("\\", "/")) == sha
    assert prior["summary"]["C00"]["maximum_absolute_reference_gap"] <= 0.01
    assert prior["integrity"][
        "all_regression_seed_banks_match_historical_initial_parameters_and_order"
    ]
