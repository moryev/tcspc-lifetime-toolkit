"""Closed protocol routing and tiny evidence fixtures, with no V00-V02 draws."""

import copy
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import pytest

from tcspc_toolkit.benchmarks import _triexponential_verification as verification
from tcspc_toolkit.benchmarks import triexponential as shared
from test_triexponential_verification import (
    complete,
    fixture_input,
    fixture_store,
    forbid,
)
from test_triexponential_integrity import software_lock, tiny_tasks

ROOT = Path(__file__).resolve().parents[1]
V1, V2 = tuple(verification.PROTOCOLS)


def manifest(scope):
    return json.loads((ROOT / verification.PROTOCOLS[scope].manifest_path).read_text())


@pytest.mark.parametrize("scope", [V1, V2])
def test_exact_protocol_metadata_and_inventory(scope, monkeypatch):
    for name in ("construct_problem", "_fit_triexponential", "run_audit"):
        monkeypatch.setattr(shared, name, forbid)
    m = manifest(scope)
    protocol = verification.validate_manifest(m)
    assert protocol is verification.PROTOCOLS[scope]
    tasks = shared.task_inventory(m, verification=True)
    spec = shared.execution_specification(m, verification=True)
    assert tuple(t["id"] for t in tasks) == protocol.task_ids
    assert len(tasks) == 12
    assert len({t["problem"]["id"] for t in tasks}) == 6
    assert not any(t["configuration"] == "audit" for t in tasks)
    assert spec["output_directory"] == protocol.output_directory
    assert spec["settings"] == manifest(V1)["configurations"]
    assert spec["worker_deadlines"] == dict.fromkeys(m["configurations"], 60)
    for i, p in enumerate(shared.problem_inventory(m, verification=True)):
        assert p["seed"] == (
            None
            if i % 2 == 0
            else [37, 2, 20261101, 2 if scope == V1 else 3, i // 2, 0]
        )
    with pytest.raises(ValueError, match="explicit verification"):
        shared.task_inventory(m)
    with pytest.raises(ValueError):
        shared.task_inventory(m, pilot=True, verification=True)


@pytest.mark.parametrize("scope", [V1, V2])
@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "scope",
        "schema",
        "conditions",
        "settings",
        "budget",
        "namespace",
        "audit",
        "task_identity",
        "output",
        "authorization",
    ],
)
def test_closed_protocol_rejects_cross_version_and_modified_contract(scope, change):
    m = manifest(scope)
    other = manifest(V2 if scope == V1 else V1)
    if change == "unknown":
        m["study_id"] = "issue37_verification_v999"
    elif change in ("scope", "schema"):
        key = "study_id" if change == "scope" else "schema"
        m[key] = other[key]
    elif change == "conditions":
        m["conditions"][0]["amplitudes"][0] += 1
    elif change == "settings":
        m["configurations"]["trf:compact"]["descent_tolerance"] = 1
    elif change == "budget":
        m["configurations"]["trf:compact"]["max_optimization_calls"] += 1
    elif change == "namespace":
        m["problems"][1]["rng_entropy"][3] = 1
    elif change == "audit":
        m["tasks"][0]["configuration"] = "audit"
    elif change == "task_identity":
        m["tasks"][0]["execution_task_identity"] = other["tasks"][0][
            "execution_task_identity"
        ]
    elif change == "output":
        m["outputs"]["directory"] = other["outputs"]["directory"]
    else:
        m["execution_authorized"] = True
    with pytest.raises(ValueError, match="exact reviewed"):
        verification.validate_manifest(m)


def test_v2_science_and_descriptive_baseline():
    v1, v2 = manifest(V1), manifest(V2)
    contract = {k: v2[k] for k in verification.CONTRACT_KEYS}
    assert (
        shared.digest(contract)
        == "27c5238c32a9aa388f614dcc2c8a6ef2813baad985e9db3de681a031a4ddeb17"
    )
    for key in ("configurations", "irfs", "numerical_acceptance", "resources"):
        assert v1[key] == v2[key]
    for old, new in zip(v1["conditions"], v2["conditions"], strict=True):
        assert {k: v for k, v in old.items() if not k.endswith("_kernel_id")} == {
            k: v for k, v in new.items() if not k.endswith("_kernel_id")
        }
    baseline = v2["previous_verification_baseline"]
    assert (
        sum(
            f["accepted"]
            for p in baseline["problems"].values()
            for f in p["fits"].values()
        )
        == 8
    )
    assert baseline["reference"] == v2["previous_verification_reference"]
    assert v2["execution_authorized"] is False
    assert v2["execution_commit"] is None
    # Source distributions do not depend on untracked historical planning files.
    proposal = (
        ROOT
        / "docs/validation/issue37_pr2c1b_numerical_verification_v2_manifest_proposal.json"
    )
    if proposal.exists():
        assert (
            hashlib.sha256(proposal.read_bytes()).hexdigest()
            == "3256669aaccdf8f3b7391f12c08ba7b20f67f500867184c26eb3e674b9a3c518"
        )
        original = json.loads(proposal.read_text())
        assert contract == {k: original[k] for k in verification.CONTRACT_KEYS}


@pytest.mark.parametrize("scope", [V1, V2])
def test_inspect_and_import_are_metadata_only(scope, monkeypatch, capsys):
    for name in ("construct_problem", "_fit_triexponential", "run_audit"):
        monkeypatch.setattr(shared, name, forbid)
    monkeypatch.setattr(np.random, "default_rng", forbid)
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["main"].__globals__
    monkeypatch.setitem(g, "execution_environment", lambda: {"fixture": True})
    monkeypatch.setitem(
        g, "repository_identity", lambda _: dict(committed_sources=False)
    )
    monkeypatch.setitem(g, "DevelopmentStore", forbid)
    script["main"](
        [
            "inspect",
            "--verification",
            "--manifest",
            str(ROOT / verification.PROTOCOLS[scope].manifest_path),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["tasks"] == 12 and out["problems"] == 6
    assert out["fingerprint"]["scope"] == scope
    assert out["execution_authorized"] is False
    assert out["fingerprint"]["repository"]["committed_sources"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "pending",
        "source",
        "environment",
        "manifest",
        "scope",
        "settings",
        "uncommitted",
        "missing_review",
    ],
)
def test_v2_requires_exact_committed_authorized_fingerprint(
    tmp_path, monkeypatch, damage
):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["fingerprint"].__globals__
    m = manifest(V2)
    monkeypatch.setitem(
        g,
        "repository_identity",
        lambda _: dict(commit="synthetic", committed_sources=damage != "uncommitted"),
    )
    monkeypatch.setitem(
        g,
        "execution_environment",
        lambda: dict(
            environment=m["environment_controls"],
            numerical_libraries=[dict(num_threads=1)],
        ),
    )
    path = ROOT / verification.PROTOCOLS[V2].manifest_path
    fp = script["fingerprint"](m, False, path, verification=True)
    lock = dict(
        schema=shared.EXECUTION_SCHEMA,
        execution_authorized=True,
        maintainer_review="synthetic",
        fingerprint=copy.deepcopy(fp),
    )
    if damage == "pending":
        lock["execution_authorized"] = False
    elif damage == "missing_review":
        lock["maintainer_review"] = None
    elif damage == "source":
        lock["fingerprint"]["source_lf_sha256"]["scripts/develop_triexponential.py"] = (
            "wrong"
        )
    elif damage == "manifest":
        lock["fingerprint"]["manifest_file_sha256"] = "wrong"
    elif damage == "scope":
        lock["fingerprint"]["scope"] = V1
    elif damage == "settings":
        lock["fingerprint"]["execution"]["settings"]["trf:compact"][
            "max_seed_starts"
        ] = 1
    elif damage == "environment":
        lock["fingerprint"]["environment"]["environment"]["OMP_NUM_THREADS"] = "2"
    pending = tmp_path / "lock.json"
    pending.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="authorized matching"):
        script["require_authorization"](pending, m, False, path, verification=True)


@pytest.mark.parametrize(
    "artifact", ["lock", "inventory", "input", "attempt", "receipt", "transport"]
)
def test_cross_version_evidence_cannot_be_reused(tmp_path, artifact):
    old, tasks, _ = fixture_store(tmp_path / "v1", scope=V1)
    new, new_tasks, _ = fixture_store(tmp_path / "v2", scope=V2)
    complete(old, tasks, tasks[0])
    complete(new, new_tasks, new_tasks[0])
    name = {
        "lock": "lock.json",
        "inventory": "input_inventory.json",
        "input": old.path("input:" + tasks[0]["problem"]["id"]).name,
        "attempt": old.path(tasks[0]["id"]).name,
        "receipt": old.path(tasks[0]["id"]).with_suffix(".started.json").name,
        "transport": old.path(tasks[0]["id"]).with_suffix(".worker.json").name,
    }[artifact]
    shutil.copyfile(old.directory / name, new.directory / name)
    with pytest.raises(shared.ExecutionIntegrityError):
        shared.report_store(new_tasks, new)
    with pytest.raises(shared.ExecutionIntegrityError):
        new.execute_attempt(new_tasks[0], new_tasks, forbid)


@pytest.mark.parametrize("state", ["completed", "failed", "timed_out", "interrupted"])
def test_v2_resume_preserves_every_attempt_and_sealed_input(tmp_path, state):
    store, tasks, _ = fixture_store(tmp_path, scope=V2)
    if state == "completed":
        complete(store, tasks, tasks[0])
    elif state == "interrupted":

        def interrupt(_):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            store.execute_attempt(tasks[0], tasks, interrupt)
    else:
        store.execute_attempt(
            tasks[0], tasks, lambda _: dict(failure=state, work_unknown=True)
        )
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    for _ in range(2):
        store.execute_attempt(tasks[0], tasks, forbid)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    store.path("input:" + tasks[0]["problem"]["id"]).unlink()
    with pytest.raises(shared.ExecutionIntegrityError):
        store.execute_attempt(tasks[-1], tasks, forbid)
    with pytest.raises(shared.ExecutionIntegrityError):
        store.prepare_inputs(tasks, forbid)


def test_v2_sealing_required_before_worker(tmp_path):
    store, tasks, _ = fixture_store(tmp_path, scope=V2, prepare=False)
    with pytest.raises(shared.ExecutionIntegrityError):
        store.execute_attempt(tasks[0], tasks, forbid)

    def interrupt(p, observation):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.prepare_inputs(tasks, interrupt)
    with pytest.raises(shared.ExecutionIntegrityError):
        store.execute_attempt(tasks[0], tasks, forbid)
    with pytest.raises(shared.ExecutionIntegrityError):
        store.prepare_inputs(tasks, fixture_input)


@pytest.mark.parametrize("scope", [V1, V2])
def test_version_specific_report_preserves_missing_denominators(tmp_path, scope):
    store, tasks, _ = fixture_store(tmp_path, scope=scope)
    complete(store, tasks, tasks[0])
    out = shared.report_store(tasks, store)
    assert out["scope"] == scope
    assert out.get("schema") == verification.PROTOCOLS[scope].report_schema
    assert out["summary"]["all_attempt_denominator"] == len(tasks)
    assert out["summary"]["outcomes"]["missing_task"] == 3
    assert out["summary"]["audit"]["status"] == "not_applicable"
    for a in out["analyses"]:
        assert a["audit"]["status"] == "not_applicable"
        assert ("final_validation" in a) == (scope == V2)


@pytest.mark.parametrize("matching", [True, False])
@pytest.mark.parametrize("poisson", [True, False])
def test_previous_verification_comparison_requires_identical_deterministic_inputs(
    matching, poisson
):
    entry = dict(
        arrays={
            k: dict(sha256=k)
            for k in ("time", "kernel", "mean", "observed", "generating_kernel")
        }
    )
    baseline = dict(
        problems=dict(
            test=dict(
                arrays={k: k for k in entry["arrays"]},
                fits={"trf:compact": dict(centered_nll=4)},
            )
        )
    )
    if not matching:
        entry["arrays"]["kernel"]["sha256"] = "changed"
    task = dict(
        problem=dict(id="test", seed=[999] if poisson else None),
        configuration="trf:compact",
    )
    r = verification._history(
        task, entry, dict(centered_nll=3), baseline, previous_verification=True
    )
    assert r["new_minus_original_centered_nll"] == (
        -1 if matching and not poisson else None
    )
    assert r["audit_status"] == "not_applicable"
    assert not r["controlled_poisson_before_after"]


def test_final_validation_reporting_keeps_rejection_and_descent_separate():
    fit = dict(
        numerical=dict(
            accepted=False,
            required_recovery_complete=False,
            validation_matches_candidate=True,
            diagnostic_improvement=0.002,
            validation=dict(
                complete=False,
                passed=None,
                best_improvement=0,
                probes=[
                    dict(
                        status="redundant_physical_candidate",
                        feasible=True,
                        centered_nll=None,
                    ),
                    dict(status="unrepresentable", feasible=True, centered_nll=None),
                ],
            ),
        ),
        optimization=dict(
            promotions=["probe"], recovery_unavailable_reason="validation_incomplete"
        ),
    )
    row = verification._validation_summary(fit)
    assert len(row["redundant_physical_candidates"]) == 1
    assert len(row["unavailable_feasible_directions"]) == 1
    assert row["detected_descent_above_tolerance"]
    assert row["passed"] is None and row["matches_returned_candidate"]
    assert row["required_recovery_complete"] is False
    assert row["promotions"] == ["probe"]
    assert (
        verification._validation_summary({})["detected_descent_above_tolerance"] is None
    )


def test_v2_report_uses_lock_bound_v1_baseline_only(tmp_path, monkeypatch):
    tasks = [t for t in tiny_tasks() if t["configuration"] != "audit"]
    lock = software_lock(tasks)
    lock["fingerprint"]["scope"] = V2
    baseline = dict(problems={})
    for task in tasks:
        p = task["problem"]
        arrays = fixture_input(p)
        baseline["problems"].setdefault(
            p["id"],
            dict(
                arrays={
                    k: shared._array_identity(arrays[k])["sha256"]
                    for k in ("time", "mean", "kernel", "observed", "generating_kernel")
                },
                fits={},
            ),
        )["fits"][task["configuration"]] = dict(centered_nll=7, accepted=False)
    lock["fingerprint"]["execution"]["previous_verification_baseline"] = baseline
    store = shared.DevelopmentStore(tmp_path, lock, verify_execution=lambda: lock)
    store.prepare_inputs(tasks, fixture_input)
    for task in tasks:
        complete(store, tasks, task)
    monkeypatch.setattr(shared, "_fit_triexponential", forbid)
    monkeypatch.setattr(shared, "run_audit", forbid)
    out = shared.report_store(tasks, store)
    for task, row in zip(tasks, out["analyses"], strict=True):
        previous = row["previous_verification_comparison"]
        if task["problem"]["seed"] is None:
            assert previous["status"] == "deterministic_identical_inputs"
            assert previous["new_minus_original_centered_nll"] == pytest.approx(-7)
        else:
            assert previous["status"] == "independent_poisson_draw"
            assert previous["new_minus_original_centered_nll"] is None


def test_registry_is_closed_and_unknown_report_scope_rejected(tmp_path):
    with pytest.raises(TypeError):
        verification.PROTOCOLS["arbitrary"] = verification.PROTOCOLS[V1]
    with pytest.raises(ValueError, match="unknown"):
        verification.protocol_for_scope("arbitrary")
    store, tasks, lock = fixture_store(tmp_path, scope="unknown-verification")
    with pytest.raises(shared.ExecutionIntegrityError, match="unknown"):
        shared.report_store(tasks, store)
