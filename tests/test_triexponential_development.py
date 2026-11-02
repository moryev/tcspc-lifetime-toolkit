"""Nonexecuting protocol, persistence and all-attempt reporting tests."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tcspc_toolkit.benchmarks import triexponential as development

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / "configs/issue37_pr2_development_v1.json").read_text())


def test_exact_population_namespaces_and_no_generation(monkeypatch):
    monkeypatch.setattr(
        development,
        "construct_problem",
        lambda *a: pytest.fail("inventory must not generate"),
    )
    m = manifest()
    comprehensive = development.task_inventory(m)
    pilot = development.task_inventory(m, pilot=True)
    assert len(comprehensive) == 560 and len(pilot) == 18
    p = development.problem_inventory(m)
    assert len(p) == 80 and len({r["observation_id"] for r in p}) == 72
    assert sum(r["control"] for r in p) == 8
    assert {tuple(r["seed"][:4]) for r in p if r["seed"]}.isdisjoint(
        {tuple(t["problem"]["seed"][:4]) for t in pilot if t["problem"]["seed"]}
    )
    assert not m["execution_authorized"]


def test_atomic_resume_failure_and_tamper_detection(tmp_path):
    store = development.DevelopmentStore(tmp_path, {"test_lock": 1})
    calls = []
    first = store.execute("toy", lambda: calls.append(1) or {"value": 3})
    assert store.execute("toy", lambda: pytest.fail("hidden retry")) == first
    assert len(calls) == 1
    failed = store.execute("failed", lambda: 1 / 0)
    assert failed["status"] == "exception"
    assert store.execute("failed", lambda: pytest.fail("retry failed")) == failed
    with pytest.raises(ValueError, match="lock"):
        development.DevelopmentStore(tmp_path, {"test_lock": 2})
    payload = json.loads(store.path("toy").read_text())
    payload["result"]["value"] = 4
    store.path("toy").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        store.read("toy")


def test_interrupted_attempt_is_preserved(tmp_path):
    store = development.DevelopmentStore(tmp_path, {"test_lock": 1})

    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        store.execute("interrupted", interrupted)
    assert store.read("interrupted")["status"] == "running"
    assert (
        store.execute("interrupted", lambda: pytest.fail("retry"))["status"]
        == "running"
    )


@pytest.mark.parametrize("interrupted", [False, True])
def test_running_status_cannot_bypass_checkpoint_integrity(tmp_path, interrupted):
    store = development.DevelopmentStore(tmp_path, {"test_lock": 1})

    def operation():
        if interrupted:
            raise KeyboardInterrupt()
        return {"value": 3}

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            store.execute("toy", operation)
    else:
        store.execute("toy", operation)
    payload = json.loads(store.path("toy").read_text())
    payload.update(status="running", result={"value": 999})
    store.path("toy").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        store.execute("toy", lambda: pytest.fail("corruption must not trigger retry"))


def test_all_attempt_reporting_keeps_missing_and_failed_cases():
    tasks = development.task_inventory(manifest())
    records = [
        dict(id=tasks[0]["id"], status="exception"),
        dict(
            id=tasks[1]["id"],
            status="completed",
            result=dict(failure="external_worker_timeout", work_unknown=True),
        ),
    ]
    summary = development.summarize_tasks(tasks, records)
    assert summary["all_attempt_denominator"] == 560
    assert all(
        g["denominator"] == 80 and g["controls"] == 8 and g["robustness"] == 72
        for g in summary["groups"].values()
    )
    assert (
        sum(g["statuses"].get("missing_task", 0) for g in summary["groups"].values())
        == 558
    )
    assert not summary["qualification"] and not summary["production_policy_selected"]
    with pytest.raises(ValueError):
        development.summarize_tasks(tasks, records + records)


def test_construct_small_software_fixture_pairs_irfs_without_study_draws():
    m = manifest()
    # Deliberately a tiny deterministic software fixture, not any Dxx/Pxx observation.
    p = dict(
        id="software:mean:matched",
        observation_id="software:mean",
        seed=None,
        assumed_irf="matched",
        condition=m["defaults"]
        | dict(bins=16, bin_width_ns=0.25, signal_photons=50, generating_irf="emg"),
    )
    first = development.construct_problem(m, p)
    second = development.construct_problem(
        m, p | dict(assumed_irf="gaussian"), observation=first
    )
    np.testing.assert_array_equal(first["observed"], second["observed"])
    assert first["observation_sha256"] == second["observation_sha256"]
    assert first["assumed_irf_sha256"] != second["assumed_irf_sha256"]


def test_scientific_recovery_does_not_invent_degenerate_truth():
    for tau, amplitudes in [([1], [3]), ([1, 1, 3], [1, 2, 3]), ([1, 2, 3], [1, 0, 3])]:
        assert not development.component_recovery(
            {"assumed_irf": "matched"},
            dict(lifetimes_ns=tau, amplitudes=amplitudes),
            dict(lifetimes_ns=[1, 2, 3]),
        )["available"]


def test_script_import_is_inert_and_execution_lock_is_required(tmp_path):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    m = manifest()
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(dict(execution_authorized=False)))
    with pytest.raises(ValueError, match="authorized"):
        script["require_authorization"](path, m, True)
    with pytest.raises(SystemExit):
        script["main"](["run", "--directory", str(tmp_path)])


@pytest.mark.parametrize(
    "change", ["source", "manifest", "scope", "threads", "permission"]
)
def test_execution_fingerprint_rejects_changed_authorization(
    tmp_path, monkeypatch, change
):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    module = script["fingerprint"].__globals__
    source = tmp_path / "src/tcspc_toolkit/control.py"
    source.parent.mkdir(parents=True)
    source.write_text("# software fixture\n")
    runner = tmp_path / "runner.py"
    runner.write_text("# software runner\n")
    monkeypatch.setitem(module, "ROOT", tmp_path)
    monkeypatch.setitem(module, "__file__", str(runner))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    m = manifest()
    lock = dict(
        execution_authorized=True,
        maintainer_review="software-test-only",
        fingerprint=script["fingerprint"](m, True),
    )
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(lock))
    assert script["require_authorization"](path, m, True) == lock
    pilot = True
    if change == "source":
        source.write_text("# changed software fixture\n")
    elif change == "manifest":
        m["defaults"]["signal_photons"] += 1
    elif change == "scope":
        pilot = False
    elif change == "threads":
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
    else:
        lock["execution_authorized"] = False
        path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="matching execution lock"):
        script["require_authorization"](path, m, pilot)


def test_report_rejects_foreign_fingerprint_before_cross_scoring(tmp_path, monkeypatch):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    (tmp_path / "lock.json").write_text(
        json.dumps(
            dict(
                execution_authorized=True,
                maintainer_review="software-test-only",
                fingerprint={"foreign": True},
            )
        )
    )
    monkeypatch.setitem(
        script["main"].__globals__,
        "report_store",
        lambda *a: pytest.fail("must verify source lock before scoring"),
    )
    with pytest.raises(ValueError, match="matching execution lock"):
        script["main"](["report", "--pilot", "--directory", str(tmp_path)])


def test_sealed_store_reporting_does_not_mutate_attempts(tmp_path, monkeypatch):
    from test_triexponential_audit import completed_control, control, fake_fit

    audit = completed_control(monkeypatch)
    time, observed, kernel = control()
    problem = dict(
        id="software:mean:matched",
        control=True,
        assumed_irf="matched",
        condition=dict(stratum="software"),
    )
    tasks = [
        dict(id=problem["id"] + ":" + config, problem=problem, configuration=config)
        for config in ("audit", "trf:compact")
    ]
    store = development.DevelopmentStore(tmp_path, {"software_test": True})
    store.execute(
        "input:" + problem["id"],
        lambda: dict(
            time=time,
            observed=observed,
            kernel=kernel,
            truth=dict(lifetimes_ns=[1, 2, 3], amplitudes=[0, 0, 0]),
        ),
    )
    store.execute(tasks[0]["id"], lambda: audit)
    p = fake_fit(audit.parameters, time, kernel, observed)
    store.execute(
        tasks[1]["id"],
        lambda: dict(
            parameters=vars(p.parameters),
            numerical=vars(p.numerical),
            expected_counts=p.expected_counts,
            centered_nll=p.centered_nll,
        ),
    )
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    report = development.report_store(tasks, store)
    assert report["attempts"][1]["result"]["cross_score"]["status"] == "agreement"
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


def test_worker_timeout_is_reported_without_retry(monkeypatch):
    import subprocess

    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))

    class Process:
        pid = 123
        returncode = None
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("software", timeout)
            assert timeout == 5
            return "", ""

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Process())
    from types import SimpleNamespace

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    assert not script["bounded_worker"](["software"], 0.01)


def test_unconfirmed_cleanup_is_bounded(monkeypatch):
    import subprocess

    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))

    class Process:
        pid = 123

        def communicate(self, timeout=None):
            assert timeout is not None
            raise subprocess.TimeoutExpired("software", timeout)

        def kill(self):
            pass

    def kill_tree(*args, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired("software-tree-kill", 5)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(subprocess, "run", kill_tree)
    with pytest.raises(script["WorkerCleanupError"], match="cleanup unconfirmed"):
        script["bounded_worker"](["software"], 0.01)


def test_cleanup_failure_persists_timeout_and_stops_execution_and_resume(
    tmp_path, monkeypatch
):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    module = script["main"].__globals__
    lock = {"software_fixture": True}
    monkeypatch.setitem(module, "require_authorization", lambda *a: lock)
    monkeypatch.setitem(
        module,
        "construct_problem",
        lambda *a, **kw: {
            "observation_sha256": "software-only-no-generated-observation",
        },
    )
    calls = []

    def unconfirmed(*args):
        calls.append(1)
        raise script["WorkerCleanupError"]("software-only cleanup failure")

    monkeypatch.setitem(module, "bounded_worker", unconfirmed)
    argv = [
        "run",
        "--pilot",
        "--directory",
        str(tmp_path),
        "--execution-lock",
        "unused",
    ]
    for _ in range(2):
        with pytest.raises(script["WorkerCleanupError"], match="run stopped"):
            script["main"](argv)
    assert len(calls) == 1  # No further task or resume retry after cleanup failure.
    store = development.DevelopmentStore(tmp_path, lock)
    tasks = development.task_inventory(manifest(), pilot=True)
    records = [store.read(t["id"]) for t in tasks]
    assert records[0]["result"]["failure"] == "external_worker_timeout"
    assert records[0]["result"]["cleanup_complete"] is False
    assert all(r["status"] == "missing_task" for r in records[1:])
    summary = development.summarize_tasks(tasks, records)
    assert summary["groups"]["audit"]["censored_runtime_count"] == 1


def test_timeout_times_are_censored_contributors_not_omitted():
    tasks = development.task_inventory(manifest(), pilot=True)
    task = next(t for t in tasks if t["configuration"] == "trf:compact")
    report = development.summarize_tasks(
        tasks,
        [
            dict(
                id=task["id"],
                status="completed",
                result=dict(
                    failure="external_worker_timeout",
                    hard_wall_seconds=60,
                    work_unknown=True,
                ),
            )
        ],
    )
    group = report["groups"]["trf:compact"]
    assert group["denominator"] == 6
    assert group["censored_runtime_count"] == 1
    assert group["runtime_summary"]["contributors"] == 1
    assert group["runtime_summary"]["median"] == 60


def test_reported_contract_violations_are_separate_from_quality():
    from test_triexponential import fit

    record = development.jsonable(fit())
    assert (
        development.reported_correctness(record)["status"]
        == "reported_contracts_consistent"
    )
    record["numerical"]["validation"]["candidate"]["background"] = 7
    assert (
        "accepted_without_exact_candidate_assessment"
        in development.reported_correctness(record)["violations"]
    )
    record["optimization"]["work"]["model_calls"] += 1
    assert (
        "work_accounting_inconsistent"
        in development.reported_correctness(record)["violations"]
    )
