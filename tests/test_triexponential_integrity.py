"""Tiny fixed arrays exercise evidence contracts; no pilot inputs or searches."""

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tcspc_toolkit.benchmarks import triexponential as development

ROOT = Path(__file__).resolve().parents[1]
Error = development.ExecutionIntegrityError


def tiny_tasks():
    return [
        dict(id=p["id"] + ":" + config, problem=p, configuration=config)
        for p in (
            dict(
                id="software:mean:matched",
                observation_id="software:mean",
                seed=None,
                assumed_irf="matched",
                condition=dict(
                    stratum="software", bins=4, bin_width_ns=0.25, time_origin_ns=0
                ),
                control=False,
            ),
            dict(
                id="software:counts:matched",
                observation_id="software:counts",
                seed=[999, 1],
                assumed_irf="matched",
                condition=dict(
                    stratum="software", bins=4, bin_width_ns=0.25, time_origin_ns=0
                ),
                control=False,
            ),
        )
        for config in ("audit", "lbfgsb:compact", "trf:compact")
    ]


def pack_input(time, observed, kernel, *, truth=None):
    def checksum(a):
        return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()

    return dict(
        time=time,
        observed=observed,
        kernel=kernel,
        generating_kernel=kernel,
        mean=observed,
        truth=truth or dict(lifetimes_ns=[1, 2, 3], amplitudes=[1, 1, 1]),
        observation_sha256=checksum(observed),
        assumed_irf_sha256=checksum(kernel),
    )


def tiny_input(problem, observation=None):
    values = [0.2, 1.5, 2.5, 1] if problem["seed"] is None else [0, 1, 2, 1]
    return pack_input([0, 0.25, 0.5, 0.75], values, [0, 2, 2, 0])


def software_lock(tasks):
    return dict(
        schema=development.EXECUTION_SCHEMA,
        execution_authorized=True,
        maintainer_review="software-fixture-only",
        fingerprint=dict(
            manifest_file_sha256="software-manifest",
            source="software-source",
            execution=dict(
                tasks=development.jsonable(tasks),
                worker_deadlines={t["configuration"]: 0.01 for t in tasks},
            ),
        ),
    )


def store_fixture(directory, tasks=None, *, prepare=True):
    tasks = tiny_tasks() if tasks is None else tasks
    lock = software_lock(tasks)
    store = development.DevelopmentStore(directory, lock, verify_execution=lambda: lock)
    if prepare:
        store.prepare_inputs(tasks, tiny_input)
    return store, tasks, lock


def rewrite_hashed(path, alter):
    record = json.loads(path.read_text())
    alter(record)
    record.pop("payload_sha256", None)
    record["payload_sha256"] = development.digest(record)
    path.write_text(json.dumps(record))


def mock_script(monkeypatch, tasks, lock):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    module = script["main"].__globals__
    monkeypatch.setitem(module, "require_authorization", lambda *a: lock)
    monkeypatch.setitem(module, "task_inventory", lambda *a, **kw: tasks)
    return script, module


def test_successful_seal_has_independent_identity_and_bound_attempts(tmp_path):
    store, tasks, lock = store_fixture(tmp_path)
    inventory = store.verify_inventory(tasks)
    assert len(inventory["inputs"]) == 2 and len(inventory["task_inputs"]) == 6
    assert inventory["payload_sha256"] == development.digest(
        {k: v for k, v in inventory.items() if k != "payload_sha256"}
    )
    assert inventory["lock_sha256"] == development.digest(lock)
    assert {e["target_kind"] for e in inventory["inputs"]} == {
        "raw_counts",
        "deterministic_expected_counts",
    }
    for task in tasks:
        result = store.execute_attempt(task, tasks, lambda binding: {"software": True})
        assert result["binding"]["inventory_sha256"] == inventory["payload_sha256"]
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    for _ in range(2):
        for task in tasks:
            store.execute_attempt(
                task, tasks, lambda binding: pytest.fail("hidden retry")
            )
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


@pytest.mark.parametrize("crash_at", ["generation", "seal"])
def test_incomplete_preparation_cannot_resume_or_run(tmp_path, monkeypatch, crash_at):
    store, tasks, lock = store_fixture(tmp_path, prepare=False)
    original = development._publish_record
    calls = []

    def builder(problem, observation):
        calls.append(problem["id"])
        if crash_at == "generation" and len(calls) == 2:
            raise KeyboardInterrupt()
        return tiny_input(problem)

    def publish(path, *args, **kwargs):
        if crash_at == "seal" and path.name == "input_inventory.json":
            raise KeyboardInterrupt()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(development, "_publish_record", publish)
    with pytest.raises(KeyboardInterrupt):
        store.prepare_inputs(tasks, builder)
    assert not (tmp_path / "input_inventory.json").exists()
    with pytest.raises(Error, match="incomplete preparation"):
        store.verify_inventory(tasks)
    with pytest.raises(Error, match="already started"):
        store.prepare_inputs(tasks, lambda *a: pytest.fail("regeneration"))
    script, module = mock_script(monkeypatch, tasks, lock)
    monkeypatch.setitem(
        module, "construct_problem", lambda *a, **k: pytest.fail("generation on resume")
    )
    monkeypatch.setitem(
        module, "bounded_worker", lambda *a: pytest.fail("worker before sealing")
    )
    with pytest.raises(Error, match="incomplete preparation"):
        script["main"](
            [
                "run",
                "--pilot",
                "--directory",
                str(tmp_path),
                "--execution-lock",
                "software",
            ]
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "observed",
        "kernel",
        "inventory",
        "inventory_rehashed",
        "input_id",
        "attempt_hash",
        "attempt_binding",
        "execution",
        "source",
        "foreign",
    ],
)
def test_resume_and_reporting_fail_closed_on_provenance_damage(
    tmp_path, monkeypatch, change
):
    store, tasks, lock = store_fixture(tmp_path)
    task = tasks[0]
    store.execute_attempt(task, tasks, lambda binding: {"software": True})
    input_path = store.path("input:" + task["problem"]["id"])
    seal = tmp_path / "input_inventory.json"
    if change == "missing":
        input_path.unlink()
    elif change in ("observed", "kernel"):

        def alter(record):
            record["result"][change][0] += 1
            key = "observation_sha256" if change == "observed" else "assumed_irf_sha256"
            record["result"][key] = development._array_identity(
                record["result"][change]
            )["sha256"]

        rewrite_hashed(input_path, alter)
    elif change == "inventory":
        seal.write_text("{")
    elif change == "inventory_rehashed":
        rewrite_hashed(seal, lambda r: r["inputs"].reverse())
    elif change == "input_id":
        rewrite_hashed(
            seal, lambda r: r["task_inputs"][0].update(input_id="input:foreign")
        )
    elif change == "attempt_hash":
        store.path(task["id"]).write_text("{}")
    elif change == "attempt_binding":
        rewrite_hashed(
            store.path(task["id"]),
            lambda r: r["binding"].update(input_id="input:foreign"),
        )
    elif change == "execution":
        rewrite_hashed(store.path(task["id"]), lambda r: r.update(lock="foreign"))
    elif change == "source":
        lock["fingerprint"]["source"] = "changed"
    else:
        (tmp_path / "duplicate.json").write_bytes(store.path(task["id"]).read_bytes())
    monkeypatch.setattr(
        development,
        "cross_score",
        lambda *a, **k: pytest.fail("scoring invalid evidence"),
    )
    with pytest.raises(Error):
        store.execute_attempt(
            tasks[1], tasks, lambda *a: pytest.fail("worker on corrupt evidence")
        )
    with pytest.raises(Error):
        development.report_store(tasks, store)


def test_running_failed_timeout_and_completed_are_preserved(tmp_path):
    store, tasks, _ = store_fixture(tmp_path)

    def interrupt(binding):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        store.execute_attempt(tasks[0], tasks, interrupt)
    store.execute_attempt(tasks[1], tasks, lambda b: 1 / 0)
    store.execute_attempt(
        tasks[2],
        tasks,
        lambda b: dict(
            failure="external_worker_timeout", work_unknown=True, hard_wall_seconds=0.01
        ),
    )
    store.execute_attempt(tasks[3], tasks, lambda b: {"software": True})
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    for task in tasks[:4]:
        store.execute_attempt(task, tasks, lambda *a: pytest.fail("retry"))
    report = development.report_store(tasks, store)
    assert report["summary"]["all_attempt_denominator"] == 6
    assert [r["status"] for r in report["attempts"]] == [
        "running",
        "exception",
        "completed",
        "completed",
        "missing_task",
        "missing_task",
    ]
    assert report["summary"]["groups"]["trf:compact"]["censored_runtime_count"] == 1
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


def test_duplicate_or_changed_task_specification_rejected(tmp_path):
    store, tasks, _ = store_fixture(tmp_path)
    with pytest.raises(Error):
        store.verify_inventory(tasks + [tasks[0]])
    changed = development.jsonable(tasks)
    changed[0]["problem"]["assumed_irf"] = "foreign"
    with pytest.raises(Error):
        development.report_store(changed, store)


def test_prepare_run_worker_transport_and_resume_are_separate(tmp_path, monkeypatch):
    tasks = tiny_tasks()
    lock = software_lock(tasks)
    directory = tmp_path / "new"
    script, module = mock_script(monkeypatch, tasks, lock)
    prepared, launched = [], []

    def prepare(manifest, problem, observation=None):
        prepared.append(problem["id"])
        return tiny_input(problem, observation)

    monkeypatch.setitem(module, "construct_problem", prepare)
    monkeypatch.setitem(
        module, "bounded_worker", lambda *a: pytest.fail("worker in preparation")
    )
    args = ["--pilot", "--directory", str(directory), "--execution-lock", "software"]
    script["main"](["prepare", *args])
    assert len(prepared) == 2
    store = development.DevelopmentStore(
        directory, lock, create=False, verify_execution=lambda: lock
    )

    def worker(command, seconds):
        task = next(
            t for t in tasks if t["id"] == command[command.index("--task-id") + 1]
        )
        arrays, binding = store.worker_input(task, tasks)
        assert len(arrays["observed"]) == 4
        launched.append(task["id"])
        store.write_worker_result(task, tasks, {"software": True}, binding)
        return True

    monkeypatch.setitem(
        module, "construct_problem", lambda *a, **k: pytest.fail("regeneration")
    )
    monkeypatch.setitem(module, "bounded_worker", worker)
    script["main"](["run", *args])
    script["main"](["run", *args])
    assert len(launched) == 6 and len(prepared) == 2
    with pytest.raises(Error, match="already started"):
        script["main"](["prepare", *args])
    with pytest.raises(Error, match="running attempt"):
        store.worker_input(tasks[0], tasks)


def test_worker_transport_is_bound_and_verified(tmp_path):
    store, tasks, _ = store_fixture(tmp_path)
    task = tasks[0]

    def operation(binding):
        store.worker_input(task, tasks)
        store.write_worker_result(task, tasks, {"software": True}, binding)
        rewrite_hashed(
            store.path(task["id"]).with_suffix(".worker.json"),
            lambda r: r["binding"].update(inventory_sha256="foreign"),
        )
        return store.read_worker_result(task, tasks, binding)

    with pytest.raises(Error, match="worker result"):
        store.execute_attempt(task, tasks, operation)
    assert store.read(task["id"])["status"] == "exception"


@pytest.mark.parametrize("corruption", ["hash", "binding"])
def test_retained_worker_transport_is_verified_on_resume_and_report(
    tmp_path, corruption
):
    store, tasks, _ = store_fixture(tmp_path)
    task = tasks[0]

    def operation(binding):
        store.worker_input(task, tasks)
        store.write_worker_result(task, tasks, {"software": True}, binding)
        return store.read_worker_result(task, tasks, binding)

    store.execute_attempt(task, tasks, operation)
    path = store.path(task["id"]).with_suffix(".worker.json")
    if corruption == "hash":
        path.write_text("{}")
    else:
        rewrite_hashed(path, lambda r: r["binding"].update(inventory_sha256="foreign"))
    with pytest.raises(Error):
        store.execute_attempt(
            tasks[1], tasks, lambda *a: pytest.fail("worker launched")
        )
    with pytest.raises(Error):
        development.report_store(tasks, store)


@pytest.mark.parametrize("status", ["completed", "exception", "running", "timeout"])
def test_missing_attempt_checkpoint_is_not_a_new_task(tmp_path, monkeypatch, status):
    store, tasks, lock = store_fixture(tmp_path)
    task = tasks[0]

    def operation(binding):
        if status == "running":
            raise KeyboardInterrupt()
        if status == "exception":
            raise RuntimeError("software failure")
        return (
            {"failure": "external_worker_timeout"}
            if status == "timeout"
            else {"software": True}
        )

    if status == "running":
        with pytest.raises(KeyboardInterrupt):
            store.execute_attempt(task, tasks, operation)
    else:
        store.execute_attempt(task, tasks, operation)
    store.path(task["id"]).unlink()
    result = store.execute_attempt(
        task, tasks, lambda *a: pytest.fail("deleted attempt retried")
    )
    assert result["status"] == "interrupted"
    report = development.report_store(tasks, store)
    assert report["attempts"][0]["status"] == "interrupted"
    assert report["summary"]["all_attempt_denominator"] == 6
    script, module = mock_script(monkeypatch, tasks, lock)
    monkeypatch.setitem(
        module, "bounded_worker", lambda *a: pytest.fail("unconfirmed prior worker")
    )
    with pytest.raises(script["WorkerCleanupError"], match="unconfirmed termination"):
        script["main"](
            [
                "run",
                "--pilot",
                "--directory",
                str(tmp_path),
                "--execution-lock",
                "software",
            ]
        )


def test_worker_cannot_start_twice_for_same_attempt(tmp_path):
    store, tasks, _ = store_fixture(tmp_path)

    def operation(binding):
        store.worker_input(tasks[0], tasks)
        with pytest.raises(FileExistsError):
            store.worker_input(tasks[0], tasks)
        return {"software": True}

    store.execute_attempt(tasks[0], tasks, operation)


def test_inventory_is_re_read_after_atomic_publication(tmp_path, monkeypatch):
    store, tasks, _ = store_fixture(tmp_path, prepare=False)
    original = development._publish_record

    def corrupt_after_publication(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == "input_inventory.json":
            input_path = store.path("input:" + tasks[0]["problem"]["id"])
            input_path.write_text("{}")
        return result

    monkeypatch.setattr(development, "_publish_record", corrupt_after_publication)
    with pytest.raises(Error, match="hash"):
        store.prepare_inputs(tasks, tiny_input)


def test_unapproved_store_cannot_prepare_inputs(tmp_path):
    tasks = tiny_tasks()
    lock = software_lock(tasks)
    lock["execution_authorized"] = False
    store = development.DevelopmentStore(tmp_path, lock, verify_execution=lambda: lock)
    with pytest.raises(Error, match="unverified"):
        store.prepare_inputs(tasks, lambda *a: pytest.fail("unapproved generation"))


def test_preparation_exception_is_retained_without_a_seal(tmp_path):
    store, tasks, _ = store_fixture(tmp_path, prepare=False)
    with pytest.raises(Error, match="preparation incomplete"):
        store.prepare_inputs(tasks, lambda *a: 1 / 0)
    assert store.read("input:" + tasks[0]["problem"]["id"])["status"] == "exception"
    with pytest.raises(Error, match="already started"):
        store.prepare_inputs(tasks, tiny_input)


@pytest.mark.parametrize("change", ["grid", "target_kind"])
def test_preparation_checks_condition_and_deterministic_target(tmp_path, change):
    store, tasks, _ = store_fixture(tmp_path, prepare=False)

    def invalid(problem, observation):
        arrays = tiny_input(problem)
        if change == "grid":
            arrays["time"][1] += 0.01
        else:
            arrays["mean"] = [1, 1, 1, 1]
        return arrays

    with pytest.raises(Error):
        store.prepare_inputs(tasks, invalid)
    assert not (tmp_path / "input_inventory.json").exists()


@pytest.mark.parametrize("mismatch", [False, True])
def test_paired_assumptions_require_identical_observations(tmp_path, mismatch):
    tasks = tiny_tasks()[:3]
    paired = development.jsonable(tasks[0]["problem"])
    paired.update(id="software:mean:gaussian", assumed_irf="gaussian")
    tasks.append(
        dict(id=paired["id"] + ":audit", problem=paired, configuration="audit")
    )
    store, _, _ = store_fixture(tmp_path, tasks, prepare=False)

    def builder(problem, observation):
        if observation is None:
            return tiny_input(problem)
        observed = list(observation["observed"])
        if mismatch:
            observed[0] += 1
        result = pack_input(observation["time"], observed, [4 / 3] * 4)
        result["generating_kernel"] = observation["generating_kernel"]
        return result

    if mismatch:
        with pytest.raises(Error, match="paired observation"):
            store.prepare_inputs(tasks, builder)
        assert not (tmp_path / "input_inventory.json").exists()
    else:
        inventory = store.prepare_inputs(tasks, builder)
        first, second = inventory["inputs"]
        assert first["arrays"]["observed"] == second["arrays"]["observed"]
        assert first["arrays"]["kernel"] != second["arrays"]["kernel"]
