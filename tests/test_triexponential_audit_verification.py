"""Closed audit-only metadata and tiny mocked-worker integrity controls."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tcspc_toolkit.benchmarks import _triexponential_audit_verification as scope
from tcspc_toolkit.benchmarks import _triexponential_verification as fit_scope
from tcspc_toolkit.benchmarks import triexponential as shared
from tcspc_toolkit.benchmarks._triexponential_audit import AuditSettings
from test_triexponential_integrity import (
    tiny_tasks,
    tiny_input,
    software_lock,
    rewrite_hashed,
)
from test_triexponential_verification import tiny_construction

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / scope.PROTOCOL.manifest_path
Error = shared.ExecutionIntegrityError


def manifest():
    return json.loads(MANIFEST.read_text())


def forbid(*args, **kwargs):
    pytest.fail("implicit construction, numerical search or production scoring")


def test_exact_registry_and_settings_without_numerical_work(monkeypatch):
    output_existed = (ROOT / scope.PROTOCOL.output_directory).exists()
    monkeypatch.setattr(shared, "construct_problem", forbid)
    monkeypatch.setattr(shared, "run_audit", forbid)
    monkeypatch.setattr(shared, "_fit_triexponential", forbid)
    m = manifest()
    assert list(scope.PROTOCOLS) == [scope.SCOPE]
    assert scope.validate_manifest(m) == scope.PROTOCOL
    tasks = shared.task_inventory(m, audit_verification=True)
    assert tuple(t["id"] for t in tasks) == scope.PROTOCOL.task_ids
    assert len(tasks) == 6 and {t["configuration"] for t in tasks} == {"audit"}
    assert len({t["problem"]["observation_id"] for t in tasks}) == 6
    actual = shared.execution_specification(m, audit_verification=True)
    assert actual["settings"] == m["configurations"]
    assert actual["settings"]["audit"] == asdict(AuditSettings(16, 8, 40000, 120))
    assert actual["preparation_model_call_limit"] == 18
    assert actual["reporting_model_call_limit"] == 0
    assert actual["worker_deadlines"] == {"audit": 120}
    assert actual["serial_workers"] == 1
    tasks[0]["problem"]["condition"]["amplitudes"][0] = -1
    assert m == manifest()
    assert (ROOT / scope.PROTOCOL.output_directory).exists() == output_existed


@pytest.mark.parametrize("field", scope.CONTRACT_KEYS)
def test_scientific_contract_is_pinned_independently(field):
    m = manifest()
    m[field] = {"changed": True}
    m["scientific_contract_sha256"] = shared.digest(
        {k: m[k] for k in scope.CONTRACT_KEYS}
    )
    with pytest.raises(ValueError, match="exact reviewed"):
        scope.validate_manifest(m)


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_id", "issue37_pr2c2_audit_verification_v2"),
        ("schema", "unknown"),
        ("execution_authorized", True),
        ("scientific_contract_sha256", "changed"),
        ("scientific_contract_keys", []),
    ],
)
def test_unknown_or_mismatched_manifest_rejected(field, value):
    m = manifest()
    m[field] = value
    with pytest.raises(ValueError):
        scope.validate_manifest(m)


def test_physical_definitions_namespaces_and_historical_proposal():
    proposal = (
        ROOT
        / "docs/validation/issue37_pr2c2_audit_verification_v1_manifest_proposal.json"
    )
    assert manifest()["historical_proposal"] == dict(
        path=proposal.relative_to(ROOT).as_posix(),
        sha256="b15c7bf828d32a7f0ae4c7c360d177b59243e388275507eb8a0addc73750ec93",
    )
    original = json.loads(
        (ROOT / "configs/issue37_pr2c1b_numerical_verification_v2.json").read_text()
    )
    m = manifest()
    for old, new in zip(original["conditions"], m["conditions"], strict=True):
        for key in old.keys() - {"id", "generating_kernel_id", "assumed_kernel_id"}:
            assert new[key] == old[key]
    assert m["irfs"] == original["irfs"]
    assert m["construction"]["namespace"] == [37, 2, 20261101, 4]
    assert (
        m["construction"]["reserved_namespaces"]["comprehensive_development"][-1] == 1
    )
    for index, p in enumerate(scope.problems(m)):
        assert p["seed"] == (
            None if index % 2 == 0 else [37, 2, 20261101, 4, index // 2, 0]
        )


@pytest.mark.parametrize("other", ["pilot", "comprehensive", "fit_v1", "fit_v2"])
def test_cross_scope_manifests_rejected_and_old_inventories_preserved(other):
    if other in ("pilot", "comprehensive"):
        m = json.loads((ROOT / "configs/issue37_pr2_development_v1.json").read_text())
        assert len(shared.task_inventory(m, pilot=other == "pilot")) == (
            18 if other == "pilot" else 560
        )
    else:
        protocol = list(fit_scope.PROTOCOLS.values())[int(other == "fit_v2")]
        m = json.loads((ROOT / protocol.manifest_path).read_text())
        assert len(shared.task_inventory(m, verification=True)) == 12
    with pytest.raises(ValueError):
        shared.task_inventory(m, audit_verification=True)
    with pytest.raises(ValueError):
        shared.task_inventory(
            manifest(), pilot=other == "pilot", verification=other.startswith("fit")
        )


@pytest.mark.parametrize(
    "function",
    [
        "settings_for_task",
        "execution_specification",
        "problem_inventory",
        "task_inventory",
    ],
)
@pytest.mark.parametrize("other", ["pilot", "verification"])
def test_exclusive_scope_flags(function, other):
    arg = "audit" if function == "settings_for_task" else manifest()
    with pytest.raises(ValueError):
        getattr(shared, function)(arg, audit_verification=True, **{other: True})


@pytest.mark.parametrize(
    "configuration", ["lbfgsb:compact", "trf:compact", "audit:other"]
)
def test_no_fit_configuration_in_audit_scope(configuration):
    with pytest.raises(ValueError):
        shared.settings_for_task(configuration, audit_verification=True)


def test_dispatch_never_passes_truth_or_production_estimates(monkeypatch):
    task = scope.tasks(manifest())[0]
    arrays = tiny_input(tiny_tasks()[0]["problem"]) | {
        "truth": object(),
        "production": object(),
    }

    def audit(time, observed, kernel, **kwargs):
        assert kwargs == dict(
            lifetime_bounds=[0.1, 10],
            shift_bounds=[-0.3, 0.3],
            settings=AuditSettings(16, 8, 40000, 120),
        )
        assert len(observed) == 4
        return "software"

    monkeypatch.setattr(shared, "run_audit", audit)
    monkeypatch.setattr(shared, "_fit_triexponential", forbid)
    assert shared.execute_task(task, arrays, audit_verification=True) == "software"


def test_preparation_reuses_three_counted_component_calls():
    m, p, reference = tiny_construction()
    actual = shared.construct_problem(m, p, audit_verification=True)
    np.testing.assert_array_equal(actual["mean"], reference["mean"])
    assert actual["preparation_work"]["model_calls"] == 3
    assert actual["preparation_work"]["charged_component_convolutions"] == 3
    assert 6 * actual["preparation_work"]["model_calls"] == 18


def fixture(directory, prepare=True):
    tasks = [t for t in tiny_tasks() if t["configuration"] == "audit"]
    lock = software_lock(tasks)
    lock["fingerprint"]["scope"] = scope.SCOPE
    lock["fingerprint"]["execution"].update(
        settings={"audit": asdict(AuditSettings(1, 1, 100, 10, validation_calls=20))},
        output_directory=str(directory),
    )
    store = shared.DevelopmentStore(directory, lock, verify_execution=lambda: lock)
    if prepare:
        store.prepare_inputs(tasks, tiny_input)
    return store, tasks, lock


def result_record(lock, complete=True):
    candidate = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.1, 0.0]
    starts = []
    for solver in ("trf", "powell"):
        work = dict(finite_evaluations=2, model_calls=2)
        if solver == "powell":
            work.update(infinite_objective=1, infeasible_trial=2, model_calls=3)
        starts.append(
            dict(
                solver=solver,
                initial=candidate,
                attempted=True,
                termination="terminated" if complete else "invocation_model_limit",
                solver_success=True if complete else None,
                calls=work["model_calls"],
                call_limit=39,
                elapsed_seconds=0.001,
                best_parameters=candidate,
                best_centered_nll=0.0,
                endpoint_centered_nll=0.0,
                endpoint_objective_consistent=True,
                work=list(work.items()),
                trial_diagnostics=[
                    dict(
                        status="infinite_objective"
                        if solver == "powell"
                        else "finite_best_discovery",
                        centered_nll=None if solver == "powell" else 0.0,
                    )
                ],
                trial_records_dropped=5,
                derivative_diagnostics=[dict(status="available")],
                derivative_records_dropped=8,
            )
        )
    return dict(
        parameters=candidate,
        expected_counts=[0.1] * 4,
        centered_nll=0.0,
        starts=starts,
        all_starts_attempted=True,
        search_complete=complete,
        assessment_complete=True,
        consistent=True,
        validation_complete=True,
        validation_passed=True,
        best_probe_improvement=0.0,
        validation_probes=[],
        validation_details=[],
        complete=complete,
        reasons=[] if complete else ["search_incomplete"],
        phase_calls=[["search", 5], ["assessment", 1], ["validation", 1]],
        component_convolutions=21,
        completed_component_convolutions=21,
        component_work_unknown_calls=0,
        elapsed_seconds=0.01,
        settings=lock["fingerprint"]["execution"]["settings"]["audit"],
        sealed=True,
        assessment_candidate=candidate,
        validation_candidate=candidate,
        work=[["model_calls", 7]],
        model_calls_by_origin=[["software", 7]],
        final_trial_diagnostics=[],
        final_trial_records_dropped=0,
    )


def complete_attempt(store, tasks, task, result):
    def operation(binding):
        _, actual = store.worker_input(task, tasks)
        assert actual == binding
        store.write_worker_result(task, tasks, result, binding)
        return store.read_worker_result(task, tasks, binding)

    return store.execute_attempt(task, tasks, operation)


def test_report_has_all_starts_status_counts_and_zero_models(tmp_path, monkeypatch):
    store, tasks, lock = fixture(tmp_path)
    for task, ok in zip(tasks, (True, False), strict=True):
        complete_attempt(store, tasks, task, result_record(lock, ok))
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    for name in (
        "run_audit",
        "_fit_triexponential",
        "_production_prediction",
        "direct_expected",
        "multiexponential_reconvolution_expected_counts",
        "cross_score",
    ):
        monkeypatch.setattr(shared, name, forbid)
    report = shared.report_store(tasks, store)
    assert report["summary"]["all_attempt_denominator"] == 2
    assert report["summary"]["scheduled_start_denominator"] == 4
    assert report["summary"]["strict_complete"] == 1
    assert report["summary"]["outcomes"] == {"audit_complete": 1, "audit_incomplete": 1}
    assert report["reporting_work"]["model_calls"] == 0
    assert (
        report["summary"]["production_minus_audit_objective_gaps"]["status"]
        == "not_applicable"
    )
    powell = report["starts"][1]
    assert powell["trial_classifications"] == dict(
        finite_objective=2,
        positive_infinite_objective=1,
        infeasible=2,
        numerical_failure=0,
    )
    assert powell["recorded"]["trial_records_dropped"] == 5
    assert powell["recorded"]["derivative_records_dropped"] == 8
    for _ in range(2):
        for task in tasks:
            store.execute_attempt(task, tasks, forbid)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


@pytest.mark.parametrize(
    "field,value",
    [
        ("search_complete", False),
        ("assessment_complete", False),
        ("consistent", False),
        ("validation_complete", False),
        ("validation_passed", False),
        ("parameters", None),
        ("assessment_candidate", [1] * 8),
        ("validation_candidate", [1] * 8),
        ("centered_nll", None),
        ("elapsed_seconds", 10),
        ("sealed", False),
        ("reasons", ["deadline"]),
    ],
)
def test_false_completion_claims_are_exposed(tmp_path, field, value):
    store, tasks, lock = fixture(tmp_path)
    r = result_record(lock)
    r[field] = value
    complete_attempt(store, tasks, tasks[0], r)
    report = shared.report_store(tasks, store)
    assert not report["analyses"][0]["strict_complete"]
    assert report["analyses"][0]["outcome"] == "inconsistent_completion_record"
    assert report["summary"]["all_attempt_denominator"] == 2
    assert report["summary"]["scheduled_start_denominator"] == 4


@pytest.mark.parametrize(
    "status", ["missing", "running", "failed", "timeout", "incomplete_result"]
)
def test_unavailable_work_retains_start_denominators(tmp_path, status):
    store, tasks, lock = fixture(tmp_path)
    if status == "running":

        def interrupt(binding):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            store.execute_attempt(tasks[0], tasks, interrupt)
    elif status in ("failed", "timeout"):
        store.execute_attempt(
            tasks[0],
            tasks,
            lambda b: dict(
                failure="external_worker_timeout"
                if status == "timeout"
                else "worker_failed",
                work_unknown=True,
                external_worker_timing=dict(
                    right_censored=status == "timeout",
                    runtime_lower_bound_seconds=0.01 if status == "timeout" else None,
                ),
            ),
        )
    elif status == "incomplete_result":
        complete_attempt(store, tasks, tasks[0], {})
    report = shared.report_store(tasks, store)
    assert len(report["starts"]) == 4 and report["summary"]["strict_complete"] == 0
    assert all(s["attempted"] is None and s["work"] is None for s in report["starts"])
    if status == "timeout":
        assert report["analyses"][0]["external_worker_timing"]["right_censored"]


@pytest.mark.parametrize(
    "damage",
    [
        "input_missing",
        "counts",
        "kernel",
        "inventory",
        "attempt",
        "worker",
        "resigned_worker",
        "source",
        "scope",
        "association",
    ],
)
def test_corruption_blocks_reporting_and_resume(tmp_path, damage):
    store, tasks, lock = fixture(tmp_path)
    complete_attempt(store, tasks, tasks[0], result_record(lock))
    path = store.path("input:" + tasks[0]["problem"]["id"])
    if damage == "input_missing":
        path.unlink()
    elif damage in ("counts", "kernel"):
        rewrite_hashed(
            path,
            lambda r: r["result"][
                "observed" if damage == "counts" else "kernel"
            ].__setitem__(0, 99),
        )
    elif damage == "inventory":
        (tmp_path / "input_inventory.json").write_text("{}")
    elif damage in ("attempt", "worker"):
        path = store.path(tasks[0]["id"])
        if damage == "worker":
            path = path.with_suffix(".worker.json")
        path.write_text("{}")
    elif damage == "resigned_worker":
        rewrite_hashed(
            store.path(tasks[0]["id"]).with_suffix(".worker.json"),
            lambda r: r["result"].__setitem__("centered_nll", 12),
        )
    elif damage in ("source", "scope"):
        lock["fingerprint"][damage] = "changed"
    else:
        tasks[0]["problem"]["id"] = "different"
    with pytest.raises(Error):
        shared.report_store(tasks, store)
    if damage != "resigned_worker":
        with pytest.raises(Error):
            store.execute_attempt(tasks[-1], tasks, forbid)


def test_unsealed_preparation_cannot_regenerate_or_start_workers(tmp_path):
    store, tasks, _ = fixture(tmp_path, prepare=False)

    def crash(*a):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.prepare_inputs(tasks, crash)
    with pytest.raises(Error):
        store.execute_attempt(tasks[0], tasks, forbid)
    with pytest.raises(Error):
        store.prepare_inputs(tasks, forbid)


def script_fixture(monkeypatch, store, tasks, lock):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["main"].__globals__
    monkeypatch.setitem(g, "require_authorization", lambda *a, **kw: lock)
    monkeypatch.setitem(g, "task_inventory", lambda *a, **kw: tasks)
    return (
        script,
        g,
        [
            "--audit-verification",
            "--manifest",
            str(MANIFEST),
            "--directory",
            str(store.directory),
            "--execution-lock",
            str(store.directory / "lock.json"),
        ],
    )


@pytest.mark.parametrize("failure", ["none", "timeout", "exception", "cleanup"])
def test_cli_mock_workers_keep_scope_timing_and_failed_attempts(
    tmp_path, monkeypatch, failure
):
    store, tasks, lock = fixture(tmp_path)
    script, g, args = script_fixture(monkeypatch, store, tasks, lock)
    calls = []

    def worker(command, ceiling):
        assert (
            "--audit-verification" in command
            and "--pilot" not in command
            and "--verification" not in command
        )
        calls.append(command)
        if failure == "timeout":
            return False
        if failure == "exception":
            raise RuntimeError("software worker failure")
        if failure == "cleanup":
            raise script["WorkerCleanupError"]("software unconfirmed termination")
        task = next(
            t for t in tasks if t["id"] == command[command.index("--task-id") + 1]
        )
        _, binding = store.worker_input(task, tasks)
        store.write_worker_result(task, tasks, result_record(lock), binding)
        return True

    monkeypatch.setitem(g, "bounded_worker", worker)
    if failure == "cleanup":
        with pytest.raises(script["WorkerCleanupError"]):
            script["main"](["run", *args])
        with pytest.raises(script["WorkerCleanupError"]):
            script["main"](["run", *args])
        assert len(calls) == 1
    else:
        script["main"](["run", *args])
        script["main"](["run", *args])
        assert len(calls) == 2
    report = shared.report_store(tasks, store)
    assert report["summary"]["scheduled_start_denominator"] == 4
    assert report["analyses"][0]["external_worker_timing"]["right_censored"] == (
        failure in ("timeout", "cleanup")
    )


def test_cli_scope_exclusive_and_inspect_nonexecuting(monkeypatch, capsys):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["main"].__globals__
    monkeypatch.setitem(g, "fingerprint", lambda *a, **kw: dict(scope=scope.SCOPE))
    for name in (
        "construct_problem",
        "execute_task",
        "report_store",
        "DevelopmentStore",
    ):
        monkeypatch.setitem(g, name, forbid)
    script["main"](["inspect", "--audit-verification"])
    result = json.loads(capsys.readouterr().out)
    assert (
        result["tasks"] == result["problems"] == 6
        and result["execution_authorized"] is False
    )
    for other in ("--pilot", "--verification"):
        with pytest.raises(SystemExit):
            script["main"](["inspect", "--audit-verification", other])


def test_pending_authorization_and_source_inventory(tmp_path, monkeypatch):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["fingerprint"].__globals__
    monkeypatch.setitem(
        g,
        "execution_environment",
        lambda: {
            "environment": manifest()["environment_controls"],
            "numerical_libraries": [{"num_threads": 1}],
        },
    )
    monkeypatch.setitem(
        g,
        "repository_identity",
        lambda paths: dict(commit="software", committed_sources=False),
    )
    fp = script["fingerprint"](manifest(), False, MANIFEST, audit_verification=True)
    assert (
        fp["scope"] == scope.SCOPE
        and fp["execution"]["reporting_model_call_limit"] == 0
    )
    assert (
        "src/tcspc_toolkit/benchmarks/_triexponential_audit_verification.py"
        in fp["source_lf_sha256"]
    )
    lock = dict(
        schema=shared.EXECUTION_SCHEMA, fingerprint=fp, execution_authorized=False
    )
    path = tmp_path / "pending.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        script["require_authorization"](
            path, manifest(), False, MANIFEST, audit_verification=True
        )
    lock.update(execution_authorized=True, maintainer_review="software")
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        script["require_authorization"](
            path, manifest(), False, MANIFEST, audit_verification=True
        )


def test_six_task_report_retains_all_144_scheduled_starts(tmp_path, monkeypatch):
    base = tiny_tasks()[0]["problem"]
    tasks = []
    for i in range(6):
        p = deepcopy(base)
        p.update(id=f"software-{i}:mean:matched", observation_id=f"software-{i}:mean")
        tasks.append(dict(id=p["id"] + ":audit", problem=p, configuration="audit"))
    lock = software_lock(tasks)
    lock["fingerprint"].update(scope=scope.SCOPE)
    lock["fingerprint"]["execution"]["settings"] = {
        "audit": asdict(AuditSettings(16, 8, 40000, 120))
    }
    store = shared.DevelopmentStore(tmp_path, lock, verify_execution=lambda: lock)
    store.prepare_inputs(tasks, tiny_input)
    for name in (
        "run_audit",
        "_fit_triexponential",
        "direct_expected",
        "multiexponential_reconvolution_expected_counts",
    ):
        monkeypatch.setattr(shared, name, forbid)
    report = shared.report_store(tasks, store)
    assert report["summary"]["all_attempt_denominator"] == 6
    assert report["summary"]["scheduled_start_denominator"] == 144
    assert report["summary"]["outcomes"] == {"missing_task": 6}
    assert report["summary"]["start_terminations"] == {
        "trf": {"unknown_worker_work": 96},
        "powell": {"unknown_worker_work": 48},
    }
    assert all(r["work"] is None for r in report["starts"])


@pytest.mark.parametrize("parent_status", ["timeout", "running"])
def test_retained_worker_payload_survives_parent_interruption(tmp_path, parent_status):
    store, tasks, lock = fixture(tmp_path)

    def operation(binding):
        store.worker_input(tasks[0], tasks)
        store.write_worker_result(tasks[0], tasks, result_record(lock), binding)
        if parent_status == "running":
            raise KeyboardInterrupt
        return dict(
            failure="external_worker_timeout",
            work_unknown=True,
            external_worker_timing=dict(right_censored=True),
        )

    if parent_status == "running":
        with pytest.raises(KeyboardInterrupt):
            store.execute_attempt(tasks[0], tasks, operation)
    else:
        store.execute_attempt(tasks[0], tasks, operation)
    report = shared.report_store(tasks, store)
    first = report["analyses"][0]
    assert first["result_available"] and first["retained_worker_result"] is not None
    assert not first["strict_complete"]
    assert first["outcome"] == (
        "running" if parent_status == "running" else "external_worker_timeout"
    )
    assert report["starts"][0]["attempted"] is True
    assert report["starts"][2]["attempted"] is None


@pytest.mark.parametrize(
    "foreign_scope",
    [fit_scope.VERIFICATION_SCOPE, "issue37_pr2c1b_numerical_verification_v2", "pilot"],
)
def test_cross_scope_checkpoint_cannot_be_imported(tmp_path, foreign_scope):
    current, tasks, lock = fixture(tmp_path / "current")
    foreign_lock = deepcopy(lock)
    foreign_lock["fingerprint"]["scope"] = foreign_scope
    foreign = shared.DevelopmentStore(
        tmp_path / "foreign", foreign_lock, verify_execution=lambda: foreign_lock
    )
    foreign.prepare_inputs(tasks, tiny_input)
    complete_attempt(foreign, tasks, tasks[0], result_record(foreign_lock))
    current.path(tasks[0]["id"]).write_bytes(foreign.path(tasks[0]["id"]).read_bytes())
    with pytest.raises(Error):
        shared.report_store(tasks, current)
    with pytest.raises(Error):
        current.execute_attempt(tasks[1], tasks, forbid)


def test_cli_preparation_seals_before_any_worker(tmp_path, monkeypatch):
    directory = tmp_path / "fresh"
    tasks = [t for t in tiny_tasks() if t["configuration"] == "audit"]
    lock = software_lock(tasks)
    lock["fingerprint"]["scope"] = scope.SCOPE
    lock["fingerprint"]["execution"].update(
        settings={"audit": asdict(AuditSettings(1, 1, 100, 10, validation_calls=20))},
        output_directory=str(directory),
    )
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["main"].__globals__
    monkeypatch.setitem(g, "require_authorization", lambda *a, **kw: lock)
    monkeypatch.setitem(g, "task_inventory", lambda *a, **kw: tasks)
    monkeypatch.setitem(
        g,
        "construct_problem",
        lambda m, p, observation=None, **kw: tiny_input(p, observation),
    )
    monkeypatch.setitem(g, "bounded_worker", forbid)
    args = [
        "--audit-verification",
        "--directory",
        str(directory),
        "--execution-lock",
        str(tmp_path / "software.json"),
    ]
    script["main"](["prepare", *args])
    store = shared.DevelopmentStore(
        directory, lock, create=False, verify_execution=lambda: lock
    )
    assert len(store.verify_inventory(tasks)["inputs"]) == 2
    with pytest.raises(Error, match="already started"):
        script["main"](["prepare", *args])


def test_cli_worker_dispatch_uses_bound_inputs_and_scope(tmp_path, monkeypatch):
    store, tasks, lock = fixture(tmp_path)
    script, g, args = script_fixture(monkeypatch, store, tasks, lock)

    def execute(task, arrays, **kwargs):
        assert task == tasks[0]
        assert kwargs == dict(pilot=False, audit_verification=True)
        assert len(arrays["observed"]) == 4
        return result_record(lock)

    monkeypatch.setitem(g, "execute_task", execute)

    def operation(binding):
        script["main"](["_worker", *args, "--task-id", tasks[0]["id"]])
        return store.read_worker_result(tasks[0], tasks, binding)

    store.execute_attempt(tasks[0], tasks, operation)
    assert shared.report_store(tasks, store)["summary"]["strict_complete"] == 1


@pytest.mark.parametrize("field", ["source", "manifest", "scope", "threads", "pools"])
def test_authorization_enforces_execution_and_environment(tmp_path, monkeypatch, field):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    fp = dict(
        repository=dict(committed_sources=True),
        source="correct",
        manifest="correct",
        scope=scope.SCOPE,
        environment=dict(
            environment=manifest()["environment_controls"],
            numerical_libraries=[dict(num_threads=1)],
        ),
    )
    lock = dict(
        schema=shared.EXECUTION_SCHEMA,
        execution_authorized=True,
        maintainer_review="software",
        fingerprint=deepcopy(fp),
    )
    if field == "threads":
        fp["environment"]["environment"]["OMP_NUM_THREADS"] = "2"
        lock["fingerprint"] = fp
    elif field == "pools":
        fp["environment"]["numerical_libraries"][0]["num_threads"] = 2
        lock["fingerprint"] = fp
    else:
        lock["fingerprint"][field] = "changed"
    path = tmp_path / "software.json"
    path.write_text(json.dumps(lock))
    monkeypatch.setitem(
        script["require_authorization"].__globals__, "fingerprint", lambda *a, **kw: fp
    )
    with pytest.raises(ValueError):
        script["require_authorization"](
            path, manifest(), False, MANIFEST, audit_verification=True
        )


def test_result_order_and_settings_mismatch_rejected(tmp_path):
    store, tasks, lock = fixture(tmp_path)
    r = result_record(lock)
    r["starts"][0]["solver"] = "powell"
    complete_attempt(store, tasks, tasks[0], r)
    with pytest.raises(Error, match="solver/order"):
        shared.report_store(tasks, store)
