"""Metadata-only real protocol checks; tiny fixtures exercise execution paths.

No test constructs V00-V02 arrays or starts an optimization experiment.
"""

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tcspc_toolkit.benchmarks import triexponential as shared
from tcspc_toolkit.benchmarks import _triexponential_verification as verification
from test_triexponential_integrity import (
    pack_input,
    rewrite_hashed,
    software_lock,
    tiny_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/issue37_pr2c1_numerical_verification_v1.json"
Error = shared.ExecutionIntegrityError


def manifest():
    return json.loads(MANIFEST.read_text())


def forbid(*args, **kwargs):
    pytest.fail("implicit generation, fitting or retry")


def test_exact_scope_inventory_without_numerical_work(monkeypatch):
    monkeypatch.setattr(shared, "construct_problem", forbid)
    monkeypatch.setattr(shared, "_fit_triexponential", forbid)
    monkeypatch.setattr(shared, "run_audit", forbid)
    m = manifest()
    tasks = shared.task_inventory(m, verification=True)
    problems = shared.problem_inventory(m, verification=True)
    assert len(tasks) == 12 and len(problems) == 6
    assert [t["id"] for t in tasks] == [t["id"] for t in m["tasks"]]
    assert {t["configuration"] for t in tasks} == {"lbfgsb:compact", "trf:compact"}
    for index, p in enumerate(problems):
        assert p["seed"] == (
            None if index % 2 == 0 else [37, 2, 20261101, 2, index // 2, 0]
        )
        assert tasks[2 * index]["problem"] == tasks[2 * index + 1]["problem"]
    specification = shared.execution_specification(m, verification=True)
    assert specification["settings"] == m["configurations"]
    assert specification["preparation_model_call_limit"] == 18
    assert specification["reporting_model_call_limit"] == 15
    assert specification["worker_deadlines"] == {
        "lbfgsb:compact": 60,
        "trf:compact": 60,
    }
    assert specification["serial_workers"] == 1
    for name in specification["settings"]:
        assert (
            shared.jsonable(shared.settings_for_task(name, pilot=True))
            == specification["settings"][name]
        )
    assert sum(p["seed"] is None for p in problems) == 3


@pytest.mark.parametrize(
    "change", ["id", "condition", "seed", "settings", "audit", "authorized"]
)
def test_scope_rejects_changed_contract(change):
    m = manifest()
    if change == "id":
        m["study_id"] += "-other"
    elif change == "condition":
        m["conditions"][0]["signal_photons"] += 1
    elif change == "seed":
        m["problems"][1]["rng_entropy"][3] = 0
    elif change == "settings":
        m["configurations"]["trf:compact"]["max_optimization_calls"] += 1
    elif change == "audit":
        m["tasks"][0]["configuration"] = "audit"
    else:
        m["execution_authorized"] = True
    with pytest.raises(ValueError, match="exact reviewed"):
        shared.task_inventory(m, verification=True)


def test_old_scopes_and_physical_metadata_unchanged():
    path = ROOT / "configs/issue37_pr2_development_v1.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "d9b33f2b88387dacb9e0edc76574616474c69d138d8eafadcda25fa22307f5aa"
    )
    old = json.loads(path.read_text())
    assert len(shared.task_inventory(old)) == 560
    pilot = shared.task_inventory(old, pilot=True)
    assert len(pilot) == 18 and sum(t["configuration"] == "audit" for t in pilot) == 6
    new = manifest()
    for previous, current in zip(
        shared.problem_inventory(old, pilot=True),
        shared.problem_inventory(new, verification=True),
        strict=True,
    ):
        for key in (
            "bins",
            "bin_width_ns",
            "time_origin_ns",
            "lifetimes_ns",
            "detected_signal_fractions",
            "signal_photons",
            "background_counts_per_bin",
            "shift_ns",
            "generating_irf",
            "lifetime_bounds_ns",
            "shift_bounds_ns",
        ):
            assert previous["condition"][key] == current["condition"][key]
        if current["seed"] is not None:
            assert previous["seed"][3] == 0 and current["seed"][3] == 2
    assert old["construction"]["observation_namespace"][3] == 1
    for name in new["irfs"]:
        assert new["irfs"][name] == old["irfs"][name]
    with pytest.raises(ValueError):
        shared.task_inventory(new)
    with pytest.raises(ValueError):
        shared.task_inventory(old, verification=True)
    with pytest.raises(ValueError):
        shared.task_inventory(new, pilot=True, verification=True)


@pytest.mark.parametrize(
    "config", ["audit", "trf:balanced", "lbfgsb:exhaustive", "other"]
)
def test_verification_never_dispatches_audit_or_other_policy(config, monkeypatch):
    monkeypatch.setattr(shared, "run_audit", forbid)
    with pytest.raises(ValueError):
        shared.settings_for_task(config, verification=True)


def fixture_input(problem, observation=None):
    p = dict(
        amplitudes=[1.0, 2.0, 3.0],
        lifetimes_ns=[0.5, 1.5, 3.0],
        background=0.1,
        shift=0.0,
    )
    time, kernel = [0, 0.25, 0.5, 0.75], [0, 2, 2, 0]
    mu = shared.multiexponential_reconvolution_expected_counts(
        time,
        kernel,
        amplitudes=p["amplitudes"],
        lifetimes=p["lifetimes_ns"],
        background=p["background"],
    )
    observed = mu if problem["seed"] is None else [0, 1, 3, 4]
    return pack_input(time, observed, kernel, truth=p)


def fixture_store(directory, *, prepare=True):
    tasks = [t for t in tiny_tasks() if t["configuration"] != "audit"]
    lock = software_lock(tasks)
    lock["fingerprint"]["scope"] = verification.VERIFICATION_SCOPE
    lock["fingerprint"]["execution"]["output_directory"] = str(directory)
    store = shared.DevelopmentStore(directory, lock, verify_execution=lambda: lock)
    if prepare:
        store.prepare_inputs(tasks, fixture_input)
    return store, tasks, lock


def fake_result(arrays, *, accepted=True):
    truth = arrays["truth"]
    p = {k: v for k, v in truth.items() if k != "shift"} | dict(
        temporal_shift_ns=truth["shift"]
    )
    mu = shared.multiexponential_reconvolution_expected_counts(
        arrays["time"],
        arrays["kernel"],
        amplitudes=p["amplitudes"],
        lifetimes=p["lifetimes_ns"],
        background=p["background"],
    )
    value = shared.calculate_poisson_deviance(arrays["observed"], mu) / 2
    return shared.jsonable(
        dict(
            parameters=p,
            expected_counts=mu,
            centered_nll=value,
            numerical=dict(
                accepted=accepted,
                consistent=True,
                required_stages_complete=True,
                validation=dict(candidate=p, passed=True),
                reasons=[] if accepted else ["software_rejection"],
            ),
            identifiability=dict(conditioning_status="software_fixture_only"),
            optimization=dict(
                starts=[],
                best_observed_centered_nll=value,
                recovery_starts_used=0,
                work=dict(
                    model_calls=0,
                    charged_component_convolutions=0,
                    total_seconds=0.001,
                    phases=dict(optimization=dict(model_calls=0)),
                ),
            ),
        )
    )


def complete(store, tasks, task, *, accepted=True):
    def operation(binding):
        arrays, actual_binding = store.worker_input(task, tasks)
        assert actual_binding == binding
        result = fake_result(arrays, accepted=accepted)
        store.write_worker_result(task, tasks, result, binding)
        return store.read_worker_result(task, tasks, binding)

    return store.execute_attempt(task, tasks, operation)


def test_sealed_paired_inputs_and_fit_only_report(tmp_path, monkeypatch):
    store, tasks, lock = fixture_store(tmp_path)
    inventory = store.verify_inventory(tasks)
    for index in (0, 2):
        a = store.task_binding(tasks[index], inventory)
        b = store.task_binding(tasks[index + 1], inventory)
        assert {
            k: v for k, v in a.items() if k not in ("task_id", "configuration")
        } == {k: v for k, v in b.items() if k not in ("task_id", "configuration")}
    for i, task in enumerate(tasks):
        complete(store, tasks, task, accepted=i != 1)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    monkeypatch.setattr(shared, "_fit_triexponential", forbid)
    monkeypatch.setattr(shared, "run_audit", forbid)
    report = shared.report_store(tasks, store)
    assert report["summary"]["all_attempt_denominator"] == 4
    assert report["summary"]["audit"]["status"] == "not_applicable"
    assert report["summary"]["outcomes"] == {"accepted": 3, "numerically_rejected": 1}
    assert report["reporting_work"]["model_calls"] == 5
    assert report["reporting_work"]["charged_component_convolutions"] == 15
    assert not report["summary"]["qualification"]
    for g in report["summary"]["groups"].values():
        assert g["denominator"] == 2 and "audit_complete" not in g and "gaps" not in g
    assert (
        report["summary"]["groups"]["trf:compact"]["accepted_only_recovery"][
            "contributors"
        ]
        == 1
    )
    for a in report["analyses"]:
        assert a["fresh_reporting_assessment"]["consistent"]
        if a["target_kind"] == "deterministic_expected_counts":
            assert (
                abs(a["deterministic_reference"]["returned_minus_generating"]) < 1e-20
            )
            assert not a["deterministic_reference"]["independent_optimizer_audit"]
        else:
            assert "deterministic_reference" not in a
    for _ in range(2):
        for task in tasks:
            store.execute_attempt(task, tasks, forbid)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}


@pytest.mark.parametrize(
    "damage",
    [
        "input_missing",
        "counts",
        "kernel",
        "inventory",
        "attempt",
        "transport",
        "resigned_transport",
        "association",
        "lock",
    ],
)
def test_invalid_evidence_blocks_reporting_and_resume(tmp_path, damage, monkeypatch):
    store, tasks, lock = fixture_store(tmp_path)
    complete(store, tasks, tasks[0])
    inp = store.path("input:" + tasks[0]["problem"]["id"])
    if damage == "input_missing":
        inp.unlink()
    elif damage in ("counts", "kernel"):
        key = "observed" if damage == "counts" else "kernel"
        rewrite_hashed(inp, lambda r: r["result"][key].__setitem__(0, 5.0))
    elif damage in ("inventory", "attempt", "transport"):
        path = (
            tmp_path / "input_inventory.json"
            if damage == "inventory"
            else store.path(tasks[0]["id"])
        )
        if damage == "transport":
            path = path.with_suffix(".worker.json")
        path.write_text("{}")
    elif damage == "resigned_transport":
        rewrite_hashed(
            store.path(tasks[0]["id"]).with_suffix(".worker.json"),
            lambda r: r["result"].__setitem__("centered_nll", 10),
        )
    elif damage == "association":
        tasks[0]["problem"]["id"] = "foreign"
    else:
        lock["fingerprint"]["source"] = "changed"
    monkeypatch.setattr(shared, "construct_problem", forbid)
    with pytest.raises(Error):
        shared.report_store(tasks, store)
    # A re-signed transport is caught specifically by report's result equality;
    # all other corruption fails the common inventory/resume gate as well.
    if damage != "resigned_transport":
        with pytest.raises(Error):
            store.execute_attempt(tasks[-1], tasks, forbid)


def test_incomplete_sealing_blocks_workers(tmp_path, monkeypatch):
    store, tasks, _ = fixture_store(tmp_path, prepare=False)

    def crash(problem, observation):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.prepare_inputs(tasks, crash)
    assert not (tmp_path / "input_inventory.json").exists()
    with pytest.raises(Error):
        store.execute_attempt(tasks[0], tasks, forbid)
    with pytest.raises(Error):
        store.prepare_inputs(tasks, forbid)


def script_for_fixture(monkeypatch, store, tasks, lock):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["main"].__globals__
    monkeypatch.setitem(g, "require_authorization", lambda *a, **kw: lock)
    monkeypatch.setitem(g, "task_inventory", lambda *a, **kw: tasks)
    monkeypatch.setattr(shared, "construct_problem", forbid)
    args = [
        "--verification",
        "--manifest",
        str(MANIFEST),
        "--directory",
        str(store.directory),
        "--execution-lock",
        str(store.directory / "lock.json"),
    ]
    return script, g, args


@pytest.mark.parametrize("outcome", ["timeout", "failure", "cleanup_failure"])
def test_worker_failures_are_censored_or_recorded_never_retried(
    tmp_path, monkeypatch, outcome
):
    store, tasks, lock = fixture_store(tmp_path)
    script, g, args = script_for_fixture(monkeypatch, store, tasks, lock)
    calls = []

    def worker(command, ceiling):
        calls.append(command)
        assert "--verification" in command and "--pilot" not in command
        if outcome == "failure":
            raise RuntimeError("synthetic worker failure")
        if outcome == "cleanup_failure":
            raise script["WorkerCleanupError"]("synthetic unconfirmed cleanup")
        return False

    monkeypatch.setitem(g, "bounded_worker", worker)
    if outcome == "cleanup_failure":
        with pytest.raises(script["WorkerCleanupError"]):
            script["main"](["run", *args])
        assert len(calls) == 1
    else:
        script["main"](["run", *args])
        assert len(calls) == 4
        script["main"](["run", *args])
        assert len(calls) == 4
    report = shared.report_store(tasks, store)
    assert report["summary"]["all_attempt_denominator"] == 4
    first = report["analyses"][0]
    assert first["external_worker_timing"]["right_censored"] == (outcome != "failure")
    if outcome != "failure":
        assert first["external_worker_timing"]["runtime_lower_bound_seconds"] == 0.01


def test_running_interrupted_and_missing_outcomes_remain_visible(tmp_path):
    store, tasks, _ = fixture_store(tmp_path)

    def interrupt(binding):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.execute_attempt(tasks[0], tasks, interrupt)
    store.execute_attempt(
        tasks[1], tasks, lambda b: dict(failure="worker_failed", work_unknown=True)
    )
    report = shared.report_store(tasks, store)
    assert report["summary"]["outcomes"] == {
        "running": 1,
        "worker_failed": 1,
        "missing_task": 2,
    }
    store.path(tasks[0]["id"]).unlink()
    assert shared.report_store(tasks, store)["summary"]["outcomes"]["interrupted"] == 1
    assert store.execute_attempt(tasks[0], tasks, forbid)["status"] == "interrupted"


def tiny_construction():
    m = dict(irfs=dict(sampled=dict(times_ns=[0, 0.25, "t_last"], values=[0, 1, 0])))
    c = dict(
        time_origin_ns=0,
        bins=4,
        bin_width_ns=0.25,
        lifetimes_ns=[0.5, 1.5, 3],
        generating_irf="sampled",
        shift_ns=0,
        signal_photons=12,
        detected_signal_fractions=[0.2, 0.3, 0.5],
        background_counts_per_bin=0.1,
    )
    p = dict(condition=c, assumed_irf="matched", seed=None)
    # Independent tiny reference setup, never a reviewed verification condition.
    original = shared.construct_problem(m, p)
    c["amplitudes"] = original["truth"]["amplitudes"]
    return m, p, original


def test_preparation_counts_calls_and_checks_declared_amplitudes_before_sampling(
    monkeypatch,
):
    m, p, original = tiny_construction()
    actual = shared.construct_problem(m, p, verification=True)
    np.testing.assert_array_equal(actual["mean"], original["mean"])
    assert actual["preparation_work"]["model_calls"] == 3
    assert actual["preparation_work"]["completed_component_convolutions"] == 3
    p["condition"]["amplitudes"] = [100, 100, 100]
    p["seed"] = [999, 7]
    monkeypatch.setattr(np.random, "Generator", forbid)
    with pytest.raises(Error, match="declared amplitudes") as error:
        shared.construct_problem(m, p, verification=True)
    assert error.value.verification_work["model_calls"] == 3


def test_preparation_failure_work_retained(tmp_path, monkeypatch):
    m, p, _ = tiny_construction()

    def fail(*a, **kw):
        raise FloatingPointError("software physical failure")

    monkeypatch.setattr(shared, "multiexponential_reconvolution_expected_counts", fail)
    store, tasks, _ = fixture_store(tmp_path, prepare=False)
    with pytest.raises(Error):
        store.prepare_inputs(
            tasks, lambda *_: shared.construct_problem(m, p, verification=True)
        )
    row = store.read("input:" + tasks[0]["problem"]["id"])
    assert row["status"] == "exception"
    assert row["preparation_work"]["model_calls"] == 1
    assert row["preparation_work"]["completed_component_convolutions"] == 0
    assert row["preparation_work"]["failures_with_unknown_partial_work"] == 1


def test_model_work_hard_limit():
    work = verification._Work(0, "software")
    with pytest.raises(Error, match="model-call limit"):
        work.model(amplitudes=[1])
    assert work.calls == 0


def test_fit_receives_no_truth_and_unchanged_compact_settings(monkeypatch):
    task = [t for t in tiny_tasks() if t["configuration"] == "trf:compact"][0]
    task["problem"]["condition"].update(
        lifetime_bounds_ns=[0.1, 10], shift_bounds_ns=[-0.3, 0.3]
    )
    arrays = fixture_input(task["problem"])
    arrays["truth"] = object()  # Must never be inspected by fitting dispatch.

    def fit(time, observed, kernel, **kwargs):
        assert "initial_guesses" not in kwargs and "truth" not in kwargs
        assert kwargs["target_kind"] == "deterministic_expected_counts"
        assert kwargs["settings"] == shared.settings_for_task("trf:compact", pilot=True)
        return "software-result"

    monkeypatch.setattr(shared, "_fit_triexponential", fit)
    monkeypatch.setattr(shared, "run_audit", forbid)
    assert shared.execute_task(task, arrays, verification=True) == "software-result"


def test_historical_hash_comparison_does_not_attribute_poisson_changes():
    entry = dict(arrays={"observed": dict(sha256="same")})
    baseline = dict(
        problems={
            "test": dict(
                arrays={"observed": "same"},
                fits={"lbfgsb:compact": dict(centered_nll=5)},
            )
        }
    )
    task = dict(problem=dict(id="test", seed=None), configuration="lbfgsb:compact")
    result = verification._history(task, entry, dict(centered_nll=4), baseline)
    assert result["status"] == "deterministic_identical_inputs"
    assert result["new_minus_original_centered_nll"] == -1
    entry["arrays"]["observed"]["sha256"] = "different"
    assert (
        verification._history(task, entry, {}, baseline)["status"]
        == "deterministic_input_mismatch"
    )
    task["problem"]["seed"] = [999]
    result = verification._history(task, entry, dict(centered_nll=4), baseline)
    assert result["status"] == "independent_poisson_draw"
    assert result["new_minus_original_centered_nll"] is None
    assert not result["controlled_poisson_before_after"]


def test_pending_gate_and_fingerprint_scope(tmp_path, monkeypatch):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["fingerprint"].__globals__
    monkeypatch.setitem(g, "execution_environment", lambda: {"software": True})
    monkeypatch.setitem(
        g,
        "repository_identity",
        lambda paths: dict(commit="software", committed_sources=False),
    )
    fp = script["fingerprint"](manifest(), False, MANIFEST, verification=True)
    assert fp["scope"] == verification.VERIFICATION_SCOPE
    assert (
        "src/tcspc_toolkit/benchmarks/_triexponential_verification.py"
        in fp["source_lf_sha256"]
    )
    pending = tmp_path / "pending.json"
    pending.write_text(
        json.dumps(
            dict(
                schema=shared.EXECUTION_SCHEMA,
                fingerprint=fp,
                execution_authorized=False,
            )
        )
    )
    with pytest.raises(ValueError, match="authorized matching"):
        script["require_authorization"](
            pending, manifest(), False, MANIFEST, verification=True
        )
    pending.write_text(
        json.dumps(
            dict(
                schema=shared.EXECUTION_SCHEMA,
                fingerprint=fp,
                execution_authorized=True,
                maintainer_review="test",
            )
        )
    )
    with pytest.raises(ValueError):
        script["require_authorization"](
            pending, manifest(), False, MANIFEST, verification=True
        )


def test_inspect_is_nonexecuting_and_scope_flags_exclusive(monkeypatch, capsys):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    g = script["main"].__globals__
    monkeypatch.setitem(
        g, "fingerprint", lambda *a, **kw: dict(scope=verification.VERIFICATION_SCOPE)
    )
    for name in (
        "construct_problem",
        "execute_task",
        "report_store",
        "DevelopmentStore",
    ):
        monkeypatch.setitem(g, name, forbid)
    script["main"](["inspect", "--verification"])
    out = json.loads(capsys.readouterr().out)
    assert (
        out["tasks"] == 12 and out["problems"] == 6 and not out["execution_authorized"]
    )
    with pytest.raises(SystemExit):
        script["main"](["inspect", "--pilot", "--verification"])


@pytest.mark.parametrize(
    "alter",
    ["source", "environment", "manifest", "scope", "settings", "threads", "pools"],
)
def test_verification_authorization_rejects_fingerprint_or_thread_mismatch(
    tmp_path, monkeypatch, alter
):
    script = runpy.run_path(str(ROOT / "scripts/develop_triexponential.py"))
    current = dict(
        repository=dict(committed_sources=True),
        source="correct",
        environment=dict(
            environment=manifest()["environment_controls"],
            numerical_libraries=[dict(num_threads=1)],
        ),
        manifest="correct",
        scope=verification.VERIFICATION_SCOPE,
        settings="correct",
    )
    lock = dict(
        schema=shared.EXECUTION_SCHEMA,
        fingerprint=json.loads(json.dumps(current)),
        execution_authorized=True,
        maintainer_review="software-fixture",
    )
    if alter == "threads":
        current["environment"]["environment"]["OMP_NUM_THREADS"] = "2"
        lock["fingerprint"] = current
    elif alter == "pools":
        current["environment"]["numerical_libraries"][0]["num_threads"] = 2
        lock["fingerprint"] = current
    else:
        lock["fingerprint"][alter] = "changed"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    monkeypatch.setitem(
        script["require_authorization"].__globals__,
        "fingerprint",
        lambda *a, **kw: current,
    )
    with pytest.raises(ValueError):
        script["require_authorization"](
            path, manifest(), False, MANIFEST, verification=True
        )


def test_successful_worker_transport_timing_and_wrong_directory_gate(
    tmp_path, monkeypatch
):
    store, tasks, lock = fixture_store(tmp_path)
    script, g, args = script_for_fixture(monkeypatch, store, tasks, lock)
    clock = [0.0]
    monkeypatch.setitem(g, "perf_counter", lambda: clock[0])
    original_read = shared.DevelopmentStore.read_worker_result

    def expensive_verification(self, *a, **kw):
        result = original_read(self, *a, **kw)
        clock[0] += 10
        return result

    monkeypatch.setattr(
        shared.DevelopmentStore, "read_worker_result", expensive_verification
    )

    def worker(command, ceiling):
        task = next(
            t for t in tasks if t["id"] == command[command.index("--task-id") + 1]
        )
        arrays, binding = store.worker_input(task, tasks)
        store.write_worker_result(task, tasks, fake_result(arrays), binding)
        clock[0] += 0.25
        return True

    monkeypatch.setitem(g, "bounded_worker", worker)
    script["main"](["run", *args])
    result = shared.report_store(tasks, store)
    assert result["summary"]["outcomes"] == {"accepted": 4}
    assert all(
        a["external_worker_timing"]["elapsed_including_cleanup_seconds"] == 0.25
        for a in result["analyses"]
    )
    lock["fingerprint"]["execution"]["output_directory"] = str(tmp_path / "wrong")
    with pytest.raises(Error, match="output directory"):
        script["main"](["run", *args])


def test_fresh_report_detects_numerical_inconsistency_without_rewriting_acceptance(
    tmp_path,
):
    store, tasks, _ = fixture_store(tmp_path)

    def operation(binding):
        arrays, _ = store.worker_input(tasks[0], tasks)
        result = fake_result(arrays)
        result["centered_nll"] += 1
        store.write_worker_result(tasks[0], tasks, result, binding)
        return result

    store.execute_attempt(tasks[0], tasks, operation)
    result = shared.report_store(tasks, store)
    analysis = result["analyses"][0]
    assert analysis["numerical"]["accepted"]
    assert not analysis["fresh_reporting_assessment"]["consistent"]
    assert not analysis["recovery"]["available"]
    assert analysis["fresh_reporting_assessment"]["signed_reported_objective_gap"] == -1
