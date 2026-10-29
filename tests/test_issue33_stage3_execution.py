"""Infrastructure fixtures only: no qualification draws or scientific fitting."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qualify_issue33_stage3 as runner
import report_issue33_stage3 as report


@pytest.fixture
def lock():
    protocol = json.loads(runner.CONFIG.read_text())
    # Aggregation fixtures use metadata, never qualification Poisson sampling.
    return dict(
        protocol=protocol,
        populations={m: runner.tasks(protocol, m) for m in ("qualification", "pilot")},
    )


@pytest.fixture
def store(tmp_path, lock):
    result = runner.Checkpoints(tmp_path / "fixtures", lock)
    result.put("authorization", runner.permit(lock, "pilot"))
    return result


def observation_task(lock):
    return lock["populations"]["pilot"][0]


def publish(store, task, payload, *, finished=1, deadline=2):
    key = runner.task_key(task)
    request = dict(task=task, coordinator_nonce="fixture", deadline_epoch=deadline)
    store.put("started-" + key, request)
    store.put(
        "transport-" + key,
        dict(
            request_sha256=runner.digest(request),
            result=payload,
            finished_epoch=finished,
        ),
    )
    return runner.adopt_transport(store, task)


def synthetic_observation(monkeypatch, store):
    # Exact deterministic integer fixture; no RNG.poisson call in the tests.
    monkeypatch.setattr(
        runner,
        "sample_photon_counts",
        lambda expected, rng: np.zeros_like(expected, dtype=int),
    )
    task = observation_task(store.lock)
    value = runner.generate_observation(
        store.lock, task, runner.permit(store.lock, "pilot")
    )
    publish(store, task, value)
    return value


def test_exact_task_counts_and_shared_observation_identity(lock):
    tasks = lock["populations"]["qualification"]
    assert [
        sum(t["kind"] == kind for t in tasks)
        for kind in ("observation", "production", "audit")
    ] == [200, 210, 133]
    assert len({runner.task_key(t) for t in tasks}) == 543
    for r in range(10):
        arms = [
            t
            for t in tasks
            if t["kind"] == "production"
            and t["condition_id"] == 26
            and t["replicate"] == r
        ]
        assert len(arms) == 2 and len({runner.observation_key(t) for t in arms}) == 1
    assert len(lock["populations"]["pilot"]) == 8
    assert {runner.task_key(t) for t in tasks}.isdisjoint(
        runner.task_key(t) for t in lock["populations"]["pilot"]
    )


def test_no_qualification_draw_before_authorization(monkeypatch, lock):
    monkeypatch.setattr(
        runner, "build_generating_case", lambda *a: pytest.fail("generator reached")
    )
    monkeypatch.setattr(np.random, "Generator", lambda *a: pytest.fail("RNG reached"))
    permission = runner.permit(lock, "pilot")
    with pytest.raises(ValueError, match="population"):
        runner.generate_observation(
            lock, lock["populations"]["qualification"][0], permission
        )
    with pytest.raises(PermissionError):
        runner.permit(lock, "qualification", {"scientific_execution_authorized": False})
    with pytest.raises(PermissionError):
        runner.generate_observation(
            lock,
            lock["populations"]["qualification"][0],
            dict(
                mode="qualification",
                lock_sha256=runner.digest(lock),
                scientific_execution_authorized=False,
            ),
        )


def test_freeze_and_verify_never_sample(monkeypatch, lock):
    monkeypatch.setattr(np.random, "Generator", lambda *a: pytest.fail("RNG reached"))
    monkeypatch.setattr(runner, "source_identity", lambda: {"fixture": "source"})
    monkeypatch.setattr(
        runner, "environment_identity", lambda: {"fixture": "environment"}
    )
    frozen = runner.make_lock(lock["protocol"])
    runner.verify_live_lock(frozen)
    assert frozen["scientific_execution_authorized"] is False
    monkeypatch.setattr(runner, "source_identity", lambda: {"fixture": "changed"})
    with pytest.raises(ValueError, match="lock mismatch"):
        runner.verify_live_lock(frozen)


def test_authorization_is_bound_to_exact_lock(lock):
    auth = dict(
        protocol=lock["protocol"]["protocol"],
        lock_sha256=runner.digest(lock),
        scientific_execution_authorized=True,
        approved_by="fixture",
        approved_utc="fixture",
    )
    assert runner.permit(lock, "qualification", auth)["mode"] == "qualification"
    changed = deepcopy(lock)
    changed["fixture"] = True
    with pytest.raises(PermissionError):
        runner.permit(changed, "qualification", auth)


def test_historical_and_other_destinations_rejected(lock, tmp_path):
    for path in (
        tmp_path,
        runner.ROOT / "data/generated/issue28_pr3_qualification_20261026",
    ):
        with pytest.raises(ValueError, match="distinct"):
            runner.ensure_destination(path, lock["protocol"], "qualification")


def test_os_lease_excludes_other_coordinators_and_releases(tmp_path):
    guard = tmp_path / "guard"
    with runner.exclusive_lease(guard):
        with pytest.raises(RuntimeError, match="active"):
            with runner.exclusive_lease(guard):
                pytest.fail("overlapping lease")
    with runner.exclusive_lease(guard):
        pass


@pytest.mark.parametrize("status", sorted(runner.TERMINAL))
def test_terminal_results_never_retried(store, status):
    task = observation_task(store.lock)
    publish(store, task, runner.failure(status, "fixture"))

    def forbidden(*args):
        pytest.fail("terminal task rerun")

    actual = runner.execute_task(
        store, task, "pilot", "new", Path("lock"), Path("auth"), launch=forbidden
    )
    assert actual["status"] == status


def test_orphan_without_transport_is_terminal_interrupted(store):
    task = observation_task(store.lock)
    store.put("started-" + runner.task_key(task), dict(task=task, deadline_epoch=0))
    first = runner.recover_started(store, task)
    assert first["status"] == "interrupted" and first["counters_available"] is False
    assert runner.recover_started(store, task) == first


def test_durable_result_recovered_without_resampling(store):
    task = observation_task(store.lock)
    key = runner.task_key(task)
    request = dict(task=task, deadline_epoch=10)
    store.put("started-" + key, request)
    store.put(
        "transport-" + key,
        dict(
            request_sha256=runner.digest(request),
            result=runner.failure("worker_error", "fixture"),
            finished_epoch=9,
        ),
    )
    assert runner.recover_started(store, task)["status"] == "worker_error"


def test_late_transport_cannot_be_adopted_as_success(store):
    actual = publish(
        store, observation_task(store.lock), dict(status="completed"), finished=3
    )
    assert actual["status"] == "timeout"


def test_transport_wrong_request_rejected(store):
    task = observation_task(store.lock)
    key = runner.task_key(task)
    store.put("started-" + key, dict(task=task, deadline_epoch=2))
    store.put(
        "transport-" + key,
        dict(request_sha256="wrong", result=dict(status="completed")),
    )
    with pytest.raises(ValueError, match="mismatch"):
        runner.recover_started(store, task)


def test_tampered_checkpoint_and_changed_lock_rejected(store):
    task = observation_task(store.lock)
    publish(store, task, runner.failure("timeout", "fixture"))
    key = runner.task_key(task)
    path = store.directory / (key + ".json")
    value = json.loads(path.read_text())
    value["payload"]["status"] = "completed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="corrupt"):
        store.get(key)
    with pytest.raises(ValueError, match="lock differs"):
        runner.Checkpoints(store.directory, dict(store.lock, changed=True))


def test_wall_timeout_kills_worker_and_reaps(tmp_path):
    status, seconds = runner.launch_worker(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        0.2,
        tmp_path / "timeout.log",
    )
    assert status == "timeout" and seconds < 10


def test_parent_timeout_is_terminal_even_with_partial_transport(store):
    task = observation_task(store.lock)

    def timed_out(*args):
        return "timeout", 300.01

    actual = runner.execute_task(
        store, task, "pilot", "epoch", Path("lock"), Path("auth"), launch=timed_out
    )
    assert actual["status"] == "timeout" and actual["counters_available"] is False
    assert store.get("timing-" + runner.task_key(task))["limit_seconds"] == 300


def test_worker_stale_epoch_cannot_compute(monkeypatch, store):
    task = observation_task(store.lock)
    (store.directory / ".coordinator-owner.json").write_text('{"nonce":"new"}')
    monkeypatch.setattr(
        runner, "compute_task", lambda *a: pytest.fail("stale computation")
    )
    with pytest.raises(ValueError, match="stale"):
        runner._worker_result(
            store,
            task,
            dict(coordinator_nonce="old"),
            runner.permit(store.lock, "pilot"),
        )


def test_resume_keeps_terminal_error_and_continues_never_started_work(
    monkeypatch, tmp_path, lock
):
    lock["populations"]["pilot"] = [
        t for t in lock["populations"]["pilot"] if t["kind"] == "observation"
    ]
    args = SimpleNamespace(
        directory=tmp_path / "resume",
        mode="pilot",
        lock=Path("lock"),
        authorization=Path("auth"),
    )
    permission = runner.permit(lock, "pilot")
    monkeypatch.setattr(runner, "ensure_destination", lambda *a: None)
    starts = []

    def execute(store, task, *args):
        previous = runner.recover_started(store, task)
        if previous is not None:
            return previous
        starts.append(runner.task_key(task))
        return publish(
            store,
            task,
            runner.failure(
                "worker_error" if len(starts) == 1 else "timeout", "fixture"
            ),
        )

    monkeypatch.setattr(runner, "execute_task", execute)
    with pytest.raises(RuntimeError, match="investigation"):
        runner.coordinate(args, lock, permission)
    runner.coordinate(args, lock, permission)
    runner.coordinate(args, lock, permission)
    assert len(starts) == len(set(starts)) == 2


def test_fitting_requires_immutable_saved_observation(store):
    task = store.lock["populations"]["pilot"][1]
    with pytest.raises(ValueError, match="checkpointed"):
        runner.load_inputs(store, task)


def test_failure_of_generation_preserves_fit_identity(store):
    publish(
        store, observation_task(store.lock), runner.failure("interrupted", "fixture")
    )
    task = store.lock["populations"]["pilot"][1]
    assert (
        runner.compute_task(store, task, runner.permit(store.lock, "pilot"))["status"]
        == "observation_unavailable"
    )


def test_exact_zero_fixture_full_gate_and_separate_calls(monkeypatch, store):
    synthetic_observation(monkeypatch, store)
    task = store.lock["populations"]["pilot"][1]
    value = runner.compute_task(store, task, runner.permit(store.lock, "pilot"))
    assert value["assessment"].accepted
    assert value["optimization"].stopping_reason == "analytical_zero_counts"
    assert value["assessment"].external_forward_evaluations == 1
    assert value["input_verification_forward_calls"] == 2
    assert value["optimization"].objective_evaluations == 0


def test_incomplete_required_stage_is_rejected_even_with_local_validation(
    monkeypatch, store
):
    synthetic_observation(monkeypatch, store)
    from dataclasses import replace

    actual = runner.fit_experimental_biexponential_reconvolution

    def incomplete(*args, **kwargs):
        fit = actual(*args, **kwargs).fit
        return SimpleNamespace(
            fit=replace(
                fit,
                optimization=replace(
                    fit.optimization,
                    numerical_validation_passed=True,
                    stopping_reason="start_budget_exhausted",
                    exhausted_budgets=("starts",),
                ),
            )
        )

    monkeypatch.setattr(
        runner, "fit_experimental_biexponential_reconvolution", incomplete
    )
    value = runner.compute_task(
        store, store.lock["populations"]["pilot"][1], runner.permit(store.lock, "pilot")
    )
    assert value["optimization"].numerical_validation_passed is True
    assert value["assessment"].accepted is False
    assert "required_stage_budget_exhausted" in value["assessment"].reasons
    assert value[
        "production_objective_usable"
    ]  # Finite rejected fit can still be audited.


def test_independent_audit_hard_call_cap(monkeypatch, store):
    synthetic_observation(monkeypatch, store)
    store.lock["protocol"]["audit"]["max_forward_calls"] = 1
    task = next(t for t in store.lock["populations"]["pilot"] if t["kind"] == "audit")
    value = runner.compute_task(store, task, runner.permit(store.lock, "pilot"))
    assert value["forward_calls"] == 1 and value["complete"] is False
    assert "forward_calls" in value["exhausted_budgets"]
    assert value["shared_forward_recheck_calls"] == 1


@pytest.mark.parametrize(
    "delta,complete,usable,expected",
    [
        (0.01, True, True, "agreement"),
        (0.01001, True, True, "disagreement"),
        (0, False, True, "audit_incomplete"),
        (0, True, False, "production_unavailable"),
        (1, False, False, "audit_incomplete"),
    ],
)
def test_audit_facts_are_separate(lock, delta, complete, usable, expected):
    production = dict(
        production_objective_usable=usable,
        assessment=dict(recomputed_centered_nll=delta),
    )
    audit = dict(status="completed", complete=complete, shared_centered_nll=0)
    row = report.audit_comparison(production, audit, lock["protocol"])
    assert row["classification"] == expected
    assert row["comparison_available"] == usable
    assert row["audit_complete"] == complete


def fixture_population(lock):
    records = {}
    for t in lock["populations"]["qualification"]:
        r = dict(status="completed", counters_available=False)
        if t["kind"] == "production":
            r.update(
                assessment=dict(
                    accepted=True,
                    diagnostic_budget_complete=True,
                    recomputed_centered_nll=10,
                ),
                production_objective_usable=True,
                recovery=dict(
                    accepted_component_relative_errors=[0, 0],
                    all_finite_component_relative_errors=[0, 0],
                ),
            )
        elif t["kind"] == "audit":
            r.update(complete=True, shared_centered_nll=10)
        records[runner.task_key(t)] = r
    return records


def test_joint_gate_all_attempt_denominators_and_reproducibility(lock):
    lock["protocol"]["acceptance"]["bootstrap_replicates"] = (
        2  # Metadata fixtures only.
    )
    records = fixture_population(lock)
    complete = report.aggregate(lock, "qualification", records)
    assert complete["primary_gates"]["overall_pass"]
    assert complete == report.aggregate(lock, "qualification", records)
    key = runner.task_key(
        next(t for t in lock["populations"]["qualification"] if t["kind"] == "audit")
    )
    records[key] = runner.failure("timeout", "fixture")
    incomplete = report.aggregate(lock, "qualification", records)
    assert incomplete["primary_gates"]["numerical_acceptance"]
    assert not incomplete["primary_gates"]["overall_pass"]
    assert incomplete["audit_tally"]["populations"]["primary"] == dict(
        agreement=99,
        disagreement=0,
        audit_incomplete=1,
        production_unavailable=0,
        not_run=0,
    )
    assert incomplete["unavailable_audit_comparisons"] == 1
    del records[key]
    missing = report.aggregate(lock, "qualification", records)
    assert missing["audit_tally"]["populations"]["primary"]["not_run"] == 1
    assert not missing["primary_gates"]["complete_untampered_population"]


@pytest.mark.parametrize(
    "accepted,errors,expected",
    [
        (95, [0.049, -0.049], (True, True, False)),
        (94, [0, 0], (False, True, True)),
        (100, [0.05, 0.05], (True, False, False)),
        (100, [0.0200001, 0.0200001], (True, True, False)),
        (100, [None, None], (True, False, False)),
    ],
)
def test_primary_criteria_independent_and_strict(lock, accepted, errors, expected):
    lock["protocol"]["acceptance"]["bootstrap_replicates"] = 1
    records = fixture_population(lock)
    for t in lock["populations"]["qualification"]:
        if t["kind"] == "production" and t["condition_id"] == 0:
            r = records[runner.task_key(t)]
            r["assessment"]["accepted"] = t["replicate"] < accepted
            r["recovery"]["accepted_component_relative_errors"] = (
                errors if r["assessment"]["accepted"] else [None, None]
            )
    gates = report.aggregate(lock, "qualification", records)["primary_gates"]
    assert (
        gates["numerical_acceptance"],
        all(gates["median_absolute_relative_errors"]),
        all(gates["absolute_mean_signed_relative_biases"]),
    ) == expected
    assert gates["complete_usable_audit_agreement_100_of_100"]


def test_exact_reported_bias_boundary_is_exclusive(monkeypatch, lock):
    def summary(*args):
        return dict(
            n_accepted=100,
            components=[
                dict(median_absolute_relative_error=0.01, signed_relative_bias=0.02),
                dict(median_absolute_relative_error=0.01, signed_relative_bias=-0.02),
            ],
        )

    monkeypatch.setattr(report, "summarize_recovery", summary)
    gates = report.aggregate(lock, "qualification", fixture_population(lock))[
        "primary_gates"
    ]
    assert gates["absolute_mean_signed_relative_biases"] == [False, False]


def test_evidence_reader_is_reproducible_and_rejects_foreign_files(monkeypatch, store):
    synthetic_observation(monkeypatch, store)
    first = report.read_evidence(store.directory, store.lock, "pilot")
    assert first == report.read_evidence(store.directory, store.lock, "pilot")
    store.put("foreign", dict(status="completed"))
    with pytest.raises(ValueError, match="foreign"):
        report.read_evidence(store.directory, store.lock, "pilot")


def test_evidence_pair_hash_mismatch_rejected(monkeypatch, store):
    synthetic_observation(monkeypatch, store)
    task = store.lock["populations"]["pilot"][1]
    publish(store, task, dict(status="completed", observation_sha256="incorrect"))
    with pytest.raises(ValueError, match="bound"):
        report.read_evidence(store.directory, store.lock, "pilot")


def test_reporting_requires_orphan_recovery_and_counts_terminal_tasks(store):
    task = observation_task(store.lock)
    key = runner.task_key(task)
    store.put("started-" + key, dict(task=task, deadline_epoch=0))
    with pytest.raises(ValueError, match="resume recovery"):
        report.read_evidence(store.directory, store.lock, "pilot")
    runner.recover_started(store, task)
    records, timings, _ = report.read_evidence(store.directory, store.lock, "pilot")
    summary = report.aggregate(store.lock, "pilot", records, timings)
    assert summary["statuses"]["interrupted"] == 1
    assert summary["statuses"]["not_run"] == 7
    assert all(r["n_terminal_tasks"] == 0 for r in summary["recovery"])
