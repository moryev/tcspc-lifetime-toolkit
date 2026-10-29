"""Stage-3 execution infrastructure; a frozen lock is NOT permission to draw.

freeze creates an execution identity and a separate DENIED authorization file.
run requires explicit authorization bound to that identity. pilot is restricted
to two nonqualification observations (stream 2), three fits and three audits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, time
from threading import Timer
import traceback
import uuid

import numpy as np
import tcspc_toolkit
from threadpoolctl import threadpool_info, threadpool_limits

from qualify_biexponential_reconvolution import (
    Checkpoints,
    digest,
    jsonable,
    source_digest,
    write_new_json,
)
from validate_issue33_stage3_protocol import CONFIG, ROOT, population, validate
from tcspc_toolkit.benchmarks.multiexponential import (
    assess_biexponential_fit,
    build_assumed_irf,
    build_generating_case,
    component_recovery,
    independent_biexponential_audit,
)
from tcspc_toolkit.estimation.multiexponential import BiexponentialFitSettings
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.measurement.adapters import (
    fit_experimental_biexponential_reconvolution,
)
from tcspc_toolkit.measurement.records import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
)
from tcspc_toolkit.simulation.histograms import sample_photon_counts


LOCK = ROOT / "docs/validation/issue33_stage3_execution_lock_v1.json"
AUTHORIZATION = ROOT / "docs/validation/issue33_stage3_execution_authorization_v1.json"
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
TERMINAL = {
    "completed",
    "timeout",
    "interrupted",
    "worker_error",
    "observation_unavailable",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def tasks(protocol, mode):
    if mode not in ("qualification", "pilot"):
        raise ValueError("unknown execution mode")
    obs, prod, audits = population(protocol)
    if mode == "pilot":
        obs = [(0, 0), (26, 0)]
        prod = [p for p in prod if p[:2] in obs]
        audits = list(prod)
    stream = 0 if mode == "qualification" else 2
    result = []
    for c, r in obs:
        result.append(
            dict(kind="observation", condition_id=c, replicate=r, stream=stream)
        )
        for kind, members in (("production", prod), ("audit", audits)):
            for cid, rep, arm in members:
                if (cid, rep) == (c, r):
                    result.append(
                        dict(
                            kind=kind,
                            condition_id=c,
                            replicate=r,
                            stream=stream,
                            arm=arm,
                        )
                    )
    return result


def task_key(task):
    key = f"{task['kind']}-s{task['stream']}-c{task['condition_id']:02d}-r{task['replicate']:03d}"
    return key + ("-" + task["arm"].replace(".", "_") if "arm" in task else "")


def observation_key(task):
    return task_key(
        {k: v for k, v in dict(task, kind="observation").items() if k != "arm"}
    )


def environment_identity():
    binaries = {}
    for name in ("numpy", "scipy", "threadpoolctl"):
        dist = metadata.distribution(name)
        for path in dist.files or []:
            if str(path).lower().endswith((".dll", ".pyd", ".so", ".py")):
                binaries[name + ":" + path.as_posix()] = hashlib.sha256(
                    Path(dist.locate_file(path)).read_bytes()
                ).hexdigest()
    with threadpool_limits(limits=1):
        pools = threadpool_info()
    return dict(
        python=sys.version,
        executable=sys.executable,
        executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        platform=platform.platform(),
        machine=platform.machine(),
        packages=sorted(
            (d.metadata["Name"], d.version)
            for d in metadata.distributions()
            if d.metadata["Name"]
        ),
        numerical_dependency_sha256=binaries,
        worker_environment=THREAD_ENV,
        numerical_threads=1,
        numerical_libraries=pools,
        workers=1,
    )


def source_identity():
    files = sorted((ROOT / "src").rglob("*.py")) + [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/report_issue33_stage3.py",
        ROOT / "scripts/validate_issue33_stage3_protocol.py",
        ROOT / "scripts/qualify_biexponential_reconvolution.py",
        ROOT / "configs/issue28_pr3_qualification.json",
        ROOT / "tests/test_issue33_stage3_execution.py",
        ROOT / "tests/test_issue33_stage3_protocol.py",
    ]
    return {p.relative_to(ROOT).as_posix(): source_digest(p) for p in files}


def make_lock(protocol):
    validate(protocol)
    return jsonable(
        dict(
            schema=1,
            protocol=protocol,
            implementation_commit=protocol["implementation_commit"],
            source_lf_sha256=source_identity(),
            environment=environment_identity(),
            populations={
                mode: tasks(protocol, mode) for mode in ("qualification", "pilot")
            },
            scientific_execution_authorized=False,
        )
    )


def verify_live_lock(lock):
    if (
        Path(tcspc_toolkit.__file__).resolve()
        != (ROOT / "src/tcspc_toolkit/__init__.py").resolve()
    ):
        raise ValueError(
            "execution must import the locked checkout, not another installation"
        )
    if make_lock(lock["protocol"]) != lock:
        raise ValueError("execution source/settings/environment lock mismatch")


def permit(lock, mode, authorization=None):
    if mode == "pilot":
        return dict(
            mode=mode, lock_sha256=digest(lock), scientific_execution_authorized=False
        )
    if mode != "qualification" or authorization is None:
        raise PermissionError("scientific execution requires separate authorization")
    if (
        authorization.get("scientific_execution_authorized") is not True
        or authorization.get("lock_sha256") != digest(lock)
        or authorization.get("protocol") != lock["protocol"]["protocol"]
        or not authorization.get("approved_by")
        or not authorization.get("approved_utc")
    ):
        raise PermissionError("scientific execution not authorized for this exact lock")
    return dict(authorization, mode=mode)


def validate_task(lock, mode, task, permission):
    if permission.get("mode") != mode or permission.get("lock_sha256") != digest(lock):
        raise PermissionError("task permission/identity mismatch")
    if mode == "qualification" and not permission.get(
        "scientific_execution_authorized"
    ):
        raise PermissionError("scientific draws are disabled")
    if task not in lock["populations"][mode]:
        raise ValueError("task outside frozen population")


@contextmanager
def exclusive_lease(path):
    """An OS lock, released by process death; never unlink the guard file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another coordinator or worker is active") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def failure(status, reason):
    return dict(status=status, reason=reason, counters_available=False)


def generate_observation(lock, task, permission):
    validate_task(
        lock, permission["mode"], task, permission
    )  # Before RNG construction.
    protocol = lock["protocol"]
    case = build_generating_case(protocol, task["condition_id"])
    entropy = [
        *protocol["root_entropy"],
        task["stream"],
        task["condition_id"],
        task["replicate"],
    ]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    counts = sample_photon_counts(case.expected_counts, rng)
    return dict(
        status="completed",
        time_ns=case.irf.time_ns,
        counts=counts,
        sample_id=f"{protocol['protocol']}-{observation_key(task)}",
        entropy=entropy,
        protocol=protocol["protocol"],
        scope=permission["mode"],
        counts_sha256=digest(counts),
        time_sha256=digest(case.irf.time_ns),
        expected_counts_sha256=digest(case.expected_counts),
        generating_kernel_sha256=digest(case.irf.kernel),
        configuration=case.configuration,
        generation_forward_calls=2,
        counters_available=True,
    )


def load_inputs(store, task):
    observation = store.get(observation_key(task))
    if observation is None:
        raise ValueError("observation must be checkpointed before fitting")
    if observation["status"] != "completed":
        return None
    protocol = store.lock["protocol"]
    case = build_generating_case(protocol, task["condition_id"])
    if (
        digest(observation["counts"]) != observation["counts_sha256"]
        or digest(observation["time_ns"]) != observation["time_sha256"]
        or digest(case.irf.time_ns) != observation["time_sha256"]
        or digest(case.expected_counts) != observation["expected_counts_sha256"]
        or digest(case.irf.kernel) != observation["generating_kernel_sha256"]
        or jsonable(case.configuration) != observation["configuration"]
        or observation["protocol"] != protocol["protocol"]
        or observation["scope"] != ("qualification" if task["stream"] == 0 else "pilot")
        or observation["sample_id"] != f"{protocol['protocol']}-{observation_key(task)}"
        or observation["entropy"]
        != [
            *protocol["root_entropy"],
            task["stream"],
            task["condition_id"],
            task["replicate"],
        ]
    ):
        raise ValueError("observation/generator hash or identity mismatch")
    measurement = TCSPCMeasurement(
        observation["time_ns"],
        observation["counts"],
        MeasurementDataKind.RAW_COUNTS,
        sample_id=observation["sample_id"],
        provenance=dict(
            protocol=protocol["protocol"],
            seed_entropy=observation["entropy"],
            scope=observation["scope"],
            origin="independent_synthetic",
        ),
    )
    assumed = (
        case.irf
        if task["arm"] == "matched"
        else build_assumed_irf(protocol, case.irf.time_ns, task["arm"])
    )
    return case, measurement, assumed, observation


def feasible_parameters(p, c):
    if p is None:
        return False
    values = [
        *p["amplitudes"],
        *p["lifetimes_ns"],
        p["background"],
        p["temporal_shift_ns"],
    ]
    lo, hi = c["lifetime_bounds_ns"]
    slo, shi = c["temporal_shift_bounds_ns"]
    return bool(
        np.all(np.isfinite(values))
        and min(p["amplitudes"]) >= 0
        and p["background"] >= 0
        and all(lo <= t <= hi for t in p["lifetimes_ns"])
        and slo <= p["temporal_shift_ns"] <= shi
    )


def compute_task(store, task, permission):
    validate_task(store.lock, permission["mode"], task, permission)
    if task["kind"] == "observation":
        return generate_observation(store.lock, task, permission)
    inputs = load_inputs(store, task)
    if inputs is None:
        return failure(
            "observation_unavailable", "saved generation task did not complete"
        )
    case, measurement, assumed, observation = inputs
    protocol, c = store.lock["protocol"], case.configuration
    if task["kind"] == "production":
        started = perf_counter()
        fitted = fit_experimental_biexponential_reconvolution(
            measurement,
            prepared_irf=assumed,
            lifetime_bounds_ns=tuple(c["lifetime_bounds_ns"]),
            temporal_shift_bounds_ns=tuple(c["temporal_shift_bounds_ns"]),
            settings=BiexponentialFitSettings(**protocol["fit_settings"]),
        )
        result = fitted.fit
        fit_seconds = perf_counter() - started
        started = perf_counter()
        assessment = assess_biexponential_fit(
            result,
            measurement.time_ns,
            measurement.values,
            assumed.kernel,
            protocol["acceptance"],
        )
        unusable = {
            "parameters_or_expectation_unavailable",
            "infeasible_parameters",
            "objective_recheck_failed",
            "curve_recheck_failed",
        }
        output = dict(
            parameters=result.parameters,
            optimization=result.optimization,
            assumptions=result.assumptions,
            statistics=result.statistics,
            identifiability=result.identifiability,
            assessment=assessment,
            recovery=component_recovery(case, result, assessment.accepted),
            production_objective_usable=not unusable.intersection(assessment.reasons),
            fit_seconds=fit_seconds,
            assessment_seconds=perf_counter() - started,
            counters_available=True,
        )
    elif task["kind"] == "audit":
        output = independent_biexponential_audit(
            measurement.time_ns,
            measurement.values,
            assumed.kernel,
            c["lifetime_bounds_ns"],
            c["temporal_shift_bounds_ns"],
            protocol["audit"],
        )
        p = output["parameters"]
        output.update(shared_centered_nll=None, shared_forward_recheck_calls=0)
        if feasible_parameters(p, c):
            output["shared_forward_recheck_calls"] = 1
            mu = (
                multiexponential_reconvolution_component_counts(
                    measurement.time_ns,
                    assumed.kernel,
                    amplitudes=p["amplitudes"],
                    lifetimes=p["lifetimes_ns"],
                    temporal_shift=p["temporal_shift_ns"],
                ).sum(axis=0)
                + p["background"]
            )
            value = 0.5 * calculate_poisson_deviance(measurement.values, mu)
            if np.isfinite(value):
                output["shared_centered_nll"] = float(value)
        output["complete"] = bool(
            output["complete"]
            and not output["exhausted_budgets"]
            and all(
                np.isfinite(s.get("centered_nll", np.inf)) for s in output["starts"]
            )
            and output["shared_centered_nll"] is not None
        )
        output["counters_available"] = True
    else:
        raise ValueError("unknown task kind")
    return dict(
        output,
        status="completed",
        observation_sha256=observation["counts_sha256"],
        assumed_kernel_sha256=digest(assumed.kernel),
        input_verification_forward_calls=2,
    )


def adopt_transport(store, task):
    key = task_key(task)
    request = store.get("started-" + key)
    transport = store.get("transport-" + key)
    if transport is None:
        return None
    if (
        request is None
        or request["task"] != task
        or transport["request_sha256"] != digest(request)
    ):
        raise ValueError("worker transport/request mismatch")
    if transport["finished_epoch"] > request["deadline_epoch"]:
        result = failure(
            "timeout", "durable worker result exceeded the original deadline"
        )
        store.put(key, result)
        return result
    result = transport["result"]
    if result.get("status") not in TERMINAL:
        raise ValueError("unknown terminal worker status")
    store.put(key, result)
    return result


def recover_started(store, task):
    """Caller holds coordinator lease and has established worker quiescence."""
    key = task_key(task)
    previous = store.get(key)
    if previous is not None:
        return previous
    if store.get("started-" + key) is None:
        return None
    result = adopt_transport(store, task)
    if result is None:
        result = failure(
            "interrupted", "started request without a completed worker result; no retry"
        )
        store.put(key, result)
    return result


def launch_worker(command, timeout, log_path):
    """Wall limit includes interpreter startup, identity checks and rechecks."""
    started = perf_counter()
    with Path(log_path).open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **THREAD_ENV},
            close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            code = process.wait(timeout=timeout)
            status = (
                "completed"
                if code == 0
                else "timeout"
                if code == 124
                else "interrupted"
            )
            return status, perf_counter() - started
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return "timeout", perf_counter() - started
        except BaseException:
            process.kill()
            process.wait()
            raise


def execute_task(
    store, task, mode, nonce, lock_path, authorization_path, launch=launch_worker
):
    key = task_key(task)
    previous = recover_started(store, task)
    if previous is not None:
        return previous
    limit = store.lock["protocol"]["execution"][
        "audit_timeout_seconds"
        if task["kind"] == "audit"
        else "production_case_timeout_seconds"
    ]
    deadline = time() + limit
    store.put(
        "started-" + key,
        dict(
            task=task,
            coordinator_nonce=nonce,
            started_utc=utc(),
            limit_seconds=limit,
            deadline_epoch=deadline,
        ),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--mode",
        mode,
        "--directory",
        str(store.directory),
        "--lock",
        str(lock_path),
        "--authorization",
        str(authorization_path),
        "--task",
        key,
    ]
    status, seconds = launch(
        command, max(0, deadline - time()), store.directory / (key + ".worker.log")
    )
    if status == "completed":
        result = adopt_transport(store, task)
        if result is None:
            result = failure("interrupted", "worker exited without a durable result")
            store.put(key, result)
    else:
        result = failure(
            status,
            "worker wall limit" if status == "timeout" else "unexpected worker exit",
        )
        store.put(key, result)
    store.put("timing-" + key, dict(wall_seconds=seconds, limit_seconds=limit))
    return result


def open_existing(directory, lock):
    path = Path(directory)
    if not (path / "lock.json").exists():
        raise ValueError("existing checkpoint store required")
    return Checkpoints(path, lock)


def ensure_destination(directory, protocol, mode):
    expected = ROOT / (
        protocol["outputs"]["raw_directory"]
        if mode == "qualification"
        else "data/generated/issue33_stage3_pilot_v1"
    )
    if Path(directory).resolve() != expected.resolve():
        raise ValueError(
            "execution output must be the distinct declared Stage-3 directory"
        )


def worker(args, lock, permission):
    store = open_existing(args.directory, lock)
    task = next(t for t in lock["populations"][args.mode] if task_key(t) == args.task)
    request = store.get("started-" + args.task)
    if request is None or request["task"] != task or store.get(args.task) is not None:
        raise ValueError("worker requires its original pending request")
    # The child also enforces the original deadline if its coordinator dies.
    watchdog = Timer(max(0, request["deadline_epoch"] - time()), lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        _worker_result(store, task, request, permission)
    finally:
        watchdog.cancel()


def _worker_result(store, task, request, permission):
    with exclusive_lease(store.directory / ".worker.guard"):
        owner = json.loads((store.directory / ".coordinator-owner.json").read_text())
        if owner["nonce"] != request["coordinator_nonce"]:
            raise ValueError("stale worker coordinator epoch")
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(info["num_threads"] != 1 for info in pools):
                raise ValueError("numerical thread limit not enforced")
            try:
                result = compute_task(store, task, permission)
            except Exception:
                result = failure("worker_error", traceback.format_exc())
        store.put(
            "transport-" + task_key(task),
            dict(
                request_sha256=digest(request),
                result=result,
                finished_epoch=time(),
                numerical_libraries=pools,
            ),
        )


def coordinate(args, lock, permission):
    ensure_destination(args.directory, lock["protocol"], args.mode)
    directory = Path(args.directory)
    with exclusive_lease(directory / ".coordinator.guard"):
        # An old child may outlive its parent. Never overlap it or assume it died.
        with exclusive_lease(directory / ".worker.guard"):
            nonce = uuid.uuid4().hex
            owner = directory / ".coordinator-owner.json"
            temporary = directory / (".owner-" + nonce)
            temporary.write_text(json.dumps(dict(nonce=nonce, pid=os.getpid())))
            os.replace(temporary, owner)
        store = Checkpoints(directory, lock)
        store.put("authorization", permission)
        for task in lock["populations"][args.mode]:
            previous = store.get(task_key(task))
            result = execute_task(
                store, task, args.mode, nonce, args.lock, args.authorization
            )
            print(task_key(task), result["status"], flush=True)
            if result["status"] == "worker_error" and previous is None:
                raise RuntimeError(
                    "saved worker error requires investigation; no retry"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "verify", "run", "pilot", "worker")
    )
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--directory", type=Path)
    parser.add_argument(
        "--mode", choices=("qualification", "pilot"), default="qualification"
    )
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.command == "freeze":
        protocol = json.loads(CONFIG.read_text())
        lock = make_lock(protocol)
        write_new_json(args.lock, lock)
        write_new_json(
            args.authorization,
            dict(
                protocol=protocol["protocol"],
                lock_sha256=digest(lock),
                scientific_execution_authorized=False,
                status="awaiting_separate_execution_approval",
            ),
        )
        print("Frozen identity; scientific execution remains disabled", digest(lock))
        return
    lock = json.loads(args.lock.read_text())
    verify_live_lock(lock)
    if args.command == "verify":
        print("Execution identity verified; no observations generated", digest(lock))
        return
    if args.command == "pilot":
        args.mode = "pilot"
    if args.command == "run" and args.mode != "qualification":
        raise ValueError("use the explicit pilot command for nonqualification work")
    auth = (
        json.loads(args.authorization.read_text())
        if args.mode == "qualification"
        else None
    )
    permission = permit(
        lock, args.mode, auth
    )  # Before output creation or RNG construction.
    if args.directory is None:
        args.directory = ROOT / (
            lock["protocol"]["outputs"]["raw_directory"]
            if args.mode == "qualification"
            else "data/generated/issue33_stage3_pilot_v1"
        )
    ensure_destination(args.directory, lock["protocol"], args.mode)
    if args.command == "worker":
        worker(args, lock, permission)
    else:
        coordinate(args, lock, permission)


if __name__ == "__main__":
    main()
