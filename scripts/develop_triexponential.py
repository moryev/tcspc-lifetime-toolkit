"""Inspect or explicitly execute a reviewed development lock. No import work."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from threadpoolctl import threadpool_info

from tcspc_toolkit.benchmarks.triexponential import (
    DevelopmentStore,
    construct_problem,
    digest,
    execute_task,
    jsonable,
    problem_inventory,
    report_store,
    task_inventory,
    execution_specification,
    EXECUTION_SCHEMA,
    ExecutionIntegrityError,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs/issue37_pr2_development_v1.json"


class WorkerCleanupError(RuntimeError):
    """Stop the run if a timed-out worker cannot be confirmed terminated."""


def bounded_worker(command, seconds):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        _, stderr = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                terminated = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
                if terminated.returncode:
                    # A failed tree kill cannot certify cleanup even if the
                    # parent alone can be reaped. Stop before another task.
                    process.kill()
                    process.communicate(timeout=5)
                    raise WorkerCleanupError("process-tree termination failed")
            else:
                process.kill()
            process.communicate(timeout=5)
        except (OSError, subprocess.SubprocessError) as error:
            raise WorkerCleanupError(
                f"timed-out worker PID {process.pid}: cleanup unconfirmed"
            ) from error
        return False
    if process.returncode:
        raise RuntimeError("worker failed: " + stderr[-2000:])
    return True


def repository_identity(paths):
    """The enforced source must be present in the named commit, not just indexed."""
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.PIPE
            )
            .decode()
            .strip()
        )
        listing = subprocess.check_output(
            ["git", "ls-tree", "-rz", head], cwd=ROOT, stderr=subprocess.PIPE
        )
        blobs = {}
        for row in listing.split(b"\0"):
            if row:
                meta, name = row.split(b"\t", 1)
                blobs[name.decode()] = meta.split()[2].decode()
        committed = True
        for path in paths:
            data = path.read_bytes().replace(b"\r\n", b"\n")
            blob = hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest()
            committed &= blobs.get(path.relative_to(ROOT).as_posix()) == blob
        return dict(commit=head, committed_sources=bool(committed))
    except (subprocess.CalledProcessError, OSError, ValueError):
        return dict(commit=None, committed_sources=False)


def execution_environment():
    """All fields here are enforced, including loaded numerical binary identities."""
    pools = sorted(threadpool_info(), key=lambda p: p["filepath"])
    for pool in pools:
        pool["binary_sha256"] = hashlib.sha256(
            Path(pool["filepath"]).read_bytes()
        ).hexdigest()
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
        "OPENBLAS_CORETYPE",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "PYTHONHOME",
    )
    return jsonable(
        dict(
            python=sys.version,
            executable=str(Path(sys.executable).resolve()),
            executable_sha256=hashlib.sha256(
                Path(sys.executable).read_bytes()
            ).hexdigest(),
            platform=platform.platform(),
            machine=platform.machine(),
            module_origins={
                name: dict(
                    path=str(Path(sys.modules[name].__file__).resolve()),
                    sha256=hashlib.sha256(
                        Path(sys.modules[name].__file__).read_bytes()
                    ).hexdigest(),
                )
                for name in (
                    "numpy",
                    "scipy",
                    "tcspc_toolkit",
                    "tcspc_toolkit.benchmarks.triexponential",
                )
            },
            dependencies=sorted(
                [
                    (d.metadata["Name"], d.version)
                    for d in importlib.metadata.distributions()
                    if d.metadata.get("Name")
                ],
                key=lambda p: (p[0].lower(), p[1]),
            ),
            environment={name: os.environ.get(name) for name in names},
            numerical_libraries=pools,
        )
    )


def fingerprint(manifest, pilot, manifest_path=None):
    manifest_path = Path(
        manifest_path or ROOT / "configs/issue37_pr2_development_v1.json"
    ).resolve()
    raw = manifest_path.read_bytes()
    if json.loads(raw) != manifest:
        raise ExecutionIntegrityError("manifest object differs from supplied file")
    paths = sorted((ROOT / "src/tcspc_toolkit").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
    ]
    return dict(
        schema=EXECUTION_SCHEMA,
        repository=repository_identity([*paths, manifest_path]),
        manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_sha256=digest(manifest),
        scope="pilot" if pilot else "comprehensive",
        source_lf_sha256={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in paths
        },
        environment=execution_environment(),
        execution=execution_specification(manifest, pilot=pilot),
    )


def require_authorization(path, manifest, pilot, manifest_path=None):
    lock = json.loads(Path(path).read_text())
    if (
        lock.get("schema") != EXECUTION_SCHEMA
        or lock.get("execution_authorized") is not True
        or not lock.get("maintainer_review")
        or lock.get("fingerprint") != fingerprint(manifest, pilot, manifest_path)
        or lock["fingerprint"]["repository"]["committed_sources"] is not True
    ):
        raise ValueError(
            "reviewed, explicitly authorized matching execution lock required"
        )
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("inspect", "prepare", "run", "report", "_worker")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    tasks = task_inventory(manifest, pilot=args.pilot)
    if args.command == "inspect":
        print(
            json.dumps(
                dict(
                    schema=EXECUTION_SCHEMA,
                    fingerprint=fingerprint(manifest, args.pilot, args.manifest),
                    execution_authorized=False,
                    maintainer_review=None,
                    tasks=len(tasks),
                    problems=len(problem_inventory(manifest, pilot=args.pilot)),
                ),
                indent=2,
            )
        )
        return
    if args.directory is None:
        parser.error("--directory required")
    if args.execution_lock is None and args.command != "report":
        parser.error(
            "--execution-lock required; the committed manifest is nonexecuting"
        )
    authorization_path = args.execution_lock or args.directory / "lock.json"

    def verify_execution():
        return require_authorization(
            authorization_path, manifest, args.pilot, args.manifest
        )

    lock = verify_execution()
    preparing = args.command == "prepare"
    if preparing and args.directory.exists():
        raise ExecutionIntegrityError(
            "preparation already started; use a separately reviewed fresh directory"
        )
    store = DevelopmentStore(
        args.directory, lock, create=preparing, verify_execution=verify_execution
    )
    if preparing:
        store.prepare_inputs(
            tasks,
            lambda p, observation: construct_problem(
                manifest, p, observation=observation
            ),
        )
        return
    inventory = store.verify_inventory(tasks)
    if args.command == "report":
        print(json.dumps(report_store(tasks, store), indent=2))
        return
    for scheduled in tasks:
        saved = store.read_attempt(scheduled, inventory)
        if saved.get("result", {}).get("cleanup_complete") is False:
            raise WorkerCleanupError(
                "saved timeout has unconfirmed cleanup; run stopped"
            )
        if saved["status"] in ("running", "interrupted") and not (
            args.command == "_worker" and scheduled["id"] == args.task_id
        ):
            raise WorkerCleanupError(
                "interrupted attempt has unconfirmed termination; run stopped"
            )
    if args.command == "_worker":
        task = next(t for t in tasks if t["id"] == args.task_id)
        arrays, binding = store.worker_input(task, tasks)
        result = jsonable(execute_task(task, arrays, pilot=args.pilot))
        store.write_worker_result(task, tasks, result, binding)
        return
    for task in tasks:

        def run_one(binding, task=task):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--manifest",
                str(args.manifest.resolve()),
                "--directory",
                str(args.directory.resolve()),
                "--execution-lock",
                str(authorization_path.resolve()),
                "--task-id",
                task["id"],
            ]
            if args.pilot:
                command.append("--pilot")
            ceiling = lock["fingerprint"]["execution"]["worker_deadlines"][
                task["configuration"]
            ]
            # Hard parent timeout includes worker startup. Internal budgets remain
            # cooperative and are recorded independently. No automatic retry.
            try:
                finished = bounded_worker(command, ceiling)
            except WorkerCleanupError as error:
                return dict(
                    failure="external_worker_timeout",
                    work_unknown=True,
                    hard_wall_seconds=ceiling,
                    cleanup_complete=False,
                    cleanup_error=str(error),
                )
            if not finished:
                return dict(
                    failure="external_worker_timeout",
                    work_unknown=True,
                    hard_wall_seconds=ceiling,
                    cleanup_complete=True,
                )
            return store.read_worker_result(task, tasks, binding)

        record = store.execute_attempt(task, tasks, run_one)
        if record.get("result", {}).get("cleanup_complete") is False:
            raise WorkerCleanupError(
                "saved timeout has unconfirmed cleanup; run stopped"
            )


if __name__ == "__main__":
    main()
