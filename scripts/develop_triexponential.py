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

from tcspc_toolkit.benchmarks.triexponential import (
    DevelopmentStore,
    construct_problem,
    digest,
    execute_task,
    jsonable,
    problem_inventory,
    report_store,
    task_inventory,
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


def fingerprint(manifest, pilot):
    paths = sorted((ROOT / "src/tcspc_toolkit").rglob("*.py")) + [Path(__file__)]
    return dict(
        manifest_sha256=digest(manifest),
        scope="pilot" if pilot else "comprehensive",
        source_lf_sha256={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in paths
        },
        python=platform.python_version(),
        platform=platform.platform(),
        thread_environment={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        dependencies={
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "tcspc-lifetime-toolkit")
        },
    )


def require_authorization(path, manifest, pilot):
    lock = json.loads(Path(path).read_text())
    if (
        lock.get("execution_authorized") is not True
        or not lock.get("maintainer_review")
        or lock.get("fingerprint") != fingerprint(manifest, pilot)
    ):
        raise ValueError(
            "reviewed, explicitly authorized matching execution lock required"
        )
    return lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run", "report", "_worker"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    tasks = task_inventory(manifest, pilot=args.pilot)
    if args.command == "inspect":
        print(
            json.dumps(
                dict(
                    fingerprint=fingerprint(manifest, args.pilot),
                    execution_authorized=False,
                    tasks=len(tasks),
                    problems=len(problem_inventory(manifest, pilot=args.pilot)),
                ),
                indent=2,
            )
        )
        return
    if args.directory is None:
        parser.error("--directory required")
    if args.command == "report":
        saved = require_authorization(
            args.directory / "lock.json", manifest, args.pilot
        )
        store = DevelopmentStore(args.directory, saved)
        print(json.dumps(report_store(tasks, store), indent=2))
        return
    if args.execution_lock is None:
        parser.error(
            "--execution-lock required; the committed manifest is nonexecuting"
        )
    lock = require_authorization(args.execution_lock, manifest, args.pilot)
    store = DevelopmentStore(args.directory, lock)
    if args.command == "_worker":
        task = next(t for t in tasks if t["id"] == args.task_id)
        arrays = store.read("input:" + task["problem"]["id"])["result"]
        result = jsonable(execute_task(task, arrays, pilot=args.pilot))
        args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf8")
        return
    # Freeze all observation/kernel bytes and hashes before any optimizer runs.
    paired = {}
    observations = {}
    for problem in problem_inventory(manifest, pilot=args.pilot):
        record = store.execute(
            "input:" + problem["id"],
            lambda p=problem: construct_problem(
                manifest, p, observation=observations.get(p["observation_id"])
            ),
        )
        if record["status"] != "completed":
            raise ValueError("input preparation incomplete; no fits started")
        checksum = record["result"]["observation_sha256"]
        observations.setdefault(problem["observation_id"], record["result"])
        previous = paired.setdefault(problem["observation_id"], checksum)
        if checksum != previous:
            raise ValueError("paired observation mismatch")
    for task in tasks:

        def run_one(task=task):
            output = args.directory / (
                hashlib.sha256(task["id"].encode()).hexdigest() + ".worker.json"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_worker",
                "--manifest",
                str(args.manifest.resolve()),
                "--directory",
                str(args.directory.resolve()),
                "--execution-lock",
                str(args.execution_lock.resolve()),
                "--task-id",
                task["id"],
                "--output",
                str(output.resolve()),
            ]
            if args.pilot:
                command.append("--pilot")
            ceiling = (
                (120 if task["configuration"] == "audit" else 60)
                if args.pilot
                else (
                    900
                    if task["configuration"] == "audit"
                    else manifest["candidate_policies"][
                        task["configuration"].split(":")[1]
                    ]["wall_seconds"]
                )
            )
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
            return json.loads(output.read_text())

        record = store.execute(task["id"], run_one)
        if record.get("result", {}).get("cleanup_complete") is False:
            raise WorkerCleanupError(
                "saved timeout has unconfirmed cleanup; run stopped"
            )


if __name__ == "__main__":
    main()
