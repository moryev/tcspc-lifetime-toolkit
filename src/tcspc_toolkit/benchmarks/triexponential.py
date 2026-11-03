"""Explicit development construction, sealed-audit cross-scoring and reporting.

Importing this module performs no generation, fitting or file I/O. This is
development infrastructure, not measurement integration or qualification.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.stats import exponnorm

from tcspc_toolkit.benchmarks._triexponential_audit import (
    AuditSettings,
    audit_terms,
    direct_expected,
    run_audit,
)
from tcspc_toolkit.estimation.triexponential import TriexponentialFitSettings
from tcspc_toolkit.estimation._fixed_k_search import _fit_triexponential
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_expected_counts,
)


def jsonable(value):
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(
            jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


EXECUTION_SCHEMA = "issue37-development-execution-v2"


class ExecutionIntegrityError(ValueError):
    """Evidence cannot be verified; never reinterpret this as numerical failure."""


def settings_for_task(configuration, *, pilot=False):
    """Resolve the actual execution settings without generation or optimization."""
    if configuration == "audit":
        settings = AuditSettings.development_candidate()
        return (
            replace(settings, model_calls=40000, wall_seconds=120)
            if pilot
            else settings
        )
    solver, policy = configuration.split(":")
    settings = replace(TriexponentialFitSettings.development(policy), solver=solver)
    return (
        replace(settings, max_optimization_calls=20000, wall_seconds=60)
        if pilot
        else settings
    )


def execution_specification(manifest, *, pilot=False):
    """The settings/task specification enforced by the execution fingerprint."""
    tasks = task_inventory(manifest, pilot=pilot)
    settings = {
        name: jsonable(settings_for_task(name, pilot=pilot))
        for name in sorted({t["configuration"] for t in tasks})
    }
    return jsonable(
        dict(
            tasks=tasks,
            settings=settings,
            worker_deadlines={k: v["wall_seconds"] for k, v in settings.items()},
            serial_workers=1,
            termination_seconds=5,
            cleanup_seconds=5,
            retries=False,
            continuations=False,
            input_generation=dict(
                procedure="benchmarks.triexponential.construct_problem",
                construction=manifest["construction"],
                irfs=manifest["irfs"],
                namespace=manifest["operational_pilot"]["namespace"]
                if pilot
                else manifest["construction"]["observation_namespace"],
                deterministic_target_kind="deterministic_expected_counts",
                sampled_target_kind="raw_counts",
            ),
        )
    )


def problem_inventory(manifest, *, pilot=False):
    """Resolve identities only: no RNG, physical model or optimizer calls."""
    if [r["id"] for r in manifest["conditions"]] != [f"D{i:02d}" for i in range(18)]:
        raise ValueError("D00-D17 required")
    common = manifest["defaults"]
    if common["deterministic_targets"] != 1 or common["poisson_replicates"] != 3:
        raise ValueError("v1 target counts changed")
    if (
        manifest["construction"]["observation_namespace"]
        == manifest["operational_pilot"]["namespace"]
    ):
        raise ValueError("pilot streams overlap")
    rows = (
        manifest["operational_pilot"]["conditions"] if pilot else manifest["conditions"]
    )
    namespace = (
        manifest["operational_pilot"]["namespace"]
        if pilot
        else manifest["construction"]["observation_namespace"]
    )
    result = []
    for index, row in enumerate(rows):
        condition = common | row
        if pilot:
            condition.update(
                generating_irf=row["irf"],
                background_counts_per_bin=row["background"],
                assumed_irfs=["matched"],
                stratum="operational_pilot",
            )
        labels = ["mean"] + [f"poisson-{r}" for r in range(1 if pilot else 3)]
        for label in labels:
            identity = row["id"] + ":" + label
            seed = (
                None
                if label == "mean"
                else [*namespace, index, int(label.split("-")[1])]
            )
            for assumed in condition["assumed_irfs"]:
                result.append(
                    dict(
                        id=identity + ":" + assumed,
                        observation_id=identity,
                        seed=seed,
                        assumed_irf=assumed,
                        condition=condition,
                        control=(
                            not pilot
                            and row["id"] in manifest["evaluation_rules"]["control_ids"]
                        ),
                    )
                )
    if len(result) != (6 if pilot else 80) or len(
        {p["observation_id"] for p in result}
    ) != (6 if pilot else 72):
        raise ValueError("development population changed")
    if not pilot and sum(p["control"] for p in result) != 8:
        raise ValueError("eight controls required")
    return tuple(result)


def task_inventory(manifest, *, pilot=False):
    policies = ("compact",) if pilot else tuple(manifest["candidate_policies"])
    return tuple(
        dict(id=p["id"] + ":" + name, problem=p, configuration=name)
        for p in problem_inventory(manifest, pilot=pilot)
        for name in (
            "audit",
            *(
                f"{solver}:{policy}"
                for solver in ("lbfgsb", "trf")
                for policy in policies
            ),
        )
    )


def _kernel(manifest, name, time):
    spec = manifest["irfs"][name]
    if name in ("gaussian", "broad_gaussian"):
        sigma = spec["fwhm_ns"] / np.sqrt(8 * np.log(2))
        values = np.exp(-0.5 * ((time - spec["center_ns"]) / sigma) ** 2)
    elif name == "emg":
        sigma = spec["gaussian_fwhm_ns"] / np.sqrt(8 * np.log(2))
        values = exponnorm.pdf(
            time, spec["tail_ns"] / sigma, loc=spec["center_ns"], scale=sigma
        )
    else:
        base = manifest["irfs"][spec.get("base", name)]
        knots = [time[-1] if v == "t_last" else v for v in base["times_ns"]]
        values = np.interp(time, knots, base["values"]) + spec.get(
            "add_before_normalization", 0
        )
    return values / np.trapezoid(values, time)


def construct_problem(manifest, problem, *, observation=None):
    """Explicit construction only; caller persists hashes before any fitting."""
    c = problem["condition"]
    if observation is not None:
        time = np.asarray(observation["time"])
        assumed = (
            np.asarray(observation["generating_kernel"])
            if problem["assumed_irf"] == "matched"
            else _kernel(manifest, problem["assumed_irf"], time)
        )
        return dict(observation) | dict(
            kernel=assumed,
            assumed_irf_sha256=hashlib.sha256(
                np.asarray(assumed, dtype="<f8").tobytes()
            ).hexdigest(),
        )
    time = c["time_origin_ns"] + np.arange(c["bins"]) * c["bin_width_ns"]
    generating = _kernel(manifest, c["generating_irf"], time)
    tau = np.asarray(c["lifetimes_ns"])
    unit = [
        multiexponential_reconvolution_expected_counts(
            time,
            generating,
            amplitudes=[1],
            lifetimes=[v],
            temporal_shift=c["shift_ns"],
        )
        for v in tau
    ]
    amplitudes = (
        c["signal_photons"]
        * np.asarray(c["detected_signal_fractions"])
        / np.sum(unit, axis=1)
    )
    mean = np.sum(amplitudes[:, None] * unit, axis=0) + c["background_counts_per_bin"]
    observed = (
        mean
        if problem["seed"] is None
        else np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(problem["seed"]))
        ).poisson(mean)
    )
    assumed = (
        generating
        if problem["assumed_irf"] == "matched"
        else _kernel(manifest, problem["assumed_irf"], time)
    )
    return dict(
        time=time,
        observed=observed,
        kernel=assumed,
        generating_kernel=generating,
        mean=mean,
        truth=dict(
            amplitudes=amplitudes,
            lifetimes_ns=tau,
            background=c["background_counts_per_bin"],
            shift=c["shift_ns"],
        ),
        observation_sha256=hashlib.sha256(
            np.asarray(observed, dtype="<f8").tobytes()
        ).hexdigest(),
        assumed_irf_sha256=hashlib.sha256(
            np.asarray(assumed, dtype="<f8").tobytes()
        ).hexdigest(),
    )


def _production_prediction(time, kernel, p):
    return multiexponential_reconvolution_expected_counts(
        time,
        kernel,
        amplitudes=p[:3],
        lifetimes=p[3:6],
        background=p[6],
        temporal_shift=p[7],
    )


def cross_score(time, observed, kernel, production, audit, *, tolerance=0.01):
    """Pure post-seal scoring: two calls to each implementation, no search."""
    if not audit.sealed:
        raise ValueError("audit must be sealed before cross-scoring")
    result = dict(
        status="unavailable",
        signed_gap=None,
        audit_complete=audit.complete,
        production_accepted=production.numerical.accepted,
        cross_consistent=False,
        production_model_calls=0,
        direct_model_calls=0,
        component_convolutions=0,
    )
    if production.parameters is None or audit.parameters is None:
        return result
    p = production.parameters
    vectors = (
        (*p.amplitudes, *p.lifetimes_ns, p.background, p.temporal_shift_ns),
        audit.parameters,
    )
    scores = []
    consistent = True
    try:
        for i, vector in enumerate(vectors):
            result["production_model_calls"] += 1
            result["component_convolutions"] += 3
            fft = _production_prediction(time, kernel, vector)
            result["direct_model_calls"] += 1
            result["component_convolutions"] += 3
            direct = direct_expected(time, kernel, vector)
            a = calculate_poisson_deviance(observed, fft) / 2
            b = float(np.sum(audit_terms(observed, direct)))
            saved_mu = production.expected_counts if i == 0 else audit.expected_counts
            saved_c = production.centered_nll if i == 0 else audit.centered_nll
            consistent &= bool(
                np.isfinite(a)
                and np.isfinite(b)
                and np.allclose(fft, direct, rtol=1e-10, atol=1e-10)
                and np.allclose(saved_mu, fft, rtol=1e-10, atol=1e-10)
                and np.isclose(a, b, rtol=1e-11, atol=1e-8)
                and np.isclose(saved_c, a, rtol=1e-11, atol=1e-8)
            )
            scores.append(a)
    except (ValueError, FloatingPointError, OverflowError):
        result["status"] = "cross_implementation_disagreement"
        return result
    gap = scores[0] - scores[1]
    result.update(
        cross_consistent=consistent, signed_gap=float(gap) if np.isfinite(gap) else None
    )
    if not consistent:
        result["status"] = "cross_implementation_disagreement"
    elif not audit.complete:
        result["status"] = "audit_incomplete"
    elif gap > tolerance:
        result["status"] = "production_worse"
    elif gap < -tolerance:
        result["status"] = "production_better_audit_limitation"
    else:
        result["status"] = "agreement"
    return result


def component_recovery(problem, truth, parameters):
    tau, amplitudes = np.asarray(truth["lifetimes_ns"]), np.asarray(truth["amplitudes"])
    if (
        parameters is None
        or len(tau) != 3
        or np.any(amplitudes == 0)
        or len(set(tau)) != 3
        or problem["assumed_irf"] != "matched"
    ):
        return dict(
            available=False,
            reason="missing_or_nonunique_or_misspecified",
            contributors=0,
        )
    order = np.argsort(tau)
    return dict(
        available=True,
        contributors=3,
        lifetime_relative_error=(
            np.asarray(parameters["lifetimes_ns"]) / tau[order] - 1
        ).tolist(),
        amplitude_relative_error=(
            np.asarray(parameters["amplitudes"]) / amplitudes[order] - 1
        ).tolist(),
    )


def _planned_inputs(tasks):
    problems, identities = {}, set()
    for task in tasks:
        p = task["problem"]
        if (
            task["id"] in identities
            or task["id"] != p["id"] + ":" + task["configuration"]
        ):
            raise ExecutionIntegrityError("duplicate or conflicting task identity")
        identities.add(task["id"])
        if p["id"] in problems and problems[p["id"]] != p:
            raise ExecutionIntegrityError("conflicting input identity")
        problems[p["id"]] = p
    if not problems:
        raise ExecutionIntegrityError("empty task inventory")
    return problems


def _input_binding(problem):
    return dict(
        problem_id=problem["id"],
        problem_sha256=digest(problem),
        observation_id=problem["observation_id"],
        target_kind="deterministic_expected_counts"
        if problem["seed"] is None
        else "raw_counts",
        assumed_irf=problem["assumed_irf"],
    )


def _array_identity(values):
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ExecutionIntegrityError("invalid input vector")
    return dict(
        shape=list(array.shape),
        dtype="<f8",
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def _read_hashed(path):
    try:
        record = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(record, dict) or record.get("payload_sha256") != digest(
            {k: v for k, v in record.items() if k != "payload_sha256"}
        ):
            raise ExecutionIntegrityError(
                "checkpoint payload hash mismatch: " + path.name
            )
        return record
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionIntegrityError(
            "missing or corrupt evidence: " + path.name
        ) from error


def _publish_record(path, payload, *, replace_existing=False):
    """Atomic publication; preparation marker gives the seal a single writer."""
    if path.exists() and not replace_existing:
        raise ExecutionIntegrityError("evidence already exists: " + path.name)
    record = dict(payload, payload_sha256=digest(payload))
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as stream:
            json.dump(record, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return record


def _start_receipt(path, payload):
    """Exclusive durable intent: even a torn receipt blocks automatic repetition."""
    record = dict(payload, payload_sha256=digest(payload))
    with path.open("x", encoding="utf8") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


class DevelopmentStore:
    """Single-writer atomic JSON checkpoints; interrupted attempts are retained.

    Resume skips all existing attempts, including running/failed attempts. A
    rerun requires a new, explicitly reviewed run directory, not silent retries.
    """

    def __init__(self, directory, lock, *, create=True, verify_execution=None):
        self.directory = Path(directory)
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = jsonable(lock)
        self.verify_execution = verify_execution
        path = self.directory / "lock.json"
        if path.exists():
            if json.loads(path.read_text()) != self.lock:
                raise ValueError("execution lock mismatch")
        elif create:
            with path.open("x", encoding="utf8") as f:
                json.dump(self.lock, f, sort_keys=True)
        else:
            raise ExecutionIntegrityError("execution directory/lock missing")

    def path(self, identity):
        return self.directory / (
            hashlib.sha256(identity.encode()).hexdigest() + ".json"
        )

    def execute(self, identity, operation, *, binding=None):
        path = self.path(identity)
        if path.exists():
            return self.read(identity, binding=binding)
        pending = dict(id=identity, status="running", lock=digest(self.lock))
        if binding is not None:
            pending["binding"] = binding
        pending["payload_sha256"] = digest(pending)
        with path.open("x", encoding="utf8") as f:
            json.dump(pending, f)
        integrity_error = None
        try:
            payload = dict(
                id=identity,
                status="completed",
                lock=digest(self.lock),
                result=jsonable(operation()),
            )
        except Exception as error:
            if isinstance(error, ExecutionIntegrityError):
                integrity_error = error
            payload = dict(
                id=identity,
                status="exception",
                lock=digest(self.lock),
                exception=type(error).__name__,
                message=str(error),
            )
        if binding is not None:
            payload["binding"] = binding
        record = _publish_record(path, payload, replace_existing=True)
        if integrity_error is not None:
            raise integrity_error
        return record

    def read(self, identity, *, binding=None):
        path = self.path(identity)
        if not path.exists():
            return dict(id=identity, status="missing_task")
        record = _read_hashed(path)
        if record["id"] != identity or record["lock"] != digest(self.lock):
            raise ExecutionIntegrityError("checkpoint identity/lock mismatch")
        if record.get("status") not in ("running", "completed", "exception"):
            raise ExecutionIntegrityError("unknown checkpoint status")
        if binding is not None and record.get("binding") != binding:
            raise ExecutionIntegrityError(
                "checkpoint input/inventory association mismatch"
            )
        return record

    def _execution_identity(self, tasks):
        if (
            self.verify_execution is None
            or self.verify_execution() != self.lock
            or self.lock.get("schema") != EXECUTION_SCHEMA
            or self.lock.get("execution_authorized") is not True
            or not self.lock.get("maintainer_review")
            or self.lock["fingerprint"]["execution"]["tasks"] != jsonable(tasks)
        ):
            raise ExecutionIntegrityError(
                "unverified execution/source/environment/task identity"
            )
        if json.loads((self.directory / "lock.json").read_text()) != self.lock:
            raise ExecutionIntegrityError("saved execution lock mismatch")
        return dict(
            lock_sha256=digest(self.lock),
            fingerprint_sha256=digest(self.lock["fingerprint"]),
            manifest_sha256=self.lock["fingerprint"]["manifest_file_sha256"],
            tasks_sha256=digest(tasks),
        )

    def _input_entry(self, problem):
        identity = "input:" + problem["id"]
        binding = _input_binding(problem)
        record = self.read(identity, binding=binding)
        if record["status"] != "completed":
            raise ExecutionIntegrityError("input preparation incomplete: " + identity)
        arrays = record["result"]
        condition = problem["condition"]
        grid = (
            condition["time_origin_ns"]
            + np.arange(condition["bins"]) * condition["bin_width_ns"]
        )
        if not np.array_equal(arrays["time"], grid):
            raise ExecutionIntegrityError("input grid differs from approved condition")
        identities = {
            name: _array_identity(arrays[name])
            for name in ("time", "observed", "kernel", "generating_kernel", "mean")
        }
        if len({tuple(v["shape"]) for v in identities.values()}) != 1:
            raise ExecutionIntegrityError("input vector alignment mismatch")
        if (
            np.any(np.diff(arrays["time"]) <= 0)
            or any(
                np.any(np.asarray(arrays[n]) < 0)
                for n in ("observed", "kernel", "generating_kernel", "mean")
            )
            or (
                binding["target_kind"] == "raw_counts"
                and np.any(
                    np.asarray(arrays["observed"]) != np.floor(arrays["observed"])
                )
            )
            or arrays["observation_sha256"] != identities["observed"]["sha256"]
            or arrays["assumed_irf_sha256"] != identities["kernel"]["sha256"]
            or (
                binding["target_kind"] == "deterministic_expected_counts"
                and not np.array_equal(arrays["observed"], arrays["mean"])
            )
        ):
            raise ExecutionIntegrityError(
                "observation/kernel hash or target-kind mismatch"
            )
        entry = dict(
            **binding,
            checkpoint_id=identity,
            filename=self.path(identity).name,
            file_sha256=hashlib.sha256(self.path(identity).read_bytes()).hexdigest(),
            payload_sha256=record["payload_sha256"],
            arrays=identities,
            truth_sha256=digest(arrays["truth"]),
        )
        return entry, arrays

    def prepare_inputs(self, tasks, builder):
        """One attempt to prepare and seal inputs. Never a resume/generation path."""
        identity = self._execution_identity(tasks)
        problems = _planned_inputs(tasks)
        if set(p.name for p in self.directory.iterdir()) != {"lock.json"}:
            raise ExecutionIntegrityError(
                "preparation already started; incomplete preparation needs controlled recovery"
            )
        marker = dict(id="preparation_started", **identity)
        # Exclusive durable marker precedes the first call to the generator.
        marker["payload_sha256"] = digest(marker)
        with (self.directory / "preparation_started.json").open(
            "x", encoding="utf8"
        ) as stream:
            json.dump(marker, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        observations, entries = {}, []
        for problem in problems.values():
            self._execution_identity(tasks)
            self.execute(
                "input:" + problem["id"],
                lambda p=problem: builder(p, observations.get(p["observation_id"])),
                binding=_input_binding(problem),
            )
            entry, arrays = self._input_entry(problem)
            observations.setdefault(problem["observation_id"], arrays)
            entries.append(entry)
        self._execution_identity(tasks)
        self._verify_pairing(entries)
        inventory = dict(
            schema=EXECUTION_SCHEMA,
            id="input_inventory",
            **identity,
            inputs=entries,
            task_inputs=self._task_inputs(tasks, entries),
        )
        _publish_record(self.directory / "input_inventory.json", inventory)
        return self.verify_inventory(tasks)

    @staticmethod
    def _verify_pairing(entries):
        paired = {}
        for entry in entries:
            shared = {
                name: entry["arrays"][name]
                for name in ("time", "observed", "generating_kernel", "mean")
            }
            shared["truth_sha256"] = entry["truth_sha256"]
            previous = paired.setdefault(entry["observation_id"], shared)
            if previous != shared:
                raise ExecutionIntegrityError("paired observation identity mismatch")

    @staticmethod
    def _task_inputs(tasks, entries):
        inputs = {e["problem_id"]: e for e in entries}
        return [
            dict(
                task_id=t["id"],
                configuration=t["configuration"],
                input_id=inputs[t["problem"]["id"]]["checkpoint_id"],
                input_payload_sha256=inputs[t["problem"]["id"]]["payload_sha256"],
            )
            for t in tasks
        ]

    def verify_inventory(self, tasks):
        """Re-read the full seal, every input, and every existing attempt. No writes."""
        identity = self._execution_identity(tasks)
        problems = _planned_inputs(tasks)
        seal = self.directory / "input_inventory.json"
        if not seal.exists():
            raise ExecutionIntegrityError(
                "incomplete preparation: sealed input inventory missing"
            )
        inventory = _read_hashed(seal)
        marker = _read_hashed(self.directory / "preparation_started.json")
        if any(
            inventory.get(k) != v or marker.get(k) != v for k, v in identity.items()
        ):
            raise ExecutionIntegrityError(
                "inventory execution/source/manifest mismatch"
            )
        entries = [self._input_entry(p)[0] for p in problems.values()]
        expected = dict(
            schema=EXECUTION_SCHEMA,
            id="input_inventory",
            **identity,
            inputs=entries,
            task_inputs=self._task_inputs(tasks, entries),
        )
        if inventory != dict(expected, payload_sha256=digest(expected)):
            raise ExecutionIntegrityError("sealed inventory/input identity mismatch")
        self._verify_pairing(entries)
        allowed = {"lock.json", "preparation_started.json", "input_inventory.json"}
        allowed.update(e["filename"] for e in entries)
        for task in tasks:
            path = self.path(task["id"])
            allowed.update(
                path.stem + suffix
                for suffix in (
                    ".json",
                    ".worker.json",
                    ".started.json",
                    ".worker_started.json",
                )
            )
            record = self.read_attempt(task, inventory)
            if record["status"] == "missing_task" and (
                path.with_suffix(".worker.json").exists()
                or path.with_suffix(".worker_started.json").exists()
            ):
                raise ExecutionIntegrityError("orphan worker result without an attempt")
            if path.with_suffix(".worker.json").exists():
                self._verified_worker_record(task, self.task_binding(task, inventory))
        if any(p.name not in allowed for p in self.directory.glob("*.json")):
            raise ExecutionIntegrityError("foreign or duplicate checkpoint evidence")
        return inventory

    @staticmethod
    def task_binding(task, inventory):
        matches = [r for r in inventory["task_inputs"] if r["task_id"] == task["id"]]
        if len(matches) != 1:
            raise ExecutionIntegrityError("missing or conflicting task input reference")
        return dict(inventory_sha256=inventory["payload_sha256"], **matches[0])

    def execute_attempt(self, task, tasks, operation):
        inventory = self.verify_inventory(tasks)
        binding = self.task_binding(task, inventory)
        existing = self.read_attempt(task, inventory)
        if existing["status"] != "missing_task":
            return existing
        _start_receipt(
            self.path(task["id"]).with_suffix(".started.json"),
            dict(
                id=task["id"],
                kind="attempt_started",
                lock=digest(self.lock),
                binding=binding,
            ),
        )
        record = self.execute(task["id"], lambda: operation(binding), binding=binding)
        self.verify_inventory(tasks)
        return record

    def read_attempt(self, task, inventory):
        binding = self.task_binding(task, inventory)
        path = self.path(task["id"])
        receipt_path = path.with_suffix(".started.json")
        if not receipt_path.exists():
            if path.exists():
                raise ExecutionIntegrityError("attempt checkpoint has no start receipt")
            return dict(id=task["id"], status="missing_task")
        receipt = _read_hashed(receipt_path)
        expected = dict(
            id=task["id"],
            kind="attempt_started",
            lock=digest(self.lock),
            binding=binding,
        )
        if receipt != dict(expected, payload_sha256=digest(expected)):
            raise ExecutionIntegrityError("attempt start receipt identity mismatch")
        worker = path.with_suffix(".worker_started.json")
        if worker.exists():
            claim = _read_hashed(worker)
            if any(
                claim.get(k) != v
                for k, v in (dict(expected, kind="worker_started")).items()
            ):
                raise ExecutionIntegrityError("worker start receipt identity mismatch")
        record = self.read(task["id"], binding=binding)
        if record["status"] == "missing_task":
            return dict(
                id=task["id"],
                status="interrupted",
                lock=digest(self.lock),
                binding=binding,
                receipt_sha256=receipt["payload_sha256"],
                reason="attempt_checkpoint_missing",
            )
        return record

    def worker_input(self, task, tasks):
        inventory = self.verify_inventory(tasks)
        binding = self.task_binding(task, inventory)
        if self.read_attempt(task, inventory)["status"] != "running":
            raise ExecutionIntegrityError(
                "worker requires its existing bound running attempt"
            )
        _start_receipt(
            self.path(task["id"]).with_suffix(".worker_started.json"),
            dict(
                id=task["id"],
                kind="worker_started",
                lock=digest(self.lock),
                binding=binding,
                pid=os.getpid(),
            ),
        )
        record = self.read(binding["input_id"], binding=_input_binding(task["problem"]))
        return record["result"], binding

    def write_worker_result(self, task, tasks, result, binding):
        inventory = self.verify_inventory(tasks)
        if binding != self.task_binding(task, inventory):
            raise ExecutionIntegrityError("worker inventory changed")
        if (
            not self.path(task["id"]).with_suffix(".worker_started.json").exists()
            or self.read_attempt(task, inventory)["status"] != "running"
        ):
            raise ExecutionIntegrityError(
                "worker result requires its active start receipt"
            )
        _publish_record(
            self.path(task["id"]).with_suffix(".worker.json"),
            dict(
                id=task["id"],
                lock=digest(self.lock),
                binding=binding,
                result=jsonable(result),
            ),
        )

    def read_worker_result(self, task, tasks, binding):
        inventory = self.verify_inventory(tasks)
        if binding != self.task_binding(task, inventory):
            raise ExecutionIntegrityError("worker inventory changed")
        return self._verified_worker_record(task, binding)["result"]

    def _verified_worker_record(self, task, binding):
        record = _read_hashed(self.path(task["id"]).with_suffix(".worker.json"))
        if (
            record.get("id") != task["id"]
            or record.get("lock") != digest(self.lock)
            or record.get("binding") != binding
            or not self.path(task["id"]).with_suffix(".worker_started.json").exists()
            or "result" not in record
        ):
            raise ExecutionIntegrityError("worker result execution/input mismatch")
        return record


def reported_correctness(fit):
    """Check saved contracts; this is not a proof of algorithmic correctness."""
    if "numerical" not in fit or "optimization" not in fit:
        return dict(status="unavailable", violations=[])
    failures = []
    numerical, p = fit["numerical"], fit.get("parameters")
    work = fit["optimization"]["work"]
    if p is not None:
        a, tau = np.asarray(p["amplitudes"]), np.asarray(p["lifetimes_ns"])
        mu = np.asarray(fit["expected_counts"], dtype=float)
        if not (
            np.all(np.isfinite(a))
            and np.all(a >= 0)
            and np.all(np.isfinite(tau))
            and np.all(tau > 0)
            and np.all(np.isfinite(mu))
            and np.all(mu >= 0)
            and np.isfinite(p["background"])
            and p["background"] >= 0
            and np.isfinite(p["temporal_shift_ns"])
        ):
            failures.append("invalid_parameters_or_predictions")
    if numerical["accepted"] and (
        p is None
        or not numerical["consistent"]
        or not numerical["required_stages_complete"]
        or not numerical.get("validation")
        or numerical["validation"]["candidate"] != p
        or numerical["validation"]["passed"] is not True
    ):
        failures.append("accepted_without_exact_candidate_assessment")
    starts = fit["optimization"]["starts"]
    if (
        sum(s["model_calls"] for s in starts)
        != work["phases"]["optimization"]["model_calls"]
        or any(s["model_calls"] > s["model_call_limit"] for s in starts)
        or sum(v["model_calls"] for v in work["phases"].values()) != work["model_calls"]
        or work["charged_component_convolutions"] != 3 * work["model_calls"]
    ):
        failures.append("work_accounting_inconsistent")
    return dict(
        status="violations" if failures else "reported_contracts_consistent",
        violations=failures,
    )


def summarize_tasks(tasks, records):
    """Every planned task contributes, including missing/interrupted attempts."""
    _planned_inputs(tasks)
    known = {r["id"]: r for r in records}
    if len(known) != len(records) or set(known) - {t["id"] for t in tasks}:
        raise ValueError("duplicate or foreign task records")
    groups = {}
    for task in tasks:
        group = groups.setdefault(
            task["configuration"],
            dict(
                denominator=0,
                controls=0,
                robustness=0,
                statuses=Counter(),
                numerical_reasons=Counter(),
                strata=Counter(),
                accepted=0,
                audit_complete=0,
                gaps=Counter(),
                recovery_available=0,
                wall_seconds=[],
                model_calls=[],
                component_convolutions=[],
                recovery_starts=[],
                conditional_lifetime_errors=[],
                conditional_amplitude_errors=[],
                unknown_work=0,
                censored_runtime_count=0,
                reported_correctness=Counter(),
            ),
        )
        group["denominator"] += 1
        group["controls" if task["problem"]["control"] else "robustness"] += 1
        row = known.get(task["id"], dict(status="missing_task"))
        group["statuses"][row["status"]] += 1
        result = row.get("result", {})
        group["reported_correctness"][reported_correctness(result)["status"]] += 1
        group["strata"][task["problem"]["condition"].get("stratum", "unknown")] += 1
        group["numerical_reasons"].update(
            result.get("numerical", {}).get("reasons", result.get("reasons", ()))
        )
        if result.get("failure"):
            group["numerical_reasons"][result["failure"]] += 1
        group["unknown_work"] += bool(
            result.get(
                "work_unknown",
                row["status"]
                in ("missing_task", "running", "interrupted", "exception"),
            )
        )
        optimization = result.get("optimization", {})
        work = optimization.get("work", {})
        wall = work.get(
            "total_seconds",
            result.get("elapsed_seconds", result.get("hard_wall_seconds")),
        )
        group["censored_runtime_count"] += int(
            result.get("failure") == "external_worker_timeout"
        )
        if wall is not None:
            group["wall_seconds"].append(wall)
        calls = work.get("model_calls")
        if calls is None and "phase_calls" in result:
            calls = sum(v for _, v in result["phase_calls"])
        if calls is not None:
            group["model_calls"].append(calls)
        components = work.get(
            "charged_component_convolutions", result.get("component_convolutions")
        )
        if components is not None:
            group["component_convolutions"].append(components)
        if "recovery_starts_used" in optimization:
            group["recovery_starts"].append(optimization["recovery_starts_used"])
        group["accepted"] += bool(result.get("numerical", {}).get("accepted", False))
        group["audit_complete"] += bool(result.get("complete", False))
        group["gaps"][result.get("cross_score", {}).get("status", "unavailable")] += 1
        group["recovery_available"] += bool(
            result.get("recovery", {}).get("available", False)
        )
        recovery = result.get("recovery", {})
        if recovery.get("available"):
            group["conditional_lifetime_errors"].append(
                recovery["lifetime_relative_error"]
            )
            group["conditional_amplitude_errors"].append(
                recovery["amplitude_relative_error"]
            )
    for group in groups.values():
        times = group["wall_seconds"]
        group["runtime_summary"] = dict(
            contributors=len(times),
            median=float(np.median(times)) if times else None,
            p90=float(np.percentile(times, 90)) if times else None,
            maximum=max(times) if times else None,
        )
        group["conditional_recovery"] = {}
        for quantity in ("lifetime", "amplitude"):
            errors = group[f"conditional_{quantity}_errors"]
            group["conditional_recovery"][quantity] = dict(
                contributors_per_component=[len(errors)] * 3,
                mean_signed_relative_error=np.mean(errors, axis=0).tolist()
                if errors
                else None,
                median_absolute_relative_error=np.median(
                    np.abs(errors), axis=0
                ).tolist()
                if errors
                else None,
            )
    return jsonable(
        dict(
            all_attempt_denominator=len(tasks),
            groups=groups,
            qualification=False,
            production_policy_selected=False,
        )
    )


def report_store(tasks, store):
    """Read sealed attempts; cross-score without changing search checkpoints."""
    inventory = store.verify_inventory(tasks)
    records = [store.read_attempt(task, inventory) for task in tasks]
    lookup = {r["id"]: r for r in records}
    enriched = jsonable(records)
    analyses = []
    for task, row in zip(tasks, enriched, strict=True):
        if task["configuration"] == "audit" or row["status"] != "completed":
            continue
        fit = row["result"]
        if "numerical" not in fit:
            continue
        inputs = store.read("input:" + task["problem"]["id"])
        audit = lookup[task["problem"]["id"] + ":audit"]
        if inputs["status"] != "completed":
            continue
        arrays = inputs["result"]
        fit["recovery"] = component_recovery(
            task["problem"], arrays["truth"], fit["parameters"]
        )
        if audit["status"] == "completed" and audit["result"].get("sealed"):
            a = SimpleNamespace(**audit["result"])
            p = SimpleNamespace(
                **(
                    fit
                    | dict(
                        parameters=SimpleNamespace(**fit["parameters"])
                        if fit["parameters"]
                        else None,
                        numerical=SimpleNamespace(**fit["numerical"]),
                    )
                )
            )
            fit["cross_score"] = cross_score(
                arrays["time"], arrays["observed"], arrays["kernel"], p, a
            )
        analyses.append(
            dict(
                id=task["id"],
                recovery=fit["recovery"],
                cross_score=fit.get("cross_score"),
            )
        )
    # Derived analysis is separate from the original hashed checkpoint payloads.
    store.verify_inventory(tasks)
    return dict(
        summary=summarize_tasks(tasks, enriched),
        attempts=records,
        analyses=analyses,
        provenance=dict(
            execution_sha256=digest(store.lock),
            inventory_sha256=inventory["payload_sha256"],
            verified=True,
        ),
    )


def execute_task(task, arrays, *, pilot=False):
    """One explicit task, used only after an external execution-lock gate."""
    c = task["problem"]["condition"]
    settings = settings_for_task(task["configuration"], pilot=pilot)
    if task["configuration"] == "audit":
        return run_audit(
            arrays["time"],
            arrays["observed"],
            arrays["kernel"],
            lifetime_bounds=c["lifetime_bounds_ns"],
            shift_bounds=c["shift_bounds_ns"],
            settings=settings,
        )
    return _fit_triexponential(
        arrays["time"],
        arrays["observed"],
        arrays["kernel"],
        lifetime_bounds_ns=c["lifetime_bounds_ns"],
        temporal_shift_bounds_ns=c["shift_bounds_ns"],
        settings=settings,
        target_kind="deterministic_expected_counts"
        if task["problem"]["seed"] is None
        else "raw_counts",
    )
