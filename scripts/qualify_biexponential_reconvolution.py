"""Issue-28 PR-3 qualification: inspect, lock, non-qualification pilot, resume.

The committed manifest blocks scientific observations until maintainer review.
No execution occurs on import. Raw checkpoints belong under data/generated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from tcspc_toolkit.benchmarks.multiexponential import (
    assess_biexponential_fit,
    build_assumed_irf,
    build_calibration_proxy,
    build_generating_case,
    component_recovery,
    independent_biexponential_audit,
    sample_case,
    summarize_recovery,
    validate_protocol,
)
from tcspc_toolkit.estimation.multiexponential import BiexponentialFitSettings
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.measurement.adapters import compare_experimental_reconvolution
from tcspc_toolkit.measurement.records import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/issue28_pr3_qualification.json"


def jsonable(value):
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def encoded(value):
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_protocol():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def source_lock(protocol):
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__), CONFIG]
    sources = {p.relative_to(ROOT).as_posix(): source_digest(p) for p in files}
    fitter = "src/tcspc_toolkit/estimation/multiexponential.py"
    if sources[fitter] != protocol["frozen_pr2_lf_sha256"]:
        raise ValueError("PR-2 source lock changed")
    return dict(
        protocol_sha256=digest(protocol),
        sources_lf_sha256=sources,
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            packages={
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pandas", "threadpoolctl")
            },
        ),
        numerical_threads=1,
    )


def write_new_json(path, payload):
    """Atomic no-overwrite publication; a crashed temporary file is not a result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    # Partial files are restartable transport, never accepted evidence.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Hard-link publication fails atomically if another process already wrote it.
    os.link(temporary, path)
    temporary.unlink()


class Checkpoints:
    def __init__(self, directory, lock):
        self.directory = Path(directory)
        self.lock = lock
        self.lock_hash = digest(lock)
        path = self.directory / "lock.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != lock:
                raise ValueError(
                    "resume rejected: source/settings/environment lock differs"
                )
        else:
            write_new_json(path, lock)

    def _path(self, key):
        if not key or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in key
        ):
            raise ValueError("unsafe checkpoint key")
        return self.directory / (key + ".json")

    def get(self, key):
        path = self._path(key)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record["key"] != key
            or record["lock_sha256"] != self.lock_hash
            or record["payload_sha256"] != digest(record["payload"])
        ):
            raise ValueError(f"corrupt or foreign checkpoint: {key}")
        return record["payload"]

    def put(self, key, payload):
        previous = self.get(key)
        if previous is not None:
            if digest(previous) != digest(payload):
                raise ValueError(f"attempt to overwrite completed record: {key}")
            return
        write_new_json(
            self._path(key),
            dict(
                key=key,
                lock_sha256=self.lock_hash,
                payload_sha256=digest(payload),
                payload=payload,
            ),
        )


def observation_checkpoint(store, protocol, condition_id, replicate, stream):
    key = f"observation-s{stream}-c{condition_id:02d}-r{replicate:03d}"
    record = store.get(key)
    if record is None:
        case = build_generating_case(protocol, condition_id)
        measurement = sample_case(protocol, case, stream=stream, replicate=replicate)
        record = dict(
            time_ns=measurement.time_ns,
            counts=measurement.values,
            provenance=measurement.provenance,
            sample_id=measurement.sample_id,
            expected_counts_sha256=digest(case.expected_counts),
            generating_irf_sha256=digest(case.irf.kernel),
        )
        store.put(key, record)  # Before either production model or any IRF arm.
        record = store.get(key)
    return key, record


def resolve_arm(protocol, case, arm, stream, proxy_record=None):
    if arm == "matched":
        return case.irf
    if arm != "calibration_proxy":
        return build_assumed_irf(protocol, case.irf.time_ns, arm)
    if proxy_record is None:
        raise ValueError("proxy calibration must be checkpointed before inference")
    if proxy_record["kernel"] is None:
        return None
    # Reconstruct the full provenance carrier rather than pretending an array
    # has verified preparation metadata.
    from tcspc_toolkit.irf.preparation import IRFPreparationDiagnostics, PreparedIRF
    from tcspc_toolkit.irf.profiles import IRFProfile, IRFSourceKind

    source = proxy_record["source"]
    profile = IRFProfile(
        source["time_ns"],
        source["values"],
        IRFSourceKind(source["source_kind"]),
        source["source_parameters"],
        source["metadata"],
        source["provenance"],
    )
    diagnostics = dict(proxy_record["preparation"])
    for name in ("operations", "flags"):
        diagnostics[name] = tuple(diagnostics[name])
    return PreparedIRF(
        profile,
        case.irf.time_ns,
        proxy_record["kernel"],
        IRFPreparationDiagnostics(**diagnostics),
    )


def proxy_checkpoint(store, protocol, stream):
    key = f"proxy-s{stream}"
    record = store.get(key)
    if record is None:
        prepared, diagnostics, measurement = build_calibration_proxy(
            protocol, stream=stream
        )
        record = dict(
            kernel=None if prepared is None else prepared.kernel,
            source=None if prepared is None else prepared.source,
            preparation=None if prepared is None else prepared.diagnostics,
            diagnostics=diagnostics,
            counts=measurement.values,
            counts_sha256=digest(measurement.values),
            provenance=measurement.provenance,
        )
        store.put(key, record)
        record = store.get(key)
    return record


def _relative_l2(fitted, reference):
    denominator = np.linalg.norm(reference)
    return (
        None
        if denominator == 0
        else float(np.linalg.norm(fitted - reference) / denominator)
    )


def production_task(protocol, case, measurement, assumed):
    c = case.configuration
    started = perf_counter()
    comparison = compare_experimental_reconvolution(
        measurement,
        prepared_irf=assumed,
        lifetime_bounds_ns=tuple(c["lifetime_bounds_ns"]),
        temporal_shift_bounds_ns=tuple(c["temporal_shift_bounds_ns"]),
        settings=BiexponentialFitSettings(**protocol["fit_settings"]),
        background_fraction=protocol["mono"]["background_fraction"],
    )
    production_seconds = perf_counter() - started
    result = comparison.bi
    started = perf_counter()
    assessment = assess_biexponential_fit(
        result,
        measurement.time_ns,
        measurement.values,
        assumed.kernel,
        protocol["acceptance"],
    )
    recheck_seconds = perf_counter() - started
    curve_metrics = None
    recovery = component_recovery(case, result, assessment.accepted)
    i = result.identifiability
    warnings = (
        *i.structural_warnings,
        *i.conditioning_warnings,
        *i.boundary_warnings,
        *i.multistart_warnings,
    )
    if result.expected_counts is not None and result.parameters is not None:
        fitted_signal = result.component_counts.sum(axis=0)
        curve_metrics = dict(
            total_relative_l2=_relative_l2(
                result.expected_counts, case.expected_counts
            ),
            signal_relative_l2=_relative_l2(
                fitted_signal, case.component_counts.sum(axis=0)
            ),
            signal_photon_error=float(fitted_signal.sum() - c["signal_photons"]),
            background_error=result.parameters.background - c["background"],
            shift_error_ns=result.parameters.temporal_shift_ns - c["shift_ns"],
        )
    return dict(
        status="completed",
        assessment=assessment,
        recovery=recovery,
        warning_free_error_above_5_percent=not warnings
        and any(
            e is not None and abs(e) > 0.05
            for e in recovery["all_finite_component_relative_errors"]
        ),
        mono=comparison.mono,
        statistics=comparison.statistics,
        bounds=comparison.bounds,
        bi_parameters=result.parameters,
        bi_components=result.components,
        optimization=result.optimization,
        identifiability=result.identifiability,
        assumptions=result.assumptions,
        curve_metrics=curve_metrics,
        production_seconds=production_seconds,
        objective_recheck_seconds=recheck_seconds,
    )


def projection_task(protocol, case, arm):
    """Use the unchanged Issue-4 projector only for compatible predeclared cases."""
    c = case.configuration
    p = protocol["pseudo_true"]
    if not (
        (arm == "matched" and c["id"] in p["eligible_matched_conditions"])
        or c["id"] in p["eligible_all_arms_conditions"]
    ):
        return dict(
            status="unavailable",
            reason="outside predeclared projector compatibility subset",
        )
    from tcspc_toolkit.benchmarks.bayesian_mismatch import (
        MismatchAssumption,
        MismatchCondition,
        construct_pseudo_true_reference,
    )

    mono = len(c["lifetimes_ns"]) == 1
    names = c["arms"] if mono else ["matched"]
    assumptions = tuple(
        MismatchAssumption(
            name,
            name,
            case.irf
            if name == "matched"
            else build_assumed_irf(protocol, case.irf.time_ns, name),
        )
        for name in names
    )
    condition = MismatchCondition(
        f"issue28-pr3-c{c['id']:02d}",
        "irf" if mono else "decay",
        "monoexponential" if mono else "biexponential",
        case.irf.time_ns,
        c["generating_irf"],
        case.irf,
        assumptions,
        c["lifetimes_ns"][0],
        None if mono else c["lifetimes_ns"][1],
        None if mono else c["detected_fractions"][1],
        c["signal_photons"],
        c["background"],
        c["shift_ns"],
    )
    try:
        reference = construct_pseudo_true_reference(
            condition,
            next(a for a in assumptions if a.assumption_id == arm),
            case.expected_counts,
            tuple(c["temporal_shift_bounds_ns"]),
        )
    except RuntimeError as exc:
        return dict(status="unavailable", reason=str(exc))
    return dict(
        status="completed",
        reference=reference,
        interpretation="deterministic prior-free model-conditional mono projection, not physical truth",
    )


def worker(store, protocol, task):
    condition_id, replicate, arm, stream, kind = (
        task[k] for k in ("condition_id", "replicate", "arm", "stream", "kind")
    )
    case = build_generating_case(protocol, condition_id)
    if kind == "projection":
        return projection_task(protocol, case, arm)
    observation = store.get(task["observation_key"])
    if observation is None:
        raise ValueError("observation was not checkpointed before fitting")
    if (
        observation["expected_counts_sha256"] != digest(case.expected_counts)
        or observation["generating_irf_sha256"] != digest(case.irf.kernel)
        or observation["time_ns"] != case.irf.time_ns.tolist()
    ):
        raise ValueError("observation/generator mismatch")
    measurement = TCSPCMeasurement(
        observation["time_ns"],
        observation["counts"],
        MeasurementDataKind.RAW_COUNTS,
        sample_id=observation["sample_id"],
        provenance=observation["provenance"],
    )
    proxy_stream = 2 if stream == 2 else 1
    assumed = resolve_arm(
        protocol, case, arm, stream, store.get(f"proxy-s{proxy_stream}")
    )
    if assumed is None:
        return dict(
            status="irf_unavailable", reason="fixed proxy calibration failed; no redraw"
        )
    with threadpool_limits(limits=1):
        if kind == "production":
            output = production_task(protocol, case, measurement, assumed)
        elif kind == "audit":
            output = independent_biexponential_audit(
                measurement.time_ns,
                measurement.values,
                assumed.kernel,
                case.configuration["lifetime_bounds_ns"],
                case.configuration["temporal_shift_bounds_ns"],
                protocol["audit"],
            )
            output["status"] = "completed"
            p = output["parameters"]
            output["shared_forward_recheck_calls"] = int(p is not None)
            output["shared_centered_nll"] = None
            if p is not None:
                expected = (
                    multiexponential_reconvolution_component_counts(
                        measurement.time_ns,
                        assumed.kernel,
                        amplitudes=p["amplitudes"],
                        lifetimes=p["lifetimes_ns"],
                        temporal_shift=p["temporal_shift_ns"],
                    ).sum(axis=0)
                    + p["background"]
                )
                output["shared_centered_nll"] = 0.5 * calculate_poisson_deviance(
                    measurement.values, expected
                )
            production_key = (
                f"production-s{stream}-c{condition_id:02d}-r{replicate:03d}"
                f"-a{task['arm_index']}"
            )
            production = store.get(production_key)
            actual = (
                production.get("assessment", {}).get("recomputed_centered_nll")
                if production is not None
                else None
            )
            difference = (
                None
                if actual is None or output["shared_centered_nll"] is None
                else abs(actual - output["shared_centered_nll"])
            )
            output["production_centered_nll"] = actual
            output["absolute_nll_difference"] = difference
            output["target_applies"] = (
                arm == "matched"
                and condition_id in protocol["audit"]["target_matched_conditions"]
            )
            output["agreement_within_0_01"] = bool(
                output["complete"]
                and difference is not None
                and difference <= protocol["audit"]["maximum_centered_nll_difference"]
            )
        else:
            raise ValueError("unknown worker kind")
    return dict(
        output,
        observation_sha256=digest(observation["counts"]),
        assumed_irf_sha256=digest(assumed.kernel),
        condition_id=condition_id,
        replicate=replicate,
        arm=arm,
        stream=stream,
    )


def execute_task(store, protocol, task):
    """One bounded child per comparison/audit; completed failures are immutable."""
    key = f"{task['kind']}-s{task['stream']}-c{task['condition_id']:02d}-r{task['replicate']:03d}-a{task['arm_index']}"
    previous = store.get(key)
    if previous is not None:
        if previous["status"] == "worker_error":
            raise RuntimeError(f"recorded worker error requires investigation: {key}")
        return previous
    store.put("request-" + key, task)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    timeout = (
        protocol["pseudo_true"]["timeout_seconds"]
        if task["kind"] == "projection"
        else protocol["execution"][
            "audit_timeout_seconds"
            if task["kind"] == "audit"
            else "production_case_timeout_seconds"
        ]
    )
    started = perf_counter()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--directory",
                str(store.directory),
                "--key",
                key,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            # Preserve diagnostics; a programming error stops orchestration after
            # recording, rather than silently becoming a scientific rejection.
            store.put(
                key,
                dict(
                    status="worker_error",
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    elapsed_seconds=perf_counter() - started,
                ),
            )
            raise RuntimeError(f"worker {key} failed: {completed.stderr}")
    except subprocess.TimeoutExpired:
        store.put(
            key,
            dict(
                status="timeout",
                limit_seconds=timeout,
                elapsed_seconds=perf_counter() - started,
            ),
        )
    store.put("runtime-" + key, dict(wall_seconds=perf_counter() - started))
    return store.get(key)


def execute_pilot(store, protocol):
    """Eight predeclared stream-2 production pairs and eight independent audits."""
    proxy_checkpoint(store, protocol, 2)
    rows = []
    for condition_id, arm in protocol["pilot"]["cases"]:
        replicate = protocol["pilot"]["replicate"]
        observation_key, _ = observation_checkpoint(
            store, protocol, condition_id, replicate, 2
        )
        c = protocol["conditions"][condition_id]
        task = dict(
            condition_id=condition_id,
            replicate=replicate,
            arm=arm,
            arm_index=c["arms"].index(arm),
            stream=2,
            observation_key=observation_key,
        )
        production = execute_task(store, protocol, dict(task, kind="production"))
        audit = execute_task(store, protocol, dict(task, kind="audit"))
        rows.append(
            dict(condition_id=condition_id, arm=arm, production=production, audit=audit)
        )
        print(
            json.dumps(
                dict(
                    condition_id=condition_id,
                    arm=arm,
                    production_status=production["status"],
                    audit_status=audit["status"],
                )
            ),
            flush=True,
        )
    store.put("pilot-summary", dict(kind="nonqualification_runtime_pilot", rows=rows))


def execute_study(store, protocol):
    if protocol["execution_status"] != "authorized":
        raise PermissionError(
            "STOP: full scientific qualification awaits maintainer approval"
        )
    proxy_checkpoint(store, protocol, 1)
    # Save each observation once before every competing arm; no policy selection.
    for c in protocol["conditions"]:
        for replicate in range(c["replicates"]):
            key, _ = observation_checkpoint(store, protocol, c["id"], replicate, 0)
            for ai, arm in enumerate(c["arms"]):
                task = dict(
                    condition_id=c["id"],
                    replicate=replicate,
                    arm=arm,
                    arm_index=ai,
                    stream=0,
                    observation_key=key,
                )
                execute_task(store, protocol, dict(task, kind="production"))
                if (
                    c["id"] == 0
                    or replicate in protocol["audit"]["secondary_replicates"]
                ):
                    execute_task(store, protocol, dict(task, kind="audit"))
        for ai, arm in enumerate(c["arms"]):
            execute_task(
                store,
                protocol,
                dict(
                    condition_id=c["id"],
                    replicate=0,
                    arm=arm,
                    arm_index=ai,
                    stream=0,
                    kind="projection",
                ),
            )


def summarize_study(store, protocol):
    summaries = []
    for c in protocol["conditions"]:
        for ai, arm in enumerate(c["arms"]):
            rows = []
            audits = []
            failures = []
            for replicate in range(c["replicates"]):
                row = store.get(f"production-s0-c{c['id']:02d}-r{replicate:03d}-a{ai}")
                if row is None:
                    raise ValueError(
                        "incomplete study; missing attempts cannot disappear from denominators"
                    )
                if row["status"] != "completed":
                    failures.append(dict(replicate=replicate, status=row["status"]))
                    row = dict(
                        assessment=dict(
                            accepted=False, diagnostic_budget_complete=False
                        ),
                        recovery=dict(accepted_component_relative_errors=[None, None]),
                    )
                rows.append(row)
                if (
                    c["id"] == 0
                    or replicate in protocol["audit"]["secondary_replicates"]
                ):
                    audit = store.get(f"audit-s0-c{c['id']:02d}-r{replicate:03d}-a{ai}")
                    if audit is None:
                        raise ValueError("incomplete study; scheduled audit missing")
                    audits.append(dict(audit, replicate=replicate))
            projection = store.get(f"projection-s0-c{c['id']:02d}-r000-a{ai}")
            if projection is None:
                raise ValueError("incomplete study; projection status missing")
            summaries.append(
                dict(
                    condition_id=c["id"],
                    arm=arm,
                    execution_failures=failures,
                    scheduled_audits=audits,
                    pseudo_true_projection=projection,
                    warning_free_error_above_5_percent_replicates=[
                        r
                        for r, row in enumerate(rows)
                        if row.get("warning_free_error_above_5_percent", False)
                    ],
                    numerical_rejection_reasons=[
                        dict(
                            replicate=r,
                            reasons=row["assessment"].get(
                                "reasons", ["execution_failure"]
                            ),
                        )
                        for r, row in enumerate(rows)
                        if not row["assessment"]["accepted"]
                    ],
                    **summarize_recovery(rows, protocol, c["id"], ai),
                )
            )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("inspect", "freeze", "pilot", "study", "summarize", "worker")
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--key")
    args = parser.parse_args()
    protocol = load_protocol()
    if args.mode == "inspect":
        print(
            json.dumps(
                dict(
                    totals=validate_protocol(protocol),
                    status=protocol["execution_status"],
                ),
                indent=2,
            )
        )
        return
    lock = source_lock(protocol)
    if args.mode == "freeze":
        if args.output is None:
            parser.error("freeze requires --output")
        write_new_json(
            args.output,
            dict(created_utc=datetime.now(timezone.utc).isoformat(), lock=lock),
        )
        return
    if args.directory is None:
        parser.error("execution requires --directory")
    store = Checkpoints(args.directory.resolve(), lock)
    if args.mode == "worker":
        request = store.get("request-" + args.key)
        store.put(args.key, worker(store, protocol, request))
    elif args.mode == "pilot":
        execute_pilot(store, protocol)
    elif args.mode == "study":
        execute_study(store, protocol)
    else:
        if args.output is None:
            parser.error("summarize requires --output")
        write_new_json(args.output, summarize_study(store, protocol))


if __name__ == "__main__":
    main()
