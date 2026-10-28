"""Issue #33: four no-retry, bounded, passive diagnostic replays.

No observation sampling, initializer bank, audit optimization or fitter edits.
The historical runner is imported only for its identity/serialization helpers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import scipy.optimize._lbfgsb_py as lbfgsb
from scipy.special import kl_div
from threadpoolctl import threadpool_limits

import tcspc_toolkit.estimation.multiexponential as fitter
from tcspc_toolkit.benchmarks.multiexponential import build_assumed_irf


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/issue33_stage1_replay.json"
SOURCES = (
    "scripts/diagnose_biexponential_termination.py",
    "tests/test_biexponential_termination_replay.py",
    "configs/issue33_stage1_replay.json",
    "docs/issue33_stage1_protocol.md",
)


def clean(value):
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return float(value)
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def encode(value):
    # Explicit nonfinite strings, including the sign, rather than ambiguous nulls.
    return json.dumps(
        clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha(path, *, lf=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encode(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def utc():
    return datetime.now(timezone.utc).isoformat()


def helpers():
    return runpy.run_path(str(ROOT / "scripts/qualify_biexponential_reconvolution.py"))


def inventory(directory, runner):
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        data = path.read_bytes()
        rows.append(
            dict(
                name=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        )
    return dict(
        files=len(rows),
        bytes=sum(r["bytes"] for r in rows),
        sha256=runner["digest"](rows),
    )


def load_case(config, replicate):
    runner = helpers()
    protocol = runner["load_protocol"]()
    lock = json.loads((ROOT / config["historical_execution_lock"]).read_text())["lock"]
    if runner["source_lock"](protocol) != lock:
        raise ValueError("Historical source/settings/environment lock mismatch")
    directory = ROOT / config["historical_directory"]
    if not (directory / "lock.json").is_file():
        raise ValueError("Existing historical directory required; never initialize it")
    store = runner["Checkpoints"](directory, lock)
    keys = {
        "observation": f"observation-s0-c00-r{replicate:03d}",
        "production": f"production-s0-c00-r{replicate:03d}-a0",
        "audit": f"audit-s0-c00-r{replicate:03d}-a0",
    }
    records = {name: store.get(key) for name, key in keys.items()}
    if any(value is None for value in records.values()):
        raise ValueError("Missing saved input; regeneration is forbidden")
    observation, production, audit = (records[k] for k in keys)
    time = np.asarray(observation["time_ns"], dtype=float)
    irf = build_assumed_irf(protocol, time, "gaussian_0.3")
    if not (
        runner["digest"](irf.kernel)
        == production["assumed_irf_sha256"]
        == audit["assumed_irf_sha256"]
        == observation["generating_irf_sha256"]
    ):
        raise ValueError("Reconstructed kernel differs from saved IRF identity")
    if not (
        runner["digest"](observation["counts"])
        == production["observation_sha256"]
        == audit["observation_sha256"]
    ):
        raise ValueError("Observation identity mismatch")
    records["kernel"] = irf.kernel
    records["input_hashes"] = {
        key + ".json": sha(directory / (key + ".json")) for key in keys.values()
    }
    return records


def make_problem(records, cls=fitter._Problem, **kwargs):
    assumptions = records["production"]["assumptions"]
    observation = records["observation"]
    problem = cls(
        np.array(observation["time_ns"], dtype=float),
        np.array(observation["counts"], dtype=float),
        np.array(records["kernel"], dtype=float, copy=True),
        tuple(assumptions["lifetime_bounds_ns"]),
        tuple(assumptions["temporal_shift_bounds_ns"]),
        fitter.BiexponentialFitSettings(**assumptions["settings"]),
        **kwargs,
    )
    scales = assumptions["parameter_scales"]
    if not (
        scales[0] == scales[1]
        and scales[2:4] == [1, 1]
        and scales[4] == problem.b_scale
        and scales[5] == problem.shift_scale
    ):
        raise ValueError("Saved scales incompatible with reconstructed problem")
    problem.a_scale = scales[0]  # Saved post-initialization scale; no initializer call.
    return problem


def vector(p):
    return np.array([*p.amplitudes, *p.lifetimes_ns, p.background, p.temporal_shift_ns])


def reconstruct(problem, physical):
    p = fitter.BiexponentialParameters(**physical)
    raw = problem.coordinates(p)
    x = np.clip(raw, problem.lower, problem.upper)
    roundtrip = vector(problem.physical(x))
    compatible = []
    for i in range(6):
        neighbors = []
        for direction in (-np.inf, np.inf):
            trial = x.copy()
            trial[i] = np.nextafter(trial[i], direction)
            neighbors.append(
                bool(
                    problem.feasible(trial)
                    and np.array_equal(vector(problem.physical(trial)), roundtrip)
                )
            )
        compatible.append(neighbors)
    return x, dict(
        unbounded_inverse=raw,
        coordinates=x,
        coordinates_hex=[float(v).hex() for v in x],
        physical_roundtrip=roundtrip,
        physical_roundtrip_minus_saved=roundtrip - vector(p),
        clipping_delta=x - raw,
        adjacent_coordinates_with_same_physical_value=compatible,
        historical_internal_coordinates_available=False,
    )


class ReplayLimit(Exception):
    pass


class Recorder:
    def __init__(self, handle=None, *, maximum=2000, seconds=300, clock=perf_counter):
        self.handle, self.maximum, self.seconds, self.clock = (
            handle,
            maximum,
            seconds,
            clock,
        )
        self.started = clock()
        self.calls = 0
        self.counts = Counter()
        self.phase = "optimization"
        self.sequence = 0
        self.callbacks = []
        self.transitions = []
        self.solver = None
        self.shared_inputs = []

    def check_time(self):
        if self.clock() - self.started >= self.seconds:
            raise ReplayLimit("wall_time_limit")

    def charge(self, engine):
        self.check_time()
        if self.calls >= self.maximum:
            raise ReplayLimit("forward_call_limit")
        self.calls += 1
        self.counts[self.phase + ":" + engine] += 1

    def emit(self, event, **fields):
        self.sequence += 1
        if self.handle is not None:
            self.handle.write(
                encode(
                    dict(
                        event=event,
                        sequence=self.sequence,
                        phase=self.phase,
                        total_forward_calls=self.calls,
                        **fields,
                    )
                )
                + "\n"
            )
            self.handle.flush()


class ObservedProblem(fitter._Problem):
    def __init__(self, *args, recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self.curves = {}
        self.current_curve = None
        self.callback_id = None

    def model(self, p, phase):
        result = super().model(p, phase)
        self.current_curve = result[0]
        return result

    def evaluate(self, x, phase="optimization"):
        self.recorder.check_time()
        before = self.recorder.calls
        self.current_curve = None
        value = super().evaluate(x, phase)
        details = {}
        if self.current_curve is not None:
            curve = self.current_curve
            self.curves[np.asarray(x).tobytes()] = curve.copy()
            details = dict(
                curve_min=np.min(curve),
                curve_max=np.max(curve),
                positive_observation_zero_expectation_bins=np.flatnonzero(
                    (self.targets > 0) & (curve == 0)
                ),
                curve_sha256=hashlib.sha256(curve.tobytes()).hexdigest(),
            )
        self.recorder.emit(
            "objective",
            callback_id=self.callback_id,
            evaluation_phase=phase,
            x=np.array(x, copy=True),
            feasible=self.feasible(x),
            value=value,
            actual_calls=self.recorder.calls - before,
            local_best=self.local_value,
            global_best=self.best_value,
            **details,
        )
        return value

    def value_and_gradient(self, x):
        self.callback_id = len(self.recorder.callbacks)
        first = self.recorder.calls
        value, gradient = super().value_and_gradient(x)
        record = dict(
            callback_id=self.callback_id,
            x=np.array(x, copy=True),
            x_hex=[float(v).hex() for v in x],
            value=value,
            gradient=np.array(gradient, copy=True),
            forward_calls=self.recorder.calls - first,
            feasible=self.feasible(x),
        )
        self.recorder.callbacks.append(record)
        self.recorder.emit("value_gradient", **record)
        self.callback_id = None
        return value, gradient


@contextmanager
def instrument(recorder, *, solver_trace=True):
    original_model = fitter.multiexponential_reconvolution_component_counts
    original_minimize = fitter.minimize
    original_setulb = lbfgsb._lbfgsb.setulb

    def model(*args, **kwargs):
        recorder.charge("shared_model")
        recorder.shared_inputs.append(encode(kwargs))
        recorder.emit("forward_call", engine="shared_model", parameters=kwargs)
        return original_model(*args, **kwargs)

    def minimize(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        recorder.solver = {
            name: clean(result[name])
            for name in (
                "x",
                "fun",
                "jac",
                "nit",
                "nfev",
                "njev",
                "status",
                "success",
                "message",
            )
        }
        recorder.emit("solver_result", result=recorder.solver)
        return result

    def state(args):
        return dict(
            x=args[1].copy(),
            f=float(args[5]),
            gradient=args[6].copy(),
            task=args[11].copy(),
            isave=args[13].copy(),
            dsave=args[14].copy(),
            line_search_task=args[16].copy(),
        )

    def setulb(*args):
        recorder.check_time()
        before = state(args)
        result = original_setulb(*args)  # Exactly one original call, unchanged objects.
        after = state(args)
        record = dict(before=before, after=after)
        recorder.transitions.append(record)
        recorder.emit("setulb_transition", **record)
        return result

    with patch.object(fitter, "multiexponential_reconvolution_component_counts", model):
        if solver_trace:
            with (
                patch.object(fitter, "minimize", minimize),
                patch.object(lbfgsb._lbfgsb, "setulb", setulb),
            ):
                yield
        else:
            yield


def independent_derivative(problem, x, recorder):
    """Analytic direct-convolution score, independent of objective differences."""
    p = problem.physical(x)
    t, kernel, y = problem.time, problem.kernel, problem.targets
    dt = t[1] - t[0]
    q = t - p.temporal_shift_ns
    shifted = np.interp(q, t, kernel, left=0, right=0)
    indices = np.clip(np.searchsorted(t, q, side="right") - 1, 0, t.size - 2)
    slope = -(kernel[indices + 1] - kernel[indices]) / (t[indices + 1] - t[indices])
    slope[(q <= t[0]) | (q >= t[-1])] = 0
    at_knot = bool(
        np.any(
            np.min(np.abs(q[:, None] - t[None, :]), axis=1)
            <= 8 * np.finfo(float).eps * max(1, float(np.max(np.abs(t))))
        )
    )

    def convolve(left, right):
        recorder.charge("direct_convolution")
        recorder.emit("forward_call", engine="direct_convolution")
        return np.convolve(left, right)[: t.size] * dt

    units, lifetime_columns, shift_columns = [], [], []
    for amplitude, tau in zip(p.amplitudes, p.lifetimes_ns, strict=True):
        decay = np.exp(-t / tau)
        units.append(convolve(decay, shifted))
        lifetime_columns.append(amplitude * convolve(decay * t / tau, shifted))
        shift_columns.append(amplitude * convolve(decay, slope) * problem.shift_scale)
    mu = p.amplitudes[0] * units[0] + p.amplitudes[1] * units[1] + p.background
    jac = np.column_stack(
        (
            problem.a_scale * units[0],
            problem.a_scale * units[1],
            *lifetime_columns,
            np.full(t.size, problem.b_scale),
            sum(shift_columns),
        )
    )
    value = float(np.sum(np.maximum(kl_div(y, mu), 0)))
    if np.any((y > 0) & (mu == 0)):
        gradient = None
    else:
        score = np.ones_like(mu)
        positive = mu > 0
        score[positive] -= y[positive] / mu[positive]
        gradient = score @ jac
    return dict(
        objective=value,
        gradient=gradient,
        shift_interpolation_knot=at_knot,
        zero_expectation_bins=int(np.sum(mu == 0)),
    ), mu


def source_snapshot():
    return {name: sha(ROOT / name, lf=True) for name in SOURCES}


def external_sources():
    return {
        "scipy_lbfgsb_python": sha(lbfgsb.__file__),
        "scipy_lbfgsb_binary": sha(lbfgsb._lbfgsb.__file__),
    }


def freeze(manifest, directory):
    config = json.loads(CONFIG.read_text())
    runner = helpers()
    original = json.loads((ROOT / config["historical_report"]).read_text())
    current = inventory(ROOT / config["historical_directory"], runner)
    if current["sha256"] != original["integrity"]["sorted_checkpoint_inventory_sha256"]:
        raise ValueError("Historical directory inventory changed")
    input_hashes = {}
    for replicate in sorted({c[0] for c in config["cases"]}):
        input_hashes.update(load_case(config, replicate)["input_hashes"])
    controls_path = ROOT / "build/reports/issue33_stage1_controls.xml"
    suites = ET.parse(controls_path).getroot().findall("testsuite")
    if not suites or any(
        int(s.get("failures", 0)) or int(s.get("errors", 0)) for s in suites
    ):
        raise ValueError("Passing non-qualification control test record required")
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Diagnostic directory must be new")
    historical = (ROOT / config["historical_directory"]).resolve()
    if directory == historical or historical in directory.parents:
        raise ValueError("Never write inside historical evidence")
    payload = dict(
        protocol=config,
        created_utc=utc(),
        sources_lf_sha256=source_snapshot(),
        scipy_sources_sha256=external_sources(),
        historical_inventory=current,
        historical_inputs_sha256=input_hashes,
        historical_execution_lock_sha256=sha(
            ROOT / config["historical_execution_lock"]
        ),
        output_directory=str(directory),
        branch_base=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        nonqualification_controls=dict(
            junit_sha256=sha(controls_path),
            tests=sum(int(s.get("tests", 0)) for s in suites),
            failures=0,
            errors=0,
        ),
    )
    write_new(manifest, payload)
    directory.mkdir(parents=True)
    print(
        encode(
            dict(
                manifest=str(manifest),
                output_directory=str(directory),
                cases=config["cases"],
            )
        )
    )


def verify_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if (
        manifest["sources_lf_sha256"] != source_snapshot()
        or manifest["scipy_sources_sha256"] != external_sources()
    ):
        raise ValueError("Frozen diagnostic implementation changed; do not bypass lock")
    config = manifest["protocol"]
    if (
        json.loads(CONFIG.read_text()) != config
        or sha(ROOT / config["historical_execution_lock"])
        != manifest["historical_execution_lock_sha256"]
    ):
        raise ValueError("Protocol/execution lock changed")
    for name, digest in manifest["historical_inputs_sha256"].items():
        if sha(ROOT / config["historical_directory"] / name) != digest:
            raise ValueError("Historical input changed: " + name)
    return manifest


def worker(manifest_path, replicate, index):
    manifest = verify_manifest(manifest_path)
    config = manifest["protocol"]
    if [replicate, index] not in config["cases"]:
        raise ValueError("Unapproved start")
    directory = Path(manifest["output_directory"]) / f"r{replicate:03d}-s{index:02d}"
    if not (directory / "attempt.json").is_file():
        raise ValueError("Parent attempt marker required")
    records = load_case(config, replicate)
    saved = records["production"]["optimization"]["starts"][index]
    result = dict(
        replicate=replicate,
        start_index=index,
        saved_start=saved,
        input_hashes=records["input_hashes"],
        saved_production_objective=records["audit"]["production_centered_nll"],
        saved_audit_objective=records["audit"]["shared_centered_nll"],
        status="incomplete",
        post_checks=[],
    )
    with (directory / "trace.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        recorder = Recorder(
            handle,
            maximum=config["maximum_forward_calls_per_replay"],
            seconds=config["maximum_seconds_per_replay"],
        )
        problem = make_problem(records, ObservedProblem, recorder=recorder)
        x, result["reconstruction"] = reconstruct(problem, saved["initial_parameters"])
        try:
            with (
                threadpool_limits(limits=config["numerical_threads"]),
                instrument(recorder),
            ):
                diagnostic = fitter._refine(problem, x, index, saved["kind"])
                result["replayed_start"] = diagnostic
                result["solver"] = recorder.solver
                result["optimization_forward_calls"] = recorder.calls
                best_x = None if problem.local_x is None else problem.local_x.copy()
                result["best_coordinates"] = best_x
                result["historical_comparison"] = dict(
                    best_objective_difference=diagnostic.best_centered_nll
                    - saved["best_centered_nll"],
                    endpoint_objective_difference=None
                    if saved["endpoint_centered_nll"] is None
                    else diagnostic.endpoint_centered_nll
                    - saved["endpoint_centered_nll"],
                    forward_call_difference=diagnostic.objective_evaluations
                    - saved["objective_evaluations"],
                    termination_matches=diagnostic.message == saved["message"],
                    best_minus_saved_production=diagnostic.best_centered_nll
                    - result["saved_production_objective"],
                    best_minus_saved_audit=diagnostic.best_centered_nll
                    - result["saved_audit_objective"],
                )
                finite_callbacks = [
                    c for c in recorder.callbacks if np.isfinite(c["value"])
                ]
                result["solver_callback_summary"] = dict(
                    count=len(recorder.callbacks),
                    nonfinite_indices=[
                        c["callback_id"]
                        for c in recorder.callbacks
                        if not np.isfinite(c["value"])
                    ],
                    finite_gradient_infinity_norms=[
                        float(np.max(np.abs(c["gradient"]))) for c in finite_callbacks
                    ],
                )
                points = []
                if finite_callbacks:
                    points += [
                        ("first_finite_callback", finite_callbacks[0]["x"].copy()),
                        ("last_finite_callback", finite_callbacks[-1]["x"].copy()),
                    ]
                if best_x is not None:
                    points.append(("retained_best", best_x))
                    recorder.phase = "post_local_validation"
                    passed, descent, _ = fitter._local_validation(problem, best_x)
                    result["post_local_validation"] = dict(
                        passed=passed,
                        detectable_descent=descent,
                        calls=problem.counts["validation"],
                    )
                derivative_problem = make_problem(
                    records, ObservedProblem, recorder=recorder
                )
                for role, p in (
                    ("saved_production", records["production"]["bi_parameters"]),
                    ("saved_audit", records["audit"]["parameters"]),
                ):
                    points.append((role, reconstruct(derivative_problem, p)[0]))
                recorder.phase = "post_derivative_checks"
                for role, point in points:
                    recorder.check_time()
                    value, gradient = derivative_problem.value_and_gradient(point)
                    direct, mu = independent_derivative(
                        derivative_problem, point, recorder
                    )
                    center = derivative_problem.curves.get(point.tobytes())
                    difference = (
                        None
                        if direct["gradient"] is None
                        else gradient - direct["gradient"]
                    )
                    result["post_checks"].append(
                        dict(
                            role=role,
                            x=point,
                            physical=derivative_problem.physical(point),
                            shared_objective=value,
                            production_finite_difference_gradient=gradient,
                            direct=direct,
                            gradient_difference=difference,
                            gradient_relative_difference=None
                            if difference is None
                            else np.abs(difference)
                            / np.maximum(1, np.abs(direct["gradient"])),
                            curve_relative_l2=None
                            if center is None
                            else float(
                                np.linalg.norm(center - mu) / np.linalg.norm(mu)
                            ),
                        )
                    )
                result["status"] = "completed"
        except ReplayLimit as error:
            result["status"] = str(error)
            recorder.emit("budget_stop", reason=str(error))
        finally:
            result["total_forward_calls"] = recorder.calls
            result["calls_by_phase_engine"] = dict(recorder.counts)
            result["elapsed_seconds"] = recorder.clock() - recorder.started
            result["best_retained_even_if_incomplete"] = dict(
                value=problem.local_value, x=problem.local_x
            )
            result["solver"] = recorder.solver
            result["trace_events"] = recorder.sequence
    result["trace_sha256"] = sha(directory / "trace.jsonl")
    write_new(directory / "result.json", result)


def run(manifest_path):
    manifest = verify_manifest(manifest_path)
    config = manifest["protocol"]
    directory = Path(manifest["output_directory"])
    # Deliberately no automatic resume/retry. A marker is an attempted replay.
    if any(
        (directory / f"r{r:03d}-s{s:02d}" / "attempt.json").exists()
        for r, s in config["cases"]
    ):
        raise FileExistsError("Existing attempt(s); do not retry or overwrite")
    for replicate, index in config["cases"]:
        case = directory / f"r{replicate:03d}-s{index:02d}"
        write_new(
            case / "attempt.json",
            dict(
                started_utc=utc(),
                manifest_sha256=sha(manifest_path),
                replicate=replicate,
                start_index=index,
            ),
        )
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "worker",
            "--manifest",
            str(Path(manifest_path).resolve()),
            "--replicate",
            str(replicate),
            "--index",
            str(index),
        ]
        started = perf_counter()
        with (case / "process.log").open("xb") as handle:
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
            )
            timed_out = wait_bounded(
                child,
                max(
                    0, config["maximum_seconds_per_replay"] - (perf_counter() - started)
                ),
            )
        if not (case / "result.json").exists():
            write_new(
                case / "result.json",
                dict(
                    status="external_timeout" if timed_out else "worker_failed",
                    exit_code=child.returncode,
                    replicate=replicate,
                    start_index=index,
                    trace_preserved=(case / "trace.jsonl").exists(),
                ),
            )
        write_new(
            case / "process.json",
            dict(
                wall_seconds=perf_counter() - started,
                exit_code=child.returncode,
                timed_out=timed_out,
            ),
        )
        print(
            encode(
                dict(
                    replicate=replicate,
                    start_index=index,
                    status=json.loads((case / "result.json").read_text())["status"],
                )
            ),
            flush=True,
        )


def wait_bounded(child, seconds):
    """Kill the Windows venv launcher AND its child interpreter on timeout."""
    try:
        child.wait(timeout=seconds)
        return False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                check=True,
                capture_output=True,
            )
        else:
            child.kill()
        child.wait()
        return True


def summarize(manifest_path, output):
    manifest = verify_manifest(manifest_path)
    config = manifest["protocol"]
    current = inventory(ROOT / config["historical_directory"], helpers())
    if current != manifest["historical_inventory"]:
        raise ValueError("Historical evidence changed")
    results = []
    for replicate, index in config["cases"]:
        directory = (
            Path(manifest["output_directory"]) / f"r{replicate:03d}-s{index:02d}"
        )
        result = json.loads((directory / "result.json").read_text())
        if (
            "trace_sha256" in result
            and sha(directory / "trace.jsonl") != result["trace_sha256"]
        ):
            raise ValueError("Trace identity changed")
        result["process"] = json.loads((directory / "process.json").read_text())
        result["replay_complete"] = (
            result["status"] == "completed"
            and not result["process"]["timed_out"]
            and result["process"]["exit_code"] == 0
        )
        results.append(result)
    write_new(
        output,
        dict(
            protocol=config["protocol"],
            manifest_sha256=sha(manifest_path),
            historical_inventory_unchanged=current,
            results=results,
            created_utc=utc(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "summarize"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.manifest, args.directory)
    elif args.command == "worker":
        worker(args.manifest, args.replicate, args.index)
    elif args.command == "run":
        run(args.manifest)
    else:
        summarize(args.manifest, args.output)


if __name__ == "__main__":
    main()
