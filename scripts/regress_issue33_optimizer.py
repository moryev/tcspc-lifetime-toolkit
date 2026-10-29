"""Stage-2B saved-observation regressions, never independent qualification.

Exclusive output directories preserve attempts. C00 reads saved observations;
PR-2 reconstructs previously used stream-2 observations and requires their saved
2026-10-25 count hashes before fitting. No historical file is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import runpy
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from tcspc_toolkit.benchmarks.multiexponential import (
    assess_biexponential_fit,
    build_assumed_irf,
)
from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitSettings,
    fit_biexponential_reconvolution,
)


ROOT = Path(__file__).resolve().parents[1]
PR2_CASES = (0, 3, 4, 5, 6, 9, 12, 13, 15, 17, 18, 19)


def sha(path, *, lf=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def c00_inputs():
    runner = runpy.run_path(
        str(ROOT / "scripts/qualify_biexponential_reconvolution.py")
    )
    protocol = runner["load_protocol"]()
    config = json.loads((ROOT / "configs/issue33_stage1_replay.json").read_text())
    manifest = json.loads(
        (ROOT / "docs/validation/issue33_stage1_manifest_2026-10-27.json").read_text()
    )
    directory = ROOT / config["historical_directory"]
    lock = json.loads((ROOT / config["historical_execution_lock"]).read_text())["lock"]
    # Read saved envelopes directly; never construct a checkpoint writer here.
    assert json.loads((directory / "lock.json").read_text()) == lock
    for replicate in (11, 58):
        records, hashes = {}, {}
        for kind in ("observation", "production", "audit"):
            key = f"{kind}-s0-c00-r{replicate:03d}" + (
                "" if kind == "observation" else "-a0"
            )
            path = directory / f"{key}.json"
            assert sha(path) == manifest["historical_inputs_sha256"][path.name]
            hashes[str(path.relative_to(ROOT))] = sha(path)
            record = json.loads(path.read_text())
            assert record["key"] == key
            assert record["lock_sha256"] == runner["digest"](lock)
            assert record["payload_sha256"] == runner["digest"](record["payload"])
            records[kind] = record["payload"]
        observation, old, audit = (
            records[k] for k in ("observation", "production", "audit")
        )
        time = np.array(observation["time_ns"])
        irf = build_assumed_irf(protocol, time, "gaussian_0.3")
        assert (
            runner["digest"](irf.kernel)
            == old["assumed_irf_sha256"]
            == audit["assumed_irf_sha256"]
        )
        assert (
            runner["digest"](observation["counts"])
            == old["observation_sha256"]
            == audit["observation_sha256"]
        )
        yield dict(
            case=f"C00-r{replicate:03d}",
            time=time,
            counts=observation["counts"],
            irf=irf,
            bounds={
                key: tuple(old["assumptions"][key])
                for key in ("lifetime_bounds_ns", "temporal_shift_bounds_ns")
            },
            old=old,
            input_hashes=hashes,
            reference_nll=audit["shared_centered_nll"],
            reference_kind="saved_independent_shared_objective",
        )


def pr2_inputs():
    history_path = ROOT / "data/generated/issue28_pr2_held_out_v1.jsonl"
    ledger = json.loads((ROOT / "docs/validation/issue28_pr2_results.json").read_text())
    assert sha(history_path) == ledger["artifacts"][history_path.name]["sha256"]
    history = [json.loads(line) for line in history_path.read_text().splitlines()][1:]
    previous = json.loads(
        (
            ROOT / "docs/validation/issue28_pr2_post_correction_2026-10-25.json"
        ).read_text()
    )
    experiment = runpy.run_path(
        str(ROOT / "scripts/benchmark_biexponential_initialization.py")
    )
    protocol = experiment["load_protocol"]()
    for condition in PR2_CASES:
        old = next(r for r in history if r["condition_id"] == condition)
        verified = next(r for r in previous["cases"] if r["condition_id"] == condition)
        time, irf, _, counts, _, case = experiment["build_case"](protocol, condition, 2)
        count_hash = hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest()
        assert count_hash == verified["counts_sha256"]
        yield dict(
            case=f"PR2-c{condition:02d}-{case['name']}",
            time=time,
            counts=counts,
            irf=irf,
            bounds={
                key: tuple(case[key])
                for key in ("lifetime_bounds_ns", "temporal_shift_bounds_ns")
            },
            old=old,
            input_hashes={
                str(history_path.relative_to(ROOT)): sha(history_path),
                "counts_int64_sha256": count_hash,
            },
            reference_nll=old["centered_nll"],
            reference_kind="historical_pr2_production_objective",
        )


def assess_result(result, case, rules):
    """Run the unchanged full gate; its work is external to fitter counters."""
    started = perf_counter()
    irf = case["irf"]
    assessment = assess_biexponential_fit(
        result,
        case["time"],
        case["counts"],
        irf.kernel if hasattr(irf, "kernel") else irf,
        rules,
    )
    elapsed = perf_counter() - started
    opt = result.optimization
    fitter_calls = sum(
        (
            opt.initialization_evaluations,
            opt.objective_evaluations,
            opt.validation_evaluations,
            opt.diagnostic_evaluations,
        )
    )
    return dict(
        numerical_acceptance=assessment.accepted,
        local_numerical_validation_passed=opt.numerical_validation_passed,
        numerical_assessment=asdict(assessment),
        external_assessment_seconds=elapsed,
        fitter_forward_evaluations=fitter_calls,
        total_forward_evaluations=fitter_calls
        + assessment.external_forward_evaluations,
        total_seconds_including_assessment=opt.total_seconds + elapsed,
    )


def save_source_snapshots(sources, output):
    # Estimation and benchmark modules share a basename. Keep relative paths.
    for source in sources:
        destination = output / "sources" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("c00", "pr2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    serializer = runpy.run_path(
        str(ROOT / "scripts/diagnose_biexponential_termination.py")
    )["encode"]
    settings = BiexponentialFitSettings()
    acceptance_path = ROOT / "configs/issue28_pr3_qualification.json"
    rules = json.loads(acceptance_path.read_text())["acceptance"]
    sources = [
        ROOT / "src/tcspc_toolkit/estimation/multiexponential.py",
        Path(__file__),
        ROOT / "src/tcspc_toolkit/benchmarks/multiexponential.py",
        acceptance_path,
    ]
    manifest = dict(
        purpose="Issue-33 Stage-2B development regressions; not independent qualification",
        group=args.group,
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_lf_sha256={str(p.relative_to(ROOT)): sha(p, lf=True) for p in sources},
        settings=asdict(settings),
        acceptance_rules=rules,
        numerical_threads=1,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        cases=["C00-r011", "C00-r058"] if args.group == "c00" else list(PR2_CASES),
    )
    (args.output / "manifest.json").write_text(serializer(manifest) + "\n")
    save_source_snapshots(sources, args.output)
    rows = []
    with threadpool_limits(limits=1):
        for case in c00_inputs() if args.group == "c00" else pr2_inputs():
            result = fit_biexponential_reconvolution(
                case["time"],
                case["counts"],
                case["irf"],
                **case["bounds"],
                settings=settings,
            )
            opt = asdict(result.optimization)
            value = (
                result.statistics.centered_negative_log_likelihood
                if result.statistics
                else None
            )
            row = dict(
                case=case["case"],
                input_hashes=case["input_hashes"],
                bounds=case["bounds"],
                parameters=asdict(result.parameters) if result.parameters else None,
                centered_nll=value,
                reference_nll=case["reference_nll"],
                reference_kind=case["reference_kind"],
                objective_gap=None if value is None else value - case["reference_nll"],
                **assess_result(result, case, rules),
                start_kinds=dict(Counter(s["kind"] for s in opt["starts"])),
                optimization=opt,
                identifiability=asdict(result.identifiability),
                historical_optimization=case["old"]["optimization"],
            )
            with (args.output / f"{case['case']}.json").open("x") as handle:
                handle.write(serializer(row) + "\n")
            compact = {
                k: v
                for k, v in row.items()
                if k not in ("optimization", "historical_optimization")
            }
            compact["optimization"] = {k: v for k, v in opt.items() if k != "starts"}
            compact["historical_optimization"] = {
                k: v for k, v in row["historical_optimization"].items() if k != "starts"
            }
            compact["raw_sha256"] = sha(args.output / f"{case['case']}.json")
            rows.append(compact)
            print(
                case["case"],
                value,
                "gap",
                row["objective_gap"],
                "accepted",
                row["numerical_acceptance"],
                "starts",
                row["start_kinds"],
                "calls",
                opt["objective_evaluations"],
                "seconds",
                opt["total_seconds"],
                flush=True,
            )
    assert all(
        sha(ROOT / p, lf=True) == digest
        for p, digest in manifest["source_lf_sha256"].items()
    )
    with (args.output / "results.json").open("x") as handle:
        handle.write(serializer(dict(manifest=manifest, cases=rows)) + "\n")


if __name__ == "__main__":
    main()
