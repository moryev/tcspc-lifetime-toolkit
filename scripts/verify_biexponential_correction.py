"""Replay previously seen observations after the authorized initialization fix.

This is regression verification, never a new untouched held-out evaluation.
The original experiment runner and its source-hash guard remain unchanged.
Requires the historical local artifacts listed in the committed hash ledger.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import runpy

import numpy as np
import scipy

from tcspc_toolkit.estimation.multiexponential import BiexponentialFitSettings
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_expected_counts,
)


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_SHA256 = "b89aebfcf4e9dba8cccd0758a48b47a621c685e90a0d68d024a273677409674f"
NEW_DIAGNOSTICS = {
    "initialization_rejected_candidates": 0,
    "initialization_scaling_failed": False,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable(value):
    """Exclude elapsed times and the explicitly different experiment label."""
    if isinstance(value, dict):
        return {
            key: comparable(item)
            for key, item in value.items()
            if key != "mode" and not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [comparable(item) for item in value]
    return value


def changed_paths(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        return [
            changed
            for key in sorted(old.keys() | new.keys())
            for changed in (
                [f"{path}/{key}"]
                if key not in old or key not in new
                else changed_paths(old[key], new[key], f"{path}/{key}")
            )
        ]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            changed
            for index, (left, right) in enumerate(zip(old, new, strict=True))
            for changed in changed_paths(left, right, f"{path}/{index}")
        ]
    return [] if old == new else [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    ledger_path = ROOT / "docs/validation/issue28_pr2_results.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    lock_path = ROOT / "data/generated/issue28_pr2_policy_lock_v1.json"
    history_path = ROOT / "data/generated/issue28_pr2_held_out_v1.jsonl"
    for path in (lock_path, history_path):
        if digest(path) != ledger["artifacts"][path.name]["sha256"]:
            raise ValueError(f"historical artifact hash mismatch: {path.name}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    header, historical_rows = history[0], history[1:]
    runner_path = ROOT / "scripts/benchmark_biexponential_initialization.py"
    config_path = ROOT / "configs/issue28_pr2_initialization.json"
    source_path = ROOT / "src/tcspc_toolkit/estimation/multiexponential.py"
    for path, key, historical_key in (
        (runner_path, "frozen_runner_sha256", "runner_sha256"),
        (config_path, "config_sha256", "config_sha256"),
    ):
        if digest(path) != lock[key] or header[historical_key] != lock[key]:
            raise ValueError(f"frozen input mismatch: {path.name}")
    if header["implementation_sha256"] != lock["frozen_implementation_sha256"]:
        raise ValueError("historical source hash does not match its policy lock")
    if digest(source_path) != CORRECTED_SHA256:
        raise ValueError("this audit is specific to the reviewed initialization fix")
    settings = BiexponentialFitSettings(**lock["selected_settings"])
    if asdict(settings) != asdict(BiexponentialFitSettings()):
        raise ValueError("current defaults differ from the original frozen policy")
    experiment = runpy.run_path(str(runner_path))
    protocol = experiment["load_protocol"]()
    expected_pairs = {(c["id"], 2) for c in protocol["conditions"]}
    if (
        len(historical_rows) != 20
        or {(r["condition_id"], r["stream"]) for r in historical_rows} != expected_pairs
        or any(
            r["mode"] != "held-out" or r["settings"] != lock["selected_settings"]
            for r in historical_rows
        )
    ):
        raise ValueError(
            "historical observations/settings are incomplete or inconsistent"
        )
    protected_paths = (ledger_path, lock_path, history_path, runner_path, config_path)
    protected = {str(p.relative_to(ROOT)): digest(p) for p in protected_paths}
    report = dict(
        purpose="Post-correction regression replay of 20 previously seen observations",
        untouched_held_out_evaluation=False,
        started_utc=datetime.now(timezone.utc).isoformat(),
        historical_source_sha256=header["implementation_sha256"],
        corrected_source_sha256=digest(source_path),
        audit_script_sha256=digest(Path(__file__)),
        protected_artifact_sha256=protected,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        settings=asdict(settings),
        comparison="Exact equality of all historical non-timing fields, including every start; mode differs by design. New diagnostic fields must be zero/false.",
        cases=[],
    )
    # Reserve distinct new outputs before doing work. Never overwrite evidence.
    with args.output.open("x", encoding="utf-8") as summary_handle:
        with args.raw_output.open("x", encoding="utf-8") as raw_handle:
            for old in historical_rows:
                condition, stream = old["condition_id"], old["stream"]
                time, kernel, _, counts, _, case = experiment["build_case"](
                    protocol, condition, stream
                )
                p = old["parameters"]
                mu = multiexponential_reconvolution_expected_counts(
                    time,
                    kernel,
                    amplitudes=p["amplitudes"],
                    lifetimes=p["lifetimes_ns"],
                    background=p["background"],
                    temporal_shift=p["temporal_shift_ns"],
                )
                historical_nll = 0.5 * calculate_poisson_deviance(counts, mu)
                if historical_nll != old["centered_nll"]:
                    raise ValueError(
                        f"historical objective not reproduced for case {condition}"
                    )
                current = experiment["_jsonable"](
                    experiment["fit_row"](
                        protocol,
                        condition,
                        stream,
                        settings,
                        "post-correction-regression-2026-10-25",
                    )
                )
                experiment["write_row"](raw_handle, current)
                compared = comparable(current)
                new_diagnostics = {
                    key: compared["optimization"].pop(key) for key in NEW_DIAGNOSTICS
                }
                differences = changed_paths(comparable(old), compared)
                if new_diagnostics != NEW_DIAGNOSTICS:
                    differences.append("/optimization/new_initialization_diagnostics")
                opt = current["optimization"]
                row = dict(
                    condition_id=condition,
                    condition=old["condition"],
                    stream=stream,
                    counts_sha256=hashlib.sha256(
                        counts.astype("<i8").tobytes()
                    ).hexdigest(),
                    lifetime_bounds_ns=case["lifetime_bounds_ns"],
                    temporal_shift_bounds_ns=case["temporal_shift_bounds_ns"],
                    exact_non_timing_match=not differences,
                    differing_fields=differences,
                    centered_nll=current["centered_nll"],
                    nll_difference=current["centered_nll"] - old["centered_nll"],
                    parameters=current["parameters"],
                    numerical_validation_passed=opt["numerical_validation_passed"],
                    identifiability=current["identifiability"],
                    exhausted_budgets=opt["exhausted_budgets"],
                    evaluations={
                        key: value
                        for key, value in opt.items()
                        if key.endswith("_evaluations")
                    },
                    new_initialization_diagnostics=new_diagnostics,
                    historical_total_seconds=old["optimization"]["total_seconds"],
                    replay_total_seconds=opt["total_seconds"],
                )
                report["cases"].append(row)
                print(
                    f"case={condition} exact_match={not differences} accepted={opt['numerical_validation_passed']} calls={opt['objective_evaluations']} seconds={opt['total_seconds']:.3f}",
                    flush=True,
                )
            report["protected_artifacts_unchanged"] = all(
                digest(ROOT / path) == value for path, value in protected.items()
            )
            report["raw_output_sha256"] = digest(args.raw_output)
            report["completed_utc"] = datetime.now(timezone.utc).isoformat()
            report["passed"] = report["protected_artifacts_unchanged"] and all(
                row["exact_non_timing_match"] and row["numerical_validation_passed"]
                for row in report["cases"]
            )
            json.dump(report, summary_handle, indent=2, allow_nan=False)
            summary_handle.write("\n")
    if not report["passed"]:
        raise SystemExit("Audit differences require investigation before publication")
    print(
        "PASS: all 20 non-timing results match exactly; historical artifacts unchanged."
    )


if __name__ == "__main__":
    main()
