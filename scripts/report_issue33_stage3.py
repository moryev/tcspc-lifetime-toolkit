"""Read-only Stage-3 aggregation; never generate observations or rerun tasks.

Execution identity comes from the saved lock. A later reporting revision gets
its own identity and new output paths, never a rewritten execution identity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from qualify_biexponential_reconvolution import (
    digest,
    jsonable,
    source_digest,
    write_new_json,
)
from qualify_issue33_stage3 import (
    TERMINAL,
    exclusive_lease,
    observation_key,
    open_existing,
    permit,
    task_key,
)
from tcspc_toolkit.benchmarks.multiexponential import summarize_recovery
from validate_issue33_stage3_protocol import ROOT, audit_tally


def finite(value):
    return value is not None and np.isfinite(value)


def audit_comparison(production, audit, protocol):
    """Completion, comparison availability and disagreement are separate facts."""
    available = bool(
        production
        and production.get("production_objective_usable")
        and finite(production.get("assessment", {}).get("recomputed_centered_nll"))
        and audit
        and finite(audit.get("shared_centered_nll"))
    )
    gap = (
        production["assessment"]["recomputed_centered_nll"]
        - audit["shared_centered_nll"]
        if available
        else None
    )
    complete = bool(
        audit and audit.get("status") == "completed" and audit.get("complete")
    )
    if audit is None:
        classification = "not_run"
    elif not complete:
        classification = "audit_incomplete"
    elif not available:
        classification = "production_unavailable"
    elif abs(gap) <= protocol["audit"]["maximum_centered_nll_difference"]:
        classification = "agreement"
    else:
        classification = "disagreement"
    return dict(
        classification=classification,
        audit_complete=complete,
        comparison_available=available,
        production_minus_audit_centered_nll=gap,
        absolute_gap=None if gap is None else abs(gap),
        substantive_disagreement=complete
        and available
        and classification == "disagreement",
    )


def recovery_row(record):
    if record is not None and record["status"] == "completed":
        return record
    return dict(
        assessment=dict(accepted=False, diagnostic_budget_complete=False),
        recovery=dict(
            accepted_component_relative_errors=[None, None],
            all_finite_component_relative_errors=[None, None],
        ),
    )


def accounting(task, record, timing):
    result = dict(
        task=task,
        status="not_run" if record is None else record["status"],
        wall_seconds=None if timing is None else timing["wall_seconds"],
        calls=None,
    )
    if record and record.get("counters_available"):
        if task["kind"] == "observation":
            calls = dict(generation=record["generation_forward_calls"])
        elif task["kind"] == "production":
            o = record["optimization"]
            calls = {
                name: o[name + "_evaluations"]
                for name in ("initialization", "objective", "validation", "diagnostic")
            }
            calls["external_assessment"] = record["assessment"][
                "external_forward_evaluations"
            ]
            calls["input_verification"] = record["input_verification_forward_calls"]
        else:
            calls = dict(
                independent_audit=record["forward_calls"],
                shared_audit_recheck=record["shared_forward_recheck_calls"],
                input_verification=record["input_verification_forward_calls"],
            )
        result["calls"] = calls
    return result


def aggregate(lock, mode, records, timings=None):
    protocol = lock["protocol"]
    task_list = lock["populations"][mode]
    expected = {task_key(t) for t in task_list}
    if set(records) - expected or any(
        r["status"] not in TERMINAL for r in records.values()
    ):
        raise ValueError("foreign task or unknown terminal status")
    summaries, audits, ledger = [], [], []
    for task in task_list:
        key = task_key(task)
        record = records.get(key)
        ledger.append(accounting(task, record, (timings or {}).get(key)))
        if task["kind"] == "audit":
            production = records.get(task_key(dict(task, kind="production")))
            audits.append(
                dict(task=task, **audit_comparison(production, record, protocol))
            )
    for condition in protocol["conditions"]:
        for arm_index, arm in enumerate(condition["arms"]):
            population = [
                t
                for t in task_list
                if t["kind"] == "production"
                and t["condition_id"] == condition["id"]
                and t["arm"] == arm
            ]
            if population:
                rows = [recovery_row(records.get(task_key(t))) for t in population]
                summary = summarize_recovery(rows, protocol, condition["id"], arm_index)
                summary["n_scheduled"] = len(population)
                summary["n_terminal_tasks"] = sum(
                    task_key(t) in records for t in population
                )
                summaries.append(dict(condition_id=condition["id"], arm=arm, **summary))
    all_terminal = len(records) == len(task_list)
    observations_complete = all(
        records.get(task_key(t), {}).get("status") == "completed"
        for t in task_list
        if t["kind"] == "observation"
    )
    gates = None
    tally = None
    if mode == "qualification":
        tally = audit_tally(
            protocol,
            [
                dict(
                    condition_id=a["task"]["condition_id"],
                    replicate=a["task"]["replicate"],
                    arm=a["task"]["arm"],
                    classification=a["classification"],
                )
                for a in audits
                if a["classification"] != "not_run"
            ],
        )
        baseline = next(s for s in summaries if s["condition_id"] == 0)
        components = baseline["components"]
        gates = dict(
            numerical_acceptance=baseline["n_accepted"] >= 95,
            median_absolute_relative_errors=[
                c["median_absolute_relative_error"] is not None
                and c["median_absolute_relative_error"] < 0.05
                for c in components
            ],
            absolute_mean_signed_relative_biases=[
                c["signed_relative_bias"] is not None
                and abs(c["signed_relative_bias"]) < 0.02
                for c in components
            ],
            complete_usable_audit_agreement_100_of_100=tally[
                "primary_audit_target_met"
            ],
            complete_untampered_population=all_terminal and observations_complete,
        )
        gates["overall_pass"] = (
            gates["numerical_acceptance"]
            and all(gates["median_absolute_relative_errors"])
            and all(gates["absolute_mean_signed_relative_biases"])
            and gates["complete_usable_audit_agreement_100_of_100"]
            and gates["complete_untampered_population"]
        )
    totals = Counter()
    for entry in ledger:
        totals.update(entry["calls"] or {})
    return jsonable(
        dict(
            protocol=protocol["protocol"],
            scope=mode,
            execution_lock_sha256=digest(lock),
            all_scheduled_tasks_terminal=all_terminal,
            scheduled_counts=dict(Counter(t["kind"] for t in task_list)),
            terminal_counts=dict(
                Counter(t["kind"] for t in task_list if task_key(t) in records)
            ),
            statuses=dict(Counter(row["status"] for row in ledger)),
            primary_gates=gates,
            audit_tally=tally,
            audit_comparisons=audits,
            complete_audits=sum(a["audit_complete"] for a in audits),
            unavailable_audit_comparisons=sum(
                not a["comparison_available"] for a in audits
            ),
            substantive_audit_disagreements=sum(
                a["substantive_disagreement"] for a in audits
            ),
            recovery=summaries,
            accounting=ledger,
            known_forward_calls_by_stage=dict(totals),
            terminal_tasks_with_unknown_call_counts=sum(
                r.get("counters_available") is not True for r in records.values()
            ),
            known_wall_seconds=sum(row["wall_seconds"] or 0 for row in ledger),
        )
    )


def read_evidence(directory, lock, mode):
    """Validate every published envelope, including unadopted worker transports."""
    store = open_existing(directory, lock)
    keys = {task_key(t) for t in lock["populations"][mode]}
    allowed = (
        {"authorization"}
        | keys
        | {prefix + k for prefix in ("started-", "transport-", "timing-") for k in keys}
    )
    inventory, payloads = {}, {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.name in ("lock.json", ".coordinator-owner.json"):
            continue
        if path.stem not in allowed:
            raise ValueError("foreign checkpoint in evidence directory")
        payloads[path.stem] = store.get(path.stem)
        inventory[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    permission = payloads.get("authorization")
    if permission is None or permit(lock, mode, permission) != permission:
        raise ValueError("checkpoint authorization mismatch")
    records = {k: payloads[k] for k in keys if k in payloads}
    for task in lock["populations"][mode]:
        key = task_key(task)
        record = records.get(key)
        request = payloads.get("started-" + key)
        if request is not None and record is None:
            raise ValueError(
                "started task has no terminal record; resume recovery before reporting"
            )
        if record is not None and (request is None or request["task"] != task):
            raise ValueError("terminal evidence lacks its original task request")
        transport = payloads.get("transport-" + key)
        if transport and (
            not request or transport["request_sha256"] != digest(request)
        ):
            raise ValueError("transport/request identity mismatch")
        if not record or record["status"] != "completed":
            continue
        if not transport or digest(transport["result"]) != digest(record):
            raise ValueError("completed task differs from durable worker result")
        if transport["finished_epoch"] > request["deadline_epoch"]:
            raise ValueError("completed evidence violates its wall limit")
        if task["kind"] == "observation":
            entropy = [
                *lock["protocol"]["root_entropy"],
                task["stream"],
                task["condition_id"],
                task["replicate"],
            ]
            if (
                record["scope"] != mode
                or record["entropy"] != entropy
                or digest(record["counts"]) != record["counts_sha256"]
                or digest(record["time_ns"]) != record["time_sha256"]
            ):
                raise ValueError("saved observation identity/hash mismatch")
        else:
            observation = records.get(observation_key(task))
            if (
                not observation
                or observation["status"] != "completed"
                or observation["counts_sha256"] != record["observation_sha256"]
            ):
                raise ValueError("fit/audit is not bound to its saved observation")
    timings = {k: payloads["timing-" + k] for k in keys if "timing-" + k in payloads}
    return records, timings, inventory


def render(summary):
    lines = [
        "# Issue #33 Stage 3 " + summary["scope"] + " report",
        "",
        "Execution lock: `" + summary["execution_lock_sha256"] + "`.",
        "",
        "Independent agreement does not prove global optimality. Audit incompleteness "
        "does not by itself establish production-fitter failure.",
        "",
        "Primary gates: `" + json.dumps(summary["primary_gates"]) + "`.",
        "",
        "| Condition / arm | Accepted / scheduled | Median absolute relative errors | Signed biases |",
        "|---|---:|---|---|",
    ]
    for row in summary["recovery"]:
        lines.append(
            f"| C{row['condition_id']:02d} / {row['arm']} | "
            f"{row['n_accepted']}/{row['n_scheduled']} | "
            f"{[c['median_absolute_relative_error'] for c in row['components']]} | "
            f"{[c['signed_relative_bias'] for c in row['components']]} |"
        )
    lines += [
        "",
        "Component errors use accepted, available components with explicit denominators "
        "in the JSON. No warning-based exclusions. Secondary conditions are diagnostic.",
        "",
        "Audit classes and all scheduled task outcomes are retained in the companion JSON.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.directory / "lock.json").read_text())
    with (
        exclusive_lease(args.directory / ".coordinator.guard"),
        exclusive_lease(args.directory / ".worker.guard"),
    ):
        records, timings, inventory = read_evidence(args.directory, lock, args.mode)
    summary = aggregate(lock, args.mode, records, timings)
    summary["evidence_sha256"] = digest(inventory)
    summary["evidence_inventory"] = inventory
    summary["reporting_source_lf_sha256"] = {
        p.relative_to(ROOT).as_posix(): source_digest(p)
        for p in (
            Path(__file__),
            ROOT / "scripts/qualify_issue33_stage3.py",
            ROOT / "scripts/qualify_biexponential_reconvolution.py",
            ROOT / "scripts/validate_issue33_stage3_protocol.py",
            ROOT / "src/tcspc_toolkit/benchmarks/multiexponential.py",
        )
    }
    if args.report.exists() or args.output.exists():
        raise FileExistsError("use new paths for each evidence/report revision")
    write_new_json(args.output, summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        handle.write(render(summary))
    print("Read-only aggregation", summary["evidence_sha256"])


if __name__ == "__main__":
    main()
