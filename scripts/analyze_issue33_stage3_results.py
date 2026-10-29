"""Supplement frozen Stage-3 results with descriptive accounting; never fit or draw.

Acceptance and recovery criteria come exclusively from the frozen reporter.
This supplementary reporting source has its own hash, separate from execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from qualify_biexponential_reconvolution import digest, source_digest, write_new_json
from qualify_issue33_stage3 import exclusive_lease, observation_key, task_key
from report_issue33_stage3 import read_evidence


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(
        n=len(values),
        minimum=min(values) if values else None,
        median=statistics.median(values) if values else None,
        maximum=max(values) if values else None,
        total=sum(values),
    )


def summarize_group(tasks, records):
    outcomes, stops, start_kinds, budgets, recovery_unavailable = (
        Counter() for _ in range(5)
    )
    warnings = {
        category: Counter()
        for category in ("structural", "conditioning", "boundary", "multistart")
    }
    starts, omitted, seeds, unused_seeds = [], [], [], []
    local_passes = accepted = warning_free = 0
    warning_free_errors = [[], []]
    fit_seconds, nlls = [], []
    for task in tasks:
        r = records[task_key(task)]
        outcomes[r["status"]] += 1
        if r["status"] != "completed":
            continue
        o = r["optimization"]
        accepted += r["assessment"]["accepted"]
        local_passes += o["numerical_validation_passed"] is True
        stops[o["stopping_reason"]] += 1
        start_kinds.update(s["kind"] for s in o["starts"])
        starts.append(len(o["starts"]))
        seeds.append(o["finite_seed_candidates"])
        unused_seeds.append(o["unattempted_seed_candidates"])
        omitted.append(len(o["optional_continuations_omitted"]))
        budgets.update(o["exhausted_budgets"])
        if o["recovery_unavailable_reason"] is not None:
            recovery_unavailable[o["recovery_unavailable_reason"]] += 1
        flagged = False
        for category in warnings:
            labels = r["identifiability"][category + "_warnings"]
            warnings[category].update(labels)
            flagged |= bool(labels)
        warning_free += not flagged
        if not flagged:
            for k, value in enumerate(
                r["recovery"]["accepted_component_relative_errors"]
            ):
                if value is not None:
                    warning_free_errors[k].append(abs(value))
        fit_seconds.append(r["fit_seconds"])
        nlls.append(r["assessment"]["recomputed_centered_nll"])
    return dict(
        scheduled=len(tasks),
        terminal_outcomes=dict(outcomes),
        numerical_accepted=accepted,
        local_validation_passes=local_passes,
        stopping_reasons=dict(stops),
        start_kinds=dict(start_kinds),
        starts=distribution(starts),
        finite_seed_bank_sizes=distribution(seeds),
        unattempted_required_seeds=distribution(unused_seeds),
        optional_continuations_omitted=distribution(omitted),
        fits_with_optional_omission=sum(n > 0 for n in omitted),
        exhausted_budgets=dict(budgets),
        required_recovery_unavailable=dict(recovery_unavailable),
        warnings={k: dict(v) for k, v in warnings.items()},
        warning_free_fits=warning_free,
        warning_free_accepted_absolute_relative_errors=[
            distribution(v) for v in warning_free_errors
        ],
        fitter_seconds=distribution(fit_seconds),
        centered_nll=distribution(nlls),
    )


def audit_details(tasks, records, comparisons):
    by_key = {task_key(r["task"]): r for r in comparisons}
    rows = []
    for task in tasks:
        r = records[task_key(task)]
        rows.append(
            dict(
                task=task,
                status=r["status"],
                comparison=by_key[task_key(task)],
                exhausted_budgets=r.get("exhausted_budgets"),
                direct_forward_calls=r.get("forward_calls"),
                elapsed_seconds=r.get("elapsed_seconds"),
                solver_statuses=[s.get("status") for s in r.get("starts", [])],
                successful_starts=sum(
                    s.get("success", False) for s in r.get("starts", [])
                ),
            )
        )
    return rows


def validate_budgets(store, records, task_list):
    protocol = store["protocol"]
    settings = protocol["fit_settings"]
    for task in task_list:
        record = records[task_key(task)]
        if record["status"] != "completed":
            continue
        if task["kind"] == "production":
            o = record["optimization"]
            assert len(o["starts"]) <= settings["max_starts"]
            for counter, maximum in (
                ("initialization", "initialization"),
                ("objective", "objective"),
                ("validation", "validation"),
                ("diagnostic", "diagnostic"),
            ):
                assert (
                    0
                    <= o[counter + "_evaluations"]
                    <= settings["max_" + maximum + "_evaluations"]
                )
            assert record["assessment"]["external_forward_evaluations"] <= 1
            assert record["input_verification_forward_calls"] == 2
        elif task["kind"] == "audit":
            assert record["forward_calls"] <= protocol["audit"]["max_forward_calls"]
            assert len(record["starts"]) <= 8
            assert all(
                s.get("nfev", 0) <= protocol["audit"]["max_nfev_per_start"]
                for s in record["starts"]
            )
            assert record["shared_forward_recheck_calls"] <= 1
            assert record["input_verification_forward_calls"] == 2
        else:
            assert record["generation_forward_calls"] == 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.directory / "lock.json").read_text())
    summary = json.loads(args.results.read_text())
    assert summary["scope"] == "qualification" and summary[
        "execution_lock_sha256"
    ] == digest(lock)
    with (
        exclusive_lease(args.directory / ".coordinator.guard"),
        exclusive_lease(args.directory / ".worker.guard"),
    ):
        records, timings, inventory = read_evidence(
            args.directory, lock, "qualification"
        )
        assert digest(inventory) == summary["evidence_sha256"]
        task_list = lock["populations"]["qualification"]
        assert len(records) == len(task_list) == 543
        validate_budgets(lock, records, task_list)
        pools_checked = 0
        for task in task_list:
            transport = args.directory / ("transport-" + task_key(task) + ".json")
            if transport.exists():
                pools = json.loads(transport.read_text())["payload"][
                    "numerical_libraries"
                ]
                assert all(p["num_threads"] == 1 for p in pools)
                pools_checked += 1
    production = [t for t in task_list if t["kind"] == "production"]
    groups = []
    for c in lock["protocol"]["conditions"]:
        for arm in c["arms"]:
            selected = [
                t
                for t in production
                if t["condition_id"] == c["id"] and t["arm"] == arm
            ]
            groups.append(
                dict(
                    condition_id=c["id"], arm=arm, **summarize_group(selected, records)
                )
            )
    paired = []
    for replicate in range(10):
        selected = [
            t
            for t in production
            if t["condition_id"] == 26 and t["replicate"] == replicate
        ]
        assert len(selected) == 2 and len({observation_key(t) for t in selected}) == 1
        obs = records[observation_key(selected[0])]
        arms = {t["arm"]: records[task_key(t)] for t in selected}
        for arm in arms.values():
            if arm["status"] == "completed":
                assert arm["observation_sha256"] == obs["counts_sha256"]
        values = {
            name: r.get("assessment", {}).get("recomputed_centered_nll")
            for name, r in arms.items()
        }
        gap = (
            None
            if any(v is None for v in values.values())
            else values["gaussian_0.3"] - values["matched"]
        )
        paired.append(
            dict(
                replicate=replicate,
                observation_status=obs["status"],
                observation_sha256=obs.get("counts_sha256"),
                same_observation_identity=True,
                numerical_acceptance={
                    a: r.get("assessment", {}).get("accepted", False)
                    for a, r in arms.items()
                },
                centered_nll=values,
                misspecified_minus_matched_centered_nll=gap,
            )
        )
    audit_tasks = [t for t in task_list if t["kind"] == "audit"]
    rejected = []
    for task in production:
        r = records[task_key(task)]
        if not r.get("assessment", {}).get("accepted", False):
            o = r.get("optimization", {})
            rejected.append(
                dict(
                    task=task,
                    status=r["status"],
                    parameters=r.get("parameters"),
                    assessment=r.get("assessment"),
                    optimization={k: v for k, v in o.items() if k != "starts"},
                    start_messages=dict(
                        Counter(s["message"] for s in o.get("starts", []))
                    ),
                    start_statuses=dict(
                        Counter(str(s["status"]) for s in o.get("starts", []))
                    ),
                    start_best_objectives=[
                        s["best_centered_nll"] for s in o.get("starts", [])
                    ],
                )
            )
    output = dict(
        schema=1,
        protocol=lock["protocol"]["protocol"],
        scope="descriptive_post_execution_analysis",
        execution_lock_sha256=digest(lock),
        evidence_sha256=digest(inventory),
        analysis_source_lf_sha256=source_digest(Path(__file__)),
        primary_gates_from_frozen_report=summary["primary_gates"],
        all_production=summarize_group(production, records),
        condition_arms=groups,
        numerically_rejected_fits=rejected,
        audits=audit_details(audit_tasks, records, summary["audit_comparisons"]),
        c26_paired_observations=paired,
        c26_paired_objective_difference=distribution(
            r["misspecified_minus_matched_centered_nll"] for r in paired
        ),
        recorded_budget_bounds_verified=True,
        single_thread_transports_checked=pools_checked,
        task_wall_seconds={
            kind: distribution(
                timings[task_key(t)]["wall_seconds"]
                for t in task_list
                if t["kind"] == kind and task_key(t) in timings
            )
            for kind in ("observation", "production", "audit")
        },
        interpretation="Descriptive secondary analysis; no new acceptance criteria, fitting, observations, or optimization retries.",
    )
    write_new_json(args.output, output)
    print(
        json.dumps(
            dict(
                groups=len(groups),
                tasks=len(task_list),
                budgets="verified",
                thread_records=pools_checked,
            )
        )
    )


if __name__ == "__main__":
    main()
