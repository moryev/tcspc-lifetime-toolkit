"""Closed audit-only protocol and sealed-record reporting. No import execution."""

from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
from math import isfinite
from types import MappingProxyType

from tcspc_toolkit.benchmarks._triexponential_audit import AuditSettings
from tcspc_toolkit.benchmarks._triexponential_verification import (
    VerificationProtocol,
    construct as construct,
)


SCOPE = "issue37_pr2c2_audit_verification_v1"
CONTRACT_KEYS = (
    "conditions",
    "construction",
    "irfs",
    "problems",
    "tasks",
    "configurations",
    "environment_controls",
    "initialization",
    "resources",
    "scheduling",
    "numerical_contract",
    "outputs",
    "integrity",
    "analysis",
)
PROTOCOL = VerificationProtocol(
    SCOPE,
    "issue37.pr2c2.audit-verification.v1",
    "e712329cbafca271ce6572e93bc383892d563485f53edbfd0d3b674baf9b0b23",
    "configs/" + SCOPE + ".json",
    "data/generated/" + SCOPE,
    tuple(
        f"A{i:02d}:{target}:matched:audit"
        for i in range(3)
        for target in ("mean", "poisson-0")
    ),
    "issue37.pr2c2.audit-verification-report.v1",
)
PROTOCOLS = MappingProxyType({SCOPE: PROTOCOL})


def validate_manifest(manifest):
    """Pin scientific content independently of claims in a supplied manifest."""
    try:
        protocol = PROTOCOLS[manifest["study_id"]]
        contract = {key: manifest[key] for key in CONTRACT_KEYS}
        encoded = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        valid = (
            manifest["schema"] == protocol.manifest_schema
            and manifest["execution_authorized"] is False
            and manifest["scientific_contract_sha256"] == protocol.contract_sha256
            and tuple(manifest["scientific_contract_keys"]) == CONTRACT_KEYS
            and hashlib.sha256(encoded.encode()).hexdigest() == protocol.contract_sha256
            and tuple(t["id"] for t in manifest["tasks"]) == protocol.task_ids
            and manifest["outputs"]["directory"] == protocol.output_directory
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("exact reviewed audit-only verification manifest required")
    return protocol


def settings(configuration):
    if configuration != "audit":
        raise ValueError("audit-only verification forbids production fits")
    return AuditSettings(16, 8, 40000, 120)


def problems(manifest):
    validate_manifest(manifest)
    conditions = {c["id"]: c for c in manifest["conditions"]}
    return tuple(
        dict(
            id=p["id"],
            observation_id=p["observation_id"],
            seed=copy.deepcopy(p["rng_entropy"]),
            assumed_irf=p["assumed_irf"],
            control=False,
            condition=copy.deepcopy(conditions[p["condition_id"]])
            | dict(stratum="audit_verification"),
        )
        for p in manifest["problems"]
    )


def tasks(manifest):
    return tuple(
        dict(id=p["id"] + ":audit", problem=p, configuration="audit")
        for p in problems(manifest)
    )


def specification(manifest):
    validate_manifest(manifest)
    actual = {"audit": asdict(settings("audit"))}
    if actual != manifest["configurations"]:
        raise ValueError("resolved audit-only settings changed")
    return dict(
        tasks=tasks(manifest),
        settings=actual,
        worker_deadlines={"audit": 120},
        serial_workers=1,
        termination_seconds=5,
        cleanup_seconds=5,
        retries=False,
        continuations=False,
        input_generation=dict(
            procedure="benchmarks.triexponential.construct_problem",
            construction=manifest["construction"],
            irfs=manifest["irfs"],
        ),
        preparation_model_call_limit=18,
        reporting_model_call_limit=0,
        output_directory=PROTOCOL.output_directory,
        report_schema=PROTOCOL.report_schema,
    )


def _trial_counts(work):
    """Use complete counters, never infer outcomes from null serialized values."""
    return dict(
        finite_objective=work.get("finite_evaluations", 0),
        positive_infinite_objective=work.get("infinite_objective", 0),
        infeasible=work.get("infeasible_trial", 0),
        numerical_failure=sum(
            work.get(k, 0)
            for k in (
                "invalid_trial",
                "unrepresentable_parameters",
                "nonfinite_model",
                "invalid_objective",
                "objective_numerical_failure",
            )
        ),
    )


def _finite(value):
    return type(value) in (int, float) and isfinite(value)


def report(tasks, store):
    """Summarize verified worker evidence without model calls or optimizer dispatch."""
    from tcspc_toolkit.benchmarks import triexponential as shared

    if store.lock["fingerprint"]["scope"] not in PROTOCOLS or any(
        t["configuration"] != "audit" for t in tasks
    ):
        raise shared.ExecutionIntegrityError(
            "foreign task or scope in audit-only report"
        )
    inventory = store.verify_inventory(tasks)
    settings_record = store.lock["fingerprint"]["execution"]["settings"]["audit"]
    planned_solvers = ["trf"] * settings_record["trf_starts"] + [
        "powell"
    ] * settings_record["powell_starts"]
    records = [store.read_attempt(t, inventory) for t in tasks]
    analyses, start_rows = [], []
    outcomes = Counter()
    termination_counts = {name: Counter() for name in ("trf", "powell")}
    for task, row in zip(tasks, records, strict=True):
        parent = row.get("result", {})
        worker = None
        terminal_success = row["status"] == "completed" and not parent.get("failure")
        if (
            terminal_success
            or store.path(task["id"]).with_suffix(".worker.json").exists()
        ):
            worker = store.read_worker_result(task, tasks, row["binding"])
        if terminal_success:
            payload = {k: v for k, v in parent.items() if k != "external_worker_timing"}
            if worker != payload:
                raise shared.ExecutionIntegrityError(
                    "attempt differs from immutable audit worker result"
                )
        result = worker if worker is not None else parent
        available = "starts" in result and "complete" in result
        starts = result.get("starts", []) if available else []
        if available and (
            result.get("settings") != settings_record
            or len(starts) > len(planned_solvers)
        ):
            raise shared.ExecutionIntegrityError(
                "audit result settings/start inventory mismatch"
            )
        task_starts = []
        for index, solver in enumerate(planned_solvers):
            saved = starts[index] if index < len(starts) else None
            if saved is not None and saved.get("solver") != solver:
                raise shared.ExecutionIntegrityError(
                    "audit start solver/order mismatch"
                )
            status = (
                saved["termination"]
                if saved is not None
                else "missing_start_record"
                if available
                else "unknown_worker_work"
            )
            work = dict(saved.get("work", ())) if saved is not None else None
            detail = dict(
                task_id=task["id"],
                index=index,
                solver=solver,
                termination=status,
                attempted=saved.get("attempted") if saved is not None else None,
                recorded=saved,
                work=work,
                trial_classifications=_trial_counts(work) if work is not None else None,
                retained_derivative_statuses=dict(
                    Counter(
                        d["status"] for d in saved.get("derivative_diagnostics", ())
                    )
                )
                if saved is not None
                else None,
                derivative_counts_cover_retained_records_only=True,
            )
            task_starts.append(detail)
            termination_counts[solver][status] += 1
        start_rows.extend(task_starts)
        search_complete = bool(
            available
            and len(starts) == len(planned_solvers)
            and all(
                s.get("attempted") is True
                and s.get("solver_success") is True
                and s.get("termination") == "terminated"
                and s.get("endpoint_objective_consistent") is True
                for s in starts
            )
        )
        candidate = result.get("parameters")
        finite_candidate = bool(
            isinstance(candidate, list)
            and len(candidate) == 8
            and all(_finite(v) for v in candidate)
            and all(v >= 0 for v in candidate[:3])
            and candidate[6] >= 0
            and all(v > 0 for v in candidate[3:6])
            and _finite(result.get("centered_nll"))
            and result["centered_nll"] >= 0
        )
        identity = bool(
            candidate is not None
            and candidate
            == result.get("assessment_candidate")
            == result.get("validation_candidate")
        )
        criteria = bool(
            search_complete
            and finite_candidate
            and identity
            and result.get("assessment_complete") is True
            and result.get("consistent") is True
            and result.get("validation_complete") is True
            and result.get("validation_passed") is True
            and result.get("sealed") is True
            and result.get("reasons") == []
            and _finite(result.get("elapsed_seconds"))
            and 0 <= result["elapsed_seconds"] < settings_record["wall_seconds"]
        )
        claim_consistent = not available or (
            result.get("search_complete") == search_complete
            and result.get("all_starts_attempted")
            == (
                len(starts) == len(planned_solvers)
                and all(s.get("attempted") is True for s in starts)
            )
            and result.get("complete") == criteria
        )
        outcome = (
            parent.get("failure")
            or (row["status"] if row["status"] != "completed" else None)
            or (
                "inconsistent_completion_record"
                if not claim_consistent
                else "audit_complete"
                if criteria
                else "audit_incomplete"
                if available
                else "incomplete_audit_result"
            )
        )
        outcomes[outcome] += 1
        analyses.append(
            dict(
                id=task["id"],
                outcome=outcome,
                result_available=available,
                reported_complete=result.get("complete"),
                strict_complete=criteria and claim_consistent and terminal_success,
                completion_record_consistent=claim_consistent,
                search_complete=search_complete,
                candidate_identity_consistent=identity,
                finite_candidate_available=finite_candidate,
                assessment_complete=result.get("assessment_complete"),
                consistent=result.get("consistent"),
                validation_complete=result.get("validation_complete"),
                validation_passed=result.get("validation_passed"),
                returned_candidate=candidate,
                centered_nll=result.get("centered_nll"),
                validation_details=result.get("validation_details"),
                best_probe_improvement=result.get("best_probe_improvement"),
                reasons=result.get("reasons"),
                work=dict(result.get("work", ())) if available else None,
                phase_calls=result.get("phase_calls"),
                model_calls_by_origin=result.get("model_calls_by_origin"),
                charged_component_convolutions=result.get("component_convolutions"),
                completed_component_convolutions=result.get(
                    "completed_component_convolutions"
                ),
                unknown_partial_model_calls=result.get("component_work_unknown_calls"),
                numerical_seconds=result.get("elapsed_seconds"),
                external_worker_timing=parent.get("external_worker_timing"),
                work_unknown=not available or parent.get("work_unknown", False),
                retained_worker_result=worker if not terminal_success else None,
                final_trial_diagnostics=result.get("final_trial_diagnostics"),
                final_trial_records_dropped=result.get("final_trial_records_dropped"),
            )
        )
    store.verify_inventory(tasks)
    return dict(
        scope=SCOPE,
        schema=PROTOCOL.report_schema,
        attempts=records,
        analyses=analyses,
        starts=start_rows,
        summary=dict(
            all_attempt_denominator=len(tasks),
            scheduled_start_denominator=len(start_rows),
            outcomes=dict(outcomes),
            strict_complete=sum(a["strict_complete"] for a in analyses),
            start_terminations={
                name: dict(values) for name, values in termination_counts.items()
            },
            production_fit_metrics=dict(status="not_applicable"),
            production_minus_audit_objective_gaps=dict(status="not_applicable"),
            qualification=False,
            production_policy_selected=False,
        ),
        preparation_work=[
            store.read("input:" + entry["problem_id"])["result"].get("preparation_work")
            for entry in inventory["inputs"]
        ],
        reporting_work=dict(
            model_calls=0,
            charged_component_convolutions=0,
            completed_component_convolutions=0,
        ),
        provenance=dict(
            inventory_sha256=inventory["payload_sha256"],
            execution_lock_sha256=inventory["lock_sha256"],
            fingerprint_sha256=inventory["fingerprint_sha256"],
            manifest_sha256=inventory["manifest_sha256"],
        ),
    )
