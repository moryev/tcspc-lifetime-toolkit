"""The reviewed six-problem fit-only verification contract; no import execution.

Shared construction, persistence and numerical routines remain in their existing
modules. Imports of the shared benchmark module are local to avoid an eager cycle.
"""

from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
from time import perf_counter
from types import MappingProxyType

import numpy as np


VERIFICATION_SCOPE = "issue37_pr2c1_numerical_verification_v1"
CONTRACT_KEYS = (
    "conditions",
    "construction",
    "irfs",
    "problems",
    "configurations",
    "tasks",
    "resources",
    "scheduling",
    "numerical_acceptance",
    "environment_controls",
    "initial_guesses",
)
# Approved draft scientific/settings/task contract. Changes require an explicit
# protocol revision, not silently accepting a similarly named arbitrary manifest.
CONTRACT_SHA256 = "f53733d261f3cee8fa0ffe1af79455897b186335a709b9b7b1cc3d7d1b842f4b"


@dataclass(frozen=True)
class VerificationProtocol:
    scope: str
    manifest_schema: str
    contract_sha256: str
    manifest_path: str
    output_directory: str
    task_ids: tuple[str, ...]
    # V1 retains its historical schema-less report representation.
    report_schema: str | None


_TASK_IDS = tuple(
    f"V{i:02d}:{target}:matched:{solver}:compact"
    for i in range(3)
    for target in ("mean", "poisson-0")
    for solver in ("lbfgsb", "trf")
)
PROTOCOLS = MappingProxyType(
    {
        p.scope: p
        for p in (
            VerificationProtocol(
                VERIFICATION_SCOPE,
                "issue37.pr2c1.verification-proposal.v1",
                CONTRACT_SHA256,
                "configs/issue37_pr2c1_numerical_verification_v1.json",
                "data/generated/issue37_pr2c1_numerical_verification_v1",
                _TASK_IDS,
                None,
            ),
            VerificationProtocol(
                "issue37_pr2c1b_numerical_verification_v2",
                "issue37.pr2c1b.verification-proposal.v2",
                "27c5238c32a9aa388f614dcc2c8a6ef2813baad985e9db3de681a031a4ddeb17",
                "configs/issue37_pr2c1b_numerical_verification_v2.json",
                "data/generated/issue37_pr2c1b_numerical_verification_v2",
                _TASK_IDS,
                "issue37.pr2c1b.verification-report.v2",
            ),
        )
    }
)


def protocol_for_scope(scope):
    """Resolve only explicitly reviewed versions, with no fallback protocol."""
    try:
        return PROTOCOLS[scope]
    except (KeyError, TypeError) as error:
        raise ValueError("unknown reviewed verification scope") from error


def validate_manifest(manifest):
    try:
        protocol = protocol_for_scope(manifest["study_id"])
        contract = {k: manifest[k] for k in CONTRACT_KEYS}
        encoded = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        valid = (
            manifest["schema"] == protocol.manifest_schema
            and manifest["execution_authorized"] is False
            and hashlib.sha256(encoded.encode()).hexdigest() == protocol.contract_sha256
            and tuple(t["id"] for t in manifest["tasks"]) == protocol.task_ids
            and manifest["outputs"]["directory"] == protocol.output_directory
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("exact reviewed fit-only verification manifest required")
    return protocol


def problems(manifest):
    """Resolve only metadata, including declared truth; never consult pilot files."""
    validate_manifest(manifest)
    conditions = {c["id"]: c for c in manifest["conditions"]}
    return tuple(
        dict(
            id=p["id"],
            observation_id=p["observation_id"],
            seed=p["rng_entropy"],
            assumed_irf=p["assumed_irf"],
            control=False,
            condition=copy.deepcopy(conditions[p["condition_id"]])
            | dict(stratum="numerical_verification"),
        )
        for p in manifest["problems"]
    )


class _Work:
    """Charge physical calls before invoking them, including failing requests."""

    def __init__(self, limit, phase):
        self.limit, self.phase = limit, phase
        self.calls = self.charged = self.completed = self.failed = 0
        self.started = perf_counter()

    def model(self, *args, **kwargs):
        from tcspc_toolkit.benchmarks import triexponential as shared

        if self.calls >= self.limit:
            raise shared.ExecutionIntegrityError(self.phase + " model-call limit")
        k = len(kwargs["amplitudes"])
        self.calls += 1
        self.charged += k
        try:
            result = shared.multiexponential_reconvolution_expected_counts(
                *args, **kwargs
            )
        except Exception:
            self.failed += 1
            raise
        self.completed += k
        return result

    def record(self):
        return dict(
            phase=self.phase,
            model_calls=self.calls,
            model_call_limit=self.limit,
            charged_component_convolutions=self.charged,
            completed_component_convolutions=self.completed,
            failures_with_unknown_partial_work=self.failed,
            elapsed_seconds=perf_counter() - self.started,
        )


def construct(manifest, problem, observation=None):
    """One explicit input preparation with three bounded K=1 component calls."""
    from tcspc_toolkit.benchmarks import triexponential as shared

    work = _Work(3, "preparation")
    try:
        arrays = shared._construct_problem(
            manifest,
            problem,
            observation=observation,
            model=work.model,
            declared_amplitudes=problem["condition"]["amplitudes"],
        )
    except Exception as error:
        # DevelopmentStore retains this bounded record in the failed checkpoint.
        error.verification_work = work.record()
        raise
    return arrays | dict(preparation_work=work.record())


def _prediction(arrays, parameters, work):
    return work.model(
        arrays["time"],
        arrays["kernel"],
        amplitudes=parameters["amplitudes"],
        lifetimes=parameters["lifetimes_ns"],
        background=parameters["background"],
        temporal_shift=parameters.get("temporal_shift_ns", parameters.get("shift")),
    )


def _objective(arrays, parameters, work):
    from tcspc_toolkit.benchmarks import triexponential as shared

    mu = _prediction(arrays, parameters, work)
    return mu, 0.5 * shared.calculate_poisson_deviance(arrays["observed"], mu)


def _history(task, entry, fit, baseline, *, previous_verification=False):
    """Compare to manifest-bound historical metadata, never initialize a search."""
    reference = baseline.get("problems", {}).get(task["problem"]["id"])
    if reference is None:
        return dict(status="historical_reference_unavailable")
    matches = {
        k: entry["arrays"][k]["sha256"] == v for k, v in reference["arrays"].items()
    }
    deterministic = task["problem"]["seed"] is None
    previous = reference["fits"][task["configuration"]]
    objective = fit.get("centered_nll")
    prior_objective = previous.get("centered_nll")
    return dict(
        status="deterministic_identical_inputs"
        if deterministic and all(matches.values())
        else "deterministic_input_mismatch"
        if deterministic
        else "independent_poisson_draw",
        array_hash_matches=matches,
        historical=previous,
        new_minus_original_centered_nll=(objective - prior_objective)
        if deterministic
        and all(matches.values())
        and objective is not None
        and prior_objective is not None
        else None,
        controlled_poisson_before_after=False,
        reused_development_conditions=True,
        **(
            dict(audit_status="not_applicable", comparison="verification_v1")
            if previous_verification
            else dict(original_audit_completeness="0/6; unchanged")
        ),
    )


def _validation_summary(fit):
    """Expose recorded final-candidate facts without inferring a solver success."""
    numerical = fit.get("numerical", {})
    validation = numerical.get("validation") or {}
    optimization = fit.get("optimization", {})
    probes = validation.get("probes", [])
    improvement = validation.get("best_improvement")
    diagnostic = numerical.get("diagnostic_improvement")
    return dict(
        complete=validation.get("complete"),
        passed=validation.get("passed"),
        candidate=validation.get("candidate"),
        matches_returned_candidate=numerical.get("validation_matches_candidate"),
        probe_statuses=dict(Counter(p["status"] for p in probes)),
        redundant_physical_candidates=[
            p for p in probes if p["status"] == "redundant_physical_candidate"
        ],
        unavailable_feasible_directions=[
            p
            for p in probes
            if p["feasible"]
            and p["centered_nll"] is None
            and p["status"] != "redundant_physical_candidate"
        ],
        detected_descent_above_tolerance=(
            (improvement is not None and improvement > 0.001)
            or (diagnostic is not None and diagnostic > 0.001)
        )
        if improvement is not None or diagnostic is not None
        else None,
        tolerance=0.001,
        promotions=optimization.get("promotions", []),
        required_recovery_complete=numerical.get("required_recovery_complete"),
        recovery_unavailable_reason=optimization.get("recovery_unavailable_reason"),
        stopping_reason=optimization.get("stopping_reason"),
        # Final validation is serialized by the fitter. No claim about an
        # unrecorded earlier pass or separate terminal-pass invocation is inferred.
        terminal_validation_history="final_candidate_record_only",
    )


def report(tasks, store):
    """Fit-only derived analysis after full provenance validation; no optimizer."""
    from tcspc_toolkit.benchmarks import triexponential as shared

    started = perf_counter()
    protocol = protocol_for_scope(store.lock["fingerprint"]["scope"])
    inventory = store.verify_inventory(tasks)
    records = [store.read_attempt(t, inventory) for t in tasks]
    if any(t["configuration"] not in ("lbfgsb:compact", "trf:compact") for t in tasks):
        raise shared.ExecutionIntegrityError("audit/foreign task in fit-only report")
    for task, row in zip(tasks, records, strict=True):
        if row["status"] == "completed" and "numerical" in row.get("result", {}):
            worker = store.read_worker_result(task, tasks, row["binding"])
            parent = {
                k: v for k, v in row["result"].items() if k != "external_worker_timing"
            }
            if worker != parent:
                raise shared.ExecutionIntegrityError(
                    "attempt differs from immutable worker result"
                )
    work = _Work(15, "reporting")
    entries = {e["problem_id"]: e for e in inventory["inputs"]}
    baseline = store.lock["fingerprint"]["execution"].get("historical_baseline", {})
    previous = store.lock["fingerprint"]["execution"].get(
        "previous_verification_baseline", {}
    )
    arrays_by_id = {p: store.read("input:" + p)["result"] for p in entries}
    truth_scores = {}
    for task in tasks:
        p = task["problem"]
        if p["seed"] is None and p["id"] not in truth_scores:
            arrays = arrays_by_id[p["id"]]
            try:
                _, truth_scores[p["id"]] = _objective(arrays, arrays["truth"], work)
            except (ValueError, FloatingPointError, OverflowError) as error:
                raise shared.ExecutionIntegrityError(
                    "invalid deterministic generating reference"
                ) from error
    analyses, enriched = [], shared.jsonable(records)
    for task, row in zip(tasks, enriched, strict=True):
        fit = row.get("result", {})
        p = task["problem"]
        arrays = arrays_by_id[p["id"]]
        numerical = fit.get("numerical", {})
        optimization = fit.get("optimization", {})
        starts = optimization.get("starts", [])
        status = row["status"]
        outcome = fit.get("failure") or (
            "accepted"
            if numerical.get("accepted")
            else "numerically_rejected"
            if "numerical" in fit
            else status
        )
        analysis = dict(
            id=task["id"],
            outcome=outcome,
            target_kind=entries[p["id"]]["target_kind"],
            audit=dict(status="not_applicable", reason="fit_only_verification_scope"),
            numerical=numerical,
            optimization=optimization,
            identifiability=fit.get("identifiability"),
            external_worker_timing=fit.get("external_worker_timing"),
            conditional_solves=[s for s in starts if s.get("fixed_coordinates")],
            joint_invocations=sum(not s.get("fixed_coordinates") for s in starts),
            returned_objective=fit.get("centered_nll"),
            best_finite_objective=optimization.get("best_observed_centered_nll"),
            reported_correctness=shared.reported_correctness(fit),
            historical_comparison=_history(task, entries[p["id"]], fit, baseline),
        )
        if protocol.report_schema is not None:
            analysis["final_validation"] = _validation_summary(fit)
            analysis["previous_verification_comparison"] = _history(
                task, entries[p["id"]], fit, previous, previous_verification=True
            )
        if (
            row["status"] == "completed"
            and "numerical" in fit
            and fit.get("parameters") is not None
        ):
            try:
                mu, value = _objective(arrays, fit["parameters"], work)
                consistent = bool(
                    np.isfinite(value)
                    and np.array_equal(mu, fit["expected_counts"])
                    and np.isclose(value, fit["centered_nll"], atol=1e-9, rtol=1e-12)
                )
                analysis["fresh_reporting_assessment"] = dict(
                    consistent=consistent,
                    centered_nll=value,
                    signed_reported_objective_gap=value - fit["centered_nll"],
                )
            except (ValueError, FloatingPointError, OverflowError) as error:
                value = None
                analysis["fresh_reporting_assessment"] = dict(
                    consistent=False, reason=str(error)
                )
            fit["recovery"] = shared.component_recovery(
                p, arrays["truth"], fit["parameters"]
            )
            if not analysis["fresh_reporting_assessment"]["consistent"]:
                fit["recovery"] = dict(
                    available=False,
                    reason="inconsistent_returned_prediction",
                    contributors=0,
                )
            elif fit["recovery"].get("available"):
                for quantity in ("lifetime", "amplitude"):
                    fit["recovery"][quantity + "_absolute_relative_error"] = np.abs(
                        fit["recovery"][quantity + "_relative_error"]
                    ).tolist()
            analysis["recovery"] = fit["recovery"]
            analysis["parameter_errors"] = dict(
                background_absolute_error=abs(
                    fit["parameters"]["background"] - arrays["truth"]["background"]
                ),
                background_relative_error=(
                    fit["parameters"]["background"] / arrays["truth"]["background"] - 1
                )
                if arrays["truth"]["background"]
                else None,
                shift_absolute_error_ns=abs(
                    fit["parameters"]["temporal_shift_ns"] - arrays["truth"]["shift"]
                ),
                component_summary=fit.get("components"),
                warning="Ordered component errors are descriptive, not proof of identifiability.",
            )
            if p["seed"] is None:
                truth_value = truth_scores[p["id"]]
                best = optimization.get("best_observed_centered_nll")
                analysis["deterministic_reference"] = dict(
                    generating_centered_nll=truth_value,
                    returned_minus_generating=value - truth_value
                    if value is not None
                    else None,
                    best_finite_minus_generating=best - truth_value
                    if best is not None
                    else None,
                    independent_optimizer_audit=False,
                )
        analyses.append(analysis)
    summary = shared.summarize_tasks(tasks, enriched)
    for group in summary["groups"].values():
        group.pop("audit_complete")
        group.pop("gaps")
        group["audit"] = dict(status="not_applicable")
    for configuration, group in summary["groups"].items():
        subset = [
            a
            for t, a in zip(tasks, analyses, strict=True)
            if t["configuration"] == configuration
        ]
        group["outcomes"] = dict(Counter(a["outcome"] for a in subset))
        group["external_worker_timing"] = [a["external_worker_timing"] for a in subset]
        group["runtime_summary"]["interpretation"] = (
            "descriptive; timeout observations are censored lower bounds, not completed runtimes"
        )
        group["conditional_recovery"]["selection"] = (
            "all finite returned candidates with defined component truth"
        )
        accepted = [
            a["recovery"]
            for a in subset
            if a["outcome"] == "accepted" and a.get("recovery", {}).get("available")
        ]
        group["accepted_only_recovery"] = dict(
            contributors=len(accepted), planned_denominator=len(subset)
        )
        for quantity in ("lifetime", "amplitude"):
            errors = [a[quantity + "_relative_error"] for a in accepted]
            group["accepted_only_recovery"][quantity] = dict(
                mean_signed_relative_error=np.mean(errors, axis=0).tolist()
                if errors
                else None,
                median_absolute_relative_error=np.median(
                    np.abs(errors), axis=0
                ).tolist()
                if errors
                else None,
            )
    summary["audit"] = dict(
        status="not_applicable", reason="fit_only_verification_scope"
    )
    summary["outcomes"] = dict(Counter(a["outcome"] for a in analyses))
    summary["target_kind_denominators"] = dict(
        Counter(a["target_kind"] for a in analyses)
    )
    pairs = []
    for pid in entries:
        pair = [a for a in analyses if a["id"].startswith(pid + ":")]
        scores = [
            a.get("fresh_reporting_assessment", {}).get("centered_nll") for a in pair
        ]
        pairs.append(
            dict(
                problem_id=pid,
                lbfgsb_minus_trf=(scores[0] - scores[1])
                if len(scores) == 2 and all(v is not None for v in scores)
                else None,
            )
        )
    store.verify_inventory(tasks)
    return shared.jsonable(
        dict(
            scope=protocol.scope,
            **(dict(schema=protocol.report_schema) if protocol.report_schema else {}),
            summary=summary,
            attempts=records,
            analyses=analyses,
            paired_solver_gaps=pairs,
            preparation_work={
                pid: arrays.get("preparation_work")
                for pid, arrays in arrays_by_id.items()
            },
            reporting_work=work.record(),
            report_elapsed_seconds=perf_counter() - started,
            provenance=dict(
                execution_sha256=shared.digest(store.lock),
                inventory_sha256=inventory["payload_sha256"],
                verified=True,
            ),
        )
    )
