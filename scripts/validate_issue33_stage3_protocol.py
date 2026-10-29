"""Validate the proposed Issue-33 protocol without generating or fitting data.

This command has no study-execution, RNG, checkpoint-writing or authorization
mode. The existing Issue-28 runner/locks remain historical and are not replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/issue33_stage3_qualification_v1.json"
CONDITIONS = (0, 4, 5, 8, 9, 10, 13, 16, 17, 21, 26)
AUDIT_CLASSES = (
    "agreement",
    "disagreement",
    "audit_incomplete",
    "production_unavailable",
)


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population(protocol):
    """Enumerate identities only; both C26 arms share one observation identity."""
    observations, production, audits = [], [], []
    for c in protocol["conditions"]:
        for replicate in range(c["replicates"]):
            observations.append((c["id"], replicate))
            for arm in c["arms"]:
                key = (c["id"], replicate, arm)
                production.append(key)
                if (
                    c["id"] == 0
                    or replicate in protocol["audit"]["secondary_replicates"]
                ):
                    audits.append(key)
    return observations, production, audits


def observation_entropy(protocol, key):
    return (*protocol["root_entropy"], protocol["streams"]["qualification"], *key)


def validate(protocol):
    """Reject population, physics, budget, source and approval-status drift."""
    require(protocol["protocol"] == "issue33_stage3_qualification_v1", "protocol ID")
    require(protocol["execution_status"] == "proposed", "review-only status required")
    require(protocol["observation_rng"]["draws_authorized"] is False, "draws forbidden")
    historical = protocol["historical_protocol"]
    require(
        historical["path"] == "configs/issue28_pr3_qualification.json",
        "historical protocol path",
    )
    path = ROOT / historical["path"]
    require(source_digest(path) == historical["lf_sha256"], "historical source hash")
    old = json.loads(path.read_text())
    require(
        source_digest(ROOT / "src/tcspc_toolkit/estimation/multiexponential.py")
        == protocol["production_source_lf_sha256"],
        "corrected fitter source hash",
    )
    require(
        protocol["implementation_commit"] == "5be88edda17ad6612a8447dca0e11a6543502b85",
        "implementation commit",
    )
    require(protocol["root_entropy"] == [33, 3, 20261029, 1], "new seed namespace")
    require(
        protocol["streams"]
        == dict(
            qualification=0,
            reserved_calibration=1,
            nonqualification=2,
            summary_bootstrap=3,
            protocol_tests=4,
        ),
        "separate stream roles",
    )
    require(protocol["observation_rng"]["bit_generator"] == "PCG64", "RNG definition")
    require(
        protocol["observation_rng"]["seed_sequence"]
        == "root_entropy + [qualification_stream, original_condition_id, replicate_index]",
        "entropy construction",
    )
    require(
        protocol["observation_rng"]["arms_share_observation"] is True, "paired arms"
    )
    for key in ("grid", "baseline", "irf", "fit_settings", "acceptance"):
        require(protocol[key] == old[key], f"historical {key} contract")
    require(
        tuple(c["id"] for c in protocol["conditions"]) == CONDITIONS, "condition IDs"
    )
    for c in protocol["conditions"]:
        original = next(v for v in old["conditions"] if v["id"] == c["id"])
        require(
            c == {**original, "replicates": 100 if c["id"] == 0 else 10},
            f"condition {c['id']} physics/replicates/arms",
        )
    primary = protocol["primary"]
    require(primary["condition_id"] == 0 and primary["attempts"] == 100, "baseline n")
    require(primary["minimum_numerically_accepted"] == 95, "baseline acceptance")
    require(primary["secondary_diagnostic_only"] is True, "secondary diagnostic role")
    require(
        primary["overall_pass_requires"]
        == [
            "baseline_numerical_target",
            "both_component_median_error_targets",
            "both_component_absolute_bias_targets",
            "baseline_audit_target",
            "complete_untampered_population",
        ],
        "primary gate conjunction",
    )
    audit = protocol["audit"]
    for key, value in old["audit"].items():
        if key not in ("secondary_replicates", "target_matched_conditions"):
            require(audit[key] == value, f"independent audit {key}")
    require(audit["secondary_replicates"] == [0, 4, 9], "fixed audit indices")
    require(audit["target_matched_conditions"] == [0], "confirmatory audit population")
    require(audit["primary_required_agreements"] == 100, "all baseline audits required")
    require(audit["secondary_diagnostic_only"] is True, "secondary audits diagnostic")
    require(audit["incomplete_counts_as_disagreement"] is False, "audit classification")
    execution = protocol["execution"]
    for key in (
        "workers",
        "numerical_threads",
        "production_case_timeout_seconds",
        "audit_timeout_seconds",
        "observations_saved_before_fit",
        "retry_completed_failures",
    ):
        require(execution[key] == old["execution"][key], f"execution {key}")
    require(execution["resume_only_never_started_tasks"] is True, "resume policy")
    require(
        execution["orphan_started_task"] == "terminal interrupted; no automatic retry",
        "orphan request policy",
    )
    require(
        execution["generation_forward_calls_separate"] is True, "generation accounting"
    )
    for key in (
        "external_production_assessment_forward_calls_per_available_candidate",
        "audit_shared_recheck_forward_calls_per_available_candidate",
    ):
        require(execution[key] == 1, "external recheck accounting")
    observations, production, audits = population(protocol)
    for keys in (observations, production, audits):
        require(len(keys) == len(set(keys)), "duplicate population identity")
    require(set(audits) <= set(production), "audits outside production population")
    require(
        len({observation_entropy(protocol, k) for k in observations}) == 200,
        "observation entropy identities",
    )
    totals = dict(
        conditions=len(protocol["conditions"]),
        observations=len(observations),
        production_biexponential_fits=len(production),
        production_mono_fits=0,
        scheduled_audits=len(audits),
        confirmatory_audits=sum(k[0] == 0 for k in audits),
        diagnostic_audits=sum(k[0] != 0 for k in audits),
        audit_solver_starts=8 * len(audits),
        paired_irf_observations=sum(
            c["replicates"] for c in protocol["conditions"] if len(c["arms"]) > 1
        ),
    )
    require(totals == protocol["expected_totals"], "declared workload totals")
    expected_outputs = dict(
        raw_directory="data/generated/issue33_stage3_qualification_v1",
        execution_lock="docs/validation/issue33_stage3_execution_lock_v1.json",
        results="docs/validation/issue33_stage3_results_v1.json",
        scientific_report="docs/issue33_stage3_qualification_v1.md",
    )
    require(protocol["outputs"] == expected_outputs, "separate evidence locations")
    return totals


def audit_tally(protocol, records):
    """Account for every scheduled audit, including missing/incomplete records.

    Records are classified terminal metadata, not observations or solver output.
    Duplicate/foreign identities are integrity errors, never extra successes.
    """
    expected = set(population(protocol)[2])
    seen = set()
    counts = {"primary": Counter(), "secondary": Counter()}
    for record in records:
        key = (record["condition_id"], record["replicate"], record["arm"])
        require(key in expected and key not in seen, "duplicate or foreign audit key")
        require(record["classification"] in AUDIT_CLASSES, "unknown audit class")
        seen.add(key)
        counts["primary" if key[0] == 0 else "secondary"][record["classification"]] += 1
    for key in expected - seen:
        counts["primary" if key[0] == 0 else "secondary"]["not_run"] += 1
    return dict(
        populations={
            group: {label: count[label] for label in (*AUDIT_CLASSES, "not_run")}
            for group, count in counts.items()
        },
        primary_audit_target_met=counts["primary"]["agreement"] == 100,
        all_scheduled_audits_terminal=not any(c["not_run"] for c in counts.values()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=CONFIG)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    print(
        json.dumps(
            dict(
                status="proposed_protocol_valid",
                observations_generated=0,
                totals=validate(protocol),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
