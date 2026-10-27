"""Verify and compact the completed Issue-28 PR-3 qualification checkpoints.

This reporting layer never generates observations, fits models, retries tasks,
or changes the frozen qualification runner. Invoke only after study completion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import runpy
import statistics


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    """Retain the finite contributor denominator for every descriptive summary."""
    values = [v for v in values if v is not None]
    return dict(
        n=len(values),
        minimum=min(values) if values else None,
        median=statistics.median(values) if values else None,
        mean=statistics.fmean(values) if values else None,
        maximum=max(values) if values else None,
    )


def warning_codes(row):
    diagnostics = row.get("identifiability", {})
    return [
        code
        for group in (
            "structural_warnings",
            "conditioning_warnings",
            "boundary_warnings",
            "multistart_warnings",
        )
        for code in diagnostics.get(group, [])
    ]


def condition_details(rows):
    """All attempts remain in counts; unavailable scalars retain explicit n."""
    completed = [r for r in rows if r["status"] == "completed"]
    finite_recovery = [
        r
        for r in completed
        if any(
            e is not None for e in r["recovery"]["all_finite_component_relative_errors"]
        )
    ]
    warning_free = [r for r in finite_recovery if not warning_codes(r)]
    accepted_warning_free = [
        r for r in warning_free if r["assessment"].get("accepted", False)
    ]
    accepted_recovery = [
        r for r in finite_recovery if r["assessment"].get("accepted", False)
    ]
    return dict(
        n_attempted=len(rows),
        n_numerically_accepted=sum(
            r["assessment"].get("accepted", False) for r in completed
        ),
        n_accepted_with_component_errors=len(accepted_recovery),
        n_accepted_any_component_error_above_5_percent=sum(
            any(
                e is not None and abs(e) > 0.05
                for e in r["recovery"]["all_finite_component_relative_errors"]
            )
            for r in accepted_recovery
        ),
        execution_status_counts=dict(Counter(r["status"] for r in rows)),
        warning_counts=dict(Counter(w for r in completed for w in warning_codes(r))),
        n_warning_free_completed=sum(not warning_codes(r) for r in completed),
        n_warning_free_with_component_errors=len(warning_free),
        n_warning_free_any_component_error_above_5_percent=sum(
            r["warning_free_error_above_5_percent"] for r in warning_free
        ),
        n_accepted_warning_free_with_component_errors=len(accepted_warning_free),
        n_accepted_warning_free_any_component_error_above_5_percent=sum(
            r["warning_free_error_above_5_percent"] for r in accepted_warning_free
        ),
        budget_exhaustion_counts=dict(
            Counter(
                b for r in completed for b in r["optimization"]["exhausted_budgets"]
            )
        ),
        n_mono_valid=sum(r["mono"]["valid_fit"] for r in completed),
        n_jacobian_available=sum(
            r["assessment"]["jacobian_available"] for r in completed
        ),
        jacobian_unavailable_reasons=dict(
            Counter(
                r["assessment"]["diagnostic_unavailable_reason"]
                for r in completed
                if not r["assessment"]["jacobian_available"]
            )
        ),
        curve_and_nuisance={
            name: describe((r.get("curve_metrics") or {}).get(name) for r in completed)
            for name in (
                "total_relative_l2",
                "signal_relative_l2",
                "signal_photon_error",
                "background_error",
                "shift_error_ns",
            )
        },
        descriptive_comparisons={
            name: describe(r["statistics"][name] for r in completed)
            for name in ("delta_aic", "delta_bic_bins", "delta_deviance")
        },
        n_negative_delta_aic=sum(
            r["statistics"]["delta_aic"] is not None
            and r["statistics"]["delta_aic"] < 0
            for r in completed
        ),
        n_negative_delta_bic_bins=sum(
            r["statistics"]["delta_bic_bins"] is not None
            and r["statistics"]["delta_bic_bins"] < 0
            for r in completed
        ),
        production_seconds=describe(r["production_seconds"] for r in completed),
        optimization_objective_calls=describe(
            r["optimization"]["objective_evaluations"] for r in completed
        ),
    )


def compact_attempt(condition, arm_index, row, payload_hash):
    """One scalar CSV row per scheduled comparison, including failed executions."""
    result = dict(
        condition_id=condition["id"],
        condition_name=condition["name"],
        arm=condition["arms"][arm_index],
        replicate=row.get("replicate"),
        status=row["status"],
        execution_failure_reason=row.get("reason"),
        execution_timeout_seconds=row.get("limit_seconds"),
        payload_sha256=payload_hash,
        observation_sha256=row.get("observation_sha256"),
        assumed_irf_sha256=row.get("assumed_irf_sha256"),
        accepted=row.get("assessment", {}).get("accepted", False),
        rejection_reasons="|".join(row.get("assessment", {}).get("reasons", [])),
        warnings="|".join(warning_codes(row)),
        exhausted_budgets="|".join(
            row.get("optimization", {}).get("exhausted_budgets", [])
        ),
        warning_free_error_above_5_percent=row.get(
            "warning_free_error_above_5_percent"
        ),
    )
    for section, names in {
        "assessment": ("diagnostic_budget_complete", "jacobian_available"),
        "bi_parameters": ("background", "temporal_shift_ns"),
        "bi_components": (
            "amplitude_weighted_lifetime_ns",
            "detected_signal_weighted_lifetime_ns",
            "ideal_photon_weighted_lifetime_ns",
        ),
        "optimization": (
            "objective_evaluations",
            "initialization_evaluations",
            "validation_evaluations",
            "diagnostic_evaluations",
            "stopping_reason",
            "numerical_validation_passed",
        ),
        "curve_metrics": (
            "total_relative_l2",
            "signal_relative_l2",
            "signal_photon_error",
            "background_error",
            "shift_error_ns",
        ),
        "mono": ("valid_fit", "failure_reason", "fitted_lifetime_ns"),
        "statistics": ("delta_aic", "delta_bic_bins", "delta_deviance"),
    }.items():
        for name in names:
            result[f"{section}_{name}"] = (row.get(section) or {}).get(name)
    for k in range(2):
        for section, names in {
            "bi_parameters": ("amplitudes", "lifetimes_ns"),
            "bi_components": (
                "amplitude_fractions",
                "detected_signal_fractions",
                "ideal_photon_fractions",
                "detected_signal_photons",
            ),
            "recovery": (
                "accepted_component_relative_errors",
                "all_finite_component_relative_errors",
                "fitted_component_available",
            ),
        }.items():
            for name in names:
                values = (row.get(section) or {}).get(name)
                result[f"{section}_{name}_{k + 1}"] = (
                    None if values is None else values[k]
                )
    for model in ("mono", "bi"):
        values = row.get("statistics", {}).get(model) or {}
        for name in (
            "deviance",
            "log_likelihood",
            "centered_negative_log_likelihood",
            "aic",
            "bic_bins",
        ):
            result[f"{model}_{name}"] = values.get(name)
    return result


def audit_summary(summaries, settings):
    """Keep scheduled failures and incomplete close objectives in denominators."""
    audits = [a for s in summaries for a in s["scheduled_audits"]]
    targeted = [
        a
        for s in summaries
        if s["arm"] == "matched"
        and s["condition_id"] in settings["target_matched_conditions"]
        for a in s["scheduled_audits"]
    ]
    tolerance = settings["maximum_centered_nll_difference"]

    def close(a):
        gap = a.get("absolute_nll_difference")
        return gap is not None and gap <= tolerance

    passed = sum(bool(a.get("complete")) and close(a) for a in targeted)
    return dict(
        n_scheduled=len(audits),
        n_completed=sum(bool(a.get("complete")) for a in audits),
        n_complete_and_agree=sum(bool(a.get("complete")) and close(a) for a in audits),
        n_best_objective_agrees=sum(close(a) for a in audits),
        n_targeted=len(targeted),
        n_targeted_complete_and_agree=passed,
        n_targeted_best_objective_agrees=sum(close(a) for a in targeted),
        target_verdict="PASS" if targeted and passed == len(targeted) else "FAIL",
        maximum_centered_nll_difference=tolerance,
        absolute_gap=describe(a.get("absolute_nll_difference") for a in audits),
        exhausted_budget_counts=dict(
            Counter(b for a in audits for b in a.get("exhausted_budgets", []))
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    runner = runpy.run_path(
        str(ROOT / "scripts/qualify_biexponential_reconvolution.py")
    )
    protocol = runner["load_protocol"]()
    lock = runner["source_lock"](protocol)
    require(
        lock == json.loads(args.execution_lock.read_text())["lock"],
        "Execution source/settings/environment lock mismatch",
    )
    require((args.directory / "lock.json").is_file(), "Existing study required")
    store = runner["Checkpoints"](args.directory, lock)
    # Completeness, fixed populations and bootstrap are the frozen implementation.
    summaries = runner["summarize_study"](store, protocol)
    inventory = []
    for path in sorted(args.directory.glob("*.json")):
        data = path.read_bytes()
        if path.name != "lock.json":
            store.get(path.stem)
        inventory.append(
            dict(
                name=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        )
    require(
        not list(args.directory.glob("*.partial")), "Unfinished checkpoint transport"
    )
    names = [r["name"] for r in inventory]
    for prefix, expected in (
        ("observation-s0-", 1550),
        ("production-s0-", 1850),
        ("audit-s0-", 205),
        ("projection-s0-", 36),
        ("proxy-s1", 1),
    ):
        require(
            sum(n.startswith(prefix) for n in names) == expected,
            f"Unexpected checkpoint count: {prefix}",
        )
    attempts, all_rows, numerical_failures = [], [], []
    summary_index = 0
    for condition in protocol["conditions"]:
        for ai, _arm in enumerate(condition["arms"]):
            rows = []
            for replicate in range(condition["replicates"]):
                key = f"production-s0-c{condition['id']:02d}-r{replicate:03d}-a{ai}"
                row = store.get(key)
                rows.append(row)
                all_rows.append(row)
                if not row.get("assessment", {}).get("accepted", False):
                    optimization = row.get("optimization", {})
                    starts = optimization.get("starts", [])
                    numerical_failures.append(
                        dict(
                            condition_id=condition["id"],
                            arm=condition["arms"][ai],
                            replicate=replicate,
                            status=row["status"],
                            assessment=row.get("assessment"),
                            parameters=row.get("bi_parameters"),
                            optimization={
                                k: v for k, v in optimization.items() if k != "starts"
                            },
                            start_kinds=dict(Counter(s["kind"] for s in starts)),
                            termination_messages=dict(
                                Counter(s["message"] for s in starts)
                            ),
                            successful_starts_with_nonfinite_endpoint=[
                                s["start_index"]
                                for s in starts
                                if s["optimizer_reported_success"]
                                and s["endpoint_centered_nll"] is None
                            ],
                        )
                    )
                compact = compact_attempt(condition, ai, row, runner["digest"](row))
                compact["replicate"] = replicate
                attempts.append(compact)
            summaries[summary_index]["descriptive_details"] = condition_details(rows)
            summary_index += 1
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(attempts[0]))
    writer.writeheader()
    writer.writerows(attempts)
    csv_data = csv_buffer.getvalue().encode("utf-8")
    timings = {
        stage: describe(
            store.get(n[:-5])["wall_seconds"]
            for n in names
            if n.startswith(f"runtime-{stage}-s0-")
        )
        for stage in ("production", "audit", "projection")
    }
    for stage, value in timings.items():
        value["total_seconds"] = None if not value["n"] else value["mean"] * value["n"]
        value["n_missing_timings"] = {
            "production": 1850,
            "audit": 205,
            "projection": 36,
        }[stage] - value["n"]
    report = dict(
        kind="completed_frozen_scientific_qualification",
        created_utc=datetime.now(timezone.utc).isoformat(),
        execution_lock=lock,
        reporting_source_lf_sha256=runner["source_digest"](Path(__file__)),
        expected_totals=protocol["expected_totals"],
        integrity=dict(
            checkpoint_files=len(inventory),
            checkpoint_bytes=sum(r["bytes"] for r in inventory),
            sorted_checkpoint_inventory_sha256=runner["digest"](inventory),
            all_checkpoint_payload_hashes_verified=True,
            attempt_csv_sha256=hashlib.sha256(csv_data).hexdigest(),
        ),
        runtime=timings,
        audits=audit_summary(summaries, protocol["audit"]),
        overall=condition_details(all_rows),
        numerical_failures=numerical_failures,
        conditions=summaries,
    )
    proxy = store.get("proxy-s1")
    report["independent_proxy_calibration"] = dict(
        counts_sha256=proxy["counts_sha256"],
        provenance=proxy["provenance"],
        kernel_sha256=None
        if proxy["kernel"] is None
        else runner["digest"](proxy["kernel"]),
        source_kind=None if proxy["source"] is None else proxy["source"]["source_kind"],
        preparation=proxy["preparation"],
        estimation_diagnostics=proxy["diagnostics"],
        interpretation="One fixed independent approximate proxy; numerical preparation does not verify experimental correctness",
    )
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    require(
        not csv_path.exists() and not json_path.exists(), "Do not overwrite evidence"
    )
    csv_path.write_bytes(csv_data)
    runner["write_new_json"](json_path, report)
    print(
        json.dumps(
            dict(
                report=str(json_path), attempts=str(csv_path), audits=report["audits"]
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
