"""Versioned Stage-6 pseudo-true correction from frozen inference records only.

This script never calls the mismatch scientific runner, samples observations,
fits noisy data, bootstraps, samples MCMC, or performs posterior prediction.
Only the five noise-free likelihood projections are optimized anew.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tcspc_toolkit.bayesian_mismatch_evaluation import (
    MismatchReferenceComparison,
    build_mismatch_workflow,
    construct_pseudo_true_reference,
    summarize_mismatch_records,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "issue4_bayesian_mismatch_workflow.json"
GENERATED = ROOT / "data" / "generated" / "issue4" / "mismatch"
CSV_PATH = GENERATED / "scientific_realizations.csv"
JSON_PATH = GENERATED / "scientific_summary.json"
OUTPUT = GENERATED / "scientific_reference_correction_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_float(value: str) -> float:
    return float(value) if value else math.nan


def _comparison_from_saved_row(
    row: dict[str, str], method: str, reference: float,
) -> MismatchReferenceComparison:
    """Recompute one target comparison from saved estimator/interval columns."""
    fields = {
        "covariance": ("classical_fit_lifetime_ns", "covariance_lifetime_std_ns",
                       "covariance_local_gaussian_lower_ns", "covariance_local_gaussian_upper_ns",
                       "covariance_valid_interval", "local_gaussian_covariance_interval"),
        "bootstrap": ("classical_fit_lifetime_ns", "bootstrap_lifetime_std_ns",
                      "bootstrap_percentile_lower_ns", "bootstrap_percentile_upper_ns",
                      "bootstrap_valid_interval", "parametric_bootstrap_percentile_interval"),
        "bayesian": ("bayesian_lifetime_median_ns", "bayesian_posterior_lifetime_std_ns",
                     "bayesian_credible_lower_ns", "bayesian_credible_upper_ns",
                     "bayesian_diagnostics_accepted", "bayesian_equal_tailed_credible_interval"),
    }
    point_key, std_key, lower_key, upper_key, valid_key, kind = fields[method]
    point = _as_float(row[point_key])
    lower, upper = _as_float(row[lower_key]), _as_float(row[upper_key])
    accepted = (row["bayesian_status"] == "success" and row[valid_key] == "True") if method == "bayesian" else row[valid_key] == "True"
    valid = accepted and all(map(math.isfinite, (lower, upper))) and lower <= upper
    point_valid = (valid if method == "bayesian" else row["classical_fit_valid_fit"] == "True") and math.isfinite(point)
    deviation = point - reference if point_valid else math.nan
    std = _as_float(row[std_key]) if valid else math.nan
    return MismatchReferenceComparison(
        method=method, interval_kind=kind, reference_name="pseudo_true_mono",
        reference_lifetime_ns=reference,
        point_lifetime_ns=point if point_valid else math.nan,
        deviation_ns=deviation, reported_std_ns=std,
        interval_lower_ns=lower if valid else math.nan,
        interval_upper_ns=upper if valid else math.nan,
        interval_width_ns=upper - lower if valid else math.nan,
        reference_included=(lower <= reference <= upper) if valid else None,
        deviation_to_reported_std=(deviation / std)
        if valid and std > 0 and math.isfinite(deviation) else math.nan,
    )


def _summaries_from_saved_rows(
    rows: list[dict[str, str]], references: dict[tuple[str, str], float],
) -> dict[tuple[str, str, str], dict]:
    records = []
    for row in rows:
        key = (row["condition_id"], row["assumption_id"])
        reference = references[key]
        comparisons = tuple(
            _comparison_from_saved_row(row, method, reference)
            for method in ("covariance", "bootstrap", "bayesian")
        )
        records.append(SimpleNamespace(
            condition_id=key[0], assumption_id=key[1], comparisons=comparisons,
        ))
    return {
        (item.condition_id, item.assumption_id, item.method): asdict(item)
        for item in summarize_mismatch_records(tuple(records))
    }


def _assert_original_aggregates_reproduced(
    original: dict, reconstructed: dict[tuple[str, str, str], dict],
) -> float:
    saved = {
        (item["condition_id"], item["assumption_id"], item["method"]): item
        for item in original["summaries"] if item["reference_name"] == "pseudo_true_mono"
    }
    if set(saved) != set(reconstructed):
        raise RuntimeError("saved and independently reconstructed group identities differ")
    maximum = 0.0
    for key, summary in reconstructed.items():
        for field, value in summary.items():
            historical = saved[key][field]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                maximum = max(maximum, abs(float(value) - float(historical)))
            elif value != historical:
                raise RuntimeError(f"saved aggregate differs at {key}/{field}")
    if maximum > 1e-10:
        raise RuntimeError(f"saved aggregate reproduction error {maximum} exceeds 1e-10")
    return maximum


def _revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    before_hashes = {"scientific_realizations.csv": _sha256(CSV_PATH),
                     "scientific_summary.json": _sha256(JSON_PATH)}
    manifest_hash = _sha256(MANIFEST)
    original = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if original["manifest_sha256"] != manifest_hash or original["profile"] != "scientific":
        raise RuntimeError("frozen Stage-6 report does not match its committed manifest")
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    workflow = build_mismatch_workflow(json.loads(MANIFEST.read_text(encoding="utf-8")), "scientific")
    expected_keys = {(c.condition_id, a.assumption_id, i)
                     for c in workflow.conditions for a in c.assumptions
                     for i in range(workflow.config.n_repeats_per_condition)}
    observed_keys = [(r["condition_id"], r["assumption_id"], int(r["realization_index"]))
                     for r in rows]
    if len(rows) != 100 or len(set(observed_keys)) != 100 or set(observed_keys) != expected_keys:
        raise RuntimeError("frozen realization table has unexpected or duplicated records")
    old_details = {(r["condition_id"], r["assumption_id"]): r
                   for r in original["pseudo_true_references"]}
    designs = {d["condition_id"]: d for d in original["condition_designs"]}
    new_details = {}
    for condition in workflow.conditions:
        curve = np.asarray(designs[condition.condition_id]["expected_generating_curve"], dtype=np.float64)
        curve_hash = hashlib.sha256(np.ascontiguousarray(curve, dtype="<f8").tobytes()).hexdigest()
        if any(row["expected_counts_sha256"] != curve_hash for row in rows
               if row["condition_id"] == condition.condition_id):
            raise RuntimeError("persisted noise-free expected curve/hash mismatch")
        for assumption in condition.assumptions:
            key = (condition.condition_id, assumption.assumption_id)
            new_details[key] = construct_pseudo_true_reference(
                condition, assumption, curve, workflow.config.temporal_shift_bounds_ns,
            )
    for row in rows:
        key = (row["condition_id"], row["assumption_id"])
        if abs(float(row["pseudo_true_mono_lifetime_ns"]) - old_details[key]["lifetime_ns"]) > 1e-12:
            raise RuntimeError("historical realization reference differs from summary")
    old_values = {key: detail["lifetime_ns"] for key, detail in old_details.items()}
    new_values = {key: detail.lifetime_ns for key, detail in new_details.items()}
    old_summaries = _summaries_from_saved_rows(rows, old_values)
    old_discrepancy = _assert_original_aggregates_reproduced(original, old_summaries)
    new_summaries = _summaries_from_saved_rows(rows, new_values)
    corrections = []
    affected_fields = set()
    for key, corrected in new_summaries.items():
        historical = old_summaries[key]
        for field in corrected:
            if corrected[field] != historical[field]:
                affected_fields.add(field)
        corrections.append({
            "condition_id": key[0], "assumption_id": key[1], "method": key[2],
            "original": historical, "corrected": corrected,
            "original_inclusion_count": historical["n_reference_included"],
            "corrected_inclusion_count": corrected["n_reference_included"],
        })
    per_inference_corrections = []
    for row in rows:
        key = (row["condition_id"], row["assumption_id"])
        if new_values[key] == old_values[key]:
            continue
        for method in ("covariance", "bootstrap", "bayesian"):
            historical = _comparison_from_saved_row(row, method, old_values[key])
            corrected = _comparison_from_saved_row(row, method, new_values[key])
            per_inference_corrections.append({
                "condition_id": key[0], "assumption_id": key[1],
                "realization_index": int(row["realization_index"]), "method": method,
                "original_deviation_ns": historical.deviation_ns,
                "corrected_deviation_ns": corrected.deviation_ns,
                "original_deviation_to_reported_std": historical.deviation_to_reported_std,
                "corrected_deviation_to_reported_std": corrected.deviation_to_reported_std,
                "original_reference_included": historical.reference_included,
                "corrected_reference_included": corrected.reference_included,
            })
    physical_summaries = {
        (item["condition_id"], item["assumption_id"], item["method"]): item
        for item in original["summaries"] if item["reference_name"] == "primary_component"
    }
    physical_decomposition = []
    for condition in workflow.conditions:
        if condition.generating_model != "biexponential":
            continue
        assumption = condition.assumptions[0]
        key = (condition.condition_id, assumption.assumption_id)
        for method in ("covariance", "bayesian"):
            physical = physical_summaries[(*key, method)]
            pseudo = new_summaries[(*key, method)]
            model_displacement = new_values[key] - condition.primary_lifetime_ns
            if abs(physical["bias_ns"] - (pseudo["bias_ns"] + model_displacement)) > 1e-12:
                raise RuntimeError("physical/pseudo-true bias decomposition failed")
            physical_decomposition.append({
                "condition_id": key[0],
                "estimator": pseudo["empirical_estimator"],
                "bias_vs_primary_ns": physical["bias_ns"],
                "bias_vs_corrected_pseudo_true_ns": pseudo["bias_ns"],
                "corrected_pseudo_true_minus_primary_ns": model_displacement,
                "mae_vs_corrected_pseudo_true_ns": pseudo["mae_ns"],
                "empirical_spread_ns": pseudo["empirical_spread_ns"],
            })
    payload = {
        "correction_schema_version": 1,
        "purpose": "Reanalysis of deterministic pseudo-true references only; not a replacement Stage-6 scientific run.",
        "no_inference_rerun_statement": "No MCMC, bootstrap, noisy-observation classical fits, PPC, or observation regeneration was performed.",
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_artifact_sha256": before_hashes,
            "manifest_sha256": manifest_hash,
            "historical_scientific_code_revision": original["code_revision"],
            "correction_git_head": _revision(),
            "correction_source_sha256": _sha256(ROOT / "src" / "tcspc_toolkit" / "bayesian_mismatch_evaluation.py"),
            "correction_script_sha256": _sha256(Path(__file__)),
            "package_versions": {name: version(name) for name in
                                 ("tcspc-lifetime-toolkit", "numpy", "scipy")},
            "reference_optimizer_policy": "centered Poisson NLL; 4 diverse starts; gaps <=1e-6 NLL and <=1e-5 ns; coordinate descent <=1e-5 NLL; scaled gradient <=1e-3; independent deviance least-squares cross-check",
        },
        "reference_corrections": [
            {"condition_id": key[0], "assumption_id": key[1],
             "original": old_details[key], "corrected": asdict(new_details[key]),
             "lifetime_delta_ns": new_details[key].lifetime_ns - old_details[key]["lifetime_ns"],
             "poisson_nll_improvement": old_details[key]["poisson_nll"] - new_details[key].poisson_nll}
            for key in old_details
        ],
        "recomputed_reference_name": "pseudo_true_mono",
        "affected_aggregate_fields": sorted(affected_fields),
        "historical_aggregate_reproduction_max_abs_error": old_discrepancy,
        "reference_dependent_summaries": corrections,
        "reference_dependent_per_inference_corrections": per_inference_corrections,
        "physical_pseudo_true_decomposition": physical_decomposition,
        "physical_reference_results_unchanged": True,
        "saved_inference_records": len(rows),
    }
    if before_hashes != {"scientific_realizations.csv": _sha256(CSV_PATH),
                         "scientific_summary.json": _sha256(JSON_PATH)}:
        raise RuntimeError("frozen artifacts changed during correction")
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} from {len(rows)} frozen inference rows; "
          f"historical aggregate max error {old_discrepancy:.3g}")


if __name__ == "__main__":
    main()
