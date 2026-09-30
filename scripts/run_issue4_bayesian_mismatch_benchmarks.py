"""Run the separate Issue-4 Stage-6 mismatch manifest; scientific is manual only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tcspc_toolkit.bayesian_mismatch_evaluation import (
    build_mismatch_workflow,
    evaluate_mismatch_workflow,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "issue4_bayesian_mismatch_workflow.json"
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "issue4" / "mismatch"


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _csv_flatten(value, prefix=""):
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        raise TypeError("CSV record must be a dataclass or dictionary")
    flat = {}
    for name, item in value.items():
        key = f"{prefix}{name}"
        if isinstance(item, dict):
            flat.update(_csv_flatten(item, f"{key}_"))
        elif isinstance(item, (tuple, list, np.ndarray)):
            flat[key] = json.dumps(_json_safe(item), separators=(",", ":"))
        else:
            flat[key] = _json_safe(item)
    return flat


def _revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "scientific"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    workflow = build_mismatch_workflow(manifest, args.profile)
    print(
        f"Stage-6 {args.profile}: {len(workflow.conditions)} generating conditions, "
        f"{len(workflow.conditions) * workflow.config.n_repeats_per_condition} observations, "
        f"{sum(len(c.assumptions) for c in workflow.conditions) * workflow.config.n_repeats_per_condition} inferences",
        flush=True,
    )

    def progress(record, index, total):
        print(
            f"[{index}/{total}] {record.condition_id}/{record.assumption_id} "
            f"r={record.realization_index} classical={record.classical_fit.valid_fit} "
            f"Bayesian={record.bayesian.status.value} PPC={record.posterior_predictive.status} "
            f"{record.orchestration_seconds:.2f}s",
            flush=True,
        )

    report = evaluate_mismatch_workflow(workflow, progress=progress)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.profile}_realizations.csv"
    json_path = args.output_dir / f"{args.profile}_summary.json"
    rows = [_csv_flatten(record) for record in report.per_inference]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected_by_condition = dict(report.generating_expected_counts)
    condition_designs = []
    for condition in workflow.conditions:
        condition_designs.append({
            "condition_id": condition.condition_id,
            "mechanism": condition.mechanism,
            "generating_model": condition.generating_model,
            "primary_lifetime_ns": condition.primary_lifetime_ns,
            "secondary_lifetime_ns": condition.secondary_lifetime_ns,
            "secondary_detected_fraction": condition.secondary_detected_fraction,
            "mono_generating_lifetime_ns": condition.mono_generating_lifetime_ns,
            "signal_photon_count": condition.signal_photon_count,
            "background_per_bin": condition.background_per_bin,
            "true_temporal_shift_ns": condition.true_temporal_shift_ns,
            "generating_irf_id": condition.generating_irf_id,
            "assumptions": [
                {"assumption_id": item.assumption_id, "assumed_irf_id": item.irf_id}
                for item in condition.assumptions
            ],
            "expected_generating_curve": expected_by_condition[condition.condition_id],
            "generating_irf_kernel": condition.generating_irf.kernel.tolist(),
            "assumed_irf_kernels": {
                item.assumption_id: item.prepared_irf.kernel.tolist()
                for item in condition.assumptions
            },
        })
    payload = {
        "profile": args.profile,
        "purpose": manifest["profiles"][args.profile]["purpose"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_path": str(args.manifest),
        "code_revision": _revision(),
        "time_grid_ns": manifest["time_grid_ns"],
        "measurement_window_ns": manifest["measurement_window_ns"],
        "predictive_windows_ns": manifest["predictive_windows_ns"],
        "profile_config": manifest["profiles"][args.profile],
        "randomness": manifest["randomness"],
        "package_versions": {
            name: version(name) for name in ("tcspc-lifetime-toolkit", "numpy", "scipy", "emcee")
        },
        "condition_designs": condition_designs,
        "pseudo_true_references": report.pseudo_true_references,
        "summaries": report.summaries,
        "observations": len(workflow.conditions) * workflow.config.n_repeats_per_condition,
        "inference_evaluations": len(report.per_inference),
        "status_counts": {
            "classical_valid": sum(row.classical_fit.valid_fit for row in report.per_inference),
            "covariance_valid": sum(row.covariance.valid_interval for row in report.per_inference),
            "bootstrap_valid": sum(row.bootstrap.valid_interval for row in report.per_inference),
            "bayesian": dict(Counter(row.bayesian.status.value for row in report.per_inference)),
            "posterior_predictive": dict(Counter(
                row.posterior_predictive.status for row in report.per_inference
            )),
        },
        "runtime_seconds": report.runtime_seconds,
        "realization_csv": str(csv_path),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Completed in {report.runtime_seconds:.2f}s. Wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
