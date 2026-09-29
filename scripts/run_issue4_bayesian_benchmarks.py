"""Run committed Issue-4 paired benchmark profiles; no scientific math lives here."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from enum import Enum
from importlib import metadata
from pathlib import Path

import numpy as np

from tcspc_toolkit.bayesian import (
    BayesianPriorConfig, BoundedUniformPrior, GammaPrior, LogNormalPrior,
)
from tcspc_toolkit.bayesian_evaluation import (
    BayesianClassicalCondition,
    BayesianClassicalEvaluationConfig,
    evaluate_issue4_condition,
)
from tcspc_toolkit.bayesian_sampling import BayesianSamplingConfig
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "issue4_bayesian_workflow.json"
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "issue4"


def _prior(spec: dict) -> BayesianPriorConfig:
    amplitude_shape, amplitude_rate = spec["amplitude_gamma_shape_rate"]
    lifetime_log_mean, lifetime_log_std = spec["lifetime_lognormal_log_mean_std"]
    background_shape, background_rate = spec["background_gamma_shape_rate"]
    shift_lower, shift_upper = spec["shift_uniform_bounds_ns"]
    return BayesianPriorConfig(
        amplitude=GammaPrior(amplitude_shape, amplitude_rate),
        lifetime_ns=LogNormalPrior(lifetime_log_mean, lifetime_log_std),
        background_per_bin=GammaPrior(background_shape, background_rate),
        temporal_shift_prior=BoundedUniformPrior(shift_lower, shift_upper),
        fixed_temporal_shift_ns=None,
    )


def _flat(prefix: str, value: object, result: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flat(f"{prefix}_{key}" if prefix else str(key), item, result)
    elif isinstance(value, (list, tuple)):
        result[prefix] = json.dumps(value)
    elif isinstance(value, Enum):
        result[prefix] = value.value
    else:
        result[prefix] = value


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "scientific"), required=True)
    parser.add_argument(
        "--cost-pilot", action="store_true",
        help="scientific sampler on one observation with 20 bootstrap refits; never a coverage run",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.cost_pilot and args.profile != "scientific":
        parser.error("--cost-pilot requires --profile scientific")
    execution_label = "cost_pilot" if args.cost_pilot else args.profile
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    profile = manifest["profiles"][args.profile]
    grid = manifest["time_grid_ns"]
    time_ns = np.arange(grid["start"], grid["stop"], grid["step"], dtype=np.float64)
    irf_spec = manifest["irf"]
    source = generate_gaussian_irf_profile(
        time_ns, gaussian_centre_ns=irf_spec["centre_ns"],
        gaussian_fwhm_ns=irf_spec["fwhm_ns"],
        provenance={"workflow": "issue4_stage5_matched"},
    )
    prepared = prepare_irf(source, time_ns)
    model = manifest["common_model"]
    condition_specs = {item["id"]: item for item in manifest["conditions"]}
    selected = list(condition_specs) if profile["condition_ids"] == "all" else profile["condition_ids"]
    policies = manifest["prior_policies"]
    sampler = BayesianSamplingConfig(
        random_seed=0, credible_interval_level=model["nominal_interval_level"],
        **profile["bayesian_sampling"],
    )
    jobs = [(cid, policy, profile["n_repeats_per_condition"])
            for cid in selected for policy in profile["prior_policy_ids"]]
    sensitivity = profile.get("prior_sensitivity")
    if sensitivity:
        jobs.extend((cid, policy, sensitivity["n_repeats_per_condition"])
                    for cid in sensitivity["condition_ids"]
                    for policy in sensitivity["prior_policy_ids"])
    if args.cost_pilot:
        jobs = [("tau2_n10000_b0p5", "baseline", 1)]
    reports = []
    started = time.perf_counter()
    for job_index, (condition_id, prior_id, repeats) in enumerate(jobs, start=1):
        spec = condition_specs[condition_id]
        condition = BayesianClassicalCondition(
            condition_id=condition_id, time_ns=time_ns,
            generating_irf=prepared, assumed_irf=prepared,
            true_lifetime_ns=spec["lifetime_ns"],
            signal_photon_count=spec["signal_photon_count"],
            background_per_bin=spec["background_per_bin"],
            true_temporal_shift_ns=model["true_temporal_shift_ns"],
            n_repeats=repeats,
        )
        config = BayesianClassicalEvaluationConfig(
            profile=execution_label, prior_policy_id=prior_id,
            priors=_prior(policies[prior_id]), sampling_config=sampler,
            n_bootstrap_resamples=20 if args.cost_pilot else profile["n_bootstrap_resamples"],
            nominal_interval_level=model["nominal_interval_level"],
            temporal_shift_bounds_ns=tuple(model["temporal_shift_bounds_ns"]),
            random_seed=manifest["randomness"]["base_seed"],
            background_fraction=model["classical_background_fraction"],
        )
        print(f"[{job_index}/{len(jobs)}] {condition_id} / {prior_id}: {repeats} paired realizations", flush=True)
        report = evaluate_issue4_condition(condition, config)
        reports.append(report)
        timings = {entry.metric: entry.median_seconds for entry in report.summary.runtimes}
        statuses = [record.bayesian.status.value for record in report.per_realization]
        print(
            f"  Bayesian statuses={statuses}; median paired={timings['paired_orchestration']:.3f}s, "
            f"bootstrap={timings['parametric_bootstrap']:.3f}s, "
            f"Bayesian={timings['bayesian_inference_total']:.3f}s",
            flush=True,
        )
    runtime_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for report in reports:
        for record in report.per_realization:
            row: dict[str, object] = {}
            _flat("", asdict(record), row)
            rows.append(row)
    csv_path = args.output_dir / f"{execution_label}_realizations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / f"{execution_label}_summary.json"
    payload = {
        "profile": execution_label,
        "scientific_coverage_claim": execution_label == "scientific",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "versions": {name: metadata.version(name) for name in ("numpy", "scipy", "emcee", "tcspc-lifetime-toolkit")},
        "runtime_seconds": runtime_seconds,
        "n_jobs": len(jobs),
        "n_realizations": len(rows),
        "summaries": [asdict(report.summary) for report in reports],
    }
    summary_path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {summary_path}; runtime={runtime_seconds:.3f}s", flush=True)
    if execution_label != "scientific":
        print(f"{execution_label} only: no scientific coverage claim.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
