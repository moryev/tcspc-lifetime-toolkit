"""Classical-only diagnostic audit of the frozen Issue-4 Stage-5 observations.

This script regenerates the saved histograms and refits them with the current
classical path. It never runs Bayesian inference or edits the scientific CSV,
summary, or committed manifest. Its CSV is a separate, ignored diagnostic.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tcspc_toolkit.bayesian_evaluation import (
    BayesianClassicalCondition,
    build_issue4_expected_counts,
    derive_issue4_seed,
    sample_issue4_observation,
)
from tcspc_toolkit.classical_evaluation import fit_single_reconvolution_curve
from tcspc_toolkit.fitting import (
    _poisson_coordinate_descent_check,
    poisson_negative_log_likelihood,
)
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import prepare_irf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "issue4_bayesian_workflow.json"
REALIZATIONS = ROOT / "data" / "generated" / "issue4" / "scientific_realizations.csv"
SUMMARY = ROOT / "data" / "generated" / "issue4" / "scientific_summary.json"
AUDIT = ROOT / "data" / "generated" / "issue4" / "stage5_optimizer_audit.csv"
PARAMETERS = (
    ("amplitude", "fitted_amplitude"),
    ("lifetime_ns", "fitted_lifetime_ns"),
    ("background_per_bin", "fitted_background"),
    ("temporal_shift_ns", "fitted_temporal_shift_ns"),
)


def main() -> None:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    report = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if hashlib.sha256(manifest_bytes).hexdigest() != report["manifest_sha256"]:
        raise ValueError("committed manifest differs from the frozen scientific report")

    grid = manifest["time_grid_ns"]
    time = np.arange(grid["start"], grid["stop"], grid["step"], dtype=np.float64)
    irf_spec = manifest["irf"]
    prepared = prepare_irf(
        generate_gaussian_irf_profile(
            time,
            gaussian_centre_ns=irf_spec["centre_ns"],
            gaussian_fwhm_ns=irf_spec["fwhm_ns"],
            provenance={"workflow": "issue4_stage5_matched"},
        ),
        time,
    )
    model = manifest["common_model"]
    conditions = {}
    for spec in manifest["conditions"]:
        condition = BayesianClassicalCondition(
            condition_id=spec["id"],
            time_ns=time,
            generating_irf=prepared,
            assumed_irf=prepared,
            true_lifetime_ns=spec["lifetime_ns"],
            signal_photon_count=spec["signal_photon_count"],
            background_per_bin=spec["background_per_bin"],
            true_temporal_shift_ns=model["true_temporal_shift_ns"],
            n_repeats=manifest["profiles"]["scientific"]["n_repeats_per_condition"],
        )
        expected, _ = build_issue4_expected_counts(condition)
        conditions[condition.condition_id] = condition, expected

    with REALIZATIONS.open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    if len(original_rows) != report["n_realizations"] or len(original_rows) != 480:
        raise ValueError("unexpected number of frozen scientific records")

    audit_rows = []
    seen = set()
    for saved in original_rows:
        condition_id = saved["condition_id"]
        prior_id = saved["prior_policy_id"]
        index = int(saved["realization_index"])
        key = condition_id, prior_id, index
        if key in seen:
            raise ValueError(f"duplicate scientific record: {key}")
        seen.add(key)
        condition, expected = conditions[condition_id]
        observation_seed = derive_issue4_seed(
            manifest["randomness"]["base_seed"], condition_id, index,
            "observation", prior_policy_id=prior_id,
        )
        if observation_seed != int(saved["seeds_observation"]):
            raise ValueError(f"observation seed mismatch: {key}")
        counts = sample_issue4_observation(
            condition, expected, observation_seed, index,
        ).require_raw_counts()
        digest = hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()
        if digest != saved["observed_counts_sha256"]:
            raise ValueError(f"observed-count hash mismatch: {key}")

        current = fit_single_reconvolution_curve(
            time=time,
            counts=counts,
            irf=prepared.kernel,
            temporal_shift_bounds=tuple(model["temporal_shift_bounds_ns"]),
            objective="poisson",
            background_fraction=model["classical_background_fraction"],
        )
        old_parameters = np.array(
            [float(saved[f"classical_fit_{name}"]) for name, _ in PARAMETERS]
        )
        new_parameters = np.array(
            [getattr(current, attribute) for _, attribute in PARAMETERS]
        )
        initial_scales = np.array([
            max(abs(current.initial_amplitude), 1.0),
            max(abs(current.initial_lifetime_ns), time[1] - time[0]),
            max(abs(current.initial_background), 1.0),
            max(
                abs(current.initial_temporal_shift_ns),
                0.1 * (model["temporal_shift_bounds_ns"][1] -
                       model["temporal_shift_bounds_ns"][0]),
                time[1] - time[0],
            ),
        ])
        scaled_bounds = list(zip(
            np.array([0.0, 1e-12, 1e-12, model["temporal_shift_bounds_ns"][0]])
            / initial_scales,
            np.array([np.inf, np.inf, np.inf, model["temporal_shift_bounds_ns"][1]])
            / initial_scales,
        ))

        def nll(scaled_parameters: np.ndarray) -> float:
            return poisson_negative_log_likelihood(
                counts,
                monoexponential_reconvolution_expected_counts(
                    time, prepared.kernel, *(scaled_parameters * initial_scales),
                ),
            )

        old_validation, old_max_descent = _poisson_coordinate_descent_check(
            nll, old_parameters / initial_scales, scaled_bounds,
        )
        old_nll = nll(old_parameters / initial_scales)
        old_valid = saved["classical_fit_valid_fit"] == "True"
        row = {
            "condition_id": condition_id,
            "prior_policy_id": prior_id,
            "realization_index": index,
            "observation_seed": observation_seed,
            "observed_counts_sha256": digest,
            "old_fit_success": old_valid,
            "old_poisson_nll": old_nll,
            "old_numerical_validation_passed": old_validation,
            "old_max_coordinate_descent_nll": old_max_descent,
            "new_optimizer_reported_success": current.optimizer_success,
            "new_fit_success": current.valid_fit,
            "new_poisson_nll": current.poisson_nll,
            "new_numerical_validation_passed": current.numerical_validation_passed,
            "new_max_coordinate_descent_nll": current.max_coordinate_descent_nll,
            "recovery_attempted": current.recovery_attempted,
            "new_failure_reason": current.failure_reason,
            "new_optimizer_status": current.optimizer_status,
            "new_optimizer_message": current.optimizer_message,
            "new_optimizer_nfev": current.optimizer_nfev,
            "new_optimizer_njev": current.optimizer_njev,
        }
        for (name, _), old, new in zip(PARAMETERS, old_parameters, new_parameters):
            row[f"old_{name}"] = old
            row[f"new_{name}"] = new
            row[f"delta_{name}"] = new - old
        audit_rows.append(row)

    with AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    old_failed = [row for row in audit_rows
                  if row["old_fit_success"] and not row["old_numerical_validation_passed"]]
    recovered = [row for row in old_failed
                 if row["new_fit_success"] and row["recovery_attempted"]]
    remaining = [row for row in old_failed if not row["new_fit_success"]]
    changes = np.array([abs(row["delta_lifetime_ns"]) for row in audit_rows])
    print(f"Audited {len(audit_rows)} rows; {len(seen)} unique realization records.")
    print(f"Old accepted fits failing local check: {len(old_failed)}")
    print(f"Recovered: {len(recovered)}; remaining failed: {len(remaining)}")
    print(f"Affected conditions: {sorted({row['condition_id'] for row in old_failed})}")
    print(f"Absolute lifetime change, max={np.max(changes):.12g} ns; "
          f"median={np.median(changes):.12g} ns")
    for row in old_failed:
        print(
            f"{row['condition_id']} / {row['prior_policy_id']} / "
            f"{row['realization_index']}: "
            f"old tau={row['old_lifetime_ns']:.12g}, "
            f"new tau={row['new_lifetime_ns']:.12g}, "
            f"old NLL={row['old_poisson_nll']:.12g}, "
            f"new NLL={row['new_poisson_nll']:.12g}"
        )
    print(f"Diagnostic CSV: {AUDIT}")


if __name__ == "__main__":
    main()
