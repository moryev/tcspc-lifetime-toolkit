"""Temporary Issue-15 trace of the frozen Week-7 generator (remove after review).

The library still constructs all 810 histograms with its original seed and
parameter order. These wrappers only copy intermediate arrays. The output is
diagnostic evidence, never a replacement for an accepted reference or test.

Run ``python scripts/diagnose_week7_sample_770.py`` in each environment.
The CI step saves NPZ/JSON files in ``build/reports``. Once both NPZ files are
available, run ``python scripts/diagnose_week7_sample_770.py --compare <other.npz>``
on Windows to report the first differing count sample and upstream differences.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np
import scipy

import tcspc_toolkit.ml.evaluation as evaluation
import tcspc_toolkit.simulation.histograms as simulation


STAGES = (
    "ideal_decay",
    "gaussian_irf",
    "normalized_irf",
    "shifted_irf",
    "renormalized_irf",
    "convolved_signal",
    "expected_counts",
    "sampled_counts",
)
EXPECTED_HASH_770 = "035f408aa945ef4df47b43ab8c7cada90e05877f5f029323804d1c8a3e1e95ca"


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _summary(values: np.ndarray) -> dict[str, object]:
    return {
        "sha256": _hash(values),
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "sum": float(np.sum(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _capture(output: Path) -> dict[str, object]:
    time = np.arange(0.0, 20.0, 0.05)
    traces: dict[str, list[np.ndarray]] = {stage: [] for stage in STAGES}
    parameters: list[dict[str, float | int]] = []
    rng_before: list[dict[str, object]] = []
    normalization_calls = 0

    def trace(stage: str, original):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            traces[stage].append(np.array(result, copy=True))
            return result

        return wrapped

    def normalize(*args, **kwargs):
        nonlocal normalization_calls
        result = original_normalize(*args, **kwargs)
        stage = "normalized_irf" if normalization_calls % 2 == 0 else "renormalized_irf"
        traces[stage].append(np.array(result, copy=True))
        normalization_calls += 1
        return result

    def sample(*args, **kwargs):
        expected = kwargs["expected_counts"]
        rng = kwargs["rng"]
        traces["expected_counts"].append(np.array(expected, copy=True))
        rng_before.append(deepcopy(rng.bit_generator.state))
        result = original_sample(*args, **kwargs)
        traces["sampled_counts"].append(np.array(result, copy=True))
        return result

    def simulate(*args, **kwargs):
        parameters.append(
            {
                name: kwargs[name]
                for name in (
                    "lifetime_ns",
                    "signal_photon_count",
                    "background_per_bin",
                    "irf_centre_ns",
                    "irf_fwhm_ns",
                    "irf_shift_ns",
                )
            }
        )
        return original_simulate(*args, **kwargs)

    original_simulate = evaluation.simulate_irf_convolved_histogram
    original_normalize = simulation.normalize_irf
    original_sample = simulation.sample_photon_counts
    wrapped_functions = {
        "monoexponential_decay": "ideal_decay",
        "generate_gaussian_irf": "gaussian_irf",
        "shift_irf": "shifted_irf",
        "convolve_decay_with_irf": "convolved_signal",
    }
    with ExitStack() as stack:
        for name, stage in wrapped_functions.items():
            stack.enter_context(
                patch.object(simulation, name, trace(stage, getattr(simulation, name)))
            )
        stack.enter_context(patch.object(simulation, "normalize_irf", normalize))
        stack.enter_context(patch.object(simulation, "sample_photon_counts", sample))
        stack.enter_context(
            patch.object(evaluation, "simulate_irf_convolved_histogram", simulate)
        )
        benchmark = evaluation.generate_benchmark_measurements(
            time=time,
            lifetimes_ns=[0.5, 1.0, 2.0, 3.0, 4.5, 6.0],
            signal_photon_counts=[1000, 10000, 100000],
            background_levels=[0.0, 0.5, 2.0],
            irf_centre_ns=1.0,
            irf_fwhm_values_ns=[0.15, 0.30, 0.60],
            irf_shift_values_ns=[-0.20, -0.05, 0.0, 0.05, 0.20],
            random_seed=42,
        )

    arrays = {stage: np.stack(traces[stage]) for stage in STAGES}
    assert all(values.shape == (810, 400) for values in arrays.values())
    np.testing.assert_array_equal(arrays["sampled_counts"], benchmark.X_histograms)
    assert len(parameters) == len(rng_before) == 810
    assert normalization_calls == 1620
    assert int(benchmark.metadata.iloc[770]["sample_id"]) == 770

    runtime = io.StringIO()
    with redirect_stdout(runtime):
        np.show_runtime()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, time_ns=time, **arrays)
    report: dict[str, object] = {
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_runtime": runtime.getvalue(),
        },
        "protocol": {
            "seed": 42,
            "time_bins": len(time),
            "sample_count": len(parameters),
            "parameter_order": list(parameters[0]),
            "parameters": parameters,
            "metadata_by_sample": benchmark.metadata.to_dict(orient="records"),
            "metadata_0": benchmark.metadata.iloc[0].to_dict(),
            "metadata_770": benchmark.metadata.iloc[770].to_dict(),
        },
        "expected_reference_counts_770_sha256": EXPECTED_HASH_770,
        "observed_counts_770": _summary(arrays["sampled_counts"][770]),
        "rng_state_before_0": rng_before[0],
        "rng_state_before_770": rng_before[770],
        "rng_state_before_by_sample": rng_before,
        "stage_0": {stage: _summary(values[0]) for stage, values in arrays.items()},
        "stage_770": {stage: _summary(values[770]) for stage, values in arrays.items()},
        "counts_sha256_by_sample": [_hash(row) for row in arrays["sampled_counts"]],
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.item()),
        encoding="utf-8",
    )
    return report


def _compare(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    report: dict[str, object] = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        with np.load(candidate_path, allow_pickle=False) as candidate:
            counts_differ = np.any(
                reference["sampled_counts"] != candidate["sampled_counts"], axis=1
            )
            indices = np.flatnonzero(counts_differ)
            first = int(indices[0]) if indices.size else None
            report["first_different_count_sample"] = first
            report["different_count_samples"] = int(indices.size)
            report["counts_770_reference_sha256"] = _hash(
                reference["sampled_counts"][770]
            )
            report["counts_770_candidate_sha256"] = _hash(
                candidate["sampled_counts"][770]
            )
            samples = sorted(
                {
                    0,
                    770,
                    *(range(max(0, first - 1), first + 1) if first is not None else []),
                }
            )
            differences: dict[str, object] = {}
            for sample_id in samples:
                stage_differences: dict[str, object] = {}
                for stage in STAGES:
                    a = reference[stage][sample_id]
                    b = candidate[stage][sample_id]
                    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    changed = np.flatnonzero(a != b)
                    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
                    stage_differences[stage] = {
                        "reference_sha256": _hash(a),
                        "candidate_sha256": _hash(b),
                        "different_bins": int(changed.size),
                        "first_different_bin": int(changed[0])
                        if changed.size
                        else None,
                        "max_absolute_difference": float(np.max(delta)),
                        "max_difference_over_peak": (
                            float(np.max(delta)) / peak if peak else 0.0
                        ),
                        "reference_sum": float(np.sum(a)),
                        "candidate_sum": float(np.sum(b)),
                        "zero_vs_positive_bins": int(
                            np.count_nonzero((a == 0) != (b == 0))
                        ),
                    }
                differences[str(sample_id)] = stage_differences
            report["sample_differences"] = differences
    reference_json = reference_path.with_suffix(".json")
    candidate_json = candidate_path.with_suffix(".json")
    if reference_json.exists() and candidate_json.exists():
        reference_report = json.loads(reference_json.read_text(encoding="utf-8"))
        candidate_report = json.loads(candidate_json.read_text(encoding="utf-8"))
        if (
            "rng_state_before_by_sample" in reference_report
            and "rng_state_before_by_sample" in candidate_report
        ):
            reference_states = reference_report["rng_state_before_by_sample"]
            candidate_states = candidate_report["rng_state_before_by_sample"]
            changed_states = [
                i
                for i, (left, right) in enumerate(
                    zip(reference_states, candidate_states, strict=True)
                )
                if left != right
            ]
            report["first_different_rng_state_before_sample"] = (
                changed_states[0] if changed_states else None
            )
            report["rng_state_equal_before_first_count_difference"] = (
                reference_states[first] == candidate_states[first]
                if first is not None
                else None
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/reports/week7-sample-770.npz")
    )
    parser.add_argument("--compare", type=Path, help="NPZ from another environment")
    args = parser.parse_args()
    report = _capture(args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")
    print("WEEK7_770_ENVIRONMENT " + json.dumps(report["environment"]))
    print(
        "WEEK7_770_SELECTED "
        + json.dumps(
            {
                "metadata_0": report["protocol"]["metadata_0"],
                "metadata_770": report["protocol"]["metadata_770"],
                "stage_0": report["stage_0"],
                "stage_770": report["stage_770"],
            }
        )
    )
    hashes = report["counts_sha256_by_sample"]
    for start in range(0, len(hashes), 100):
        print(
            f"WEEK7_770_COUNT_HASHES_{start} " + json.dumps(hashes[start : start + 100])
        )
    if args.compare:
        print("WEEK7_770_COMPARISON " + json.dumps(_compare(args.output, args.compare)))


if __name__ == "__main__":
    main()
