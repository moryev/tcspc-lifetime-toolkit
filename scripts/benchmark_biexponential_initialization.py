"""Declared Issue-28 PR-2 development experiment; never frozen benchmark data.

The held-out mode requires a previously written policy lock. Selection reads
development rows only. JSONL outputs are exclusive-create, so attempts cannot
silently overwrite earlier evidence. Scientific fitting stays in the package.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import kl_div
from scipy.stats import exponnorm

from tcspc_toolkit.estimation.multiexponential import (
    BiexponentialFitSettings,
    _fit_core,
    fit_biexponential_reconvolution,
)
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/issue28_pr2_initialization.json"
IMPLEMENTATION = ROOT / "src/tcspc_toolkit/estimation/multiexponential.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def build_case(protocol, condition_id, stream=None):
    """Explicit independent synthetic source; no real measurement claim."""
    c = dict(protocol["baseline"])
    c.update(
        next(item for item in protocol["conditions"] if item["id"] == condition_id)
    )
    t = np.linspace(0, c["window_ns"], c["bins"], endpoint=c["endpoint"])
    sigma = c["irf_fwhm_ns"] / np.sqrt(8 * np.log(2))
    kernel = np.exp(-0.5 * ((t - c["irf_center_ns"]) / sigma) ** 2)
    if c["irf_kind"] == "emg":
        kernel = exponnorm.pdf(
            t, c["irf_tail_ns"] / sigma, loc=c["irf_center_ns"], scale=sigma
        )
    elif c["irf_kind"] == "double_peak":
        weight = c["secondary_irf_weight"]
        kernel = (1 - weight) * kernel + weight * np.exp(
            -0.5
            * ((t - c["irf_center_ns"] - c["secondary_irf_offset_ns"]) / sigma) ** 2
        )
    kernel /= np.trapezoid(kernel, x=t)
    components = multiexponential_reconvolution_component_counts(
        t,
        kernel,
        amplitudes=[1, 1],
        lifetimes=c["lifetimes_ns"],
        temporal_shift=c["shift_ns"],
    )
    amplitudes = (
        c["signal_photons"] * np.array(c["detected_fractions"]) / components.sum(axis=1)
    )
    expected = amplitudes @ components + c["background"]
    observations = None
    if stream is not None:
        rng = np.random.default_rng(
            np.random.SeedSequence([*protocol["seed_entropy"], condition_id, stream])
        )
        observations = rng.poisson(expected)
    return t, kernel, expected, observations, amplitudes, c


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_row(handle, row):
    handle.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
    handle.flush()


def settings_for(protocol, policy):
    return BiexponentialFitSettings(policy=policy, **protocol["experimental_limits"])


def fit_row(protocol, condition_id, stream, settings, mode, repetition=0):
    t, kernel, expected, counts, amplitudes, c = build_case(
        protocol, condition_id, stream
    )
    result = fit_biexponential_reconvolution(
        t,
        counts,
        kernel,
        lifetime_bounds_ns=tuple(c["lifetime_bounds_ns"]),
        temporal_shift_bounds_ns=tuple(c["temporal_shift_bounds_ns"]),
        settings=settings,
    )
    opt = asdict(result.optimization)
    row = dict(
        kind="fit",
        mode=mode,
        condition_id=condition_id,
        condition=c["name"],
        stream=stream,
        repetition=repetition,
        policy=settings.policy,
        settings=asdict(settings),
        optimization=opt,
        parameters=None if result.parameters is None else asdict(result.parameters),
        identifiability=asdict(result.identifiability),
        centered_nll=None,
        curve_relative_l2=None,
        lifetime_relative_errors=None,
        amplitude_relative_errors=None,
        detected_fraction_errors=None,
    )
    if result.parameters is not None:
        row["centered_nll"] = float(
            0.5 * calculate_poisson_deviance(counts, result.expected_counts)
        )
        row["curve_relative_l2"] = float(
            np.linalg.norm(result.expected_counts - expected) / np.linalg.norm(expected)
        )
        if min(amplitudes) > 0 and c["lifetimes_ns"][0] != c["lifetimes_ns"][1]:
            row["lifetime_relative_errors"] = (
                np.asarray(result.parameters.lifetimes_ns) / c["lifetimes_ns"] - 1
            ).tolist()
            row["amplitude_relative_errors"] = (
                np.asarray(result.parameters.amplitudes) / amplitudes - 1
            ).tolist()
        if result.components.detected_signal_fractions is not None:
            row["detected_fraction_errors"] = (
                np.asarray(result.components.detected_signal_fractions)
                - c["detected_fractions"]
            ).tolist()
    return row


def independent_audit(protocol, condition_id, stream=0):
    """Different solver, physical taus, direct convolution, independent loss."""
    t, kernel, _, y, _, c = build_case(protocol, condition_id, stream)
    settings = settings_for(protocol, "exhaustive")
    result = fit_biexponential_reconvolution(
        t,
        y,
        kernel,
        lifetime_bounds_ns=tuple(c["lifetime_bounds_ns"]),
        temporal_shift_bounds_ns=tuple(c["temporal_shift_bounds_ns"]),
        settings=settings,
    )
    scale = max(float(np.sum(y)) / t.size, 1)
    calls = 0
    dt = t[1] - t[0]

    def direct_mu(x):
        nonlocal calls
        calls += 1
        shifted = np.interp(t - x[5], t, kernel, left=0, right=0)
        curves = [
            np.convolve(np.exp(-t / tau), shifted)[: t.size] * dt for tau in x[2:4]
        ]
        return scale * (x[0] * curves[0] + x[1] * curves[1] + x[4])

    def residuals(x):
        mu = direct_mu(x)
        terms = np.maximum(kl_div(y, mu), 0)
        return np.sign(y - mu) * np.sqrt(2 * terms)

    lower = [
        0,
        0,
        c["lifetime_bounds_ns"][0],
        c["lifetime_bounds_ns"][0],
        0,
        c["temporal_shift_bounds_ns"][0],
    ]
    upper = [
        np.inf,
        np.inf,
        c["lifetime_bounds_ns"][1],
        c["lifetime_bounds_ns"][1],
        np.inf,
        c["temporal_shift_bounds_ns"][1],
    ]
    best_mu, best = None, np.inf
    started = perf_counter()
    # These starts depend only on bounds and observations, never fitted or
    # generating parameters. They differ from the production five-point bank.
    grid = np.geomspace(c["lifetime_bounds_ns"][0], c["lifetime_bounds_ns"][1], 7)
    records = []
    for i, j in ((1, 4), (2, 5), (1, 5), (2, 4)):
        for shift in (-0.1, 0.1):
            start = [
                1.0,
                1.0,
                grid[i],
                grid[j],
                0.01,
                float(np.clip(shift, lower[5], upper[5])),
            ]
            solved = least_squares(
                residuals,
                start,
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-9,
                max_nfev=1500,
            )
            mu = direct_mu(solved.x)
            value = 0.5 * calculate_poisson_deviance(y, mu)
            records.append(
                dict(
                    nll=float(value),
                    success=bool(solved.success),
                    nfev=int(solved.nfev),
                )
            )
            if value < best:
                best, best_mu = value, mu
    actual = result.statistics.centered_negative_log_likelihood
    return dict(
        kind="audit",
        condition_id=condition_id,
        stream=stream,
        production_nll=actual,
        independent_nll=float(best),
        absolute_difference=float(abs(actual - best)),
        passed=bool(
            abs(actual - best)
            <= protocol["criteria"]["independent_audit_nll_difference"]
        ),
        independent_forward_evaluations=calls,
        independent_seconds=perf_counter() - started,
        curve_difference=float(
            np.linalg.norm(best_mu - result.expected_counts) / np.linalg.norm(best_mu)
        ),
        independent_starts=records,
    )


def select_policy(protocol, source, output):
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    headers = [r for r in rows if r["kind"] == "metadata"]
    if (
        len(headers) != 1
        or headers[0]["mode"] != "development"
        or headers[0]["config_sha256"] != digest(CONFIG)
    ):
        raise ValueError("selection requires one matching development experiment")
    fits = [r for r in rows if r["kind"] == "fit"]
    expected_keys = {
        (c["id"], stream, policy)
        for c in protocol["conditions"]
        for stream in protocol["development_streams"]
        for policy in protocol["policies"]
    }
    keys = [(r["condition_id"], r["stream"], r["policy"]) for r in fits]
    if (
        len(keys) != len(set(keys))
        or set(keys) != expected_keys
        or any(r["mode"] != "development" for r in fits)
    ):
        raise ValueError(
            "selection requires exactly all declared development rows, never held-out rows"
        )
    summaries = []
    for policy in protocol["policies"]:
        group = [r for r in fits if r["policy"] == policy]
        gaps, failures = [], []
        for row in group:
            reference = min(
                (
                    r["centered_nll"]
                    for r in fits
                    if r["condition_id"] == row["condition_id"]
                    and r["stream"] == row["stream"]
                    and r["centered_nll"] is not None
                ),
                default=np.inf,
            )
            gap = (
                np.inf
                if row["centered_nll"] is None
                else row["centered_nll"] - reference
            )
            gaps.append(gap)
            opt = row["optimization"]
            if (
                gap > protocol["criteria"]["objective_gap_absolute"]
                or opt["numerical_validation_passed"] is not True
                or any(
                    b in opt["exhausted_budgets"]
                    for b in ("optimization", "validation", "starts")
                )
            ):
                failures.append(
                    dict(
                        condition_id=row["condition_id"],
                        stream=row["stream"],
                        gap=gap,
                        numerical_validation=opt["numerical_validation_passed"],
                        exhausted=opt["exhausted_budgets"],
                    )
                )
        elapsed = [r["optimization"]["total_seconds"] for r in group]
        summaries.append(
            dict(
                policy=policy,
                eligible=not failures,
                failures=failures,
                maximum_objective_gap=max(gaps),
                median_total_elapsed_seconds=float(np.median(elapsed)),
                p90_total_elapsed_seconds=float(np.quantile(elapsed, 0.9)),
                total_forward_evaluations=sum(
                    sum(
                        r["optimization"][key]
                        for key in (
                            "initialization_evaluations",
                            "objective_evaluations",
                            "validation_evaluations",
                            "diagnostic_evaluations",
                        )
                    )
                    for r in group
                ),
            )
        )
    eligible = sorted(
        (s for s in summaries if s["eligible"]),
        key=lambda s: (
            s["median_total_elapsed_seconds"],
            s["p90_total_elapsed_seconds"],
            s["total_forward_evaluations"],
            s["policy"],
        ),
    )
    chosen = None
    if eligible:
        policy = eligible[0]["policy"]
        group = [r for r in fits if r["policy"] == policy]
        initial = settings_for(protocol, policy)

        def budget(key, quantum, ceiling):
            return min(
                ceiling,
                max(
                    quantum,
                    int(
                        np.ceil(
                            1.5 * max(r["optimization"][key] for r in group) / quantum
                        )
                    )
                    * quantum,
                ),
            )

        chosen = asdict(
            replace(
                initial,
                max_starts=min(
                    46,
                    max(
                        4,
                        int(
                            np.ceil(
                                1.5
                                * max(len(r["optimization"]["starts"]) for r in group)
                            )
                        ),
                    ),
                ),
                max_objective_evaluations=budget("objective_evaluations", 1000, 100000),
                max_validation_evaluations=budget("validation_evaluations", 50, 2000),
                max_diagnostic_evaluations=budget("diagnostic_evaluations", 10, 100),
            )
        )
    report = dict(
        kind="selection",
        config_sha256=digest(CONFIG),
        comparison_implementation_sha256=headers[0]["implementation_sha256"],
        selected_settings=chosen,
        summaries=summaries,
        selection_observations="development_streams_0_and_1_only",
        development_references=[
            dict(
                condition_id=c["id"],
                stream=stream,
                centered_nll=min(
                    (
                        r["centered_nll"]
                        for r in fits
                        if r["condition_id"] == c["id"]
                        and r["stream"] == stream
                        and r["centered_nll"] is not None
                    ),
                    default=np.inf,
                ),
            )
            for c in protocol["conditions"]
            for stream in protocol["development_streams"]
        ],
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2, allow_nan=False)
    print(json.dumps(_jsonable(report), indent=2))


def noise_free_qualification(settings):
    time = np.linspace(0, 20, 512)
    kernel = np.exp(-0.5 * ((time - 1) / (0.3 / np.sqrt(8 * np.log(2)))) ** 2)
    kernel /= np.trapezoid(kernel, x=time)
    shifted = np.interp(time - 0.075, time, kernel, left=0, right=0)
    unit = np.array(
        [
            np.convolve(np.exp(-time / tau), shifted)[: time.size] * (time[1] - time[0])
            for tau in (0.8, 3.2)
        ]
    )
    amplitudes = 1e6 * np.array([0.4, 0.6]) / unit.sum(axis=1)
    expected = amplitudes @ unit + 2
    result = _fit_core(
        time,
        expected,
        kernel,
        lifetime_bounds_ns=(0.1, 8),
        temporal_shift_bounds_ns=(-0.3, 0.3),
        settings=settings,
        raw_counts=False,
    )
    return dict(
        kind="noise_free",
        settings=asdict(settings),
        parameters=asdict(result.parameters),
        amplitude_relative_errors=(
            np.asarray(result.parameters.amplitudes) / amplitudes - 1
        ).tolist(),
        lifetime_relative_errors=(
            np.asarray(result.parameters.lifetimes_ns) / [0.8, 3.2] - 1
        ).tolist(),
        background_relative_error=result.parameters.background / 2 - 1,
        shift_absolute_error_ns=abs(result.parameters.temporal_shift_ns - 0.075),
        curve_relative_l2=float(
            np.linalg.norm(result.expected_counts - expected) / np.linalg.norm(expected)
        ),
        optimization=asdict(result.optimization),
        identifiability=asdict(result.identifiability),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "pilot",
            "development",
            "select",
            "freeze",
            "held-out",
            "timing",
            "audit",
            "development-check",
            "noise-free",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument(
        "--policies", nargs="+", choices=("economical", "adaptive", "exhaustive")
    )
    args = parser.parse_args()
    protocol = load_protocol()
    if args.mode == "select":
        select_policy(protocol, args.source, args.output)
        return
    if args.mode == "freeze":
        selection = json.loads(args.source.read_text(encoding="utf-8"))
        if not selection["selected_settings"] or selection["config_sha256"] != digest(
            CONFIG
        ):
            raise ValueError("a qualifying development-only selection is required")
        if args.validation is None:
            raise ValueError(
                "freeze requires --validation with the derived-budget development check"
            )
        checked = [
            json.loads(line)
            for line in args.validation.read_text(encoding="utf-8").splitlines()
        ]
        header = checked[0]
        if (
            header["mode"] != "development-check"
            or header["implementation_sha256"] != digest(IMPLEMENTATION)
            or header["config_sha256"] != digest(CONFIG)
        ):
            raise ValueError(
                "development check does not match the current implementation and configuration"
            )
        rows = [r for r in checked if r["kind"] == "fit"]
        references = {
            (r["condition_id"], r["stream"]): r["centered_nll"]
            for r in selection["development_references"]
        }
        if len(rows) != len(references) or {
            (r["condition_id"], r["stream"]) for r in rows
        } != set(references):
            raise ValueError("development check is incomplete")
        for row in rows:
            if (
                row["settings"] != selection["selected_settings"]
                or row["optimization"]["numerical_validation_passed"] is not True
                or row["optimization"]["exhausted_budgets"]
                or row["centered_nll"]
                - references[(row["condition_id"], row["stream"])]
                > protocol["criteria"]["objective_gap_absolute"]
            ):
                raise ValueError(
                    "selected settings failed the development budget check"
                )
        selection["frozen_implementation_sha256"] = digest(IMPLEMENTATION)
        selection["frozen_runner_sha256"] = digest(__file__)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2, allow_nan=False)
        return
    if args.mode in ("held-out", "development-check"):
        lock = json.loads(args.source.read_text(encoding="utf-8"))
        if args.mode == "held-out" and (
            lock["config_sha256"] != digest(CONFIG)
            or lock["frozen_implementation_sha256"] != digest(IMPLEMENTATION)
            or lock["frozen_runner_sha256"] != digest(__file__)
        ):
            raise ValueError(
                "held-out evaluation requires the unchanged frozen implementation, runner and configuration"
            )
        policies = [BiexponentialFitSettings(**lock["selected_settings"])]
        streams = (
            protocol["held_out_streams"]
            if args.mode == "held-out"
            else protocol["development_streams"]
        )
    else:
        policies = [
            settings_for(protocol, p) for p in (args.policies or protocol["policies"])
        ]
        streams = protocol["development_streams"] if args.mode == "development" else [0]
    conditions = [c["id"] for c in protocol["conditions"]]
    if args.mode == "pilot":
        conditions = protocol["pilot_conditions"]
    elif args.mode == "timing":
        conditions = protocol["timing_repeat_conditions"]
    elif args.mode == "audit":
        conditions = protocol["independent_audit_conditions"]
    repetitions = protocol["timing_repetitions"] if args.mode == "timing" else 1
    with args.output.open("x", encoding="utf-8") as handle:
        write_row(
            handle,
            dict(
                kind="metadata",
                mode=args.mode,
                config_sha256=digest(CONFIG),
                implementation_sha256=digest(IMPLEMENTATION),
                runner_sha256=digest(__file__),
                python=platform.python_version(),
                platform=platform.platform(),
                processor=platform.processor(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                execution="serial",
                seed_entropy=protocol["seed_entropy"],
            ),
        )
        # Warm imports/FFT/objective, without a held-out observation or fit.
        t, k, expected, _, _, _ = build_case(protocol, 0)
        calculate_poisson_deviance(expected, expected)
        if args.mode == "noise-free":
            settings = (
                BiexponentialFitSettings(
                    **json.loads(args.source.read_text(encoding="utf-8"))[
                        "selected_settings"
                    ]
                )
                if args.source
                else settings_for(protocol, "economical")
            )
            row = noise_free_qualification(settings)
            write_row(handle, row)
            print(json.dumps(_jsonable(row), indent=2))
            return
        for condition in conditions:
            if args.mode == "audit":
                row = independent_audit(protocol, condition)
                write_row(handle, row)
                print(json.dumps(_jsonable(row)), flush=True)
                continue
            for stream in streams:
                for repeat in range(repetitions):
                    order = (
                        policies
                        if (condition + stream + repeat) % 2 == 0
                        else list(reversed(policies))
                    )
                    for settings in order:
                        row = fit_row(
                            protocol, condition, stream, settings, args.mode, repeat
                        )
                        write_row(handle, row)
                        opt = row["optimization"]
                        print(
                            f"{args.mode} case={condition} stream={stream} policy={settings.policy} nll={row['centered_nll']} accepted={opt['numerical_validation_passed']} calls={opt['objective_evaluations']} seconds={opt['total_seconds']:.3f} exhausted={opt['exhausted_budgets']}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
