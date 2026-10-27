"""Independent Issue-28 qualification machinery; no policy selection or tuning.

Observation streams 0/1 require an explicitly authorized manifest. The current
pre-review manifest permits only stream 2. Generation uses the PR-1 model;
the audit deliberately uses a different solver, loss and direct convolution.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import kl_div
from scipy.stats import norm

from tcspc_toolkit.estimation.multiexponential import BiexponentialFitSettings
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance
from tcspc_toolkit.irf.estimation import estimate_irf_from_leading_edge
from tcspc_toolkit.irf.preparation import (
    PreparedIRF,
    irf_profile_from_sampled_irf,
    prepare_irf,
)
from tcspc_toolkit.irf.profiles import (
    generate_emg_irf_profile,
    generate_gaussian_irf_profile,
)
from tcspc_toolkit.measurement.records import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
)
from tcspc_toolkit.physics.forward_model import (
    multiexponential_reconvolution_component_counts,
)
from tcspc_toolkit.simulation.histograms import sample_photon_counts


def matrix_totals(protocol):
    """Derive work from arms, not from hand-maintained fit-count constants."""
    conditions = protocol["conditions"]
    observations = sum(c["replicates"] for c in conditions)
    comparisons = sum(c["replicates"] * len(c["arms"]) for c in conditions)
    audits = sum(
        len(c["arms"])
        * (
            c["replicates"]
            if c["id"] == 0
            else len(protocol["audit"]["secondary_replicates"])
        )
        for c in conditions
    )
    return dict(
        conditions=len(conditions),
        observations=observations,
        comparisons=comparisons,
        production_fits=2 * comparisons,
        scheduled_audits=audits,
    )


def validate_protocol(protocol):
    """Reject an incomplete matrix or drift from the merged production policy."""
    if matrix_totals(protocol) != protocol["expected_totals"]:
        raise ValueError("condition/arm totals differ from declared totals")
    if [c["id"] for c in protocol["conditions"]] != list(range(30)):
        raise ValueError("the protocol requires the ordered 30 conditions")
    if (
        BiexponentialFitSettings(**protocol["fit_settings"])
        != BiexponentialFitSettings()
    ):
        raise ValueError("qualification must preserve the frozen PR-2 settings")
    for c in protocol["conditions"]:
        if len(c["arms"]) != len(set(c["arms"])) or not c["arms"]:
            raise ValueError("assumed-IRF arms must be unique and nonempty")
        if c["replicates"] != (100 if c["id"] == 0 else 50):
            raise ValueError("replicate counts are frozen")
    if protocol["pilot"]["stream"] != 2:
        raise ValueError("pilot must use the non-qualification stream")
    return matrix_totals(protocol)


def observation_rng(protocol, stream, condition_id, replicate):
    """Explicit PCG64 streams; no qualification draws before review approval."""
    if stream not in (0, 1, 2):
        raise ValueError("unknown observation stream")
    if stream in (0, 1) and protocol["execution_status"] != "authorized":
        raise PermissionError(
            "scientific observations require prequalification approval"
        )
    entropy = [*protocol["root_entropy"], stream, condition_id, replicate]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def condition_configuration(protocol, condition_id):
    item = next(c for c in protocol["conditions"] if c["id"] == condition_id)
    return {**protocol["baseline"], **item}


def condition_grid(protocol, condition):
    g = protocol["grid"]
    return np.linspace(g["start_ns"], g["stop_ns"], g["bins"], endpoint=g["endpoint"])[
        : condition.get("first_bins", g["bins"])
    ]


def build_assumed_irf(protocol, time, name):
    """Explicit synthetic provenance, including the synthetic sampled arm."""
    irf = protocol["irf"]
    provenance = {"origin": "issue28_pr3_synthetic", "assumption": name}
    if name.startswith("gaussian_"):
        source = generate_gaussian_irf_profile(
            time,
            gaussian_centre_ns=irf["center_ns"],
            gaussian_fwhm_ns=float(name.removeprefix("gaussian_")),
            provenance=provenance,
        )
        return prepare_irf(source, time)
    if name not in ("emg", "sampled_emg"):
        raise ValueError(f"unknown IRF source: {name}")
    source_time = time if name == "emg" else np.linspace(0, 20, irf["sampled_bins"])
    source = generate_emg_irf_profile(
        source_time,
        gaussian_centre_ns=irf["center_ns"],
        gaussian_fwhm_ns=irf["emg_fwhm_ns"],
        tail_time_ns=irf["emg_tail_ns"],
        provenance=provenance,
    )
    if name == "sampled_emg":
        source = irf_profile_from_sampled_irf(
            SampledIRF(
                source.time_ns,
                source.values,
                metadata={
                    "synthetic_source_parameters": dict(source.source_parameters)
                },
                provenance={**provenance, "experimentally_measured": False},
            )
        )
    return prepare_irf(source, time, resampling="none" if name == "emg" else "linear")


@dataclass(frozen=True)
class GeneratingCase:
    configuration: dict
    irf: PreparedIRF
    amplitudes: tuple[float, ...]
    component_counts: np.ndarray
    expected_counts: np.ndarray


def build_generating_case(protocol, condition_id, *, overrides=None):
    """Deterministic expectations only; does not draw qualification observations."""
    c = condition_configuration(protocol, condition_id)
    if overrides is not None:
        c.update(overrides)
    time = condition_grid(protocol, c)
    irf = build_assumed_irf(protocol, time, c["generating_irf"])
    unit = multiexponential_reconvolution_component_counts(
        time,
        irf.kernel,
        amplitudes=np.ones(len(c["lifetimes_ns"])),
        lifetimes=c["lifetimes_ns"],
        temporal_shift=c["shift_ns"],
    )
    amplitudes = (
        c["signal_photons"] * np.asarray(c["detected_fractions"]) / unit.sum(axis=1)
    )
    components = multiexponential_reconvolution_component_counts(
        time,
        irf.kernel,
        amplitudes=amplitudes,
        lifetimes=c["lifetimes_ns"],
        temporal_shift=c["shift_ns"],
    )
    expected = components.sum(axis=0) + c["background"]
    components.setflags(write=False)
    expected.setflags(write=False)
    return GeneratingCase(c, irf, tuple(map(float, amplitudes)), components, expected)


def sample_case(protocol, case, *, stream, replicate):
    c = case.configuration
    rng = observation_rng(protocol, stream, c["id"], replicate)
    return TCSPCMeasurement(
        case.irf.time_ns,
        sample_photon_counts(case.expected_counts, rng),
        MeasurementDataKind.RAW_COUNTS,
        sample_id=f"issue28-pr3-s{stream}-c{c['id']:02d}-r{replicate:03d}",
        provenance={
            "origin": "independent_synthetic",
            "stream": stream,
            "seed_entropy": [*protocol["root_entropy"], stream, c["id"], replicate],
        },
    )


def build_calibration_proxy(protocol, *, stream):
    """One independent calibration, fixed across the 50 C28 observations.

    Stream 2 uses a distinct replicate (- represented by 1000000, not a negative
    seed) to avoid sharing the C28 pilot observation. Scientific calibration is
    stream 1, replicate 0. Construction failures are returned without redraw.
    """
    if stream not in (1, 2):
        raise ValueError("proxy calibration requires stream 1 or 2")
    case = build_generating_case(protocol, 28, overrides=protocol["proxy"])
    measurement = sample_case(
        protocol, case, stream=stream, replicate=0 if stream == 1 else 1000000
    )
    p = protocol["proxy"]
    result = estimate_irf_from_leading_edge(
        measurement,
        background_window_ns=tuple(p["background_window_ns"]),
        rising_edge_window_ns=tuple(p["rising_window_ns"]),
        smoothing_window_bins=p["smoothing_window"],
        polynomial_order=p["polynomial_order"],
    )
    prepared = (
        None
        if result.profile is None
        else prepare_irf(result.profile, case.irf.time_ns)
    )
    return prepared, result.diagnostics, measurement


@dataclass(frozen=True)
class NumericalAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    recomputed_centered_nll: float | None
    objective_difference: float | None
    external_forward_evaluations: int
    diagnostic_budget_complete: bool
    jacobian_available: bool
    diagnostic_unavailable_reason: str | None


def assess_biexponential_fit(result, time, counts, kernel, rules):
    """Qualification acceptance independent of warnings and optimizer status.

    Exactly one extra forward call rechecks physical feasibility, returned curves
    and objective. It is recorded separately from all PR-2 work counters.
    Legitimately unavailable Jacobians do not become numerical fit failures.
    """
    o = result.optimization
    reasons = []
    value = difference = None
    calls = 0
    p = result.parameters
    if p is None or result.expected_counts is None or result.statistics is None:
        reasons.append("parameters_or_expectation_unavailable")
    else:
        lo, hi = result.assumptions.lifetime_bounds_ns
        slo, shi = result.assumptions.temporal_shift_bounds_ns
        feasible = (
            np.all(
                np.isfinite(
                    [*p.amplitudes, *p.lifetimes_ns, p.background, p.temporal_shift_ns]
                )
            )
            and np.all(np.asarray(p.amplitudes) >= 0)
            and p.background >= 0
            and np.all(
                (np.asarray(p.lifetimes_ns) >= lo) & (np.asarray(p.lifetimes_ns) <= hi)
            )
            and slo <= p.temporal_shift_ns <= shi
        )
        if not feasible:
            reasons.append("infeasible_parameters")
        else:
            calls = 1
            components = multiexponential_reconvolution_component_counts(
                time,
                kernel,
                amplitudes=p.amplitudes,
                lifetimes=p.lifetimes_ns,
                temporal_shift=p.temporal_shift_ns,
            )
            expected = components.sum(axis=0) + p.background
            value = 0.5 * calculate_poisson_deviance(counts, expected)
            difference = abs(value - result.statistics.centered_negative_log_likelihood)
            if not np.isfinite(value) or not np.isclose(
                value,
                result.statistics.centered_negative_log_likelihood,
                atol=rules["objective_recheck_atol"],
                rtol=rules["objective_recheck_rtol"],
            ):
                reasons.append("objective_recheck_failed")
            if not np.all(np.isfinite(result.expected_counts)) or not np.allclose(
                expected, result.expected_counts, rtol=1e-12, atol=1e-12
            ):
                reasons.append("curve_recheck_failed")
    if o.numerical_validation_passed is not True:
        reasons.append("local_validation_failed_or_unavailable")
    if set(o.exhausted_budgets).intersection(rules["required_budget_stages"]):
        reasons.append("required_stage_budget_exhausted")
    # Analytical all-zero solutions complete required stages without optimization.
    if o.stopping_reason not in ("candidate_bank_completed", "analytical_zero_counts"):
        reasons.append("required_fitting_stages_incomplete")
    i = result.identifiability
    return NumericalAssessment(
        not reasons,
        tuple(reasons),
        value,
        difference,
        calls,
        p is not None and "diagnostics" not in o.exhausted_budgets,
        i.weighted_jacobian_rank is not None,
        i.jacobian_unavailable_reason,
    )


def component_recovery(case, result, accepted):
    """Canonical component errors, with explicit per-component availability.

    No warning-based filtering. Exact zero fitted amplitude/contribution removes
    that component only from its error denominator, never from fit acceptance.
    Generating equal lifetimes, mono and absent signal have no component truth.
    """
    c = case.configuration
    truth = np.asarray(c["lifetimes_ns"])
    identifiable_truth = (
        truth.size == 2
        and truth[0] < truth[1]
        and c["signal_photons"] > 0
        and np.all(np.asarray(c["detected_fractions"]) > 0)
    )
    finite_errors = [None, None]
    available = [False, False]
    p = result.parameters
    if identifiable_truth and p is not None and result.component_counts is not None:
        lo, hi = result.assumptions.lifetime_bounds_ns
        for k in range(2):
            available[k] = bool(
                p.amplitudes[k] > 0
                and np.sum(result.component_counts[k]) > 0
                and np.isfinite(p.lifetimes_ns[k])
                and lo <= p.lifetimes_ns[k] <= hi
            )
            if available[k]:
                finite_errors[k] = float((p.lifetimes_ns[k] - truth[k]) / truth[k])
    return dict(
        truth_components_available=bool(identifiable_truth),
        fitted_component_available=available,
        accepted_component_relative_errors=finite_errors if accepted else [None, None],
        all_finite_component_relative_errors=finite_errors,
    )


class _AuditLimit(Exception):
    pass


def independent_biexponential_audit(
    time, counts, kernel, lifetime_bounds, shift_bounds, settings
):
    """Eight TRF least-squares starts; direct convolution, physical tau coords.

    Every direct prediction (including numerical derivatives and endpoints)
    consumes the common hard call limit. The best evaluated point survives
    exhaustion; incomplete audits are never called passed.
    """
    time, y, kernel = map(lambda a: np.asarray(a, dtype=float), (time, counts, kernel))
    scale = max(float(np.mean(y)), 1.0)
    lower = [0, 0, lifetime_bounds[0], lifetime_bounds[0], 0, shift_bounds[0]]
    upper = [
        np.inf,
        np.inf,
        lifetime_bounds[1],
        lifetime_bounds[1],
        np.inf,
        shift_bounds[1],
    ]
    grid = np.geomspace(*lifetime_bounds, settings["lifetime_grid_points"])
    calls = 0
    best_value = np.inf
    best_x = best_mu = None
    records = []
    exhausted = False
    started = perf_counter()

    def direct(x):
        nonlocal calls, best_x, best_mu, best_value
        if calls >= settings["max_forward_calls"]:
            raise _AuditLimit
        calls += 1
        shifted = np.interp(time - x[5], time, kernel, left=0, right=0)
        unit = [
            np.convolve(np.exp(-time / tau), shifted)[: time.size] * (time[1] - time[0])
            for tau in x[2:4]
        ]
        mu = scale * (x[0] * unit[0] + x[1] * unit[1] + x[4])
        terms = np.maximum(kl_div(y, mu), 0)
        value = float(np.sum(terms))
        if np.isfinite(value) and value < best_value:
            best_value, best_x, best_mu = value, np.array(x, copy=True), mu.copy()
        return mu, terms

    def residuals(x):
        mu, terms = direct(x)
        return np.sign(y - mu) * np.sqrt(2 * terms)

    for i, j in settings["lifetime_index_pairs"]:
        for shift in settings["initial_shifts_ns"]:
            start = [1, 1, grid[i], grid[j], 0.01, np.clip(shift, *shift_bounds)]
            before = calls
            try:
                solved = least_squares(
                    residuals,
                    start,
                    bounds=(lower, upper),
                    method="trf",
                    x_scale="jac",
                    ftol=settings["ftol"],
                    xtol=settings["xtol"],
                    gtol=settings["gtol"],
                    max_nfev=settings["max_nfev_per_start"],
                )
                mu, _ = direct(solved.x)
                records.append(
                    dict(
                        success=bool(solved.success),
                        status=int(solved.status),
                        nfev=int(solved.nfev),
                        forward_calls=calls - before,
                        centered_nll=0.5 * calculate_poisson_deviance(y, mu),
                    )
                )
            except _AuditLimit:
                exhausted = True
                records.append(
                    dict(
                        success=False,
                        status="forward_budget_exhausted",
                        forward_calls=calls - before,
                    )
                )
                break
        if exhausted:
            break
    parameters = None
    if best_x is not None:
        order = np.lexsort((-best_x[:2], best_x[2:4]))
        parameters = dict(
            amplitudes=(scale * best_x[:2][order]).tolist(),
            lifetimes_ns=best_x[2:4][order].tolist(),
            background=float(scale * best_x[4]),
            temporal_shift_ns=float(best_x[5]),
        )
    return dict(
        parameters=parameters,
        centered_nll=None
        if best_mu is None
        else 0.5 * calculate_poisson_deviance(y, best_mu),
        forward_calls=calls,
        elapsed_seconds=perf_counter() - started,
        budget_exhausted=exhausted,
        exhausted_budgets=(
            (["forward_calls"] if exhausted else [])
            + (["start_nfev"] if any(r.get("status") == 0 for r in records) else [])
        ),
        complete=not exhausted
        and len(records) == 8
        and all(r["success"] for r in records),
        starts=records,
    )


def summarize_recovery(rows, protocol, condition_id, arm_index):
    """All-attempt acceptance and accepted-component metrics with explicit n.

    Bootstrap whole observation rows, retaining missingness. Intervals describe
    Monte Carlo uncertainty of this study, not per-observation uncertainty.
    """
    n = len(rows)
    if not n:
        raise ValueError("cannot summarize an empty population")
    rules = protocol["acceptance"]
    accepted = np.array([r["assessment"]["accepted"] for r in rows])
    proportion = float(np.mean(accepted))
    z = float(norm.ppf(0.5 + rules["confidence_level"] / 2))
    center = (proportion + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z
        / (1 + z * z / n)
        * np.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
    )
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [*protocol["root_entropy"], 3, condition_id, arm_index]
            )
        )
    )
    indices = rng.integers(0, n, (rules["bootstrap_replicates"], n))
    components = []
    joint = accepted.copy()
    for k in range(2):
        errors = np.array(
            [r["recovery"]["accepted_component_relative_errors"][k] for r in rows],
            dtype=float,
        )
        valid = np.isfinite(errors)
        values = errors[valid]
        median = float(np.median(abs(values))) if values.size else None
        bias = float(np.mean(values)) if values.size else None
        bootstrap = []
        for selected in indices:
            v = errors[selected]
            v = v[np.isfinite(v)]
            if v.size:
                bootstrap.append((np.median(abs(v)), np.mean(v)))
        interval = (
            None
            if not bootstrap
            else np.quantile(bootstrap, [0.025, 0.975], axis=0).T.tolist()
        )
        passed = bool(
            median is not None
            and median
            < rules["maximum_component_median_absolute_relative_error_exclusive"]
            and abs(bias) < rules["maximum_component_absolute_relative_bias_exclusive"]
        )
        components.append(
            dict(
                n_contributing=int(values.size),
                n_unavailable_among_accepted=int(np.sum(accepted & ~valid)),
                median_absolute_relative_error=median,
                signed_relative_bias=bias,
                bootstrap_intervals=interval,
                n_defined_bootstraps=len(bootstrap),
                targets_met=passed,
            )
        )
        all_finite = np.array(
            [
                r["recovery"].get("all_finite_component_relative_errors", [None, None])[
                    k
                ]
                for r in rows
            ],
            dtype=float,
        )
        for label, mask in (
            ("all_finite", np.isfinite(all_finite)),
            ("rejected_finite", np.isfinite(all_finite) & ~accepted),
        ):
            sensitivity = all_finite[mask]
            components[-1][label] = dict(
                n=int(sensitivity.size),
                median_absolute_relative_error=None
                if not sensitivity.size
                else float(np.median(abs(sensitivity))),
                signed_relative_bias=None
                if not sensitivity.size
                else float(np.mean(sensitivity)),
            )
        joint &= valid & (abs(errors) < 0.05)
    return dict(
        n_attempted=n,
        n_accepted=int(accepted.sum()),
        numerical_acceptance=proportion,
        wilson_interval=[center - half, center + half],
        components=components,
        joint_accepted_both_within_5_percent_fraction=float(joint.mean()),
        diagnostic_budget_complete_fraction=float(
            np.mean([r["assessment"]["diagnostic_budget_complete"] for r in rows])
        ),
        baseline_targets_met=bool(
            condition_id == 0
            and n == protocol["conditions"][0]["replicates"]
            and proportion >= rules["minimum_baseline_numerical_fraction"]
            and all(c["targets_met"] for c in components)
        ),
    )
