"""Candidate-specific checks and descriptive fixed-K structural diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from tcspc_toolkit.estimation._fixed_k import _Evaluation, _readonly, _scalar
from tcspc_toolkit.estimation._fixed_k_objective import _model_derivatives


@dataclass(frozen=True, eq=False)
class _ValidationProbe:
    direction: str
    vector: object
    signed_step: float
    feasible: bool
    evaluation: _Evaluation | None


@dataclass(frozen=True, eq=False)
class _LocalValidation:
    candidate: _Evaluation
    submitted_objective: float | None
    tolerance: float
    probes: tuple[_ValidationProbe, ...]
    best_probe: _Evaluation | None
    best_improvement: float
    complete: bool
    passed: bool | None


def _validation_directions(coordinates):
    p, k = coordinates.dimension, coordinates.n_components
    directions = [(name, np.eye(p)[i]) for i, name in enumerate(coordinates.names)]
    for i, j in combinations(range(k), 2):
        for offset, name in ((0, "amplitude_transfer"), (k, "lifetime_split")):
            vector = np.zeros(p)
            vector[offset + i], vector[offset + j] = 1, -1
            directions.append((f"{name}_{i + 1}_{j + 1}", vector))
    return tuple((name, _readonly(v)) for name, v in directions)


def _validate_candidate(
    problem, candidate, *, steps, absolute_tolerance, relative_tolerance
):
    """Probe a submitted point. Better probes are evidence, never replacements."""
    steps = tuple(_scalar(s, "validation step") for s in steps)
    if not steps or any(s <= 0 for s in steps):
        raise ValueError("explicit positive validation steps required")
    atol = _scalar(absolute_tolerance, "absolute_tolerance", minimum=0)
    rtol = _scalar(relative_tolerance, "relative_tolerance", minimum=0)
    if candidate.problem_id is not problem._identity or not candidate.finite:
        return _LocalValidation(
            candidate, candidate.centered_nll, atol, (), None, 0, False, None
        )
    tolerance = atol + rtol * abs(candidate.centered_nll)
    if not np.isfinite(tolerance):
        raise ValueError("unrepresentable validation tolerance")
    probes = []
    best = None
    complete = True
    for name, vector in _validation_directions(problem.coordinates):
        for step in steps:
            for sign in (-1, 1):
                x = candidate.coordinates + sign * step * vector
                feasible = problem.coordinates.feasible(x)
                result = None
                if feasible:
                    if np.array_equal(x, candidate.coordinates):
                        complete = False
                    else:
                        result = problem.evaluate(x, phase="validation", origin=name)
                        if result.finite and (
                            best is None or result.centered_nll < best.centered_nll
                        ):
                            best = result
                        if result.status not in ("finite", "infinite_objective"):
                            complete = False
                probes.append(
                    _ValidationProbe(name, vector, sign * step, feasible, result)
                )
                if not complete or problem.work.deadline_reached:
                    complete = False
                    break
            if not complete:
                break
        if not complete:
            break
    improvement = (
        max(0.0, candidate.centered_nll - best.centered_nll)
        if best is not None
        else 0.0
    )
    return _LocalValidation(
        candidate,
        candidate.centered_nll,
        tolerance,
        tuple(probes),
        best,
        improvement,
        complete,
        improvement <= tolerance if complete else None,
    )


@dataclass(frozen=True, eq=False)
class _NumericalAssessment:
    candidate: _Evaluation
    fresh: _Evaluation
    signed_objective_gap: float | None
    consistent: bool
    validation_matches: bool
    validation_passed: bool | None
    complete: bool
    reasons: tuple[str, ...]


def _assess_candidate(
    problem, candidate, *, validation=None, absolute_tolerance, relative_tolerance
):
    """Recompute the exact returned coordinates, not a validation anchor."""
    atol = _scalar(absolute_tolerance, "absolute_tolerance", minimum=0)
    rtol = _scalar(relative_tolerance, "relative_tolerance", minimum=0)
    fresh = problem.evaluate(
        candidate.coordinates, phase="assessment", origin="fresh_candidate_check"
    )
    reasons = []
    gap = None
    complete = bool(fresh.finite and not problem.work.deadline_reached)
    if candidate.problem_id is not problem._identity:
        reasons.append("different_problem")
    if not candidate.finite or not fresh.finite:
        reasons.append("nonfinite_candidate_or_recheck")
    else:
        gap = fresh.centered_nll - candidate.centered_nll
        if abs(gap) > atol + rtol * abs(candidate.centered_nll):
            reasons.append("objective_mismatch")
        if candidate.parameters != fresh.parameters:
            reasons.append("parameter_mismatch")
        if not np.array_equal(candidate.expected_counts, fresh.expected_counts):
            reasons.append("expectation_mismatch")
        if not np.array_equal(candidate.component_counts, fresh.component_counts):
            reasons.append("component_mismatch")
    matches = bool(validation is not None and validation.candidate is candidate)
    if validation is not None and not matches:
        reasons.append("validation_for_different_candidate")
    return _NumericalAssessment(
        candidate,
        fresh,
        gap,
        complete and not reasons,
        matches,
        validation.passed if matches else None,
        complete,
        tuple(reasons),
    )


@dataclass(frozen=True)
class _ComponentStructure:
    n_components: int
    nominal_parameters: int
    zero_amplitude_components: tuple[int, ...]
    equal_lifetime_pairs: tuple[tuple[int, int], ...]
    lifetime_ratios: tuple[tuple[int, int, float], ...]
    amplitude_fractions: tuple[float, ...] | None
    infinite_window_area_fractions: tuple[float, ...] | None
    finite_window_signal_fractions: tuple[float, ...] | None
    background_zero: bool


def _fractions(values):
    values = np.asarray(values, dtype=float)
    maximum = np.max(values)
    if maximum == 0 or not np.isfinite(maximum):
        return None
    scaled = values / maximum
    return tuple(map(float, scaled / scaled.sum()))


def _component_structure(candidate):
    if not candidate.finite:
        raise ValueError("finite candidate required")
    p = candidate.parameters
    a, tau = np.asarray(p.amplitudes), np.asarray(p.lifetimes_ns)
    pairs = tuple(combinations(range(p.n_components), 2))
    # Log weights avoid overflow of A*tau. Zero weights remain exactly zero.
    positive = a > 0
    areas = np.zeros_like(a)
    if np.any(positive):
        logs = np.log(a[positive]) + np.log(tau[positive])
        areas[positive] = np.exp(logs - logs.max())
    # Scaling before summing also avoids overflow in finite-window totals.
    maximum = np.max(candidate.component_counts)
    totals = (
        candidate.component_counts.sum(axis=1)
        if maximum == 0
        else (candidate.component_counts / maximum).sum(axis=1)
    )
    with np.errstate(over="ignore"):
        ratios = tuple(
            (i, j, float(max(tau[i], tau[j]) / min(tau[i], tau[j]))) for i, j in pairs
        )
    return _ComponentStructure(
        p.n_components,
        2 * p.n_components + 2,
        tuple(map(int, np.flatnonzero(a == 0))),
        tuple((i, j) for i, j in pairs if tau[i] == tau[j]),
        ratios,
        _fractions(a),
        _fractions(areas),
        _fractions(totals),
        p.background == 0,
    )


@dataclass(frozen=True, eq=False)
class _ConditioningDiagnostic:
    status: str
    n_bins: int
    n_parameters: int
    rank: int | None
    rank_tolerance: float | None
    singular_values: tuple[float, ...] | None
    condition_number: float | None
    coordinate_names: tuple[str, ...]
    derivative: object
    active_bounds: tuple[str, ...]


def _conditioning_diagnostic(problem, candidate, policy):
    """Local Poisson-weighted sensitivity, not physical identifiability or a CI."""
    derivative = _model_derivatives(problem, candidate, policy, phase="diagnostics")
    n, p = problem.inputs.time_ns.size, problem.coordinates.dimension
    status = "available"
    rank = tolerance = singular = condition = None
    if derivative.jacobian is None:
        status = "derivative_unavailable"
    elif np.any(candidate.expected_counts <= 0):
        status = "nonregular_zero_expectation"
    elif problem.work.deadline_reached:
        status = "wall_time_limit"
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            weight = 1 / np.sqrt(candidate.expected_counts)
            jacobian = derivative.jacobian * weight[:, None]
            error = derivative.column_errors * weight[:, None]
        if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(error)):
            status = "nonfinite_weighted_jacobian"
        else:
            try:
                singular_array = np.linalg.svd(jacobian, compute_uv=False)
                tolerance = float(
                    max(
                        max(n, p) * np.finfo(float).eps * singular_array[0],
                        np.linalg.norm(error),
                    )
                )
                if not np.isfinite(tolerance):
                    status = "unrepresentable_rank_tolerance"
                else:
                    rank = int(np.sum(singular_array > tolerance))
                    singular = tuple(map(float, singular_array))
                    condition = (
                        float(singular_array[0] / singular_array[-1])
                        if rank == p
                        else np.inf
                    )
            except np.linalg.LinAlgError:
                status = "svd_failure"
    if problem.work.deadline_reached and status == "available":
        status = "wall_time_limit"
    return _ConditioningDiagnostic(
        status,
        n,
        p,
        rank,
        tolerance,
        singular,
        condition,
        problem.coordinates.names,
        derivative,
        tuple(
            name
            for name, value, lo, hi in zip(
                problem.coordinates.names,
                candidate.coordinates,
                problem.coordinates.lower,
                problem.coordinates.upper,
                strict=True,
            )
            if value == lo or value == hi
        ),
    )
