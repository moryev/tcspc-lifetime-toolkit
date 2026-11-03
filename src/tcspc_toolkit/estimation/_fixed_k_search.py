"""Bounded K=3 seed-first search and explicit candidate assessment."""

from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from time import monotonic
from types import MappingProxyType

import numpy as np

from tcspc_toolkit.estimation import triexponential as api
from tcspc_toolkit.estimation._fixed_k import (
    _CoordinateMap,
    _FixedKInputs,
    _FixedKParameters,
    _canonicalize_components,
    _readonly,
)
from tcspc_toolkit.estimation._fixed_k_objective import (
    _FixedKProblem,
    _WorkLimits,
    _shift_geometry,
)
from tcspc_toolkit.estimation._fixed_k_initialization import _build_seed_bank, _SeedBank
from tcspc_toolkit.estimation._fixed_k_solvers import _run_lbfgsb, _run_trf, _lease
from tcspc_toolkit.estimation._poisson_residuals import _poisson_residuals
from tcspc_toolkit.estimation._fixed_k_diagnostics import (
    _validate_candidate,
    _ValidationProbe,
    _assess_candidate,
    _component_structure,
    _conditioning_diagnostic,
)
from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_fit_statistics
from tcspc_toolkit.irf.preparation import PreparedIRF


def _public(parameters):
    if parameters is None:
        return None
    return api.TriexponentialParameters(
        parameters.amplitudes,
        parameters.lifetimes_ns,
        parameters.background,
        parameters.temporal_shift_ns,
    )


def _neighbor_shifts(time, shift, bounds):
    """At most three nearby knots and sided interior points, using actual time."""
    q = time - shift
    left = np.searchsorted(time, q, side="left") - 1
    right = np.searchsorted(time, q, side="right")
    knots = []
    mask = left >= 0
    if np.any(mask):
        knots.append(shift + float(np.min(q[mask] - time[left[mask]])))
    mask = right < time.size
    if np.any(mask):
        knots.append(shift - float(np.min(time[right[mask]] - q[mask])))
    if np.any(
        np.searchsorted(time, q, side="left") != np.searchsorted(time, q, side="right")
    ):
        knots.append(shift)
    result = []
    unavailable = []
    for knot in knots:
        if bounds[0] <= knot <= bounds[1]:
            result.append(knot)
            distance = _shift_geometry(
                time, np.zeros_like(time), knot
            ).distance_to_next_knot_ns
            for sign in (-1, 1):
                room = knot - bounds[0] if sign < 0 else bounds[1] - knot
                h = min(distance / 4, room / 2)
                value = knot + sign * h
                if h > 0 and value != knot and np.isfinite(value):
                    result.append(value)
                elif room > 0:
                    unavailable.append(sign)
    return tuple(dict.fromkeys(result)), tuple(dict.fromkeys(unavailable))


def _mathematically_feasible(coordinates, vector, step, lower, upper):
    """Check declared float inputs exactly, without overflow in probe arithmetic."""
    for x, v, lo, hi in zip(coordinates, vector, lower, upper, strict=True):
        value = Fraction(float(x)) + Fraction(float(step)) * Fraction(float(v))
        if (np.isfinite(lo) and value < Fraction(float(lo))) or (
            np.isfinite(hi) and value > Fraction(float(hi))
        ):
            return False
    return True


def _probe_status(probe, candidate):
    if probe.evaluation is not None:
        if probe.feasible and probe.evaluation.parameters == candidate.parameters:
            return "unrepresentable_probe"
        return probe.evaluation.status
    if not probe.feasible:
        return "infeasible"
    if probe.direction.startswith("shift_interval_unrepresentable"):
        return "unrepresentable_shift_probe"
    return "unrepresentable_probe"


def _validate_full(problem, candidate, settings):
    with np.errstate(over="ignore", invalid="ignore"):
        check = _validate_candidate(
            problem,
            candidate,
            steps=settings.validation_steps,
            absolute_tolerance=settings.descent_tolerance,
            relative_tolerance=0,
        )
    # A failed numeric construction is not proof that the direction violates
    # physical bounds. Also reject coordinate steps lost on physical decoding.
    probes = []
    complete = check.complete
    for probe in check.probes:
        if not probe.feasible and _mathematically_feasible(
            candidate.coordinates,
            probe.vector,
            probe.signed_step,
            problem.coordinates.lower,
            problem.coordinates.upper,
        ):
            probe = replace(probe, feasible=True)
        if _probe_status(probe, candidate) == "unrepresentable_probe":
            complete = False
        probes.append(probe)
    check = replace(
        check,
        probes=tuple(probes),
        complete=complete,
        passed=check.passed if complete else None,
    )
    if not check.complete:
        return check
    p = problem.coordinates
    extra = []
    for size in range(1, 5):
        for indices in combinations((0, 1, 2, 6), size):
            x = candidate.coordinates.copy()
            x[list(indices)] = 0
            extra.append(("zero_face:" + ",".join(map(str, indices)), x))
    for i in (3, 4, 5, 7):
        for value in (p.lower[i], p.upper[i]):
            x = candidate.coordinates.copy()
            x[i] = value
            extra.append((f"bound:{p.names[i]}", x))
    shifts, unavailable = _neighbor_shifts(
        problem.inputs.time_ns,
        candidate.parameters.temporal_shift_ns,
        p.temporal_shift_bounds_ns,
    )
    for shift in shifts:
        x = candidate.coordinates.copy()
        x[7] = (shift - p.shift_mid) / p.shift_scale
        extra.append(("shift_interval", x))
    probes = list(check.probes)
    best = check.best_probe
    complete = not unavailable
    for sign in unavailable:
        probes.append(
            _ValidationProbe(
                f"shift_interval_unrepresentable:{sign}",
                _readonly(np.eye(8)[7]),
                float(sign) * 0.0,
                True,
                None,
            )
        )
    seen = {tuple(candidate.coordinates)}
    for name, x in extra:
        if tuple(x) in seen:
            continue
        seen.add(tuple(x))
        feasible = p.feasible(x)
        result = (
            problem.evaluate(x, phase="validation", origin=name) if feasible else None
        )
        probe = _ValidationProbe(
            name, _readonly(x - candidate.coordinates), 1.0, feasible, result
        )
        probes.append(probe)
        if _probe_status(probe, candidate) == "unrepresentable_probe":
            complete = False
        if result is not None:
            if result.finite and (
                best is None or result.centered_nll < best.centered_nll
            ):
                best = result
            if (
                result.status not in ("finite", "infinite_objective")
                or problem.work.deadline_reached
            ):
                complete = False
                break
    improvement = (
        max(0.0, candidate.centered_nll - best.centered_nll)
        if best is not None
        else 0.0
    )
    return replace(
        check,
        probes=tuple(probes),
        best_probe=best,
        best_improvement=improvement,
        complete=complete,
        passed=improvement <= check.tolerance if complete else None,
    )


def _canonical_candidate(problem, candidate):
    p, order = _canonicalize_components(candidate.parameters)
    if np.array_equal(order, np.arange(3)):
        return candidate
    x = candidate.coordinates.copy()
    x[:3] = x[:3][order]
    x[3:6] = x[3:6][order]
    checked = problem.evaluate(x, phase="assessment", origin="canonical_candidate")
    if checked.finite:
        return checked
    # Retain the finite evidence on exhaustion. Fresh assessment must still pass.
    return replace(
        candidate,
        coordinates=_readonly(x),
        parameters=p,
        component_counts=_readonly(candidate.component_counts[order]),
        origin="canonical_unchecked",
    )


def _validation_record(check):
    if check is None:
        return None
    return api.TriexponentialLocalValidation(
        _public(check.candidate.parameters),
        check.submitted_objective,
        check.tolerance,
        tuple(
            api.TriexponentialValidationProbe(
                probe.direction,
                tuple(map(float, probe.vector)),
                probe.signed_step,
                probe.feasible,
                _public(probe.evaluation.parameters) if probe.evaluation else None,
                probe.evaluation.centered_nll if probe.evaluation else None,
                _probe_status(probe, check.candidate),
            )
            for probe in check.probes
        ),
        _public(check.best_probe.parameters) if check.best_probe else None,
        check.best_improvement,
        check.complete,
        check.passed,
    )


def _fit_triexponential(
    time_ns,
    targets,
    irf,
    *,
    lifetime_bounds_ns,
    temporal_shift_bounds_ns,
    initial_guesses=None,
    settings=None,
    target_kind="raw_counts",
    clock=monotonic,
):
    started = clock()
    if not isinstance(settings, api.TriexponentialFitSettings):
        raise TypeError(
            "explicit TriexponentialFitSettings required; no production default selected"
        )
    inputs = _FixedKInputs(time_ns, targets, irf, target_kind)
    coordinates = _CoordinateMap(
        3,
        lifetime_bounds_ns,
        temporal_shift_bounds_ns,
        max(float(inputs.targets.sum()), 1),
        max(float(inputs.targets.mean()), 1),
    )
    guesses = []
    for guess in () if initial_guesses is None else initial_guesses:
        if not isinstance(guess, api.TriexponentialParameters):
            raise TypeError("initial guesses must be TriexponentialParameters")
        p = _FixedKParameters(
            guess.amplitudes,
            guess.lifetimes_ns,
            guess.background,
            guess.temporal_shift_ns,
        )
        coordinates.encode(p)
        p, _ = _canonicalize_components(p)
        if p not in guesses:
            guesses.append(p)
    if len(guesses) > settings.max_seed_starts - 10:
        raise ValueError("supplied guesses must leave ten protected seed slots")
    phases = dict(
        initialization=settings.max_initialization_calls,
        optimization=settings.max_optimization_calls,
        validation=settings.max_validation_calls,
        assessment=settings.max_assessment_calls,
        diagnostics=settings.max_diagnostic_calls,
    )
    total = sum(phases.values())
    problem = _FixedKProblem(
        inputs,
        coordinates,
        _WorkLimits(phases, total, total, 3 * total, settings.wall_seconds),
        clock=clock,
    )
    problem.work.started = started
    best_observed = None

    def retain(evaluation):
        nonlocal best_observed
        if evaluation.finite and (
            best_observed is None
            or evaluation.centered_nll < best_observed.centered_nll
        ):
            best_observed = evaluation

    problem.evaluation_observer = retain
    seconds = dict.fromkeys(phases, 0.0)

    def canonical(candidate):
        mark = clock()
        result = _canonical_candidate(problem, candidate)
        seconds["assessment"] += clock() - mark
        return result

    mark = clock()
    analytic = not np.any(inputs.targets)
    if analytic:
        bank = _SeedBank()
        p = _FixedKParameters(
            (0, 0, 0),
            (float(np.exp(coordinates.log_mid)),) * 3,
            0,
            float(np.clip(0, *coordinates.temporal_shift_bounds_ns)),
        )
        problem.evaluate(
            coordinates.encode(p),
            phase="initialization",
            origin="analytical_zero_counts",
        )
    else:
        with _lease(
            problem,
            "initialization",
            settings.max_initialization_calls,
            0.05 * settings.wall_seconds,
        ):
            bank = _build_seed_bank(problem, settings, guesses)
    seconds["initialization"] = clock() - mark
    starts = []
    attempted = []
    promotions = []
    recovery_used = recovery_calls = 0
    recovery_reason = None
    seed_calls = settings.max_optimization_calls - settings.recovery_calls
    n = len(bank.selected)
    for i, (name, seed) in enumerate(bank.selected):
        if problem.work.deadline_reached:
            break
        allowance = seed_calls // n + int(i < seed_calls % n)
        if allowance == 0:
            break
        runner = _run_trf if settings.solver == "trf" else _run_lbfgsb
        invocation = runner(
            problem,
            seed,
            settings,
            kind="seed",
            seed_id=name,
            calls=allowance,
            seconds=0.65 * settings.wall_seconds / n,
            invocation_id=len(starts),
        )
        starts.append(invocation)
        attempted.append(name)
        seconds["optimization"] += invocation.seconds
    candidate = canonical(best_observed) if best_observed is not None else None
    check = assessment = conditioning = diagnostic_anchor = None
    diagnostic_best = None
    diagnostic_improvement = None
    objective_available = residual_available = None
    if candidate is not None:
        while True:
            mark = clock()
            check = _validate_full(problem, candidate, settings)
            seconds["validation"] += clock() - mark
            if check.passed is True:
                break
            if (
                check.best_probe is not None
                and check.best_probe.centered_nll < candidate.centered_nll
            ):
                promotions.append("validation_probe")
                candidate = canonical(check.best_probe)
            if not check.complete:
                recovery_reason = "validation_incomplete"
                break
            if recovery_used >= settings.reserved_recovery_starts:
                recovery_reason = "required_recovery_start_limit"
                break
            if problem.work.deadline_reached:
                recovery_reason = "wall_time_limit"
                break
            allowance = settings.recovery_calls // settings.reserved_recovery_starts
            allowance += int(
                recovery_used
                < settings.recovery_calls % settings.reserved_recovery_starts
            )
            if allowance == 0:
                recovery_reason = "required_recovery_call_limit"
                break
            recovery_solver = settings.recovery_solver or settings.solver
            runner = _run_trf if recovery_solver == "trf" else _run_lbfgsb
            invocation = runner(
                problem,
                candidate,
                settings,
                kind="required_recovery",
                seed_id=None,
                calls=allowance,
                seconds=0.2 * settings.wall_seconds / settings.reserved_recovery_starts,
                fix_zero_face=check.best_probe is not None
                and check.best_probe.origin.startswith("zero_face"),
                invocation_id=len(starts),
            )
            starts.append(invocation)
            recovery_used += 1
            recovery_calls += invocation.calls
            seconds["optimization"] += invocation.seconds
            if best_observed.centered_nll < candidate.centered_nll:
                promotions.append("recovery_best")
                candidate = canonical(best_observed)
        mark = clock()
        assessment = _assess_candidate(
            problem,
            candidate,
            validation=check,
            absolute_tolerance=1e-9,
            relative_tolerance=1e-12,
        )
        seconds["assessment"] += clock() - mark
        # Freeze selection after assessment. Diagnostic probes are retained as
        # separate evidence and can reject acceptance, never silently replace it.
        mark = clock()
        diagnostic_anchor = candidate

        def diagnostic_observer(evaluation):
            nonlocal diagnostic_best
            retain(evaluation)
            if evaluation.finite and (
                diagnostic_best is None
                or evaluation.centered_nll < diagnostic_best.centered_nll
            ):
                diagnostic_best = evaluation

        problem.evaluation_observer = diagnostic_observer
        try:
            conditioning = _conditioning_diagnostic(
                problem, candidate, settings.derivative_policy
            )
        finally:
            problem.evaluation_observer = retain
        seconds["diagnostics"] += clock() - mark
        diagnostic_improvement = (
            max(0.0, candidate.centered_nll - diagnostic_best.centered_nll)
            if diagnostic_best is not None
            else 0.0
        )
        derivative = conditioning.derivative
        objective_available = derivative.objective.complete
        residual_available = False
        if derivative.jacobian is not None:
            residual = _poisson_residuals(inputs.targets, candidate.expected_counts)
            if np.all(residual.derivative_available):
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_available = (
                        bool(
                            np.all(
                                np.isfinite(
                                    derivative.jacobian * residual.derivatives[:, None]
                                )
                            )
                        )
                        and not problem.work.deadline_reached
                    )

    unattempted = tuple(name for name, _ in bank.selected if name not in attempted)
    stages = (
        bank.complete
        and not bank.missing
        and not unattempted
        and recovery_reason is None
        and check is not None
        and check.complete
        and assessment is not None
        and assessment.complete
    )
    reasons = []
    if not stages:
        reasons.append("required_stages_incomplete")
    if candidate is None:
        reasons.append("no_finite_candidate")
    if assessment is None or not assessment.complete or not assessment.consistent:
        reasons.append("fresh_assessment_failed_or_incomplete")
    if (
        check is None
        or not check.complete
        or check.passed is not True
        or check.candidate is not candidate
    ):
        reasons.append("candidate_validation_failed_or_unavailable")
    if problem.work.deadline_reached:
        reasons.append("wall_time_limit")
    if (
        diagnostic_improvement is not None
        and diagnostic_improvement > settings.descent_tolerance
    ):
        reasons.append("diagnostic_descent_detected")
    if assessment is not None:
        reasons.extend(assessment.reasons)
    summary = stats = structure = None
    if candidate is not None:
        structure = _component_structure(candidate)
        summary = api.TriexponentialComponentSummary(
            structure.amplitude_fractions,
            tuple(map(float, candidate.component_counts.sum(axis=1))),
            structure.finite_window_signal_fractions,
            structure.infinite_window_area_fractions,
        )
        if target_kind == "raw_counts":
            stats = calculate_poisson_fit_statistics(
                inputs.targets, candidate.expected_counts, n_parameters=8
            )
    same_diagnostic = (
        candidate is diagnostic_anchor
        and conditioning is not None
        and conditioning.derivative.objective.base is candidate
        and conditioning.candidate_id == candidate.evaluation_id
    )
    if not same_diagnostic:
        objective_available = residual_available = None
    identifiability = api.TriexponentialIdentifiabilityDiagnostics(
        8,
        structure.zero_amplitude_components if structure else (),
        structure.equal_lifetime_pairs if structure else (),
        structure.lifetime_ratios if structure else (),
        tuple(
            name
            for name, x, lo, hi in zip(
                problem.coordinates.names,
                candidate.coordinates,
                problem.coordinates.lower,
                problem.coordinates.upper,
                strict=True,
            )
            if x == lo or x == hi
        )
        if candidate
        else (),
        conditioning.status
        if same_diagnostic and conditioning is not None
        else "candidate_changed_or_unavailable",
        conditioning.rank if same_diagnostic and conditioning else None,
        conditioning.singular_values if same_diagnostic and conditioning else None,
        conditioning.condition_number if same_diagnostic and conditioning else None,
        conditioning.rank_tolerance if same_diagnostic and conditioning else None,
        tuple(d.status for d in conditioning.derivative.objective.derivatives)
        if same_diagnostic and conditioning
        else (),
        _public(candidate.parameters) if same_diagnostic else None,
        candidate.evaluation_id if same_diagnostic else None,
        conditioning.restricted if same_diagnostic else None,
        conditioning.derivative.objective.derivatives if same_diagnostic else (),
    )
    records = tuple(
        api.TriexponentialStartDiagnostics(
            i,
            s.kind,
            s.seed_id,
            _public(s.initial.parameters),
            _public(s.best.parameters) if s.best else None,
            s.best.centered_nll if s.best else None,
            s.endpoint.centered_nll if s.endpoint else None,
            s.reported_success,
            s.solver_status,
            s.termination,
            tuple(problem.coordinates.names[j] for j in s.fixed),
            s.call_limit,
            s.calls,
            s.seconds,
            s.derivative_statuses,
            s.safeguard_trials,
            s.solver,
            s.residual_evaluations,
            s.jacobian_evaluations,
            s.inward_start_coordinates,
            s.failure_events,
            s.failure_diagnostics,
        )
        for i, s in enumerate(starts)
    )
    work = problem.work
    work_record = api.TriexponentialWork(
        work.requests,
        work.objective_attempts,
        work.objective_calls,
        work.model_calls,
        work.charged_component_convolutions,
        work.completed_component_convolutions,
        work.model_failures_with_unknown_partial_work,
        MappingProxyType(
            {p: MappingProxyType(dict(counts)) for p, counts in work.phases.items()}
        ),
        MappingProxyType(seconds),
        tuple(work.blocked),
        int(stats is not None),
        clock() - started,
        0.0,
        problem.failure_events,
        tuple(problem.recent_failures),
    )
    if problem.work.deadline_reached and "wall_time_limit" not in reasons:
        reasons.append("wall_time_limit")
    numerical = api.TriexponentialNumericalAssessment(
        not reasons,
        tuple(dict.fromkeys(reasons)),
        stages,
        _public(candidate.parameters) if candidate else None,
        assessment.fresh.centered_nll if assessment else None,
        assessment.signed_objective_gap if assessment else None,
        assessment.consistent if assessment else False,
        _validation_record(check),
        assessment.prediction_consistent if assessment else None,
        assessment.prediction_reasons if assessment else (),
        assessment.validation_matches if assessment else False,
        candidate is not None and recovery_reason is None,
        objective_available,
        residual_available,
        diagnostic_improvement,
        _public(diagnostic_best.parameters) if diagnostic_best else None,
        diagnostic_best.centered_nll if diagnostic_best else None,
    )
    optimization = api.TriexponentialOptimizationDiagnostics(
        bank.generated,
        bank.screened,
        bank.rejected,
        bank.duplicates,
        len(bank.candidates),
        tuple(name for name, _ in bank.selected),
        unattempted,
        tuple(bank.missing),
        bank.complete,
        bank.projections,
        bank.projection_failures,
        bank.repairs,
        bank.scaling_fallback,
        records,
        settings.reserved_recovery_starts,
        recovery_used,
        settings.recovery_calls,
        recovery_calls,
        recovery_reason,
        candidate.origin if candidate else None,
        _public(best_observed.parameters) if best_observed else None,
        best_observed.centered_nll if best_observed else None,
        tuple(promotions),
        "accepted" if numerical.accepted else (recovery_reason or reasons[0]),
        work_record,
    )
    prepared = isinstance(inputs.irf, PreparedIRF)
    source = inputs.irf.source if prepared else None
    assumptions = api.TriexponentialFitAssumptions(
        problem.coordinates.lifetime_bounds_ns,
        problem.coordinates.temporal_shift_bounds_ns,
        settings,
        problem.coordinates.amplitude_scale,
        problem.coordinates.background_scale,
        target_kind,
        "prepared_irf" if prepared else "caller_declared_array",
        source.source_kind.value if source else None,
        MappingProxyType(dict(source.source_parameters))
        if source
        else MappingProxyType({}),
        MappingProxyType(dict(source.metadata)) if source else MappingProxyType({}),
        MappingProxyType(dict(source.provenance)) if source else MappingProxyType({}),
        inputs.irf.diagnostics if prepared else None,
    )
    result = api.MultiexponentialFitResult(
        _public(candidate.parameters) if candidate else None,
        candidate.expected_counts if candidate else None,
        candidate.component_counts if candidate else None,
        candidate.centered_nll if candidate else None,
        summary,
        stats,
        optimization,
        numerical,
        identifiability,
        assumptions,
    )
    elapsed = clock() - started
    work_record = replace(
        work_record,
        total_seconds=elapsed,
        overhead_seconds=max(0.0, elapsed - sum(seconds.values())),
    )
    if elapsed >= settings.wall_seconds and "wall_time_limit" not in numerical.reasons:
        numerical = replace(
            numerical, accepted=False, reasons=(*numerical.reasons, "wall_time_limit")
        )
        optimization = replace(optimization, stopping_reason="wall_time_limit")
    return replace(
        result,
        numerical=numerical,
        optimization=replace(optimization, work=work_record),
    )
