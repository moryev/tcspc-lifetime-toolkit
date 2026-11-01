"""Deterministic K=3 template screening; NNLS is initialization only."""

from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import nnls

from tcspc_toolkit.estimation._fixed_k import (
    _FixedKParameters,
    _canonicalize_components,
)


@dataclass
class _SeedBank:
    generated: int = 0
    screened: int = 0
    rejected: int = 0
    duplicates: int = 0
    projections: int = 0
    projection_failures: int = 0
    repairs: int = 0
    scaling_fallback: bool = False
    complete: bool = True
    candidates: list = field(default_factory=list)
    selected: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    # A duplicate may represent more than one protected template bucket.
    buckets: dict = field(default_factory=dict)


def _key(evaluation):
    p = evaluation.parameters
    return (*p.lifetimes_ns, *p.amplitudes, p.background, p.temporal_shift_ns)


def _templates(coordinates):
    lo, hi = coordinates.lifetime_bounds_ns
    tau = np.exp(np.linspace(np.log(lo), np.log(hi), 5))
    tau[[0, -1]] = (lo, hi)
    shifts = (
        *coordinates.temporal_shift_bounds_ns[:1],
        coordinates.shift_mid,
        coordinates.temporal_shift_bounds_ns[1],
    )
    for shift_index, shift in enumerate(shifts):
        for indices in combinations_with_replacement(range(5), 3):
            bucket = f"shift{shift_index}:distinct{len(set(indices))}"
            yield (
                f"grid-{shift_index}-{''.join(map(str, indices))}",
                tuple(tau[list(indices)]),
                shift,
                bucket,
                False,
            )
    interior = tuple(
        np.exp(np.log(lo) + np.array([0.125, 0.5, 0.875]) * (np.log(hi) - np.log(lo)))
    )
    for i, shift in enumerate(shifts):
        yield f"interior-{i}", interior, shift, None, True


def _build_seed_bank(problem, settings, guesses):
    bank = _SeedBank()
    p = problem.coordinates
    y = problem.inputs.targets
    seen = {}
    guess_ids = []

    def basis(tau, shift):
        problem.work.request("initialization")
        values, failure = problem._model(
            _FixedKParameters((1, 1, 1), tau, 0, shift),
            "initialization",
            objective=False,
        )
        if failure and "limit" in failure:
            bank.complete = False
        return values

    def add(parameters, name, bucket=None, repair=False):
        canonical, _ = _canonicalize_components(parameters)
        try:
            x = problem.coordinates.encode(canonical)
        except (ValueError, FloatingPointError, OverflowError):
            bank.rejected += 1
            return None
        bank.screened += 1
        result = problem.evaluate(x, phase="initialization", origin=name)
        if repair and result.status == "infinite_objective":
            bank.repairs += 1
            return add(
                replace(
                    canonical,
                    background=max(canonical.background, 0.01 * float(y.mean())),
                ),
                name + ":background_repair",
                bucket,
            )
        if not result.finite:
            bank.rejected += 1
            if "limit" in result.status:
                bank.complete = False
            return None
        key = _key(result)
        if key in seen:
            bank.duplicates += 1
            name, result = seen[key]
        else:
            seen[key] = (name, result)
            bank.candidates.append((name, result))
        if bucket:
            bank.buckets.setdefault(bucket, []).append((name, result))
        return name

    # Scale before creating any coordinate-bearing candidate. This is charged.
    mid_tau = (float(np.exp(p.log_mid)),) * 3
    reference = basis(mid_tau, p.shift_mid)
    if reference is not None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scale = p.amplitude_scale / reference[0].sum()
        if np.isfinite(scale) and scale > 0:
            problem.coordinates = replace(p, amplitude_scale=float(scale))
        else:
            bank.scaling_fallback = True
    else:
        bank.scaling_fallback = True

    for i, guess in enumerate(guesses):
        name = add(guess, f"supplied-{i}")
        if name is not None and name not in guess_ids:
            guess_ids.append(name)

    for name, tau, shift, bucket, interior in _templates(p):
        bank.generated += 1
        unit = basis(tau, shift)
        if unit is None:
            bank.rejected += 1
            continue
        try:
            with np.errstate(
                over="raise", invalid="raise", divide="raise", under="ignore"
            ):
                if interior:
                    coefficients = np.r_[
                        0.99 * y.sum() / (3 * unit.sum(axis=1)), 0.01 * y.mean()
                    ]
                else:
                    design = np.column_stack((unit.T, np.ones(y.size)))
                    largest = np.max(design, axis=0)
                    largest[largest == 0] = 1
                    normalized = design / largest
                    norms = np.linalg.norm(normalized, axis=0)
                    norms[norms == 0] = 1
                    target_scale = max(float(y.max()), 1)
                    bank.projections += 1
                    try:
                        coefficients, _ = nnls(
                            normalized / norms, y / target_scale, maxiter=100
                        )
                        coefficients = coefficients / norms / largest * target_scale
                    except (RuntimeError, np.linalg.LinAlgError):
                        bank.projection_failures += 1
                        coefficients = np.r_[np.zeros(3), y.mean()]
                if not np.all(np.isfinite(coefficients)):
                    raise FloatingPointError("nonfinite coefficients")
        except (FloatingPointError, OverflowError):
            bank.rejected += 1
            continue
        add(
            _FixedKParameters(
                tuple(coefficients[:3]), tau, float(coefficients[3]), shift
            ),
            name,
            bucket,
            repair=True,
        )
    bank.generated += 1
    add(
        _FixedKParameters((0, 0, 0), mid_tau, float(y.mean()), p.shift_mid),
        "background",
        "background",
    )

    required = [f"shift{i}:distinct{j}" for i in range(3) for j in (1, 2, 3)] + [
        "background"
    ]
    selected = []
    for bucket in required:
        items = bank.buckets.get(bucket, [])
        if not items:
            bank.missing.append(bucket)
            continue
        item = min(items, key=lambda row: (row[1].centered_nll, _key(row[1])))
        if item[0] not in [s[0] for s in selected]:
            selected.append(item)
    for name in guess_ids:
        item = next(row for row in bank.candidates if row[0] == name)
        if name not in [s[0] for s in selected]:
            selected.append(item)
    if len(selected) > settings.max_seed_starts:
        raise ValueError(
            "supplied guesses exceed seed capacity with protected coverage"
        )

    remaining = [
        row for row in bank.candidates if row[0] not in {s[0] for s in selected}
    ]
    lo, hi = np.log(p.lifetime_bounds_ns)

    def location(row):
        q = row[1].parameters
        return np.r_[
            (np.log(q.lifetimes_ns) - lo) / (hi - lo),
            (q.temporal_shift_ns - p.shift_mid) / (2 * p.shift_scale),
        ]

    farthest = False
    while remaining and len(selected) < settings.max_seed_starts:

        def ranking(row):
            distance = min(
                (np.linalg.norm(location(row) - location(other)) for other in selected),
                default=0,
            )
            return (-distance if farthest else 0, row[1].centered_nll, _key(row[1]))

        item = min(remaining, key=ranking)
        remaining.remove(item)
        selected.append(item)
        farthest = not farthest
    bank.selected = selected
    bank.complete = bank.complete and not problem.work.deadline_reached
    return bank
