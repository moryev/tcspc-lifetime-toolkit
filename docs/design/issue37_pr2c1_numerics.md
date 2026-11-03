# Issue #37 PR 2C-1 — boundary derivatives and final diagnostics

This checkpoint follows the read-only forensics of Operational Pilot v1 at
`263d4f229d84690f7ee5ec40080cc89e08a5318b`. It is a numerical implementation and
deterministic software-test correction, not a pilot replay, policy comparison,
or qualification. The pilot is already-inspected development evidence. Its six
inputs, 18 attempts, locks and execution reports remain immutable. Its 6/6
L-BFGS-B acceptance, 4/6 TRF acceptance and 0/6 audit completeness are unchanged.

## Scope and preserved contracts

Only the private fixed-K foundations, dedicated K=3 result/finalization records,
and controlled tests change. Existing mono/bi fitters, physical forward models,
IRFs, Poisson semantics, 66 root exports, schema v1 and release metadata do not.
No seed counts, numerical tolerances, model-call budgets, audit code/policies or
development-manifest bytes change. Explicit settings remain required. Existing
PR 1/2A/2B documents describe the earlier checkpoints; this document supersedes
their derivative-side selection and diagnostic ordering for the corrected code.

## Near-boundary derivative construction

The earlier implementation shortened each side to one quarter of its distance
to the bound and required both sides at interior points. A solver's tiny inward
adjustment could therefore demand differences dominated by rounding on the
bound-facing side, despite a usable inward derivative.

Exact coordinate bounds remain `bound`. A feasible interior side with less
than four requested step lengths of room is `insufficient_bound_room`, not an
infeasible direction and not a snapped endpoint. It is not evaluated using a
collapsed step. A sufficiently separated opposite side must independently pass
the existing two-scale, three-point derivative check. It can supply a
`feasible_side_near_bound` column for smooth lifetime dependence or a smooth
shift-interpolation region. If neither side supports the stencil, no column is
manufactured. Two available sides still must agree under the unchanged policy.

For each side, the three probe points must have distinct decoded physical values
and signed, representable displacements. The derivative of the interpolating
quadratic uses their actual spacings, rather than assuming exactly h/2, h, 2h.
Shift displacement is measured physically and divided by the shift scale;
lifetime displacement uses log1p((tau_probe-tau_base)/tau_base). Effective and
internal displacements must agree to the existing 1e-6 representability check.
Coarse and fine estimates retain the existing absolute/relative error checks
and bounded refinement count. The base parameters are never moved.

The smooth ideal lifetime derivative exp(-t/tau)*t/tau was considered. It is
used in an independent direct-convolution test oracle, not substituted into
production: the authoritative implementation includes FFT arithmetic and a
nonnegative roundoff clamp. Safeguarded differences continue to evaluate that
unchanged implementation and expose numerical unavailability where necessary.

Actual supplied interpolation coordinates remain authoritative. Continuous
kinks, support jumps, extremely short interpolation intervals, nonfinite
objectives and unrepresentable physical changes still withhold full derivatives.
Meaningful directional information remains available separately. Zero-amplitude
lifetime columns remain exactly zero; scalar-objective availability never
fabricates a Poisson residual Jacobian at zero expectation.

## Bounded failure evidence

New fields are additive and defaulted; historical JSON is neither migrated nor
rewritten. Private immutable scalar records describe:

- invocation index/kind/seed/solver, evaluation ID, phase and origin;
- exact internal coordinates, decoded physical parameters when available,
  coordinate/physical bounds and active-coordinate indices;
- shift interpolation geometry and zero-expectation counts;
- each coordinate's derivative status and signed directional scores;
- requested, actual internal, physical and effective step sizes, probe evaluation
  IDs, stencil type, refinement counts and failure statuses.

Each side retains at most 12 recent scalar probe records plus its total attempted
point count. Each problem retains eight recent failure events plus a total;
each invocation snapshots up to eight of its own events with its total count.
Omitted older events are therefore explicit. No event retains expected-count
arrays. Per-invocation records keep their submitted component labels; final
conditioning records use the canonical returned component order.
Malformed coordinates retain their original shape and at most one coordinate
vector's worth of flattened values. Foreign anchors retain explicit rejection;
their counts and shift geometry are not interpreted using a different problem.

Failed physical requests and derivative probes continue through the same ledger.
Recording a failure or computing a small SVD adds no physical call. Failed
infeasible-coordinate requests incur no physical call; charged/unknown partial
physical work retains its existing semantics. No extra solver invocation occurs.

## Finalization and acceptance

Search and bounded required recovery run first. Existing explicit validation
promotion remains, including truthful rejection when the last promoted point
cannot inherit the previous anchor's validation. The selected candidate is then
independently recomputed. Conditioning runs once on that frozen evaluation under
the existing diagnostic budget. Candidate identity is checked before attachment.

Diagnostic probes cannot promote a candidate or initiate recovery. Better finite
diagnostic points remain in best-observed evidence and are also reported in
`diagnostic_best_parameters`, `diagnostic_best_centered_nll` and
`diagnostic_improvement`. Improvement above the existing descent tolerance rejects
acceptance with `diagnostic_descent_detected`; smaller improvement is descriptive.
This is the same local-descent threshold, not a new production quality policy.
There is no finalization loop. Diagnostic unavailability alone is not converted
into failure or success of the scalar fit.

The legacy `consistent` field retains its aggregate meaning. Additive fields
separate fresh prediction/objective consistency and reasons, validation identity,
required-recovery completion, full objective-gradient availability, and full
residual-Jacobian availability. A fresh-consistent return with stale validation
remains rejected. Solver termination is still insufficient for acceptance.

When only some model columns exist, a separately labeled restricted sensitivity
record fixes every excluded coordinate. It reports its own coordinate list,
rank, singular values, error-aware tolerance and condition number; full rank
remains unavailable. A support jump never receives a full shift derivative.
Zero expectation, exhausted diagnostic budget, failed derivatives and SVD/deadline
failures are explicit. Rank and conditioning remain local, scaled, model-conditional
facts, not physical identifiability or component-recovery claims.

## Validation and subsequent review

New fixtures exercise exact/ULP/inward-adjusted bounds, analytic lifetime and
independent complex-step/direct-convolution oracles on suitable smooth operations,
nonrepresentable physical changes, conflicting sided estimates, bounded failure
records, controlled solver calls, final-candidate identity, explicit diagnostic
descent, stale-anchor rejection, restricted rank and diagnostic exhaustion.
Existing knot, approximate-grid, residual-zero, accounting and mono/bi guards
remain required. No test reads or replays pilot observations.

A separate verification proposal should first review this checkpoint and its
deterministic regression results, then authorize a new versioned source/environment
lock and fresh verification fixtures. Preserve original pilot evidence; do not
reuse its approved lock for changed source. Predeclare a small bounded population,
both solvers, unchanged candidate settings, all-attempt reporting, failure-coordinate
inspection and final diagnostic coverage before any execution. Any pilot-informed
algorithm or protocol choices remain development, not untouched qualification.
Audit trial handling/completeness and broader optimizer quality remain separate
checkpoints. No new pilot, large comparison, production default or PR 3 is authorized.
