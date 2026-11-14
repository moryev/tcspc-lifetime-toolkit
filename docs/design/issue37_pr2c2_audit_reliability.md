# Issue #37 PR 2C-2 independent-audit reliability

## Checkpoint scope

Stages A–E add independent audit trial classification, representable residual
differences, work records and candidate-specific assessment diagnostics. The
base checkpoint is `7ebcc400d1c9fe891557ebccb66bbfe1f5d63df9`. The implementation
is confined to `benchmarks._triexponential_audit` and its deterministic tests.

The preserved pilot recorded six incomplete audits, 96 TRF starts and 48 Powell
starts. Its Powell outcomes comprised 31 invocation limits, 12 nonfinite
objectives and five infeasible trials. TRF recorded 54 successful terminations,
25 unstable Jacobians and 17 invocation limits. All original candidates and
cross-scores remain historical development evidence. This checkpoint's software
tests use tiny synthetic controls and mocked solvers.

## Scalar trial and endpoint contracts

The audit retains its direct convolution, sampled-coordinate interpolation,
finite-window background semantics, centered Poisson calculation, coordinate
scaling and independent data-only seed formula. Its only dependencies are
NumPy, SciPy and standard-library modules.

Each scalar trial has an explicit outcome.

| Outcome | Powell callback | Recorded work |
| --- | --- | --- |
| Feasible finite objective | Return the calculated objective | Charge the actual direct-model and objective calculations |
| Positive target at zero expectation | Return positive infinity | Charge the physical evaluation and record `infinite_objective` |
| Finite coordinates outside physical bounds | Return positive infinity as the extended-real objective outside the domain | Record rejection before any model charge |
| Invalid coordinates or unrepresentable decoding | Abort with an explicit trial failure | Charge zero physical calls |
| Model failure or objective NaN/overflow | Abort with a numerical-failure reason | Preserve work already performed and finite candidates |
| Invocation/global call or time exhaustion | Abort with a distinct resource reason | Record the blocked request and zero additional physical calls |
| Solver numerical exception | Preserve the exception class and message | Retain all completed work and finite evidence |

Positive infinity is accepted only when impossible Poisson bins explain it and
the remaining objective bins are finite. An overflowing finite-expectation
objective remains a numerical failure. Resource gates also apply to rejected
callbacks, so repeated infeasible requests cannot evade elapsed-time limits.

The solver's endpoint is evaluated through the same feasibility and counted
objective path. Successful solver termination with an infeasible or infinite
endpoint leaves the invocation incomplete. Every invocation records its own
best finite candidate independently of the global best.

The fresh endpoint objective is also compared with Powell's reported scalar
objective or half the squared norm of TRF's reported residual vector. The check
uses the existing fresh-objective tolerances, 1e-9 absolute and 1e-12 relative.
Missing, malformed, nonfinite or inconsistent solver objective records leave
the invocation incomplete. Reported and independently recomputed values and
the consistency verdict are retained. This check adds no physical evaluations.

A bounded installed-SciPy Powell test exercises a finite–infinite–finite search
and returns a finite solution. SciPy can emit intermediate invalid-arithmetic
warnings involving infinities. Its behavior on arbitrary disconnected finite
regions remains a numerical limitation. All-infinite endpoints, numerical
exceptions and exhaustion retain explicit failure outcomes.

## Independent residual derivatives

TRF continues to use signed Poisson-deviance residuals and two-scale one-sided
finite differences with the existing `rtol=2e-3`, `atol=1e-5` stability check.
The requested coordinate step remains `1e-5 * max(1, abs(coordinate))`.

The derivative construction now performs the following operations.

1. Establish physical bound room and, for shift, distances to interpolation
   events on the actual supplied time coordinates.
2. Select a feasible side and construct two distinct representable physical
   points. Up to eight metadata attempts enlarge collapsed steps within the
   available room. A metadata-only attempt on the opposite side is allowed if
   the first side cannot supply distinct points.
3. Evaluate each selected point once through the charged audit evaluator.
4. Divide residual differences by actual physical displacements and apply the
   independent linear coordinate scale to obtain optimizer-coordinate columns.
5. Compare both slopes using the existing stability tolerances. Numerical
   failure or instability is reported explicitly.

Metadata construction creates no physical evaluations. A failed evaluated
stencil is retained as unavailable. There are no numerical stencil retries or
additional solver invocations in this correction.

`math.fsum` computes interpolation-event distances from supplied binary64 time
and shift values. This avoids cancellation from first rounding `time - shift`,
including Windows runtimes whose `longdouble` offers binary64 precision. The physical
model continues to evaluate the original `np.interp` expression.

Geometry records distinguish smooth intervals, continuous agreeing-slope knots,
continuous unequal-slope knots and endpoint support discontinuities. Overflowing
one-sided slopes yield explicitly unavailable geometry. A full
shift Jacobian is unavailable at an unequal-slope knot or support jump. Initial
physical knots retain the existing conservative shift-fixed invocation policy,
including agreeing-slope knots. Free coordinates and initial geometry are
recorded explicitly. A query that merely rounds onto a knot is classified using
the underlying supplied coordinates and representable feasible sides.

Exactly zero expectation retains a valid scalar objective when the target is
also zero. The residual Jacobian remains conservatively unavailable there.
Conditional fixed-shift solves retain their seven-coordinate status. Additional
active-face solve orchestration requires a separate numerical-policy decision.

Derivative diagnostics record the anchor, coordinate, requested step, selected
side, exact-bound status, interpolation geometry, left/right room, metadata-side
failures, physical trial points, actual optimizer/physical displacements and
stability disagreement or failure reason. The first TRF residual point records
any inward start adjustment separately from the submitted seed.

## Work and diagnostic records

The legacy `phase_calls` and `component_convolutions` fields remain available.
The latter is charged component work, three components per actual direct-model
call. Additional fields report completed component work and calls with unknown
partial component work after physical calculation failures.

The ledger distinguishes evaluator requests, scalar/residual/Jacobian callback
requests, model calls by phase and origin, objective calculations, feasibility
rejections, valid infinite objectives, numerical failures, finite-best
discoveries and blocked calls. Each Jacobian's anchor and probe evaluations are
charged once. A coordinate-decoding failure is detected before reserving a
physical call. Objective failures after a completed model retain that completed
component work.

Each start stores counter deltas, actual calls and its lease, elapsed time,
termination, solver status/message, endpoint, finite best, fixed/free coordinates
and inward adjustment. Trial and derivative diagnostics each retain the latest
32 records per start, with exact dropped-record counts. Complete counters cover
all events regardless of retention. Final assessment/validation trials have a
separate bounded diagnostic record and all 32 declared scalar validation
probes retain detailed records when reached. Incomplete validation records
the stopping reason and completed subset.

The pilot allocation remains 16 TRF starts and eight Powell starts under 40,000
total model calls. Search receives 39,742 calls, validation 256 and assessment
two. The first 22 invocations receive 1,656 search calls and the last two receive
1,655. At the 120-second ceiling, each start receives a 4.25-second time lease
within the global deadline. Unused call allocations remain unallocated to other
starts. The external worker deadline and cleanup controls belong to the runner.

## Candidate identity and strict completion

After search, a copy of the best finite coordinate vector is canonically ordered.
Fresh assessment and validation explicitly identify the same returned physical
candidate. The returned prediction and objective are freshly evaluated when
assessment succeeds. The original search-best objective remains recorded if
assessment fails or reveals inconsistency.

Validation keeps the existing coordinate directions, step magnitudes 0.01 and
0.0001 in each sign, and descent tolerance 0.001. Steps are shortened to feasible
bound room. Exact bound-blocked directions and feasible but unrepresentable
directions have different statuses. Valid positive-infinite probes complete
their directional evaluation. Better finite probes record descent without
promoting a new candidate. Fresh consistency uses the existing prediction and
objective tolerances.

Strict completeness still requires every scheduled start to terminate
successfully within its lease, a finite candidate, fresh consistent assessment,
complete passing local validation and an unexpired audit deadline. Failed
exploratory starts remain in all-start denominators. Cross-scoring uses the
sealed result and retains signed production-minus-audit objective differences.
Incomplete audits remain ineligible for confirmed agreement.

A possible future completeness-policy review could add a distinct measure of
scheduled work exhausted under declared budgets. Agreement eligibility under
such a definition would require separate approval and prospective evidence.
The historical 0/6 outcome continues to use its original strict contract.

## Nonexecuting operational-verification proposal

Proposed study identity is `issue37_pr2c2_audit_verification_v1`.
The future observation namespace is `[37, 2, 20261101, 4, condition_index, 0]`.
The comprehensive-development namespace retains branch 1.

This proposal reuses the inspected physical definitions V00–V02 from the
canonical verification-v2 manifest, SHA-256
`5afdf91cd1be72181491a7b39ef383c6c367f57e3c22bceab8663f00f0312769`.
The new labels A00–A02 identify newly sealed execution inputs. Deterministic
targets overlap the original pilot and both verification studies. New Poisson
observations would use independent draws. These controls remain operational
development evidence after their use in implementation decisions.

| Condition | Bins | Lifetimes in ns | Signal photons | Background/bin | Shift in ns | IRF |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A00 | 256 | 0.6, 2, 4.8 | 20,000 | 0.5 | 0.043 | Gaussian |
| A01 | 256 | 1, 1.2, 3 | 5,000 | 1 | -0.043 | EMG |
| A02 | 128 | 0.6, 2, 4.8 | 2,000 | 0 | 0 | Sampled with nonzero endpoints |

All use origin zero, 0.03125 ns bin width, detected fractions 0.25/0.45/0.30,
lifetime bounds 0.1–10 ns and shift bounds -0.3–0.3 ns. The exact declared
amplitudes, normalized-kernel construction and IRF parameters come from those
three pinned manifest conditions. Gaussian center/FWHM are 0.8/0.18 ns. EMG
adds a 0.35 ns tail. The sampled fixture adds 0.02 before normalization using
the pinned sampled coordinates and values.

The proposed six problem identities are
`A00:mean:matched`, `A00:poisson-0:matched`,
`A01:mean:matched`, `A01:poisson-0:matched`,
`A02:mean:matched`, `A02:poisson-0:matched`.
Observation IDs omit the `:matched` suffix. Each problem has one task formed
by appending `:audit`. This gives six workers, 96 TRF starts and 48 Powell starts.
Production fits and production/audit agreement statistics are outside this
proposed audit-reliability population.

Each audit uses the existing 40,000-call/120-second policy and strict assessment
rules described above, with serial execution. Aggregate worker limits are
240,000 direct calls, 720,000 charged component convolutions and 720 seconds.
The established five-second termination and five-second cleanup limits add at
most 60 seconds for six timed-out workers. Unconfirmed termination stops the
run. Preparation and reporting overhead are reported separately.

A separately authorized execution scope must first bind a newly committed
source inventory, environment and one-thread controls, exact manifest bytes,
six input identities, six tasks and every budget in an approved execution lock.
Preparation must publish six fresh hashed input checkpoints, atomically seal
the complete inventory and verify it before any worker. Task checkpoints must
bind the same execution and inventory identities. Resume verifies all evidence
and executes only genuinely missing tasks. Failed, interrupted and timed-out
attempts remain preserved. An incomplete preparation requires a controlled
recovery decision.

Proposed output is
`data/generated/issue37_pr2c2_audit_verification_v1`.
Required reports include all six scheduled attempts and all 144 starts,
termination reasons, candidate/assessment/validation identities, infinities,
rejections, unavailable derivatives, retained/dropped diagnostics, phase and
origin work, finite-best progress summaries, numerical and external runtimes,
censored outcomes and strict completeness. Timing and call sufficiency must be
measured from those results before any resource-policy revision. Future mean
target truth scoring may be performed only after sealing search results, with
its reporting work separately charged.

This document supplies a proposal for review. Executable scope registration,
manifest creation, execution authorization and input preparation require the
subsequent stage-F approval.
