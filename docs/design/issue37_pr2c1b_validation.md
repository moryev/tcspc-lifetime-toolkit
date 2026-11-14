# Issue #37 PR 2C-1b — scalar validation and final-candidate recovery

This implementation checkpoint starts from commit
`ecb04f08fc4315010bced8beec947059698c9f67` and the maintainer's November 5, 2026
authorization. Its evidence consists of deterministic software fixtures and
read-only investigation of the completed PR 2C-1 verification. Scientific
execution requires a subsequent reviewed source and environment lock.

## Recorded rejection mechanisms

The sealed verification contains six inputs, 12 attempts and 57 checkpoint files.
Its approved lock file has SHA-256
`e5d280a726f7feb20247b56cb7691919a14bfa6eeca843a74ffbf0ab6750174f`.
Its inventory file has SHA-256
`34add9b0f81ea5d24af7697cc21c6a7ed22e74f0867579bcfe2111e6b1dbb160`.
The recorded acceptance count is 8/12. The following investigation reads those
records and interpolation-coordinate arithmetic only.

| Saved case | Mechanism | Classification |
| --- | --- | --- |
| V01 mean TRF | Reconstructing a knot through `shift + (time - shift - time[j])` produced `-0.06249999999999999`. The remaining distance of `6.938893903907228e-18` gave a quarter-step that rounded back to that value on both sides. | Cancellation in probe construction is an implementation defect. Explicit rejection of unavailable coverage is a conservative safeguard. |
| V00 mean L-BFGS-B | The same construction produced `0.062499999999999986` and a collapsed quarter-step. A better validation point was promoted before stopping on incomplete validation, leaving the old validation anchor attached. Final diagnostics recorded additional descent of about 0.00160284. | Knot construction and missing final-candidate validation are implementation defects. Rejection of stale evidence and descent above 0.001 are intentional safeguards. |
| V00 Poisson TRF | A reconstructed `0.03125000000000001` gave a collapsed quarter-step and two unavailable shift probes. | The same knot-construction defect. The saved record reports complete workers but incomplete numerical validation. |
| V01 Poisson TRF | A lifetime-bound probe moved an internal coordinate by `4.440892098500626e-16` and decoded to the identical physical lifetime of 10 ns. | The endpoint was already attained physically. Treating this redundant endpoint request as a missing physical direction is an implementation defect. |

Both V01 TRF runs used zero of four reserved recovery starts because the scheduler
stops on incomplete validation before allocating recovery. That ordering is part
of the existing numerical contract. Evaluating whether recovery should run despite
unusable validation requires a separate policy review. A solver invocation ending
at its model-call lease also remains distinguishable from a mathematical derivative
singularity. Corrected acceptance and objective quality on these six problems
remain unmeasured at this implementation checkpoint.

## Feasible scalar-objective probing

Scalar validation lives in `estimation._fixed_k_diagnostics`, with K=3 endpoint,
face and neighboring-shift checks in `estimation._fixed_k_search`. The objective
module supplies counted evaluation and derivative geometry to these callers.

Each requested direction is checked against decoded physical bounds. A direction
pointing out of an attained physical bound is infeasible. A direction with positive
physical room can be numerically unavailable. This distinction also applies when
an internal coordinate is one ULP inside a bound but decodes to the endpoint.

The requested coordinate step is shortened to the available box-bound room.
Construction then checks the actual decoded movement of every active coordinate
in the direction. If rounding or underflow loses that movement, construction can
enlarge the step from a machine-resolution estimate using coordinate precision,
log-coordinate offset precision and adjacent physical-value spacing. At most eight
metadata constructions are attempted. Every construction respects the box bounds.
Failure retains a feasible, unavailable probe and incomplete validation.

Only a distinct decoded candidate reaches the physical model. Its actual internal
displacement and evaluated physical parameters are retained in the probe record.
Scalar model calls, including impossible Poisson trials, use the validation ledger.
Metadata constructions consume wall time and zero physical-model calls. The
existing validation call limit and global deadline still govern evaluations.

Additional endpoint and zero-face requests are deduplicated by exact physical
parameter records. An already attained endpoint or previously checked extra point
receives `redundant_physical_candidate`, with no model call or new directional
coverage. Ordinary inward directional coverage is checked separately. A genuinely
different requested shift that decodes to the anchor remains unavailable.

Neighboring interpolation knots are constructed directly from the supplied
abscissa differences `time[i] - time[j]`. Closest distinct knots on each side are
kept, including real closely spaced knots on approximately uniform grids.
At most three nearby knots and two sided interior points per knot are requested.
Each side uses its own next knot or physical bound. A rounded quarter-step can
advance with `nextafter` only when a representable interior exists. An interval
with no representable interior retains explicit unavailability. This procedure
preserves real support discontinuities and uses scalar evaluations across them.
Derivative availability remains a separate diagnostic.

The 0.001 descent tolerance applies to every detected improvement as before.
A same-anchor physical evaluation cannot establish directional coverage.

## Promotion, finalization and recovery stages

The seed-first search retains the best finite evaluated evidence. Existing
validation promotion and counted required-recovery ordering continue to select
the returned candidate. Recovery uses its reserved starts, reserved calls and
per-invocation leases.

If the search stops immediately after a promotion, one terminal validation pass
checks that exact returned candidate using the remaining validation budget.
Selection is frozen for this terminal pass. Better terminal probes remain in the
best-observed fields and can reject acceptance. This pass adds zero solver
invocations and cannot start a validation/promotion loop.

An earlier unavailable required stage remains recorded. In particular, a passing
terminal validation cannot waive `required_recovery_start_limit` or the existing
`validation_incomplete` stop. The public record may therefore show a passing
final-candidate validation together with incomplete required stages and numerical
rejection. Changing that recovery policy requires separate maintainer review.

Fresh assessment and conditioning follow this terminal check and refer to the
exact returned candidate. Diagnostic descent above 0.001 continues to reject
acceptance. Diagnostic availability, conditioning rank, component recovery and
physical identifiability retain their separate interpretations.

## Deterministic review coverage

New fixtures exercise K=1/2/3/4 structures, exact and one/five-ULP lifetime and
shift bounds, feasible inward probes, tiny coordinate steps, transfer directions,
decoded endpoint aliases, large-offset shifts, approximate grids, intervals without
representable interiors, continuous knots, support jumps, exact zero faces and
an independently computed constant-count Poisson objective.

Controlled solver fixtures cover counted promotion, successful validation after
required recovery, terminal validation with exhausted calls, passing terminal
validation with an omitted required stage, best-finite retention, fresh assessment
and conditioning identity, and rejection of genuine descent. Existing scheduling
tests retain checks for seed coverage, reserved capacity, independent physical-call
instrumentation and false solver convergence. The fixtures use small newly defined
mathematical problems.

## Proposed subsequent verification gate

After review and a separately authorized local commit, prepare a new versioned
execution proposal against that commit and its enforced environment. Obtain
explicit approval before preparing observations or starting workers.

A candidate bounded follow-up would retain the six V00–V02 physical condition
definitions, two explicit compact solver configurations and 12 serial fits, with
all current numerical budgets, tolerances and external deadlines. It would use a
new reviewed random-stream branch, proposed branch 3, plus separate observation,
task, output and execution identities. Source/manifest/environment fingerprints,
two-stage input sealing and all-attempt reporting are required. Independent audits
would remain outside that proposed workload.

Predeclare reports of probe statuses and actual decoded displacement, final
validation identity, required-stage reasons, promotions, seed/recovery leases,
phase work, deadlines, objective quality and conditioning. Compare deterministic
targets only after exact input/kernel checks. Describe new Poisson realizations
as independent draws. The previous pilot and verification are already inspected
development evidence, with their historical acceptance and audit outcomes retained.

This proposed follow-up awaits authorization. Production policy selection,
independent qualification and subsequent audit/recovery checkpoints retain their
own review gates.
