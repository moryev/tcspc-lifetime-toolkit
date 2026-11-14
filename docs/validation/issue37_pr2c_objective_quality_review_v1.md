# Issue #37 PR 2C reporting correction and objective-quality review v1

## Scope and evidence identity

This is a read-only analysis of saved numerical verification v2 evidence and a
reporting-only software correction. The branch is
`feat/issue-37-pr2-triexponential-fitting`, based on
`f8f02349cce80a62053f525022b1383680336ed0`. The correction is uncommitted.
All historical numerical acceptance decisions retain their recorded values.
The twelve fits were accepted, six for each solver. The study scheduled zero audits.

The original report remains byte-identical, with SHA-256
`179f1defe14a47ef24d3cb08e534d1833691e8f90dc4d2b3abd40a92958658ee`.
The approved execution lock has SHA-256
`0ca894bc4cacc330c14b18bd94a57acc9cf4099f33912e160485cebbe6b7be63`.
The sealed input inventory identity is
`df3584336b6e7377180d2a9b4dceca08e484be08bb94178c6398576a011a0f7f`.
Its file SHA-256 is
`9fa1dbc6e4e0a9be16ddc5ca411f7f145a2506bed0bda74a26e875d567a6bbf4`.

Before the reporting edit, the ordinary execution-identity gate verified the
actual source and environment. Read-only store verification confirmed all six
input checkpoints and twelve task associations. All 57 checkpoint file hashes
matched the final execution receipt. Subsequent reporting-source edits have a
new identity and would require a separately reviewed execution lock for future
execution. This analysis reads the preserved original records without running
the report's physical-model reassessment or any optimizer.

The machine-readable erratum records each corrected probe's task and index,
the original report identity, the new reporting-source raw and LF-normalized
hashes, and all 34 V00/V01 L-BFGS-B invocation summaries. Its source hash is
`2a8ed422c0c8990332f56251ed7b6f2d0f15961a718aa220bfd086d671e4bddb`.
[Machine-readable erratum](issue37_pr2c_reporting_erratum_v1.json)

## Reporting correction

`_validation_summary()` previously selected feasible probes with a null
serialized objective as unavailable. JSON serialization maps nonfinite floats
to null. The evaluator explicitly records `infinite_objective` when a positive
target has zero expectation. That valid positive-infinite Poisson objective
therefore entered the unavailable category incorrectly.

The reporting helper now partitions recorded statuses into finite objective,
positive-infinite objective, unrepresentable feasible direction, numerical
evaluation failure, infeasible trial, redundant physical candidate, and
incomplete or missing record. Objective values are checked for consistency
with recorded finite/infinite statuses. Null alone supplies no classification.
Unknown statuses, missing fields, contradictory finite values and resource-limit
statuses remain visible in the incomplete category. Numerical overflow and
invalid model output have their own failure category.

`unavailable_feasible_directions` now contains only explicitly recorded
representability failures. The added classification summary preserves every
supplied probe record and its denominator. The same helper applies to an
explicitly supplied earlier validation record. Earlier complete validation
histories were not serialized by the fitter, so their probe populations remain
unknown. Retained derivative-stencil failure snapshots have separate semantics
and are not counted as additional final-validation probes.

| Problem | Solver | Reclassified positive-infinite probes |
| --- | --- | ---: |
| V00 mean | L-BFGS-B | 2 |
| V00 mean | TRF | 4 |
| V00 Poisson | L-BFGS-B | 4 |
| V00 Poisson | TRF | 4 |
| V01 mean | L-BFGS-B | 8 |
| V01 mean | TRF | 1 |
| V01 Poisson | L-BFGS-B | 4 |
| V01 Poisson | TRF | 8 |
| V02 mean | L-BFGS-B | 4 |
| V02 mean | TRF | 1 |
| V02 Poisson | L-BFGS-B | 2 |
| V02 Poisson | TRF | 1 |
| **All 12 fits** | | **43** |

Across 1,026 final-validation records, the corrected partition is 944 finite,
43 positive-infinite, 16 infeasible and 23 redundant. There are zero recorded
unrepresentable feasible directions, numerical evaluation failures, or
incomplete/malformed records in this final-validation population. This result
does not describe every earlier optimization evaluation. The original twelve
fit decisions and signed objective gaps remain unchanged.

## Observed L-BFGS-B results

Both deterministic targets have saved generating-truth centered NLL exactly
zero. This attainable objective provides a useful quality reference. Acceptance
under the declared finite local-probe tolerance permits a positive objective
deficit. All values below are read from saved evidence or computed by subtraction.

| Quantity | V00 mean | V01 mean |
| --- | ---: | ---: |
| Returned centered NLL and gap from zero | 0.187558199670 | 0.0293862511923 |
| Best finite recorded objective | 0.186706067424 | 0.0292741797483 |
| Best search-invocation objective | 0.187558199670 | 0.0293862511923 |
| Final validation best objective | 0.186840444810 | 0.0293177480631 |
| Final validation improvement | 0.000717754860 | 0.0000685031292 |
| Diagnostic best improvement | 0.000852132246 | 0.000112071444 |
| Saved TRF returned objective on identical input | 0.0000281916889 | 0.00000000135205210 |
| Selected and attempted seeds | 16/16 | 16/16 |
| Seed calls used / allocated | 8,768/16,000 | 12,374/16,000 |
| Seed invocation call limits reached | 2/16 | 10/16 |
| Recovery starts used / reserved | 2/4 | 0/4 |
| Recovery calls used / reserved | 2,000/4,000 | 0/4,000 |
| Total optimization calls | 10,768 | 12,374 |
| Full-coordinate weighted Jacobian rank | 8/8 | 6/8 |
| Active physical bounds at returned candidate | none | none |

### Initialization and coverage

Both records contain 109 generated templates, 109 available candidates, 105
NNLS projections, complete initialization, zero rejected candidates and no
missing required coverage. V00 screened 123 candidates including 14 background
repairs. V01 screened 125 including 16 repairs. All selected starts were attempted
in the recorded order. Each selected bank includes four starts at shift -0.3 ns,
seven at zero and five at +0.3 ns. Both use lifetime grid values approximately
0.1, 0.31623, 1, 3.16228 and 10 ns, together with a background template.
The full selected physical parameters are retained in the erratum.

The records preserve selected seed IDs and initial parameters. They do not
preserve the entire screened candidate ranking or every initial objective.
The compact bank's coverage contract was completed. Its sufficiency for finding
the best basin cannot be established from these records.

### Invocation outcomes and leases

Every seed and recovery invocation had a 1,000-model-call lease. The scheduler
divides 16,000 seed calls equally among 16 seeds and 4,000 recovery calls among
four reserved slots. Unused capacity from an early seed termination is not
redistributed by this implementation. The physical-call ledger includes
derivative stencils and safeguards.

V00's best seed, `grid-2-002` at index 7, retained objective 0.284952437039 at
its 1,000-call limit. Required recovery invocations 16 and 17 retained
0.234279692175 and 0.187558199670, respectively, also at their 1,000-call limits.
All three terminated with `derivative_unavailable`, accompanied by explicit
`invocation_model_limit` failures in derivative work. These records establish
resource-censored derivative availability at termination. They provide no
complete sequence of accepted iterates or evidence of a stationary local minimum.

The other V00 lease exhaustion was seed 4. Five seed invocations reported SciPy
relative-reduction convergence with shift fixed at zero. Four of these ended
near objective 88.09980614 and the background start ended at 289.08880735.
Nine further seeds terminated before their call limit after infinite-objective
or unavailable-derivative events. Some retained failure snapshots also include
infeasible coordinates. These conditional convergences describe their restricted
searches. The recorded final candidate came from a joint search.

V01's best seed, `grid-2-222` at index 6, reached objective 0.0293862511923 at
its 1,000-call limit. Ten of its sixteen seeds exhausted their leases, with
explicit limit failures. Its sole reported SciPy convergence was the
shift-fixed background start at 23.00596560. Five other seeds stopped early
after infinite-objective or unavailable-derivative events. The winning seed's
returned lifetimes are 1.0442063687987655, 1.044206368800776 and
2.5831761854424005 ns. Its first two amplitudes are also nearly equal.
The lifetime ratio of the first pair is 1.0000000000019256. They are close
enough to produce numerical rank 6, while the exact-equality diagnostic remains
false. The saved record establishes a nearly coincident returned representation.

Both fits retain finite candidates from unsuccessful invocations. No invocation
in either fit exhausted a time lease. The total unused seed allocations were
7,232 and 3,626 calls. These numbers describe the existing scheduling contract
and do not authorize reallocating work.

### Validation, promotions and finalization

V00 records two `validation_probe` promotions alternating with two
`recovery_best` promotions. Its final candidate has its own completed, passing
validation and a consistent fresh assessment. Both recovery slots were charged.
V01 records no promotions or recovery invocations because its candidate passed
the declared local validation. Required recovery is recorded complete for both.
Neither fit exhausted reserved recovery capacity.

The largest subsequently detected improvements were 0.000852132246 and
0.000112071444, below the unchanged 0.001 tolerance. Retaining those diagnostic
probes as the best-observed evidence without silently promoting them accounts
for the small returned-versus-best differences. Those differences explain only
about 0.45% and 0.38% of the returned objective deficits from zero.
Candidate selection/finalization therefore supplies a small measured part of
the deficit. The saved records show no stale candidate acceptance.

V00's final full-coordinate condition number is 1,474.911. V01 has two singular
values below its recorded rank tolerance and an unavailable full-rank condition
number. All final objective-coordinate derivative values are available.
Their maximum absolute values are about 10.5001 and 0.6914, respectively.
Passing the fixed local-probe tolerance does not establish gradient stationarity.
Conditioning describes these particular returned parameterizations and conveys
no physical-identifiability guarantee.

## Mechanisms supported and unresolved

| Candidate mechanism | Saved evidence and limits |
| --- | --- |
| Invocation-call allocation | Directly observed at both winning seeds and both V00 recovery endpoints. Unused global capacity coexists with exhausted invocation leases. Trajectories are needed to measure marginal benefit from reallocating calls. |
| Derivative or line-search failures | Many early starts encounter valid infinite objectives or derivative unavailability. Lease failures account for the winning-invocation terminations. Full line-search histories and separate iteration/callback counts were not saved. |
| Initialization and local minima | Compact coverage completed. Conditional convergences at substantially worse objectives and near-coincident V01 components support investigating basin access and symmetry. Saved evidence cannot certify a genuine joint local minimum or identify which omitted seed would help. |
| Coordinate scaling | No scaling fallback was recorded. Coordinate derivatives and final conditioning warrant investigation, but no alternate-scale comparison exists. A causal scaling defect is unestablished. |
| Recovery scheduling | V00 used two reserved slots and V01 passed before requiring recovery. Scheduling followed the reviewed local-descent rule. That rule does not require closing a known truth-objective gap. |
| Candidate finalization | Fresh candidate identity and validation match. The measured unpromoted improvements are small relative to the deficits. |
| Model-conditional weak identifiability | V01's returned near-coincident pair is rank deficient. V00 is full rank with substantial conditioning. These facts help explain parameter sensitivity but cannot explain away attainable objective deficits. |

## Proposed subsequent checkpoint for review

First add bounded diagnostic instrumentation with deterministic tests, preserving
current policy decisions. Record the initial objective, per-invocation scalar
and derivative call totals, accepted-iterate objective progress, projected
gradient norms where valid, line-search outcomes, lease exhaustion location,
seed ranking/selection reasons, and each validation anchor, improvement and
recovery decision. Bound event storage and report truncation explicitly.
Keep this instrumentation separate from fitting acceptance.

A later, separately authorized paired development experiment could compare
the current equal leases with one predeclared redistribution of unused seed
capacity, under the same total call/start/phase limits and unchanged selected
seed bank. Complete required seed coverage first and protect reserved recovery
capacity. Define the allocation and stopping rules before observing new
results. Any restart must occupy a declared invocation slot. If this design
cannot fit those constraints, return it for review before execution.

Use objective progress and termination mechanisms as outcomes, with every
attempt retained. Separately investigate whether nearly coincident initialization
persists in V01 once progress is recorded. A larger total model-call budget,
changed coordinate scaling, or a production-default choice is unsupported by
the current saved evidence alone. These inspected deterministic conditions
remain development evidence. Future scientific qualification requires its
separate protocol and random streams.

## Validation and protection

The reporting tests cover valid infinite Poisson limits, exact zero targets and
expectations, finite positive objectives, numerical failures, representability,
redundancy, malformed records, common aggregation across recorded passes,
unchanged acceptance, signed gaps and all-attempt denominators. An older test
fixture used the unrecognized shorthand `unrepresentable`; it now uses the
actual recorded status `unrepresentable_probe`.

Read-only hash verification covers 81 original pilot checkpoints, 57 verification
v1 checkpoints, 57 verification v2 checkpoints, 7,825 Issue #28 evidence files,
2,173 Issue #33 evidence files, 72 Issue #33 source locks and all 81 pre-existing
validation/planning artifacts. Historical 119/121 and 99/100 audit outcomes
remain preserved. Full-suite and packaging results are recorded in the final
handoff after validation completes.
