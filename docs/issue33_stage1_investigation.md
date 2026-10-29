# Issue #33 Stage 1: bounded optimizer-termination investigation

October 27, 2026. Refs #33 and #28. **Investigation only; no production correction
or new qualification.** The four authorized starts were executed once each.
All completed within 2,000 actual model calls and 300 seconds per start, including
the instrumented derivatives and subsequent numerical checks.

The replay demonstrates premature termination involving nonfinite objective or
gradient callbacks at zero background. It reproduces the saved per-start results
exactly at the level that the historical records permit. It does not establish
that changing one callback alone will recover the independent-audit objectives,
or that those objectives are globally optimal.

## Repository and scientific boundary

Before creating `investigate/issue-33-optimizer-termination`, local `main` was clean
and matched both `origin/main` and the remote main ref at
`209f3e6b8b6dd2e88f2a2f1b580333719e3b84d1` (merged PR #34).
AGENTS.md, Issues #33/#28 and roadmap Issue #5, the production fitter, relevant
tests, the PR-3 report and the proposed follow-up issue were reviewed. The older
planning-only text in AGENTS.md is historical; the current maintainer instruction
authorizes this investigation and branch, without authorizing a fitter edit.

The implementation boundaries remain:

- `physics.forward_model` supplies the unchanged PR-1 component expectations:
  shifted IRF, bin-width-scaled finite-window convolution, pre-exponential
  amplitudes, and one background added after summing components.
- `evaluation.diagnostics` supplies the centered Poisson NLL, one half of deviance.
  Positive counts with zero expectation have infinite cost; no floor is applied.
- `estimation.multiexponential` owns scaling, initialization, L-BFGS-B, candidate
  retention, local validation and descriptive identifiability diagnostics.
- The PR-3 benchmark/runner supplies saved observations, assumptions, source
  locks and independent-audit records. Its execution and acceptance populations
  were not rerun or revised.

The frozen exhaustive policy remains 46 starts and 56,000 optimization calls,
with initialization/validation/diagnostic limits of 128/100/100. Lifetime bounds
are `(0.1, 8)` ns and shift bounds `(-0.3, 0.3)` ns for these cases. The recorded
`ftol=1e-12`, `gtol=1e-6`, finite-difference step `1e-5`, 1,000-iteration cap,
`maxls=40`, local-descent tolerance `.01`, and warning settings are unchanged.

## Diagnostic implementation and pre-execution lock

The [protocol](issue33_stage1_protocol.md) and
[configuration](../configs/issue33_stage1_replay.json) were frozen before execution
in the [dated manifest](validation/issue33_stage1_manifest_2026-10-27.json).
The manifest binds the harness, its tests, protocol/configuration, installed
SciPy Python wrapper and binary, six historical input files, historical lock and
whole-directory inventory. The original 66-file source/settings/environment
identity was checked through the original runner, including the reference
Windows/Python 3.12.0, NumPy 2.5.1 and SciPy 1.18.0 environment.

`scripts/diagnose_biexponential_termination.py` calls the original `_refine()`
and `_Problem.value_and_gradient()`. Passive wrappers copy and log objective
probes, callbacks, actual model calls, and the arguments/results of each original
SciPy `setulb` call. They do not add a solver callback, alter options, reorder
evaluations or inject numerical probes during optimization. Synthetic controls
verify exact equality of the entire forward-input and objective/gradient callback
sequence with logging enabled versus disabled, excluding timing.

The saved physical starts are converted using the original coordinate transform
and saved post-initialization scales. Neither the initialization bank nor any
other start is run. The Gaussian prepared kernel is reconstructed from the locked
implementation and required to match its original saved hash; no observation is
generated. One thread is used. An external 300-second child-process limit includes
imports, input verification and checks. A common counter covers optimization,
local validation and derivatives; six independent direct convolutions per
analytic-derivative point are conservatively charged as six calls. Exclusive
attempt markers prohibit retries and overwriting partial/completed attempts.

After freezing each solver result, one local validation is run at its retained
best point. Its improved candidate is never optimized. Five predeclared derivative
points are checked using a separate problem instance: first finite-objective
callback, last finite-objective callback, retained best, saved production result,
and saved audit result. These checks do not revise the solver result or historical
qualification. A finite-objective callback can still have an invalid gradient;
the trace and derivative records retain that distinction.

Raw traces are separate, ignored files in
`data/generated/issue33_stage1_20261027/r011-s10`, `r011-s45`, `r058-s05`, and
`r058-s23`. Each directory contains an attempt marker, JSONL trace, result,
process outcome and process log. The compact
[results](validation/issue33_stage1_results_2026-10-27.json) include parameters,
checks, counters, reconstruction details and trace hashes. These four attempts
must not be repeated automatically.

## Four replay results

All objective values below are centered Poisson NLL. `Best` includes finite
derivative probes, which the original fitter intentionally retains. It is not
necessarily the optimizer's endpoint. The last column is diagnostic descent
from that retained point, not a new fit or revised qualification result.

| Case/start | Best | Endpoint | Solver termination | Iterations / reported nfev | Optimization / all calls | Child wall seconds | Local descent |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| C00 r11 / 10 | 2106.895908 | 2107.500969 | Relative reduction | 13 / 25 | 334 / 478 | 3.930 | 368.532377 |
| C00 r11 / 45 | 1511.035566 | 1511.371337 | Relative reduction | 3 / 7 | 80 / 224 | 5.133 | 282.593621 |
| C00 r58 / 5 | 337.320597 | 337.379840 | Relative reduction | 27 / 43 | 556 / 700 | 4.359 | 5.355715 |
| C00 r58 / 23 | 298937.749118 | infinity | Projected gradient | 1 / 3 | 29 / 195 | 4.911 | 17702.171592 |

All four solvers report success/status 0. All four retained points fail the
unchanged `.01` local-descent check. No replay exhausted either cap or was
incomplete. Totals are 999 optimization model calls, 190 validation calls,
288 shared-model derivative-check calls and 120 direct convolutions: **1,597
calls**, with approximately **18.33 seconds** summed child wall time. These are
four isolated-start costs, not full-fit performance estimates.

Best parameters, best objective, objective-evaluation count, and termination
message match each saved start exactly. The three finite endpoint objectives
also match exactly; the fourth endpoint remains nonfinite (historically stored
as null, explicitly recorded as `Infinity` here).

All 24 saved physical initial values have exactly zero round-trip difference;
none required clipping after inverse transformation. Adjacent binary64 internal
coordinates can map to the same saved physical value for r11/s45's first log
lifetime, r58/s5's zero shift, and r58/s23's second amplitude. Original internal
coordinates and iteration traces were not saved. Thus the replay matches all
available historical summaries, but a bit-identical historical trajectory cannot
be demonstrated. The synthetic logging control establishes instrumentation
equivalence, not recovery of missing historical state.

## Demonstrated termination mechanisms

Trace callback and event indices below are zero-based. Solver task codes are
interpreted using the locked installed `scipy.optimize._lbfgsb_py` mappings.
Raw `isave`/`dsave` arrays are preserved without assigning undocumented meanings
to their individual entries. Internal line-search operations that do not return
to Python are not exposed.

**r11/s10: failed finite differences followed by a zero-displacement stop.**
The last successful finite point has NLL 2107.5009687206166. Callback 23 requests
zero background, obtains finite NLL 7016.523823943518, then fails to calculate the
first log-lifetime derivative after all eight step reductions. Fourteen probes
have infinite cost with positive observed counts in zero-expectation bins 0/1.
`value_and_gradient()` returns six NaNs. SciPy next requests exactly the previous
finite point. Events 729/730 show line-search warning `[6,601]` (rounding errors
prevent progress), `NEW_X`, then `[4,402]` (relative-reduction convergence), with
identical coordinates and objective to the previous accepted point. Its finite
gradient infinity norm is about 60,564.5, so this is not stationarity.

**r11/s45 and r58/s5: infinite cost with a zero placeholder gradient.**
Callbacks 5 and 41, respectively, propose zero background and receive infinite
cost with an all-zero gradient. Each solver then evaluates the previous finite
point again, with identical coordinates and objective. Line-search warning
`[6,601]` is followed by `NEW_X` and relative-reduction success. The decisive
transition events are 175/176 and 1223/1224. Endpoint gradient infinity norms
are about 33,582.4 and 5,930.0. Returning `(infinity, zeros)` did not produce a
useful backtracking step or prevent a success termination. The production
callback docstring's assertion that infinite cost necessarily forces safe
backtracking is contradicted by these replays.

**r58/s23: NaN gradients, invalid coordinates, then false projected-gradient
success.** The initial shared-model objective is finite, 299467.2828863418, but
all eight inward shift-difference pairs include impossible expectations. Sixteen
probes have infinite cost; the callback returns six NaNs. The solver requests
NaN coordinates, which the feasibility guard rejects without model calls and
reports as `(infinity, zeros)`. It next requests the feasible all-zero
amplitude/background corner, also with infinite cost and a zero gradient.
Events 66/67 show `[6,601]`, `NEW_X`, and projected-gradient convergence
`[4,401]`. The returned endpoint has infinite cost, despite success/status 0.
The best finite derivative probe survives in the original candidate-retention
logic. SciPy reports three `nfev`, while four calls to the fitter's value/gradient
callback are observed, including a repeated NaN-coordinate request. Solver
`nfev` is therefore not the actual forward-model work count; the separately
instrumented 29 calls are authoritative.

These are reproduced failure paths at the application/solver interface. They
support premature termination rather than local-minimum trapping at the retained
points. They do not prove a SciPy defect for a well-defined finite callback, nor
that placeholder gradients alone explain every historical start. In r11/s10,
the decisive callback has a finite objective and NaN gradient, not the
infinite-objective/zero-gradient form.

## Derivative and convolution findings

Independent derivatives use direct discrete convolution, exact amplitude and
background derivatives, the analytic log-lifetime derivative, and the
piecewise-linear IRF shift slope. They are independent of objective differences.
At the three positive-background endpoints, direct and finite-difference gradients
agree well relative to their large nonzero values. The maximum componentwise
`abs(FD-direct)/max(1,abs(direct))` differences are approximately `2.62e-7`,
`5.22e-6`, and `1.46e-4`. This quantity is a scaled comparison, not a relative
error for a near-zero derivative. The largest absolute discrepancies are .0159,
.0679 and .2914, predominantly in the background coordinate. Their magnitude
does not explain a stationarity claim at gradients of thousands.

The finite-difference method is less reliable near zero background. The initial
r11/s10 background derivative differs by approximately 79.36% from its analytic
value. At r58/s23's retained point, background 0.0391148 counts/bin, the difference
is 4.235%. At the saved audit solutions the analytic scaled background derivatives
are +.0026713 and -.0159980, whereas the finite differences give -1.563894 and
-2.062740. The fixed scaled step corresponds to about .01956 counts/bin here;
finite-difference truncation/conditioning near low background deserves targeted
Stage-2 tests. No step size has been tuned. r58/s5's initial shift is exactly an
interpolation knot; that derivative is not treated as a smooth stationarity test.

Direct and FFT curves differ in relative L2 by less than `3e-16` at all twenty
checked points, but this aggregate measure hides a scientifically consequential
boundary effect. At r58/s23's zero-background initial point, direct convolution
has eight zero-expectation bins and infinite cost, while tiny FFT contributions
give a finite shared-model cost. Neighboring FFT evaluations clip some small
negative roundoff values to zero and become impossible for positive counts.
This explains the observed unstable finite/infinite neighborhood and failed
shift stencil. The analogous zero-background trial in r11/s10 has unstable
log-lifetime probes. This is an interaction of exact Poisson zero semantics,
convolution roundoff and the optimizer's boundary exploration; large curve norms
or finite central costs cannot certify a differentiable neighborhood. The shared
forward model, its zero semantics, and time-origin convention were not changed.

## Recovery capacity and scientific consequences

The saved complete fits used:

| Case | Seeds + recovery | Optimization calls / 56,000 | Validation calls / 100 | Saved production NLL | Saved audit NLL | Historical gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C00 r11 | 45 + 1 | 11,504 | 98 | 1228.441945 | 247.844923 | 980.597023 |
| C00 r58 | 46 + 0 | 11,267 | 49 | 331.964882 | 259.114084 | 72.850798 |

The separate local checks at r11/s45 and r58/s5 produce exactly the saved
descents 282.5936205035539 and 5.3557148815608. Their improved objective values
equal the historical returned production objectives. There is no unexplained
loss of a better finite candidate: the final returned candidates came from
validation, not the solver endpoint.

Both fits have no remaining start slot. `_fit_core()` checks candidate-bank
completion before reporting start exhaustion; both historical records therefore
say `candidate_bank_completed` and have empty `exhausted_budgets`. This is an
observability/recovery-capacity limitation, not an objective-budget overrun.
For r11, only two validation evaluations also remain; simply adding another
recovery start would not fund a full subsequent 49-call interior validation.
More recovery capacity might help, but these four replays do not demonstrate
that it is sufficient. Increasing starts or evaluations is not an established
remedy and was not attempted.

The saved production candidates still have independent gradient infinity norms
about 23,254.6 and 4,793.3. The saved audit candidates are much closer to
stationarity, but no global-optimality claim follows. This investigation does not
trace the other seven descriptive disagreements: their full iteration records
do not exist, and additional starts were not authorized. A common mechanism for
those cases remains unestablished.

The original scientific conclusions remain unchanged: C00 numerical acceptance
98/100 and both component median-error/bias categories passed; designated
independent objective agreement 119/121 failed. Both C00 disagreements were
rejected by local validation. The public API's explicit finite-best-candidate
failure contract worked; the private callback's safe-backtracking assertion and
optimizer-success interpretation are demonstrably inadequate. Neither descriptive
Jacobian conditioning nor identifiability warnings substitute for numerical
validation. Secondary statistical-recovery and IRF-mismatch findings are unchanged.

## Smallest justified Stage 2 scope, requiring separate approval

First correct the invalid callback/termination contract, retaining L-BFGS-B,
the shared forward model, physical bounds and exact zero semantics. Do not feed
NaN gradients into the solver or use a fabricated zero gradient as evidence of
stationarity at infinite cost. A failed stencil or unusable callback should have
an explicit numerical-failure path that preserves its best finite candidate and
charges every probe. Guard endpoint/status interpretation independently of the
solver success flag. Merely replacing zero by NaN, tightening `ftol`, or adding
an expectation floor is not a justified solution.

To improve recovery, rather than only make termination honest, separately design
and test bounded handling of invalid trial/stencil neighborhoods. Candidate
approaches are a finite, feasible one-sided stencil when a central stencil fails,
and explicit safeguarded retreat/recovery from a finite point; neither is proven
sufficient by Stage 1. Any new evaluations or restart must be visible and charged.
Keep synthetic boundary controls separate from the saved C00 regression cases.
Do not adopt an untested finite penalty that changes the Poisson objective.

Report recovery unavailability explicitly when the start cap or remaining
validation work prevents it. First assess the corrected callback under the
existing policy. If reallocation of seed/recovery slots or validation work is
needed, predeclare that as a versioned policy change rather than silently
increasing the 46-start/56,000-call limits. This investigation does not authorize
such a change or select new default settings.

Required regression coverage for that later correction:

1. Finite central objective with an impossible derivative neighborhood, including
   zero-background FFT tails: no NaN callback sent to L-BFGS-B, honest termination,
   retained best candidate and exact work accounting.
2. Nonfinite objective at feasible zero-amplitude/background corners and infeasible
   NaN coordinates: no false numerical acceptance or fabricated stationarity.
3. Repeated identical accepted point after an invalid trial: relative-reduction
   success cannot override failed independent descent/stationarity validation.
4. Derivative agreement on independent positive-background controls, very small
   background, one-sided bounds, shift knots, and unavailable derivatives. Include
   absolute tolerances near zero derivatives, without tuning on qualification data.
5. Full seed bank versus one rejected seed, no recovery slot, insufficient remaining
   validation calls, and exhaustion during a stencil/recovery. No hidden retries.
6. Retained zero components/background, equal lifetimes, ordered outputs, raw-count
   and IRF provenance guards, mono/PR-1 behavior and all phase budgets.
7. Separately authorized C00 r11/r58 regression runs under explicitly recorded
   corrected settings, targeting the original `.01` independent-objective gap,
   or preserving and investigating any miss. These observations are development
   regressions now, not untouched qualification or proof of global optimality.

Any production change requires a new, predeclared, versioned qualification before
claiming that the modified fitter meets scientific targets. Preserve the original
PR-3 results and failed target permanently. Stage 1 establishes a concrete defect
mechanism to address; it does not yet establish a successful correction.

## Integrity, validation and changed files

The original execution lock remains immutable and valid. Its current Git checkout
uses CRLF for some JSON text; the manifest records the current file-byte hash
separately from the original canonical payload/source identity. No lock was
regenerated or bypassed. The 7,825 historical checkpoint files remain exactly
123,390,600 bytes with sorted inventory digest
`d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`.
All six selected checkpoint hashes, observation links, kernel identity and
checkpoint envelopes verified. The original numerical implementations, PR-1/2
evidence, seeds, thresholds, exports, SQLite schema and release metadata are
unchanged. This findings document is post-replay reporting, outside both the
historical execution lock and the pre-execution diagnostic source lock.

Validation: all 11 instrumentation/control tests passed before freezing; the
post-replay relevant regression group passed **431 tests in 66.54 seconds**.
It includes the harness, fitter/identifiability, experiment/qualification helpers,
experimental adapters, mono/multi forward models, Poisson statistics and release
boundary/tooling tests. Ruff over `src tests examples scripts` passed; formatting
passed with 129 unchanged audited debt files. Whitespace and final source/evidence
integrity checks passed. The full repository suite, distribution rebuild and
scientific qualification were not repeated: no production implementation or
packaging file changed, and Stage 1 does not conclude Issue #33.

Seven new, uncommitted files comprise this stage:

- `configs/issue33_stage1_replay.json`
- `docs/issue33_stage1_protocol.md`
- `docs/issue33_stage1_investigation.md`
- `docs/validation/issue33_stage1_manifest_2026-10-27.json`
- `docs/validation/issue33_stage1_results_2026-10-27.json`
- `scripts/diagnose_biexponential_termination.py`
- `tests/test_biexponential_termination_replay.py`

Raw diagnostic artifacts total approximately 1.56 MB and remain ignored, outside
historical evidence. Test reports are under ignored `build/reports/`. No existing
tracked file was edited. No commit, push, PR, issue update, production correction
or new scientific qualification was performed. Stage 1 stops for maintainer review.
