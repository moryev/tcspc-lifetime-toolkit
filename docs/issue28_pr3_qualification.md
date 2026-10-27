# Issue #28 PR 3: completed independent scientific qualification

Status, October 27: **execution complete; primary C00 targets PASS; independent
optimizer agreement target FAIL (119/121 designated cases)**. The maintainer
accepted Disposition A: preserve the frozen study and document the numerical
limitation, with a separately proposed follow-up investigation. This does not
turn the failed target into a pass or establish experimental validity. No fitter,
warning threshold, scientific setting or analysis population changed. Refs #28.

The complete [machine-readable report](validation/issue28_pr3_results_2026-10-27.json)
contains all 36 condition/IRF-arm summaries, 205 audit records and 36 projection
statuses. The [attempt table](validation/issue28_pr3_results_2026-10-27.csv)
retains all 1,850 comparisons, including numerical failures and unavailable
quantities. The [post-study reporting record](validation/issue28_pr3_reporting_2026-10-27.json)
binds these artifacts and the revised documentation to the immutable execution
identity. The [proposed follow-up issue](issue28_optimizer_followup.md) has not
been created online; no diagnostic replays or Git publication are authorized.

The [manifest](../configs/issue28_pr3_qualification.json) is the executable
source of conditions, arms, seeds, bounds, numerical settings and thresholds.
The [runner](../scripts/qualify_biexponential_reconvolution.py) uses the unchanged
PR-1 physics/statistics and PR-2 fitter. The
[PR-2 numerical record](issue28_pr2_numerics.md) and all its original source locks
and observations remain historical evidence.

## Corrected population and workload

There are **30 generating conditions, 1,550 independent Poisson observations,
1,850 paired mono/bi comparisons, 3,700 production fits and 205 scheduled
independent optimization audits**. The independent proxy calibration adds one
calibration observation, outside all these denominators.

The arithmetic is executable and tested: `100 + 29*50 = 1550`; C25 adds two
extra assumed-IRF arms and C26–C29 one each, so comparisons are
`1550 + (2+4)*50 = 1850`. Audits are `100 + (29+2+4)*3 = 205`.
The earlier proposed 1,900/3,800/208 figures counted one extra 50-replicate arm.
That arm is absent from the listed scientific matrix; no condition was added
to match the erroneous totals.

## Grid, physical generation and matrix

Use `linspace(0, 20, 512, endpoint=True)`, lifetimes 0.8/3.2 ns, finite-window
detected signal fractions 0.4/0.6, background 2 counts/bin, positive shift
0.075 ns and a Gaussian source centered at 1 ns with FWHM 0.3 ns. C00 has
1e6 expected detected signal photons and 100 replicates. Every other condition
has 50 replicates and 1e5 signal photons unless overridden below. Bi lifetime
bounds are [0.1, 8] ns and shared shift bounds [-0.3, 0.3] ns.

`G(w)` is a Gaussian source with FWHM w ns. `E` is an EMG with Gaussian
center 1 ns, FWHM 0.3 ns and tail time 0.2 ns. Each source is explicitly
prepared before inference; shifting does not renormalize its kernel.

| C | Override | Assumed IRF arms |
| --- | --- | --- |
| 00 | Confirmatory baseline, S=1e6, 100 replicates | Matched |
| 01–04 | S=1e5, 1e4, 1e3, 100 respectively | Matched |
| 05–06 | True mono tau=2 ns, S=1e6 / 1e3 | Matched |
| 07–08 | Taus (2,2) / (2,2.1) ns | Matched |
| 09–12 | Secondary detected fractions 0 / .005 / .02 / .1 | Matched |
| 13–14 | B=0 / 20 per bin | Matched |
| 15–18 | Shift -.075 / -.3 / +.3 / +.35 ns | Matched; C18 is constraint mismatch |
| 19–20 | Taus (.1,3.2) / (.8,8) ns | Matched |
| 21 | First 103 coordinates of the baseline grid | Matched on truncated grid |
| 22 | Taus (.12,.4) ns | Matched |
| 23 | Generating G(.6) | Matched G(.6) |
| 24 | Signal=0, background=2/bin | G(.3) |
| 25 | Generating G(.3) | G(.3), G(.2), G(.45) |
| 26 | Generating E | E, G(.3) |
| 27 | E sampled on 257 coordinates over 0–20 ns, linearly prepared on 512 bins | Matched sampled E, G(.3) |
| 28 | Generating G(.3) | Matched, fixed independent calibration proxy |
| 29 | True mono tau=2 ns, generating E | E, G(.3) |

For each component, evaluate a unit-amplitude PR-1 curve at the actual shift,
then set `A_k = S*f_k / sum(unit_curve_k)`. Amplitude retains its pre-exponential
meaning. Sum components and add background once. C21 keeps S fixed within its
shorter window to isolate information loss from exposure loss. C27 is explicitly
synthetic sampled input, not an experimentally measured IRF. Exact all-zero
observations are a deterministic CI case, not 50 redundant stochastic cases.

C28 uses a separate mono tau=2 ns calibration with S=1e6, B=2, shift=0 and
G(.3). The existing leading-edge routine uses background window [0,.4), rising
window [.4,1.8), seven smoothing bins and polynomial order three. Its source,
counts, diagnostics and preparation are retained. One constructed proxy is fixed
across evaluation observations; a construction failure is retained without
redraw. This estimates performance conditional on one calibration, not calibration
uncertainty or experimental correctness.

## Numerical acceptance and availability

The primary acceptance fraction uses **all 100 attempted C00 fits**. A bi fit
is accepted only when its estimates are finite and feasible, its expectation
and centered NLL pass an independent re-evaluation, local numerical validation
passes, the candidate bank (or analytical zero-count solution) is complete,
and initialization, optimization, validation and start budgets are not exhausted.
Objective agreement uses atol=1e-8 and rtol=1e-10; returned-curve recheck uses
rtol=atol=1e-12. The one extra forward call is recorded separately from PR-2
counters. Optimizer termination alone cannot pass this assessment.

**Diagnostic budget completeness is separate.** Diagnostic exhaustion alone
does not invalidate numerical acceptance. Report the fraction completing the
diagnostic budget, Jacobian availability and its reason separately. Structural
degeneracy and interpolation-knot limitations may legitimately make a diagnostic
unavailable even when its work budget is complete. Warnings do not filter the
numerical acceptance or component-recovery populations.

The unchanged C00 targets are at least 95% numerical acceptance, and for
**each** canonical component: median absolute relative lifetime error <5% and
absolute relative bias <2%. Signed error is `(tau_hat-tau_generating)/tau_generating`;
relative bias is its mean, not the mean absolute error. Each component uses
accepted fits with finite bounded lifetime, positive fitted amplitude and
positive detected contribution, provided the generating model has two distinct
positive components. Exact fitted zero amplitude/contribution makes that
component unavailable; the fit can still be numerically accepted. Report its
unavailability count and exclude it only from that component's error denominator.
An equal fitted lifetime pair is retained in recovery errors when generating
components are distinct; it is not discarded to improve recovery.

Report all-attempt joint acceptance-and-both-errors-below-5% frequency, per-component
contributor counts, all-finite sensitivity and rejected-finite estimates.
An undefined component summary cannot pass. Neither component may compensate
for a miss by the other. Equal generating lifetimes, absent components, true
mono and background-only data have no arbitrary two-component relative truth
errors. Report curve relative L2 (undefined for a zero reference norm), total
signal, nuisance errors, fractions and descriptive lifetime averages instead.

Report a 95% Wilson interval for numerical acceptance and 5,000 whole-observation
bootstrap resamples for aggregate component summaries, preserving unavailable
rows. These are Monte Carlo study intervals, not component intervals on a fitted
histogram. A result of 95/100 does not establish that the population probability
is at least .95.

Noise-free qualification retains the existing independent PR-2 test: amplitude,
lifetime and background recovery within .1%, shift error <=1e-4 ns and curve
relative L2 <1e-6. Noisy component-recovery thresholds are not substituted for
these deterministic targets.

## Independent numerical audit

Audit all C00 observations and replicates 0, 24, 49 of every secondary IRF arm.
The different implementation uses direct `np.convolve`, linear shifted-kernel
interpolation, physical lifetime coordinates, mean-count amplitude/background
scaling and SciPy `kl_div` signed residuals. Eight TRF least-squares starts use
lifetime pairs (1,4), (2,5), (1,5), (2,4) from a seven-point geometric bound grid,
with shifts -.1 and +.1 ns clipped to the bounds. No production or generating
parameter initializes the audit.

Per start: `x_scale='jac'`, ftol=xtol=1e-12, gtol=1e-9, max_nfev=1,500.
A common hard 100,000 direct-forward-call limit counts derivative and endpoint
evaluations. One separately counted PR-1 forward recheck compares the final audit
point and the production point under the same centered objective. Complete audit
status also records termination of all eight starts. Preserve partial best points
and exhaustion; never convert incomplete audits into passed audits.

NLL agreement <=.01 is a target for matched C00, C01, C14, C15, C23, C25,
C26 and C27. Other regimes are diagnostic; retain both disagreement directions.
Agreement of these two finite searches is not a global-optimum certificate.
Any additional failure investigation must be separately labelled, never replace
the 205 scheduled audits or change the scientific matrix.

The existing Issue-4 pseudo-true projector is reused without alteration only
for matched C00/C01/C26/C27 and both C29 arms. All other projections are explicitly
unavailable by protocol. Its original four starts, strict objective/lifetime,
local-descent/gradient checks and independent least-squares check remain intact.
Failed projections retain the original reason and never become stochastic fitted
references. Each projection has a 300-second external timeout.

## Reproducibility, locks and interruption

Use PCG64 with SeedSequence entropy `[28,3,20261025,1,stream,condition,replicate]`.
Stream 0 is qualification; stream 1/condition 28/replicate 0 is proxy calibration.
Stream 2 is exclusively CI/example/pilot; pilot proxy calibration uses replicate
1,000,000, separate from pilot observations. Aggregate bootstrap uses stream 3
with condition and assumption indices. One observation is shared by every IRF
arm and both fitters. No frozen Week-7/8/9 or PR-2 observation is reused.

The full PR-2 settings are serialized in the manifest: exhaustive policy,
46 starts, 56,000 optimization calls, plus 128 initialization, 100 validation
and 100 diagnostic calls. No thresholds are tuned on PR-3 data. Source locks
hash LF-normalized package sources, runner and configuration and record Python,
NumPy, SciPy, pandas, threadpoolctl and platform. The original PR-2 fitter hash
is checked, not rewritten. Run serially with one numerical-library thread.

Observations are checkpointed and hashed before fitting. Each production pair,
audit and projection has an immutable key and atomic JSON publication. A pair
is the production checkpoint unit: interruption within it can repeat the pair,
but cannot select its most favorable result. Resume requires exact source,
settings and environment locks; changed or corrupt evidence is rejected.
Completed failures/timeouts are not automatically retried. Worker programming
errors stop orchestration for investigation. Each production pair and audit has
a 300-second external timeout. A terminated parent can leave a completed child
record; partial transport files are not completed scientific results.

Raw arrays and detailed start diagnostics stay under ignored `data/generated/`.
Compact review records belong under `docs/validation/`; generated raw logs do
not belong in distributions. The original pre-execution lock is preserved.
Maintainer approval changed only `execution_status` to `authorized` and created
the separate October 26 execution lock before the first qualification draw.
The scientific matrix/settings were unchanged. The completed study must not be
rerun merely to regenerate its summaries.

## Interpretation and failure classifications

Keep optimizer failure, numerical rejection, budget exhaustion, incomplete
diagnostics, unavailable IRF/projection, physical constraint mismatch, structural
degeneracy and finite-sample recovery misses distinguishable. A warning-free
error above 5% is recorded explicitly. The PR-2 warning-free 10.18% secondary
lifetime error remains a negative result; this study does not tune warnings to
detect it. Good conditioning, small deviance or favorable AIC/BIC does not
certify physical identifiability, generating model order or correct experimental
IRF assumptions.

The study is synthetic and conditional on the stated physical model. It does
not qualify periodic excitation, pile-up, time-varying backgrounds, IRF uncertainty
or acquired experimental ground truth. Issue #13 remains deferred.

## October 26 non-qualification runtime pilot

The [compact pilot record](validation/issue28_pr3_pilot_2026-10-26.json) retains
both passes and their source locks. The second pass adds explicit per-start
budget labels, atomic transport and process-wall timings; it does not change
the fitter, study matrix, thresholds or independent solver. Compared parameters,
objectives, acceptance, warnings and call counts were identical between passes.
All observations were stream 2. No scientific observation or scientific proxy
calibration was generated or inspected.

| C / assumed IRF | Production pair compute (s) | Pair process wall (s) | Independent audit compute (s) | Audit forward calls | Absolute NLL difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00 matched | 8.406 | 12.138 | .264 | 601 | 1.82e-8 |
| 04 matched | 15.504 | 19.854 | 1.013 | 1,658 | 1.25e-12 |
| 08 matched | 10.209 | 16.652 | 1.236 | 3,504 | .0105877 |
| 10 matched | 16.335 | 23.953 | 1.754 | 4,532 | 3.25e-9 |
| 21 matched | 10.607 | 16.576 | .198 | 1,978 | 2.54e-11 |
| 22 matched | 16.122 | 24.103 | 4.405 | 11,836 | 3.78e-11 |
| 26 Gaussian assumption | 8.224 | 14.524 | .332 | 1,003 | 1.59e-11 |
| 28 independent proxy | 6.535 | 10.289 | .174 | 687 | 6.03e-12 |

All eight production fits completed their required budgets and local numerical
validation; this is a feasibility result, not an eight-observation qualification
claim. C08's production NLL 254.008122 is **lower** than the audit's 254.018710.
The solutions allocate a weak short component at .323/.368 ns with the main
component near 2.064 ns. Production warns about weak contribution and near-optimal
start disagreement. The gap exceeds .01, but C08 is a diagnostic regime, outside
the predeclared agreement-target subset; neither search certifies a global optimum.

C22 has one audit start reaching 1,500 `nfev`, with a worse NLL of 671.79;
other starts reached the best NLL 278.188741. Its best objective agrees with
production, but the full audit remains **incomplete**. The global forward-call
budget was not exhausted; `exhausted_budgets=['start_nfev']` distinguishes the
per-start limit. Both limitations are retained without changing limits or policy.

Mean production compute is 11.493 s/pair (median 10.408, range 6.535–16.335),
including both fits, validation and identifiability calculations. Mean audit
compute is 1.172 s (median .673, range .174–4.405). Mean bounded-child wall
times are 17.261 and 7.291 s respectively. Startup, imports, source checks and
checkpoint I/O therefore materially affect total cost on this Windows setup.

Extrapolation to the unchanged matrix gives **5.91 compute hours / 8.87 wall
hours for 1,850 production pairs**, and **4.00 compute minutes / 24.91 wall
minutes for 205 audits**. Scenario ranges using pilot extrema are 5.29–12.39
production wall hours and 16.18–40.31 audit wall minutes, not statistical confidence
intervals. Allow roughly **10–14 hours** for a serial run including projections,
summaries and variability. Six eligible pseudo-true projections have a combined
30-minute timeout ceiling; unsupported projections remain labelled unavailable.
The original 205-audit schedule is computationally feasible and retained.

## October 26 authorized execution and completed primary baseline

Before any qualification draw, the accepted 66-file source/settings/environment
lock matched exactly. PR-1/2 implementations and frozen evidence had no diff;
matrix totals, streams and replicate counts matched the declared protocol.
The checkpoint failure/resume probe passed, with approximately 282 GiB free.
The [preflight record](validation/issue28_pr3_execution_preflight_2026-10-26.json)
preserves these checks. Following the documented approval procedure, only
`execution_status` changed to `authorized`; the
[execution lock](validation/issue28_pr3_execution_lock_2026-10-26.json) records
that transition. The original checkpoint/pilot locks remain unchanged. Raw
checkpoints use `data/generated/issue28_pr3_qualification_20261026/`, separate
from both pilot directories and the example.

All 100 C00 realizations and audits have completed. Numerical acceptance is
**98/100**, meeting the >=95% target; the 95% Wilson interval is
**92.999%–99.450%**. This interval does not establish a population acceptance
probability >=95%. All 98 accepted fits have both components available, and
all 98 have both relative lifetime errors below 5%. Diagnostic budgets completed
in all 100 attempts. No C00 warning-free component error above 5% was observed.

| Component | Accepted n | Median absolute relative error | Signed relative bias | 95% bootstrap median interval | 95% bootstrap bias interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short, 0.8 ns | 98 | 0.3961% | +0.0220% | 0.3088%–0.4853% | -0.0933%–+0.1402% |
| Long, 3.2 ns | 98 | 0.1529% | +0.0274% | 0.1218%–0.2329% | -0.0261%–+0.0813% |

Each component meets the unchanged <5% median-error and <2% absolute-bias
targets. All 5,000 predeclared whole-observation bootstrap resamples had defined
summaries. The all-finite population includes all 100 attempts: median errors
are 0.4147%/0.1549%, and signed biases +0.0282%/+0.0305%. The two rejected but
finite fits remain in that sensitivity population and in the acceptance
denominator; no fitted component was excluded as unavailable.

The independent optimization agreement target **fails for two C00 cases**:

| Replicate | Production centered NLL | Audit centered NLL | Absolute gap |
| --- | ---: | ---: | ---: |
| 11 | 1228.441945 | 247.844923 | 980.597023 |
| 58 | 331.964882 | 259.114084 | 72.850798 |

All 100 audits completed all eight starts; 98 agree within .01. Both discrepant
production fits were independently rejected by local validation. Their exact
start composition and termination records are:

| C00 replicate | Seeded / recovery starts | Termination messages | Optimization calls / 56,000 | Validation calls / 100 | Last validation descent |
| --- | --- | --- | --- | --- | --- |
| 11 | 45 / 1 | 46 relative-function-reduction terminations | 11,504 | 98 | 282.593621 |
| 58 | 46 / 0 | 45 relative-function-reduction; one projected-gradient termination | 11,267 | 49 | 5.355715 |

Both screened 46 candidates, used 92/128 initialization calls and 26/100
diagnostic calls. One initialization candidate was rejected for r11, leaving
one recovery slot; r58 had no recovery slot. The 46-start cap includes recovery.
No exhausted-budget flag was recorded because the candidate bank completed,
but the start cap constrained further recovery. There was no guaranteed reserved
recovery attempt. Both returned a better validation probe with numerical failure
explicitly retained; neither selected result was assigned optimizer success.

An October 27 read-only investigation reconstructed curves from saved parameters
and the hash-verified assumed kernel, without sampling or fitting. Production NLLs
reproduced exactly; direct/shared audit NLL differences were below 5e-12. The
production-versus-audit curve relative L2 differences were 4.574% and 0.942%
(audit curve denominator). Both production candidates are interior, with rank-six
weighted Jacobians and condition numbers 35.0/43.8. Their physical amplitude and
background NLL derivatives are (0.53412, -0.54237, 11.00812) and
(-0.11180, 0.07596, 1.11759). These feasible descent directions support premature
termination rather than stationary local-minimum trapping. Both had the
`insufficient_starts_for_agreement_assessment` warning; neither had structural,
conditioning or boundary warnings. Such diagnostic distinctions do not certify
physical correctness.

R58 start 23 reported projected-gradient convergence with a nonfinite endpoint.
The callback's zero placeholder gradient at nonfinite cost is a plausible hazard,
not an established cause of the selected fit's objective gap. Per-iteration
coordinates, gradients and line-search trials were not saved. **The exact
optimizer-level cause remains unresolved.** All eight independent starts agreed
closely in each C00 case, but this does not prove global optimality. No production
fit or audit was rerun, substituted or tuned. The retained API behavior is an
explicit numerical failure with its best finite candidate. The accepted
Disposition A preserves both audit-target misses and proposes a separately
authorized investigation rather than modifying qualification outcomes.

## Historical implementation checkpoint validation

The pilot's production JSON averages 55 KB/pair and observation JSON 10.4 KB.
Plan approximately 120–160 MB for raw checkpoints, plus temporary/report margin;
reserve 0.5 GB disk. Only one child runs at a time; reserve 1–2 GB RAM for parent,
scientific imports and child working storage (a planning allowance, not measured
peak RSS). No GPU or parallel worker pool is required.

The [checkpoint lock](validation/issue28_pr3_checkpoint_lock_2026-10-26.json)
matches the second pass. A real resume took 3.766 seconds, left all **59 JSON
checkpoints byte-identical**, and performed no refits. Focused tests separately
reject changed sources/settings, corrupted payloads and overwritten failures;
timeouts are retained, and programming errors stop execution. Run only one
coordinator per directory and let an interrupted child exit before resuming.

Preliminary integration validation: **72 focused tests passed**, including the
executable example outside the checkout, plus the repaired structural checks.
Ruff and the unchanged-debt formatting check passed. All original measurement
adapter functions/classes are AST-identical after required mechanical formatting.
PR-1/2 numerical sources, original qualification evidence, the Issue-4 projector,
root exports, schema and release metadata have no diff. The final full suite
passed **2,296 tests**, with **one intentional Windows-reference skip**, three
established warnings (two finite-invalid mismatch warnings and the Windows
physical-core detection fallback), and **83.62% combined statement/branch
coverage**, in 722.00 seconds. Coverage exceeds the unchanged 80% floor.
Those pre-execution tests did not run the large scientific study or establish its
then-untested acceptance targets. JUnit and coverage details remain in ignored
`build/reports/`. The completed scientific results follow.

## Completed study accounting and target verdicts

All 1,550 qualification observations and 1,850 paired fits completed, covering
3,700 production model fits and all scheduled condition/replicate/IRF arms.
There were no production timeouts, missing attempts or proxy-construction failures.
Bi numerical acceptance was **1,844/1,850**: C00 r11/r58 and C05 r11/r25/r46
failed local validation; C29 matched r45 reached 56,000 optimization calls before
completing the required fitting stages. Its passing local check did not override
that incomplete execution. Mono validity was 1,840/1,850; all ten invalid mono
fits were in C23. Finite curves from rejected fits remain descriptive outputs.

| Predeclared target | Actual | Verdict |
| --- | --- | --- |
| C00 numerical acceptance >=95/100 | 98/100 | PASS |
| C00 tau1 median absolute relative error <5% | 0.3961%, n=98 | PASS |
| C00 tau2 median absolute relative error <5% | 0.1529%, n=98 | PASS |
| C00 absolute tau1 mean signed relative bias <2% | +0.0220%, n=98 | PASS |
| C00 absolute tau2 mean signed relative bias <2% | +0.0274%, n=98 | PASS |
| Designated independent centered-NLL agreement <=.01 | 119/121 complete and agreeing | FAIL |
| Noise-free parameter/shift/curve tolerances | Independent deterministic test passed in checkpoint suite | PASS |

Noise-free test results are distinct from the repeated-Poisson study. No fitted
parameters were replaced with audit parameters. **Not all qualification targets
passed.** The maintainer accepted documenting the numerical limitation and
tracking it separately; this is not a threshold change.

All 1,850 diagnostic budgets completed. Weighted Jacobians were available in
1,763 fits; 39 were unavailable at shift-interpolation knots and 48 had
zero-expectation nonregular information. Diagnostic completeness, local rank
and physical identifiability remain different questions.

## Complete condition and assumed-IRF results

Errors and biases below are percentages in canonical lifetime order, conditional
on numerical acceptance and the frozen component-availability rule. The n pair
gives the actual component denominators; a dash means undefined, never zero
error. For mono/equal/absent-component/background-only generating models,
two-component truth errors are undefined by protocol. In other cases a fitted
vanishing component is excluded only from its component denominator; it remains
an attempted fit and a joint-recovery failure. Warning-free errors count accepted
fits with any defined component error >5%, over warning-free accepted fits with
at least one defined component error. Paired arms share observations, so these
rows are not independent populations to pool into a universal success probability.

| C | Assumed IRF | Accepted / attempted | Component n | Median absolute errors (%) | Signed biases (%) | Warning-free errors / eligible |
| --- | --- | --- | --- | --- | --- | --- |
| C00 | matched | 98/100 | 98/98 | 0.396 / 0.153 | 0.022 / 0.027 | 0/25 |
| C01 | matched | 50/50 | 50/50 | 1.505 / 0.846 | 0.242 / 0.054 | 1/50 |
| C02 | matched | 50/50 | 50/50 | 5.224 / 3.117 | 0.639 / 1.672 | 31/50 |
| C03 | matched | 50/50 | 50/50 | 26.877 / 28.850 | 7.073 / 40.431 | 39/40 |
| C04 | matched | 50/50 | 47/47 | 74.136 / 60.025 | -38.060 / 43.781 | 2/2 |
| C05 | matched | 47/50 | 0/0 | - / - | - / - | 0/0 |
| C06 | matched | 50/50 | 0/0 | - / - | - / - | 0/0 |
| C07 | matched | 50/50 | 0/0 | - / - | - / - | 0/0 |
| C08 | matched | 50/50 | 41/47 | 3.296 / 2.297 | -31.094 / 84.017 | 3/3 |
| C09 | matched | 50/50 | 0/0 | - / - | - / - | 0/0 |
| C10 | matched | 50/50 | 50/50 | 0.389 / 53.814 | -2.113 / 29.472 | 12/12 |
| C11 | matched | 50/50 | 50/50 | 0.396 / 14.676 | -0.281 / 1.188 | 41/49 |
| C12 | matched | 50/50 | 50/50 | 0.555 / 2.377 | -0.239 / -1.721 | 13/50 |
| C13 | matched | 50/50 | 50/50 | 1.445 / 0.717 | 0.364 / 0.217 | 0/0 |
| C14 | matched | 50/50 | 50/50 | 2.244 / 1.151 | -0.361 / -0.017 | 4/50 |
| C15 | matched | 50/50 | 50/50 | 1.361 / 0.950 | -0.315 / 0.156 | 1/50 |
| C16 | matched | 50/50 | 50/50 | 1.121 / 0.658 | -0.589 / -0.026 | 0/17 |
| C17 | matched | 50/50 | 50/50 | 1.340 / 0.939 | 0.281 / 0.036 | 1/24 |
| C18 | matched | 50/50 | 50/50 | 26.577 / 7.206 | 26.927 / 7.402 | 0/0 |
| C19 | matched | 50/50 | 50/50 | 0.000 / 0.442 | 0.402 / 0.012 | 0/12 |
| C20 | matched | 50/50 | 50/50 | 0.815 / 0.000 | -0.221 / -0.328 | 0/12 |
| C21 | matched | 50/50 | 50/50 | 11.305 / 20.639 | 1.763 / 20.684 | 44/45 |
| C22 | matched | 50/50 | 50/50 | 2.603 / 0.873 | 0.869 / 0.292 | 11/50 |
| C23 | matched | 50/50 | 50/50 | 1.552 / 0.928 | 0.599 / 0.264 | 1/50 |
| C24 | matched | 50/50 | 0/0 | - / - | - / - | 0/0 |
| C25 | matched | 50/50 | 50/50 | 1.746 / 0.791 | 0.084 / -0.097 | 2/50 |
| C25 | gaussian_0.2 | 50/50 | 50/50 | 21.062 / 2.560 | 21.171 / 2.546 | 50/50 |
| C25 | gaussian_0.45 | 50/50 | 50/50 | 22.919 / 4.544 | -23.075 / -4.710 | 50/50 |
| C26 | matched | 50/50 | 50/50 | 1.430 / 0.608 | -0.324 / -0.195 | 6/50 |
| C26 | gaussian_0.3 | 50/50 | 50/50 | 40.004 / 7.303 | 40.232 / 7.295 | 50/50 |
| C27 | matched | 50/50 | 50/50 | 1.376 / 0.871 | 0.347 / 0.355 | 2/50 |
| C27 | gaussian_0.3 | 50/50 | 50/50 | 41.212 / 7.578 | 41.262 / 7.742 | 50/50 |
| C28 | matched | 50/50 | 50/50 | 1.779 / 0.793 | 1.184 / 0.593 | 1/50 |
| C28 | calibration_proxy | 50/50 | 50/50 | 5.592 / 1.618 | 5.455 / 1.588 | 32/50 |
| C29 | matched | 49/50 | 0/0 | - / - | - / - | 0/0 |
| C29 | gaussian_0.3 | 50/50 | 0/0 | - / - | - / - | 0/0 |

### Physical recovery and model discrepancy

- Photon budgets: C01-C04 retained 50/50 numerical acceptance each while median
  errors deteriorated from 1.51%/0.85% at 1e5 signal photons to 74.14%/60.02% at
  100. C04 has three unavailable accepted estimates for each component; joint
  acceptance and both-errors-below-5% is zero. At 1e3 photons only 1/50 achieved
  that joint criterion.
- Weak components: at secondary fractions .005/.02/.1, median secondary errors
  were 53.81%/14.68%/2.38%. A zero generating component has no secondary-lifetime
  relative error. Component fractions and clearly named lifetime averages remain
  in the attempt table without being promoted to unique physical truths.
- Equal/near-equal lifetimes: equal generating lifetimes have no separately
  recoverable labels. C08 near-equal medians look modest, but biases are
  -31.09%/+84.02%, with 9/3 unavailable accepted components and only 1/50 joint
  successes. Extreme tails and unavailable components cannot be hidden by medians.
- Background and shifts: zero background (C13) recovered well and appropriately
  triggered boundary warnings. Elevated background (C14) increased errors to
  2.24%/1.15%. Negative shift and exact shift-bound cases remained recoverable.
  C18's generating +.35 ns shift exceeds the +.3 ns fit bound: all fits accepted,
  but errors reached 26.58%/7.21% and the median shift error was -.05 ns.
- Lifetime boundaries: C19/C20 place one true lifetime exactly at a bound.
  Approximately zero median error for that component partly reflects clipping,
  not unrestricted precision. These are descriptive boundary cases.
- Short window: C21 has median signal-curve relative L2 error 0.52% despite
  11.30%/20.64% lifetime errors; only 2/50 joint successes. C22's .12/.4 ns
  lifetimes gave 2.60%/0.87% errors but 11/50 warning-free recovery failures.
  The broader matched C23 IRF remained recoverable with the bi fitter.
- IRF mismatch: wrong widths in C25 caused 21.06%/22.92% median primary-component
  errors; Gaussian substitution for EMG/sampled EMG in C26/C27 caused 40.00%/41.21%.
  All 200 such fits were accepted, warning-free and failed >5% recovery.
- Independent proxy: C28's one independent stream-1 calibration constructed a
  .284444 ns FWHM proxy, with retained `negative_derivative_clipped_for_proxy`
  provenance/diagnostic information. Its 5.59%/1.62% errors and 32/50 warning-free
  failures exceed matched-IRF recovery errors. Successful normalization/preparation
  is not experimental verification, and one calibration does not assess
  calibration-to-calibration uncertainty.
- True mono (C05/C06/C29), equal lifetime (C07), zero component (C09) and pure
  background (C24) outcomes are described through curves, contributions and
  nuisance quantities. Signal-relative curve error is undefined in C24.

### Warnings and descriptive model comparisons

Among 1,498 accepted fits with at least one defined component error, 628 had an
error above 5%; **447 had no identifiability warnings**. Equivalently 447/1,041
(42.94%) warning-free accepted fits with defined errors failed that recovery
criterion. C00 contributed none (0/25 eligible warning-free fits). These are
descriptive results of a deliberately difficult, partly paired matrix, not
calibrated diagnostic sensitivity/specificity for experimental data.

All comparison deltas are bi minus mono. Across all 1,850 finite comparisons,
bi had higher full likelihood/lower deviance in 1,840, lower AIC in 1,419 and
lower bin-based BIC in 1,347. Among 1,834 pairs with both numerical-validity flags
passing, the respective counts were 1,827, 1,407 and 1,335. Rejected finite
curves are not silently removed from the all-finite descriptive population.

For C00, median likelihood gain was 19,779.47, median deviance difference
-39,558.95, AIC difference -39,554.95 and BIC difference -39,546.47.
True-mono conditions usually preferred mono after the complexity penalty.
C29 with a wrong Gaussian IRF preferred mono in all 50 AIC/BIC comparisons, yet
its median mono lifetime was 2.0762 ns against generating 2 ns. Conversely every
wrong-IRF bi-generating C25-C28 arm preferred bi by AIC/BIC while biased recovery
persisted. Information criteria do not verify the assumed IRF or physical model.

The comparison reuses full Poisson likelihood including factorial terms,
deviance = twice centered NLL, nominal p=4/6 even on singular boundaries, and
BIC using all bins (512, or 103 for C21). Observations, grid/window and assumed
IRF are shared. Historical mono lifetime bounds are [1e-12, infinity), with
background >=1e-12; bi uses [.1,8] ns and permits zero background. Shared shift
bounds do not make these identical fitting domains. No automatic model selection
or ordinary chi-square likelihood-ratio p-value is inferred.

## Completed independent audits and projections

All 205 audit attempts returned records, but only **204 completed every required
start**. There were 195 complete agreements within .01, and 196 close best-point
comparisons if the incomplete audit is counted descriptively. The designated
well-conditioned subset has **119/121 complete agreements: FAIL**. Its only misses
are C00 r11/r58 above. The largest gap among its 119 agreements is 4.97e-8.

Across all 205 available objective comparisons, absolute-gap median/p90/p95/max
are 5.10e-9 / 2.60e-8 / 4.95e-7 / 980.597. The following seven additional gaps
are descriptive, outside the predeclared agreement-target subset. Signed gap
means production minus independent centered NLL.

| Condition / replicate | Signed gap | Lower objective |
| --- | --- | --- |
| C05 / 0 | +0.405887 | Independent |
| C05 / 49 | +0.077165 | Independent |
| C07 / 0 | -0.159967 | Production |
| C07 / 24 | -1.701383 | Production |
| C09 / 0 | -0.081913 | Production |
| C09 / 24 | -0.491288 | Production |
| C29 matched / 0 | -0.166782 | Production |

All seven production fits passed the declared local check and had weak-component,
boundary, structural or start-disagreement warnings. Their competing curves
differed by only .011-.655% relative L2 despite different decompositions. This is
consistent with boundary/near-degenerate solution sensitivity, not evidence that
the independent optimizer always finds the global optimum or that the C00
premature-termination mechanism explains every disagreement.

C22 r0 is **incomplete**: one start reached 1,500 nfev (its final NLL 694.0684);
other starts found the best point, agreeing with production within 3.41e-11.
Its 11,077 forward calls did not exhaust the global ceiling. No audit hit the
100,000-call limit or external timeout. A close best point does not erase the
unfinished required start.

The 36 projection records contain **3 successful, 3 failed eligible, and 30
unavailable-by-protocol** outcomes. C00 and C01 failed unchanged strict start
agreement (NLL gaps .000866483 / 5.94378e-6; lifetime gaps 8.47955e-5 /
1.43092e-5 ns). C29 matched failed numerical validation with `ABNORMAL`.
Successful pseudo-true mono lifetimes were 1.982155 ns (C26 matched), 1.983815 ns
(C27 matched), and 2.076120 ns (C29 Gaussian assumption). These deterministic,
prior-free model-conditional projections are not generating physical truths.
Failed projections were neither relaxed nor replaced with noisy fitted values.

## Execution resources and integrity

The resumed process exited 0 on October 26 at 21:08:53 UTC. Its **6h57m01.6s**
covers the resumed segment only, not the full study. Whole-study worker wall
times sum to 27,884.27 s for production pairs, 919.47 s for audits and 120.43 s
for projection-status tasks: approximately **8h02m04s**, excluding orchestration
overhead and the interruption gap. All stage timing records are present.
Median production-pair worker wall time was 13.77 s; the internal fit timer
median was 10.69 s. These are measured workstation costs, not runtime guarantees.

Raw checkpoints occupy 123,390,600 bytes (7,825 JSON files). Intermittent process
samples observed roughly 323 MB for Python processes in the first segment;
monitoring stopped at the maintainer's request, so this is not a full-run peak
memory measurement. The interruption/resumption preserved all 1,162 preexisting
checkpoint files byte-for-byte; completed failures were not retried or deleted.
All 1,550 observation keys, 1,850 pair keys, 205 scheduled audit keys and 36
projection keys, random-stream assignments and paired observation/IRF hashes
were verified. No unfinished transport files remain.

The immutable execution lock hashes 66 sources/settings files and records the
environment. It matches the finished directory's lock; all checkpoint payload
hashes verify. Its SHA-256 is
`29b1f22449d67f05c47c926bfa05afe65273d18b3cfa09e9ab9d54ed8c94626f`.
The sorted checkpoint inventory SHA-256 remains
`d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`.

Documentation, tests and the post-study reporter are outside those 66 files.
Their revised hashes and final compact-result hashes are recorded separately in
[the reporting evidence](validation/issue28_pr3_reporting_2026-10-27.json);
neither historical lock was rewritten. Full raw observations/traces remain
local ignored evidence, not distributed raw logs. Public compact artifacts retain
per-attempt hashes, failures, unavailable quantities, per-arm aggregates and all
scheduled audits/projection statuses. A fresh source checkout can inspect these
summaries; full reaggregation additionally requires the original raw directory
and locked environment.
