# Issue #33 Stage 2B: seed-first scheduling

Development/regression correction only. Stage 3 scientific qualification is not
authorized. The branch remains `investigate/issue-33-optimizer-termination`;
Stage 1 commit `3bff378f` and all original Stage 2 records are preserved.

## Approved design and numerical contract

Stage 2's immediate safeguarded-continuation chains consumed starts before the
finite seed bank completed. Both C00 runs and 11 of 12 historical PR-2 regression
cases therefore failed the required-work acceptance gate despite passing local
validation and objective comparisons. All 14 saved traces already contained a
seed-origin candidate within .01 of their respective reference. This supports
restoring seed coverage, not relaxing the gate or increasing the budget.

For exhaustive fits, every finite seed retains its original order and receives
one bounded optimizer invocation. Per-start safeguarded probes remain evaluated,
charged and eligible for best-candidate selection, but their requests for another
solver invocation are omitted. Independent local validation then determines
whether recovery is required. Recovery retains the existing maximum of two
invocations per tier and the shared start/objective budgets. A partial bank may
still have a locally validated best candidate, but cannot be accepted as complete.

Economical/adaptive continuation priority, tiers, early stopping, validation and
limits are unchanged. The exhaustive-only decision does not alter the physical
model, Poisson objective, derivative safeguards, initializer, tolerances, root API,
zero amplitude/background support, equal lifetimes, canonical ordering or mono
fitting. All original default budgets remain: 46 starts, 56,000 optimization,
128 initialization, 100 validation and 100 diagnostic forward calls.

Three additive result diagnostics have defaults:

- `finite_seed_candidates`: actual finite bank size after initialization.
- `unattempted_seed_candidates`: remaining bank entries, including entries beyond
  a legitimate economical/adaptive early stop; those policies retain their own
  completion semantics.
- `optional_continuations_omitted`: originating start indices, including any
  validation-recovery starts whose per-start continuation request was omitted.
  These are not discarded finite probes or uncharged solver invocations.

`recovery_unavailable_reason` remains separate. Required validation recovery at
the cap is a failure; a complete bank with passing validation at exactly 46
starts is not automatically exhausted. Objective/initialization/validation
exhaustion remains a failure under the unchanged qualification gate. Diagnostics
remain separately available/unavailable and do not determine fit acceptance.

The regression runner now invokes the unchanged `assess_biexponential_fit()`:
physical feasibility, independent curve/objective recomputation, local validation,
required budgets and fitting-stage completion all determine `numerical_acceptance`.
Local validation is also reported separately. The assessment's forward call and
elapsed time are external to the fitter's phase counters and included in explicit
combined totals. References are the saved independent shared objectives for C00
and historical production objectives for PR-2, with their kinds recorded.

## Predeclared regression execution

After focused numerical/physical tests, run each of the existing 14 cases once
with unchanged settings and one numerical thread. Use exclusive directories
`data/generated/issue33_stage2b_c00_v1` and
`data/generated/issue33_stage2b_pr2_v1`. Preserve all attempts. The runner records
input hashes, source/settings/acceptance rules, source snapshots, full per-start
records and the external assessment. Source files remain fixed during execution.
No new qualification observation is generated. PR-2 stream-2 reconstructions
must match their saved count hashes before fitting. C00 absolute objective gaps
must be <= .01; PR-2 comparisons are historical regression comparisons, not
independent audits. Any miss is retained and investigated before further action.

## Results and validation

All **14/14** cases pass both independent local validation and the complete
external numerical assessment. Every assessment has an empty rejection-reason
list and charges exactly one external forward call. All cases stop with
`candidate_bank_completed`, have zero unattempted seeds, no exhausted budgets,
no unavailable required recovery, and zero validation-recovery starts. Maximum
detected local descent is zero in every case. R11 attempts 45 seeds (one screened
candidate was nonfinite); every other case attempts 46. The initializer and its
finite bank are unchanged. All 14 complete lists of initial seed parameters and
their order match the historical per-start records exactly.

The full machine-readable [Stage 2B record](validation/issue33_stage2b_results_2026-10-28.json)
contains parameters, changes from Stage 2, diagnostics, all counters/timings,
assessments, references, manifests and integrity hashes. Raw records retain every
start. Each saved case was fitted once in Stage 2B; there were no retries or
threshold/budget changes.

Calls below are initialization / optimization / validation / diagnostics;
**each row additionally has one external assessment call**. Seconds measure the
fit itself, serially with one numerical thread, before full-suite/build work.
The ledger also records assessment time and combined calls/time.

| Case | Centered NLL | Signed reference gap | Seeds | Omitted continuations | Phase calls | Seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| C00 r11 | 247.844922542221 | -3.666e-12 | 45 | 37 | 92 / 10303 / 49 / 28 | 7.066 |
| C00 r58 | 259.114084194550 | -9.265e-12 | 46 | 39 | 92 / 9460 / 49 / 27 | 6.200 |
| PR2 c00 separated | 187.794974371074 | -1.143e-11 | 46 | 26 | 92 / 15325 / 49 / 44 | 9.825 |
| PR2 c03 close | 210.506076769129 | -2.599e-9 | 46 | 32 | 92 / 11921 / 49 / 35 | 7.776 |
| PR2 c04 equal | 213.755727645381 | -6.807e-11 | 46 | 33 | 92 / 13951 / 49 / 34 | 8.645 |
| PR2 c05 zero secondary | 217.693491583417 | -5.068e-11 | 46 | 31 | 92 / 16078 / 46 / 31 | 10.446 |
| PR2 c06 weak secondary | 200.917175296909 | -1.767e-10 | 46 | 32 | 92 / 20307 / 49 / 35 | 12.893 |
| PR2 c09 100 photons | 225.261561635970 | +2.842e-14 | 46 | 0 | 92 / 12556 / 46 / 72 | 8.086 |
| PR2 c12 million photons | 176.594716314480 | -7.109e-8 | 46 | 16 | 92 / 19733 / 49 / 53 | 12.353 |
| PR2 c13 zero background | 190.280751388901 | +3.979e-13 | 46 | 11 | 92 / 13039 / 46 / 60 | 9.199 |
| PR2 c15 negative shift | 182.710170280242 | -6.764e-12 | 46 | 11 | 92 / 19085 / 49 / 60 | 12.156 |
| PR2 c17 short window | 178.527449952124 | -3.374e-11 | 46 | 32 | 92 / 15284 / 49 / 37 | 9.199 |
| PR2 c18 EMG | 195.914971528012 | -4.661e-12 | 46 | 36 | 92 / 15565 / 49 / 36 | 9.994 |
| PR2 c19 sampled double peak | 200.070809011950 | -3.865e-12 | 46 | 27 | 92 / 15354 / 49 / 41 | 9.408 |

C00 references are the saved independent optimizer's shared objectives
247.84492254222425 and 259.1140841945589. Both satisfy the unchanged absolute
.01 target. PR-2 references are the historical production fits of these same
stream-2 observations; no independent audit reference is asserted for them.
Neither comparison proves global optimality or fresh scientific qualification.

| Returned parameter | C00 r11 | C00 r58 |
| --- | ---: | ---: |
| Amplitude 1 | 19133.95508808706 | 19222.94412304516 |
| Amplitude 2 | 7240.234912761089 | 7185.450449451257 |
| Lifetime 1, ns | 0.8058896285220275 | 0.8090979170065407 |
| Lifetime 2, ns | 3.216767365471741 | 3.222513035565356 |
| Background, counts/bin | 1.4843480362122492 | 1.37838143659433 |
| Shift, ns | 0.07422834906337668 | 0.07489319788270302 |

### Cost and parameter/diagnostic changes from Stage 2

Across all cases, seed coverage increases from 354 to 643 attempts. The original
290 recovery starts are replaced by seed coverage; no recovery is needed by the
final validation. There are 363 omitted per-start continuation requests. Their
finite safeguarded probes remain charged and retained; these are not 363 hidden
solver invocations or discarded evaluations.

| Quantity | Stage 2 | Stage 2B |
| --- | ---: | ---: |
| C00 fitting seconds, total | 13.220 | 13.267 |
| PR-2 fitting seconds, total | 117.547 | 119.980 |
| All fitting seconds, total | 130.767 | 133.247 |
| PR-2 median fitting seconds | 9.093 | 9.616 |
| Optimization calls | 202298 | 207961 |
| All fitter forward calls | 204944 | 210519 |
| Separately recorded external assessment calls | not run by old runner | 14 |

Total fitting time increases about 1.9%, and optimization calls about 2.8%.
Individual times move in both directions. These are single serial regression
measurements on one workstation, not a platform-wide performance qualification.
The largest centered-NLL change from Stage 2 is 7.14e-12, so no objective-quality
regression is detected at the declared tolerance.

Parameters are exactly unchanged in PR-2 c00, c04, c09, c12, c15 and c17.
Other changes are small: the largest absolute amplitude change is 0.000247
(c05), lifetime change 0.000014985 ns (c03's second lifetime), background change
3.05e-7 counts/bin (c03), and shift change 2.09e-9 ns (c05). C00 lifetime changes
are below 3.3e-9 ns. The zero-background case still has exactly zero background.
The ledger retains all six signed parameter changes for every case; no parameter
or warning tolerance was adjusted to suppress these differences.

The only warning-list change is C00 r58 gaining
`insufficient_starts_for_agreement_assessment`. Its near-optimal start count falls
from eight to one when continuation chains are omitted, reducing the available
descriptive agreement evidence.
This is reported rather than suppressed; numerical acceptance does not depend on
warning absence. Other diagnostic numbers, including near-optimal start counts,
may change with the attempted candidates and are retained in the ledger.

### Evidence preservation and source chronology

All seven Stage-1 files match their recorded hashes. The 7,825 original PR-3
JSON files still total 123,390,600 bytes and retain inventory SHA-256
`d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`.
The historical execution lock is unchanged; only the authorized fitter differs
among its 66 locked sources. Historical Stage 2 raw records and consolidated
report are not rewritten. The new record includes their comparison provenance.

Both regression runs passed their end-of-run source guards. Review then found
that flat source snapshots collided because the fitter and assessor are both
named `multiexponential.py`. Before any source edit, complete relative-path
snapshots were added to each new run directory and verified against all original
manifest hashes. Existing flat snapshots, manifests and results remain intact.
The runner then received only the snapshot-path correction and its regression
test. Fitter, assessment, settings and numerical reporting are unchanged from the
14 runs; no fit was repeated. The ledger distinguishes executed and final runner
hashes.

### Validation and changed files

Focused numerical, physical, qualification and measurement-adapter checks passed
**407 tests**. After the snapshot correction, all **17** scheduling/reporting
controls passed. Existing Stage-2 safeguards also remain covered, including
installed-SciPy behavior, invalid objectives/stencils/endpoints, retained finite
candidates, exact Poisson zeros and budget limits. New controls cover complete
and incomplete banks, optional omission, blocked/available required recovery,
the unchanged two-recovery limit, all three policies, phase accounting and the
complete external assessment with its separately counted forward call.

The full repository suite passes **2,352 tests**, with **one intentional skip**
and three non-failing warnings, in 709.37 seconds on Windows/Python 3.12.0.
Combined statement/branch coverage is **83.75%**, above the unchanged 80% gate.
The skipped test is the explicitly invoked Week-7 reference regeneration; it was
not rerun. This local validation does not qualify every supported platform.

Ruff passes; the formatting ratchet passes with 129 unchanged audited debt files;
`pip check` and whitespace checks pass. Fresh wheel/sdist builds and strict Twine
metadata checks pass; both artifacts contain the final fitter source. Isolated
wheel and sdist installation checks pass, including canonical imports, the
ordered 66-name API, removed module paths, CLI/README/examples, base-only optional
dependency isolation, dependency consistency and a small Bayesian integration.
The artifact environments resolved NumPy 2.5.4 / SciPy 1.18.1; these smoke checks
are separate from the full project-environment suite and saved fits on NumPy
2.5.1 / SciPy 1.18.0. No package was published or project SDK dependency replaced.

Logs are `build/issue33-stage2b-focused.log`, `build/issue33-stage2b-pytest.log`,
`build/issue33-stage2b-final-build.log`, `build/issue33-stage2b-twine.log` and
`build/issue33-stage2b-distribution.log`; XML/coverage artifacts are under
`build/reports/issue33-stage2b-*`. The ledger records source and artifact hashes.

Stage 2B changes only the fitter's scheduling and additive diagnostics in
production. It updates `scripts/regress_issue33_optimizer.py`, adds
`tests/test_biexponential_scheduling.py`, parameterizes the existing continuation
control for the two unchanged policies, updates `docs/api.md`, and adds this
report and its validation JSON. The prior Stage-2 diff remains uncommitted.

## Remaining limitations and proposed Stage 3

A 46-seed bank still leaves no start for required recovery if its best candidate
fails validation. That remains an explicit rejection, not a guarantee that all
future observations will succeed. The 56,000-call limit may stop a future bank
early. A usable Armijo probe does not establish stationarity, omitted continuation
chains could discover other basins, and local feasible-descent checks do not prove
global optimality. Shape interpolation remains nonsmooth at shift knots. Equal
or weak components can remain poorly identified despite objective agreement.
Economical/adaptive early stops retain their policy semantics and still do not
meet the exhaustive completion requirement of the historical qualification gate.

Recommend a separately approved, newly versioned Stage-3 protocol using this
seed-first source/settings/assessment lock. Before generating observations,
freeze a new unused seed namespace, output directory, environment, serial thread
policy and all acceptance/audit rules. Preserve the 30-condition physical study
design and its 1,550 observations, 1,850 comparisons and 3,700 fits; retain the
baseline >=95/100 numerical acceptance, component median relative errors <5%,
absolute relative biases <2%, and .01 designated independent-objective agreement.
Retain the proposed 205 audits and 121 designated agreement targets from the
Stage-2 protocol proposal, with all-attempt denominators and explicit failures.
Record required versus optional work and the extra assessment call for every fit.

Predeclare independent direct-convolution optimization, time/work limits, no
post-failure threshold tuning, and handling of interrupted attempts. Stage-2/2B
C00 and PR-2 observations remain development regressions and cannot enter the new
independent denominator. No Stage-3 observations, scientific outcome changes,
commit, push, PR, merge, release or issue closure are authorized by this report.
