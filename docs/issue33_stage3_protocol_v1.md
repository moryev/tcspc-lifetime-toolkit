# Issue #33: proposed independent Stage 3 qualification v1

**Status: protocol review only; scientific execution is not authorized.**
Protocol: `issue33_stage3_qualification_v1`. Machine-readable definition:
[`configs/issue33_stage3_qualification_v1.json`](../configs/issue33_stage3_qualification_v1.json).
This smaller design supersedes the earlier Stage-2/2B suggestion to repeat the
entire 30-condition PR-3 matrix. Those earlier reports remain historical records.

## Frozen implementation and verified evidence

Local implementation commit:
`5be88edda17ad6612a8447dca0e11a6543502b85`
(`fix: safeguard biexponential optimization and complete seed scheduling`).
Its parent is the separate Stage-1 commit
`3bff378fd7a41a6c46ff3bf21696cb101b8402bb`.
Exactly the twelve reviewed Stage-2/2B files were staged; the index was compared
with the validated working files before committing. No push occurred.

The final fitter LF SHA-256 is
`56ed8bf360ce500c8df26d7e6f555028cdbfe95785440a38721531a7e11413f6`.
Verification on October 29 checked the Stage-1 hashes, Stage-2/2B source hashes,
all recorded final test/build/install logs and artifact hashes, and all fourteen
full numerical assessments. Current implementation/test Python files and API
documentation match the validated sdist. The full-suite XML confirms **2,352
passed and one intentional skip**, with no failures/errors; coverage is 83.75%.
Ruff and formatting were rerun and pass (129 unchanged audited debt files).
Fresh wheel/sdist, strict metadata, isolated installations and their integration
checks passed in the retained Stage-2B evidence. These scientific fits/full tests
were not rerun merely to commit unchanged code.

All 7,825 original PR-3 evidence files retain inventory hash
`d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`;
the source/settings execution lock is unchanged. The only changed production
source among its locked files is the authorized fitter. The original **119/121**
audit agreement outcome remains failed and immutable.

## Final numerical contract

- Nonfinite objectives and unavailable/nonfinite/unrepresentable derivatives
  unwind the current L-BFGS-B invocation explicitly. Neither zero placeholders
  nor NaN derivatives are handed to the optimizer as valid stationary evidence.
- Amplitude/background derivatives use the exact Poisson score. Zero-count bins
  have score one even at zero expectation; positive counts at zero expectation
  still give infinite cost. Lifetime/shift differences use bounded finite
  stencils, finite one-sided fallbacks and bounded step reductions.
- Invalid trials may trigger a projected-gradient Armijo search (coefficient
  1e-4, at most 40 halvings). Every prediction is charged. Every finite candidate,
  including probes from failed steps, remains eligible for best selection.
- Returned solver endpoints are re-evaluated. Solver success is descriptive;
  independent feasible-descent validation and the complete external numerical
  assessment determine acceptance. The external curve/objective check uses one
  separately recorded forward call when a feasible candidate is available.
- Exhaustive fitting attempts each finite seed in its original order before
  local validation and required recovery. Optional per-start continuation
  requests are omitted and recorded; their finite probes are retained.
- Failed validation permits at most two recovery invocations per tier within
  the same budgets. A new invocation always consumes a start. At 46 complete
  seed starts there is **no spare recovery capacity**: if validation then fails,
  the fit is rejected with required recovery unavailable. Exactly 46 starts with
  complete required work and passing validation is not itself exhaustion.
- Initialization, optimization, validation and required-start exhaustion prevent
  full acceptance. Diagnostic exhaustion is reported separately. Best finite
  output can survive numerical rejection; no artificial success is assigned.
- Economical uses its four-seed tier; adaptive retains 4/8/16/full-bank tiers
  and its stability/disagreement early-stop rules. Both retain continuation
  priority. Their legitimate early stops do not satisfy this qualification's
  exhaustive-completion gate. Only exhaustive is qualified here.
- Public fit signatures, curated root API and existing result fields are
  preserved. Four optimization diagnostics were appended with defaults:
  `recovery_unavailable_reason`, `finite_seed_candidates`,
  `unattempted_seed_candidates`, `optional_continuations_omitted`.

Limits remain 46 starts, 56,000 optimization, 128 initialization, 100 validation
and 100 diagnostic forward calls, 1,000 optimizer iterations, `maxls=40`,
`ftol=1e-12`, `gtol=1e-6`, finite-difference step 1e-5 and detectable-descent
tolerance .01. The original Poisson likelihood, physical forward model, shift
and IRF preparation conventions, exact zero amplitudes/background, equal lifetime
support, canonical ordering, raw-count guard and historical mono fitter remain
unchanged. No expectation/background floor or acceptance-threshold change occurs.

## Scientific question and condition matrix

Does the corrected fitter converge reliably and agree with independent
optimization on unseen observations while preserving component recovery and
honest failure reporting?

Conditions retain their original PR-3 IDs and generating definitions. Unless a
row overrides them: 512 endpoint-inclusive bins from 0 to 20 ns; lifetimes
0.8/3.2 ns; detected signal fractions 0.4/0.6; **100,000 signal photons**;
background 2 counts/bin; shift .075 ns; Gaussian IRF centered at 1 ns with
FWHM .3 ns; fit lifetime bounds [.1, 8] ns and shift bounds [-.3, .3] ns.
The primary C00 override is **one million signal photons**.

| Original ID | Role / change from defaults | Replicates | Assumed IRF arms | Bi fits | Audits |
| --- | --- | ---: | --- | ---: | ---: |
| C00 | Primary; 1,000,000 signal photons | 100 | matched | 100 | 100 |
| C04 | Low signal: 100 photons | 10 | matched | 10 | 3 |
| C05 | True mono: lifetime 2 ns, fraction 1, 1,000,000 photons | 10 | matched | 10 | 3 |
| C08 | Nearly equal lifetimes: 2 / 2.1 ns | 10 | matched | 10 | 3 |
| C09 | Vanishing component: fractions 1 / 0 | 10 | matched | 10 | 3 |
| C10 | Weak component: fractions .995 / .005 | 10 | matched | 10 | 3 |
| C13 | Exactly zero background | 10 | matched | 10 | 3 |
| C16 | True shift at lower fitted bound: -.3 ns | 10 | matched | 10 | 3 |
| C17 | True shift at upper fitted bound: +.3 ns | 10 | matched | 10 | 3 |
| C21 | Short window: first 103 original-grid bins | 10 | matched | 10 | 3 |
| C26 | EMG generating IRF: Gaussian FWHM .3 ns, tail .2 ns, center 1 ns | 10 | matched EMG; misspecified Gaussian .3 ns | 20 | 6 |
| **Total** | **11 conditions; ten diagnostic secondary conditions** | **200** | | **210** | **133** |

The short window ends at 102 * 20 / 511 ns (about 3.992 ns). As in PR-3,
amplitudes are determined from unit component sums on each condition's actual
grid/window to give its declared detected signal photon total and fractions.
This does not renormalize a shifted IRF. Fractions are detected-photon fractions,
not pre-exponential amplitude fractions. The C04 count of 100 is signal only;
its expected background still contributes 1,024 counts over 512 bins.

The two C26 arms share exactly the same saved histogram, grid and window; they
are paired comparisons, not twenty independent observations. The study includes
**no production mono fits, pseudo-true projections, proxy calibrations, Bayesian
fits or ML workflows**. C05 is a mono *generator* fitted by the bi-exponential
method. This bounds the new numerical study rather than repeating PR-3's wider
measurement-integration qualification.

## Confirmatory targets and denominators

The primary population is fixed at C00 replicates **0 through 99**. Numerical
acceptance must be **at least 95/100 scheduled attempts**, using the unchanged
complete gate, including its required-budget and completion checks. Failures,
timeouts and unavailable results cannot disappear from that denominator. Missing
or unstarted records make the study incomplete; they cannot support early success.

For each canonical component, retain PR-3's definitions:

1. Relative error is (estimated lifetime - generating lifetime) / generating
   lifetime. Median absolute relative error must be **strictly below .05**.
2. Absolute mean signed relative error must be **strictly below .02**.
3. Each metric uses numerically accepted fits where that component has positive
   amplitude and detected contribution and finite bounded lifetime. Report its
   exact contributing n and unavailable-among-accepted count. Undefined metrics
   fail. No warnings, Jacobian-rank flags or unfavorable finite errors are filtered.

These are the existing *conditional* component metrics, not a claim that every
attempt recovers both components. Also report all-100 joint acceptance/both
components-within-5% frequency, unavailable components, all-finite and
rejected-finite sensitivity summaries. No new availability threshold is silently
added. This conditional-denominator limitation remains explicit.

Report Wilson 95% intervals for acceptance and 5,000 whole-observation bootstrap
replicates for component summary uncertainty, retaining missingness and C26
pairing. These intervals are descriptive study uncertainty, not additional
pass/fail criteria or per-observation lifetime intervals.

Secondary results have **no confirmatory numerical/recovery thresholds**. Report
each arm's acceptance out of ten, all failure reasons, component availability,
errors where defined, warnings, cost and paired C26 changes. Mono/vanishing
generators do not supply two identifiable component truths; nearly equal/weak
components remain difficult even if numerically accepted. Under misspecified IRF,
generating-parameter error describes model discrepancy and estimation together;
it is not by itself an optimization failure.

## Independent audit population, method and decision rule

Audit **all 100 C00 observations**, regardless of production acceptance. For
every secondary condition, audit zero-based replicate indices **[0, 4, 9] in
every arm**, also regardless of production outcome. This gives 100 confirmatory
and 33 diagnostic audits: **133 audits and 1,064 scheduled TRF starts**.
No production-driven audit selection or adaptive expansion is permitted.

Reuse `benchmarks.multiexponential.independent_biexponential_audit`: independently
implemented direct convolution, physical lifetime coordinates and signed Poisson
deviance residuals via `scipy.special.kl_div`, optimized by trust-region reflective
least squares. It must not call the corrected production optimizer or initialize
from its fit. Use the same saved counts, assumed kernel, grid and parameter bounds.
IRF preparation and the final shared-objective recheck remain shared dependencies;
this is optimizer/implementation independence, not independent measurement physics.

Keep the established seven-point geometric lifetime grid and index pairs
[(1,4), (2,5), (1,5), (2,4)] at initial shifts -.1 and +.1 ns. The original
observation-mean scale, amplitude starts 1/1 and scaled background .01 remain.
Use `x_scale="jac"`, `ftol=xtol=1e-12`, `gtol=1e-9`, at most 1,500 SciPy
function evaluations per start and **100,000 total direct predictions per audit**.
All derivative and endpoint predictions count toward the latter limit.

An audit is complete only when all eight starts finish successfully with finite
endpoint objectives, no budget is exhausted, and a feasible finite best candidate
and shared-objective recheck are available. Preserve best finite results from
incomplete audits for diagnosis; a small gap does not make such an audit complete.
Charge the shared audit recheck separately (one call when a candidate is available).

For complete audits compare the independently optimized candidate and the
production candidate using the same shared centered NLL. Production comparison
requires a feasible finite candidate that passes objective/curve recomputation;
it may still be numerically rejected for budget or local-validation reasons.
Require **absolute gap <= .01**; a substantially better production objective is
also a disagreement, not an automatic audit success.

**The primary audit target is 100/100 complete, available comparisons within .01.**
There is no allowance for excluding incomplete audits or changing the population
after inspection. Primary qualification passes only when the numerical target,
both component error/bias targets, this audit target and study-integrity/completion
requirements all pass. The historical 119/121 result is not its denominator.

Classify every scheduled audit in exactly one primary accounting category:

- `agreement`: complete, usable production objective, gap <= .01;
- `disagreement`: complete, usable production objective, gap > .01;
- `audit_incomplete`: timeout, interruption, failure, exhausted budget, unsuccessful
  start, or invalid/unavailable audit endpoint/recheck;
- `production_unavailable`: audit complete but production objective unusable;
- `not_run`: pending record, hence incomplete study, not a terminal success.

Retain orthogonal production-rejection and detailed audit failure flags even
when categories have precedence. Incomplete audits are not counted as proven
disagreements, but prevent the primary audit target from passing. Report the
agreement fraction out of all 100/33 scheduled audits as well as explicitly
labeled complete-comparison fractions. Secondary audit disagreements/incompletes
are diagnostic. Agreement never proves a global optimum.

## Leakage prevention, locks and checkpoints

New root entropy: **[33, 3, 20261029, 1]**, not present in the existing configuration
or study records examined for this proposal. Use NumPy `SeedSequence` and PCG64:
`root_entropy + [stream, original_condition_id, replicate_index]`.
Stream 0 is qualification; 1 is reserved calibration (unused); 2 is nonqualification
(no pilot scheduled); 3 is summary bootstrap; 4 is protocol tests (no draws needed).
Summary entropy is `root_entropy + [3, condition_id, arm_index]`; paired C26
differences resample observation indices using the matched arm's index 0 stream.
No optimizer randomness or seed derived from fit outcomes is introduced.

The original PR-3 observations, C00 r11/r58 and twelve PR-2 comparisons remain
development/regression evidence. None can enter any new independent denominator.
Keep old observations, source/settings locks, audit records and conclusions intact.

The accompanying [candidate lock](validation/issue33_stage3_candidate_lock_2026-10-29.json)
is a review snapshot, not an execution lock or authorization. It records the
implementation commit, all current package
sources, generator/audit/validator/protocol dependencies, explicit populations,
entropy identities and the configured environment. Source text hashes normalize
CRLF to LF; canonical JSON hashes bind populations and envelope payloads.

Before any scientific draw, separately review and freeze a new execution runner,
its tests, final protocol, sources, acceptance rules and complete populations.
Pin the full Python/dependency environment and numerical-library binaries; abort
on drift rather than silently refreshing a lock. The intended environment is
the validated Windows/Python 3.12.0, NumPy 2.5.1, SciPy 1.18.0 project environment,
with one worker, one numerical thread and explicit OMP/OpenBLAS/MKL thread limits.
The artifact smoke-test environment is not a substitute qualification environment.

The new runner must reuse the established deterministic generating physics and
independent audit, and the canonical raw-measurement bi adapter. It must assign
new Stage-3 measurement IDs/seed/protocol provenance; reused generator definitions
must not make these look like old PR-3 observations. Do not call the old paired
mono/bi production task, which would add unauthorized workload. No historical
runner, source lock or source-hash exception needs rewriting.

Predeclare 200 observation identities, 210 production tasks and 133 audit tasks.
Publish each observation before any fit. Use exclusive, atomic, content-hashed
envelopes binding protocol/source/environment lock, task identity, counts, grid,
assumed kernel, generating configuration and payload. Save per-start traces,
finite-best results, local/full validation, optional omissions, required recovery,
all phase/direct/external calls, durations and terminal failure categories.
Keep deterministic generation/template work separate from fitting counters.

Write durable started requests before work. Resume only never-started tasks.
Completed failures and timeouts are terminal and never retried automatically.
An orphan started task becomes an interrupted record after confirming no worker
remains; do not silently repeat it. The same applies to a generation request
without a published observation: preserve generation failure, do not redraw.
Integrity errors halt execution and require investigation. Source changes or
method fixes require a new reviewed version, never overwriting this evidence.

Each production/audit worker has a 300-second wall limit including its assessment
or shared recheck. Limit exhaustion remains a terminal result even when a useful
finite candidate survives. Preserve late/unpublished worker artifacts separately;
do not let them overwrite a timeout. Denominators come from the frozen population,
not whichever result files happen to be present.

New locations, currently uncreated:

- `data/generated/issue33_stage3_qualification_v1/` — raw immutable checkpoints;
- `docs/validation/issue33_stage3_execution_lock_v1.json` — approved execution lock;
- `docs/validation/issue33_stage3_results_v1.json` — machine-readable scientific results;
- `docs/issue33_stage3_qualification_v1.md` — new scientific report.

## Work estimate, validation and approval boundary

There are **343 fit/audit tasks**, at most 9,660 production solver invocations
and 1,064 scheduled independent starts. Fitter phase limits total 56,328 calls
per fit, plus one external assessment when available. Audit work is separately
limited to 100,000 direct predictions plus one shared recheck per audit. Two
component-model calls per condition template would add 22 generation calls if
templates are cached once; any reconstruction/verification work must also be
recorded, never hidden inside or outside these counters.

Stage-2B C00 averaged 6.63 fitting seconds; the selected PR-2 median was 9.62 s.
Historical independent C00 audits had median .231 s and maximum .800 s; the
selected secondary regimes' audit medians ranged .093 to 1.655 s. These are
planning evidence, not timing qualification of new observations. Allow roughly
30-45 minutes of numerical work and **1-2 hours wall time** for process startup,
checkpointing and difficult cases. The sum of all 343 worker timeout ceilings is
28 h 35 min; that is a pessimistic ceiling, not an expected runtime. One CPU
worker, no GPU, roughly 8 GB available RAM and 1 GB spare disk provide practical
headroom; measure actual usage without changing numerical settings.

Routine CI uses only protocol metadata and existing small deterministic controls,
never the new scientific stream. The new validator has no execution mode and
uses no numerical/random generator. **36 protocol tests pass**, covering exact
physics/settings/populations, hash drift, disjoint entropy identities, paired
arms, budget/threshold changes, audit denominators, incomplete/disagreement
separation, duplicate/foreign keys and pre-RNG blocking of scientific streams.
No observation or fit was generated for these tests.

Before execution, runner-specific tests must additionally establish atomic
checkpoint publication, tamper/lock rejection, source/environment drift refusal,
interruption/timeout resumption without retry, observation-before-fit ordering,
all-attempt production denominators and unavailable-component accounting, paired
arm identity, audit-completeness classification, external call accounting and
prevention of summary publication from incomplete populations. Reuse the existing
checkpoint abstractions where possible. These are execution prerequisites, not
claims that a new runner already exists.

The new protocol/configuration, validation helper/tests and candidate lock remain
uncommitted for review. Only Stage 2/2B was authorized for a commit. No scientific
execution, push, PR, merge, release or issue closure has occurred. Approval of
this protocol must precede runner finalization, final execution locking and any
scientific observation generation.
