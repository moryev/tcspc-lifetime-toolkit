# Issue #37 PR 2A: tri-exponential optimization checkpoint

This is a development checkpoint for maintainer review, not a selected production
policy or scientific qualification. PR #38 supplies the unchanged physical and
statistical foundations. Existing mono/bi fitters, their defaults, historical
Issue #28/#33 evidence, the ordered 66-name root API, schema v1 and release
metadata remain unchanged. Issue #33 remains open and deferred.

## Public boundary

The module-qualified entry point is
`tcspc_toolkit.estimation.triexponential.fit_triexponential_reconvolution`:

```text
fit_triexponential_reconvolution(
    time_ns, counts, irf, *, lifetime_bounds_ns, temporal_shift_bounds_ns,
    initial_guesses=None, settings=None,
)
```

`settings=None` raises `TypeError`. Callers must supply a
`TriexponentialFitSettings`; `TriexponentialFitSettings.development("compact")`
explicitly requests one unqualified development candidate. No root export,
arbitrary-K optimizer, measurement adapter or model-order selection is added.
The low-level caller asserts raw-count semantics; the canonical measurement
guard/integration remains PR 3. Noninteger, negative or nonfinite counts are
rejected. The private deterministic-expected-count route is for mathematical
tests and future development references, not relabelled Poisson observations.

The dedicated records are:

- `TriexponentialParameters`: three pre-exponential amplitudes, three lifetimes
  in ns, counts-per-bin background and one shared shift in ns.
- `TriexponentialFitSettings`: explicit budgets, tolerances and solver limits.
  Resolved settings, not a policy label alone, identify an invocation.
- `MultiexponentialFitResult`: K=3 parameters, readonly total/component curves,
  centered NLL, component summaries, Poisson statistics, optimization,
  numerical assessment, identifiability and assumptions records. A rejected
  finite result is retained; absence is explicit `None`, never NaN placeholders.
- `TriexponentialOptimizationDiagnostics` and `TriexponentialStartDiagnostics`:
  selected and unattempted seed IDs, coverage, projections, repairs, each
  invocation's lease/work/termination, recovery, promotions and best observed
  finite evidence, including probes not selected as the returned candidate.
- `TriexponentialNumericalAssessment`, `TriexponentialLocalValidation` and
  `TriexponentialValidationProbe`: exact candidate/objective, feasible coordinate
  directions, evaluated probes, best improvement, completeness, tolerance,
  fresh consistency and explicit rejection reasons.
- `TriexponentialIdentifiabilityDiagnostics`: zero amplitudes, equal-lifetime
  pairs, ratios, bounds and conditional weighted-Jacobian sensitivity or explicit
  unavailability. No physical-identifiability claim or uncertainty interval.
- `TriexponentialComponentSummary`, `TriexponentialFitAssumptions` and
  `TriexponentialWork`: pre-exponential, detected-window and ideal-area fractions;
  resolved IRF source/preparation provenance and coordinate scales; all work.

Fields describing physical parameters, returned-candidate assessment, provenance
and work have explicit semantics. Template names, derivative status strings,
private coordinate arrays and solver internals remain development implementation
details. Public records are frozen; result arrays and work mappings are readonly.
Provenance is snapshotted by the foundation input carrier.

## Modules and physical invariants

`triexponential.py` owns the dedicated records and public entry point.
`_fixed_k_initialization.py` owns templates, projection and selection.
`_fixed_k_solvers.py` owns a single safeguarded L-BFGS-B invocation and its lease.
`_fixed_k_search.py` owns seed-first scheduling, explicit promotion and assessment.
All compose PR 1. Only `_fixed_k_objective.py` gains narrow optional hooks: a
reservation guard, an evaluation observer, and derivative active indices for
explicit conditional solves. Defaults preserve PR 1 behavior; no mono/bi caller
uses these hooks. The complete foundation regression group remains required.

Coordinates remain linear scaled amplitudes/background, centered log lifetimes
and centered/scaled shared shift. A charged unit-component reference sets the
amplitude scale from the observation total and finite-window response; an
unrepresentable scale explicitly falls back to the validated count scale.
No expectation floor, lifetime separation, positivity transform on amplitudes,
silent IRF preparation, time snapping or component deletion is introduced.
Zero amplitudes/background and equal lifetimes remain representable. Returned
components are ordered by lifetime then descending amplitude, with stable ties.
Canonicalization is followed by a charged check if ordering changes. Log/linear
round trips can incur ordinary floating-point roundoff away from exact zeros.
All returned component curves, fractions and component-indexed diagnostics use
this order. Start/probe evidence instead preserves its own exact submitted
candidate and coordinate labels; those labels must not be interpreted as the
final candidate's component indices.

The existing `shift_irf()` and FFT physical model are unchanged. Positive shift
means later IRF; zero extrapolation, normalization, finite-window truncation and
the supplied approximately uniform coordinates keep their original meanings.
One model call evaluates three components; no basis cache hides physical work.

## Deterministic initialization

The bank has exactly **109 templates**, not a guarantee of 109 distinct starts:

1. Five log-spaced lifetime anchors, including the exact bounds, give
   35 unordered triples with replacement. At lower/mid/upper shift this gives
   105 NNLS templates. All multiplicities, including equal lifetimes, remain.
2. Three interior mixtures use lifetime log-fractions `(0.125, 0.5, 0.875)` at
   those same shifts, with 99% signal and 1% background starting allocation.
3. One background-only template uses the observed mean and midpoint shift.

Each NNLS solve uses three separately charged unit components and a constant
background column. Column/target scaling protects the projection, which is
initialization only and does not replace Poisson fitting. NNLS uses an explicit
100-iteration cap. Failure is recorded and supplies a background-only candidate,
without another NNLS attempt. A scored infinite objective permits one explicit,
counted 1%-mean-background repair. This is a starting-point choice, not a floor
on fitted expectations. Model failures and unavailable buckets remain visible.

Canonical physical-parameter equality deduplicates candidates. The best candidate
from each of nine shift/lifetime-multiplicity buckets and the background bucket
is protected; duplicates may cover multiple buckets. Unique supplied guesses are
retained next. Inputs must leave ten protected slots (`max_seed_starts - 10`
unique supplied guesses); excess guesses raise before fitting. A supplied guess
uses the same initialization and start budgets as any other candidate.
Remaining slots alternate objective rank and farthest lifetime/shift diversity
in normalized log-lifetime/shift space. Tie breaks use physical values, not
completion timing. The compact selection is a prefix of the larger selection
for identical observations/settings except capacity. Permutation-equivalent
supplied guesses do not consume multiple selected slots.

## Solver and nonsmooth behavior

The adapter makes at most one SciPy `minimize(..., method="L-BFGS-B", jac=True)`
call per scheduled slot. Every objective, basis and derivative stencil passes
through the same phase/global/invocation guards. Finite candidate observations
are retained even if a later derivative or endpoint fails. A reported solver
success is recorded but never used as numerical acceptance.

At a smooth region or a continuous knot with agreeing slopes, safeguarded PR-1
derivatives are used. At a continuous unequal-slope knot or a support jump,
the invocation explicitly fixes shift. It requests only the other derivative
coordinates; the full gradient remains unavailable. Exact zero faces discovered
by validation can fix the corresponding amplitudes/background and zero-amplitude
lifetimes during required recovery. A fully fixed face occupies its scheduled
slot and reports `no_free_coordinates` without manufacturing solver success.

A newly encountered unusable objective/gradient aborts that solver invocation.
If a usable finite gradient anchor exists, up to `max_line_search` (40) counted
Armijo probes may retain descent within the same lease. They are finite probes,
not another solver or continuation chain. Derivative estimates across a support
jump are never supplied as smooth gradients. Full PR-1 diagnostics retain
mathematically meaningful sided derivative information; public conditioning is
unavailable when a complete weighted Jacobian is unavailable.

Local validation evaluates PR-1 coordinate/transfer/splitting directions at
steps `1e-4` and `1e-2`, exact zero faces, lifetime/shift bounds and nearby shift
interval probes. Neighbor geometry uses actual coordinates, not nominal `dt`.
Only finite, representable, in-bounds sided points are evaluated; an unrepresentable
feasible neighboring point is recorded and makes validation incomplete. These
checks also distinguish mathematical infeasibility from overflowing coordinate
arithmetic, and mark a coordinate step that decodes to unchanged physical
parameters as unavailable. Any evaluation already made is still charged; neither
case can support acceptance. These probes can detect descent but do not prove
all directional minima or global
optimality, particularly on dense knots of approximately uniform grids.

## Scheduling, budgets and candidate selection

The candidates below remain unqualified; callers select one explicitly or give
custom settings. No settings change is permitted in response to development
observations without a versioned protocol revision.

| Candidate | Seed slots | Reserved recovery | Total slots | Optimization model calls | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact | 16 | 4 | 20 | 60,000 | 120 |
| Balanced | 32 | 8 | 40 | 120,000 | 300 |
| Exhaustive reference | 109 | 8 | 117 | 300,000 | 900 |

`floor(optimization_calls / 5)` is reserved exclusively for required recovery.
Selected seeds receive fixed, near-equal integer leases from the other 80%;
unused leases are not loaned to other starts. Every selected seed is scheduled
before any recovery. Zero seed allowances/deadline exhaustion leave explicit
unattempted IDs. Recovery cannot starve seed coverage. Even if recovery slots
are configured as zero, the reserved fraction is not silently reassigned.

Initialization has 512 model calls; validation 1024 (exhaustive 2048), final
assessment 32 and conditioning 256. These are additional to optimization, with
global call/component ceilings equal to the sum of phase limits and three times
that sum. Objective attempts/calculations, model calls, NNLS projections, charged
and completed component work, unknown partial failures and reported statistics
calculations are separate. Rejected/infeasible requests cost no physical call.
No hidden retries or optimization work occur in another phase.

Initialization has a 5%-wall lease, each selected seed a fixed share of 65%,
each reserved recovery a share of 20%. Validation/diagnostics/assessment use the
remaining global deadline and their own call caps. Unused wall shares are not
extensions of another solver's lease. Checks are cooperative before physical
calls; an in-flight call, NNLS or SciPy overhead can cross a deadline. Measured
overshoot is retained and acceptance then fails. Phase time plus explicit
overhead time accounts for total time through result assembly; it is not a
preemptive real-time guarantee.

All finite evaluations are observed, including stencils/probes. Diagnostic/search
improvements are explicitly promoted before validation. A validation improvement
above the declared absolute centered-NLL tolerance (0.001) can be promoted and
scheduled for required recovery, after all seeds. **The promoted point never
inherits its anchor's validation**. It needs its own completed validation and a
fresh exact-candidate assessment. Promotion is recorded. If recovery is exhausted,
the finite promoted point is returned with rejected assessment and the old anchor
still visibly recorded. A sub-tolerance better validation probe is retained as
`best_observed_*`; the validated anchor may remain selected, explicitly distinct
from the lowest observed objective. No optional continuation chains are present.

Fresh assessment recomputes the exact returned coordinates, total/component
curves, parameters and objective, with objective tolerances `1e-9 + 1e-12*|NLL|`.
Acceptance additionally requires completed initialization/protected coverage,
all selected seed attempts, any required recovery, candidate-specific validation
and no expired global deadline. A failed solver with valid independently checked
finite output can be accepted; a successful solver alone cannot. Conditioning is
computed once and attached only if it describes the same final candidate;
otherwise `candidate_changed_or_unavailable` is explicit. Missing conditioning
does not by itself imply numerical rejection or physical nonidentifiability.
All-zero observations have an exact analytic zero-count result, with no nonlinear
starts, but still require charged validation and fresh assessment.

## Nonexecuting development protocol and PR 2B gate

[`issue37_pr2_development_v1.json`](../../configs/issue37_pr2_development_v1.json)
is a versioned, nonexecuting draft. It contains D00–D17, exact inherited/overridden
conditions, 72 observation targets and 80 assumption-specific problems, paired
streams and work limits. Six eventual solver/policy configurations would mean
480 fits and 80 independent audits. No observations have been generated here.

Mandatory numerical correctness is separate from objective quality and component
recoverability. D00 and D01 define eight proposed control problems requiring
numerical acceptance, complete audits and absolute centered-NLL gap <=0.01.
The other 72 problems remain separately reported robustness strata, including the
nonregular zero-background D09 boundary. Signed gap
is production minus audit: positive >0.01 is production worse; negative <-0.01
is audit insufficiency, not production failure. Incomplete audit agreement is
unestablished even when its provisional gap is small. All missing/failed attempts
remain in denominators. 80/80 agreement is aspirational and descriptive, not the
sole eligibility criterion. Unresolved negative control gaps leave reference
eligibility unresolved. No eligible candidate means stop and review, not threshold
relaxation or silent budget expansion.

The proposed operational pilot has disjoint stream namespace, six problems,
12 fits and six audits, with shorter explicit caps (24 worker-minute ceiling).
It is not authorized/executed and cannot become untouched qualification evidence.
Actual bounded timings and incomplete/censored outcomes must be measured after
approval before requesting the comprehensive run. Its 78.67 worker-hour sum of
candidate ceilings is a ceiling, not a measured forecast. Deterministic test
durations from this checkpoint do not estimate full K=3 recovery capacity.

PR 2B must independently implement and test bounded TRF, including signed
Poisson-deviance residual derivatives at exact zero; review an independently
implemented direct-convolution TRF/Powell audit; validate objective consistency,
boundary behavior, completion rules and work leases; then obtain pilot/study
authorization. The 16 TRF/8 Powell/200,000 calls/900 seconds audit is a candidate,
not validated capacity. Truth and production estimates must not seed the audit.
Agreement is not proof of a global optimum. PR 3 retains its separate measurement
integration and independent, separately frozen qualification review gate with
new streams. The strict 100/100 target remains proposed, not frozen.

## Deterministic validation

The five new test files cover dedicated API/input limits, template arithmetic and
selection, scaled NNLS failure, permutations and exact zero components, nonsmooth
conditional solves on exact/approximately uniform grids, unavailable derivatives,
failed/nonfinite trials, finite-best retention, independently instrumented work,
leases/deadlines, seed-first recovery, explicit promotion, stale-validation
rejection, fresh canonical assessment, unavailable conditioning and manifest
arithmetic. A bounded 180-call actual L-BFGS-B test checks finite descent; controlled
solver stubs test adverse termination. These are software tests, not the numerical
development population. The existing PR-1 derivative oracles use complex steps
only on analytic amplitude/exponential operations and an explicit piecewise-linear
shift slope away from knots; no complex-step oracle crosses interpolation/clipping.

Run these tests and the four PR-1 foundation files, then existing forward-model,
Poisson, mono/bi numerical, public API and structural tests. Full pytest with
combined branch coverage, Ruff, the unchanged-debt formatter check, fresh
wheel/sdist and isolated-install checks remain required. Read-only historical
hash verification must not regenerate studies. Machine-specific validation
artifacts belong under ignored `build/issue37-pr2a/`, not historical records.
