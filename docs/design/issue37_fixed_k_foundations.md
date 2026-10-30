# Issue #37 PR 1 — private fixed-K numerical foundations

Status: implemented for review; no public K=3 fitter and no scientific
qualification claim. The active [Issue #37](https://github.com/moryev/tcspc-lifetime-toolkit/issues/37)
and the maintainer's PR-1 authorization define the scope. K=1/2/3/4 tests exercise
dimensional contracts; they do not establish unrestricted arbitrary-K fitting.

## Boundaries and dependencies

`estimation._fixed_k` owns immutable parameter/input/evaluation records and
coordinates. `_fixed_k_objective` composes those records with the unchanged
general-K component forward model and stable Poisson deviance. It owns explicit
work limits and derivative safeguards. `_fixed_k_diagnostics` uses the counted
evaluator for local probes, fresh candidate checks and local sensitivity.

There is no optimizer, public result class, seed bank, recovery scheduler,
measurement adapter, covariance estimator or model-order selector in these
modules. Existing mono/bi implementations do not import them. No root or
subpackage initializer exports were added. The ordered 66-name root API,
SQLite schema v1, release metadata and old scientific records remain unchanged.
Current production inventory: 50 substantive modules, 12 support modules and
5 retained package APIs (67 Python files).

## Physical and coordinate contracts

For K components, the 2K+2 coordinates are
`[A_1/s_A, ..., A_K/s_A, log(tau_1)-m, ..., log(tau_K)-m, B/s_B, (shift-c)/r]`.
Here `m` is the midpoint of the log-lifetime bounds; `c,r` are the shift midpoint
and half-range. Scales and finite increasing lifetime/shift bounds are explicit
inputs. PR 1 does not choose count-based scales or a production policy. All
lifetimes use the same positive interval. Linear coordinates have lower bound
zero and no artificial upper ceiling; shift coordinates range from -1 to 1.
Unrepresentable transformations fail explicitly, including positive linear
values that would underflow to zero. Exact physical endpoints are retained.

Amplitudes are pre-exponential scales, not photon totals. B is one background
count per bin, added after summing components. Zero amplitudes/background and
equal lifetimes are legal. Coordinates retain submitted component labels.
Canonicalization is an explicit separate operation: increasing lifetime, then
decreasing amplitude for exact lifetime ties, with a returned row permutation.
It is deterministic but does not resolve physical permutation nonidentifiability.
An optimizer must not sort inside its objective or derivative calculation.

Inputs are copied read-only. PreparedIRF source/provenance and preparation
diagnostics are snapshotted; arrays must already have unit trapezoidal area.
There is no implicit registration, normalization, grid change or time rebasing.
The existing positive-shift convention, zero extrapolation, convolution dt
factor, FFT roundoff clamp and finite-window truncation remain authoritative.
Approximately uniform grids accepted by the existing validator remain accepted.

Targets explicitly declare raw counts or deterministic expected counts. Raw
counts must be integer-valued below 2**53. This internal assertion cannot detect
processed measurements; PR 3 must retain the canonical raw-measurement guard.
The objective is the existing stable deviance divided by two, including its
exact-zero limits: (y,mu)=(0,0) contributes zero; y>0,mu=0 is infinite. There is
no expectation floor. Invalid numerical output and a mathematically infinite
objective have distinct statuses.

## Work and failure accounting

`_WorkLimits` requires explicit per-phase model-call limits, global model-call
and objective-attempt ceilings, a component-convolution ceiling, and elapsed
wall seconds. Phases are initialization, optimization, validation, assessment
and diagnostics. No production limits are selected by PR 1.

The ledger distinguishes evaluation requests, reserved objective attempts,
actually computed Poisson objectives, physical model calls, charged component
convolutions and known completed component convolutions. One successful call
to the existing K-component model costs K convolutions, even for zero amplitudes.
The derivative's unit-amplitude basis is a charged model call with no Poisson
objective. Every finite-difference probe uses the counted evaluator. Reuse of
an already evaluated point within a single stencil refinement incurs no new
physical work. The ledger records per-phase counts and blocked-work reasons.
An infeasible coordinate request uses no physical call. Failed physical calls
consume their reserved K-work ceiling; unknown partial completion is recorded
separately, not reported as K successfully completed convolutions.

Wall limits are cooperative: checked before each physical call and exposed
afterwards. A running NumPy/SciPy call cannot be preempted. A finite result
completed after the deadline remains available, while validation/assessment
cannot report completed acceptance after a deadline. Time includes the whole
problem lifetime, including caller overhead. No retries occur in evaluation;
derivative reductions are explicit, bounded by their policy and charged.
Programming errors propagate; expected floating-point failures are records.

The best finite search evaluation is retained with its coordinates, physical
parameters, curve, components, objective, origin and phase. Only initialization
and optimization evaluations participate in this retention. Validation,
assessment and diagnostics never silently select a new candidate. There are no
planned/reserved/total-start counters yet: PR 2 must add and distinguish them
from the forward-model, component-work and wall-time accounting defined here.

## Derivative contract and interpolation geometry

Amplitude and background derivatives use the exact linear model basis and the
Poisson score, with score=1 for y=0 even at mu=0. Zero-amplitude lifetime
derivatives and zero-signal shift partial slopes are exactly zero. At a support
discontinuity, however, simultaneous amplitude/shift changes can have different
directional limits even at zero signal: the full derivative remains unavailable.
Other lifetime
and shift columns use feasible three-point one-sided differences at h and h/2,
checked for finite values, representability and agreement under explicit
absolute/relative tolerances. Bounded reductions are numerical safeguards,
not optimizer recovery. Left/right objective directional slopes are retained
separately. Failed estimates are `None` with a reason, never fabricated zeros.

Shift geometry uses `time-shift` and the actual interpolation abscissae, not
rounded shift/dt. It distinguishes smooth regions, continuous knots with
identical one-sided IRF slopes, continuous knots with unequal slopes, and
support discontinuities caused by nonzero endpoint samples extrapolated to
zero. Stencils stop short of the next actual knot. At a support discontinuity,
the side that jumps away from the point is not differenced against its value;
the continuous side may retain a meaningful directional derivative. If both
endpoint samples jump, neither side is available. No full derivative is claimed
at such a discontinuity. Continuous IRF kinks conservatively withhold a full
shift derivative while preserving usable sides, even if an objective-level
cancellation could make a particular scalar directional score agree.

Local weighted-Jacobian availability is distinct from a finite objective and
from numerical optimization success. Extremely short interpolation regions,
roundoff-dominated differences and failed stencils explicitly remain unavailable.
PR 2 must handle this availability contract rather than send dummy gradients
to a solver. The existing `shift_irf` implementation is untouched.

## Validation, consistency and structural diagnostics

Local validation records the exact submitted evaluation and objective,
tolerance, direction vectors, signed step sizes, infeasible skips, attempted
evaluations, best probe, best detected improvement, completeness and pass
(`None` if incomplete). Directions are all coordinate axes and all component
pairs' amplitude-transfer and log-lifetime-splitting directions. This is a finite
local search, not a proof of stationarity or global optimality.

A better probe does not inherit acceptance and is never promoted in PR 1.
Fresh assessment recomputes the exact candidate supplied to it and compares
parameters, component curves, total expectation and signed objective gap
(fresh minus submitted). A validation record applies only to its original
candidate object. Replacing or promoting a point requires its own validation.
Different-problem records cannot inherit acceptance. Numerical consistency,
local-validation success and solver termination are separate facts; no solver
termination flag or overall fit-success policy is implemented here.

Component diagnostics report nominal p=2K+2, exact zero amplitudes, all equal-tau
pairs and lifetime ratios, and separately amplitude fractions, nominal A*tau
infinite-window area fractions and actual finite-window signal fractions.
Zero signal has undefined fractions, not invented zero or uniform fractions.
Local sensitivity uses a Poisson-weighted Jacobian in the declared scaled
coordinates. Rank tolerance includes machine precision and measured stencil
disagreement; singular values and threshold are reported. Rank is bounded by
both bins and parameters, and deficient condition number is infinite. Zero
expectations, unavailable derivatives, nonfinite weights and SVD failures have
explicit statuses. These are descriptive structural/conditioning diagnostics,
not physical identifiability, component recoverability or uncertainty intervals.
Active coordinate bounds and one-sided stencil statuses and actual step sizes
remain inspectable. A deadline crossed during SVD is reported even when the
completed singular-value/rank facts can be retained.

## Tests and review gate

The four `tests/test_fixed_k_*.py` files cover K=1/2/3/4, permutations, physical
boundaries, direct-convolution and complex-step derivative oracles, exact-zero
Poisson limits and a high-precision near-equality objective, actual-grid knot
geometry, failed derivatives, phase budgets, deadlines, candidate identity,
fresh consistency checks, degeneracy and rank. Tests are deterministic and
small; none performs K=3 multi-start fitting or repeated-Poisson qualification.
Existing physics/statistics, mono/bi regressions, API/inventory tests, full
coverage, Ruff/formatting, fresh artifacts and isolated installation remain
required technical gates. Historical locks are checked read-only.

## PR 2 handoff and exclusions

PR 2 owns a dedicated public K=3 fitter and structured fit result, count-based
scale selection, seed generation/screening, planned seed starts, separately
reserved recovery starts and total starts, full multi-start orchestration,
recovery scheduling and solver adapters. It must explicitly validate any
promoted probe and assess the actual returned candidate. Compact, balanced and
exhaustive policies remain development candidates; no default is frozen here.
Measure L-BFGS-B/TRF alternatives, derivative-unavailable handling, runtime,
forward calls and component work on fresh development data. Independent
TRF/Powell audit methods and their work limits also require measured validation.

PR 3 owns measurement integration and separately predeclared scientific
qualification. Report audit completeness, signed objective gaps with stated
orientation, numerical acceptance, component recovery and all-attempt
denominators. 100/100 independent agreement remains a proposed strict target,
not a criterion frozen before numerical development. Numerical agreement,
structural recoverability and conditional parameter uncertainty stay separate.

No mono/bi refactor, unrestricted arbitrary-K promise, automatic model selection,
component confidence intervals, Bayesian/ML/CLI integration, persistence change,
historical replay for tuning or Issue-33 recovery-capacity work is included.
Issue #28's failed 119/121 and Issue #33's failed 99/100 audit outcomes remain
immutable. Issue #33 is open and deferred and does not block this development.
