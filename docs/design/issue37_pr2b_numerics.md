# Issue #37 PR 2B numerical infrastructure

Implementation and deterministic software validation only. PR 2A is preserved
at `56466959c9f2948bb71adee3abb75739c5a464b5`. No pilot, development comparison,
qualification, production policy selection or PR 3 integration is authorized.
Existing mono/bi implementations, physics, IRFs, root exports, SQLite schema v1,
release metadata and frozen historical evidence are unchanged.

## TRF contract

Explicit K=3 settings now accept `solver="trf"` or `"lbfgsb"` and an optional
`recovery_solver`. The latter switches only required recovery invocations; it
does not add continuations or extra starts. Omitting it retains the seed solver.
The PR 2A L-BFGS-B setting remains backward compatible, not a newly selected
production default. `settings=None` is still rejected.

`_poisson_residuals` computes signed residuals with sign(y-mu). For positive y,
near equality it evaluates the convergent series for
`h(d)=2*(d-log1p(d))/d**2`, d=(mu-y)/y, including h(0)=1:
`r=-d*sqrt(y)*sqrt(h)`, `dr/dmu=-1/(sqrt(y)*(1+d)*sqrt(h))`.
Away from equality it uses the log difference and centered objective. Square
roots are factored to avoid premature overflow. For y=0, mu>0,
`r=-sqrt(2)*sqrt(mu)` and `dr/dmu=-1/(sqrt(2)*sqrt(mu))`.
At y=mu=0 the objective and residual are zero but the expectation derivative is
unavailable. At y>0, mu=0 the objective/residual are infinite. Unrepresentable
scores/Jacobians remain unavailable. No expectation floor is introduced.

TRF composes this derivative with counted safeguarded model columns. The PR 1
derivative record exposes individual columns while retaining its original full
Jacobian/gradient availability semantics. A restricted solve never claims a
full Jacobian. Shift is fixed at nonsmooth knots/support jumps; required recovery
can fix explicit zero faces and their inactive lifetimes. A zero model slope
alone does not establish differentiability of sqrt(mu), so zero-expectation bins
conservatively abort TRF Jacobian construction. The zero-observation analytical
path and original boundary candidates remain available. Local validation can
probe feasible zero faces and neighboring shift intervals without gradients.

SciPy may move a boundary starting coordinate inward. Its first actual residual
point is recorded if changed; the exact submitted candidate remains retained.
Residual requests, Jacobian requests, model calls, charged component work and
solver status are distinct. Jacobian anchors are freshly evaluated and charged;
there is no hidden callback cache. All work uses the original phase ledger,
per-invocation leases and seed-first 80/20 optimization-call partition.
Every solver invocation occupies one start. Solver success alone cannot accept
a fit. Candidate promotion, validation and final assessment remain PR 2A's.
TRF uses SciPy APIs available since the supported 1.13 minimum, explicit x_scale
and analytic callback Jacobians; newer callback/workers options are not used.

## Independent audit

`benchmarks._triexponential_audit` imports only NumPy, SciPy and standard-library
records/time. It has its own data validation, coordinates (linear amplitudes and
background scaled by mean counts; physical lifetimes and shifts), deterministic
irrational-fraction seed bank, direct convolution, interpolation, stable centered
objective, bounded TRF/Powell invocations, work leases and local assessment.
No production coordinate, seed, callback, derivative or scheduler is imported.
Truth and production results are absent from its search interface.

The audit objective independently uses a symmetric ratio series near equality.
TRF uses independently differenced residuals at two step sizes, within the
actual interpolation interval, and refuses unavailable/unstable Jacobians.
Initial interpolation knots conservatively fix shift even if the knot happens
to have agreeing slopes. Powell is derivative free and may traverse knots.

The explicit candidate is 16 TRF starts, eight Powell starts, 200,000 total
direct-model calls and 900 seconds. Validation reserves 256 calls, fresh
assessment two; search receives the remainder in disjoint start leases, with
85% of time apportioned to starts. These are configurable candidate limits.
Every model request charges three component convolutions, including uncertain
partial work on failures. Finite evidence survives solver failure/exhaustion.
Local validation records coordinate probes and objective improvements at two
step scales; it does not promote a probe or transfer acceptance. Completion
requires every scheduled start, successful termination without exhaustion,
finite candidate, fresh consistency, completed passing local validation and an
unexpired global deadline. This conservative rule can reject a good candidate;
its effectiveness must be measured in the separately authorized pilot.

The result is sealed before cross-scoring. Cross-scoring evaluates each candidate
with both implementations and checks saved predictions/objectives. Its two
production and two direct calls (12 component convolutions) are separate from
sealed search budgets. The signed gap is production minus audit centered NLL.
Incomplete audits never yield established agreement. Negative gaps below -0.01
identify audit limitations; positive gaps above 0.01 identify a better audited
candidate. Cross-implementation inconsistency supersedes objective agreement.
Finite-search agreement does not prove global optimality or component recovery.

## Development runner and evidence

The v1 manifest is unchanged, including its nonexecuting flags and historical
`bounded_trf_deferred`/`implemented=false` planning labels. This document records
the software's implementation status without revising the scientific protocol.
Any scientific/settings/population change still requires a versioned manifest.

`benchmarks.triexponential` resolves 72 observation identities, 80 assumption
problems, eight controls and 72 robustness problems; six solver/policy pairs
give 480 fits plus 80 audits. Construction is explicit and separated from task
inventory. Matched/misspecified IRFs retain paired observation hashes. Missing,
failed, interrupted and externally timed-out tasks remain in all-attempt reports.
Numerical rejection reasons, signed gaps, runtime distributions, work/recovery
counts and component recovery contributors are separate. Absent/equal components,
mono/bi generation and IRF mismatch do not receive invented unique K=3 truth.

`scripts/develop_triexponential.py inspect` performs no generation or fitting.
`run` requires an external lock with explicit execution authorization, maintainer
review identifier, and an exact fingerprint of source, manifest, environment
and scope. It prepares and hashes all inputs before fitting. JSON checkpoints
have immutable task identities, execution-lock hashes and payload hashes;
atomic replacement publishes results. Resume skips existing attempts, including
failures/interruption; no hidden retries occur. A revised/repeated run needs a
new reviewed directory. SQLite is untouched. `report` cross-scores sealed saved
results without changing search checkpoints or initiating an optimizer.
Reporting verifies the same manifest/source/environment fingerprint before
cross-scoring. The fingerprint includes thread-control environment variables;
set and record them before issuing a lock, and preserve them on resume.
Interrupted `running` records also carry verified payload hashes.

Each worker also has a parent-enforced timeout including interpreter startup.
On Windows timeout termination covers its process tree. A killed worker reports
unknown partial work, not zero calls or successful convergence. Internal timers
are cooperative; external timeout enforcement/OS scheduling overhead is separate.
Tree termination and pipe cleanup each have a five-second timeout. Unconfirmed
cleanup is saved as a censored timeout and stops further execution, including
resume. It requires manual process reconciliation and a new reviewed run, not
automatic retry. The 24-minute search ceiling excludes up to ten seconds of
cleanup per timed-out worker; OS termination latency is not a numerical guarantee.
No source code, tests, import or installation automatically executes this runner.

## Proposed pilot, not executed

The unchanged candidate pilot uses P00/P01/P02, one deterministic target and one
Poisson observation each: six problems, two solvers with compact start layout,
12 fits and six independent audits, 18 tasks. Streams use
`[37,2,20261101,0,condition,replicate]`, disjoint from comprehensive stream 1.

Each fit: at most 16 seed plus four recovery invocations, 20,000 optimization
model calls (4,000 recovery-reserved), 512 initialization, 1,024 validation,
32 assessment and 256 diagnostics: 21,824 model calls, 65,472 charged component
convolutions, 60 seconds per worker. Each audit: 16 TRF/eight Powell starts,
40,000 total direct calls, 120 seconds per worker. Twelve cross-scores add 48
model calls/144 component convolutions outside search. The optimization-worker
ceiling is 1,440 seconds (24 minutes), excluding input preparation, reporting
and process cleanup. Record actual/censored costs before requesting the full
560-task development comparison; no measured runtime forecast exists yet.

PR 2B does not freeze a production policy, audit capacity or future 100/100
qualification criterion. Pilot authorization and review remain the next gate.

## Final checkpoint review

Deterministic adversarial review found and corrected intermediate overflow in
the new residual/audit objective calculations at representable floating-point
extremes, including equal counts near 1e308. Decimal objective/derivative oracles
cover these cases. An unavailable/nonfinite residual never claims an available
derivative. This does not alter the existing physics or Poisson-statistics code.
Checkpoint verification now covers interrupted records as well as completed
ones; changing a record's status cannot bypass its integrity hash. Reporting
rejects a mismatched source/manifest/environment lock before cross-scoring.
Numerical solver exceptions retain finite evidence and failed invocation records.
Timeout cleanup is bounded and an unconfirmed process termination stops the run.
These corrections use software fixtures only, with no pilot/development draws.

## Two-stage input integrity correction

The original pending proposal
`docs/validation/issue37_pr2_operational_pilot_v1_execution_lock_proposal.json`
(SHA-256 `a12ce4d42f13242d910544d128f97791b219998ec9e5904d4fa50f38376626db`)
and its preflight report remain unchanged historical planning artifacts. They
cannot authorize the corrected runner. No pilot inputs have been generated.

Execution now uses the explicit `issue37-development-execution-v2` schema.
One authoritative fingerprint enforces all production Python files, the runner,
`pyproject.toml`, exact manifest bytes and canonical contents, the source commit
and confirmation that these sources are committed. It also enforces Python and
interpreter identity, OS, installed dependency versions, relevant Python/thread
environment controls, loaded numerical-library binary hashes and effective
thread counts, and package/module origins. Resolved conditions, observation
identities, RNG construction, settings, task order, budgets and deadlines are
part of that same fingerprint. Settings resolution is shared with execution;
this correction changes no numerical settings or algorithms. Any mismatch is
fatal before preparation, worker execution, resume or verified reporting.

Timestamps, explanatory provenance, review notes and aggregate display counts
outside `fingerprint` are informational. Authorization still requires a true
`execution_authorized` field and a real, nonempty `maintainer_review`; the whole
approved lock is also hashed into the evidence. `inspect` is metadata-only and
emits a pending lock. A proposal made from this uncommitted correction records
`committed_sources=false`, and cannot execute even if its approval flag is flipped.
After a separately approved commit, regenerate and review the lock against the
new commit before authorizing execution. Old-schema locks fail closed.

The commands are deliberately separate:

1. `prepare --pilot --directory RUN --execution-lock APPROVED`: requires a fresh
   directory. Writes the approved lock and a durable preparation-start marker,
   prepares every required input exactly once, saves hashed input checkpoints,
   and atomically publishes `input_inventory.json` only after all inputs pass.
   It re-reads the seal and every checkpoint before returning. It never launches
   a fit or audit. An exception/interruption leaves evidence and an incomplete
   preparation; neither `prepare` nor `run` retries it in that directory.
2. `run --pilot --directory RUN --execution-lock APPROVED`: generates no inputs.
   It verifies the current execution fingerprint, seal, every input, complete
   task mappings and all existing attempts before starting workers. The worker
   repeats these checks before accessing its input and publishing its result.
   Missing/corrupted sealed inputs are fatal, never reconstructed from a seed.
3. `report --pilot --directory RUN --execution-lock APPROVED`: requires the same
   verified provenance. It performs no fitting. Missing numerical outcomes stay
   in the planned denominator, but unverified input/execution provenance blocks
   cross-scoring and the report. Save report output outside the run's checkpoint
   directory; unknown JSON evidence inside that directory is rejected.

Each sealed entry binds problem/observation/assumed-IRF identity, deterministic
versus raw-count kind, shape/dtype and little-endian float64 hashes of grid,
observation, assumed/generating kernels and mean, truth metadata, input filename,
raw checkpoint hash and canonical payload hash. The inventory binds the complete
task-to-input map, approved lock, fingerprint and manifest. Its independently
verifiable canonical payload hash is the inventory identity. Paired assumptions
must have identical observation/grid/generating/truth identities. Declared array
hashes are recomputed from saved arrays, not trusted as labels.

Per-task immutable start receipts prevent a deleted result from being treated
as a never-attempted task. Worker start receipts permit only one worker per
attempt. Running, exception, timeout and completed attempts are never replaced
on resume. A start receipt with a missing checkpoint is reported as interrupted;
running/interrupted attempts block new workers until termination is reconciled
under separate controlled authorization. There is no automatic recovery command.
Reports may still describe these verified incomplete attempts. Unknown, duplicate
or conflicting evidence and receipts fail verification.

Task checkpoints and worker transports bind the same execution-lock hash,
inventory hash and exact input checkpoint. Worker transport is verified before
publication as a result and rechecked on every resume and report, including
transports retained beside terminal attempts. Derived cross-scores/recovery summaries are separate
from original hashed result payloads. Signed gaps, censored time and unknown
partial work remain explicit. SHA-256 checks detect inconsistent evidence;
they are integrity records, not signatures against an actor rewriting an entire
evidence directory. The workflow remains single-writer and serial.

All correction tests use tiny fixed software arrays. The six pilot problems,
D00-D17 manifest, namespaces, settings, 24-minute worker ceiling and bounded
cleanup procedure remain unchanged. Source/environment verification overhead
is distinct from physical-model work; external worker deadlines include startup
and worker verification. Operational cost remains unmeasured until separately
authorized execution. No production default or qualification claim is implied.
