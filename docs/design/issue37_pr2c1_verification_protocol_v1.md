# PR 2C-1 numerical verification v1 â€” pending protocol

Study: `issue37_pr2c1_numerical_verification_v1`.
Numerical checkpoint: `903a31921624c6ab258ec1bac68de82bf48d3a8c`.
The [complete manifest](../../configs/issue37_pr2c1_numerical_verification_v1.json)
is a nonexecuting proposal with `execution_authorized=false`. It is not an
execution lock. No inputs, kernels or results have been generated for this study.
The fit-only infrastructure and protocol are pending maintainer review and a new
source commit. Execution remains unauthorized.

## Purpose and scientific overlap

Evaluate near-boundary derivative construction, failure instrumentation and
final-candidate diagnostics with the original compact settings. This is numerical
development, not production-policy selection, an independent audit, qualification
or PR 2C-2. No optimizer or audit policy is changed by this proposal.

The three scientific conditions intentionally match the inspected original
pilot's P00/P01/P02 definitions. Their deterministic targets are therefore reused
controls in scientific terms, even though future preparation will create separate
checkpoints under a new identity. They are not untouched evidence. New Poisson
draws use a separate random stream and are not paired before/after software tests.
Original pilot evidence, including its 0/6 complete audits, remains immutable.

## Conditions and input construction

All time units are ns. For each condition, use `t = arange(bins) * 0.03125`.
Lifetime bounds are `[0.1, 10]`; shared-shift bounds are `[-0.3, 0.3]`.
Detected-window signal fractions are `[0.25, 0.45, 0.30]`.

| Condition | Pilot analogue | Bins / final time | Lifetimes | Signal photons | Background per bin | Shift | Generating/assumed IRF |
|---|---|---|---|---:|---:|---:|---|
| V00 | P00 | 256 / 7.96875 | 0.6, 2, 4.8 | 20,000 | 0.5 | +0.043 | Gaussian / matched |
| V01 | P01 | 256 / 7.96875 | 1, 1.2, 3 | 5,000 | 1 | -0.043 | EMG / matched |
| V02 | P02 | 128 / 3.96875 | 0.6, 2, 4.8 | 2,000 | 0 | 0 | Sampled with nonzero endpoints / matched |

The generating pre-exponential amplitudes copied from the sealed historical
generating-parameter records are:

| Condition | A1 | A2 | A3 |
|---|---:|---:|---:|
| V00 | 253.75289213587928 | 143.54246275610197 | 50.25038078226651 |
| V01 | 38.50094632228498 | 58.03373958171306 | 17.295449818881043 |
| V02 | 26.213845249683697 | 18.69393821313789 | 8.70618886723992 |

These are generating parameters, never optimizer starts. The authoritative
construction is `A[j] = signal_photons * fraction[j] / sum(unit_component[j])`
using the shifted generating kernel and finite observation window. Future
preparation must verify that it reproduces the declared binary64 amplitudes;
a discrepancy stops preparation rather than silently updating this protocol.

IRFs are explicit synthetic assumptions:

- Gaussian: center 0.8, FWHM 0.18, sigma `0.18 / sqrt(8*log(2))`.
- EMG: Gaussian center 0.8, Gaussian FWHM 0.18, exponential tail 0.35;
  the existing `scipy.stats.exponnorm.pdf` convention has shape `tail/sigma`,
  location `center` and scale `sigma`.
- Sampled: linear interpolation through times
  `[0, 0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.75, 3.96875]` and values
  `[0, 0, 0.15, 1, 0.7, 0.35, 0.08, 0, 0]`, then add 0.02 to every target-grid
  value before normalization. This is a synthetic sampled fixture, not measured
  data. Its nonzero endpoint values preserve the support discontinuity at shift 0.

Normalize each generated kernel by the existing trapezoidal area on its actual
target grid. Do not zero endpoints or alter the unchanged `shift_irf` semantics:
positive shift moves the IRF later; extrapolation is zero, with no shift-dependent
renormalization. Retain bin-width convolution scaling, finite-window truncation
and one post-convolution background. The matched assumed kernel equals the
generating kernel byte-for-byte but keeps its separate assumption identity.

## Six problems and twelve tasks

| Observation ID | Kind | RNG entropy | Problem ID |
|---|---|---|---|
| V00:mean | deterministic expectation | none | V00:mean:matched |
| V00:poisson-0 | raw counts | `[37,2,20261101,2,0,0]` | V00:poisson-0:matched |
| V01:mean | deterministic expectation | none | V01:mean:matched |
| V01:poisson-0 | raw counts | `[37,2,20261101,2,1,0]` | V01:poisson-0:matched |
| V02:mean | deterministic expectation | none | V02:mean:matched |
| V02:poisson-0 | raw counts | `[37,2,20261101,2,2,0]` | V02:poisson-0:matched |

Use `Generator(PCG64(SeedSequence(entropy))).poisson(mean)` exactly once per
Poisson observation. Branch 0 remains the original pilot and branch 1 the
comprehensive-development namespace. Branch 2 is this proposed verification.
Each observation and kernel identity is additionally namespaced by the new
study ID. Deterministic targets use the existing explicit deterministic-objective
path and are never described as observed raw counts.

Each problem receives `lbfgsb:compact` then `trf:compact`, using the identical
sealed input checkpoint. There are exactly 12 tasks and no audit tasks. The JSON
lists every task ID, input mapping and immutable SHA-256-derived filename.
Initial guesses are absent; initialization uses the unchanged data-only template
bank and NNLS screening. Generating truth and old estimates must not initialize
any fit. The original pilot directory is never an input store for the new run.

## Unchanged fit settings and bounded work

The manifest copies both production-setting records exactly from the original
approved lock. Each fit has 16 planned seed starts and four reserved recovery
starts, with a total limit of 20 invocations. Optimization has 20,000 calls:
16,000 for seeds and 4,000 reserved for recovery. Other phase ceilings are 512
initialization, 1,024 validation, 32 assessment and 256 diagnostic model calls.
That is at most 21,824 model calls and 65,472 charged K=3 component convolutions
per fit. NNLS work, actual residual/Jacobian evaluations and failed probes remain
separately visible. Optional continuation chains and automatic retries are disabled.

The seed call pool is divided by actual selected seed count using the existing
quotient/remainder rule (1,000 calls each when 16 are selected); seed time leases
are `0.65*60/n_selected` (2.4375 seconds for 16). Recovery has four 1,000-call,
three-second leases. Global phase and deadline checks still apply. Unused recovery
capacity is not loaned to seeds; incomplete required coverage remains a failure.

Unchanged numerical settings include max iterations 1,000, max line search 40,
`ftol=1e-12`, `gtol=1e-6`, derivative step `1e-4`, derivative relative/absolute
tolerances `2e-4`/`1e-6`, three reductions, validation steps `[1e-4, 1e-2]`,
and descent tolerance 0.001. Fresh objective consistency keeps absolute/relative
tolerances `1e-9`/`1e-12`. Recovery uses the same solver as its fit.

Run one worker at a time with a 60-second external deadline and unchanged
60-second cooperative numerical limit. Startup and worker-side integrity checking
are inside the external deadline. Use the existing Windows process-tree procedure:
five seconds for termination and five for cleanup. Stop if termination cannot be
confirmed; never launch a replacement automatically.

Across 12 fits: at most 240 solver invocations, 261,888 fit model calls, 785,664
charged component convolutions, 720 worker-seconds, plus at most 120 cleanup-seconds.
The 840-second sum excludes preparation, coordinator work and reporting; record
those separately rather than implying an end-to-end ceiling.

The explicit verification scope separately counts preparation of at most 18 K=1 unit-model
calls (six inputs, three components each) and reporting of at most 15 K=3 model
calls (12 returned predictions and three deterministic generating references).
These are implemented non-optimization work caps, not permission to execute the
study. Calls are charged before physical evaluation, including failing calls;
completed and unknown partial component work remain separate. Failed preparation
retains its work record. Each report invocation has its own 15-call ceiling;
repeated reporting is additional explicitly invoked work, not a new fit attempt.

## Execution lock, sealing and resume

First approve this protocol, then prepare a separate execution lock. Its enforced
identity must include exact committed execution HEAD, all numerical and invoked
execution/reporting sources, `pyproject.toml`, manifest bytes and canonical digest,
conditions, RNG procedure, settings, complete task inventory and output location.
Enforce Python/executable, dependency versions, module origin hashes, OS/machine,
loaded numerical-library binary identities and actual thread counts. Timestamps
and human review prose are descriptive, not substitutes for enforced fields.

Set OMP, OpenBLAS, MKL, VECLIB, NUMEXPR and BLIS thread counts to one;
`OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`, `PYTHONHASHSEED=0`; leave
`OPENBLAS_CORETYPE`, `PYTHONPATH` and `PYTHONHOME` unset. Record and enforce them.
Any executable-source change requires a new reviewed, committed execution HEAD
and lock. Neither an old lock nor a changed authorization Boolean suffices.

Only after explicit execution approval may preparation create
`data/generated/issue37_pr2c1_numerical_verification_v1`. It must write a durable
preparation marker, prepare all six inputs, atomically seal `input_inventory.json`,
then re-read every input/hash/mapping before any worker starts. Bind attempts and
worker transports to the approved lock, authoritative fingerprint, manifest,
inventory identity and exact input payload. Reports go outside the checkpoint
directory under the new study prefix.

Resume verifies all identities, inputs and attempts before running genuinely
missing tasks. Missing/corrupted inputs must never be regenerated. Unsealed
preparation stops explicitly. Failed, timed-out, interrupted or completed attempts
cannot be overwritten or silently retried. Invalid provenance blocks verified
reporting; missing numerical results still remain in planned denominators.

## Reporting and interpretation

Retain all 12 planned attempts, six per solver, six deterministic and six Poisson.
Report accepted, rejected, failed, timed-out, interrupted, not-attempted and
invalid-provenance outcomes separately. Runtime summaries include censored
observations and explicit unknown partial work rather than fabricated zeros.

For every available result report numerical reasons, exact candidate identity,
fresh consistency, validation identity/completeness, required recovery completion,
promotions, returned and best-finite objectives/parameters, full/restricted
conditioning and derivative/Jacobian availability. Include failure coordinates,
bounds, actual stencil steps and geometry. Bounded failure records can omit older
events; report their total and retained counts, not a claim of complete tracing.

Separate joint and conditional invocations using their free/fixed coordinates.
Report each solver's success/termination and failed conditional solves, but never
equate a restricted solver success with numerical acceptance of the full fit.
Show seed coverage, recovery usage/exhaustion, all phase/model/component work,
numerical time, external worker time, cleanup, preparation and reporting time.

For deterministic targets compute the signed, unclipped gap
`C(y_mean, mu_returned) - C(y_mean, mu_generating)`, and also the best-finite gap.
Record the computed generating objective rather than forcing floating-point zero.
This is a generating-reference comparison, not an independent optimizer audit.
For each shared input report signed `C_LBFGSB - C_TRF` alongside both outcomes.

Report canonical ordered lifetime/amplitude errors where meaningful, background
absolute error (relative only for nonzero truth), shift error in ns and component
fractions. Flag undefined or nonidentifiable associations. Show all-finite and
accepted-only summaries with their denominators. A close objective or warning-free
fit does not establish component recoverability.

Compare deterministic outcomes with the original pilot only after checking target,
grid and kernel hashes, and disclose the reused controls. New Poisson draws permit
descriptive comparisons only. Original audit completeness remains 0/6; new audit
status is explicitly `not_applicable` (not scheduled by this scope). No 12/12 acceptance target, policy selection, scientific
qualification target, automatic tuning, data exclusion or budget increase is set.

## Implemented infrastructure and review handoff

The existing `scripts/develop_triexponential.py` now accepts `--verification`,
mutually exclusive with `--pilot`. The new private
`benchmarks._triexponential_verification` validates the exact versioned scientific,
settings and task contract; it does not expose a generic audit-removal option.
The old pilot still resolves 18 tasks and comprehensive development 560 tasks.
The manifest's scientific contract hash is
`f53733d261f3cee8fa0ffe1af79455897b186335a709b9b7b1cc3d7d1b842f4b`.
Metadata-only regression checks compare all physical parameters and IRF
specifications with the original pilot. The generating functions are identical
mathematically. Exact new array-byte identity remains to be checked during future
preparation/reporting; no V00-V02 arrays were created during infrastructure work.

The manifest includes read-only historical metadata (array and result hashes,
acceptance, objective, recovery, conditioning and work) extracted after verifying
all 81 original checkpoint files against the pilot report receipt. It is bound by
the new manifest hash, never used to supply missing physical parameters or starts.
Reporting compares actual saved input hashes before calculating a deterministic
before/after objective difference. Independent Poisson draws receive only
descriptive comparisons. Reused controls cannot become untouched qualification
observations.

All source Python files, the runner and `pyproject.toml`, manifest raw/canonical
hashes, environment, resolved settings, tasks and the output directory are enforced
by the existing execution fingerprint. Human review prose is informational.
Preparation and reports reuse `DevelopmentStore`'s immutable input and attempt
checks. A numerical attempt must also match its immutable worker transport before
reporting; only the parent's separately measured external timing is additional.
Failed, timed-out and missing outcomes remain in planned denominators. Numerical
runtime summaries containing timeout lower bounds are descriptive, not uncensored
performance estimates. External timing includes process startup and cleanup, not
CPU time. A process termination can leave partial numerical work unknown.

After review, commit only this infrastructure, manifest, protocol and approved
inventory/test updates. Do not include the existing untracked historical pilot
artifacts opportunistically. The current numerical checkpoint is the baseline,
not a valid execution HEAD for this changed infrastructure. Following the future
commit, regenerate a pending lock from the exact committed source and intended
environment, review it, then obtain separate explicit execution authorization.
A pending proposal (`execution_authorized=false`) is rejected by the gate.
No old lock can be reused and no output directory is created by `inspect`.

The following PowerShell commands describe the supported future workflow. **They
are not execution authorization and have not been run for the study.** Run from
the repository root with the reviewed interpreter and no concurrent workers:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:OMP_DYNAMIC = 'FALSE'
$env:MKL_DYNAMIC = 'FALSE'
$env:PYTHONHASHSEED = '0'
Remove-Item Env:OPENBLAS_CORETYPE, Env:PYTHONPATH, Env:PYTHONHOME -ErrorAction SilentlyContinue
$verificationPython = '.venv\Scripts\python.exe'
$verificationManifest = 'configs/issue37_pr2c1_numerical_verification_v1.json'
$verificationProposal = 'docs/validation/issue37_pr2c1_numerical_verification_v1_execution_lock_proposal.json'
$verificationLock = 'docs/validation/issue37_pr2c1_numerical_verification_v1_execution_lock_approved.json'
$verificationDirectory = 'data/generated/issue37_pr2c1_numerical_verification_v1'
# After the future source commit: metadata only; creates a PENDING proposal.
& $verificationPython scripts/develop_triexponential.py inspect --verification --manifest $verificationManifest | Set-Content -Encoding utf8 $verificationProposal
# Only after separate review and execution authorization, using the new approved lock:
& $verificationPython scripts/develop_triexponential.py prepare --verification --manifest $verificationManifest --directory $verificationDirectory --execution-lock $verificationLock
& $verificationPython scripts/develop_triexponential.py run --verification --manifest $verificationManifest --directory $verificationDirectory --execution-lock $verificationLock
& $verificationPython scripts/develop_triexponential.py report --verification --manifest $verificationManifest --directory $verificationDirectory --execution-lock $verificationLock | Set-Content -Encoding utf8 docs/validation/issue37_pr2c1_numerical_verification_v1_results.json
```

Use UTF-8 without BOM (PowerShell 7 `-Encoding utf8`) and check each process exit
code before continuing. Preserve the pending proposal; the approved lock must
retain the identical fingerprint with explicit authorization and maintainer review.
Fresh preparation saves six `sha256("input:" + problem_id).json` checkpoints,
`preparation_started.json` and sealed `input_inventory.json`. Twelve task payloads
use `sha256(task_id).json`, with durable `.started.json`, `.worker_started.json`
and `.worker.json` records as applicable. The exact filenames are in the manifest.
Reinvoke only `run` for an explicitly requested resume: it verifies every identity
and skips existing attempts. Never rerun `prepare` or delete receipts to recover.
Unconfirmed cleanup or interrupted/running attempts stop further workers pending
separate investigation. Reports are outside the checkpoint directory; preserve
prior reports rather than overwriting evidence on a later invocation.

The new full source/environment fingerprints must be resolved after the future
commit in the final execution environment. No current uncommitted fingerprint,
passing software test, or completed packaging check authorizes study execution.
