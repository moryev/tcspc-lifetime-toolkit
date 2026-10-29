# Issue #33 Stage 3 execution infrastructure v1

Scientific design approved; infrastructure awaiting final review. **Scientific
execution remains disabled.** The candidate protocol and its earlier review lock
are retained separately from the final execution lock. No Stage 3 qualification
observation may be generated until separate authorization names that exact lock.

The unchanged [protocol](../configs/issue33_stage3_qualification_v1.json) and
[design rationale](issue33_stage3_protocol_v1.md) define the scientific study.
This document specifies its implementation, execution safeguards and validation.
The corrected production fitter remains commit
`5be88edda17ad6612a8447dca0e11a6543502b85`, with Stage 1 separately committed as
`3bff378fd7a41a6c46ff3bf21696cb101b8402bb`. No Stage 3 implementation commit is
made by this work.

## Frozen populations and gates

| Original condition | Difference from original baseline | New observations | Bi fits | Audits |
|---|---|---:|---:|---:|
| C00 | 1,000,000 signal photons | 100 | 100 | 100 |
| C04 | 100 signal photons; background still 2/bin | 10 | 10 | 3 |
| C05 | True mono, 2 ns; 1,000,000 signal photons | 10 | 10 | 3 |
| C08 | Lifetimes 2 and 2.1 ns | 10 | 10 | 3 |
| C09 | Fractions 1 and 0 | 10 | 10 | 3 |
| C10 | Fractions .995 and .005 | 10 | 10 | 3 |
| C13 | Exactly zero background | 10 | 10 | 3 |
| C16 | Shift -.3 ns | 10 | 10 | 3 |
| C17 | Shift +.3 ns | 10 | 10 | 3 |
| C21 | First 103 bins of original grid; short window | 10 | 10 | 3 |
| C26 | EMG generating IRF; matched and Gaussian-.3 arms | 10 | 20 | 6 |
| **Total** | **11 conditions, 12 condition/arm groups** | **200** | **210** | **133** |

The baseline is 512 bins over 0–20 ns, lifetimes .8/3.2 ns, detected fractions
.4/.6, 100,000 signal photons, background 2/bin, shift .075 ns, Gaussian IRF
FWHM .3 ns centered at 1 ns. All physical overrides come directly from the
original PR 3 definitions. C21 calibrates signal photons within its truncated
window, exactly as before. There are no mono-production or proxy-calibration
tasks. The 133 audits schedule 1,064 independent trust-region starts.

Report these primary criteria independently, then their conjunction:

- Numerical acceptance: at least 95 of the 100 scheduled C00 fits, under the
  established full external assessment, including physical/objective/curve
  checks, local validation and completion of required fitting work.
- Each lifetime's median absolute relative error strictly below .05.
- Each lifetime's absolute mean signed relative bias strictly below .02.
- **New deliberately strict Stage 3 audit criterion:** all 100 C00 audits must
  complete and yield usable shared-objective comparisons with absolute centered
  NLL difference at most .01. No incomplete or unavailable pair counts as an
  agreement. This is not a reinterpretation of the historical PR 3 criterion.
- Complete, untampered population: all 543 scheduled task identities terminal
  and all 200 observations durably saved. Missing tasks never disappear from
  denominators or constitute successful completion.

Component metrics retain the original accepted/available-component denominator,
per-component counts, missingness, all-finite/rejected-finite sensitivities and
all-attempt joint recovery fraction. Empty component metrics fail their gate.
No warning-based exclusions. Secondary results are diagnostic only.

Audit completeness requires eight successful finite endpoints, no exhausted
audit budget, a feasible finite best candidate and a finite shared-model
objective recheck. Comparison uses the same shared centered-NLL convention at
both retained candidates. A finite production candidate can be compared even
when production numerical acceptance fails. The signed gap, absolute gap,
completion and comparison availability are separate fields. Mutually exclusive
classes are agreement, substantive disagreement, audit incomplete, production
objective unavailable, and not run. Incompleteness and unavailable comparisons
also have separate overlapping counts. Incompleteness alone is not evidence of
production failure; it does fail the joint audit gate. Agreement does not prove
global optimality.

## Identity, seeds and immutable evidence

PCG64 observation entropy is `[33, 3, 20261029, 1, stream, condition, replicate]`.
Stream 0 is qualification; 1 reserved calibration; 2 nonqualification pilot;
3 summary bootstrap; 4 protocol tests. C26 arms share one saved observation and
count hash. Original PR 3, C00 r11/r58 and PR 2 development observations are never
loaded by this runner. Their outcomes remain development/historical evidence.

The final lock embeds the complete protocol and ordered task lists, LF-normalized
source hashes for all package modules and the runner/report/validator/tests,
interpreter version/path/executable hash, platform, all installed distribution
versions, NumPy/SciPy/threadpoolctl Python and native-library hashes, and numerical
library details. Production imports must resolve to the locked checkout.
Every worker re-verifies the live identity before computation. Worker environment
sets OMP/OPENBLAS/MKL threads to one; threadpoolctl also enforces and records one
thread for every loaded numerical library. One coordinator and one worker run at
a time. Locks must be checked on the actual reference machine; environment drift
requires review, not silent lock replacement.

The existing PR 3 atomic no-overwrite checkpoint format is reused. Every payload
has its task key, execution-lock digest and payload digest in an envelope.
Observations additionally retain integer counts, time grid, RNG entropy,
configuration, sample identity, expectation/kernel/time/count hashes and scope.
Generation finishes and publishes its observation before either fit or audit.
Each consumer verifies observation hashes and regenerates only deterministic
expectations, never a new random observation. Assumed-kernel hashes identify arms.

## Budgets, interruption and restart

Production keeps exhaustive policy, 46 starts, 56,000 optimization calls, 128
initialization calls, 100 validation calls and 100 diagnostic calls. The audit
keeps eight TRF starts, 1,500 nfev/start and 100,000 actual direct-model calls.
No new retries, tolerances, floors or fitter tuning are introduced.

Each task has a 300-second wall limit, including interpreter startup and source
verification. The parent kills and reaps a timed-out worker. A child watchdog
also uses the original deadline if its coordinator disappears. Numerical work
has independent call caps; timeout does not convert an incomplete fit into a
successful result. Timing is descriptive and platform dependent.

OS file leases exclude simultaneous coordinators and workers; process death
releases the lease. A new coordinator must establish worker quiescence before
recovery. An epoch nonce rejects a delayed worker from an older coordinator.

Restart has three cases:

1. A terminal task stays immutable, including failed numerical fits, incomplete
   audits, timeouts, worker exceptions and interrupted attempts. It is never
   retried or duplicated.
2. A started task with a valid durable worker transport completed within its
   original deadline is adopted without recomputation. Late transports become
   terminal timeout results.
3. A started task without a durable result becomes terminal `interrupted`, after
   worker quiescence. Its original observation/request identity remains intact.
   An interrupted generation leaves dependent fits/audits explicitly
   `observation_unavailable`; there is no redraw.

Only never-started tasks execute on resume. A new unexpected worker exception
is saved and halts the coordinator for investigation. A subsequent explicit
resume retains that terminal error and continues other never-started tasks.
Ordinary numerical rejections/incomplete audits are saved and the schedule
continues. Power loss can make the precise interruption cause unknowable;
`interrupted` does not pretend to identify a numerical defect or timeout.

Two generation forward calls per observation, two deterministic input-verification
calls per fit/audit, the external production assessment (at most one call) and
the shared audit recheck (at most one) are separately charged. Killed/error tasks
can have unknown partial call totals: these remain null/unknown, never zero or
invented exact totals. Per-stage known totals are explicitly lower bounds when
such tasks exist. Recovered transports can lack parent timing; this stays null.

## Commands and outputs

`scripts/validate_issue33_stage3_protocol.py` is validation-only and cannot draw.
`scripts/qualify_issue33_stage3.py freeze` publishes the final lock and a separate
denied authorization file. `verify` checks the live identity without creating
observations. `run` requires a separately reviewed positive authorization bound
to the exact lock digest, approver and approval time. The candidate protocol's
review-only flag stays unchanged; authorization is a separate record.

`pilot` is restricted to C00 r0 and C26 r0 on stream 2: two observations, three
fits, three audits. It uses `data/generated/issue33_stage3_pilot_v1`. It cannot
select qualification identities or write the qualification directory. These
pilot data provide runtime/infrastructure evidence only.

Scientific raw records, when separately authorized, go to
`data/generated/issue33_stage3_qualification_v1`. The final execution lock is
`docs/validation/issue33_stage3_execution_lock_v1.json`; the separate authorization
file is `docs/validation/issue33_stage3_execution_authorization_v1.json`.
Scientific results/report locations remain the protocol's declared paths.

`scripts/report_issue33_stage3.py --directory ... --mode qualification|pilot
--output NEW.json --report NEW.md` validates evidence and aggregates read-only.
It holds leases during its snapshot, rejects foreign/changed records, preserves
the saved execution-lock identity, records its own reporting-source identity and
evidence inventory, and refuses output overwrite. Later reporting revisions
must use new output paths and cannot rewrite the original execution identity.
If a started request lacks a terminal record, reporting refuses to classify it
as never run: resume recovery first. `n_terminal_tasks` explicitly counts durable
terminal outcomes; the scheduled denominator is independent of that count.

## Execution identity and nonqualification pilot

The final canonical execution-lock digest is
`7abb3188578458b57f9372572a8d5f0e9567c30864c1bf9cfbc3460f554be73a`.
It locks 72 source/settings/test files, 1,524 numerical-dependency Python/native
files and 139 installed distribution versions. Reference software is Windows,
Python 3.12.0, NumPy 2.5.1 and SciPy 1.18.0, with the recorded BLAS/OpenMP libraries
limited to one thread. Its separate authorization file remains denied.

The small stream-2 pilot completed two observations, three production fits and
three audits. All eight tasks completed in **66.175 seconds** of summed worker
wall time. The three production fits passed the full numerical gate; all three
audits completed and agreed within .01. These are runtime/infrastructure checks,
not primary or secondary qualification results. No qualification stream was used.

| Pilot condition / arm | Fit NLL | Fit seconds | Total worker seconds | Fitter forward calls | Starts |
|---|---:|---:|---:|---:|---:|
| C00 matched | 256.131378 | 7.010 | 12.990 | 10,852 | 46 |
| C26 matched | 260.792739 | 8.661 | 12.721 | 13,540 | 46 |
| C26 Gaussian .3 | 489.395497 | 7.066 | 10.707 | 11,416 | 45 |

Audit worker times were 8.831, 4.213 and 6.201 seconds, respectively. Generation
worker times were 4.896 and 5.617 seconds. Worker totals include startup, identity
verification and checkpoint work; fitter totals do not. The pilot had 37,978
known forward calls across all categories, including 12 input-verification,
four generation, three external assessment and three shared audit-recheck calls.
The independent audit used 2,148 direct-model calls. No counter was unavailable.

The pilot's execution identity is retained unchanged in
`docs/validation/issue33_stage3_pilot_execution_lock_v1.json`, digest
`fa830e76cc756c2d618e0a31ffcddcd871c20bd1cd92719905297326188ed6eb`.
Its denied authorization record is retained alongside it. A subsequent review
clarified terminal-task counting and made reporting reject unrecovered started
requests. Only the reporting script and its tests differ in the final lock;
generation, fitting, audit, runner, settings and environment are identical.
No pilot task was rerun for this reporting change. The earlier source is also
retained in `build/issue33-stage3-dist`'s sdist; final distributions use a distinct
directory. The original pilot report remains immutable. A resume check reused
all existing results, and aggregation before this clarification was byte-identical
before and after resume. A new report under the final reporting implementation
keeps that original execution identity and records its separate reporting hashes.

Simple extrapolation of these two observations' worker times gives approximately
**74 minutes** for 200 observation, 210 production and 133 audit tasks. This small
pilot does not sample the weak/degenerate tails: reserve **2–4 hours** on an otherwise
idle reference machine, with one worker and one numerical thread. The 300-second
task caps permit a much longer worst case (45.25 hours for all 543 tasks, excluding
small coordinator overhead); this is a safety bound, not a runtime prediction.

## Validation record and limitations

The review inventory comprises these new, uncommitted files; no tracked file is
modified. The first five are unchanged artifacts from the earlier protocol review:

- `configs/issue33_stage3_qualification_v1.json`
- `docs/issue33_stage3_protocol_v1.md`
- `docs/validation/issue33_stage3_candidate_lock_2026-10-29.json`
- `scripts/validate_issue33_stage3_protocol.py`
- `tests/test_issue33_stage3_protocol.py`
- `scripts/qualify_issue33_stage3.py`
- `scripts/report_issue33_stage3.py`
- `tests/test_issue33_stage3_execution.py`
- `docs/issue33_stage3_infrastructure_v1.md`
- `docs/validation/issue33_stage3_infrastructure_v1.json`
- `docs/validation/issue33_stage3_execution_lock_v1.json`
- `docs/validation/issue33_stage3_execution_authorization_v1.json`
- `docs/validation/issue33_stage3_pilot_execution_lock_v1.json`
- `docs/validation/issue33_stage3_pilot_execution_authorization_v1.json`
- `docs/issue33_stage3_pilot_v1.md`
- `docs/validation/issue33_stage3_pilot_v1.json`

The ignored pilot checkpoint directory and build/test logs are retained locally.
No qualification output directory exists. The original PR 3 inventory is unchanged:
7,825 files, 123,390,600 bytes, inventory SHA-256
`d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`.
The seven Stage 1 files and all tracked Stage 2/2B files remain unchanged. The
original 1,550 observations, 1,850 comparison outcomes, 205 audits, 66-file lock,
C00 failures and failed 119/121 agreement outcome are preserved.

Focused fixtures cover populations/seeds/pairing, denied draws, hash/lock drift,
task leases, stale workers, interruption adoption, terminal failures, resume,
timeouts, hard audit call limits, full numerical assessment, Poisson zero
fixtures, separate call accounting, strict gates, missingness and reproducible
aggregation. They never sample qualification observations. Routine CI does not
run the pilot or study. The final validation ledger and pilot timing report are
separate from future scientific results.

The final focused suite passes **76 tests** in 4.66 seconds. The complete
repository suite passes **2,428 tests**, with one intentional reference-only
skip and three non-failing warnings, in **770.33 seconds**. Combined statement/
branch coverage is **83.75%**, above the unchanged 80% floor. The earlier
usage-limit interruption and the later reporting review left incomplete full-suite
logs; they remain retained and are not counted as completed validation.

Ruff, the formatting ratchet (129 unchanged audited debt files), new-file and
tracked-file whitespace checks, SDK dependency consistency and strict artifact
metadata checks pass. All 72 final execution-locked source/settings/test files
match the fresh final sdist. Validation narrative updates followed artifact
creation; the sdist is an execution-source snapshot, not a later scientific report.

Final wheel and sdist isolated installations and integration checks both pass:
canonical imports and 66-name API, 38 removed module paths, base-only optional
dependency isolation, CLI, executable examples, dependency consistency and the
small Bayesian integration. These temporary environments resolved NumPy 2.5.4
and SciPy 1.18.1; they are packaging smoke environments, separate from the locked
scientific reference environment. The initial sandboxed build could not access
PyPI; the isolated build and installations succeeded with authorized network
access. No SDK package was replaced and no artifact was published.

The consolidated record is
[`docs/validation/issue33_stage3_infrastructure_v1.json`](validation/issue33_stage3_infrastructure_v1.json).
It records source chronology, historical integrity, exact test counts/timings,
artifact and log hashes, pilot accounting and the runtime extrapolation.

The infrastructure cannot establish scientific success in advance. A complete
46-seed bank still leaves no recovery slot if local validation fails. Weak/mono
components may be unrecoverable despite numerical acceptance. Deliberate IRF
mismatch is not a defect cured by convergence. The strict all-100 audit gate may
fail because the independent solver is incomplete, separately from fitter
failure. Clock adjustment or severe machine load can affect wall-time handling;
the qualification should run on a stable, otherwise idle reference machine.
No criterion or numerical setting may change in response to qualification data.
