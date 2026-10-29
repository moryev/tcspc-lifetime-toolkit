# Issue #33: independent Stage 3 qualification v1

Final scientific review accepts the correction of the original premature-
termination mechanisms and the numerical robustness improvement for publication.
This disposition does not assert complete numerical qualification: Issue #33
remains open. The final documentation/evidence-consistency review is recorded
separately from the original execution identity and immutable terminal results.

**Verdict: the frozen joint primary qualification failed.** C00 passed its
numerical-acceptance and all four lifetime-recovery targets, but achieved
99/100 complete independent objective agreements against the required 100/100.
The one disagreement was also explicitly rejected by the production numerical
gate. No target, setting, observation, or terminal result was changed in response.

The authorized study completed on October 29, 2026, under
`issue33_stage3_qualification_v1`. All 200 observations, 210 production fits and
133 independent audits have terminal records. All tasks completed normally;
numerical rejection is distinct from process failure. The full numerical gate
accepted 206/210 fits. All 133 audits were complete and usable; 129 agreed within
the frozen absolute centered-NLL tolerance of 0.01 and four exceeded it.

## Evidence and execution identity

The authoritative machine-readable [results](validation/issue33_stage3_results_v1.json)
were produced by the frozen aggregation code. Supplementary
[diagnostics](validation/issue33_stage3_diagnostics_v1.json) retain per-condition
starts, omissions, warnings, rejected fits, all audit comparisons and C26 pairing.
The [postflight integrity record](validation/issue33_stage3_integrity_v1.json)
reconciles the population, hashes, budgets, environment and historical evidence.
Raw observations and task checkpoints remain in
`data/generated/issue33_stage3_qualification_v1/`.

The execution lock's canonical SHA-256 is
`7abb3188578458b57f9372572a8d5f0e9567c30864c1bf9cfbc3460f554be73a`.
The complete checkpoint-inventory digest is
`a44df9408aec60d63e2910003bb9cbc9f004e4a2a6961eb0dc93ef79d2b2be43`.

The branch remains `investigate/issue-33-optimizer-termination`, with Stage 2/2B
at `5be88edda17ad6612a8447dca0e11a6543502b85` and Stage 1 separately preserved
at `3bff378f`. The tracked worktree was unchanged during scientific execution.
No fitter, physical model,
Poisson likelihood, zero-background support, IRF convention, tolerance, or
historical record was modified.

Execution used the separately recorded maintainer
[approval](validation/issue33_stage3_execution_approval_2026-10-29.json), bound
to the exact lock and 200/210/133 scope. The candidate protocol and earlier
denied-authorization record remain unchanged as review history. Their original
`proposed`/false fields were not edited to bypass the execution check. The
[preflight](validation/issue33_stage3_preflight_2026-10-29.json) passed before
any scientific draw, including exclusive coordinator/worker checks.

All 72 locked source/settings/test files, the exact Python executable, 139
distribution versions and 1,524 numerical dependency files were verified.
The reference environment was Windows, Python 3.12.0, NumPy 2.5.1 and SciPy
1.18.0, with serial workers and one numerical thread. All 543 worker transport
records confirm one numerical thread. No simultaneous coordinator was launched.

The root entropy was `[33, 3, 20261029, 1]`; observation seeds appended
`[0, original_condition_id, replicate_index]`. All 200 qualification identities
and entropies are unique. Every observation was saved and hashed before fitting.
The two C26 assumed-IRF arms share the same observation identity and counts hash
for each of their ten replicates. Pilot and historical development observations
are absent from the qualification population. Summary bootstrap calculations use
the separate frozen summary stream, not additional Poisson observations.

## Primary C00 results

C00 retains the original 512-bin, 0--20 ns, one-million-signal-photon generating
condition, lifetimes 0.8 and 3.2 ns, detected fractions 0.4/0.6, background 2
counts/bin, shift 0.075 ns and matched Gaussian IRF of FWHM 0.3 ns.

| Frozen criterion | Result | Target | Verdict |
|---|---:|---:|---|
| Full numerical acceptance | 99/100 | At least 95/100 | Pass |
| Median absolute relative error, tau 1 | 0.332001% | Below 5% | Pass |
| Median absolute relative error, tau 2 | 0.236241% | Below 5% | Pass |
| Absolute mean signed relative bias, tau 1 | 0.005075% | Below 2% | Pass |
| Absolute mean signed relative bias, tau 2 | 0.005351% | Below 2% | Pass |
| Complete usable independent agreement | 99/100 | 100/100 within 0.01 NLL | **Fail** |
| Complete, untampered population | 100/100 | All scheduled observations | Pass |

Both signed biases are positive. Lifetime statistics use the frozen
accepted-and-available-component denominator: 99 for each component, with no
unavailable component among accepted fits and no warning-based exclusion. The
rejected fit stays in the original numerical and audit denominators. The
all-attempt joint accepted-and-both-within-5% fraction is 99/100. As a descriptive
sensitivity, all 100 finite returned candidates give median errors of 0.337390%
and 0.240204%, and signed biases of +0.027879% and +0.010636%. These sensitivity
statistics do not replace the frozen gate.

All 100 primary audits completed all eight independent starts with usable finite
comparisons. There were zero incomplete audits, unavailable comparisons, missing
tasks, or timeouts. The failed audit target is an objective disagreement, not an
audit-completeness failure. Agreement does not establish global optimality.

### New C00 r050 failure

The new observation `observation-s0-c00-r050` produced centered NLL
245.763158651684 in production and 237.259286039528 in the independent audit:
production minus audit is **+8.503872612156**.

| Parameter | Retained production candidate | Independent audit candidate |
|---|---:|---:|
| Amplitude 1 | 18940.6540813 | 18973.7621510 |
| Amplitude 2 | 7186.3928554 | 7369.0945779 |
| Lifetime 1, ns | 0.818283736 | 0.798537185 |
| Lifetime 2, ns | 3.217082105 | 3.186329367 |
| Background, counts/bin | 1.432762418 | 1.494763790 |
| Temporal shift, ns | 0.074332674 | 0.075231849 |

All 46 finite seeds were attempted. Independent local validation detected
0.370458 NLL descent at its validation anchor and retained its improved finite
candidate. No recovery slot remained. The fitter reported local validation
false, exhausted budget `starts`, recovery unavailable `start_budget_exhausted`,
and stopping reason `recovery_start_budget_exhausted`. The external assessment
rejected the fit for failed/unavailable local validation, exhausted required
work budget and incomplete required fitting stages.

Recorded start messages show safeguarded nonfinite-objective or failed-derivative
termination for all 46 invocations; 43 reported safeguarded descent available.
This supports the known limitation that completing a 46-seed bank leaves no slot
for required recovery. It does not establish which unperformed recovery would
have succeeded. The run retained the failure and its best finite candidate; no
additional production or audit invocation was made. Neither a successful
extra recovery nor a uniquely determined solver-level path to the full audit
gap has been demonstrated by these saved records. This new r050 is unrelated
to the historical C00 r11/r58 development regressions.

## Secondary diagnostic conditions

There are ten secondary conditions, ten independent observations per condition,
and eleven production arms because C26 is paired. Audit replicates are exactly
0, 4 and 9 for every arm. All 33 secondary audits are complete; 30 agree and
three disagree. These conditions have no new pass/fail recovery thresholds.

The baseline for secondary conditions is 100,000 signal photons, lifetimes
0.8/3.2 ns, detected fractions 0.4/0.6, background 2 counts/bin, shift 0.075 ns
and the matched Gaussian IRF unless the row changes those quantities.

| Condition / arm | Accepted / fits | Audit agreements / scheduled | Component n (1, 2) | Median absolute relative errors (1, 2) | Signed biases (1, 2) |
|---|---:|---:|---|---|---|
| C04: 100 signal photons | 10/10 | 3/3 | 9, 9 | 72.77%, 55.48% | -58.27%, +23.02% |
| C05: true mono, 2 ns, one million signal photons | 7/10 | 1/3 | 0, 0 | N/A | N/A |
| C08: lifetimes 2/2.1 ns | 10/10 | 3/3 | 9, 9 | 3.44%, 1.97% | -28.08%, +102.77% |
| C09: detected fractions 1/0 | 10/10 | 3/3 | 0, 0 | N/A | N/A |
| C10: detected fractions 0.995/0.005 | 10/10 | 2/3 | 10, 10 | 1.03%, 70.74% | -5.77%, +25.44% |
| C13: exactly zero background | 10/10 | 3/3 | 10, 10 | 1.39%, 0.95% | -1.07%, -0.45% |
| C16: shift -0.3 ns | 10/10 | 3/3 | 10, 10 | 1.41%, 1.02% | -0.26%, +0.13% |
| C17: shift +0.3 ns | 10/10 | 3/3 | 10, 10 | 1.61%, 1.30% | -0.13%, -0.78% |
| C21: first 103 acquisition bins | 10/10 | 3/3 | 10, 10 | 10.93%, 17.87% | -10.28%, -11.04% |
| C26: generating EMG, matched EMG | 10/10 | 3/3 | 10, 10 | 0.99%, 0.60% | +0.13%, +0.09% |
| C26: same observations, assumed Gaussian | 10/10 | 3/3 | 10, 10 | 39.15%, 7.11% | +40.04%, +7.36% |

Component denominators retain the frozen availability rules. One accepted
component in each position is unavailable in C04 and C08. C05/C09 do not have
two positive distinct generating components, so the paired-component recovery
statistics are undefined by design. This does not erase their numerical fits,
warnings or audits. Canonically sorted fitted components do not create two
physical truths for a mono or zero-component generating process.

Low signal, weak components and nearly equal lifetimes illustrate the difference
between numerical acceptance and recoverability. C04 has only 100 expected
signal photons against 1,024 expected background counts. C08's small median
errors coexist with severe tails and signed biases; the median alone is
misleading. The short-window C21 retains the first 103 points of the original
grid (approximately 3.99 ns), with the original within-window photon calibration.
Its numerical fits and audits pass while component errors remain substantial.

C13 preserves exactly zero background support: nine returned backgrounds are
exactly zero and the remaining one is 0.121723 counts/bin. All ten pass the full
gate. Boundary warnings occur for all ten; a warning is not a numerical failure.
C16/C17 also pass all fits and scheduled audits. Three lower-bound cases and
five upper-bound cases return shifts exactly at their fitted bound; six and five
respectively carry shift-boundary warnings, including nearby solutions.

C26 generates an EMG IRF with FWHM 0.3 ns, tail 0.2 ns and center 1 ns. On the
same ten observations, deliberate Gaussian misspecification increases centered
NLL by 205.772708--244.953256, median 218.753367. Both arms are numerically
accepted and warning-free in every case, and all six audits agree. The large
misspecified-arm component bias is physical model mismatch, not failed numerical
convergence. No likelihood-ratio p-value or universal mismatch threshold is
inferred from this descriptive paired comparison.

### Every substantive independent disagreement

| Observation / matched arm | Production accepted? | Production NLL | Audit shared NLL | Production minus audit |
|---|---|---:|---:|---:|
| C00 r050 | No | 245.763159 | 237.259286 | +8.503873 |
| C05 r004 | No | 333.261726 | 264.582596 | +68.679130 |
| C05 r009 | No | 293.882582 | 251.810150 | +42.072432 |
| C10 r004 | Yes | 310.785769 | 311.250190 | -0.464421 |

C10 r004 is a complete audit disagreement with a **lower production objective**.
Production returned lifetimes 0.804186/8.0 ns; the independent search returned
0.724996/0.894065 ns. Eight successful independent starts therefore do not prove
the best objective was reached. The negative gap is retained as a disagreement
under the symmetric absolute tolerance, not mislabeled as production failure.
No independent rerun or extra start was performed.

C05 r001 was also numerically rejected, with retained NLL 1459.850283, but was
not in the predeclared audit population. No independent objective is invented
for it. C05 r001/r004/r009 all detected feasible descent and exhausted the start
cap before required recovery could complete. The retained validation-anchor
descent amounts were 447.927070, 103.792195 and 215.765866 respectively.

## Starts, omissions, budgets and warnings

All 210 declared finite seed banks completed: 9,521 seed starts and zero
unattempted required seeds. Bank sizes ranged from 44 to 46 after recorded
initialization filtering. Six required validation-recovery starts were charged,
for **9,527 total production starts**. Every fit used at most 46 starts.

Four recovery invocations succeeded in C00 r039/r061/r079/r097. The remaining
two, C05 r004/r009, used their available slot but still failed local validation;
further required recovery was unavailable. C00 r050 and C05 r001 already used
all 46 slots on seeds. These are the four explicit required-recovery failures.

Exactly 46 starts were used by 78 fits: 74 were accepted and four were rejected.
Using the cap is not itself failure. The 206 accepted fits report
`candidate_bank_completed`; the other four report
`recovery_start_budget_exhausted`. Only these four report an exhausted budget,
always `starts`. No initialization, objective, validation, diagnostic, audit-call
or wall-time budget was exhausted.

There were **6,504 optional continuation omissions across 203 fits**, including
199 accepted fits and all four rejected fits. These are logged per-start requests
that were deferred; they are not 6,504 additional attempted starts or predicted
successful recoveries. Passing validation can make their omission harmless under
the established contract. Failed validation still requires recovery, which is
reported separately and cannot be excused as optional work.

| Population | Seed starts | Recovery starts | Optional omissions | Accepted fits |
|---|---:|---:|---:|---:|
| C00 | 4510 | 4 | 3813 | 99/100 |
| C04 | 460 | 0 | 3 | 10/10 |
| C05 | 452 | 2 | 318 | 7/10 |
| C08 | 451 | 0 | 278 | 10/10 |
| C09 | 455 | 0 | 296 | 10/10 |
| C10 | 452 | 0 | 308 | 10/10 |
| C13 | 460 | 0 | 79 | 10/10 |
| C16 | 460 | 0 | 112 | 10/10 |
| C17 | 450 | 0 | 335 | 10/10 |
| C21 | 460 | 0 | 328 | 10/10 |
| C26 matched | 460 | 0 | 304 | 10/10 |
| C26 Gaussian | 451 | 0 | 330 | 10/10 |

The full external numerical gate is distinct from both local validation and
the independent optimization audit. It checks the feasible finite candidate,
recomputed unchanged Poisson objective, local validation and completion of
required fitting/budget stages. Diagnostics remain separately reported. Here
206 local-validation passes also passed the full gate; the equality of these
counts does not redefine the gate as a local Boolean.

Warnings remain descriptive and were never used to exclude an observation.
Across 210 fits, 92 had no identifiability warning. Among those accepted,
available components, absolute relative errors reached 77.40% and 255.67%.
All ten C26 misspecified fits were warning-free despite substantial bias; nine
of ten short-window fits were warning-free. Absence of a warning is not a
calibrated recovery or model-validity guarantee. Weak-component, rank-deficiency,
boundary and near-optimal-start disagreement warnings occur prominently in
degenerate conditions; exact category counts remain in the diagnostics JSON.

## Work accounting and runtime

| Work stage | Forward-model evaluations |
|---|---:|
| Observation generation | 400 |
| Input/template verification | 686 |
| Production initialization | 19,320 |
| Production optimization objective | 2,607,293 |
| Production independent local validation | 10,437 |
| Production diagnostics | 7,585 |
| External production assessment | 210 |
| Independent direct-convolution audit | 128,450 |
| Audit shared-model recheck | 133 |
| **Total** | **2,774,514** |

No task has unavailable call counts. Production objective usage ranged from
8,858 to 40,890 calls (median 10,756), below 56,000. Each initialization used
92 of 128 calls; validation used 43--98 of 100; diagnostics used 26--72 of 100.
Independent audits used all 1,064 scheduled starts, with 570--6,234 direct calls
per audit (median 643), below their 100,000-call limit. Each start respected
the frozen 1,500 function-evaluation limit. External assessment and shared
rechecks are explicitly outside the fitter and direct-audit counters.

The serial study ran from **10:15:55.251 to 11:26:36.273 UTC** on October 29:
**4,241.022 seconds, or 70.684 minutes**. The sum of task worker wall times was
4,222.960 seconds. Median/max task times, including startup and verification,
were 3.927/13.371 s for observations, 11.535/32.095 s for production, and
4.333/15.208 s for audits. Every task was below its 300-second wall cap.

Production fitter time alone totaled 1,664.467 s (median 6.971 s, maximum
25.119 s). Independent numerical audit time totaled 41.308 s (median 0.214 s,
maximum 1.908 s). Fresh process startup, lock/environment verification and
checkpointing account for much of the remaining worker time. The measured
70.7-minute study is consistent with the pilot's approximately 74-minute
extrapolation. CPU-time and peak resident-memory profiling were not recorded;
wall time must not be described as CPU time. The final raw directory contains
2,720 files totaling 24,975,082 bytes, including 2,173 hashed evidence envelopes,
execution lock, logs and lease bookkeeping. No GPU work was used.

## Reconciliation, historical protection and review boundary

Postflight verified all 543 expected terminal identities and timing records,
all observation/count/grid bindings, all checkpoint envelopes and the exact
frozen aggregation. There were no missing tasks, terminal process errors,
timeouts, orphan interruptions, silent retries or redrawn observations. The
coordinator stderr was empty and both execution leases were quiescent. A
usage-limit pause in the assistant conversation did not stop or duplicate the
background coordinator.

The historical PR-3 directory is unchanged: 7,825 files, 123,390,600 bytes,
inventory digest
`d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`.
Its 1,550 observations, 1,850 paired outcomes, 205 audits, original 66-file
source/settings/environment lock, C00 r11/r58 failure evidence and failed
119/121 audit outcome remain historical evidence. None was rewritten or entered
the new independent denominator. All seven Stage-1 files are unchanged.

The frozen infrastructure's pre-execution validation remains 76 focused tests
and 2,428 full-suite passes with one intentional skip, 83.75% combined coverage,
passing Ruff/formatting, wheel/sdist builds, strict metadata and both isolated
installation/integration checks. Those checks were completed before execution;
they are not new tests of scientific success. No locked code changed during
execution. The separate post-execution descriptive utility
`scripts/analyze_issue33_stage3_results.py` passes Ruff and formatting checks;
its source hash is recorded in the diagnostic output, separate from the frozen
execution identity. It neither draws observations nor invokes a fitter. Its
actual-evidence assertions verify budget bounds, pairing, thread records and
the frozen evidence digest. The frozen aggregate was reproduced exactly during
postflight without rerunning any scientific task.

Read-only aggregation can be reproduced with the frozen reporter using the raw
checkpoint directory and **new output/report paths**. Existing result paths are
write-once. The supplementary utility likewise requires a new output path.
Raw checkpoint files are retained locally in the ignored generated-data area;
publication requires preserving that evidence alongside its manifest, rather
than assuming a Git push would include it. Reporting revisions must retain the
original execution digest and state their separate reporting-source identity.

**Recommendation:** Issue #33 is not ready to be presented as fully qualified or
resolved under the approved Stage-3 criterion. The implementation and this
negative qualification record have been accepted by the maintainer for transparent
PR publication, with Issue #33 remaining open. The residual required-recovery-capacity
problem, especially in high-count mono data, needs a separate design decision.
The weak-component audit disagreement also limits claims about the independent
search. Further diagnostic replay or algorithm changes would make these new
observations development evidence and would require a newly frozen independent
qualification for subsequent confirmatory claims. No such work was performed
or authorized here. Scientific execution itself made no Git history or GitHub
changes. The subsequent publication review authorizes a commit, branch push and
PR using `Refs #33`; it does not authorize merging, release or issue closure.
