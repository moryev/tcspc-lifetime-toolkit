# Issue #33 Stage 2: numerical correction

Refs #33. Development/regression work, not independent qualification.
Stage 1 is preserved in commit `3bff378` on
`investigate/issue-33-optimizer-termination`. The original PR-3 evidence and
source/settings locks remain historical records, not locks for this correction.

## Design declared before production edits and saved-case runs

Stage 1 demonstrated two unusable callback forms: infinite objective with a
zero placeholder gradient, and finite objective with NaN finite differences.
Neither is a valid differentiable function/gradient pair for L-BFGS-B. Rejected
trials were followed by unchanged iterates and relative-reduction success; one
invalid endpoint even reported projected-gradient success. Finite probes were
retained correctly, and independent validation rejected the historical results.

A scalar control with the installed SciPy 1.18.0 minimizes
`1000*x - log(x)` on `[0, 2]`, starting at `0.1`. Both `(inf, 0)` and
`(inf, NaN)` at zero terminate successfully at `0.1` (cost 102.302585).
Raising on the invalid trial, then explicitly restarting after finite descent,
reaches `0.001` (cost 7.907755) in eight counted solver invocations. Replacing
the placeholder with NaN alone is therefore not the correction.

The proposed bounded correction keeps the shared physics, exact Poisson
objective, scaled coordinates, initializer and L-BFGS-B:

1. Use the exact Poisson score for the linear amplitude/background coordinates.
   For zero-count bins the score is one even when the expectation is zero;
   positive counts at zero expectation still have infinite objective. No floor
   or transformed likelihood is introduced. Shared component curves supply the
   amplitude derivatives; a zero amplitude uses a charged unit-amplitude call.
2. Retain bound-aware finite objective differences for lifetime/shift. Permit
   finite one-sided stencils if the opposite side is impossible, with at most
   the original eight step reductions. An unavailable/nonfinite derivative
   raises a numerical failure; no invalid pair reaches SciPy. Final review also
   guards steps too small to move a coordinate: identical stencil coordinates
   are an explicit failure, never evidence of a zero derivative.
3. On an unusable callback, terminate that solver invocation explicitly.
   From its best usable callback, seek feasible projected-gradient Armijo
   descent (coefficient `1e-4`, at most 40 halvings, first step one). All trial
   evaluations use the original objective and optimization counter. No penalty
   objective is supplied to SciPy. Only actual finite descent can request a
   continuation; the best finite probe survives every failure.
4. Prioritize a requested continuation before the next seed. Every new
   `minimize` invocation gets its own start record (`safeguarded_recovery`) and
   consumes the existing start cap. This is an explicit allocation change;
   fewer seed starts may fit within the same cap. No increased work limits,
   hidden retries, altered optimizer tolerances or warning thresholds.
5. Re-evaluate solver endpoints and retain independent feasible-descent
   validation. Report recovery unavailable due to start/optimization/validation
   limits separately from completion of the seed bank.

All original default settings remain: 46 starts, 56,000 optimization calls,
128 initialization, 100 validation and 100 diagnostic calls; `ftol=1e-12`,
`gtol=1e-6`, difference step `1e-5`, local-descent tolerance `.01`,
1,000 iterations, `maxls=40`. No claim of global optimality or calibrated
identifiability follows from numerical acceptance.

First test synthetic numerical and physical contracts. Then run the saved C00
r11/r58 observations once under these declared settings, targeting absolute
centered-NLL gaps <= .01 from 247.844923 / 259.114084. Preserve any miss and
investigate it before changing the mechanism/settings. These observations are
development regressions. Historical PR-2 comparisons reuse existing observations;
there will be no fresh scientific observations in Stage 2.

## Implementation and regression findings

Only `estimation.multiexponential` changes production behavior. Its additional
diagnostic field, `recovery_unavailable_reason`, defaults to `None`; existing
public function signatures, parameter/result fields, settings values, root
exports, mono fitting, likelihood and forward model remain unchanged.
Per-start messages identify unusable callbacks, unavailable safeguarded descent,
invalid solver endpoints and exhaustion during a safeguard. An interrupted
solver has no reported success/status; no success flag is fabricated.

The exact score removes six usual objective probes per callback: a regular
interior callback now uses seven forward calls instead of thirteen. A zero
amplitude requires an additional charged component-model call. Failed shape
stencils, one-sided alternatives and safeguarded probes can cost more. Every
`minimize` call has a start record. All derivative/trial calls share the hard
optimization counter, and best finite candidates remain available on failure.

### Saved C00 observations

The two saved observations ran under the declared unchanged numerical settings,
one numerical thread, Python 3.12.0 / NumPy 2.5.1 / SciPy 1.18.0. They are
development regressions, not new independent validation. The independent
optimizer's saved objective is a comparison value, not a global-optimum proof.

| Quantity | C00 r11 | C00 r58 |
| --- | ---: | ---: |
| Historical production centered NLL | 1228.441945222 | 331.964882 |
| Corrected centered NLL | 247.844922542217 | 259.114084194548 |
| Gap from requested rounded audit objective | -0.000000458 | +0.000000195 |
| Local numerical validation | pass | pass |
| Maximum detected local descent | 0 | 0 |
| Seed / safeguarded starts | 16 / 30 | 15 / 31 |
| Initialization / optimization calls | 92 / 9498 | 92 / 8741 |
| Validation / diagnostic calls | 49 / 38 | 49 / 34 |
| Total forward calls | 9677 | 8916 |
| Runtime, seconds | 6.882 | 6.339 |
| Historical runtime, seconds | 7.757 | 7.126 |
| Required-budget qualification acceptance | **fail: starts exhausted** | **fail: starts exhausted** |

Returned parameters use canonical lifetime order:

| Parameter | r11 | r58 |
| --- | ---: | ---: |
| Amplitude 1 | 19133.9551159 | 19222.9441563 |
| Amplitude 2 | 7240.2349123 | 7185.4504415 |
| Lifetime 1, ns | 0.805889629 | 0.809097916 |
| Lifetime 2, ns | 3.216767368 | 3.222513039 |
| Background, counts/bin | 1.484348098 | 1.378381276 |
| Shift, ns | 0.074228349 | 0.074893198 |

Both stop with `recovery_start_budget_exhausted` and explicitly report
`recovery_unavailable_reason="start_budget_exhausted"`. The seed bank remains
incomplete. Local validation does **not** override this missing work. Under
the unchanged PR-3 required-budget gates, numerical acceptance is **0/2**,
despite objective agreement **2/2** and local validation **2/2**. No acceptance
rule was relaxed to conceal this result.

### Historical PR-2 comparisons

The selected existing stream-2 observations are conditions
`0, 3, 4, 5, 6, 9, 12, 13, 15, 17, 18, 19`: separated, close/equal, zero/weak
secondary component, 100 and one million signal photons, zero background,
negative shift, short window, EMG, and sampled double peak. Every reconstructed
count array matches the October 25 saved hash before fitting. This is reuse of
previous observations, not a newly untouched stream.

All 12 local validations pass; all objective differences from the historical
PR-2 fits are below `7.2e-8` in magnitude. The fitted zero-background case still
returns **exactly zero** background. Parameters can differ in degenerate or
weakly constrained directions without materially changing the objective; the
full parameter/diagnostic records are retained.

However, **11/12 exhaust the start budget**, so only **1/12** meet the historical
required-budget acceptance rule. The 100-photon case completes its 46-seed bank
without a restart. Runtime medians are 9.093 seconds corrected versus 12.946
seconds historical; optimization calls total 184,059 versus 267,093. These
timings describe selected existing cases on one workstation. Fewer explored
seeds partly explain the reduced cost; this is not equal-work speed qualification.

The exact-degeneracy tests now generate their exact optimum with the shared
forward model. Direct-convolution targets differ from FFT targets at roundoff
level and can prefer a roughly `1.6e-14` nonzero amplitude or unequal-ulp
lifetimes. The corrected optimizer exposed those tiny differences. Exact zeros
and equal lifetimes remain admissible and tested; scientific warning thresholds
were not changed. Independent direct-convolution physical recovery tests remain.

### Evidence and source chronology

Raw per-start regression records and exclusive attempt manifests are in
`data/generated/issue33_stage2_c00_v1` and `issue33_stage2_pr2_v1`.
Each contains a hash-verified snapshot of the fitter used. C00 completed normally.
During the PR-2 comparison, a reporting-only edit was made after Python had
imported the fitter. All 12 fits completed, but the runner's end-of-run source
guard correctly refused summary publication. The exact loaded source was
reconstructed and hash-verified; `recovered_results.json` was assembled from the
unchanged raw records without repeating any fit. This bookkeeping failure is
retained, not relabeled as an uninterrupted locked run.

The final reporting edit resets a stale recovery reason between adaptive tiers
and reports unavailable recovery after incomplete validation. It does not change
the objective, derivatives, trial handling or settings, and those new reporting
paths do not occur in the 14 completed-validation regressions. Dedicated tests
exercise the final reporting paths. Final review added the unrepresentable-step
guard and its numerical-limit regression. A final-source verification reuses the
same 14 observations with unchanged settings, comparing all non-timing outputs
to the first correction attempt. This is justified by the new guard, not tuning
against observed errors. Final verification timings may overlap other validation
work; the isolated first-attempt timings above remain the cost comparison. All
attempts and source hashes remain separate.

Final-source verification completed for all 14 observations. Every non-timing
field matches the original correction attempt exactly, including all parameters,
per-start records, objective values, diagnostics and phase counters. The final
source SHA-256 is
`58950f84e24cb289c9b03f53fc664827767b2bb6b544a4d59e52a76adf1b846c`.
Both completed final-verification manifests and their source snapshots are
preserved under `issue33_stage2_c00_final` and `issue33_stage2_pr2_final`.
The raw worker's `numerical_acceptance` field denotes local validation only;
the [consolidated record](validation/issue33_stage2_results_2026-10-28.json)
explicitly separates `local_numerical_validation_passed` from
`historical_required_budget_acceptance`.

The PR-3 qualification runner intentionally refuses the changed fitter. Its
source-lock test now asserts that refusal and checks transport against the saved
historical lock. Neither the runner nor the original lock/configuration changes.
The seven Stage-1 files remain equal to commit `3bff378`, their pre-execution
hashes match, and all 7,825 historical PR-3 JSON files (123,390,600 bytes) retain
inventory SHA-256 `d2f9acc213768751a333fe1b78dc008fc932c3012d5889e606d429680d63ba14`.

## Remaining numerical limitations and disposition

The reproduced invalid-callback mechanisms are corrected and the saved-objective
regression target is met. **The current allocation is not qualified for promotion
under the unchanged required-budget acceptance gates.** Safeguarded restarts
consume slots that previously explored independent seeds. This is a real
compatibility limitation, not grounds to call start exhaustion harmless or
silently raise the cap. Stage 2 stops with this result for maintainer review.

Finite difference neighborhoods remain vulnerable to FFT-tail roundoff at zero
background. They now fail explicitly when no finite stencil can be found.
The guarded descent can be unavailable (no usable callback, nonfinite slope,
no Armijo descent within 40 trials, or exhausted work). No method here proves
global optimality. Shift interpolation knots remain nonsmooth; local validation,
weighted-Jacobian rank and warning absence do not certify component recovery or
physical model validity. No intervals, Bayesian/ML extensions, model selection,
persistence migrations, release changes or frozen benchmark regeneration occur.

## Changed-file inventory and diff review

Stage 1 is the separate, requested seven-file commit `3bff378`; Stage 2 is an
uncommitted nine-file diff:

| File | Change |
| --- | --- |
| `src/tcspc_toolkit/estimation/multiexponential.py` | Only production change: guarded derivatives/trials, charged continuations and truthful recovery reporting. |
| `tests/test_biexponential_optimizer_safeguards.py` | 22 focused numerical controls, including actual SciPy interaction, failed/unrepresentable stencils, zero/small expectations and exhausted budgets. |
| `tests/test_biexponential_reconvolution.py` | Replace the unsafe placeholder-gradient assertion with explicit failure semantics. |
| `tests/test_biexponential_diagnostics.py` | Define exact-degeneracy targets on the evaluated model; retain independent physical recovery tests. |
| `tests/test_biexponential_qualification.py` | Require rejection of the changed source by the historical qualification lock; retain saved-lock checkpoint tests. |
| `scripts/regress_issue33_optimizer.py` | Separately invoked saved-observation regressions with exclusive outputs, input hashes and complete per-start records. |
| `docs/api.md` | Document the current numerical/recovery contract and qualification limitation. |
| `docs/issue33_stage2_correction.md` | Design rationale, evidence, limitations, validation and proposed Stage 3. |
| `docs/validation/issue33_stage2_results_2026-10-28.json` | Machine-readable comparisons, source/artifact hashes and validation evidence. |

Review confirms no edits to physics/likelihood/mono implementations, measurement
or IRF contracts, benchmark seeds/configuration, historical scientific records,
root exports, persistence schema, version, tag or release metadata. All new
scientific work is regression/development evidence. No push, PR, merge, release
or issue closure is performed.

## Final validation

The final source above passed **2,334 tests**, with **one intentional skip** and
three non-failing warnings, on Windows/Python 3.12.0. Combined statement/branch
coverage is **83.69%**, exceeding the unchanged 80% gate. The skipped test is
the explicitly invoked Week-7 exact stochastic regeneration; that expensive
historical reference workflow was not rerun. These checks qualify this local
environment, not every supported Python/platform combination or experimental
validity.

| Check | Result / evidence |
| --- | --- |
| Focused numerical and physical tests | Initial 195-test group passed; final guard/physical group passed 112 tests; final complete suite includes all 22 new safeguard controls. |
| Full suite and branch coverage | PASS; `build/issue33-stage2-resumed-pytest.log`, `build/reports/issue33-stage2-resumed-pytest.xml`, `build/reports/issue33-stage2-resumed-coverage.json`; 689.90 seconds. |
| Ruff | `ruff check src tests examples scripts`: PASS. |
| Formatting ratchet | `scripts/check_format.py`: PASS, 129 unchanged audited debt files; no baseline refresh. |
| Dependency consistency | `pip check`: PASS in the project environment and disposable artifact environments. |
| Build and strict metadata | Final wheel/sdist built successfully; `twine check --strict`: both PASS. |
| Isolated artifact installations | `scripts/check_distribution.py --dist-dir build/issue33-stage2-final-dist`: PASS for wheel and sdist, including metadata/module inventory, ordered 66-name API, removed paths, CLI, README/examples, optional-dependency isolation and small Bayesian integration. Log: `build/issue33-stage2-resumed-distribution.log`. |
| Artifact source identity | Both final artifacts contain exactly the final fitter source; artifact hashes are in the consolidated record. |
| Saved regressions | All 14 final-source records exactly match all first-attempt non-timing fields; C00 objectives agree with saved independent shared objectives to within `1.1e-11`. |
| Preservation and diff | Seven Stage-1 files, historical evidence inventory and locks unchanged; only the authorized fitter differs among the historical locked sources; `git diff --check` passes. |

An earlier full run passed 2,333 tests before the final unrepresentable-step
guard. The subsequent final validation was interrupted by the session usage
limit at the import checks; its partial logs remain in
`build/issue33-stage2-final-pytest.log` and
`build/issue33-stage2-final-distribution.log`. Neither interrupted process was
still running when work resumed. The authoritative completed checks above were
then run against the unchanged final source, with separate `resumed` logs.
The completed 14-observation regressions were not repeated on resume.

The first sandboxed build could not fetch its isolated build dependency; a
no-isolation attempt also confirmed that setuptools was absent from the project
environment. The isolated build and installation checks subsequently completed
with the necessary network permission, without modifying the project SDK or
publishing artifacts. Historical version/release metadata remains unchanged.

## Proposed Stage 3 protocol — not authorized or executed

1. **Resolve the capacity issue before generating observations.** Review the
   present 46-start allocation on development controls only. Any revised policy
   or budget needs an explicit new version and review; do not remove the
   required-start-budget acceptance gate. If the present settings are retained,
   predeclare that all exhausted cases fail numerical acceptance. Freeze code,
   settings, environment and this decision before independent generation.
2. Use a new protocol identifier `issue33-stage3-v1`, new directory and root
   entropy **[33, 3, 20261028, 1]**. Reserve stream 0 for qualification, 1 for
   independent proxy calibration, 2 for development/pilot, and 3 for summary
   bootstrap. Do not draw any of these streams in Stage 2. Neither C00 r11/r58
   nor any PR-2/old PR-3 observation can enter the new independent denominator.
3. Copy the physical definitions, replicate counts and arm mapping of the
   immutable `issue28-pr3-v1` manifest into a **new** reviewed manifest. Keep
   all 30 conditions: 100 baseline replicates and 50 for each other condition,
   totaling **1,550 observations**, **1,850 paired mono/bi comparisons** and
   **3,700 fits**. Preserve the endpoint-inclusive 512-bin 0–20 ns baseline,
   0.8/3.2 ns lifetimes, 0.4/0.6 detected fractions, one million signal photons,
   background 2/bin, shift .075 ns and Gaussian FWHM .3 ns. The short-window
   condition retains the first 103 bins. Preserve lower counts, mono/equal/close
   and weak/zero components, backgrounds, shifts/bounds, IRF limits, Gaussian,
   EMG, sampled and explicitly approximate proxy arms and paired misspecification.
4. The proposed starting numerical settings are the **unchanged values listed
   above**, pending the explicit capacity decision. Use the same histogram,
   assumed IRF, grid/window and raw-count conventions for each mono/bi pair.
   Keep nominal parameter counts 4/6 and descriptive deviance/AIC/bin-BIC;
   do not use an ordinary chi-square mono/bi likelihood-ratio p-value.
5. Require baseline numerical acceptance **>=95/100**, both component median
   absolute relative errors **<5%**, and both absolute relative biases **<2%**.
   Preserve all-attempt denominators, missing/unavailable component counts and
   unconditional joint failure rates; conditional component summaries keep the
   existing positive/distinct-component eligibility, with no warning filtering.
   Retain the original 5,000 summary-bootstrap replicates and 95% intervals.
   Generating components, deterministic pseudo-true mono projections and optional
   trusted experimental references remain distinct.
6. Schedule **205 independent audits**: all 100 baseline replicates, plus the
   original secondary indices `[0,24,49]` and arm rules. Preserve the designated
   **121** agreement targets and all descriptive audits. Use independent direct
   convolution, physical lifetime coordinates and eight trust-region starts
   from the seven-point lifetime grid pairs `(1,4),(2,5),(1,5),(2,4)` at shifts
   `-.1,+.1` ns. Keep 100,000 forward calls, 1,500 evaluations/start,
   `ftol=xtol=1e-12`, `gtol=1e-9`, and absolute shared-objective gap <=.01.
   Count failures/timeouts/misses without a global-optimality claim.
7. Save each observation before fitting, with source/kernel/count hashes,
   complete start/call accounting, numerical and descriptive diagnostics and
   explicit interrupted/failed records. Run serially with one numerical thread
   and separate 300-second production/audit case limits. Do not retry completed
   failures or overwrite old evidence. The current selected-case times imply
   several hours of bi-exponential work alone; measure an approved development
   pilot after the capacity decision before scheduling the full study.
8. Freeze the protocol and obtain separate approval before execution. CI runs
   deterministic controls only. Report every target miss without changing
   thresholds, settings, warning policies or denominators after inspection.
