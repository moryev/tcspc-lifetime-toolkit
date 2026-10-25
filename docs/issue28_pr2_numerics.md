# Issue #28 PR 2 numerical development record

## Predeclared protocol

The experiment in `configs/issue28_pr2_initialization.json` was specified before
implementing or running the fitter (2026-10-24). It contains only new synthetic
development conditions. It neither imports nor regenerates Week 7-9 references.
The immutable condition ID and replicate ID extend the listed SeedSequence
entropy. Every grid is `linspace(0, window_ns, bins, endpoint=True)`.
Signal amplitudes are computed from the component sums at the generating shift
to achieve the declared **detected**, finite-window photon fractions.

Streams 0 and 1 alone select the default policy. Stream 2 is not generated or
evaluated until the selected settings and implementation hash have been recorded.
Held-out failures must be reported and investigated without using them to revise
the selected policy. Any later policy revision requires a newly declared
evaluation design; stream 2 cannot become held out again.

The pilot uses stream 0 of four declared conditions. Its purpose is to measure
optimization, validation, and diagnostic cost and find implementation defects.
The 46-start/100,000-objective limits are experimental ceilings. Validation and
diagnostics have separate hard evaluation budgets; their costs are reported.
All full-development attempts, failures, and retries are retained in the report.
No quality thresholds or difficult conditions are removed after seeing results.

The development comparison uses the same observations for all three policies.
A policy must pass local numerical validation on every development observation,
have no incomplete validation or optimization-budget exhaustion, and lie within
0.01 centered-NLL units of the best evaluated policy on every observation.
Eligible policies are ranked by median total elapsed time, then 90th percentile,
then total forward evaluations, then policy name. Repeat timings on the fixed
three-condition subset; record system/library versions and serial execution.
Multi-start agreement is not evidence of global optimality or physical validity.

Before the pilot, the production-budget rule is fixed as follows: take 150% of
the selected policy's maximum development usage, round objective calls upward
to 1,000, validation calls to 50, diagnostic calls to 10, and starts to an
integer (at least four). Cap each at its experimental ceiling. Initialization
retains its explicit 128-call cap. Verify these derived budgets on development
observations before freezing. They are not inferred from held-out observations.

Component recovery is reported only where generating components are distinct
and nonzero. For zero components, relative amplitude error and that component's
lifetime recovery are undefined. All attempts remain in numerical acceptance
denominators; conditional recovery summaries include their sample counts.

## Independent numerical qualification

The noise-free Issue-28 baseline uses 512 endpoint-inclusive coordinates from
0 to 20 ns, lifetimes 0.8/3.2 ns, detected fractions 0.4/0.6, 1e6 signal photons,
background 2/bin, shift 0.075 ns, Gaussian center 1 ns/FWHM 0.3 ns, lifetime
bounds 0.1-8 ns and shift bounds +/-0.3 ns. Required component relative error is
at most 0.001 and curve relative L2 error is below 1e-6. Background relative
error must be at most 0.001 and shift absolute error at most 1e-4 ns.
Noninteger expectations exercise a private numerical core, never the public
raw-observation API or full observed-count likelihood statistics.

An independent bounded trust-region residual optimization uses physical lifetime
coordinates, direct convolution, and independently calculated Poisson residuals.
Its designated well-conditioned development conditions are 0, 2, and 12. Both
solutions are evaluated with the shared centered objective; NLL agreement must
be within 0.01. This is an independent optimization audit, not a global proof.

The Issue-28 repeated high-count targets (at least 95% numerical acceptance,
median lifetime relative error below 5%, relative bias below 2%) remain unchanged.
The independent repeated-Poisson qualification and measurement integration belong
to PR 3 and are not claimed by this development experiment.

## Implementation and measurements

The first noise-free check exposed L-BFGS-B termination after finite differences
crossed an infinite-cost trial. The fitter now supplies bounded three-point
differences of finite objectives. Impossible line-search trials retain infinite
cost; their placeholder gradient cannot pass independent numerical validation.
No expectation floor or observation transformation was introduced. The baseline
then reached relative curve error 1.72e-8, within the unchanged 1e-6 target.

The first pilot (`data/generated/issue28_pr2_pilot_v1.jsonl`) contains all 12
attempts. Economical fits took 1.04-1.60 s, adaptive 3.07-22.11 s, and exhaustive
7.39-15.71 s on this workstation. All pilot endpoints passed local checks and
the economical objectives were within 0.01 of the best evaluated policy. One
adaptive run exhausted its start budget. These four observations do not select
the default. Optimization, validation and diagnostic calls/times are recorded
separately in each row.

The initial full-development attempt (`issue28_pr2_development_v1.jsonl`) was
stopped during condition 1; its 11 completed rows are retained. Review identified
that a better validation probe must be retained even if recovery cannot run.
The selected candidate now records its source, and a validation-selected trial
is not attributed to an optimizer termination. Validation also reports the
anchor parameters and largest detected descent in its declared neighborhood.
The held-out stream was not generated or inspected.

Diagnostic review corrected apparent full rank at exactly equal lifetimes:
the rank threshold includes the measured weighted difference between h and h/2
Jacobians, in addition to the usual floating-point SVD tolerance. The threshold
is returned explicitly. At shift interpolation knots a smooth Jacobian is
unavailable, even when central differences happen to agree across step sizes.
Neither rank nor a warning-free result certifies physical identifiability.

## Development-only policy selection

The completed second comparison contains all 120 fits (20 conditions, two
development streams, three policies). All attempts are included. The first
incomplete comparison remains a separate artifact, not pooled with this run.

| Policy | Median total seconds | 90th percentile seconds | Total forward calls | Eligible |
| --- | ---: | ---: | ---: | --- |
| Economical | 1.422 | 1.759 | 98,179 | No |
| Adaptive | 11.551 | 16.282 | 791,175 | No |
| Exhaustive | 13.295 | 16.916 | 885,963 | Yes |

These serial workstation times include initialization, optimization, local
validation and identifiability diagnostics. They are measurements, not runtime
guarantees. The comparison used Python 3.12.0, NumPy 2.5.1 and SciPy 1.18.0.

Economical initialization missed the 0.01 objective-gap target on four of 40
observations: close lifetimes (condition 3/stream 0, gap 0.1294), equal generating
lifetimes (4/1, 1.5435), zero secondary component (5/1, 220.6684), and 0.5%
secondary contribution (6/0, 4.4812). The zero-secondary case also failed local
validation. These are failures of that optimization policy; generating-model
degeneracy does not excuse a worse objective. Conversely, a good objective in
these regimes would not imply recoverable component parameters.

Adaptive fitting passed local validation on 40/40 observations and matched the
best objective, but exhausted its start allowance in five cases: (2/1), (4/1),
(5/0), (5/1), and (16/0). Recovery attempts use the same allowance as seed starts.
The predeclared eligibility rule therefore excludes it. No condition or
threshold was dropped to select a cheaper default.

**Exhaustive initialization is the selected default.** It passed local checks
on 40/40 observations without exhausted budgets, with maximum objective gap
3.13e-6. The measured maximum optimization usage was 36,803 forward calls.
The predeclared budget rule yields 46 starts, 56,000 optimization calls, 128
initialization calls, 100 validation calls and 100 diagnostic calls, with 1,000
optimizer iterations per start. These are finite work limits, not promises
that arbitrary observations will converge. The economical/adaptive policies
remain explicit options with their limitations reported here.

## Architecture and interpretation

`estimation.multiexponential` owns the result records and fitter; the PR-1
physical model and Poisson statistics are reused without edits. Amplitudes and
background use scaled linear coordinates that retain exact zero. Lifetimes
use independent log coordinates with shared explicit finite bounds. Shift uses
an affine coordinate on its supplied finite interval. Sorting is applied only
to results, with amplitudes and component curves reordered together. Equal
lifetimes are retained, with descending amplitude as the deterministic tie rule.

The initializer uses a deterministic five-point log-lifetime grid, three shift
positions and nonnegative least-squares amplitude/background seeds. Those seeds
are refined with bounded L-BFGS-B on centered Poisson NLL. Least squares does not
replace the final Poisson objective. All-zero observations have an exact
zero-expectation solution. Invalid inputs raise; numerical failures retain the
best finite evaluated candidate, including a better validation probe, with its
source recorded. When no finite candidate is available, estimate fields are
`None`.

Local validation probes the six scaled coordinate directions and amplitude
transfer/lifetime splitting directions at three step sizes. Its result refers
to the recorded anchor and a 0.01-NLL descent tolerance in this finite
neighborhood. A returned validation probe may improve that anchor within the
tolerance. Neither this check nor agreement between starts proves global
optimality. Budget exhaustion, unavailable diagnostics and optimizer termination
remain separate fields.

The weighted Jacobian uses scaled optimization coordinates and weights
`1/sqrt(expected_counts)`. Its singular values/condition number depend on these
coordinates and supplied bounds. Rank uses both floating-point tolerance and
the observed h-versus-h/2 derivative discrepancy. Zero expectations, shift
interpolation knots, unstable derivatives and exhausted diagnostic work can
make local information unavailable. Structural zero/equal-component warnings,
descriptive local-conditioning warnings and near-optimal-start disagreements
are separate. Configurable thresholds are not universal scientific criteria.

Prepared kernels preserve available preparation diagnostics and source records;
array kernels carry caller-declared assumptions. Both enter one validated
kernel path, with unit trapezoidal area required within rtol 1e-10/atol 1e-12.
Neither representation establishes experimental IRF correctness. AIC/BIC retain
six nominal fitted parameters even at degeneracies and do not constitute
automatic model selection or a regular-asymptotic validity claim.

The final-code derived-budget check passed all 40 development observations with
no exhausted budget. Across those fits, initialization took 2.37 s in total,
optimization 548.47 s, validation 1.16 s and identifiability diagnostics 0.82 s.
The counters include finite-difference forward calls. Most cost is optimization,
but the other phases remain bounded and measured.

The declared three-condition timing subset was repeated three times per policy
(27 fits). Each observation/policy reproduced its objective and evaluation
count. Economical total times were 1.06-1.34 s (median 1.276), adaptive
11.19-16.86 s (median 12.999), and exhaustive 11.11-16.23 s (median 12.732).
These timing repeats do not override the full-development eligibility rules.

### Noise-free recovery and independent audit

The selected settings satisfy every predeclared deterministic baseline target:

| Quantity | Measured error | Required bound |
| --- | ---: | ---: |
| Maximum component amplitude relative error | 2.81e-7 | 0.001 |
| Maximum component lifetime relative error | 1.61e-7 | 0.001 |
| Background relative error | 2.18e-5 | 0.001 |
| Shift absolute error | 2.93e-9 ns | 1e-4 ns |
| Expected-curve relative L2 error | 1.48e-8 | below 1e-6 |

The independent trust-region audit passed all three declared observations:

| Condition (stream 0) | Absolute centered-NLL difference | Relative curve difference |
| --- | ---: | ---: |
| Separated (0) | 8.70e-12 | 5.08e-9 |
| Long separated (2) | 5.92e-12 | 1.19e-8 |
| Million photons (12) | 4.03e-8 | 1.23e-7 |

Each independent audit used eight starts and a 1,500-`nfev` cap per start,
including finite-difference calls in its separately reported actual forward
count (552, 583 and 770). All 24 independent optimizer runs terminated
successfully. The comparison production fits used the experimental exhaustive
ceilings and matched the final-budget development-check objectives. Direct
convolution, physical lifetime coordinates and `scipy.special.kl_div` residuals
provide an optimization cross-check distinct from the production objective
gradient. These three well-conditioned audits are fast (0.12-0.18 s for the
independent solves), but do not qualify that solver across the full experiment
or authorize replacing the selected policy from this limited subset.

### Regression qualification

The final Python code passed **2,214 tests**, with one intentional reference-only
skip and three previously established warnings, in 566.76 s on Windows/Python
3.12.0. Combined statement/branch coverage was **83.37%**, above the unchanged
80% gate; the new fitter module reached **96.41%** combined coverage. All 90 new
fitter/diagnostic/experiment tests passed. The suite includes
the PR-1 statistics and forward-model tests, existing mono regressions, public
API and isolated-import guards, measurement/IRF paths and persistence tests.
The warnings were the two existing finite-invalid mismatch warnings and the
Windows physical-core-detection fallback. No reference regeneration was run.

Ruff passed; the formatting-baseline check passed with 130 unchanged audited
debt files. No baseline hash was refreshed. The static compatibility audit found
17 relevant modules/initializers AST-identical to `HEAD`, including both existing
mono fitter modules and the PR-1 forward/statistics modules. Existing package
exports, schema, frozen references, versions and release files are unchanged.
Structural guard changes add only the new module's required imports and update
the current inventory from 61 to 62; the small multiline parametrization change
is mechanical formatting of the extended list.

```bash
python -m pytest --durations=25 --junitxml=build/reports/issue28_pr2_pytest.xml --cov=tcspc_toolkit --cov-branch --cov-report=term-missing --cov-report=json:data/generated/issue28_pr2_coverage_v1.json
python -m ruff check src tests examples scripts
python scripts/check_format.py
git diff --check
```

The 20 held-out observations are evaluated only after this successful regression
run and the policy lock. The frozen fitter SHA-256 is
`4ac3d3e9b7a7904c91f2b169cf0ef7e5a56185de3a04049846a084dfdc44a332`,
and the frozen runner SHA-256 is
`e8b91bbf8138076a06775d33fba910236ab5b02b3ac1b4a54ca0d8a7c4bb068f`.
No fitter or runner changes occurred between freezing and the original held-out
evaluation. The later initialization correction and its separate regression
verification are recorded below.

### Frozen-policy held-out evaluation

All **20/20** stream-2 observations passed local numerical validation, with no
exhausted budgets. Maximum optimization usage was 42,171 calls (the 1,000-photon
condition), below the frozen 56,000-call cap; initialization/validation/diagnostic
maxima were 92/49/72. Total forward calls were 447,985. Median elapsed time was
12.626 s, 90th percentile 17.270 s, and maximum 25.499 s. The frozen policy and
source hashes were checked again after evaluation and remain unchanged.

The separated held-out baseline returned 0.647738/2.787607 ns, and the
million-photon case 0.650356/2.805977 ns. Poor component recovery remains visible:
the 0.5%-secondary case missed the secondary lifetime by 36.2%, the 100-photon
case had a zero fitted component, and the 1,000-photon case returned
1.19749/9.0 ns. The 10%-secondary case missed the second lifetime by 10.18% with
no descriptive warning. A warning-free fit is therefore explicitly **not** a
recovery or physical-identifiability certificate. Equal generating lifetimes
and zero generating amplitudes have no component-wise relative-recovery scores.

These observations did not select or alter settings. The post-hoc independent
investigation of conditions 8 and 10 is recorded separately from the predeclared
three-case audit and from all selection evidence; it diagnoses poor recovery,
not a new policy comparison. The held-out run itself supplies local numerical
checks, not an exhaustive global-optimum comparison on unseen observations.

The independent solver agrees on condition 8 within 4.15e-11 NLL and on
condition 10 within 5.69e-14 NLL (relative curve differences about 1e-8). All
16 independent starts terminated successfully. The fitted centered NLLs are
195.8734 and 224.7605; evaluating the generating parameters gives the worse
values 200.3699 and 229.5639 on these noisy observations. This supports a
sampling/information limitation in these two recovery examples rather than a
detected optimization failure. It is conditional on this synthetic model and
does not establish a global proof, experimental validity, or calibrated
component uncertainty. No settings or warning thresholds were adjusted.

### Descriptive development examples

On the final-code development check, the separated 100,000-photon condition
returned lifetimes 0.64275/2.81893 ns and 0.65402/2.79623 ns across its two
streams (generating 0.65/2.8 ns). The million-photon condition returned
0.650147/2.799384 ns and 0.650649/2.800622 ns. These small examples report
observed recovery; they are not the PR-3 bias or acceptance study.

The zero-secondary condition in stream 0 returned a zero amplitude, Jacobian
rank 5, and an unconstrained second lifetime; assigning a relative recovery
error to that lifetime would be misleading. With close generating lifetimes,
stream 0 returned a weak component at the 9-ns bound and stream 1 a zero
component. Both had near-optimal lifetime/contribution disagreements despite
passing the local numerical check. In the 100-signal-photon condition, one
solution reached 9 ns and another 0.08 ns; both showed near-optimal disagreement.
The 0.5%-secondary condition's second-lifetime errors were approximately 20% and
100% while local numerical validation passed. These are recovery limitations
under weak information, not a reason to relabel numerical convergence as
physical identification or suppress the observations.

Tests also force false optimizer-success flags and misleading cached objectives,
retain a better failed-run candidate over a worse successful one, exhaust each
work budget explicitly, preserve exact equal/zero solutions, and mark unavailable
derivatives at interpolation knots. All-zero observations produce an exact
zero curve, undefined component fractions, and no optimizer-success claim.

## Reproduction and evidence

Run from the repository root with the configured development environment.
Choose unused output names: every experiment artifact is exclusive-create.
The development commands use only streams 0/1; the final command unlocks stream
2 only when configuration, fitter and runner hashes match the policy lock.

```bash
python scripts/benchmark_biexponential_initialization.py pilot --output data/generated/issue28_pr2_pilot_v1.jsonl
python scripts/benchmark_biexponential_initialization.py development --output data/generated/issue28_pr2_development_v2.jsonl
python scripts/benchmark_biexponential_initialization.py select --source data/generated/issue28_pr2_development_v2.jsonl --output data/generated/issue28_pr2_selection_v1.json
python scripts/benchmark_biexponential_initialization.py development-check --source data/generated/issue28_pr2_selection_v1.json --output data/generated/issue28_pr2_development_check_v1.jsonl
python scripts/benchmark_biexponential_initialization.py timing --output data/generated/issue28_pr2_timing_v1.jsonl
python scripts/benchmark_biexponential_initialization.py audit --output data/generated/issue28_pr2_audit_v1.jsonl
python scripts/benchmark_biexponential_initialization.py noise-free --source data/generated/issue28_pr2_selection_v1.json --output data/generated/issue28_pr2_noise_free_v1.jsonl
python scripts/benchmark_biexponential_initialization.py freeze --source data/generated/issue28_pr2_selection_v1.json --validation data/generated/issue28_pr2_development_check_v1.jsonl --output data/generated/issue28_pr2_policy_lock_v1.json
python scripts/benchmark_biexponential_initialization.py held-out --source data/generated/issue28_pr2_policy_lock_v1.json --output data/generated/issue28_pr2_held_out_v1.jsonl
```

Timing requires serial execution without concurrent test or experiment jobs.
The final code can reproduce the declared comparison, while prototype pilot
and interrupted-attempt hashes describe earlier implementation states. Do not
silently replace those historical attempts with a final-code rerun. The
development budget check establishes the final-code result before freezing.

The compact [results and hash ledger](validation/issue28_pr2_results.json)
contains environment metadata, selection details, numerical audits, coverage,
compatibility checks and every raw artifact's hash. The
[observation table](validation/issue28_pr2_observations.csv) retains all **230**
completed fits: 12 pilot, 11 interrupted-attempt, 120 full-development,
40 derived-budget checks, 27 timing repeats and 20 held-out observations.
These are purpose-labeled attempts, not 230 independent replicates. Full
per-start records remain in the ignored local `data/generated/` JSONL files;
the tracked table retains each fit's outcomes, recovery quantities, work/time
counts, optimizer-status counts and diagnostic warnings/ranges. In JSON,
nonfinite numbers are encoded as `null`; rank, availability reasons and warning
fields distinguish singular/undefined cases. Empty CSV fields are unavailable
quantities, not zeros. The post-hoc held-out audit is an additional labeled
record, never selection input.

## Final scope and review

| Files | Change |
| --- | --- |
| `src/tcspc_toolkit/estimation/multiexponential.py` | One shared-kernel fitter, parameter/settings/result records, bounded optimization and grouped diagnostics. |
| `tests/test_biexponential_reconvolution.py`, `tests/test_biexponential_diagnostics.py`, `tests/test_biexponential_experiment.py` | 90 initial numerical, physical, degeneracy, invalid-input, failure and experiment-isolation tests; 100 after the initialization correction. |
| `configs/issue28_pr2_initialization.json`, `scripts/benchmark_biexponential_initialization.py` | Predeclared conditions/criteria, deterministic experiment, policy selection/locking and independent audits. |
| `docs/api.md`, this record, `docs/validation/issue28_pr2_results.json`, `docs/validation/issue28_pr2_observations.csv` | API contract, qualification evidence, assumptions and limitations. |
| `tests/test_package_structure.py`, `scripts/check_distribution.py`, `tests/test_release_tooling.py` | Narrow import/inventory guard updates for one added module. |
| `docs/package_structure.md`, `docs/release_readiness.md`, `AGENTS.md` | Current inventory and approved-task orientation; historical qualification remains historical. |

All declared PR-2 deterministic recovery and independent-objective targets were
met. The cheaper policies' failures remain reported. Local numerical acceptance
does not eliminate the documented component-recovery limitations, the absence
of calibrated uncertainty, or the cost of exhaustive initialization. The broad
Issue-28 repeated high-count acceptance/median-error/bias targets remain
unchanged and **not yet qualified**; that independent study is PR 3.

At the original implementation handoff, PR 2 remained reviewed and uncommitted
on `main`; no Git publication, release change, Issue-13 work or PR-3 implementation
had been performed. Existing mono/PR-1 source, curated exports, schema v1 and
frozen scientific references remain unchanged. Partial PRs use `Refs #28`.

## October 25 initialization correction

The final independent review identified an initialization failure outside the
zero-origin development cases. With `time_ns = np.linspace(-1, 5, 64)`, a
unit-area Gaussian kernel, one count per bin, lifetime bounds `(0.001, 8)` and
shift bounds `(-0.2, 0.2)`, a generated short-lifetime template overflowed before
objective screening. The resulting `ValueError` aborted the fit despite a
supplied exact solution: amplitudes `(0, 0)`, lifetimes `(1, 2)`, background `1`
and shift `0`. A sufficiently small lower lifetime bound could also overflow
the preliminary amplitude-scale reference before supplied guesses were tried.

The correction is confined to initialization failure handling and its reporting.
Floating-point overflow/invalid operations raise within generated model and
linear-projection trials; nonfinite predictions/projections are explicitly
rejected. Only `FloatingPointError` is newly caught, so unrelated validation or
programming exceptions propagate. The existing caller-input and initial-guess
checks remain unchanged. A failed scale reference keeps the original finite
count-based scale. The shared forward model, absolute time origin, optimizer,
policy, tolerances, thresholds and all numerical work limits are unchanged.

Every attempted model call still consumes initialization budget, including
failed calls. `starts_screened` now includes templates rejected before their
objective can be evaluated; `initialization_rejected_candidates` identifies
rejected candidates, and `initialization_scaling_failed` separately identifies
a failed scale reference. Exhaustion propagates through the existing budget
path. If no finite candidate exists, parameters/curves/statistics remain
unavailable with `no_finite_candidate`, no optimizer-success claim, and no
optimization/validation/diagnostic calls.

Four physical regression cases failed on the pre-correction source with
`decay must contain only finite values`: the reported example, the scale-reference
variant, and all-overflow cases with initialization budgets 5 and 128. All pass
after correction. Ten added cases also cover nonfinite predictions, propagation
of unrelated errors, and preservation of valid candidates under exhausted work
budgets. In the all-overflow example, the unrestricted run charges 47 calls
(one reference, 45 templates, one background candidate) and rejects 46 candidates;
a five-call budget charges exactly five calls and rejects four candidates.

Before editing the fitter, initialization states were saved for all 40 designated
development observations (20 conditions, streams 0/1). After correction, ordered
candidate coordinates, best objective/parameters/curves/components, amplitude
scale, screened counts, phase evaluation counts and exhausted-budget flags match
**exactly**. The two new diagnostics are zero/false on every case. Settings and
the optimizer, numerical-validation and identifiability implementations are
AST-identical. No held-out observation was used in this correction audit and
the initialization-efficiency experiment was not rerun. Elapsed times are not
claimed to be identical or requalified by this comparison.

The original configuration, experiment runner, observation table, results/hash
ledger and policy lock are preserved. Their results remain evidence for the
recorded pre-correction implementation, not a relabeled run of the new source.
The fitter SHA-256 changed from
`4ac3d3e9b7a7904c91f2b169cf0ef7e5a56185de3a04049846a084dfdc44a332` to
`b89aebfcf4e9dba8cccd0758a48b47a621c685e90a0d68d024a273677409674f`.
Consequently the old held-out lock correctly rejects the corrected source;
it must not be rewritten to imply a new held-out evaluation. The separate local
audit files are `data/generated/issue28_pr2_initialization_fix_before.json`
and `data/generated/issue28_pr2_initialization_fix_comparison.json`.

Correction validation (Windows, configured Python 3.12.0):

- Focused initialization, failure and budget checks: **17 passed**.
- All PR-2 fitter, diagnostic and experiment tests: **100 passed**.
- Affected mono/PR-1 forward, statistics and classical tests: **287 passed**.
- Complete repository suite: **2,224 passed, one intentionally skipped** in
  494.32 seconds, with **83.40% combined statement/branch coverage**; the fitter
  has **96.55% combined coverage**. The two established finite-invalid mismatch
  warnings and Windows CPU-count fallback warning remain non-failing.
- Ruff, unchanged-debt formatting baseline and whitespace checks passed.

The full-suite JUnit and coverage XML/JSON files use the separate
`build/reports/issue28-pr2-initialization-fix-*` names. The previous qualification
ledger is not overwritten. Existing mono/PR-1 implementation, package-initializer
APIs, frozen scientific references, schema and release metadata remain unchanged.
The correction handoff remained uncommitted on `main`. Publication was later
authorized separately, subject to the post-correction consistency audit below;
PR 3 remains unauthorized.

## October 25 post-correction consistency audit and publication

After accepting the initialization correction, the maintainer authorized a
replay of the 20 previously evaluated stream-2 observations as a publication
gate. This is **post-correction regression verification**, not a new untouched
held-out evaluation and not policy-selection evidence. No settings, bounds,
budgets or warning thresholds were selected or tuned from this replay.

The separate [verification script](../scripts/verify_biexponential_correction.py)
checks the historical artifact hashes against the original ledger, the original
source/runner/configuration correspondence, the exact frozen settings and the
reviewed corrected-source hash. It reconstructs the observations using the
unchanged configuration/runner and verifies every historical fitted objective
on those counts before replaying the fit. The original runner's held-out mode
and source-hash validation are unchanged. Both new output files use exclusive
creation; the original policy lock and qualification artifacts are never rewritten.

The [dated compact comparison](validation/issue28_pr2_post_correction_2026-10-25.json)
records the source/artifact hashes, environment, frozen settings, observation
hashes, bounds, per-case outcomes and timing differences. All **20/20** cases
match **exactly** across every historical non-timing result field, including
objective values, fitted parameters, local numerical validation, optimizer
status, warnings, conditioning and disagreement diagnostics, all per-start
records, evaluation counts, stopping reasons and budget status. Maximum absolute
centered-NLL difference is **0**. The new rejected-initialization count is zero
and the scaling-failure flag is false throughout.

All cases again passed local numerical validation, with no exhausted budget.
Total forward calls remain **447,985**, and maximum optimization usage remains
**42,171** of 56,000. The warning-free 10.18% secondary-lifetime error is reproduced
unchanged; the replay does not improve or reinterpret its statistical recovery.
Elapsed times are intentionally compared separately: historical/replay medians
are **12.626/12.691 s**; replay 90th percentile/max are **16.821/25.873 s**, and
the 20 fits total **270.482 s**. Runtime variation does not change any evaluation
count or numerical outcome. No substantive discrepancy required investigation.

Reproduce with the recorded local historical artifacts and unused output names:

```bash
python scripts/verify_biexponential_correction.py --output docs/validation/issue28_pr2_post_correction_2026-10-25.json --raw-output data/generated/issue28_pr2_post_correction_2026-10-25.jsonl
```

The approximately 60-kB compact audit is included in PR 2; detailed per-start
replay logs remain ignored under `data/generated/`. Comparator smoke checks
confirm detection of objective, lifetime, warning, validation, count and budget
changes while excluding only elapsed-time fields and the explicit replay label.
The corrected fitter retains its previously validated source hash and the
2,224-pass/one-skip/83.40% full-suite result; the full suite was not repeated for
publication. Ruff, the formatting baseline and staged whitespace checks are
publication gates. PR-2 branch creation, commit, push and pull-request opening
are authorized after this passing audit; merge and PR 3 require later approval.
