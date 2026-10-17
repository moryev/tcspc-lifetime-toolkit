# Issue-12 scientific acceptance

Use the configured project interpreter from the repository checkout:

```bash
python scripts/run_issue12_scientific_reproducibility.py --week7 --reference issue12-post-correction
python scripts/run_issue12_scientific_reproducibility.py --week8 --reference issue12-post-correction
# Or run both independent workflows:
python scripts/run_issue12_scientific_reproducibility.py --all --reference issue12-post-correction
```

This explicitly invoked harness is not part of ordinary pytest. It calls canonical
library functions, blocks the 38 removable flat facades, and writes a compact JSON
report to a new system temporary directory printed at startup. It does not execute
notebooks, write databases, or update scientific reference artifacts.

Week 7 uses Notebook 12's 810-curve factorial design, seed 42, fixed 648/162 split,
and paired 10% secondary-component experiment. Week 8 uses the separate frozen
`week8-day55-v2` protocol: 64 development curves, seeds 50001–50006 for 192 external
observations each, and development-only PCA/estimator fitting. Week-7 mismatch is
not Test F. Test F retains its primary-component scoring reference and separate
descriptive signal-photon-weighted reference, not a Bayesian pseudo-true target.

## Reference policy

The authoritative current regression record is the versioned
[Issue-12 post-correction scientific reference](../data/reference/issue12_week7_week8_post_correction.json).
It records the reviewed fitter after parameter scaling, numerical convergence
validation and bounded half-bin recovery. It is **not updated notebook results**.

Notebook 12 cell `83354e7feeb8b834`, and Notebook 13 cells `e802c319fc9ab6ad`,
`93c3d55a39cf7b40` and `f5e40edbc2b8ea2b` remain historical pre-correction
evidence. They are read without evaluating source, and their outputs remain
untouched. `historical_notebook_comparison` reports their differences separately;
those classical differences do not fail current acceptance. The 79 established
ML comparisons remain an independent invariance check and must still pass.

`current_reference_acceptance` determines current regression PASS/FAIL. Exact
checks cover schemas, IDs, seeds, ordering, input fingerprints, manifests and
valid/failure populations (including failed sample IDs/reasons). The JSON stores
full-precision metrics and explicit absolute tolerances: 5e-7 for Week-7 metrics,
5e-5 for Week-8 errors/failure rates and 5e-4 for Week-8 degradation ratios,
retaining the established six/four/three-decimal reference precision. No relative
tolerance or blanket tolerance for scientific configuration is applied. New
paired and severity metrics use their week's corresponding error precision.
Dependency versions are provenance, not unconditional acceptance constraints.

Classical errors use valid fits only, always accompanied by total/valid/failed
counts. Paired error changes use complete accepted pairs, with their own count;
MAE ratios remain descriptive when the marginal populations differ. The complete
72-row A-F scorecard covers every ML representation, both baselines and classical
fitting. Test C compares the same observations under correct and deliberately
misspecified IRFs. Test F retains 96 weak and 96 moderate cases and distinguishes
primary-component from descriptive signal-photon-weighted references.

Timing is context only, never a frozen millisecond oracle. Structural checks cover
method order, batch/per-fit scopes, repetitions, warmups and finite nonnegative
measurements. Classical timings reuse the existing fits; ML fitting occurs outside
the prediction timing. Hardware-sensitive relative timings require contextual
review, not fixed regression thresholds.

## Independent repeat and candidate evidence

To verify exact deterministic reruns in the same environment, run the command
above twice, adding `--repeat-report <first-temporary-report>/acceptance.json` to
the second invocation. Each fresh process regenerates all data and fits. Reports
compare exact scientific summaries plus byte-level digests of metadata,
predictions and classical returned parameters/NLL; runtimes are excluded.
Environment equality is required only for this exact-repeat check, not for
ordinary current-reference acceptance.

For a separately authorized reference review, `--collect-candidate` replaces
`--reference issue12-post-correction`. It writes temporary evidence and reports
`CANDIDATE`, never acceptance PASS and never writes/overwrites the reference file.
Stage 3A.3 used two such independent matching runs before creating version 1, then
ran current-reference acceptance with the facade blocker. Future reference changes
likewise require explicit review; do not widen tolerances to hide discrepancies.

Reports record the commit, environment, selected reference/hash, protocols, input
fingerprints, metrics/populations, tolerances, historical differences and timing
context. Nonzero exit means a current discrepancy or workflow error. The two
weeks are independent; `--all` still reports the other week after a failure.

Lightweight pytest tests cover reference integrity, parsing, comparison boundaries,
population handling, repeat verification and the import blocker without running
the scientific workloads. `--all` remains Week 7 and Week 8 only.

## Stage 3B.1: Week-9 ML/conformal acceptance

```bash
python scripts/run_issue12_scientific_reproducibility.py --week9-ml --reference issue12-week9-ml
# Independent fresh-process repeat (use the first run's printed report path):
python scripts/run_issue12_scientific_reproducibility.py --week9-ml --reference issue12-week9-ml --repeat-report <first-report>/acceptance.json
```

This separate mode uses the version-1
[Issue-12 current Week-9 ML/conformal reference](../data/reference/issue12_week9_ml_uncertainty_post_correction.json).
The Week-7/8 reference and modes are unchanged. `--collect-candidate` is available
instead of `--reference` for explicitly authorized reference review; two independent
candidates must agree before a reference is created. The harness never writes it.

The protocol is 64 development observations (seed 42), split into 48 training and
16 calibration observations (split seed 57), followed by untouched A–F evaluation
(seeds 50001–50006, 192 observations each). Input fingerprints must match the
accepted Week-8 reference. Features are stateless. Shared Week-8 preparation fits
PCA on development only, but these uncertainty methods do not use PCA. Ridge's
learned scaling uses only the 48 training rows or their bootstrap resamples.

The three quantile HGB fits use 0.05/0.50/0.95 quantiles; RF spread is the population
standard deviation across 100 tree predictions; Ridge spread is the sample
standard deviation across 200 training-row bootstrap fits. Model/resampling seed
is 42. Instrumentation checks all 205 pipeline fits against the permitted rows and
exact seeded resamples, then rejects any refitting during calibration or external
evaluation. RF and Ridge spread remain **scores**, never calibrated intervals.

Conformal calibration remains in `benchmarks.uncertainty`: scores are
`max(lower - y, y - upper, 0)` on the 16 held-out development rows, using the
finite-sample rank `min(n, ceil((n + 1) * nominal_level))`. Here the rank is 16 and
the correction is zero. Re-evaluating the calibration rows is descriptive, not an
independent coverage test; external A–F results are reported separately. Neither
those results nor conformal construction establish coverage under arbitrary shift.

Reports retain factual point, interval and score identities, nominal levels,
attempted/valid/failed populations, paired response diagnostics and separate
96-row weak/moderate Test-F summaries. Test F is scored against the primary
component; its signal-photon-weighted reference remains separately identified
and descriptive, not a unique truth or a pseudo-true projection.

Exact independent-repeat checks cover scientific summaries and 99 fingerprints of
features, metadata, predictions/attachments, conformity scores and paired/residual
profiles. Current-reference comparisons use explicit named-metric tolerances of
5e-5, retaining the saved four-decimal Week-9 table precision; additional point,
pinball, paired and severity summaries use that same precision. Configuration,
nominal levels, seeds, identities, counts, failures and input fingerprints remain
exact. Versions are provenance, not mandatory pins.

Notebook 14 cells `910a6d01bbf11f44`, `339769e99c73f7e3` and `7dde16d16f42f0e6`
are read only as historical evidence. All 191 saved numerical comparisons match;
unexplained historical differences stop this workflow rather than silently
approving a new reference. No notebook is executed or modified.

The residual check runs **classical point fits only** for A/F (384 observations),
with the corrected current fitter. It reproduces the negative residual-structure
finding, not a model-validity guarantee. This mode does not call local covariance,
classical parametric bootstrap or repeated-Poisson workflows, does not touch the
Day-62 classical cache, and writes no database. Their scientific acceptance is
reserved for Stage 3B.2. Passing this mode does not complete Issue #12.

## Stage 3B.2B: full classical uncertainty profile

The explicit profile `issue12-week9-classical-full-v1` retains the historical
scientific scale, with the reviewed current fitter. These modes are intentionally
expensive and are **not** part of ordinary pytest:

```bash
python scripts/run_issue12_scientific_reproducibility.py --week9-classical-day61 --collect-candidate
python scripts/run_issue12_scientific_reproducibility.py --week9-classical-day61 --collect-candidate --repeat-report <day61-first>/acceptance.json
# Continue only after exact independent Day-61 agreement:
python scripts/run_issue12_scientific_reproducibility.py --week9-classical-day62 --collect-candidate
python scripts/run_issue12_scientific_reproducibility.py --week9-classical-day62 --collect-candidate --repeat-report <day62-first>/acceptance.json
```

Day 61 uses all eight Notebook-14 physical conditions in their published order,
seeds 61001–61008, 50 repeated measurements and 100 bootstrap refits per valid
primary. Within each condition **one RNG** produces a primary observation, its
bootstrap draws, then the next primary. Do not separate observation generation
from bootstrap execution. Day 62 evaluates all 192 observations in each frozen
A–F test, in order, with 200 refits per valid primary and **one global bootstrap
RNG seeded 62001**. Invalid primaries consume no bootstrap draws. These are serial
production workflows, not acceptance-script reimplementations of the algorithms.
Budget roughly 15–40 minutes per Day-61 pass and 1.5–4 hours per Day-62 pass;
timings are provenance, not regression thresholds. The external mode stops for
review if, after four hours, observed progress projects beyond six hours.

Observers record actual inputs, returned parameters/NLL, strict failures,
covariance diagnostics, bootstrap refit populations and optimizer recovery calls.
Detailed records go to the temporary report directory, while bootstrap draws and
individual refit results are digested rather than retained in large arrays.
Instrumentation never changes starts, draws, validity or the returned objects.
Continuation and half-bin fallback are identified from the three reviewed
optimizer call sites; changed call structure requires an instrumentation review.

Coverage means covered **valid intervals / valid intervals**. Reports also retain
all attempted observations and actual method evaluations, including skipped
invalid primaries. A bootstrap interval still needs only two successful refits,
as specified by production; requested/valid/failed refit counts, minimum support
and observations with at least 10% failed refits remain explicit. That 10% flag is
descriptive, not a new validity threshold. Repeated-Poisson variability remains an
empirical reference, never a per-point interval attachment. Test F uses the
primary-component reference; weighted lifetime is descriptive only.

The legacy Day-62 `fit_failure_rate` counts nonfinite predictions. It is retained
separately from strict primary-fit failure rate and IDs/reasons. Covariance can
fail at a boundary even when the primary is valid. Neither kind of unavailable
interval is silently removed from availability reporting.

Two independent complete passes must agree on deterministic records, input/RNG
fingerprints, populations and metrics before establishing the current reference.
There is no third expensive pass solely for reference comparison. After separate
review creates `data/reference/issue12_week9_classical_uncertainty_post_correction.json`,
compare the already-produced second evidence file read-only:

```bash
python scripts/run_issue12_scientific_reproducibility.py --week9-classical-day61 --reference issue12-week9-classical --compare-report <day61-second>/acceptance.json
python scripts/run_issue12_scientific_reproducibility.py --week9-classical-day62 --reference issue12-week9-classical --compare-report <day62-second>/acceptance.json
```

Omit `--compare-report` for an explicitly requested fresh scientific traversal;
`--week9-classical` selects both protocols. Candidate collection and reference
acceptance never create or overwrite the reference artifact. Notebook 14 and the
ignored Day-62 cache are neither executed nor updated.

The historical Day-61 findings table and Notebook-14 table are separate evidence:
they already differed before Issue #12, and the documentation table does not pin
the notebook's explicit RNG provenance. Later fitter changes can change both
fitted means used for bootstrap draws and primary validity. In Day 62, changed
validity decisions additionally shift the single bootstrap RNG stream (notably
after historical B/36). Historical cache equality is therefore not the current
regression oracle. Preserve those sources and distinguish pre-divergence checks,
aggregate historical interpretation and current candidate-to-candidate equality.

With 50 valid intervals, Day-61 coverage changes by 0.02. Descriptive 95% Wilson
intervals accompany coverage; they are not regression tolerances and do not define
success as an arbitrary small deviation from nominal 0.90. Exact fixed-seed
covered/valid counts and failure identities protect current regression behavior.

### Reviewed classical reference and offline acceptance

The current version-1 reference is
[`data/reference/issue12_week9_classical_uncertainty_post_correction.json`](../data/reference/issue12_week9_classical_uncertainty_post_correction.json).
Its scientific source is `b21e3411f85f68a4db8cd304277f0e65acd5e27d`, not the
future commit containing the reference. Separate `day61` and `day62` sections
retain complete compact summaries, exact protocol identities, strict/legacy
populations and fingerprints. No histograms or per-refit estimate vectors are
stored. Provenance includes historical/fitter commits, environment, hashes of
both evidence reports and exact-repeat results. Dependency versions are context,
not mandatory pins.

The approved complete passes took 20.48/19.70 minutes (Day 61) and
126.44/126.61 minutes (Day 62). The finalization used **offline comparison only**;
ordinary pytest and ordinary users are not expected to repeat this multi-hour
profile. The expensive commands above are for a separately requested scientific
rerun. Given completed second-pass reports, use `--compare-report` as shown above:
that branch never calls either full scientific workflow, and lightweight tests
enforce this. Candidate mode cannot create/overwrite the committed reference;
reference comparison reads it without mutation. Do not rerun science merely to
test the comparator.

Named absolute tolerances are explicit in the JSON. SDs, widths and scale ratios
use half a four-decimal unit (5e-5), consistent with the historical displayed
precision; point MAE/bias and empirical RMSE use half a six-decimal unit (5e-7),
consistent with the earlier Issue-12 point summaries. Coverage/failure fractions
allow only 1e-12 arithmetic rounding and retain exact integer numerators and
denominators. No tolerance applies to nominal coverage, IDs, seeds, order,
populations, failures or fingerprints. Exact result fingerprints intentionally
flag numerical/implementation drift for review even when aggregate differences
would lie inside displayed precision. There is no universal tolerance and no
test requiring every condition to achieve exactly 0.90 coverage.

Day-61 availability is 400/400 primaries and both interval types, with
39,999/40,000 valid bootstrap refits. Day-62 availability is 1,151/1,152 primaries,
1,150/1,152 covariance intervals and 1,151/1,152 bootstrap intervals, with
230,104/230,200 valid bootstrap refits. C ID 69 remains a strict primary failure;
F ID 165 remains a covariance boundary failure. Moderate Test-F coverage is
33/96 for covariance and 32/96 for bootstrap. See the appended current
[scientific findings](scientific_findings.md) for condition-level results and
interpretation. Historical differences (Day 61: 62/112; Day 62: 64/146) remain
informational; they do not turn an agreed current-reference comparison into a
regression failure. This stage does not declare Issue #12 complete.
