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
