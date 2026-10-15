# Issue-12 Week-7/8 acceptance

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
the scientific workloads. Week-9 acceptance is outside this harness.
