# Issue-12 Week-7/8 acceptance

Use the configured project interpreter from the repository checkout:

```bash
python scripts/run_issue12_scientific_reproducibility.py --week7
python scripts/run_issue12_scientific_reproducibility.py --week8
# Or run both independent workflows:
python scripts/run_issue12_scientific_reproducibility.py --all
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

The numerical oracles are **saved outputs**, read without evaluating notebook
source: Notebook 12 cell `83354e7feeb8b834`, and Notebook 13 cells
`e802c319fc9ab6ad`, `93c3d55a39cf7b40`, and `f5e40edbc2b8ea2b`.
The reports include notebook hashes and dependency versions. Exact checks cover
IDs, counts, ordering, and repeated deterministic arrays/predictions. Metric
tolerances are half the displayed last decimal unit: 5e-7 for Week-7 six-decimal
metrics, 5e-5 for Week-8 four-decimal MAE/failure rates, and 5e-4 for three-decimal
degradation. These are reference-precision checks, not universal cross-version
numerical guarantees. Timings are not compared to exact historical milliseconds.

A discrepancy exits the affected workflow before subsequent phases and records
FAIL; `--all` can still inspect the other independent week. Do not widen tolerances
or regenerate the saved tables to make acceptance pass. In particular, the saved
Week-7/8 classical analyses predate the documented Day-61 parameter scaling and
later Issue-4 Poisson convergence validation. Any resulting difference requires
an explicit scientific-reference acceptance decision, not a silent rebaseline.

Normal lightweight harness tests validate reference-table parsing, the import
blocker, comparison boundaries, and failure-report serialization. They do not run
the full scientific benchmarks. Week-9 acceptance is outside this harness.
