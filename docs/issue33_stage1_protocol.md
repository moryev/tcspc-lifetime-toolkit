# Issue #33 Stage 1: predeclared bounded diagnostic replay

October 27, 2026. The maintainer authorized only the four starts in Issue #33.
This is forensic regression work on already reviewed observations, not a new
qualification or an untouched evaluation. No production code, historical record,
setting, warning threshold or qualification criterion may change.

## Inputs and reconstruction

Use C00 r11 starts 10/45 and r58 starts 5/23, in that order. Require the original
66-file source/settings/environment lock, checkpoint envelopes and whole-directory
inventory digest to match. Use saved counts, grids, physical initial parameters,
parameter scales, bounds and settings. The raw records save an assumed-IRF hash,
not a kernel array: reconstruct only that deterministic Gaussian prepared kernel
through the locked implementation and require exact equality to its saved hash.
Never generate observations or rerun the initialization bank or other starts.

Convert saved physical parameters with the unchanged `coordinates()` method and
clip to the original bounds, as the original initialization does. Record the
unclipped/clipped coordinates and binary64 hex, physical round-trip differences,
and adjacent floating-point coordinates that map to the same stored parameter.
These quantify reconstruction ambiguity; original internal coordinates and
iteration traces were not saved, so matching endpoint summaries alone cannot
establish a bit-identical historical trajectory.

## Instrumentation and work limits

Call the unchanged `_refine()` and `_Problem.value_and_gradient()`; delegate all
arithmetic to the originals. Temporary in-process wrappers record objective
probes, gradient callbacks, actual forward calls and SciPy `setulb` transitions.
They add no optimizer callback, numerical probe, option or retry during solving.
Copy arrays when logging, never mutate solver-owned buffers. Record exposed
task/line-search codes and raw `isave`/`dsave` arrays; internal line-search steps
that do not return to Python are unavailable. Separate solver requests from the
finite-difference probes and the final endpoint re-evaluation.

One common hard cap is **2,000 actual model calls per replay**, including all
post-replay derivatives and validation. A shared component-count invocation is
one call (the production convention). Independently calculated analytic
derivatives use six direct convolutions per point; conservatively charge each
convolution separately. Do not reserve calls by changing the original optimizer
limit: stop at the common cap wherever reached. Preserve partial results.

The parent gives each child **300 seconds from process launch**, including imports,
input verification and post checks, and terminates the child on timeout. The
child also checks elapsed time before evaluations and solver transitions. Use
one numerical thread. Exclusive attempt markers prevent automatic retries or
overwriting any completed, exhausted or interrupted replay.

Raw JSONL events go only to a new `data/generated/issue33_stage1_20261027/`
directory. Each event includes its sequence number, stage and actual-call count.
Record coordinates, feasibility, objective, gradient, curve hash/range/zero-bin
summary, best-candidate updates, solver status, nit/nfev/njev and exposed states.
Nonfinite numbers have explicit string encodings. Flush events so interruptions
preserve prior evidence. Compact results and hashes go under `docs/validation/`.

## Post-replay numerical checks

Freeze the replay's best point and solver result before diagnostic probes. Run
the unchanged local validator once at that best point as a **separate diagnostic**;
never feed its improved point to another optimization start. Use a separate
problem instance for derivative checks, with identical scales/settings. Check
the first finite solver callback, last finite callback, retained best, saved
production result and saved audit result, in that order. Retain coincident points
under their distinct roles and report any omitted checks on exhaustion.

Independent derivatives differentiate the direct discrete convolution, using
`d exp(-t/tau) / d log(tau) = exp(-t/tau)*t/tau`, exact linear amplitude/background
derivatives, and the piecewise-linear interpolation slope for shift. Combine with
the Poisson score `1-y/mu`. Record interpolation knots, zero expectations and
direct-versus-shared curves/objectives; a derivative at a knot or impossible
expectation is not a smooth stationarity certificate. Report the original finite
differences alongside this independent calculation, without tuning thresholds.

## Integrity and interpretation

Before executing any authorized replay, freeze the harness, tests, this protocol,
configuration, SciPy wrapper/binary hashes, environment, historical lock and input
hashes in a separate dated manifest. Synthetic non-qualification control tests
must demonstrate identical objective/gradient sequences, results and model-call
counts with passive instrumentation enabled/disabled, plus strict cap/timeout,
no-overwrite and derivative correctness checks.

Compare each replay to its saved start's best/endpoint objectives and call count,
and to the saved final production/audit objectives. Distinguish demonstrated
behavior in reconstructed replays from an inferred historical explanation.
Independent-audit values are comparison points, not certified global optima.
The isolated start does not restore the full bank's global-best state; this
state affects retention, not the objective or L-BFGS-B arithmetic.

Only after the four attempts, inspect traces and write a separate findings report.
No correction, additional replay, new qualification or Git publication is
authorized by this protocol. Refs #33 and #28.
