# Proposed issue: investigate bi-exponential optimizer premature termination

**Draft only; not created on GitHub. Refs #28.** Maintainer-approved Disposition A
preserves the completed PR-3 study and its failed independent agreement target.
This document does not authorize a replay, fitter change or new qualification.

## Evidence and problem

C00 r11/r58 have production centered NLLs 1228.441945/331.964882 versus independent
audit values 247.844923/259.114084: gaps **980.597023/72.850798**, exceeding the
unchanged .01 target. Both were rejected by local numerical validation. The
primary C00 acceptance/recovery/bias targets passed; designated audit agreement
failed at 119/121. Preserve that distinction and all original denominators.

Production r11 used 45 seeds plus one recovery; r58 used 46 seeds with no recovery
slot. All starts reported optimizer success, but r58 start 23 had a nonfinite
endpoint and projected-gradient termination. Other r58 starts and all r11 starts
reported relative-function-reduction termination. Large analytic amplitude and
background derivatives at the returned interior candidates support premature
termination. The callback returns a zero placeholder gradient at nonfinite cost;
this is a hazard to investigate, not a proven cause of either selected-fit gap.
The exact solver-level cause remains unresolved. Independent audit agreement
across eight starts does not prove global optimality.

The 46-start cap includes recovery. Candidate-bank completion can leave no
recovery slot despite unused objective evaluations and no exhausted-budget flag.
Review reporting of this distinction without silently increasing the frozen
budget or redefining numerical acceptance.

## Proposed bounded diagnostic replay

Subject to separate authorization, replay only these recorded starts:

| Observation | Start index | Kind |
| --- | --- | --- |
| C00 r11 | 10 | Best seed |
| C00 r11 | 45 | Validation recovery |
| C00 r58 | 5 | Best seed |
| C00 r58 | 23 | Nonfinite endpoint / projected-gradient termination |

Use the saved observations, original prepared-kernel hash, recorded parameter
scales, bounds, settings and environment. Cap each replay at **2,000 forward
calls and 300 seconds**; preserve cap exhaustion without retries. Record all
objective/gradient evaluation coordinates, nonfinite costs, derivative steps,
line-search trial/accepted points where exposed, iteration/evaluation counters
and termination messages. Check callback derivatives against independent
derivatives at finite logged points. The original internal coordinates were not
saved: check reconstruction differences before claiming bit-identical replay.
Keep this diagnostic artifact outside the original qualification directory.

## Acceptance tests for any eventual correction

- Establish a reproducible cause before selecting a correction; preserve the
  original failures and add focused regression coverage for that mechanism.
- Nonfinite endpoints must never become numerically accepted results, regardless
  of optimizer status. Check finite objective/gradient consistency, best-candidate
  retention and independently validated feasible descent.
- Explicitly test start-cap/recovery-slot accounting, objective/validation limits,
  and honest incomplete/failure diagnostics. No hidden retries or uncharged calls.
- Recover the saved C00 regression cases to within .01 of the independently
  recorded objectives under explicitly declared corrected settings, or retain and
  investigate any miss. These are now regression/development cases, not untouched
  qualification observations or global-optimality certificates.
- Preserve zero components/background, equal lifetimes, physical shift/scaling
  conventions, IRF provenance, raw-count guards and all mono/PR-1 contracts.
- Any modified fitter requires **new, versioned qualification**: predeclare source,
  settings, seeds, evaluation budgets, numerical/recovery targets and independent
  audit rules before generating fresh evaluation observations. Do not tune on the
  old PR-3 study and relabel it held out, relax targets, or overwrite old locks.

See the [completed scientific record](issue28_pr3_qualification.md) and
[machine-readable evidence](validation/issue28_pr3_results_2026-10-27.json).
