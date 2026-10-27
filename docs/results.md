# Scientific results: evidence and interpretation

The [educational examples](../examples/README.md) demonstrate API use on small
inputs. Their printed estimates are **not** the established benchmarks below.
All reported evidence here comes from controlled synthetic data; it does not
establish performance on acquired TCSPC measurements.

## Which evidence is current?

Issue #28 PR 3's [completed independent study](issue28_pr3_qualification.md)
preserves all 1,850 paired comparisons and 205 audit attempts. C00 numerical
acceptance (98/100), both component median errors and both biases passed their
targets. Independent objective agreement **failed in two designated cases**
(119/121 passed); both production fits were rejected by local validation.
Low counts, weak components, short windows and wrong IRFs can still yield poor
recovery in accepted, warning-free fits. Inspect the complete
[summary JSON](validation/issue28_pr3_results_2026-10-27.json) and
[attempt CSV](validation/issue28_pr3_results_2026-10-27.csv). The accepted reporting
disposition retains these limitations and a proposed numerical follow-up; it
does not claim all qualification targets passed. This study does not replace
the historical evidence below.

For Weeks 7–9, the authoritative current regression records are the three
committed Issue-12 post-correction references:

- [Week 7/8](../data/reference/issue12_week7_week8_post_correction.json): current
  Poisson fitter, paired mismatch, and frozen A–F point-estimation scorecards.
- [Week-9 ML uncertainty](../data/reference/issue12_week9_ml_uncertainty_post_correction.json):
  direct/conformal quantile intervals and RF/training-bootstrap scores.
- [Week-9 classical uncertainty](../data/reference/issue12_week9_classical_uncertainty_post_correction.json):
  repeated-Poisson calibration and frozen A–F covariance/bootstrap analysis.

These records preserve populations, references, seeds, environment provenance,
and acceptance tolerances. See the [acceptance policy](issue12_scientific_acceptance.md#reference-policy)
and [final ledger](issue12_scientific_acceptance.md#stage-5-final-technical-acceptance).
Historical Notebook-12/13/14 outputs remain intact; some pre-correction classical
numbers differ from these references. Do not substitute them silently.

Issue-8 and Issue-4 findings retain their own study provenance. For Bayesian
mismatch, the Stage-6.5-corrected deterministic references documented in
[scientific findings](scientific_findings.md#deterministic-reference-audit--stage-65)
are authoritative. Those are separate studies, not extensions of frozen A–F.

## Point estimation and distribution shift

Frozen Week-8 evaluation uses 64 development observations and 192 observations
per external test, with test seeds 50001–50006. Development fitting and tuning
remain separate from external A–F data. A is familiar distribution, B changes
photon counts, C broadens the IRF, D raises background, E increases misalignment,
and F introduces controlled bi-exponential mismatch.

| Method | Representation | Test | MAE (ns) | Contributing / attempted | Caveat |
| --- | --- | --- | ---: | ---: | --- |
| Poisson reconvolution | Raw histogram | A | 0.082750 | 192 / 192 | Matched mono-exponential model and IRF |
| Random Forest | Engineered features | A | 0.149010 | 192 / 192 | Synthetic development support |
| Random Forest | Engineered features | B | 0.428854 | 192 / 192 | Aggregate of low/high photon OOD regimes |
| Poisson reconvolution | Raw histogram | C, correct IRF | 0.096351 | 191 / 192 | Accepted-fit population; ID 69 remains optimizer-unsuccessful |
| Poisson reconvolution | Raw histogram | F | 0.156723 | 192 / 192 | Scored against primary-component lifetime, not unique mono-exponential truth |

Values are rounded from the `weeks.week8.scorecard` and classical population
records in the [Week-7/8 reference](../data/reference/issue12_week7_week8_post_correction.json).
Classical error summaries above use accepted fits. In contrast, the retained
Week-9 legacy Test-C point MAE is 0.095926 ns over **finite predictions**, including
the invalid fit. These are intentionally different populations.

For Test C, assuming 0.4 ns FWHM for observations generated with 0.6 ns FWHM gives
MAE 0.116739 ns over 192 accepted fits. The comparable paired analysis has 191
accepted pairs and mean absolute-error increase 0.020550 ns. Comparing the two
marginal MAEs alone would hide their different populations.
The [full findings](scientific_findings.md#issue-12-post-correction-week-78-acceptance-reference)
retain all A–F results, paired analyses, and Week-7's separate mismatch protocol.

## Statistical uncertainty under model mismatch

For frozen A–F, the current Week-9 classical reference uses nominal 90% intervals
and 200 bootstrap refits per valid primary fit. Coverage is **covered valid intervals / valid
intervals**, with availability recorded separately.

| Method | Test | Valid / attempted intervals | Coverage | Mean width (ns) |
| --- | --- | ---: | ---: | ---: |
| Local Poisson/Fisher covariance | A | 192 / 192 | 0.890625 | 0.332571 |
| Parametric Poisson bootstrap | A | 192 / 192 | 0.885417 | 0.327137 |
| Local Poisson/Fisher covariance | F | 191 / 192 | 0.518325 | 0.366182 |
| Parametric Poisson bootstrap | F | 192 / 192 | 0.494792 | 0.360921 |

Test F combines 96 weak and 96 moderate mixtures, scored against the generating
primary component. F/165 has an invalid boundary covariance result. The weighted
lifetime remains descriptive; no Bayesian pseudo-true projection is used here.
See the [current classical findings](scientific_findings.md#issue-12-current-full-classical-uncertainty-acceptance-stage-3b2b)
for severity, failures, and repeated-Poisson calibration.

The ML uncertainty study uses a different split: 48 training and 16 calibration
observations. Broad direct/conformal quantile intervals cover all 192 targets in
each external test; mean width on A is 2.4583 ns. The conformal correction is zero
because all calibration targets already lie inside the base intervals. This
finite-suite coverage does not establish arbitrary-shift guarantees.
Its RF point MAE on A is 0.1799 ns, **not** the Week-8 RF result above.
RF tree spread and Ridge training-bootstrap spread remain ranking/sensitivity
scores. Details are in the [ML/conformal findings](scientific_findings.md#issue-12-current-week-9-mlconformal-acceptance-stage-3b1).

**An estimator can correctly quantify statistical uncertainty while remaining
confidently wrong because its physical model is incomplete.** The modest interval
widening under F does not compensate for model-induced bias. Bootstrap generates
data from the same fitted model; it does not supply model-form uncertainty.

## Generalized IRFs and Bayesian inference

The [Issue-8 findings](scientific_findings.md#post-week-9--issue-8-generalized-irfs-and-failure-awareness)
show that matching an IRF's peak and full FWHM does not remove shape-family bias.
Leading-edge proxies can be numerically constructible and physically poor.
Single-realization proxy controls, a three-bootstrap demonstration, and a small
discrete-target ML fixture are not general robustness or coverage evidence.

The [Bayesian study](scientific_findings.md#post-week-9--issue-4-bayesian-poisson-inference-and-model-conditional-uncertainty)
compares posterior, covariance, and bootstrap uncertainty conditional on explicit
models. Its matched-model analysis uses 50 observations per condition; its
mismatch study uses 20 per condition and remains separate from A–F.

Under weak/moderate bi-exponential mismatch, all three nominal 90% interval types
include the primary lifetime in **0/20** cases per group. Inclusion of the corrected
model-conditional pseudo-true projection is **16/20** and **19/20**, respectively.
All intervals in those groups are valid. A pseudo-true lifetime is a deterministic,
prior-free likelihood projection, not a physical true lifetime or a posterior estimate.

With the same 20 EMG-generated observations, replacing the assumed matched EMG by
a matched-peak/FWHM Gaussian reduces generating-lifetime inclusion from **19/20**
to **1/20** for all three methods. Statistical precision can remain high around
the wrong-model projection. The Stage-5 classical numerical-audit qualification
and Stage-6.5 correction provenance remain attached to the detailed findings.

Sampler diagnostics assess posterior exploration, not physical correctness.
Posterior-predictive checks are descriptive model-conditional diagnostics, not
universally calibrated model-validity p-values. Weak mismatch can evade global
Poisson-deviance checks. Neither good scalar deviance nor narrow intervals
establish a correct decay or IRF model.

## Reproducing or extending a comparison

Read the [reproducibility guide](reproducibility.md) before launching studies.
Preserve the selected reference, validity policy, paired observations, and
training/calibration/external roles. Report attempted and contributing counts.
Wall-clock timings depend on hardware and environment and are not regression
goldens. The accepted evidence does not cover vendor formats, realistic detector
artifacts, broad decay-family selection, or validated synthetic-to-real transfer.
