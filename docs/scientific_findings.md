# Scientific findings

This document records durable scientific conclusions established during
development of the TCSPC Lifetime Toolkit. Detailed numerical evidence,
plots, and exploratory analyses remain in the corresponding notebooks.

## Week 8 — Robust evaluation and OOD generalization

Detailed analysis: `notebooks/13_generalization_and_robustness.ipynb`

### Main findings

1. **Strong in-distribution cross-validation does not guarantee OOD robustness.**
   Engineered-feature Ridge gives the strongest repeated development-CV
   performance, while Random Forest is the strongest ML estimator across
   most final A-F robustness tests.

2. **Photon statistics and elevated background are the dominant current
   weaknesses of the learned estimators.**
   Photon-starved measurements drive much of the Test-B degradation, while
   background shift substantially degrades all current engineered-feature
   ML estimators.

3. **IRF broadening is comparatively benign when the IRF is known.**
   Explicit IRF misspecification is a separate and more consequential
   instrument-model mismatch.

4. **Explicit physical nuisance modelling provides strong robustness.**
   Classical reconvolution remains the lowest-MAE principal estimator across
   Tests A-F and is particularly robust to background and temporal shift when
   those quantities are represented in the forward model.

5. **Physics-based fitting can fail scientifically without failing
   numerically.**
   Under weak bi-exponential Test-F mismatch, classical reconvolution has
   zero fit failures but develops systematic positive lifetime bias.

6. **Smaller ML degradation under Test F does not imply correct
   bi-exponential modelling.**
   The ML models were trained only on mono-exponential data; partial
   insensitivity to the secondary component can appear robust when scoring
   against the dominant lifetime tau_1.

### Overall conclusion

The estimators have learned physically meaningful lifetime information,
but that information remains entangled with properties of the acquisition
distribution. Robustness is estimator-, representation-, and
mismatch-specific rather than universal.

Classical reconvolution benefits strongly from explicit nuisance modelling,
but becomes particularly sensitive when the physical decay model itself is
violated.

### Consequence for Week 9

Point-estimate accuracy and optimizer success are insufficient indicators
of reliability. Week 9 should therefore investigate uncertainty and
failure-awareness while retaining Tests A-F as untouched external
robustness tests.


## Week 9 — Uncertainty calibration and failure-awareness

### Day 58 — Direct quantile gradient-boosting baseline

Day 58 established the first predictive-uncertainty baseline using three 
independent `HistGradientBoostingRegressor` models for the 0.05, 0.50, and 0.95 conditional lifetime quantiles. 
The median model provides the central lifetime estimate and the outer quantiles define a nominal 90% prediction interval.
The uncertainty-development pool contains 64 familiar-domain samples and is split reproducibly into 48 uncertainty-training 
samples and 16 held-out calibration samples. Final robustness Tests A-F remain excluded from model fitting and calibration.
On the held-out development calibration subset, the nominal 90% interval achieved 100% empirical coverage with zero interval 
failures and zero quantile crossings. However, the mean interval width was 2.56 ns and the median width was 3.0 ns. 
The learned quantiles were strongly discretized: the lower quantile collapsed to 1 ns for every calibration sample, 
the median prediction took approximately 1, 2, or 3 ns, and the upper quantile took approximately 3 or 4 ns. 
The perfect empirical coverage therefore reflects conservative and poorly sharp intervals rather than finely resolved conditional uncertainty.
Frozen external evaluation preserved 100% coverage and zero quantile crossing across low-photon, high-photon, elevated-background, 
weak bi-exponential, and moderate bi-exponential conditions. Because the intervals are broad, coverage alone is not sufficiently informative for judging failure-awareness.
Paired Test-A comparisons provide the stronger diagnostic. Under low-photon shift, mean absolute error increased from 0.398 ns to 0.539 ns 
while mean interval width increased from 2.461 ns to 2.883 ns. This demonstrates a regime-level uncertainty response, 
but the paired Spearman association between interval-width change and absolute-error change was only 0.077, indicating weak sample-level failure ranking.
Under elevated background, mean absolute error increased from 0.411 ns to 0.578 ns while mean interval width increased only from 2.458 ns to 2.641 ns. 
The uncertainty response therefore substantially under-reacted in magnitude to the loss of accuracy. However, the paired width-error change correlation was approximately 0.48, 
showing that the estimator retained some useful relative failure-awareness within this regime.
High photon counts produced a small improvement in accuracy together with narrower intervals, which is physically consistent, 
although the paired width-error correlation remained close to zero.
Controlled bi-exponential mismatch produced essentially no meaningful uncertainty response. 
Mean interval width changed by only about 0.01 ns for weak mismatch and 0.04 ns for moderate mismatch relative to paired Test A, 
while the paired width-error correlations were approximately -0.26 and 0.01 respectively. 
The apparent reduction in error relative to the dominant-component lifetime must not be interpreted as improved modelling: 
the bi-exponential decay has no unique mono-exponential lifetime, and the highly discretized median estimator can move closer 
to the selected primary reference without recognizing the underlying physical model violation.

### Overall conclusion

Direct quantile gradient boosting provides valid, non-crossing but highly conservative prediction intervals on the current small development dataset. 
The interval width responds qualitatively to some acquisition changes, particularly photon statistics, and retains partial failure-awareness under elevated background. 
However, its uncertainty estimates are coarse, only weakly aligned with individual prediction failures, and do not meaningfully recognize controlled bi-exponential model mismatch.
Nominal predictive uncertainty learned on the mono-exponential development distribution is therefore not sufficient by itself 
for robust failure-awareness under physical distribution shift and model misspecification. This result establishes the direct quantile-regression method 
as a baseline against which subsequent Week 9 uncertainty methods can be evaluated.


### Day 59 — Ensemble disagreement and ML training-bootstrap spread

Day 59 evaluated two uncertainty-score approaches that make no nominal
prediction-interval claim: Random-Forest tree-to-tree disagreement and
non-parametric training-data bootstrap spread for Ridge regression.

Both methods used the same Week 9 development split: 48
uncertainty-training samples and 16 held-out calibration samples.
The final robustness Tests A, B, D, and F were used only after fitting,
with the same matched `pair_id` design used for the Day 58 analysis.

Random-Forest tree spread provided a strong error-ranking signal on the
held-out development calibration subset. The Spearman correlation between
tree spread and absolute lifetime error was approximately 0.94. The
lowest-spread 20% of calibration predictions had an MAE of approximately
0.01 ns, compared with approximately 0.50 ns for the highest-spread 20%.
Because the calibration subset contains only 16 samples, the exact
correlation magnitude should not be overinterpreted, but the observed
ranking separation is strong.

Under low-photon Test B, Random-Forest MAE increased by a factor of
approximately 3.32 relative to matched Test A, while mean tree spread
increased by approximately 1.66. Under elevated-background Test D, MAE
increased by approximately 2.76 while tree spread increased by only 1.32.
Despite this smaller regime-level increase in spread, tree disagreement
remained highly informative within Test D: the shifted-condition
spread-error Spearman correlation was approximately 0.90 and the paired
correlation between spread change and absolute-error change was
approximately 0.65.

Random-Forest tree spread did not exhibit the hypothesized complete
"confident but wrong" failure under moderate bi-exponential mismatch.
Relative to matched Test A, moderate Test F increased MAE by approximately
36%, while mean tree spread increased by only approximately 9%. However,
tree spread still ranked individual Test-F errors well, with a
shifted-condition spread-error correlation of approximately 0.87 and a
paired spread-change/error-change correlation of approximately 0.56.

Weak Test-F mismatch reduced both MAE relative to the selected dominant
lifetime tau_1 and mean tree spread. This must not be interpreted as
recognition of correct bi-exponential physics. Tree spread is an
error-ranking diagnostic relative to a chosen target, not a model-validity
test. A mono-exponential estimator can move closer to tau_1 while remaining
physically misspecified.

Training-data bootstrap spread for Ridge showed a different behaviour. On
the held-out familiar-domain calibration subset, bootstrap spread had only
a weak monotonic association with absolute error, with Spearman correlation
approximately 0.18.

Under acquisition distribution shift, however, bootstrap spread often
responded strongly at the regime level. Low-photon Test B increased Ridge
MAE by approximately 4.57 and mean bootstrap spread by approximately 1.97;
the paired spread-change/error-change correlation was approximately 0.78.
Elevated background increased MAE by approximately 4.18 and spread by
approximately 1.76, with paired correlation approximately 0.70.

High-photon Test B produced the clearest distinction between regime
detection and sample-level failure ranking. Ridge MAE increased by
approximately 5.65 and mean bootstrap spread by approximately 6.20, and
bootstrap spread increased for every matched pair. Nevertheless, the
within-condition spread-error correlation was only approximately 0.26 and
the paired spread-change/error-change correlation only approximately 0.08.
Bootstrap spread therefore strongly recognized that this regime was
unstable for Ridge without reliably ranking the individual prediction
errors inside that regime.

Controlled bi-exponential mismatch exposed the main limitation of
training-data bootstrap uncertainty. Under moderate Test F, Ridge MAE
increased by approximately 28%, while mean bootstrap spread remained
essentially unchanged, with a shifted/reference spread ratio of
approximately 1.00. The shifted spread-error correlation was approximately
0.06 and the paired spread-change/error-change correlation approximately
0.05.

This provides a direct example of uncertainty blindness under physical
model mismatch. Resampling the mono-exponential uncertainty-training data
measures sensitivity to finite training-sample composition, but every
bootstrap replica remains confined to the same assumed physical data
distribution. Agreement between bootstrap models therefore does not imply
that the underlying physical model is valid.

### Day 59 conclusion

Random-Forest tree disagreement and Ridge training-bootstrap spread capture
different aspects of estimator reliability.

Random-Forest tree spread is a strong sample-level error-ranking heuristic
on the present benchmark and remains informative under photon and
background shifts. Its magnitude is not calibrated to lifetime error and
should not be interpreted as a predictive standard deviation.

Ridge bootstrap spread is a weaker in-domain ranking signal but can act as
a strong detector of acquisition-distribution instability. In particular,
it reacts strongly to the photon-count and background regimes in which
Ridge becomes brittle.

Neither mechanism should be interpreted as a general detector of physical
model validity. The moderate bi-exponential experiment shows especially
clearly that training-data bootstrap replicas can remain mutually
consistent while prediction error increases under an unseen decay model.

Together with Day 58, these results demonstrate that predictive interval
width, ensemble disagreement, and training-data sensitivity quantify
different notions of uncertainty and can fail in different ways.