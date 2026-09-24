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


## Day 61 — Classical bootstrap versus empirical repeated-Poisson error

### Scientific question

Day 61 asked whether uncertainty reported by the classical Poisson reconvolution estimator reflects the variability that 
would actually be observed if the same physical TCSPC measurement were repeated many times.

Three quantities were compared:

* local covariance uncertainty obtained from the expected Poisson Fisher information;
* parametric Poisson-bootstrap uncertainty obtained by resampling from the fitted expected-count curve and refitting;
* empirical estimator variability obtained from statistically independent Poisson realizations generated under the same known physical condition.

The repeated-Poisson distribution serves as the empirical reference. Unlike covariance and bootstrap uncertainty, 
it requires knowledge of the true simulation condition and is therefore a benchmark rather than a deployable per-curve uncertainty estimate.

### Parametric bootstrap formulation

For one fitted TCSPC histogram with fitted expected counts $\hat{\mu}_i$, bootstrap measurements were generated as

$$
k_i^\ast \sim \operatorname{Poisson}(\hat{\mu}_i).
$$

Each bootstrap realization was then passed through the same histogram-derived initialization and Poisson reconvolution 
fitting pipeline as the original measurement.

Time bins were not resampled. This preserves the ordered physical structure of the TCSPC histogram.

The resulting bootstrap lifetime distribution provides a bootstrap standard deviation, median, percentile interval, 
and refit-failure diagnostics.

### Empirical repeated-Poisson reference

For a fixed known physical condition,

$$
(\tau, N_{\mathrm{photons}}, B, \mathrm{IRF}, \Delta t),
$$

independent Poisson measurements were generated and fitted repeatedly.

From the resulting lifetime estimates,

$$
\hat{\tau}^{(1)},\ldots,\hat{\tau}^{(R)},
$$

the empirical bias, standard deviation, and RMSE were calculated.

The principal uncertainty-calibration diagnostic was

$$
R_{\sigma}
=
\frac{
\text{mean estimated uncertainty}
}{
\text{empirical repeated-Poisson standard deviation}
}.
$$

A value near one indicates agreement between the reported uncertainty and the observed sampling variability. 
Values below one indicate optimistic uncertainty, while values above one indicate conservative uncertainty.

### Important numerical finding: raw Poisson optimization was poorly scaled

The first Day 61 calibration run revealed an unexpected high-photon failure.

For a representative condition with

$$
\tau = 2.0\ \mathrm{ns},\qquad
N_{\mathrm{photons}}=100\,000,\qquad
B=0.5,
$$

the histogram-derived initialization produced

$$
\tau_0 = 1.8176\ \mathrm{ns}.
$$

The original raw-parameter L-BFGS-B Poisson fit reported successful convergence but returned

$$
\hat{\tau}=1.8986\ \mathrm{ns},
$$

with reduced Poisson negative log-likelihood

$$
\mathrm{NLL}=-580595.56.
$$

Starting the same optimizer near the known physical truth produced

$$
\hat{\tau}=2.0036\ \mathrm{ns},
$$

with the substantially better objective value

$$
\mathrm{NLL}=-580800.51.
$$

The difference of approximately

$$
\Delta\mathrm{NLL}\approx205
$$

showed that optimizer success did not imply convergence to the best relevant solution.

The underlying problem was numerical parameter scaling. The reconvolution parameters can differ by many orders of magnitude: 
amplitude may be of order $10^3-10^6$, while lifetime, background, and temporal shift are typically of order $10^{-2}-10^1$.

The Poisson optimizer was therefore changed to operate internally on dimensionless scaled parameters while retaining the 
same physical model, likelihood, parameter bounds, and public fitting API.

With scaled optimization and the same imperfect initial guess, the fitted lifetime became

$$
\hat{\tau}=2.00016\ \mathrm{ns},
$$

with

$$
\mathrm{NLL}=-580800.80.
$$

Thus the apparent high-photon uncertainty failure was primarily a numerical optimization failure rather than a failure of 
the Poisson statistical model.

A dedicated regression test now verifies that the full histogram-derived initialization and Poisson reconvolution pipeline 
recovers the high-count lifetime even when the initial lifetime estimate is substantially biased.

### Final calibration experiment

After correcting Poisson parameter scaling, the uncertainty experiment was repeated using

$$
R=50
$$

independent Poisson measurements per condition and

$$
B_{\mathrm{bootstrap}}=100
$$

bootstrap refits per measurement.

The nominal interval coverage was 90%.

Eight physical conditions were evaluated across photon count, background level, and lifetime while keeping the mono-exponential 
reconvolution model correctly specified.

| Condition                         | Empirical std (ns) | Covariance std (ns) | Covariance ratio | Bootstrap std (ns) | Bootstrap ratio | Covariance coverage | Bootstrap coverage |
| --------------------------------- | -----------------: | ------------------: | ---------------: | -----------------: | --------------: | ------------------: | -----------------: |
| 1k photons, low background        |             0.1082 |              0.0923 |            0.853 |             0.0930 |           0.860 |                0.84 |               0.80 |
| 10k photons, low background       |             0.0245 |              0.0239 |            0.977 |             0.0238 |           0.974 |                0.94 |               0.84 |
| 100k photons, low background      |             0.0075 |              0.0070 |            0.927 |             0.0071 |           0.942 |                0.88 |               0.90 |
| 1k photons, elevated background   |             0.1569 |              0.1474 |            0.940 |             0.1473 |           0.939 |                0.86 |               0.84 |
| 10k photons, elevated background  |             0.0326 |              0.0294 |            0.899 |             0.0297 |           0.910 |                0.90 |               0.88 |
| 100k photons, elevated background |             0.0077 |              0.0076 |            0.978 |             0.0076 |           0.980 |                0.92 |               0.86 |
| 1 ns lifetime                     |             0.0132 |              0.0113 |            0.858 |             0.0116 |           0.879 |                0.84 |               0.84 |
| 4 ns lifetime                     |             0.0641 |              0.0604 |            0.942 |             0.0608 |           0.949 |                0.86 |               0.80 |

Across these conditions, the mean estimated-to-empirical standard-deviation ratios were

$$
\overline{R}_{\mathrm{cov}}
\approx0.922,
$$

and

$$
\overline{R}_{\mathrm{bootstrap}}
\approx0.929.
$$

Both methods therefore tracked empirical repeated-measurement variability reasonably well but were mildly optimistic on average.

The mean empirical interval coverage was

$$
C_{\mathrm{cov}}
\approx0.880
$$

for local covariance intervals and

$$
C_{\mathrm{bootstrap}}
\approx0.845
$$

for bootstrap percentile intervals, compared with the nominal value of 0.90.

### Photon-count dependence

The uncertainty estimates reproduced the physically expected improvement with increasing photon statistics.

For $\tau=2$ ns and low background, the empirical lifetime standard deviation decreased from

$$
0.1082\ \mathrm{ns}
$$

at \(10^3\) signal photons to

$$
0.0245\ \mathrm{ns}
$$

at \(10^4\) photons and

$$
0.0075\ \mathrm{ns}
$$

at \(10^5\) photons.

Both covariance and bootstrap uncertainty followed the same trend closely.

This confirms that the uncertainty machinery responds to changing photon information rather than merely reporting an 
approximately constant model-dependent scale.

### Background dependence

Elevated background increased lifetime uncertainty most strongly in the low-photon regime.

At $10^3$ photons, increasing the background from 0.5 to 5 counts per bin increased the empirical lifetime standard deviation from approximately

$$
0.108\ \mathrm{ns}
$$

to

$$
0.157\ \mathrm{ns}.
$$

Both covariance and bootstrap methods reproduced this increase.

At high photon count, the effect of the same background increase was much smaller because signal information remained dominant.

### Covariance versus bootstrap

Under the correctly specified mono-exponential model, the local Fisher-information covariance estimate performed surprisingly well.

Its mean uncertainty ratio was comparable to that of the parametric bootstrap, while its average interval coverage was closer to the nominal 90% value.

This does not establish covariance as universally superior. The covariance approximation is local and depends on a regular, 
well-conditioned likelihood surface, an adequate physical model, and solutions away from parameter bounds.

The result instead shows that under these regular conditions, the much cheaper local covariance calculation can provide 
a useful approximation to repeated-measurement lifetime uncertainty.

The parametric bootstrap also reproduced empirical standard deviations well. Its simple percentile intervals showed modest 
undercoverage in the present experiment, suggesting that variance estimation and interval calibration should be treated as separate questions.

### Failure diagnostics

No primary reconvolution-fit failures or parameter-boundary hits occurred in the final repeated-Poisson calibration experiment.

Bootstrap refit failures were essentially absent, with only negligible failure rates in the short- and long-lifetime conditions.

The observed differences between estimated and empirical uncertainty therefore cannot be explained by selective removal of failed fits.

### Main conclusion

For a correctly specified mono-exponential TCSPC reconvolution model, both local Poisson/Fisher covariance and parametric 
Poisson bootstrap provide useful estimates of lifetime sampling uncertainty.

Across the tested photon-count, background, and lifetime conditions, both uncertainty scales were generally close to the 
empirical variability observed over repeated Poisson measurements, although both were mildly optimistic on average.

The most important Day 61 finding was methodological: uncertainty calibration exposed a hidden numerical-conditioning problem 
in the principal Poisson reconvolution estimator. Once the optimizer was reformulated in scaled dimensionless coordinates, 
the high-photon systematic bias disappeared and uncertainty calibration became physically consistent.

This demonstrates why uncertainty analysis is valuable not only for reporting confidence in final predictions, but also 
as a diagnostic tool for identifying failures in the underlying estimator itself.


## Day 62 — Uncertainty calibration under A-F and model-mismatch failure-awareness

### Scientific question

Day 62 froze the Week 9 uncertainty methods and evaluated them across the
complete Week 8 A-F robustness suite.

The central question was no longer only whether lifetime predictions become
less accurate under distribution shift, but whether the corresponding
uncertainty estimates recognize that loss of reliability.

The final A-F tests remained external evaluation data only. They were not
used for model fitting, uncertainty calibration, threshold selection, or
conformal calibration.

For Test F, uncertainty was scored against the dominant-component lifetime

$$
\tau_1,
$$

which remains the primary frozen Week 8 reference. The
signal-photon-weighted component lifetime is descriptive only: a
bi-exponential decay does not possess a unique mono-exponential true
lifetime.


### Quantile gradient boosting across A-F

The direct 0.05/0.50/0.95 gradient-boosting quantile estimator retained
100% empirical interval coverage across all six A-F tests.

However, the nominal 90% intervals were already highly conservative on the
held-out development calibration subset, where coverage was also 100%.
Across A-F, mean interval widths remained approximately 2.46-2.67 ns.

The resulting high coverage therefore does not imply strong
failure-awareness. The intervals are broad relative to the lifetime errors
and provide limited discrimination between familiar conditions and shifted
or misspecified conditions.

The conditional analysis confirmed the same pattern.

Under low-photon Test B, median-prediction MAE increased relative to the
development calibration reference while mean interval width increased only
modestly. High-photon Test B produced narrower intervals and somewhat
improved accuracy.

Elevated-background Test D increased prediction error while interval width
changed comparatively little.

Controlled bi-exponential Test F produced essentially no useful interval
response. Weak and moderate mismatch both retained 100% coverage with mean
interval widths close to those observed for familiar conditions.

Thus the direct quantile interval is conservative enough to contain the
selected lifetime target, but interval width is not a sensitive indicator
of physical model mismatch.


### Conformalized quantile regression

Split conformalization was applied to the frozen Day 58 quantile estimator
using only the 16-sample held-out uncertainty-calibration subset.

For every valid calibration sample, the original 90% quantile interval
already contained the true lifetime. With the non-negative conformal
nonconformity score

$$
s_i
=
\max
\left(
L_i-y_i,\;
y_i-U_i,\;
0
\right),
$$

the finite-sample conformal correction was therefore

$$
q_{\mathrm{conf}} = 0.
$$

Consequently, conformalized intervals were identical to the original
quantile intervals on both the calibration subset and all A-F robustness
tests.

This is not a conformal failure. It shows that the underlying quantile
intervals are already sufficiently conservative that this split-conformal
correction has nothing to enlarge.

In the present small-data setting, conformalization therefore adds no
additional practical failure-awareness.


### Random-Forest ensemble spread

Random-Forest tree-to-tree spread remained the strongest current
sample-level ML uncertainty score.

Across the complete A-F suite, the Spearman association between absolute
lifetime error and tree spread remained approximately 0.72-0.91.

The method responded particularly clearly to acquisition-statistics shifts.

For low-photon Test B, matched MAE increased by approximately

$$
3.32\times,
$$

while mean tree spread increased by approximately

$$
1.66\times.
$$

For elevated-background Test D, matched MAE increased by approximately

$$
2.76\times,
$$

while mean tree spread increased by approximately

$$
1.32\times.
$$

Within elevated-background Test D, the spread-error correlation remained
approximately 0.90.

The response to decay-model mismatch was substantially weaker. Under
moderate Test F, matched MAE increased by approximately 37%, while mean
tree spread increased by only approximately 9%.

Tree disagreement therefore remains useful for ranking prediction
difficulty, but its magnitude should not be interpreted as a calibrated
lifetime uncertainty or as a general test of physical model validity.


### Ridge training-bootstrap spread

Ridge training-bootstrap spread again behaved primarily as a detector of
instability in the learned mapping rather than as a calibrated
sample-specific uncertainty measure.

It responded strongly to photon-count and background distribution shifts.
However, its error-ranking performance was inconsistent across A-F, with
weak or negative correlations in some regimes.

Most importantly, moderate bi-exponential Test F increased lifetime error
while mean Ridge bootstrap spread remained essentially unchanged.

This confirms the Day 59 interpretation: resampling the
mono-exponential training data measures sensitivity to finite training-set
composition, but all bootstrap models remain confined to the same assumed
physical data distribution. Agreement between them does not establish that
the underlying decay model is valid.


### Classical covariance and parametric-bootstrap calibration across A-F

The strongest Day 62 result came from applying the Day 60-61 classical
uncertainty methods to every curve in the frozen A-F suite.

Both methods wrapped the same Poisson mono-exponential reconvolution
estimator:

* local Poisson/Fisher covariance;
* parametric Poisson bootstrap with 200 refits per measured curve.

Every test was fitted using the correct per-curve IRF width. Test F
therefore isolates decay-model misspecification rather than IRF
misspecification.

The nominal interval coverage was 90%.

| Test | Covariance coverage | Bootstrap coverage | Covariance mean width (ns) | Bootstrap mean width (ns) |
| --- | ---: | ---: | ---: | ---: |
| A | 0.891 | 0.885 | 0.333 | 0.327 |
| B | 0.864 | 0.822 | 0.875 | 0.938 |
| C | 0.872 | 0.854 | 0.348 | 0.347 |
| D | 0.912 | 0.917 | 0.473 | 0.475 |
| E | 0.896 | 0.901 | 0.337 | 0.337 |
| F | 0.518 | 0.490 | 0.366 | 0.361 |

For familiar Test A, both uncertainty methods were close to nominal
calibration.

Tests D and E also remained approximately calibrated.

Test B showed moderate undercoverage overall, reflecting the difficulty of
the low-photon subset.

The decisive failure occurred under Test F. Lifetime MAE increased from

$$
0.0827\ \mathrm{ns}
$$

for Test A to

$$
0.1567\ \mathrm{ns}
$$

for Test F, an increase of approximately

$$
1.89\times.
$$

However, the reported uncertainty scale increased by only approximately

$$
1.10\times
$$

for covariance and

$$
1.11\times
$$

for the parametric bootstrap.

As a result, nominal 90% coverage collapsed to approximately 52% for local
covariance and 49% for the parametric bootstrap.

The mean normalized absolute error

$$
\frac{
|\hat{\tau}-\tau_{\mathrm{ref}}|
}{
\hat{\sigma}_{\tau}
}
$$

increased from approximately 0.84 under Test A to approximately 2.52-2.54
under Test F.

This is a direct example of confident physical misspecification: the
estimator becomes systematically less accurate without reporting a
commensurate increase in statistical uncertainty.


### Photon-statistics failure-awareness

Classical uncertainty responded strongly and physically to photon
statistics.

For low-photon Test B, classical MAE was approximately

$$
0.367\ \mathrm{ns},
$$

and covariance lifetime uncertainty was approximately

$$
0.393\ \mathrm{ns}.
$$

For high-photon Test B, MAE fell to approximately

$$
0.014\ \mathrm{ns},
$$

with covariance uncertainty of approximately

$$
0.0145\ \mathrm{ns}.
$$

Despite this large change in precision, covariance coverage remained
similar between the low- and high-photon subsets.

The parametric bootstrap showed the same strong photon-count dependence.

This demonstrates that both classical uncertainty methods successfully
recognize statistical information loss caused by photon starvation.


### Elevated-background failure-awareness

Elevated-background Test D increased the classical uncertainty scale while
maintaining approximately nominal coverage.

Covariance mean interval width increased from approximately

$$
0.333\ \mathrm{ns}
$$

under Test A to

$$
0.473\ \mathrm{ns}
$$

under Test D.

Bootstrap width behaved almost identically.

Thus the classical statistical uncertainty estimates respond appropriately
when the measurement becomes less informative because of background
contamination.


### Severity dependence under bi-exponential mismatch

The Test-F severity analysis made the model-mismatch failure especially
clear.

For weak mismatch, classical MAE was approximately

$$
0.107\ \mathrm{ns}.
$$

Covariance and bootstrap coverage fell to approximately 0.695 and 0.646,
respectively.

For moderate mismatch, MAE increased to approximately

$$
0.207\ \mathrm{ns},
$$

while covariance and bootstrap coverage collapsed further to approximately
0.344 and 0.333.

At the same time, the reported lifetime standard deviation changed only
slightly between weak and moderate mismatch.

The normalized error increased from approximately 1.36-1.40 estimated
standard deviations under weak mismatch to approximately 3.67-3.68 under
moderate mismatch.

The uncertainty failure therefore becomes substantially more severe as the
unmodelled secondary component becomes stronger, even though the reported
statistical uncertainty itself changes very little.


### Covariance and bootstrap identify statistical uncertainty, not model uncertainty

The close agreement between local covariance and the 200-resample
parametric bootstrap is scientifically important.

The Test-F failure cannot be explained as a weakness unique to the local
Gaussian covariance approximation.

The parametric bootstrap also samples exclusively from the fitted
mono-exponential model:

$$
k_i^\ast
\sim
\operatorname{Poisson}
\left(
\hat{\mu}_i^{\mathrm{mono}}
\right).
$$

It therefore measures sampling variability conditional on the assumed
physical model.

When the real synthetic data are bi-exponential, bootstrap resampling from
the fitted mono-exponential model cannot represent the missing structural
uncertainty.

Day 62 therefore establishes a central distinction:

> Good calibration of statistical uncertainty under the assumed model does
> not imply robustness to physical model misspecification.


### Residual diagnostics under Test F

Day 62 also revisited whether signed Poisson deviance residual structure
could expose model mismatch that scalar goodness-of-fit metrics miss.

For every valid curve, the signed deviance residual

$$
r_i(t)
$$

was retained after mono-exponential Poisson reconvolution.

The mean signed residual profile was then calculated as

$$
\bar{r}(t)
=
\frac{1}{N}
\sum_{i=1}^{N}
r_i(t)
$$

for familiar Test A, weak Test F, and moderate Test F.

A paired diagnostic also compared matched Test-F and Test-A curves through

$$
\overline{
r_F(t)-r_A(t)
}.
$$


### Scalar Poisson deviance was almost insensitive to mismatch

Mean Poisson deviance per time bin was approximately

$$
1.0561
$$

for Test A,

$$
1.0602
$$

for weak Test F, and

$$
1.0627
$$

for moderate Test F.

Relative to Test A, the mean-deviance ratios were therefore only

$$
1.0039
$$

and

$$
1.0063.
$$

Thus scalar Poisson goodness-of-fit remained almost unchanged despite the
substantial uncertainty-calibration failure observed under Test F.


### Signed residual profiles provided only weak additional mismatch information

The RMS magnitude of the mean signed residual profile increased by
approximately 11% under both weak and moderate Test F.

The maximum absolute mean residual increased by approximately 26% for weak
mismatch and 29% for moderate mismatch.

However, the residual-profile RMS showed essentially no severity ordering:
weak and moderate Test F produced almost identical increases.

The time-domain profiles were noisy and strongly overlapping. The paired
Test-F-minus-Test-A profiles also fluctuated around zero without a clear
reproducible temporal signature that strengthened systematically from weak
to moderate mismatch.

Test A itself exhibited a non-zero average residual structure, particularly
a weak negative late-time baseline. Therefore absolute residual-profile
magnitude is not a clean standalone model-validity statistic in the current
benchmark.

The scientifically supported conclusion is consequently negative but
useful:

> Scalar Poisson goodness-of-fit largely misses the controlled
> bi-exponential mismatch, while aggregation of signed residual structure
> provides only modest additional discrimination and no clear
> severity-dependent temporal signature.

Residual diagnostics should therefore not be presented as a reliable
current detector of Test-F model misspecification.


### Day 62 main conclusion

Day 62 demonstrates that uncertainty quality is strongly
failure-mechanism-dependent.

Classical covariance and parametric Poisson bootstrap respond appropriately
to changes in photon statistics and background because those perturbations
alter statistical information within the assumed physical model.

The same methods can become severely overconfident when the physical decay
model itself is wrong. Under bi-exponential Test F, lifetime error increases
far more strongly than the reported uncertainty, causing nominal 90%
coverage to collapse.

ML uncertainty mechanisms show analogous limitations in different forms.
Random-Forest tree spread remains a useful error-ranking heuristic, while
Ridge training-bootstrap spread can detect some acquisition-distribution
instabilities. Neither constitutes a general physical-model-validity test.

Direct quantile intervals are highly conservative on the current small
development dataset, and split conformalization adds zero correction because
all calibration targets already lie inside those intervals.

Finally, neither scalar Poisson deviance nor the current aggregate signed
residual profiles provide a strong warning of the controlled model mismatch.

The principal Week 9 lesson is therefore:

> An estimator can know how uncertain it is about statistical noise while
> remaining unaware that its physical model is wrong.

Robust TCSPC lifetime inference should consequently report statistical
uncertainty together with explicit distribution-shift and model-validity
diagnostics rather than treating a single uncertainty estimate as a
universal measure of trust.