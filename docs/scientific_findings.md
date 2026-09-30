# Scientific findings

This document records durable scientific conclusions established during
development of the TCSPC Lifetime Toolkit. Detailed numerical evidence,
plots, and exploratory analyses remain in the corresponding notebooks and
reproducible benchmark workflows.

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


## Day 63 — Week 9 synthesis

Notebook 14 consolidates the complete Week 9 uncertainty and
failure-awareness workflow using the committed library APIs developed during
Days 57–62.

The final workflow preserves three distinct data roles: uncertainty training,
held-out development calibration, and frozen external Tests A–F. The A–F
suite remains excluded from estimator fitting, uncertainty calibration, and
conformal calibration.

Across the investigated methods, uncertainty is strongly
estimator-specific. Direct quantile regression produces conservative but
broad prediction intervals. Random-Forest tree disagreement provides a
useful sample-level error-ranking signal, while training-data bootstrap
spread is more sensitive to some acquisition-distribution shifts than to
individual prediction error.

For the classical Poisson reconvolution estimator, both local
Fisher-information covariance and parametric Poisson bootstrap reproduce
sampling variability reasonably well when the mono-exponential forward model
is correctly specified. Their uncertainty increases appropriately when
photon information decreases or detector background increases.

The decisive limitation appears under controlled bi-exponential model
mismatch. Lifetime error increases substantially while the classical
covariance and bootstrap uncertainty scales increase only modestly, causing
nominal 90% coverage to collapse. This confirms that these methods quantify
statistical variability conditional on the assumed model rather than the
uncertainty associated with an incorrect physical model.

The ML uncertainty mechanisms exhibit related limitations. None of the
current predictive intervals, ensemble-disagreement scores, or
training-bootstrap scores constitutes a general detector of physical-model
validity.

Split conformalization provides no additional correction in the present
experiment because the original quantile intervals already contain every
held-out calibration target. Conformal calibration can improve coverage
relative to a specified calibration distribution, but it does not by itself
supply awareness of unseen physical model misspecification.

Finally, neither scalar Poisson deviance nor the current aggregate signed
residual profiles provide a strong independent warning of the controlled
Test-F mismatch.

The main Week 9 conclusion is therefore:

> **An estimator can correctly quantify statistical uncertainty while
> remaining confidently wrong because its physical model is incomplete.**

For robust TCSPC lifetime analysis, statistical uncertainty should therefore
be reported together with explicit distribution-shift, model-validity, and
failure diagnostics rather than interpreted as a universal measure of
trustworthiness.

Notebook 14 provides the reproducible synthesis of these results and closes
the Week 9 uncertainty and failure-awareness stage.

---

## Post-Week-9 — Issue #8: generalized IRFs and failure awareness

### Reproducible scope

[Notebook 16](../notebooks/16_generalized_irf_and_failure_awareness.ipynb)
reproduces the committed Stage-6 fixed-seed demonstrations using library
functions and [the Issue-#8 manifest](../configs/issue8_irf_workflow.json).
These experiments are independent of the frozen Week-8 A–F protocol, which
is unchanged. The results below describe small controlled fixtures, not a
general robustness or uncertainty-coverage study.

Notebook 16 was executed from a fresh Python 3.12 kernel with NumPy 2.5.1,
SciPy 1.18.0, pandas 3.0.3, and scikit-learn 1.9.0. Its outputs reproduce the
Stage-6 results below at the reported precision; versions are also displayed
in the notebook rather than assumed from an unrecorded environment.

The shape study uses [0, 12) ns at 0.02 ns spacing, lifetimes {1, 2, 3, 4} ns,
200,000 expected signal photons within that acquisition window, background
1 count/bin, and seed 8601. The EMG has Gaussian-component centre 1.5 ns,
Gaussian-component FWHM 0.35 ns, and exponential tail time 0.18 ns. There are
two external Poisson realizations per lifetime per population (eight each).
The ML development population has 30 Gaussian realizations per lifetime
(120 total); development and both external populations use independent RNG
streams. Fitted residual-shift bounds are ±0.2 ns throughout this comparison.

### Generalized IRF handling

IRF source/origin, sampled source values, preparation history, and the
forward-model kernel are distinct. `IRFProfile` preserves supplied source
values, including nonuniform grids; `PreparedIRF` contains a normalized,
uniform-target-grid kernel and preparation diagnostics. Gaussian, EMG,
imported `SampledIRF`, and explicit leading-edge proxies all reach the same
array-based convolution/reconvolution machinery. An imported sampled trace
is not necessarily an independently measured physical IRF.

Registration and linear resampling are explicit. Neither infers zero time;
known registration is distinct from residual shift fitted during
reconvolution. Original imported values and provenance remain intact.

Geometric support loss is separated from target-grid quadrature change.
For the notebook's piecewise-linear source with times [0, 0.3, 0.7, 1] ns and
values [0, 1, 0, 0], the source area is 0.35. A target grid [0, 0.5, 1] ns
retains all available support, yet has pre-normalization area 0.25. A clipped
target [0.25, 0.5, 0.75] ns retains source area 0.245833 (loss fraction
0.297619), while its sampled area is 0.229167. Boundary integration is exact
for the available piecewise-linear source, not for unknown physical tails.
Normalization does not remedy undersampling; acceptable support loss is
application dependent.

### IRF shape mismatch with matched peak and full FWHM

The comparison Gaussian is derived only from the noiseless generating EMG.
Both have sampled peak 1.62 ns and full sampled FWHM approximately
0.45145465565 ns (agreement better than 1e-9 ns). The EMG's 0.35 ns
Gaussian-component FWHM is not its full width. Matching peak and FWHM does
not make Gaussian and EMG shapes equivalent.

Each EMG histogram is fit twice using exactly the same observed counts,
Poisson objective, background treatment, shift bounds, and initialization
policy: once with its matched EMG and once with the comparison Gaussian.
The Gaussian-generated control has matched nuisance conditions and new
Poisson counts. Results for eight histograms per row are:

| Generated data / assumed IRF | MAE (ns) | Bias (ns) | Mean Poisson deviance |
|---|---:|---:|---:|
| EMG / matched EMG | 0.005909 | +0.000906 | 591.609359 |
| Same EMG counts / Gaussian | 0.032795 | +0.032795 | 709.710133 |
| Gaussian control / Gaussian | 0.003490 | +0.002798 | 615.431037 |

All 24 fits are valid with no reported boundary hits. Thus higher-order IRF
shape differences produce a material positive lifetime bias here despite
matched timing and full width. The notebook retains per-realization signed
deviance residuals and illustrates additional early-time residual structure
under the Gaussian assumption. The mean deviance rises in this fixture;
this does not require matched deviance to win for every noisy realization
or establish deviance/residuals as universal physical-validity tests.

### Conditional statistical uncertainty under a wrong IRF

For external pair 0, with true lifetime 1 ns:

| Assumed fixed IRF | Lifetime estimate (ns) | Local Poisson/Fisher std (ns) | Three-resample bootstrap endpoints (ns) |
|---|---:|---:|---:|
| Matched EMG | 1.004166 | 0.002646 | 0.999421–1.006630 |
| Gaussian | 1.028703 | 0.002688 | 1.025463–1.030904 |

The bootstrap uses seed 1119 and requested nominal level 0.9. **Three
resamples demonstrate the interface only; these endpoints are not a
calibrated interval study or a coverage estimate.** Both covariance and
bootstrap hold the assumed IRF fixed, propagating no IRF-shape uncertainty.
Conditional statistical uncertainty can be small even when IRF shape is
misspecified: the Gaussian estimate is biased while its model-conditional
uncertainty is narrow.

### Gaussian-trained ML transfer: discrete-target fixture limitation

Ridge, Random Forest, and HistGradientBoosting are fit only on the 120
Gaussian development histograms. Feature settings are tail start 5 ns,
early stop 3 ns, late start 7 ns, and at least three tail points; estimator
random state is 42. Only the established histogram-derived features enter
models, not lifetime labels, source-kind labels, IRF parameters, or experiment
metadata. External data do not enter representation/model fitting,
hyperparameter selection, or calibration.

| Model | Gaussian-control MAE (ns) | EMG MAE (ns) | Gaussian-control bias (ns) | EMG bias (ns) |
|---|---:|---:|---:|---:|
| Ridge | 0.022274 | 0.204581 | +0.005659 | +0.204581 |
| Random Forest | 0.000000 | 0.013750 | 0.000000 | +0.013750 |
| HistGradientBoosting | 0.000027 | 0.000027 | approximately 0 | approximately 0 |

Both external populations have two new Poisson realizations at each of
{1, 2, 3, 4} ns; those discrete target values are identical to the development
support (30 realizations each). The near-perfect RF/HGB control performance
reflects this tiny high-count, controlled-nuisance, discrete-target fixture:
tree models can predict already-seen lifetime levels. This is not
unseen-lifetime generalization. HGB's absence of measurable EMG degradation
is a negative result for this fixture only, **not evidence of general
robustness to asymmetric IRFs**. The experiment was not changed to force a
degradation result.

Mean RF tree spread changes from 0 to 0.056929 ns. It is an ensemble
disagreement score, not a nominal prediction interval, conformal interval,
or classical model-conditional parameter standard deviation. No ML interval
calibration or coverage claim is made here.

### Leading-edge proxy: favorable and adverse regimes

For mono-exponential fluorescence, `S'(t) = A h(t) - S(t)/tau`; therefore
`h(t) != S'(t)` in general. The derivative proxy does not reconstruct the
missing `S/tau` term and is never labelled a measured or recovered true IRF.

The favorable condition uses lifetime 8 ns, Gaussian peak 2 ns/FWHM 0.4 ns,
2,000,000 finite-window signal photons, background 1 count/bin, and [0, 20)
ns at 0.02 ns spacing. Seed 8603 gives one independent Poisson realization
per regime. The explicit half-open background window is [0, 1) ns and
candidate rising-edge window [1.2, 4) ns. Savitzky–Golay settings are 17 bins,
polynomial order 3, first derivative in count/ns, and `mode="interp"`.

The favorable proxy has peak 1.96 ns (error −0.04 ns), full FWHM 0.383277 ns
(error −0.016723 ns), and normalized L1 shape error 0.091641. L1 denotes the
trapezoidal integral of the absolute difference of unit-area profiles, not
an exact-reconstruction criterion. Its fitted lifetime is 8.008096 ns,
versus 7.995458 ns using the true synthetic IRF.

| Regime / intended change | Proxy status | L1 shape error | Fit with true IRF (ns) | Fit with proxy (ns) | Additional observation |
|---|---|---:|---:|---:|---|
| Favorable baseline | Constructed | 0.091641 | 7.995458 | 8.008096 | Useful approximation |
| Low photons: 2,000 | Constructed | 1.009749 | 7.481860 | 7.157688 | Proxy FWHM unavailable |
| High background: 5,000/bin | Constructed | 0.120736 | 8.009286 | 8.033546 | Background ≥ half of observed rising-window counts flagged |
| Comparable lifetime: 0.4 ns | Constructed | 0.496261 | 0.400538 | 0.428982 | Valid proxy despite shape error |
| Very short lifetime: 0.08 ns | Constructed | 0.912417 | 0.079596 | 0.152618 | Valid proxy despite large error |
| Bi-exponential: 0.4/8 ns, short amplitude fraction 0.9 | Constructed | 0.449348 | 1.498194 | 1.532768 | No unique mono-exponential truth |
| +2.2 ns translation, correct window | Constructed | 0.103228 | 8.021581 | 8.027358 | Window shifted explicitly |
| +2.2 ns translation, wrong window | Failed | — | 7.998500 | — | Smoothed peak at analysis-window boundary |
| Strong EMG asymmetry: tail 0.4 ns | Constructed | 0.210944 | 7.988998 | 8.190455 | Proxy FWHM unavailable |
| Finite physical rise: rise 0.5 ns / decay 8 ns | Constructed | 1.031587 | 8.659080 | 8.246961 | No unique mono-exponential truth |

All constructed proxies report negative derivative clipping, including the
favorable one. This factual flag alone is not a severity classifier. The
low-photon and asymmetric cases additionally lack identifiable FWHM; high
background triggers a background-fraction flag but this realization remains
fairly accurate. Comparable/very-short lifetimes and finite rise can remain
numerically constructible and physically poor without a diagnostic that
identifies the underlying physical cause. Failure, warning, and actual
physical error must be reported separately.

The asymmetry control matches baseline sampled peak and full FWHM (within
one 0.02 ns bin and 1e-9 ns, respectively); it is not principally a
width/timing test. The finite-rise control uses
`d(u) ∝ exp(-u/tau_decay) - exp(-u/tau_rise)` for `u >= 0`, solely as a local
emission-dynamics confound. Bi-exponential and finite-rise scalar fits are
estimates under a misspecified mono-exponential decay model, **not errors
against a unique mono-exponential truth**; their truth-based lifetime-error
fields are absent/NaN.

The correctly translated timing control uses edge window [3.4, 6.2) ns;
the wrong-window case keeps [1.2, 4) ns. These controls use independent
Poisson realizations, **not an identical-count window ablation**. Translation
also changes the remaining finite acquisition window while preserving the
expected detected signal budget. No analysis window is silently realigned.

### Scientific interpretation

Matching peak and FWHM does not eliminate shape-family bias. Leading-edge
proxies can be useful in favorable regimes, but numerical constructibility,
warning flags, and physical accuracy are different quantities. Single
realizations do not estimate regime-wise failure probabilities, and a proxy
cannot generally identify lifetime/IRF scale separation, mixed decay,
physical rise, or unseen IRF tails from one histogram.

Issue #8 extends the Week-9 conclusion: **uncertainty within an assumed model
and evidence that the assumed model is physically correct are different
questions**. Report model assumptions and controlled failure evidence
alongside statistical uncertainty, rather than interpreting a narrow
interval or a well-formed proxy as physical validation.

## Post-Week-9 — Issue #4: Bayesian Poisson inference and model-conditional uncertainty

Week 9 established that statistical precision can coexist with physical
model error, and Issue #8 extended that distinction to the assumed IRF.
Issue #4 asks whether joint Bayesian inference changes this conclusion and
what posterior information adds to the existing classical uncertainty
analysis.

The Bayesian framework consumes canonical raw-count measurements and the
shared mono-exponential reconvolution model with a fixed prepared IRF.
Explicit priors on amplitude, lifetime, constant background, and residual
shift combine with the Poisson likelihood. `emcee` sampling and independent
ensemble diagnostics support posterior means/medians, credible intervals,
parameter correlations, and posterior-predictive checks. This preserves the
experimental-workflow distinction between a measurement, an assumed model,
and separately supplied benchmark truth.

### Matched-model calibration — Stage 5

The [Stage-5 manifest](../configs/issue4_bayesian_workflow.json) defines eight
principal mono-exponential conditions with 50 repeated Poisson observations
each. At a lifetime of 2 ns, signal budgets of 1,000, 10,000, and 100,000
photons are crossed with backgrounds of 0.5 and 5 counts/bin; 1 ns and 4 ns
lifetimes are also evaluated at 10,000 photons and 0.5 counts/bin. The same
raw histogram, time grid, fixed Gaussian IRF, background convention, and
residual-shift bounds feed classical Poisson fitting, local Poisson/Fisher
covariance, a 100-refit parametric Poisson bootstrap, and Bayesian inference.
Four predeclared prior-sensitivity jobs add 20 analyses each, paired to the
corresponding baseline subsets, for 480 inference records in total.

Under this correctly specified model, all three uncertainty approaches
broadly track empirical repeated-Poisson estimator variability. Covariance
and bootstrap standard deviations are compared with the spread of classical
fits; posterior standard deviations are compared with the spread of Bayesian
posterior medians. These are different estimator-specific denominators.
All baseline 90% interval-coverage estimates are compatible with nominal
coverage within finite-sample uncertainty: their 95% Wilson intervals
contain 0.90. Coverage is conditional on valid intervals or accepted Bayesian
runs, with failures counted separately. Fifty repetitions per condition do
not establish exact nominal calibration or justify ranking methods from
small coverage differences.

Bayesian posterior medians do not materially improve lifetime MAE over
classical Poisson reconvolution in these matched regimes. More photons
reduce lifetime error, empirical spread, and reported uncertainty. Higher
background increases uncertainty and strengthens parameter coupling,
particularly when photon information is limited; individual finite-sample
metrics need not change monotonically. The joint posterior adds direct
information about lifetime–background and lifetime–residual-shift
correlations, exposing identifiability effects that a marginal lifetime
interval alone cannot show.

That information has a substantial computational cost. Representative median
times from the scientific run are approximately 0.03 s for a classical
Poisson fit, 3.1 s for 100 bootstrap refits, and 60 s for Bayesian inference.
These are measurements on this hardware and configuration, not universal
algorithmic ratios. The full Stage-5 run took 31,573.440 s (about 8 h 46 min).

### Classical numerical audit — Stage 5.5

A post-run audit of all 480 saved Stage-5 records found one prematurely
terminated L-BFGS-B fit: `tau4_n10000_b0p5`, baseline realization 22.
Optimizer success had been accepted despite a large gradient and a feasible
descent direction. The toolkit subsequently added post-fit local objective
validation, one deterministic bounded continuation for suspicious solutions,
separate optimizer/recovery diagnostics, and robust propagation of classical
failures to unavailable covariance/bootstrap results. All 480 audited refits
passed; only that one required recovery. Its lifetime changed from 4.139127
to 3.921927 ns.

The broad Stage-5 conclusions are unaffected, but comparisons involving that
historical classical estimate and its conditional covariance/bootstrap
intervals retain this qualification. The frozen Stage-5 outputs were not
rewritten, and Bayesian inference was not repeated. Stage 6 uses the hardened
classical path. Its practical local check guards against gross numerical
nonstationarity; it does not establish global optimality or model correctness.

### From statistical precision to model mismatch — Stage 6

The next scientific question is whether a narrow uncertainty interval implies
that the assumed physical model is correct. **Covariance, parametric
bootstrap, and Bayesian posterior uncertainty are all conditional on the
assumed model.** They do not automatically include uncertainty from a wrong
decay family, a wrong IRF shape, or other model-form errors. This is a
cross-method limitation.

The independent [Stage-6 manifest](../configs/issue4_bayesian_mismatch_workflow.json)
specifies a matched mono-exponential control, weak and moderate bi-exponential
decays fitted as mono-exponential, and an EMG-IRF condition analysed under two
IRF assumptions. Each generating condition has 20 observations: 80 distinct
histograms and 100 inference evaluations. All use 100,000 expected detected
signal photons, background 0.5 counts/bin, and the [0, 12) ns window at
0.05 ns spacing. Local covariance intervals, 100-refit bootstrap percentile
intervals, and Bayesian credible intervals all use a nominal 90% level and
shared inference assumptions. These observations are independent of Stage 5
and frozen Week-8/9 Tests A–F.

### The pseudo-true mono-exponential reference

A bi-exponential decay has no unique physical mono-exponential lifetime.
The generating component lifetimes and detected-photon fractions therefore
remain separate from a deterministic projection onto the assumed model:

$$
\theta^\star
= \arg\min_{\theta=(A,\tau,B,\Delta t)}
\sum_i \left[
\mu_i(\theta)
- \lambda_i^{\mathrm{true}}\log\mu_i(\theta)
\right].
$$

Here $\lambda_i^{\mathrm{true}}$ is the noise-free generating histogram, and
$\mu_i(\theta)$ uses the same assumed IRF, measurement window, background
model, and shift bounds as inference. The lifetime coordinate $\tau^\star$
is a prior-free, deterministic, model-conditional likelihood projection. It
is not a physical "true lifetime" or a Bayesian posterior estimate.

The decomposition

$$
\hat{\tau}-\tau_{\mathrm{primary}}
= (\hat{\tau}-\tau^\star)
+ (\tau^\star-\tau_{\mathrm{primary}})
$$

separates estimation error inside the assumed model from the physical
discrepancy introduced by restricting the model family. An estimator can be
precise near $\tau^\star$ while that projection is displaced from the
generating primary-component lifetime. The matched Gaussian and matched EMG
controls both recover $\tau^\star=2.000000000$ ns.

### Weak and moderate decay-model mismatch

Both mixtures have primary lifetime 2 ns and secondary lifetime 4 ns. The
secondary fractions, 5% and 15%, refer to **detected signal photons in the
finite measurement window**, following the Week-9/Test-F convention; they
are not exponential amplitude fractions. Each convolved component is
normalized by its own finite-window signal sum before the two components
are mixed. The Stage-6.5-corrected references are 2.073921931 ns for weak
mismatch and 2.227022556 ns for moderate mismatch.

| Mixture | Point estimator | Bias vs primary (ns) | Bias vs corrected τ* (ns) | MAE vs corrected τ* (ns) |
|---|---|---:|---:|---:|
| Weak, 5% | Classical Poisson | +0.072230 | −0.001692 | 0.005807 |
| Weak, 5% | Bayesian posterior median | +0.071817 | −0.002105 | 0.005921 |
| Moderate, 15% | Classical Poisson | +0.224940 | −0.002083 | 0.006157 |
| Moderate, 15% | Bayesian posterior median | +0.224424 | −0.002599 | 0.006375 |

For weak mismatch, reported lifetime standard deviations remain approximately
0.0076–0.0077 ns. The physical displacement is about 9.3 times the reported
statistical uncertainty, while both point estimators remain comparatively
close to the best mono-exponential approximation. For example, the classical
mean deviation decomposes as approximately
$0.072230 = -0.001692 + 0.073922$ ns.

For moderate mismatch, reported standard deviations are still only
0.0082–0.0085 ns, despite a physical displacement of about 26–27 reported
standard deviations. The corresponding classical decomposition is
$0.224940 = -0.002083 + 0.227023$ ns. These ratios describe group mean
deviation divided by mean reported standard deviation; they are not a
universal calibration score.

| Mixture / reference | Covariance interval inclusion | Bootstrap percentile interval inclusion | Bayesian credible-interval inclusion |
|---|---:|---:|---:|
| Weak / primary 2 ns | 0/20 | 0/20 | 0/20 |
| Weak / corrected τ* | 16/20 | 16/20 | 16/20 |
| Moderate / primary 2 ns | 0/20 | 0/20 | 0/20 |
| Moderate / corrected τ* | 19/20 | 19/20 | 19/20 |

All intervals in these two groups are valid, so the inclusion fractions
also equal the successful-and-including fractions. With only 20 repetitions,
pseudo-true inclusion is descriptive evidence about targeting the projection,
not a claim of nominal calibration. All three uncertainty approaches can
remain narrow and internally consistent around the wrong-model projection
while excluding the physically meaningful primary-component lifetime.

### Paired IRF-shape mismatch

The IRF experiment holds the mono-exponential generating lifetime at 2 ns
and uses an asymmetric EMG (Gaussian-component centre 1.5 ns, component FWHM
0.35 ns, tail time 0.18 ns). Each of the same 20 observed histograms is
analysed with the matched EMG and a peak/full-FWHM-matched Gaussian constructed
using Issue #8's matching procedure. Raw counts, priors, sampler settings,
bootstrap budget, background convention, residual-shift treatment, and
corresponding random streams are shared within each pair. Only the assumed
IRF and its provenance change.

The corrected wrong-Gaussian projection is $\tau^\star=2.024957161$ ns.
Replacing the assumed EMG by the Gaussian shifts the classical fitted
lifetime by +0.024976 ns on average and the Bayesian posterior median by
+0.024894 ns. Both closely match the deterministic projection shift of
+0.024957 ns. Bayesian posterior standard deviation barely changes, from
about 0.00745 to 0.00753 ns.

For all three uncertainty approaches, generating-lifetime inclusion falls
from 19/20 with the matched EMG to 1/20 with the wrong Gaussian. Inclusion of
the wrong-model pseudo-true lifetime remains 19/20. The interval remains
conditional on the fixed assumed IRF; IRF-shape uncertainty has not been
propagated. Agreement with the wrong-model projection must not be described
as physical-lifetime calibration.

### Sampling diagnostics and posterior-predictive checks

Stage 6 records 99 successful Bayesian evaluations and one
`insufficient_sampling` rejection in the matched mono control. All
deliberately misspecified evaluations pass the configured sampler checks:
20/20 weak mixtures, 20/20 moderate mixtures, and 20/20 wrong-Gaussian IRFs.
The matched EMG group also passes 20/20. Successful MCMC diagnostics support
adequate sampling of the assumed posterior under the operational policy;
they do not establish that the assumed physical model is correct.

Posterior-predictive checks (PPC) are available for the 99 accepted runs,
using 200 replicated draws per run and explicit early [1, 3) ns and tail
[7, 12) ns windows. The saved discrepancies include Poisson deviance,
signed-deviance residual RMS and maximum magnitude, total counts, peak count
and time, and early/tail counts. Their tail probabilities are descriptive
model-conditional diagnostics, not universally calibrated frequentist
p-values. Small deviance tail probabilities mean replicated data rarely
have as much deviance as the observed histogram.

The matched mono and matched EMG groups provide their respective PPC
reference distributions. Weak 5% mismatch is not clearly separated from
the mono control by global deviance: its median tail probability is about
0.385, and only 1/20 probabilities are at most 0.05. Tail-window counts are
somewhat more sensitive, with 6/20 at most 0.05. Thus weak mismatch can evade
global PPC diagnostics even when the physical lifetime displacement is about
nine reported standard deviations.

Moderate 15% mismatch produces strong predictive tension: deviance tail
probabilities are at most 0.05 in 20/20 runs, early-window probabilities in
18/20, and tail-window probabilities in 19/20. The wrong Gaussian IRF also
has deviance probabilities at most 0.05 in 20/20 runs. Its peak-count
probability is at least 0.95 in 19/20, indicating that replicated peaks are
systematically too high relative to the EMG-generated observations. These
are useful warnings for these controlled mismatches, without defining a
general model-validity classifier.

The RMS of the across-realization mean signed residual profile gives a
complementary description of systematic temporal structure:

| Condition / assumed IRF | RMS of mean signed residual profile |
|---|---:|
| Mono control / Gaussian | 0.246 |
| Weak bi-exponential / Gaussian | 0.321 |
| Moderate bi-exponential / Gaussian | 0.619 |
| EMG / matched EMG | 0.252 |
| EMG / assumed Gaussian | 0.572 |

Profiles are averaged over accepted runs (19 mono controls and 20 in each
other group). The systematic residual structure strengthens substantially
for moderate decay mismatch and wrong IRF shape, but only modestly for weak
mismatch. This is consistent with Week 9's warning that scalar goodness of
fit and aggregate residuals can provide weak discrimination under modest
mismatch; the high-count Stage-6 study does not change the frozen Test-F
findings. Per-draw Poisson deviance and scalar residual RMS satisfy
$\mathrm{RMS}=\sqrt{D/n_{\mathrm{bins}}}$ here, so they are monotonic
transformations with identical tail-probability ordering, not independent
evidence. The RMS of the mean signed profile above is a distinct aggregation,
also not a universal mismatch detector.

### Deterministic reference audit — Stage 6.5

The read-only Stage-6 audit detected insufficient numerical accuracy in the
original deterministic bi-exponential projections. Stage 6.5 hardened that
offline reference calculation with a numerically centered Poisson objective,
diverse deterministic starts including background, strict local descent and
gradient checks, start-agreement checks, and an independent Poisson-deviance
least-squares cross-check. This reference calculation has a deliberately
tighter accuracy contract than routine noisy-data fitting.

The corrected references and inclusion counts above are authoritative for
this scientific interpretation. The scientific inference records were not
rerun: only reference-dependent deviations and inclusion summaries were
recalculated from saved estimates and intervals. Physical-reference results,
paired IRF displacements, sampler diagnostics, and PPC findings are unchanged.
The original Stage-5 and Stage-6 scientific CSV/JSON files remain frozen.
The separate ignored artifact
`data/generated/issue4/mismatch/scientific_reference_correction_v1.json`
records original hashes, corrected references, numerical validation, and
reanalysis provenance; the corresponding
[correction script](../scripts/correct_issue4_mismatch_references.py) uses
the persisted inference records and noise-free curves.

### Scientific interpretation and limits

The Bayesian study extends the Week-9 result across all three inference
approaches. With the correct model, statistical uncertainty broadly follows
repeated-Poisson variability. With an incorrect model, equally narrow
uncertainty can surround a physically displaced effective parameter because
inference is accurately approximating the best description available within
the wrong model family. Joint posterior correlations describe parameter
coupling, sampler diagnostics guard against sampling failure, and PPC adds
evidence about model adequacy; these answer different questions.

The results are limited to 50 baseline repetitions per Stage-5 condition,
20-realization predeclared prior-sensitivity subsets, and 20 observations per
Stage-6 generating condition. Coverage and inclusion therefore have finite
Monte Carlo uncertainty. The tested priors, fixed assumed IRFs, finite MCMC
budget, and operational convergence diagnostics do not establish general
prior robustness, formal convergence, or physical validity. The controlled
2/4 ns mixtures and specified EMG/Gaussian mismatch do not represent every
experimental model error, and measured runtimes depend on hardware and
configuration. PPC sensitivity depends on mismatch magnitude and the chosen
discrepancy; weak mismatch remains difficult to identify. Issue #4 remains
open pending the final Bayesian notebook and final documentation/API and
regression review.
