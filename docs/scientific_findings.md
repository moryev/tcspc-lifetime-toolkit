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