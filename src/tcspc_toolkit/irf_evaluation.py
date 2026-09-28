"""Controlled Issue-#8 IRF-shape and leading-edge proxy experiments.

These populations are independent of the frozen Week-8 A--F protocol. All
inference uses the existing sampled-kernel convolution and Poisson fit paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq

from tcspc_toolkit.classical_evaluation import ReconvolutionCurveResult
from tcspc_toolkit.classical_uncertainty import (
    ParametricPoissonBootstrapResult,
    PoissonLocalCovarianceResult,
    _reconstruct_reconvolution_fit_result,
    estimate_parametric_poisson_bootstrap,
    estimate_poisson_reconvolution_local_covariance,
)
from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.evaluation import (
    calculate_lifetime_errors,
    calculate_poisson_deviance_residuals,
)
from tcspc_toolkit.experimental import fit_experimental_reconvolution
from tcspc_toolkit.features import extract_feature_table
from tcspc_toolkit.irf import (
    IRFProfile,
    IRFSourceKind,
    generate_emg_irf_profile,
    generate_gaussian_irf_profile,
)
from tcspc_toolkit.irf_estimation import (
    LeadingEdgeIRFResult,
    estimate_irf_from_leading_edge,
)
from tcspc_toolkit.irf_preparation import PreparedIRF, _sampled_fwhm, prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.ml_evaluation import BenchmarkMeasurements, evaluate_regression
from tcspc_toolkit.ml_models import (
    make_hist_gradient_boosting_pipeline,
    make_ridge_pipeline,
)
from tcspc_toolkit.ml_uncertainty import RandomForestTreeSpreadEstimator
from tcspc_toolkit.models import biexponential_decay, monoexponential_decay
from tcspc_toolkit.preprocessing import validate_time_axis
from tcspc_toolkit.simulation import build_expected_counts_from_irf, sample_photon_counts


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class IRFShapeMismatchConfig:
    """Explicit design for paired EMG/Gaussian shape-family comparisons."""

    time_ns: FloatArray
    lifetimes_ns: tuple[float, ...]
    signal_photon_counts: tuple[int, ...]
    tail_times_ns: tuple[float, ...]
    gaussian_component_centre_ns: float
    gaussian_component_fwhm_ns: float
    background_per_bin: float
    development_repeats: int = 8
    external_repeats: int = 2
    random_seed: int = 8_601
    temporal_shift_bounds_ns: tuple[float, float] = (-0.2, 0.2)

    def __post_init__(self) -> None:
        time = validate_time_axis(self.time_ns)
        copied = np.array(time, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "time_ns", copied)
        for name in ("lifetimes_ns", "tail_times_ns"):
            values = tuple(getattr(self, name))
            if not values or any(isinstance(x, (bool, np.bool_))
                                 or not np.isfinite(x) or x <= 0.0 for x in values):
                raise ValueError(f"{name} must contain finite positive values")
            object.__setattr__(self, name, values)
        photon_counts = tuple(self.signal_photon_counts)
        object.__setattr__(self, "signal_photon_counts", photon_counts)
        if not photon_counts or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n <= 0
            for n in photon_counts
        ):
            raise ValueError("signal_photon_counts must contain positive integers")
        if not np.isfinite(self.gaussian_component_centre_ns):
            raise ValueError("gaussian_component_centre_ns must be finite")
        if not np.isfinite(self.gaussian_component_fwhm_ns) or self.gaussian_component_fwhm_ns <= 0:
            raise ValueError("gaussian_component_fwhm_ns must be finite and positive")
        if not np.isfinite(self.background_per_bin) or self.background_per_bin < 0:
            raise ValueError("background_per_bin must be finite and nonnegative")
        for name in ("development_repeats", "external_repeats", "random_seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "random_seed" else 1):
                raise ValueError(f"{name} is out of range")
        lower, upper = self.temporal_shift_bounds_ns
        if not (np.isfinite(lower) and np.isfinite(upper) and lower < 0 < upper):
            raise ValueError("temporal_shift_bounds_ns must be finite and straddle zero")


@dataclass(frozen=True)
class IRFShapeMismatchDataset:
    """Separate Gaussian development and paired external populations.

    Generic ``BenchmarkMeasurements`` retain raw histograms, true synthetic
    lifetimes, and nuisance metadata. IRFs are aligned by external row index;
    ``pair_id`` identifies the same nuisance condition across external sets.
    """

    config: IRFShapeMismatchConfig
    gaussian_development: BenchmarkMeasurements
    gaussian_control: BenchmarkMeasurements
    emg_external: BenchmarkMeasurements
    true_emg_irfs: tuple[PreparedIRF, ...]
    comparison_gaussian_irfs: tuple[PreparedIRF, ...]


def _match_gaussian_to_emg_profile(emg_profile: IRFProfile) -> IRFProfile:
    """Match sampled principal peak and full FWHM, not asymmetric tail shape."""
    if emg_profile.source_kind is not IRFSourceKind.SYNTHETIC_EMG:
        raise ValueError("a synthetic EMG profile is required")
    time = emg_profile.time_ns
    target_width = _sampled_fwhm(time, emg_profile.values)
    if target_width is None:
        raise ValueError("EMG sampled FWHM is unavailable; no Gaussian match exists")
    target_peak = float(time[int(np.argmax(emg_profile.values))])

    def width_difference(gaussian_fwhm_ns: float) -> float:
        candidate = generate_gaussian_irf_profile(
            time,
            gaussian_centre_ns=target_peak,
            gaussian_fwhm_ns=gaussian_fwhm_ns,
        )
        width = _sampled_fwhm(time, candidate.values)
        if width is None:
            raise ValueError("Gaussian sampled FWHM is unavailable on this grid")
        return width - target_width

    lower, upper = target_width * 1e-3, target_width * 4.0
    if width_difference(lower) >= 0.0 or width_difference(upper) <= 0.0:
        raise ValueError("cannot bracket a sampled Gaussian FWHM match")
    component_fwhm = float(brentq(width_difference, lower, upper, xtol=1e-12))
    return generate_gaussian_irf_profile(
        time,
        gaussian_centre_ns=target_peak,
        gaussian_fwhm_ns=component_fwhm,
        metadata={"comparison": "matched_sampled_peak_and_full_fwhm"},
        provenance={"comparison_to": "noiseless_synthetic_emg_profile"},
    )


def _benchmark_measurements(
    time: FloatArray, histograms: list[IntArray], rows: list[dict[str, Any]]
) -> BenchmarkMeasurements:
    return BenchmarkMeasurements(
        time=time,
        X_histograms=np.stack(histograms),
        y=np.asarray([row["true_lifetime_ns"] for row in rows], dtype=np.float64),
        metadata=pd.DataFrame(rows),
    )


def generate_irf_shape_mismatch_dataset(
    config: IRFShapeMismatchConfig,
) -> IRFShapeMismatchDataset:
    """Generate disjoint Gaussian development and paired external histograms.

    The comparison Gaussian is determined only from the noiseless EMG source,
    never from fluorescence outcomes. Its external control and the EMG test
    share lifetime, photon budget, background, sampled peak, and full FWHM.
    Independent seeded Poisson streams generate each population.
    """
    if not isinstance(config, IRFShapeMismatchConfig):
        raise TypeError("config must be IRFShapeMismatchConfig")
    time = config.time_ns
    streams = np.random.SeedSequence(config.random_seed).spawn(3)
    development_rng, control_rng, emg_rng = (
        np.random.default_rng(stream) for stream in streams
    )
    conditions = list(product(
        config.lifetimes_ns, config.signal_photon_counts, config.tail_times_ns
    ))
    irf_pairs: dict[float, tuple[PreparedIRF, PreparedIRF, float, float]] = {}
    for tail in config.tail_times_ns:
        emg_profile = generate_emg_irf_profile(
            time,
            gaussian_centre_ns=config.gaussian_component_centre_ns,
            gaussian_fwhm_ns=config.gaussian_component_fwhm_ns,
            tail_time_ns=tail,
        )
        gaussian_profile = _match_gaussian_to_emg_profile(emg_profile)
        peak = float(time[int(np.argmax(emg_profile.values))])
        width = _sampled_fwhm(time, emg_profile.values)
        irf_pairs[tail] = (
            prepare_irf(emg_profile, time),
            prepare_irf(gaussian_profile, time),
            peak,
            width,
        )

    development_hist: list[IntArray] = []
    development_rows: list[dict[str, Any]] = []
    control_hist: list[IntArray] = []
    control_rows: list[dict[str, Any]] = []
    emg_hist: list[IntArray] = []
    emg_rows: list[dict[str, Any]] = []
    true_irfs: list[PreparedIRF] = []
    gaussian_irfs: list[PreparedIRF] = []

    for repeat in range(config.development_repeats):
        for condition_id, (lifetime, photons, tail) in enumerate(conditions):
            _, gaussian_irf, peak, width = irf_pairs[tail]
            decay = monoexponential_decay(time, 1.0, lifetime, 0.0)
            expected = build_expected_counts_from_irf(
                decay, gaussian_irf,
                signal_photon_count=photons,
                background_per_bin=config.background_per_bin,
            )
            development_hist.append(sample_photon_counts(expected, development_rng))
            development_rows.append({
                "population": "gaussian_development",
                "development_id": len(development_rows),
                "condition_id": condition_id,
                "replicate": repeat,
                "true_lifetime_ns": lifetime,
                "signal_photon_count_target": photons,
                "background_per_bin": config.background_per_bin,
                "irf_peak_ns": peak,
                "irf_full_fwhm_ns": width,
                "emg_tail_time_ns_design": tail,
            })

    for repeat in range(config.external_repeats):
        for condition_id, (lifetime, photons, tail) in enumerate(conditions):
            true_irf, gaussian_irf, peak, width = irf_pairs[tail]
            decay = monoexponential_decay(time, 1.0, lifetime, 0.0)
            common = {
                "pair_id": len(emg_rows),
                "condition_id": condition_id,
                "replicate": repeat,
                "true_lifetime_ns": lifetime,
                "signal_photon_count_target": photons,
                "background_per_bin": config.background_per_bin,
                "irf_peak_ns": peak,
                "irf_full_fwhm_ns": width,
                "emg_tail_time_ns_design": tail,
            }
            emg_expected = build_expected_counts_from_irf(
                decay, true_irf,
                signal_photon_count=photons,
                background_per_bin=config.background_per_bin,
            )
            control_expected = build_expected_counts_from_irf(
                decay, gaussian_irf,
                signal_photon_count=photons,
                background_per_bin=config.background_per_bin,
            )
            emg_hist.append(sample_photon_counts(emg_expected, emg_rng))
            control_hist.append(sample_photon_counts(control_expected, control_rng))
            emg_rows.append({**common, "population": "emg_external"})
            control_rows.append({**common, "population": "gaussian_control"})
            true_irfs.append(true_irf)
            gaussian_irfs.append(gaussian_irf)

    return IRFShapeMismatchDataset(
        config=config,
        gaussian_development=_benchmark_measurements(time, development_hist, development_rows),
        gaussian_control=_benchmark_measurements(time, control_hist, control_rows),
        emg_external=_benchmark_measurements(time, emg_hist, emg_rows),
        true_emg_irfs=tuple(true_irfs),
        comparison_gaussian_irfs=tuple(gaussian_irfs),
    )


@dataclass(frozen=True)
class IRFClassicalPair:
    """Three fits indexed to the same external nuisance condition.

    The first two fits use *identical* EMG-generated counts. Residual profiles
    are signed Poisson-deviance residuals, or None when a fit is invalid.
    """

    pair_id: int
    matched_emg: ReconvolutionCurveResult
    gaussian_assumed: ReconvolutionCurveResult
    gaussian_control: ReconvolutionCurveResult
    matched_emg_residuals: FloatArray | None
    gaussian_assumed_residuals: FloatArray | None
    gaussian_control_residuals: FloatArray | None


@dataclass(frozen=True)
class IRFClassicalEvaluation:
    """Paired raw results and a tidy per-fit table for later analysis."""

    dataset: IRFShapeMismatchDataset
    pairs: tuple[IRFClassicalPair, ...]
    per_fit: pd.DataFrame


def _synthetic_measurement(time: FloatArray, counts: IntArray) -> TCSPCMeasurement:
    return TCSPCMeasurement(
        time_ns=time,
        values=counts,
        data_kind=MeasurementDataKind.RAW_COUNTS,
    )


def _fit_residuals(
    time: FloatArray, counts: IntArray, irf: PreparedIRF,
    fit: ReconvolutionCurveResult,
) -> FloatArray | None:
    if not fit.valid_fit:
        return None
    physical_fit = _reconstruct_reconvolution_fit_result(
        time=time, irf=irf.kernel, curve_result=fit,
    )
    return calculate_poisson_deviance_residuals(counts, physical_fit.fitted_curve)


def evaluate_irf_shape_classical(
    dataset: IRFShapeMismatchDataset,
) -> IRFClassicalEvaluation:
    """Fit each EMG histogram with matched EMG and matched-width Gaussian.

    A separately generated Gaussian control uses the same nuisance condition.
    All fits use the unchanged Poisson reconvolution adapter and bounds.
    """
    if not isinstance(dataset, IRFShapeMismatchDataset):
        raise TypeError("dataset must be IRFShapeMismatchDataset")
    time = dataset.config.time_ns
    pairs: list[IRFClassicalPair] = []
    rows: list[dict[str, Any]] = []
    for index, (emg_counts, control_counts, emg_irf, gaussian_irf) in enumerate(zip(
        dataset.emg_external.X_histograms,
        dataset.gaussian_control.X_histograms,
        dataset.true_emg_irfs,
        dataset.comparison_gaussian_irfs,
        strict=True,
    )):
        emg_measurement = _synthetic_measurement(time, emg_counts)
        control_measurement = _synthetic_measurement(time, control_counts)
        bounds = dataset.config.temporal_shift_bounds_ns
        matched = fit_experimental_reconvolution(
            emg_measurement, temporal_shift_bounds_ns=bounds, prepared_irf=emg_irf,
        )
        misspecified = fit_experimental_reconvolution(
            emg_measurement, temporal_shift_bounds_ns=bounds, prepared_irf=gaussian_irf,
        )
        control = fit_experimental_reconvolution(
            control_measurement, temporal_shift_bounds_ns=bounds, prepared_irf=gaussian_irf,
        )
        pairs.append(IRFClassicalPair(
            pair_id=index,
            matched_emg=matched,
            gaussian_assumed=misspecified,
            gaussian_control=control,
            matched_emg_residuals=_fit_residuals(time, emg_counts, emg_irf, matched),
            gaussian_assumed_residuals=_fit_residuals(
                time, emg_counts, gaussian_irf, misspecified,
            ),
            gaussian_control_residuals=_fit_residuals(
                time, control_counts, gaussian_irf, control,
            ),
        ))
        design = dataset.emg_external.metadata.iloc[index]
        truth = float(design["true_lifetime_ns"])
        for label, population, fit in (
            ("matched_emg", "emg_external", matched),
            ("gaussian_assumed", "emg_external", misspecified),
            ("gaussian_control", "gaussian_control", control),
        ):
            if fit.valid_fit:
                errors = calculate_lifetime_errors(
                    np.array([truth]), np.array([fit.fitted_lifetime_ns]),
                )
                signed = float(errors[0][0])
                absolute = float(errors[1][0])
                relative = float(errors[2][0])
            else:
                signed = absolute = relative = float("nan")
            rows.append({
                "pair_id": index,
                "condition_id": int(design["condition_id"]),
                "replicate": int(design["replicate"]),
                "population": population,
                "assumed_irf": label,
                "true_lifetime_ns": truth,
                "estimated_lifetime_ns": fit.fitted_lifetime_ns,
                "signed_error_ns": signed,
                "absolute_error_ns": absolute,
                "relative_error": relative,
                "valid_fit": fit.valid_fit,
                "optimizer_success": fit.optimizer_success,
                "boundary_hit": fit.boundary_hit,
                "poisson_deviance": fit.poisson_deviance,
                "poisson_nll": fit.poisson_nll,
                "failure_reason": fit.failure_reason,
                "signal_photon_count_target": int(design["signal_photon_count_target"]),
                "emg_tail_time_ns": float(design["emg_tail_time_ns_design"]),
            })
    return IRFClassicalEvaluation(dataset, tuple(pairs), pd.DataFrame(rows))


def summarize_irf_shape_classical(evaluation: IRFClassicalEvaluation) -> pd.DataFrame:
    """Summarize valid-fit lifetime error and deviance by assumed IRF."""
    rows: list[dict[str, Any]] = []
    for assumption, group in evaluation.per_fit.groupby("assumed_irf", sort=False):
        valid = group[group["valid_fit"]]
        rows.append({
            "assumed_irf": assumption,
            "n_histograms": len(group),
            "n_valid_fits": len(valid),
            "n_boundary_hits": int(group["boundary_hit"].sum()),
            "mae_ns": float(valid["absolute_error_ns"].mean()) if len(valid) else np.nan,
            "bias_ns": float(valid["signed_error_ns"].mean()) if len(valid) else np.nan,
            "mean_poisson_deviance": (
                float(valid["poisson_deviance"].mean()) if len(valid) else np.nan
            ),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class IRFConditionalUncertaintyEvaluation:
    """Model-conditional classical uncertainty on selected external pairs.

    Covariance and bootstrap keep each assumed IRF fixed. Neither incorporates
    uncertainty in the IRF source, shape family, or registration.
    """

    per_fit: pd.DataFrame
    detailed_results: tuple[
        tuple[int, str, PoissonLocalCovarianceResult, ParametricPoissonBootstrapResult], ...
    ]


def evaluate_irf_shape_conditional_uncertainty(
    dataset: IRFShapeMismatchDataset,
    classical: IRFClassicalEvaluation,
    *,
    pair_ids: tuple[int, ...],
    n_bootstrap_resamples: int = 6,
    nominal_coverage: float = 0.90,
    random_seed: int = 8_602,
) -> IRFConditionalUncertaintyEvaluation:
    """Run existing Fisher covariance and Poisson bootstrap for selected fits.

    The caller chooses a small subset explicitly; invalid fits are reported
    as skipped. Intervals describe sampling variation *conditional on the
    assumed fixed IRF*, not physical model adequacy or IRF uncertainty.
    """
    if not isinstance(dataset, IRFShapeMismatchDataset):
        raise TypeError("dataset must be IRFShapeMismatchDataset")
    if not isinstance(classical, IRFClassicalEvaluation):
        raise TypeError("classical must be IRFClassicalEvaluation")
    if classical.dataset is not dataset:
        raise ValueError("classical evaluation must come from the supplied dataset")
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_ids must be nonempty and distinct")
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or i < 0 or i >= len(classical.pairs)
           for i in pair_ids):
        raise ValueError("pair_ids contain an out-of-range index")
    if isinstance(n_bootstrap_resamples, (bool, np.bool_)) or not isinstance(
        n_bootstrap_resamples, (int, np.integer)
    ) or n_bootstrap_resamples < 2:
        raise ValueError("n_bootstrap_resamples must be at least 2")
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(
        random_seed, (int, np.integer)
    ) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not np.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must lie strictly between zero and one")
    rows: list[dict[str, Any]] = []
    detailed = []
    streams = np.random.SeedSequence(random_seed).spawn(2 * len(pair_ids))
    for pair_index, pair_id in enumerate(pair_ids):
        pair = classical.pairs[pair_id]
        truth = float(dataset.emg_external.y[pair_id])
        for offset, (label, fit, irf) in enumerate((
            ("matched_emg", pair.matched_emg, dataset.true_emg_irfs[pair_id]),
            ("gaussian_assumed", pair.gaussian_assumed,
             dataset.comparison_gaussian_irfs[pair_id]),
        )):
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "assumed_irf": label,
                "true_lifetime_ns": truth,
                "estimated_lifetime_ns": fit.fitted_lifetime_ns,
                "fit_valid": fit.valid_fit,
                "uncertainty_scope": "conditional_on_assumed_fixed_irf",
            }
            if fit.valid_fit:
                reconstructed = _reconstruct_reconvolution_fit_result(
                    time=dataset.config.time_ns,
                    irf=irf.kernel,
                    curve_result=fit,
                )
                covariance = estimate_poisson_reconvolution_local_covariance(
                    time=dataset.config.time_ns,
                    irf=irf.kernel,
                    fit_result=reconstructed,
                    temporal_shift_bounds=dataset.config.temporal_shift_bounds_ns,
                )
                bootstrap = estimate_parametric_poisson_bootstrap(
                    time=dataset.config.time_ns,
                    irf=irf.kernel,
                    fit_result=reconstructed,
                    temporal_shift_bounds=dataset.config.temporal_shift_bounds_ns,
                    rng=np.random.default_rng(streams[2 * pair_index + offset]),
                    n_resamples=n_bootstrap_resamples,
                    nominal_coverage=nominal_coverage,
                )
                detailed.append((pair_id, label, covariance, bootstrap))
                row.update({
                    "covariance_valid": covariance.covariance_valid,
                    "local_lifetime_std_ns": covariance.lifetime_std,
                    "local_failure_reason": covariance.failure_reason,
                    "bootstrap_valid": bootstrap.bootstrap_valid,
                    "bootstrap_lifetime_std_ns": bootstrap.bootstrap_std_ns,
                    "bootstrap_lower_ns": bootstrap.lower_ns,
                    "bootstrap_upper_ns": bootstrap.upper_ns,
                    "bootstrap_n_successful_fits": bootstrap.n_successful_fits,
                    "bootstrap_failure_reason": bootstrap.failure_reason,
                })
            else:
                row.update({
                    "covariance_valid": False,
                    "local_lifetime_std_ns": np.nan,
                    "local_failure_reason": "source fit invalid",
                    "bootstrap_valid": False,
                    "bootstrap_lifetime_std_ns": np.nan,
                    "bootstrap_lower_ns": np.nan,
                    "bootstrap_upper_ns": np.nan,
                    "bootstrap_n_successful_fits": 0,
                    "bootstrap_failure_reason": "source fit invalid",
                })
            rows.append(row)
    return IRFConditionalUncertaintyEvaluation(pd.DataFrame(rows), tuple(detailed))


@dataclass(frozen=True)
class IRFMLTransferEvaluation:
    """Gaussian-only training and two untouched external evaluations.

    ``rf_tree_spread_ns`` is ensemble disagreement, not a nominal or
    calibrated prediction interval. No external population fits a model,
    scaler, or representation.
    """

    n_gaussian_development_samples: int
    feature_names: tuple[str, ...]
    per_prediction: pd.DataFrame
    summary: pd.DataFrame


def evaluate_gaussian_trained_irf_transfer(
    dataset: IRFShapeMismatchDataset,
    *,
    feature_config: FeatureConfig,
    random_state: int = 42,
) -> IRFMLTransferEvaluation:
    """Fit Ridge, HGB, and RF on Gaussian development features only.

    Feature extraction is a fixed, per-histogram transformation. All fitted
    model/scaler state comes from ``gaussian_development``; both external
    populations are evaluated without tuning or refitting.
    """
    if not isinstance(dataset, IRFShapeMismatchDataset):
        raise TypeError("dataset must be IRFShapeMismatchDataset")
    if not isinstance(feature_config, FeatureConfig):
        raise TypeError("feature_config must be FeatureConfig")
    training = dataset.gaussian_development
    train_features = extract_feature_table(
        training.X_histograms, training.time, feature_config,
    )
    feature_names = tuple(train_features.columns)
    X_train = train_features.to_numpy(dtype=np.float64)
    y_train = np.asarray(training.y, dtype=np.float64)
    models = {
        "ridge": make_ridge_pipeline(),
        "hist_gradient_boosting": make_hist_gradient_boosting_pipeline(
            random_state=random_state,
        ),
        "random_forest": RandomForestTreeSpreadEstimator(random_state=random_state),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, population in (
        ("gaussian_control", dataset.gaussian_control),
        ("emg_external", dataset.emg_external),
    ):
        features = extract_feature_table(
            population.X_histograms, population.time, feature_config,
        )
        if tuple(features.columns) != feature_names:
            raise ValueError("external feature columns differ from development")
        X = features.to_numpy(dtype=np.float64)
        y_true = np.asarray(population.y, dtype=np.float64)
        for model_name, model in models.items():
            if model_name == "random_forest":
                scored = model.predict(X)
                y_pred = scored.prediction
                tree_spread = scored.uncertainty_score
            else:
                y_pred = np.asarray(model.predict(X), dtype=np.float64)
                tree_spread = np.full(y_true.shape, np.nan)
            signed, absolute, relative = calculate_lifetime_errors(y_true, y_pred)
            metrics = evaluate_regression(y_true, y_pred)
            summaries.append({
                "population": label,
                "model": model_name,
                "n_histograms": len(y_true),
                "mae_ns": metrics.mae_ns,
                "rmse_ns": metrics.rmse_ns,
                "bias_ns": float(np.mean(signed)),
                "mean_relative_error": metrics.mean_relative_error,
                "mean_rf_tree_spread_ns": float(np.mean(tree_spread))
                if model_name == "random_forest" else np.nan,
            })
            for index in range(len(y_true)):
                design = population.metadata.iloc[index]
                rows.append({
                    "population": label,
                    "model": model_name,
                    "pair_id": int(design["pair_id"]),
                    "condition_id": int(design["condition_id"]),
                    "true_lifetime_ns": float(y_true[index]),
                    "estimated_lifetime_ns": float(y_pred[index]),
                    "signed_error_ns": float(signed[index]),
                    "absolute_error_ns": float(absolute[index]),
                    "relative_error": float(relative[index]),
                    "rf_tree_spread_ns": float(tree_spread[index]),
                    "signal_photon_count_target": int(
                        design["signal_photon_count_target"]
                    ),
                    "emg_tail_time_ns_design": float(
                        design["emg_tail_time_ns_design"]
                    ),
                })
    return IRFMLTransferEvaluation(
        n_gaussian_development_samples=len(y_train),
        feature_names=feature_names,
        per_prediction=pd.DataFrame(rows),
        summary=pd.DataFrame(summaries),
    )


@dataclass(frozen=True)
class LeadingEdgeFailureConfig:
    """One-factor-at-a-time synthetic proxy study, independent of Week 8."""

    time_ns: FloatArray
    gaussian_irf_peak_ns: float = 2.0
    gaussian_irf_fwhm_ns: float = 0.4
    long_lifetime_ns: float = 8.0
    signal_photon_count: int = 2_000_000
    background_per_bin: float = 1.0
    background_window_ns: tuple[float, float] = (0.0, 1.0)
    rising_edge_window_ns: tuple[float, float] = (1.2, 4.0)
    smoothing_window_bins: int = 31
    polynomial_order: int = 3
    low_photon_count: int = 2_000
    high_background_per_bin: float = 5_000.0
    comparable_lifetime_ns: float = 0.4
    very_short_lifetime_ns: float = 0.08
    biexponential_short_lifetime_ns: float = 0.4
    biexponential_short_amplitude_fraction: float = 0.9
    misregistration_ns: float = 2.2
    asymmetric_tail_time_ns: float = 0.4
    finite_rise_time_ns: float = 0.5
    n_repeats: int = 1
    random_seed: int = 8_603
    temporal_shift_bounds_ns: tuple[float, float] = (-0.2, 0.2)

    def __post_init__(self) -> None:
        time = validate_time_axis(self.time_ns)
        copied = np.array(time, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "time_ns", copied)
        for name in (
            "gaussian_irf_fwhm_ns", "long_lifetime_ns", "comparable_lifetime_ns",
            "very_short_lifetime_ns", "biexponential_short_lifetime_ns",
            "asymmetric_tail_time_ns", "finite_rise_time_ns",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.gaussian_irf_peak_ns):
            raise ValueError("gaussian_irf_peak_ns must be finite")
        if not 0 < self.biexponential_short_amplitude_fraction < 1:
            raise ValueError("biexponential_short_amplitude_fraction must lie in (0, 1)")
        for name in ("signal_photon_count", "low_photon_count", "n_repeats"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, (bool, np.bool_)) or not isinstance(
            self.random_seed, (int, np.integer)
        ) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        for name in ("background_per_bin", "high_background_per_bin"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.misregistration_ns) or self.misregistration_ns <= 0:
            raise ValueError("misregistration_ns must be finite and positive")
        if self.finite_rise_time_ns >= self.long_lifetime_ns:
            raise ValueError("finite_rise_time_ns must be shorter than long_lifetime_ns")
        if self.low_photon_count >= self.signal_photon_count:
            raise ValueError("low_photon_count must be below signal_photon_count")
        if self.high_background_per_bin <= self.background_per_bin:
            raise ValueError("high_background_per_bin must exceed background_per_bin")
        if not self.very_short_lifetime_ns < self.comparable_lifetime_ns < self.long_lifetime_ns:
            raise ValueError("lifetime regimes must increase from very short to long")
        if self.rising_edge_window_ns[1] + self.misregistration_ns > copied[-1]:
            raise ValueError("translated rising-edge window exceeds the acquisition grid")


@dataclass(frozen=True)
class LeadingEdgeRegimeResult:
    """Raw synthetic realization, proxy status, truth comparison, and fits."""

    regime: str
    realization: int
    generating_model: str
    monoexponential_reference_lifetime_ns: float | None
    generating_parameters: dict[str, float]
    measurement: TCSPCMeasurement
    true_prepared_irf: PreparedIRF
    proxy: LeadingEdgeIRFResult
    true_irf_fit: ReconvolutionCurveResult
    proxy_irf_fit: ReconvolutionCurveResult | None
    proxy_l1_shape_error: float | None
    proxy_peak_time_error_ns: float | None
    proxy_fwhm_error_ns: float | None


@dataclass(frozen=True)
class LeadingEdgeFailureEvaluation:
    """Per-realization controlled regimes and tidy metrics for Notebook 16."""

    cases: tuple[LeadingEdgeRegimeResult, ...]
    per_case: pd.DataFrame


def _finite_rise_decay(
    time_ns: FloatArray, *, tau_rise_ns: float, tau_decay_ns: float,
) -> FloatArray:
    """Issue-#8 control confound, not a general fluorescence-rise model.

    d(u) is proportional to exp(-u/tau_decay)-exp(-u/tau_rise) for u>=0,
    and zero before onset. Stage-4 photon-budget scaling handles amplitude.
    """
    if not 0 < tau_rise_ns < tau_decay_ns:
        raise ValueError("finite-rise times require 0 < tau_rise_ns < tau_decay_ns")
    u = np.asarray(time_ns, dtype=np.float64)
    return np.where(u >= 0.0,
                    np.exp(-np.maximum(u, 0.0) / tau_decay_ns)
                    - np.exp(-np.maximum(u, 0.0) / tau_rise_ns),
                    0.0)


def _asymmetric_regime_irf(config: LeadingEdgeFailureConfig) -> PreparedIRF:
    """Set EMG full sampled FWHM/peak near the Gaussian baseline."""
    time = config.time_ns
    target_width = config.gaussian_irf_fwhm_ns

    def width_difference(component_width: float) -> float:
        profile = generate_emg_irf_profile(
            time,
            gaussian_centre_ns=config.gaussian_irf_peak_ns,
            gaussian_fwhm_ns=component_width,
            tail_time_ns=config.asymmetric_tail_time_ns,
        )
        width = _sampled_fwhm(time, profile.values)
        if width is None:
            raise ValueError("EMG sampled FWHM unavailable for asymmetric regime")
        return width - target_width

    lower, upper = target_width * 0.1, target_width * 4.0
    if width_difference(lower) >= 0 or width_difference(upper) <= 0:
        raise ValueError("cannot match EMG full FWHM to Gaussian baseline")
    component_width = brentq(width_difference, lower, upper, xtol=1e-12)
    profile = generate_emg_irf_profile(
        time,
        gaussian_centre_ns=config.gaussian_irf_peak_ns,
        gaussian_fwhm_ns=component_width,
        tail_time_ns=config.asymmetric_tail_time_ns,
    )
    sampled_peak = float(time[int(np.argmax(profile.values))])
    return prepare_irf(
        profile, time, resampling="linear",
        registration_offset_ns=config.gaussian_irf_peak_ns - sampled_peak,
    )


def _leading_edge_regime_specifications(
    config: LeadingEdgeFailureConfig,
    baseline_irf: PreparedIRF,
) -> tuple[dict[str, Any], ...]:
    """Only the named factor differs from the favorable baseline."""
    time = config.time_ns
    shifted_irf = prepare_irf(
        generate_gaussian_irf_profile(
            time,
            gaussian_centre_ns=config.gaussian_irf_peak_ns,
            gaussian_fwhm_ns=config.gaussian_irf_fwhm_ns,
        ),
        time, resampling="linear", registration_offset_ns=config.misregistration_ns,
    )
    translated_window = (
        config.rising_edge_window_ns[0] + config.misregistration_ns,
        config.rising_edge_window_ns[1] + config.misregistration_ns,
    )
    return (
        {"regime": "favorable", "irf": baseline_irf,
         "lifetime": config.long_lifetime_ns},
        {"regime": "low_photon_count", "irf": baseline_irf,
         "lifetime": config.long_lifetime_ns,
         "photons": config.low_photon_count},
        {"regime": "high_background", "irf": baseline_irf,
         "lifetime": config.long_lifetime_ns,
         "background": config.high_background_per_bin},
        {"regime": "comparable_lifetime", "irf": baseline_irf,
         "lifetime": config.comparable_lifetime_ns},
        {"regime": "very_short_lifetime", "irf": baseline_irf,
         "lifetime": config.very_short_lifetime_ns},
        {"regime": "bi_exponential", "irf": baseline_irf,
         "model": "bi_exponential"},
        {"regime": "translated_irf_correct_window", "irf": shifted_irf,
         "lifetime": config.long_lifetime_ns,
         "edge_window": translated_window},
        {"regime": "misregistered_analysis_window", "irf": shifted_irf,
         "lifetime": config.long_lifetime_ns},
        {"regime": "strongly_asymmetric_irf", "irf": _asymmetric_regime_irf(config),
         "lifetime": config.long_lifetime_ns},
        {"regime": "finite_physical_rise", "irf": baseline_irf,
         "model": "finite_physical_rise"},
    )


def evaluate_leading_edge_failure_regimes(
    config: LeadingEdgeFailureConfig,
) -> LeadingEdgeFailureEvaluation:
    """Study eight isolated departures plus a registered timing control.

    An available proxy is only numerically constructible; shape error against
    the known synthetic IRF and downstream fit error are separate outcomes.
    Windows stay fixed except the explicit correctly translated control.
    Bi-exponential data have no single mono-exponential ground-truth lifetime.
    """
    if not isinstance(config, LeadingEdgeFailureConfig):
        raise TypeError("config must be LeadingEdgeFailureConfig")
    time = config.time_ns
    baseline_irf = prepare_irf(generate_gaussian_irf_profile(
        time,
        gaussian_centre_ns=config.gaussian_irf_peak_ns,
        gaussian_fwhm_ns=config.gaussian_irf_fwhm_ns,
    ), time)
    regimes = _leading_edge_regime_specifications(config, baseline_irf)
    streams = np.random.SeedSequence(config.random_seed).spawn(
        len(regimes) * config.n_repeats
    )
    cases: list[LeadingEdgeRegimeResult] = []
    rows: list[dict[str, Any]] = []
    for regime_index, spec in enumerate(regimes):
        for realization in range(config.n_repeats):
            irf = spec["irf"]
            model = spec.get("model", "mono_exponential")
            if model == "bi_exponential":
                short_fraction = config.biexponential_short_amplitude_fraction
                ideal = biexponential_decay(
                    time, short_fraction, config.biexponential_short_lifetime_ns,
                    1.0 - short_fraction, config.long_lifetime_ns, 0.0,
                )
                reference_lifetime = None
                parameters = {
                    "short_lifetime_ns": config.biexponential_short_lifetime_ns,
                    "long_lifetime_ns": config.long_lifetime_ns,
                    "short_amplitude_fraction": short_fraction,
                }
            elif model == "finite_physical_rise":
                ideal = _finite_rise_decay(
                    time, tau_rise_ns=config.finite_rise_time_ns,
                    tau_decay_ns=config.long_lifetime_ns,
                )
                reference_lifetime = None  # a mono-exponential fit is misspecified
                parameters = {
                    "rise_time_ns": config.finite_rise_time_ns,
                    "decay_time_ns": config.long_lifetime_ns,
                }
            else:
                reference_lifetime = float(spec["lifetime"])
                ideal = monoexponential_decay(time, 1.0, reference_lifetime, 0.0)
                parameters = {"lifetime_ns": reference_lifetime}
            photons = int(spec.get("photons", config.signal_photon_count))
            background = float(spec.get("background", config.background_per_bin))
            if spec["regime"] in (
                "translated_irf_correct_window", "misregistered_analysis_window"
            ):
                parameters["known_irf_translation_ns"] = config.misregistration_ns
            if spec["regime"] == "strongly_asymmetric_irf":
                parameters["emg_tail_time_ns"] = config.asymmetric_tail_time_ns
            expected = build_expected_counts_from_irf(
                ideal, irf, signal_photon_count=photons,
                background_per_bin=background,
            )
            rng = np.random.default_rng(
                streams[regime_index * config.n_repeats + realization]
            )
            measurement = _synthetic_measurement(time, sample_photon_counts(expected, rng))
            proxy = estimate_irf_from_leading_edge(
                measurement,
                background_window_ns=config.background_window_ns,
                rising_edge_window_ns=spec.get(
                    "edge_window", config.rising_edge_window_ns,
                ),
                smoothing_window_bins=config.smoothing_window_bins,
                polynomial_order=config.polynomial_order,
            )
            true_fit = fit_experimental_reconvolution(
                measurement,
                temporal_shift_bounds_ns=config.temporal_shift_bounds_ns,
                prepared_irf=irf,
            )
            proxy_fit = None
            l1 = peak_error = width_error = None
            if proxy.profile is not None:
                prepared_proxy = prepare_irf(proxy.profile, time)
                l1 = float(np.trapezoid(
                    np.abs(prepared_proxy.kernel - irf.kernel), x=time,
                ))
                true_peak = float(time[int(np.argmax(irf.kernel))])
                proxy_peak = float(time[int(np.argmax(prepared_proxy.kernel))])
                peak_error = proxy_peak - true_peak
                true_width = _sampled_fwhm(time, irf.kernel)
                proxy_width = _sampled_fwhm(time, prepared_proxy.kernel)
                if true_width is not None and proxy_width is not None:
                    width_error = proxy_width - true_width
                proxy_fit = fit_experimental_reconvolution(
                    measurement,
                    temporal_shift_bounds_ns=config.temporal_shift_bounds_ns,
                    prepared_irf=prepared_proxy,
                )
            cases.append(LeadingEdgeRegimeResult(
                regime=spec["regime"], realization=realization,
                generating_model=model,
                monoexponential_reference_lifetime_ns=reference_lifetime,
                generating_parameters=parameters,
                measurement=measurement,
                true_prepared_irf=irf,
                proxy=proxy,
                true_irf_fit=true_fit,
                proxy_irf_fit=proxy_fit,
                proxy_l1_shape_error=l1,
                proxy_peak_time_error_ns=peak_error,
                proxy_fwhm_error_ns=width_error,
            ))
            rows.append({
                "regime": spec["regime"],
                "realization": realization,
                "generating_model": model,
                "monoexponential_reference_lifetime_ns": reference_lifetime,
                "signal_photon_count_target": photons,
                "background_per_bin": background,
                "proxy_status": proxy.diagnostics.status,
                "proxy_failure_reason": proxy.diagnostics.failure_reason,
                "proxy_flags": proxy.diagnostics.flags,
                "requested_rising_edge_window_ns": (
                    proxy.diagnostics.requested_rising_edge_window_ns
                ),
                "background_estimate_counts_per_bin": (
                    proxy.diagnostics.background_estimate_counts_per_bin
                ),
                "estimated_background_fraction_of_rising_counts": (
                    proxy.diagnostics.estimated_background_fraction_of_rising_counts
                ),
                "negative_derivative_fraction": (
                    proxy.diagnostics.negative_derivative_fraction
                ),
                "proxy_fwhm_ns": proxy.diagnostics.proxy_fwhm_ns,
                "proxy_l1_shape_error": l1,
                "proxy_peak_time_error_ns": peak_error,
                "proxy_fwhm_error_ns": width_error,
                "true_irf_lifetime_estimate_ns": true_fit.fitted_lifetime_ns,
                "proxy_irf_lifetime_estimate_ns": (
                    proxy_fit.fitted_lifetime_ns if proxy_fit is not None else np.nan
                ),
                "true_irf_fit_valid": true_fit.valid_fit,
                "proxy_irf_fit_valid": (
                    proxy_fit.valid_fit if proxy_fit is not None else False
                ),
                "true_irf_poisson_deviance": true_fit.poisson_deviance,
                "proxy_irf_poisson_deviance": (
                    proxy_fit.poisson_deviance if proxy_fit is not None else np.nan
                ),
                "true_irf_lifetime_error_ns": (
                    true_fit.fitted_lifetime_ns - reference_lifetime
                    if reference_lifetime is not None and true_fit.valid_fit else np.nan
                ),
                "proxy_irf_lifetime_error_ns": (
                    proxy_fit.fitted_lifetime_ns - reference_lifetime
                    if reference_lifetime is not None and proxy_fit is not None
                    and proxy_fit.valid_fit else np.nan
                ),
            })
    return LeadingEdgeFailureEvaluation(tuple(cases), pd.DataFrame(rows))


def summarize_leading_edge_regimes(
    evaluation: LeadingEdgeFailureEvaluation,
) -> pd.DataFrame:
    """Summarize construction and truth-comparison metrics without hiding cases."""
    rows: list[dict[str, Any]] = []
    for regime, group in evaluation.per_case.groupby("regime", sort=False):
        rows.append({
            "regime": regime,
            "n_realizations": len(group),
            "n_proxy_constructed": int((group["proxy_status"] == "proxy_constructed").sum()),
            "mean_proxy_l1_shape_error": float(group["proxy_l1_shape_error"].mean()),
            "mean_abs_proxy_lifetime_error_ns": float(
                group["proxy_irf_lifetime_error_ns"].abs().mean()
            ),
            "mean_abs_true_irf_lifetime_error_ns": float(
                group["true_irf_lifetime_error_ns"].abs().mean()
            ),
        })
    return pd.DataFrame(rows)
