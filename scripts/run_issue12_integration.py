"""Small cross-feature acceptance, not a scientific benchmark/reference generator.

Run with --block-emcee for base-only acceptance or --bayesian for a tiny real
sampling smoke. All four workflows run twice; temporary CSVs/databases are
removed. No notebook, reference, benchmark cache, or production file is written.
"""

from __future__ import annotations

import argparse
from contextlib import closing, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import importlib.abc
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
# Reuse the existing acceptance blocker; importing this module runs no science.
_spec = importlib.util.spec_from_file_location(
    "issue12_scientific_helpers", ROOT / "scripts/run_issue12_scientific_reproducibility.py",
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
BlockFlatFacades = _helpers.BlockFlatFacades
require = _helpers.require


class BlockEmcee(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "emcee" or fullname.startswith("emcee."):
            raise ModuleNotFoundError("Stage-4 deliberately blocked emcee", name="emcee")
        return None


def fingerprint(array):
    import numpy as np
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((values.dtype.str, values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def classical_projection(measurement, prepared, *, reference=None):
    """Compose public carriers; no second fit or uncertainty algorithm here."""
    import numpy as np
    from tcspc_toolkit import fit_experimental_reconvolution
    from tcspc_toolkit.estimation.fitting import ReconvolutionFitResult
    from tcspc_toolkit.physics.forward_model import monoexponential_reconvolution_expected_counts
    from tcspc_toolkit.evaluation.core import build_point_evaluation
    from tcspc_toolkit.evaluation.results import EvaluationBatch, MethodDescriptor
    from tcspc_toolkit.uncertainty.classical import (
        estimate_poisson_reconvolution_local_covariance,
        project_poisson_local_covariance_interval,
    )

    curve = fit_experimental_reconvolution(
        measurement, prepared_irf=prepared, temporal_shift_bounds_ns=(-.3, .3),
    )
    # The measurement adapter returns diagnostic curve facts, whereas covariance
    # consumes the low-level fit carrier. Reconstruct only that public carrier,
    # evaluating its fitted curve with the shared production forward model.
    fit = ReconvolutionFitResult(
        amplitude=curve.fitted_amplitude, lifetime=curve.fitted_lifetime_ns,
        background=curve.fitted_background, temporal_shift=curve.fitted_temporal_shift_ns,
        fitted_curve=monoexponential_reconvolution_expected_counts(
            measurement.time_ns, prepared.kernel, amplitude=curve.fitted_amplitude,
            lifetime=curve.fitted_lifetime_ns, background=curve.fitted_background,
            temporal_shift=curve.fitted_temporal_shift_ns,
        ),
        success=curve.valid_fit, optimizer_reported_success=curve.optimizer_success,
        numerical_validation_passed=curve.numerical_validation_passed,
    )
    covariance = estimate_poisson_reconvolution_local_covariance(
        time=measurement.time_ns, irf=prepared.kernel, fit_result=fit,
        temporal_shift_bounds=(-.3, .3),
    )
    batch = EvaluationBatch(
        "integration/classical", [measurement.sample_id],
        references={} if reference is None else {reference.reference_id: reference},
    )
    method = MethodDescriptor("integration/poisson", "classical")
    points = build_point_evaluation(
        batch=batch, method=method, lifetime_estimates_ns=[fit.lifetime],
        is_valid=[curve.valid_fit], failure_reasons=[curve.failure_reason],
    )
    attachment = project_poisson_local_covariance_interval(
        batch=batch, points=points, method=method, sample_id=measurement.sample_id,
        curve=curve, covariance=covariance, nominal_level=.9,
    )
    require(curve.valid_fit and np.isfinite(curve.poisson_deviance), "Classical wiring failed")
    return curve, fit, covariance, batch, method, points, attachment


def synthetic_workflow():
    import numpy as np
    from tcspc_toolkit import (
        MeasurementDataKind, TCSPCMeasurement, monoexponential_decay, prepare_irf,
        normalize_histogram_batch, fit_experimental_reconvolution,
    )
    from tcspc_toolkit.config import CountNormalization
    from tcspc_toolkit.irf.profiles import generate_gaussian_irf_profile
    from tcspc_toolkit.simulation.histograms import build_expected_counts_from_irf, sample_photon_counts
    from tcspc_toolkit.evaluation.results import LifetimeReference

    time = np.arange(0., 12., .1)
    source = generate_gaussian_irf_profile(
        time, gaussian_centre_ns=.8, gaussian_fwhm_ns=.35,
        metadata={"role": "synthetic generating source"},
    )
    prepared = prepare_irf(source, time)
    original_source = source.values.copy()
    expected = build_expected_counts_from_irf(
        monoexponential_decay(time, 1., 2., 0.), prepared,
        signal_photon_count=30000, background_per_bin=3.,
    )
    counts = sample_photon_counts(expected, np.random.default_rng(120401))
    measurement = TCSPCMeasurement(
        time, counts, MeasurementDataKind.RAW_COUNTS, sample_id="synthetic-0",
        metadata={"observation_seed": 120401},
    )
    require(np.issubdtype(counts.dtype, np.integer), "Simulation lost raw counts")
    normalized = normalize_histogram_batch(counts[None, :], mode=CountNormalization.TOTAL)
    require(np.isclose(normalized.sum(), 1.), "Normalization failed")
    processed = TCSPCMeasurement(time, normalized[0], MeasurementDataKind.PROCESSED_INTENSITY)
    try:
        fit_experimental_reconvolution(processed, prepared_irf=prepared,
                                      temporal_shift_bounds_ns=(-.3, .3))
    except ValueError as exc:
        require("raw" in str(exc).lower(), "Unexpected processed-data rejection")
    else:
        raise AssertionError("Processed intensities reached Poisson fitting")
    projection = classical_projection(
        measurement, prepared,
        reference=LifetimeReference("generating/2ns", "generating_mono", [2.], [True]),
    )
    curve, fit, covariance, _, _, points, attachment = projection
    require(abs(fit.lifetime - 2.) < .25 and covariance.covariance_valid,
            "Synthetic classical/covariance sanity failed")
    np.testing.assert_array_equal(measurement.values, counts)
    np.testing.assert_array_equal(source.values, original_source)
    require(prepared.source is source and prepared is not source, "IRF provenance collapsed")
    report = {
        "counts": fingerprint(counts), "source": fingerprint(source.values),
        "kernel": fingerprint(prepared.kernel), "preparation": asdict(prepared.diagnostics),
        "source_kind": source.source_kind.value,
        "parameters": [fit.amplitude, fit.lifetime, fit.background, fit.temporal_shift],
        "nll": curve.poisson_nll, "covariance_std": covariance.lifetime_std,
        "points": points.points.to_dict("records"),
        "references": points.reference_comparisons.to_dict("records"),
        "intervals": attachment.intervals.to_dict("records"),
        "processed_rejected": True,
    }
    return measurement, prepared, projection, report


def ml_workflow():
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tcspc_toolkit import (
        EstimatorSpec, fit_regressors, predict_regressors,
        generate_monoexponential_dataset, extract_feature_table, FEATURE_NAMES,
    )
    from tcspc_toolkit.config import FeatureConfig
    from tcspc_toolkit.evaluation.results import EvaluationBatch, LifetimeReference, MethodDescriptor
    from tcspc_toolkit.evaluation.core import build_point_evaluation

    data = generate_monoexponential_dataset(
        n_curves=32, time=np.arange(0., 16., .1), lifetime_range=(1., 3.),
        amplitude_range=(100., 200.), background_range=(1., 3.),
        photon_count_range=(20000, 30000), random_seed=120402,
    )
    features = extract_feature_table(histograms=data.X, time=data.time, config=FeatureConfig(
        tail_start_ns=4., early_stop_ns=2., late_start_ns=5., min_tail_points=5,
    ))
    require(list(features.columns) == list(FEATURE_NAMES), "Feature schema changed")
    # Stateless features may be extracted for all rows. Learned scaling/model
    # fitting receives only the first 24 rows; evaluation preserves an explicit
    # deliberately non-sorted order, independently of DataFrame index labels.
    train = np.arange(24)
    test = np.array([31, 24, 30, 25, 29, 26, 28, 27])
    X_train, X_test = features.to_numpy()[train], features.to_numpy()[test]
    targets = data.metadata.lifetime_true.to_numpy()
    fitted = fit_regressors(
        estimator_specs=(EstimatorSpec("user-linear", lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=1.)), ("features",)),),
        X_train_by_representation={"features": X_train}, y_train=targets[train],
    )
    model = fitted["user-linear"]["features"]
    scaler = model.named_steps["standardscaler"]
    require(scaler.n_samples_seen_ == 24, "Held-out data entered learned preprocessing")
    np.testing.assert_array_equal(scaler.mean_, X_train.mean(axis=0))
    predictions = predict_regressors(
        fitted_estimators=fitted, X_by_representation={"features": X_test},
    )["user-linear"]["features"]
    np.testing.assert_array_equal(predictions, model.predict(X_test))
    batch = EvaluationBatch(
        "integration/ml", [f"development-{i}" for i in test],
        representations={"features": X_test},
        references={"generating": LifetimeReference("generating", "generating_mono",
                                                   targets[test], np.ones(8, dtype=bool))},
    )
    method = MethodDescriptor("user-linear", "ml", "features")
    points = build_point_evaluation(batch=batch, method=method,
                                   lifetime_estimates_ns=predictions, is_valid=np.isfinite(predictions))
    require(points.points.sample_id.tolist() == list(batch.sample_ids), "ML sample ordering changed")
    require(np.isfinite(predictions).all(), "Nonfinite ML predictions")
    return data, test, points, {
        "counts": fingerprint(data.X), "train_ids": train.tolist(), "test_ids": test.tolist(),
        "training_scaler_mean": scaler.mean_.tolist(), "predictions": predictions.tolist(),
        "points": points.points.to_dict("records"),
        "references": points.reference_comparisons.to_dict("records"),
    }


def experimental_workflow(directory, *, bayesian=False, block_emcee=False):
    import numpy as np
    from tcspc_toolkit import (
        load_sampled_irf_csv, load_tcspc_measurement_csv, irf_profile_from_sampled_irf,
        prepare_irf, monoexponential_decay, measurement_to_histogram_batch,
        measurement_to_feature_table, FEATURE_NAMES, estimate_irf_from_leading_edge,
    )
    from tcspc_toolkit.config import FeatureConfig
    from tcspc_toolkit.irf.profiles import generate_emg_irf_profile
    from tcspc_toolkit.simulation.histograms import build_expected_counts_from_irf, sample_photon_counts
    from tcspc_toolkit.estimation.bayesian.posterior import (
        BayesianPriorConfig, GammaPrior, LogNormalPrior, IRFModelRelation,
    )
    from tcspc_toolkit.estimation.bayesian.sampling import (
        BayesianSamplingConfig, fit_bayesian_monoexponential_reconvolution,
    )

    time = np.arange(0., 12., .1)
    generated = generate_emg_irf_profile(time, gaussian_centre_ns=1.,
                                       gaussian_fwhm_ns=.35, tail_time_ns=.15)
    kernel = prepare_irf(generated, time)
    expected = build_expected_counts_from_irf(monoexponential_decay(time, 1., 2., 0.), kernel,
                                            signal_photon_count=30000, background_per_bin=3.)
    counts = sample_photon_counts(expected, np.random.default_rng(120403))
    # Temporary experimental-style import, with explicit ps -> ns conversion.
    # These are synthetic fixtures, not an actual experimental ground truth claim.
    irf_path, counts_path = directory / "source.csv", directory / "measurement.csv"
    np.savetxt(irf_path, np.column_stack([time * 1000, generated.values]),
               delimiter=",", header="time,irf", comments="")
    np.savetxt(counts_path, np.column_stack([time * 1000, counts]),
               delimiter=",", header="time,counts", comments="")
    sampled = load_sampled_irf_csv(irf_path, time_unit="ps")
    measurement = load_tcspc_measurement_csv(counts_path, time_unit="ps", data_kind="raw_counts",
                                            irf=sampled, sample_id="imported-0")
    source = irf_profile_from_sampled_irf(sampled)
    prepared = prepare_irf(source, measurement.time_ns)
    np.testing.assert_allclose(measurement.time_ns, time, rtol=0., atol=2e-15)
    np.testing.assert_array_equal(measurement.require_raw_counts(), counts)
    np.testing.assert_array_equal(measurement_to_histogram_batch(
        measurement, training_time_ns=measurement.time_ns), counts[None, :])
    features = measurement_to_feature_table(
        measurement, training_time_ns=measurement.time_ns,
        feature_config=FeatureConfig(tail_start_ns=4., early_stop_ns=2., late_start_ns=5.,
                                     min_tail_points=5), training_feature_names=FEATURE_NAMES,
    )
    curve, fit, covariance, _, _, points, _ = classical_projection(measurement, prepared)
    require(points.reference_comparisons.empty, "Experimental import invented a reference")
    proxy = estimate_irf_from_leading_edge(
        measurement, background_window_ns=(0., .4), rising_edge_window_ns=(.4, 2.),
        smoothing_window_bins=5, polynomial_order=2,
    )
    require(proxy.profile is not None, f"Leading-edge proxy could not be constructed: {proxy.diagnostics}")
    require(proxy.profile.source_kind != source.source_kind, "Proxy relabelled as measured source")
    require(prepared.source is source and measurement.irf is sampled, "Source identity lost")
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(shape=2., rate=2./10000.),
        lifetime_ns=LogNormalPrior(log_mean=np.log(2.), log_std=.7),
        background_per_bin=GammaPrior(shape=2., rate=2./3.),
        temporal_shift_prior=None, fixed_temporal_shift_ns=0.,
    )
    config = BayesianSamplingConfig(random_seed=120404, n_walkers=12, warmup_steps=20,
                                    production_steps=30, max_production_steps=30,
                                    extension_steps=10, n_ensembles=2)
    sampling = {"status": "NOT RUN", "reason": "explicit --bayesian required"}
    if block_emcee or bayesian:
        try:
            run = fit_bayesian_monoexponential_reconvolution(
                measurement, prepared_irf=prepared, priors=priors, sampling_config=config,
                irf_model_relation=IRFModelRelation.UNSPECIFIED,
            )
        except ImportError as exc:
            require("optional emcee extra" in str(exc), "Unexpected Bayesian import failure")
            sampling = {"status": "OPTIONAL_DEPENDENCY_ERROR", "message": str(exc)}
        else:
            require(not block_emcee, "Blocked emcee was loaded")
            require(run.samples.physical.shape == (2, 30, 12, 4), "Tiny sampler wiring failed")
            require(np.isfinite(run.samples.physical).all(), "Nonfinite posterior samples")
            require(run.context.prepared_irf is prepared, "Bayesian assumed IRF changed")
            sampling = {"status": run.result.status.value,
                        "sample_fingerprint": fingerprint(run.samples.physical),
                        "shape": list(run.samples.physical.shape),
                        "claim": "wiring only; no convergence or calibration acceptance"}
    return {
        "counts": fingerprint(measurement.values), "features": fingerprint(features.to_numpy()),
        "source_kind": source.source_kind.value, "proxy_kind": proxy.profile.source_kind.value,
        "proxy_diagnostics": asdict(proxy.diagnostics), "kernel": fingerprint(prepared.kernel),
        "preparation": asdict(prepared.diagnostics), "lifetime_ns": fit.lifetime,
        "nll": curve.poisson_nll, "reference_count": len(points.reference_comparisons),
        "bayesian": sampling,
    }


def mismatch_workflow():
    import numpy as np
    from tcspc_toolkit import TCSPCMeasurement, MeasurementDataKind, prepare_irf
    from tcspc_toolkit.irf.profiles import generate_gaussian_irf_profile
    from tcspc_toolkit.simulation.histograms import simulate_irf_convolved_biexponential_histogram
    from tcspc_toolkit.evaluation.results import LifetimeReference
    from tcspc_toolkit.evaluation.diagnostics import calculate_poisson_deviance_residuals

    time = np.arange(0., 12., .1)
    counts, metadata = simulate_irf_convolved_biexponential_histogram(
        time=time, primary_lifetime_ns=2., secondary_lifetime_ns=4., secondary_fraction=.2,
        signal_photon_count=30000, background_per_bin=3., irf_centre_ns=.8,
        irf_fwhm_ns=.35, irf_shift_ns=0., rng=np.random.default_rng(120405),
    )
    prepared = prepare_irf(generate_gaussian_irf_profile(
        time, gaussian_centre_ns=.8, gaussian_fwhm_ns=.35), time)
    measured = TCSPCMeasurement(time, counts, MeasurementDataKind.RAW_COUNTS, sample_id="mismatch-0")
    curve, fit, _, _, _, points, _ = classical_projection(
        measured, prepared,
        reference=LifetimeReference("primary/2ns", "primary_component", [2.], [True]),
    )
    residuals = calculate_poisson_deviance_residuals(counts, fit.fitted_curve)
    require(np.isfinite(residuals).all(), "Mismatch residuals unavailable")
    require(points.reference_comparisons.reference_kind.tolist() == ["primary_component"],
            "Mismatch target silently changed")
    return {"counts": fingerprint(counts), "generating_parameters": metadata,
            "weighted_lifetime_descriptive_ns": .8 * 2. + .2 * 4.,
            "lifetime_ns": fit.lifetime, "poisson_deviance": curve.poisson_deviance,
            "residuals": fingerprint(residuals),
            "references": points.reference_comparisons.to_dict("records")}


def persistence_workflow(directory, measurement, prepared, projection, ml):
    import numpy as np
    import tcspc_toolkit.persistence.sqlite as store
    from tcspc_toolkit import TCSPCMeasurement, MeasurementDataKind
    from tcspc_toolkit.evaluation.core import build_point_evaluation
    from tcspc_toolkit.uncertainty.classical import (
        CLASSICAL_UNCERTAINTY_METHODS, POISSON_LOCAL_COVARIANCE_METHOD_ID,
        estimate_poisson_reconvolution_local_covariance, project_poisson_local_covariance_interval,
    )
    from tcspc_toolkit.uncertainty.evaluation import PredictionIntervalResult, evaluate_prediction_intervals

    curve, fit, covariance, batch, method, points, attached = projection
    # Explicit injected *failure* fixture tests propagation, not numerical failure incidence.
    failed_curve = replace(curve, valid_fit=False, optimizer_success=False,
                           failure_reason="optimizer_unsuccessful")
    failed_covariance = estimate_poisson_reconvolution_local_covariance(
        time=measurement.time_ns, irf=prepared.kernel, fit_result=replace(fit, success=False),
        temporal_shift_bounds=(-.3, .3),
    )
    failed_points = build_point_evaluation(batch=batch, method=method,
        lifetime_estimates_ns=[fit.lifetime], is_valid=[False], failure_reasons=[failed_curve.failure_reason])
    failed_attachment = project_poisson_local_covariance_interval(
        batch=batch, points=failed_points, method=method, sample_id=measurement.sample_id,
        curve=failed_curve, covariance=failed_covariance, nominal_level=.9,
    )
    require(not failed_covariance.covariance_valid and
            not failed_attachment.intervals.is_valid_interval.any(), "Failure became valid uncertainty")
    intervals = PredictionIntervalResult(
        prediction=np.array([fit.lifetime, fit.lifetime]),
        lower=np.array([attached.intervals.lower_ns.iloc[0], failed_attachment.intervals.lower_ns.iloc[0]]),
        upper=np.array([attached.intervals.upper_ns.iloc[0], failed_attachment.intervals.upper_ns.iloc[0]]),
        nominal_coverage=.9, method_id=POISSON_LOCAL_COVARIANCE_METHOD_ID,
    )
    summary = evaluate_prediction_intervals(np.array([2., 2.]), intervals)
    require(summary.n_samples == 2 and summary.n_valid_intervals == 1 and
            summary.interval_failure_rate == .5, "Attempted interval population disappeared")
    path = directory / "integration.sqlite"
    store.initialize_database(path)
    data, test, ml_points, _ = ml
    with closing(store.connect_database(path)) as db, store.transaction(db):
        run_id = store.record_run(db, run_key="integration", run_type="evaluation", status="complete",
            recorded_at_utc="2026-10-17T00:00:00Z", origin="new", package_version="0.7.0",
            configuration={"profile": "issue12-cross-feature-v1"}, base_seed=120401)
        other_run = store.record_run(db, run_key="unrelated", run_type="evaluation", status="complete",
            recorded_at_utc="2026-10-17T00:00:00Z", origin="new", package_version="0.7.0", configuration={})
        source_id = store.record_irf_source(db, prepared.source, source_key="source")
        prepared_id = store.record_prepared_irf(db, prepared, irf_source_id=source_id, preparation_key="prepared")
        condition_id = store.record_simulation_condition(db, condition_key="mono", condition_id="mono",
            generating_model="monoexponential", mono_lifetime_ns=2., signal_photon_count=30000,
            background_per_bin=3., true_temporal_shift_ns=0., generating_irf_id=prepared_id,
            origin_run_id=run_id)
        measurement_id = store.record_measurement(db, measurement, measurement_key="synthetic-0",
            source_type="synthetic", condition_pk=condition_id, origin_run_id=run_id, observation_seed=120401)
        store.link_run_measurement(db, run_id=run_id, measurement_id=measurement_id, data_role="evaluation")
        assumption = store.record_model_assumption(db, assumption_key="mono-poisson",
            assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
            background_convention="per_bin", context_completeness="complete", prepared_irf_id=prepared_id,
            temporal_shift_lower_ns=-.3, temporal_shift_upper_ns=.3)
        model = store.record_model_version(db, model_key=method.method_id, estimator_name=method.method_id,
            family=method.family, configuration={"objective": "poisson"})
        result_ids = []
        for key, source_curve, source_covariance in (
            ("fitted", curve, covariance), ("injected-failure", failed_curve, failed_covariance),
        ):
            result_id = store.record_reconvolution_fit(db, source_curve, run_id=run_id,
                measurement_id=measurement_id, model_id=model, assumption_id=assumption,
                analysis_key=key, irf_model_relation="matched", execution={"evaluation_id": batch.evaluation_id,
                    "reference_id": "generating/2ns", "reference_kind": "generating_mono"})
            result_ids.append(result_id)
            store.record_classical_covariance_uncertainty(db, source_covariance, result_id=result_id,
                method_configuration={"temporal_shift_bounds_ns": [-.3, .3]})
        store.record_prediction_intervals(db, intervals, result_ids=result_ids,
            method=CLASSICAL_UNCERTAINTY_METHODS[POISSON_LOCAL_COVARIANCE_METHOD_ID],
            method_configuration={}, interval_kind=attached.intervals.interval_kind.iloc[0])
        ml_model = store.record_model_version(db, model_key="user-linear", estimator_name="user-linear",
            family="ml", representation_id="features", configuration={"alpha": 1.},
            provenance={"training_ids": list(range(24)), "training_seed": 120402})
        for index, row in zip(test, ml_points.points.itertuples(index=False)):
            sample = TCSPCMeasurement(data.time, data.X[index], MeasurementDataKind.RAW_COUNTS,
                                     sample_id=row.sample_id)
            source_condition = store.record_simulation_condition(db, condition_key=row.sample_id,
                condition_id=row.sample_id, generating_model="monoexponential",
                mono_lifetime_ns=float(data.metadata.iloc[index].lifetime_true),
                signal_photon_count=int(round(data.metadata.iloc[index].photon_count_expected -
                    data.metadata.iloc[index].background_true * data.time.size)),
                background_per_bin=float(data.metadata.iloc[index].background_true), true_temporal_shift_ns=0.,
                generating_parameters={"total_count_target": int(data.metadata.iloc[index].photon_count_target),
                    "amplitude_true": float(data.metadata.iloc[index].amplitude_true),
                    "signal_count_storage": "nearest integer to expected signal sum"})
            mid = store.record_measurement(db, sample, measurement_key=row.sample_id, source_type="synthetic",
                                           condition_pk=source_condition, observation_seed=120402)
            store.link_run_measurement(db, run_id=run_id, measurement_id=mid, data_role="evaluation")
            store.record_scalar_prediction(db, row.lifetime_estimate_ns, run_id=run_id, measurement_id=mid,
                model_id=ml_model, source_result_type="tcspc_toolkit.evaluation.results.PointEvaluationResult",
                execution={"evaluation_id": row.evaluation_id, "sample_id": row.sample_id,
                           "reference_id": "generating", "reference_kind": "generating_mono"})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with closing(store.connect_database(path, readonly=True)) as reader:
        require(reader.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "Database corrupt")
        results = store.query_to_dataframe(store.query_results(reader, filters={"run_id": run_id},
                                                               include_context=True, include_fit_details=True))
        uncertainty = store.query_to_dataframe(store.query_uncertainty(reader, filters={"run_id": run_id}))
        measured = store.query_to_dataframe(store.query_measurements(reader, include_generating=True))
        require(len(results) == 10 and len(uncertainty) == 4, "Stored populations differ")
        require(len(store.query_results(reader, filters={"run_id": other_run}).rows) == 0, "Run filter leaked")
        require(results.is_valid.tolist() == [1, 0] + [1] * 8, "Persisted validity changed")
        np.testing.assert_array_equal(results.lifetime_estimate_ns.to_numpy(),
                                      [fit.lifetime, fit.lifetime] + ml_points.points.lifetime_estimate_ns.tolist())
        require(set(uncertainty.result_id) == set(result_ids), "Interval result linkage lost")
        require(uncertainty.is_valid.tolist() == [1, 0, 1, 0], "Invalid covariance disappeared")
        require(set(results.model_family) == {"classical", "ml"}, "Method family lost")
        require(measured.generating_mono_lifetime_ns.notna().all(), "Generating reference missing")
        require(results.estimator_name.tolist() == [method.method_id] * 2 + ["user-linear"] * 8,
                "Persisted method identity changed")
        require(results.representation_id.iloc[2:].tolist() == ["features"] * 8,
                "Persisted representation identity changed")
        require(results.failure_reason.iloc[1] == "optimizer_unsuccessful", "Failure reason lost")
        provenance = [json.loads(value) for value in results.execution_json]
        require([row["reference_id"] for row in provenance] == ["generating/2ns"] * 2 + ["generating"] * 8,
                "Explicit evaluation reference identity lost")
        require(all(row["reference_kind"] == "generating_mono" for row in provenance),
                "Generating reference changed kind")
        expected_references = [2.] + data.metadata.iloc[test].lifetime_true.tolist()
        np.testing.assert_array_equal(measured.generating_mono_lifetime_ns, expected_references)
        stored_irf = store.query_to_dataframe(store.query_prepared_irfs(reader, include_source=True))
        require(stored_irf.irf_source_id.tolist() == [source_id], "Prepared/source IRF linkage changed")
        require(results.assumed_prepared_irf_id.iloc[:2].tolist() == [prepared_id] * 2,
                "Assumed inference IRF linkage changed")
        try:
            reader.execute("DELETE FROM experiment_runs")
        except Exception as exc:
            import sqlite3
            require(isinstance(exc, sqlite3.OperationalError) and "readonly" in str(exc),
                    "Unexpected read-only failure")
        else:
            raise AssertionError("Read-only connection allowed writes")
    require(before == hashlib.sha256(path.read_bytes()).hexdigest(), "Queries changed database bytes")
    # Compare identity/scientific columns only, not measured runtime or paths.
    return {"result_ids": results.result_id.tolist(), "run_ids": results.run_id.tolist(),
            "model_ids": results.model_id.tolist(), "validity": results.is_valid.tolist(),
            "lifetimes": results.lifetime_estimate_ns.tolist(), "uncertainty_rows": len(uncertainty),
            "reference_provenance": provenance, "generating_references_ns": expected_references,
            "method_families": results.model_family.tolist(),
            "interval_summary": asdict(summary), "read_only_bytes_unchanged": True,
            "failed_covariance_reason": failed_covariance.failure_reason}


def cli_workflow(directory):
    import numpy as np
    from unittest.mock import patch
    from tcspc_toolkit import fit_monoexponential_decay
    from tcspc_toolkit.simulation.histograms import simulate_monoexponential_decay
    from tcspc_toolkit.cli import main

    time = np.linspace(0., 12., 121)
    _, counts = simulate_monoexponential_decay(time, 2000., 2., 3., random_seed=120406)
    path = directory / "cli.csv"
    np.savetxt(path, np.column_stack([time, counts]), delimiter=",", header="time_ns,measured_counts", comments="")
    api = fit_monoexponential_decay(time, counts, (1800., 1.8, 2.))
    output = io.StringIO()
    with redirect_stdout(output), patch.object(sys, "argv", [
        "tcspc", "fit", "--input", str(path), "--time-unit", "ns", "--data-kind", "raw_counts",
        "--initial-amplitude", "1800", "--initial-lifetime", "1.8", "--initial-background", "2",
    ]):
        status = main()
    require(status in (0, None) and f"{api.lifetime:.2f}" in output.getvalue(), "CLI/API fit mismatch")
    require("raw_counts" in output.getvalue(), "CLI lost data semantics")
    return {"lifetime_ns": api.lifetime, "output": output.getvalue(), "model": "unconvolved least squares"}


def run_once(*, bayesian=False, block_emcee=False):
    with tempfile.TemporaryDirectory(prefix="tcspc-issue12-integration-") as temporary:
        directory = Path(temporary)
        measured, prepared, projection, synthetic = synthetic_workflow()
        ml = ml_workflow()
        report = {"synthetic": synthetic, "ml": ml[3],
                  "experimental": experimental_workflow(directory, bayesian=bayesian, block_emcee=block_emcee),
                  "mismatch": mismatch_workflow(),
                  "persistence": persistence_workflow(directory, measured, prepared, projection, ml),
                  "cli": cli_workflow(directory)}
    require(not directory.exists(), "Temporary CSV/database cleanup failed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bayesian", action="store_true")
    group.add_argument("--block-emcee", action="store_true")
    args = parser.parse_args(argv)
    # Keep CLI INFO logging from making the second run noisier than the first;
    # this does not filter Python/scientific warnings.
    logging.basicConfig(level=logging.WARNING)
    blocker = BlockFlatFacades()
    require(not blocker.names.intersection(sys.modules), "Use a fresh process, without loaded flat facades")
    sys.meta_path.insert(0, blocker)
    if args.block_emcee:
        require("emcee" not in sys.modules, "Use a fresh process for optional-dependency isolation")
        sys.meta_path.insert(0, BlockEmcee())
    started = perf_counter()
    first = run_once(bayesian=args.bayesian, block_emcee=args.block_emcee)
    second = run_once(bayesian=args.bayesian, block_emcee=args.block_emcee)
    # JSON comparison also tolerates identical unavailable NaN diagnostics while
    # preserving exact floating values; timing is intentionally absent.
    require(json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True), "Deterministic rerun differs")
    require(not blocker.names.intersection(sys.modules), "A flat facade bypassed the blocker")
    print(json.dumps({"profile": "issue12-cross-feature-v1", "status": "PASS", "runs": 2,
                      "facades_blocked": len(blocker.names), "elapsed_seconds": perf_counter() - started,
                      "workflows": second}, indent=2))


if __name__ == "__main__":
    main()
