"""Small SQLite roundtrip, not a scientific benchmark or a Bayesian inference run.

From an installed checkout:
    python examples/persistence_roundtrip.py --output data/generated/roundtrip.sqlite

The classical fit and Fisher covariance are computed. All Bayesian posterior,
sampler and timing summaries are explicitly illustrative fixtures; no MCMC runs.
PPC diagnostics use three fixed fixture draws and seeded Poisson replicates.
Large arrays are not retained. An existing output is never overwritten.
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tcspc_toolkit import persistence as store
from tcspc_toolkit.bayesian import (
    BayesianModelContext, BayesianPriorConfig, BoundedUniformPrior, GammaPrior,
    IRFModelRelation, LogNormalPrior,
)
from tcspc_toolkit.bayesian_predictive import (
    BayesianPosteriorPredictiveResult,
    calculate_bayesian_posterior_predictive_diagnostics, summarize_predictive_band,
)
from tcspc_toolkit.bayesian_sampling import (
    BayesianModelContextSummary, BayesianParameterSummary, BayesianReconvolutionResult,
    BayesianRuntime, BayesianSamplingConfig, BayesianSamplingDiagnostics,
    BayesianSamplingStatus,
)
from tcspc_toolkit.classical_uncertainty import (
    DEFAULT_INFORMATION_CONDITION_LIMIT, DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP,
    estimate_poisson_reconvolution_local_covariance,
)
from tcspc_toolkit.fitting import fit_monoexponential_reconvolution
from tcspc_toolkit.forward_model import monoexponential_reconvolution_expected_counts
from tcspc_toolkit.generalization_evaluation import calculate_robustness_metrics
from tcspc_toolkit.irf import generate_gaussian_irf_profile
from tcspc_toolkit.irf_preparation import PreparedIRF, prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.simulation import sample_photon_counts

if TYPE_CHECKING:
    from pandas import DataFrame


OBSERVATION_SEED = 7301
SAMPLING_SEED = 2**64 - 1  # Fixture execution seed, not reusable model identity.
PPC_SEED = 7302
SHIFT_BOUNDS = (-0.2, 0.2)
DATASET_KEY = "roundtrip:singleton"


def _query_result_counts_by_family(
    connection: sqlite3.Connection, *, run_id: int,
) -> store.QueryResult:
    """One row per family in this run: point-result count and stored valid count.

    Called on the example's read-only connection. This counts stored is_valid
    flags, not accuracy, physical-model correctness or recomputed diagnostics.
    """
    with closing(connection.cursor()) as cursor:
        cursor.execute("""
            SELECT mv.family AS model_family,
                   COUNT(*) AS n_results,
                   SUM(CASE WHEN er.is_valid = 1 THEN 1 ELSE 0 END) AS n_valid_results
            FROM estimator_results AS er
            JOIN model_versions AS mv ON mv.model_id = er.model_id
            WHERE er.run_id = ?
            GROUP BY mv.family
            ORDER BY mv.family
        """, (run_id,))
        return store.QueryResult(
            columns=tuple(column[0] for column in cursor.description),
            rows=tuple(tuple(row) for row in cursor.fetchall()),
        )


def _bayesian_fixture(
    measurement: TCSPCMeasurement, prepared: PreparedIRF,
) -> tuple[BayesianReconvolutionResult, BayesianPosteriorPredictiveResult]:
    """Hand-constructed example summaries, NOT inference from this measurement."""
    priors = BayesianPriorConfig(
        amplitude=GammaPrior(2.0, 0.01), lifetime_ns=LogNormalPrior(np.log(2.0), 0.5),
        background_per_bin=GammaPrior(2.0, 1.0),
        temporal_shift_prior=BoundedUniformPrior(*SHIFT_BOUNDS),
        fixed_temporal_shift_ns=None,
    )
    sampler = BayesianSamplingConfig(
        random_seed=SAMPLING_SEED, n_walkers=10, n_ensembles=2, warmup_steps=8,
        production_steps=20, max_production_steps=40, extension_steps=10,
        credible_interval_level=0.9,
    )
    summaries = (
        BayesianParameterSummary("amplitude", "scale", 595.0, 590.0, 20.0, 550.0, 630.0),
        BayesianParameterSummary("lifetime_ns", "ns", 1.92, 1.9, 0.1, 1.75, 2.1),
        BayesianParameterSummary("background_per_bin", "counts_per_bin", 2.1, 2.0, 0.4, 1.4, 2.8),
        BayesianParameterSummary("temporal_shift_ns", "ns", 0.01, 0.0, 0.04, -0.08, 0.08),
    )
    context = BayesianModelContext(
        measurement, prepared_irf=prepared, irf_model_relation=IRFModelRelation.MATCHED,
    )
    result = BayesianReconvolutionResult(
        status=BayesianSamplingStatus.SUCCESS, parameter_summaries=summaries,
        inferred_parameter_names=tuple(item.name for item in summaries),
        inferred_parameter_units=tuple(item.unit for item in summaries),
        correlation_matrix=np.eye(4),
        diagnostics=BayesianSamplingDiagnostics(
            production_steps=20, retained_samples=400, extension_count=0,
            acceptance_fraction=np.full((2, 10), 0.4), mean_acceptance_fraction=0.4,
            autocorrelation_time_steps=np.full((2, 4), 2.0),
            autocorrelation_relative_change=np.full((2, 4), 0.05),
            approximate_effective_samples=np.full(4, 200.0),
            maximum_ensemble_mean_difference_sd=0.1,
            maximum_ensemble_median_difference_sd=0.1, failure_reasons=(),
        ),
        runtime=BayesianRuntime(0.0, 0.0, 0.0, 0.0),  # Fixture, not measured timing.
        priors=priors, sampling_config=sampler,
        model_context=BayesianModelContextSummary(
            measurement.sample_id, context.irf_selection, context.irf_source_kind,
            context.irf_model_relation,
        ),
    )
    draws = np.array([[580.0, 1.85, 1.8, -0.02], [590.0, 1.9, 2.0, 0.0],
                      [600.0, 1.95, 2.2, 0.02]])
    expected = np.array([
        monoexponential_reconvolution_expected_counts(measurement.time_ns, prepared.kernel, *draw)
        for draw in draws
    ])
    replicated = sample_photon_counts(expected, np.random.default_rng(PPC_SEED))
    predictive = BayesianPosteriorPredictiveResult(
        selected_posterior_indices=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        selection_with_replacement=False, retained_posterior_sample_count=400,
        physical_parameter_draws=draws, expected_counts=expected, replicated_counts=replicated,
        posterior_expected_count_band=summarize_predictive_band(expected, interval_level=0.9),
        posterior_predictive_count_band=summarize_predictive_band(replicated, interval_level=0.9),
        diagnostics=calculate_bayesian_posterior_predictive_diagnostics(
            context, expected, replicated,
            early_window_ns=(0.0, 1.5), tail_window_ns=(8.0, 11.75),
        ),
        model_context=context, priors=priors, sampling_config=sampler,
        inference_sampling_status=result.status, random_seed=PPC_SEED,
        interval_level=0.9, runtime_seconds=0.0,  # Fixture timing, too.
    )
    return result, predictive


def run_persistence_roundtrip(
    database_path: Path,
) -> tuple[dict[str, store.QueryResult], dict[str, DataFrame]]:
    """Create a NEW database; close, reopen read-only, query and convert to frames.

    The parent directory must exist. FileExistsError protects any existing file.
    Scientific payloads use fixed seeds; database/run creation timestamps record
    the actual invocation time. Returned rows/frames contain persisted facts and
    descriptive counts of those facts, without scientific recomputation.
    """
    database_path = Path(database_path)
    # Exclusive creation also guards against a concurrent creator; no unlink/overwrite.
    with database_path.open("xb"):
        pass

    time_ns = np.arange(0.0, 12.0, 0.25)
    generating_lifetime, signal_photons, background = 2.0, 5000, 2.0
    source = generate_gaussian_irf_profile(
        time_ns, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4,
        provenance={"example": "persistence_roundtrip"},
    )
    prepared = prepare_irf(source, time_ns)
    unit_signal = monoexponential_reconvolution_expected_counts(
        time_ns, prepared.kernel, 1.0, generating_lifetime, 0.0, 0.0,
    )
    # Finite-window expected signal budget is NOT the forward-model amplitude.
    generating_amplitude = signal_photons / float(unit_signal.sum())
    expected = monoexponential_reconvolution_expected_counts(
        time_ns, prepared.kernel, generating_amplitude, generating_lifetime, background, 0.0,
    )
    measurement = TCSPCMeasurement(
        time_ns, sample_photon_counts(expected, np.random.default_rng(OBSERVATION_SEED)),
        MeasurementDataKind.RAW_COUNTS, sample_id="curve-0",
        provenance={"example": "persistence_roundtrip", "synthetic": True},
    )
    fit = fit_monoexponential_reconvolution(
        time_ns, measurement.values, prepared.kernel,
        initial_guess=(generating_amplitude, 2.0, 2.0, 0.0),
        temporal_shift_bounds=SHIFT_BOUNDS, objective="poisson",
    )
    covariance = estimate_poisson_reconvolution_local_covariance(
        time=time_ns, irf=prepared.kernel, fit_result=fit,
        temporal_shift_bounds=SHIFT_BOUNDS,
        relative_step=DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP,
        condition_number_limit=DEFAULT_INFORMATION_CONDITION_LIMIT,
    )
    bayesian, predictive = _bayesian_fixture(measurement, prepared)
    metrics = calculate_robustness_metrics(
        y_true=np.array([generating_lifetime]), y_pred=np.array([fit.lifetime]),
    )

    # All scientific computation is finished before the write transaction starts.
    store.initialize_database(database_path)
    with closing(store.connect_database(database_path)) as writer, store.transaction(writer):
        run_id = store.record_run(
            writer, run_key="roundtrip:run", run_type="evaluation", status="complete",
            recorded_at_utc=datetime.now(timezone.utc).isoformat(), origin="new",
            package_version=version("tcspc-lifetime-toolkit"),
            configuration={"example": "persistence_roundtrip", "bayesian_is_fixture": True},
            profile="persistence_demo", base_seed=OBSERVATION_SEED,
        )
        source_id = store.record_irf_source(writer, source, source_key="roundtrip:gaussian")
        prepared_id = store.record_prepared_irf(
            writer, prepared, irf_source_id=source_id, preparation_key="roundtrip:prepared",
        )
        condition_id = store.record_simulation_condition(
            writer, condition_key="roundtrip:condition", condition_id="mono-2ns",
            generating_model="monoexponential", mono_lifetime_ns=generating_lifetime,
            signal_photon_count=signal_photons, background_per_bin=background,
            true_temporal_shift_ns=0.0, true_reconvolution_amplitude=generating_amplitude,
            generating_irf_id=prepared_id, origin_run_id=run_id,
        )
        measurement_id = store.record_measurement(
            writer, measurement, measurement_key="roundtrip:measurement",
            source_type="synthetic", condition_pk=condition_id, origin_run_id=run_id,
            observation_seed=OBSERVATION_SEED,
        )
        store.link_run_measurement(
            writer, run_id=run_id, measurement_id=measurement_id, data_role="evaluation",
            test_id="roundtrip", regime_id="mono-2ns", dataset_key=DATASET_KEY,
        )
        assumption_id = store.record_model_assumption(
            writer, assumption_key="roundtrip:mono-poisson",
            assumed_decay_model="monoexponential", observation_model="poisson_reconvolution",
            background_convention="per_bin", context_completeness="complete",
            prepared_irf_id=prepared_id, temporal_shift_lower_ns=SHIFT_BOUNDS[0],
            temporal_shift_upper_ns=SHIFT_BOUNDS[1],
        )
        classical_model_id = store.record_model_version(
            writer, model_key="roundtrip:classical", estimator_name="Poisson reconvolution",
            family="classical", configuration={"objective": "poisson"},
        )
        classical_id = store.record_reconvolution_fit(
            writer, fit, run_id=run_id, measurement_id=measurement_id,
            model_id=classical_model_id, assumption_id=assumption_id,
            analysis_key="small-fit", irf_model_relation="matched",
        )
        store.record_classical_covariance_uncertainty(
            writer, covariance, result_id=classical_id,
            method_configuration={
                "relative_step": DEFAULT_LOCAL_COVARIANCE_RELATIVE_STEP,
                "condition_number_limit": DEFAULT_INFORMATION_CONDITION_LIMIT,
                "temporal_shift_bounds_ns": SHIFT_BOUNDS,
            },
        )
        bayesian_model_id = store.record_model_version(
            writer, model_key="roundtrip:bayesian", estimator_name="Bayesian fixture",
            family="bayesian", prior_policy_id="roundtrip:illustrative-priors",
            configuration=store.bayesian_model_configuration(bayesian.priors, bayesian.sampling_config),
        )
        store.record_bayesian_result(
            writer, bayesian, run_id=run_id, measurement_id=measurement_id,
            model_id=bayesian_model_id, assumption_id=assumption_id,
            analysis_key="fixture-not-inference", posterior_predictive=predictive,
            execution={"illustrative_fixture": True, "mcmc_executed": False,
                       "timings_are_fixture_values": True},
        )
        # One actual source metric object -> six facts, not invented demo values.
        store.record_robustness_metrics(
            writer, metrics, scope=store.BenchmarkMetricScope(
                run_id=run_id, population_key="roundtrip:classical-singleton",
                dataset_key=DATASET_KEY, measurement_ids=[measurement_id],
                model_id=classical_model_id, assumption_id=assumption_id,
                condition_pk=condition_id, test_id="roundtrip", regime_id="mono-2ns",
                reference_kind="generating_mono",
                reference_semantics="generating_mono: known simulation lifetime",
                signal_photon_count=signal_photons, background_per_bin=background,
            ),
        )

    # The writer is committed AND CLOSED. Nothing below fits or evaluates science.
    with closing(store.connect_database(database_path, readonly=True)) as reader:
        queries = {
            "runs": store.query_runs(reader),
            "measurements": store.query_measurements(reader, include_generating=True),
            "irf_sources": store.query_irf_sources(reader),
            "prepared_irfs": store.query_prepared_irfs(reader, include_source=True),
            "results": store.query_results(reader, include_context=True, include_fit_details=True),
            "uncertainty": store.query_uncertainty(reader, include_context=True),
            "bayesian": store.query_bayesian(reader, include_context=True, decode_json=True),
            "metrics": store.query_metrics(reader, decode_json=True),
            "result_counts_by_family": _query_result_counts_by_family(reader, run_id=run_id),
        }
        frames = {name: store.query_to_dataframe(rows) for name, rows in queries.items()}

    # Structural/ownership assertions, not assertions about estimator accuracy.
    assert {name: len(rows.rows) for name, rows in queries.items()} == {
        "runs": 1, "measurements": 1, "irf_sources": 1, "prepared_irfs": 1,
        "results": 2, "uncertainty": 1, "bayesian": 1, "metrics": 6,
        "result_counts_by_family": 2,
    }
    assert queries["result_counts_by_family"].rows == (("bayesian", 1, 1), ("classical", 1, 1))
    points, observed = frames["results"], frames["measurements"].iloc[0]
    assert set(points["measurement_id"]) == {observed["measurement_id"]}
    assert points["result_id"].nunique() == 2
    assert points["fit_result_id"].notna().sum() == 1
    assert "generating_mono_lifetime_ns" in observed.index
    assert "generating_mono_lifetime_ns" not in points.columns
    classical = points.loc[points["model_family"] == "classical"].iloc[0]
    assert frames["uncertainty"].iloc[0]["result_id"] == classical["result_id"]
    posterior = frames["bayesian"].iloc[0]
    assert posterior["result_id"] == points.loc[points["model_family"] == "bayesian", "result_id"].iloc[0]
    assert posterior["ppc_status"] == "success"
    assert posterior["point_summary"] == "posterior_median"
    lifetime = next(item for item in posterior["parameter_summaries_json"]["parameters"]
                    if item["name"] == "lifetime_ns")
    assert posterior["lifetime_estimate_ns"] == lifetime["median"]
    assert posterior["lifetime_mean_ns"] == lifetime["mean"]
    assert frames["metrics"]["model_id"].eq(classical["model_id"]).all()
    assert frames["metrics"]["reference_id"].isna().all()  # No invented reference row.
    return queries, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New database in an existing directory")
    args = parser.parse_args()
    try:
        queries, frames = run_persistence_roundtrip(args.output)
    except FileExistsError:
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    print(f"Saved, closed, and reopened read-only: {args.output}")
    print("Bayesian/PPC inputs are illustrative fixtures, NOT sampled inference; no MCMC ran.")
    print("Row grains: " + ", ".join(f"{name}={len(rows.rows)}" for name, rows in queries.items()))
    columns = {
        "measurements": ["measurement_key", "data_kind", "generating_mono_lifetime_ns",
                         "generating_signal_photon_count", "observed_total_counts"],
        "results": ["result_id", "estimator_name", "point_summary", "lifetime_estimate_ns", "is_valid"],
        "uncertainty": ["result_id", "method_id", "output_kind", "reported_std_ns", "nominal_coverage"],
        "bayesian": ["result_id", "lifetime_mean_ns", "sampling_status", "diagnostics_accepted", "ppc_status"],
        "metrics": ["metric_name", "metric_value", "denominator_kind", "n_contributing"],
    }
    for name, selected in columns.items():
        print(f"\n{name}:")
        print(frames[name][selected].to_string(index=False))
    print("\nSQL aggregation - estimator results by model family (this run):")
    print(frames["result_counts_by_family"].to_string(index=False))
    print("\nOwnership assertions passed. Chains/predictive arrays are optional external artifacts; none retained.")


if __name__ == "__main__":
    main()
