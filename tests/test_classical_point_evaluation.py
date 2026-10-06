"""Portable classical reporting oracles, independent of the factual projection.

Controlled fit outputs lock all columns (including timings/diagnostics) without
dependency-version snapshots. Existing numerical tests exercise real fitting.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit import generalization_evaluation as evaluation
from tcspc_toolkit.classical_evaluation import ReconvolutionBenchmarkResult, ReconvolutionBenchmarkSummary
from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.evaluation_core import combine_point_evaluations
from tcspc_toolkit.generalization_datasets import generate_generalization_test_suite


@pytest.fixture(scope="module")
def tests_by_id():
    indices = [0, 1, 6, 7]  # Paired observations, both Test-F severities.
    return {
        test.test_id: replace(
            test, X_histograms=test.X_histograms[indices], y=test.y[indices],
            metadata=test.metadata.iloc[indices].set_axis([90, 30, 70, 50]),
        )
        for test in generate_generalization_test_suite().tests
    }


def diagnostic_fixture(test, mode="correct_test_irf", nominal_width=None):
    table = test.metadata.copy().reset_index(drop=True)
    delta = ("ABCDEF".index(test.test_id) + 1) / 4 + (0.125 if nominal_width is not None else 0)
    values = [test.y[0] + delta, test.y[1] + 2, np.nan, np.inf]
    table["true_lifetime_ns"] = test.y
    table["fitted_lifetime_ns"] = values
    table["valid_fit"] = [test.test_id != "B", False, False, False]
    table["failure_reason"] = [None if test.test_id != "B" else "optimizer_failed",
                               "boundary_hit", "fit_exception", "nonfinite_parameters"]
    table["optimizer_success"] = [True, True, False, False]
    table["boundary_hit"] = [False, True, False, False]
    for column in ("initial_amplitude", "initial_lifetime_ns", "initial_background",
                   "initial_temporal_shift_ns", "fitted_amplitude", "fitted_background",
                   "poisson_nll", "poisson_deviance", "runtime_ms"):
        table[column] = [1.0, 2.0, 3.0, 4.0]
    table["exception_message"] = [None, None, "controlled failure", None]
    table["classical_irf_mode"] = mode
    table["assumed_irf_fwhm_ns"] = test.metadata.irf_fwhm_ns.to_numpy() if nominal_width is None else nominal_width
    table["irf_fwhm_error_ns"] = table.assumed_irf_fwhm_ns - table.irf_fwhm_ns
    table["fitted_temporal_shift_ns"] = table.irf_shift_ns + 0.025
    table["temporal_shift_error_ns"] = table.fitted_temporal_shift_ns - table.irf_shift_ns
    return table


def expected_predictions(test, diagnostics, name, *, extra_diagnostics=True):
    """Independent pre-migration schema, order, masking and dtype policy."""
    table = test.metadata.copy().reset_index(drop=True)
    values = diagnostics.fitted_lifetime_ns.to_numpy(dtype=float)
    valid = diagnostics.valid_fit.to_numpy(dtype=bool)
    errors = np.full(test.y.size, np.nan)
    errors[valid] = values[valid] - test.y[valid]
    table["estimator"] = name
    table["representation"] = "raw_histogram"
    table["true_lifetime_ns"] = test.y
    table["predicted_lifetime_ns"] = values
    table["valid_prediction"] = valid
    table["error_ns"] = errors
    table["absolute_error_ns"] = np.abs(errors)
    if extra_diagnostics:
        for column in ("classical_irf_mode", "assumed_irf_fwhm_ns",
                       "fitted_temporal_shift_ns", "temporal_shift_error_ns"):
            table[column] = diagnostics[column].to_numpy()
    return table


def expected_summary(predictions):
    rows = []
    for (name, representation, test_id), group in predictions.groupby(
        ["estimator", "representation", "test_id"], sort=False,
    ):
        errors = group.loc[group.valid_prediction, "error_ns"].to_numpy()
        # Controlled diagnostics have exactly one accepted fit or none.
        assert len(errors) in (0, 1)
        error = errors[0] if len(errors) else np.nan
        rows.append((name, representation, test_id, len(group), len(errors),
                     abs(error), abs(error), abs(error), error, abs(error), abs(error),
                     1 - len(errors) / len(group)))
    return pd.DataFrame(rows, columns=[
        "estimator", "representation", "test_id", "n_total_samples", "n_valid_predictions",
        "mae_ns", "median_absolute_error_ns", "rmse_ns", "bias_ns",
        "p90_absolute_error_ns", "p95_absolute_error_ns", "classical_failure_rate",
    ])


@pytest.mark.parametrize("path", ["instrument", "mismatch", "full"])
def test_classical_frozen_tables_match_independent_legacy_expectations(tests_by_id, monkeypatch, path):
    calls = []
    def fit_test(**kwargs):
        calls.append((kwargs["test"].test_id, kwargs["irf_mode"], kwargs["assumed_irf_fwhm_ns"]))
        assert kwargs["irf_centre_ns"] == 1.0
        assert kwargs["temporal_shift_bounds"] == (-0.4, 0.6)
        assert kwargs["objective"] == "poisson" and kwargs["background_fraction"] == 0.15
        return diagnostic_fixture(kwargs["test"], kwargs["irf_mode"], kwargs["assumed_irf_fwhm_ns"])
    monkeypatch.setattr(evaluation, "_evaluate_classical_generalization_test", fit_test)
    functions = {
        "instrument": evaluation.evaluate_classical_instrument_acquisition_benchmark,
        "mismatch": evaluation.evaluate_classical_model_mismatch_benchmark,
        "full": evaluation.evaluate_classical_generalization_suite_benchmark,
    }
    ids = {"instrument": "ACDE", "mismatch": "AF", "full": "ABCDEF"}[path]
    result = functions[path](tests=tests_by_id, irf_centre_ns=1.0,
                             temporal_shift_bounds=(-0.4, 0.6), background_fraction=0.15)
    expected_calls = [(test_id, "correct_test_irf", None) for test_id in ids]
    if path == "instrument":
        expected_calls.append(("C", "nominal_familiar_irf", 0.4))
    assert calls == expected_calls
    tables, diagnostics = [], []
    for test_id, mode, width in expected_calls:
        test = tests_by_id[test_id]
        diagnostic = diagnostic_fixture(test, mode, width)
        name = "classical_reconvolution_mono_model"
        if path == "instrument":
            name = "classical_reconvolution_correct_irf" if width is None else "classical_reconvolution_nominal_irf"
        tables.append(expected_predictions(test, diagnostic, name))
        if path != "mismatch":
            diagnostic["estimator"] = name
        if path != "instrument":
            diagnostic["classical_decay_model"] = "monoexponential_reconvolution"
        if path == "mismatch":
            diagnostic["poisson_deviance_per_bin"] = diagnostic.poisson_deviance / test.time.size
        diagnostics.append(diagnostic)
    predictions = pd.concat(tables, ignore_index=True)
    fit_diagnostics = pd.concat(diagnostics, ignore_index=True)
    summary = expected_summary(predictions)
    pd.testing.assert_frame_equal(result.predictions, predictions, check_exact=True)
    pd.testing.assert_frame_equal(result.summary, summary, check_exact=True)
    pd.testing.assert_frame_equal(result.fit_diagnostics, fit_diagnostics, check_exact=True)
    # These scientifically specialized builders remain unchanged in this stage.
    if path == "instrument":
        expected_degradation = evaluation._build_classical_instrument_degradation_table(summary)
        pd.testing.assert_frame_equal(result.test_c_irf_comparison,
                                      evaluation._build_test_c_classical_irf_comparison(summary), check_exact=True)
    else:
        expected_degradation = evaluation.build_reference_mae_degradation_table(summary, reference_test_id="A")
    pd.testing.assert_frame_equal(result.degradation, expected_degradation, check_exact=True)
    if path == "mismatch":
        pd.testing.assert_frame_equal(result.paired_diagnostics,
                                      evaluation.build_classical_af_paired_diagnostics(fit_diagnostics), check_exact=True)
        pd.testing.assert_frame_equal(result.severity_summary,
                                      evaluation.summarize_classical_test_f_model_mismatch(fit_diagnostics), check_exact=True)
    if "F" in ids:
        np.testing.assert_array_equal(result.predictions.loc[result.predictions.test_id == "F", "true_lifetime_ns"],
                                      tests_by_id["F"].metadata.primary_lifetime_ns)


def test_principal_ab_classical_projection_keeps_original_result_objects(tests_by_id, monkeypatch):
    prepared = evaluation.prepare_generalization_ab_data(
        development_measurements=evaluation.build_generalization_development_measurements(),
        test_a=tests_by_id["A"], test_b=tests_by_id["B"],
        feature_config=FeatureConfig(tail_start_ns=2.0, early_stop_ns=2.0, late_start_ns=3.0),
    )
    fitted = evaluation.fit_generalization_ml_estimators(prepared)
    irf = np.array([0.0, 1.0])  # Passed through; the controlled fitter does not consume it.
    originals = {}
    def fit(**kwargs):
        assert kwargs["irf"] is irf
        assert kwargs["objective"] == "poisson" and kwargs["temporal_shift_bounds"] == (-0.5, 0.5)
        test_id = kwargs["metadata"].test_id.iloc[0]
        diagnostic = diagnostic_fixture(tests_by_id[test_id])
        summary = (
            ReconvolutionBenchmarkSummary(4, 1, 3, 0.25, 0.75, 0.25, 0.25, 0.25, 2.5, 2.5)
            if test_id == "A" else
            ReconvolutionBenchmarkSummary(4, 0, 4, 0.0, 1.0, np.nan, np.nan, np.nan, 2.5, 2.5)
        )
        originals[test_id] = ReconvolutionBenchmarkResult(diagnostic, summary)
        return originals[test_id]
    monkeypatch.setattr(evaluation, "evaluate_reconvolution_benchmark", fit)
    result = evaluation.evaluate_principal_ab_benchmark(prepared=prepared, fitted_estimators=fitted, classical_irf=irf)
    expected = pd.concat([
        expected_predictions(tests_by_id[test_id], diagnostic_fixture(tests_by_id[test_id]),
                             "classical_reconvolution", extra_diagnostics=False)
        for test_id in "AB"
    ], ignore_index=True)
    rows = result.predictions.loc[result.predictions.estimator == "classical_reconvolution"].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows, expected, check_exact=True)
    summary = expected_summary(expected)
    pd.testing.assert_frame_equal(result.summary.loc[result.summary.estimator == "classical_reconvolution"].reset_index(drop=True),
                                  summary, check_exact=True)
    pd.testing.assert_frame_equal(result.degradation.loc[result.degradation.estimator == "classical_reconvolution"].reset_index(drop=True),
                                  evaluation.build_mae_degradation_table(summary), check_exact=True)
    for test_id in "AB":
        assert result.classical_results[test_id] is originals[test_id]
        pd.testing.assert_frame_equal(originals[test_id].per_curve, diagnostic_fixture(tests_by_id[test_id]), check_exact=True)


@pytest.mark.parametrize("test_id", ["A", "B", "F"])
def test_classical_adapter_preserves_fit_facts_and_legacy_visibility(tests_by_id, test_id):
    test = tests_by_id[test_id]
    diagnostics = diagnostic_fixture(test)
    original = diagnostics.copy(deep=True)
    batch = evaluation._build_generalization_evaluation_batch(test=test, representations={})
    # Deliberately no classical prefix: family comes from the adapter's role.
    result = evaluation._build_classical_point_evaluation(batch=batch, diagnostics=diagnostics, method_id="physical-fit")
    points, comparisons = result.points, result.reference_comparisons
    assert points.method_family.eq("classical").all()
    assert points.representation_id.eq("raw_histogram").all()
    np.testing.assert_array_equal(points.lifetime_estimate_ns, diagnostics.fitted_lifetime_ns)
    assert points.is_valid.tolist() == diagnostics.valid_fit.tolist()
    # Pandas string columns may represent an absent reason as NaN; canonical
    # optional labels normalize absence to None without changing actual reasons.
    assert points.failure_reason.tolist() == [
        None if pd.isna(reason) else reason for reason in diagnostics.failure_reason
    ]
    assert comparisons.reference_kind.eq("primary_component" if test_id == "F" else "generating_mono").all()
    assert comparisons.error_ns.iloc[1] == diagnostics.fitted_lifetime_ns.iloc[1] - test.y[1]
    assert not comparisons.is_valid_comparison.iloc[1]
    assert comparisons.error_ns.iloc[2:].isna().all()
    assert not {"runtime_ms", "poisson_nll", "boundary_hit", "fitted_amplitude"} & set(points.columns)
    projected = evaluation._project_generalization_predictions(
        batch=batch, result=result, reference_id=next(iter(batch.references)),
    )
    pd.testing.assert_frame_equal(projected, expected_predictions(test, diagnostics, "physical-fit", extra_diagnostics=False), check_exact=True)
    summary = evaluation._summarize_generalization_predictions(projected, method_families={"physical-fit": "classical"})
    pd.testing.assert_frame_equal(summary, expected_summary(projected), check_exact=True)
    pd.testing.assert_frame_equal(diagnostics, original, check_exact=True)


def test_distinct_irf_methods_coexist_but_duplicate_identity_is_rejected(tests_by_id):
    test = tests_by_id["C"]
    batch = evaluation._build_generalization_evaluation_batch(test=test, representations={})
    correct = evaluation._build_classical_point_evaluation(
        batch=batch, diagnostics=diagnostic_fixture(test), method_id="classical_reconvolution_correct_irf",
    )
    nominal = evaluation._build_classical_point_evaluation(
        batch=batch, diagnostics=diagnostic_fixture(test, "nominal_familiar_irf", 0.4),
        method_id="classical_reconvolution_nominal_irf",
    )
    combined = combine_point_evaluations([correct, nominal])
    assert len(combined.points) == 8
    assert combined.points.method_id.drop_duplicates().tolist() == [
        "classical_reconvolution_correct_irf", "classical_reconvolution_nominal_irf",
    ]
    with pytest.raises(ValueError, match="Duplicate point identity"):
        combine_point_evaluations([correct, correct])


@pytest.mark.parametrize("misalignment", ["row_count", "sample_order"])
def test_classical_adapter_rejects_misaligned_diagnostics(tests_by_id, misalignment):
    test = tests_by_id["A"]
    diagnostics = diagnostic_fixture(test)
    diagnostics = diagnostics.iloc[:-1] if misalignment == "row_count" else diagnostics.iloc[::-1]
    batch = evaluation._build_generalization_evaluation_batch(test=test, representations={})
    with pytest.raises(ValueError, match="one row per test sample|not aligned"):
        evaluation._build_classical_point_evaluation(batch=batch, diagnostics=diagnostics, method_id="fit")


def test_real_full_suite_fits_keep_point_projection_and_diagnostics(tests_by_id):
    result = evaluation.evaluate_classical_generalization_suite_benchmark(tests=tests_by_id, irf_centre_ns=1.0)
    expected = pd.concat([
        expected_predictions(test, result.fit_diagnostics.loc[result.fit_diagnostics.test_id == test_id],
                             "classical_reconvolution_mono_model")
        for test_id, test in tests_by_id.items()
    ], ignore_index=True)
    pd.testing.assert_frame_equal(result.predictions, expected, check_exact=True)
    assert result.summary.test_id.tolist() == list("ABCDEF")
    for test_id, diagnostic in result.fit_diagnostics.groupby("test_id", sort=False):
        np.testing.assert_array_equal(diagnostic.assumed_irf_fwhm_ns, tests_by_id[test_id].metadata.irf_fwhm_ns)
    # Actual fits are not compared by runtime or dependency-specific snapshots.
    summary = evaluation.summarize_generalization_predictions(expected)
    pd.testing.assert_frame_equal(result.summary, summary, check_exact=True)
    pd.testing.assert_frame_equal(result.degradation,
                                  evaluation.build_reference_mae_degradation_table(summary, reference_test_id="A"), check_exact=True)
