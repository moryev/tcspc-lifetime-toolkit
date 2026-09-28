"""Controlled Issue-#8 evaluation; never modifies frozen Week-8 A--F tests."""

import numpy as np
import pytest

from tcspc_toolkit.config import FeatureConfig
from tcspc_toolkit.irf_evaluation import (
    IRFShapeMismatchConfig,
    LeadingEdgeFailureConfig,
    evaluate_gaussian_trained_irf_transfer,
    evaluate_irf_shape_classical,
    evaluate_irf_shape_conditional_uncertainty,
    evaluate_leading_edge_failure_regimes,
    generate_irf_shape_mismatch_dataset,
    summarize_irf_shape_classical,
    summarize_leading_edge_regimes,
)
from tcspc_toolkit.irf_preparation import _sampled_fwhm


@pytest.fixture(scope="module")
def shape_dataset():
    config = IRFShapeMismatchConfig(
        time_ns=np.arange(0.0, 12.0, 0.02),
        lifetimes_ns=(1.0, 2.0, 3.0, 4.0),
        signal_photon_counts=(200_000,),
        tail_times_ns=(0.18,),
        gaussian_component_centre_ns=1.5,
        gaussian_component_fwhm_ns=0.35,
        background_per_bin=1.0,
        development_repeats=30,
        external_repeats=2,
        random_seed=8_601,
    )
    return generate_irf_shape_mismatch_dataset(config)


@pytest.fixture(scope="module")
def classical_result(shape_dataset):
    return evaluate_irf_shape_classical(shape_dataset)


@pytest.fixture(scope="module")
def leading_result():
    config = LeadingEdgeFailureConfig(
        time_ns=np.arange(0.0, 20.0, 0.02),
        smoothing_window_bins=17,
    )
    return evaluate_leading_edge_failure_regimes(config)


def test_shape_population_pairing_and_gaussian_match(shape_dataset):
    dataset = shape_dataset
    assert len(dataset.gaussian_development.y) == 120
    assert len(dataset.emg_external.y) == len(dataset.gaussian_control.y) == 8
    np.testing.assert_array_equal(dataset.emg_external.y, dataset.gaussian_control.y)
    for column in (
        "signal_photon_count_target", "background_per_bin", "irf_peak_ns",
        "irf_full_fwhm_ns", "emg_tail_time_ns_design",
    ):
        np.testing.assert_array_equal(
            dataset.emg_external.metadata[column].to_numpy(),
            dataset.gaussian_control.metadata[column].to_numpy(),
        )
    for i, (emg, gaussian) in enumerate(zip(
        dataset.true_emg_irfs, dataset.comparison_gaussian_irfs, strict=True,
    )):
        assert dataset.emg_external.metadata.iloc[i]["pair_id"] == i
        assert dataset.gaussian_control.metadata.iloc[i]["pair_id"] == i
        emg_peak = dataset.config.time_ns[np.argmax(emg.kernel)]
        gaussian_peak = dataset.config.time_ns[np.argmax(gaussian.kernel)]
        assert emg_peak == gaussian_peak
        assert abs(
            _sampled_fwhm(dataset.config.time_ns, emg.kernel)
            - _sampled_fwhm(dataset.config.time_ns, gaussian.kernel)
        ) < 1e-9
        assert not np.allclose(emg.kernel, gaussian.kernel)


def test_shape_generation_is_seeded_and_config_grid_is_defensive(shape_dataset):
    config = shape_dataset.config
    assert not config.time_ns.flags.writeable
    repeated = generate_irf_shape_mismatch_dataset(config)
    np.testing.assert_array_equal(
        repeated.gaussian_development.X_histograms,
        shape_dataset.gaussian_development.X_histograms,
    )
    np.testing.assert_array_equal(
        repeated.gaussian_control.X_histograms,
        shape_dataset.gaussian_control.X_histograms,
    )
    np.testing.assert_array_equal(
        repeated.emg_external.X_histograms,
        shape_dataset.emg_external.X_histograms,
    )


def test_same_emg_histogram_fit_twice_and_control_kept_separate(
    shape_dataset, classical_result,
):
    dataset = shape_dataset
    result = classical_result
    assert len(result.pairs) == len(dataset.emg_external.y)
    assert len(result.per_fit) == 3 * len(result.pairs)
    assert set(result.per_fit["assumed_irf"]) == {
        "matched_emg", "gaussian_assumed", "gaussian_control",
    }
    for pair in result.pairs:
        assert pair.matched_emg.valid_fit
        assert pair.gaussian_assumed.valid_fit
        assert pair.gaussian_control.valid_fit
        assert pair.matched_emg_residuals.shape == dataset.config.time_ns.shape
        assert pair.gaussian_assumed_residuals.shape == dataset.config.time_ns.shape
        assert pair.gaussian_control_residuals.shape == dataset.config.time_ns.shape
        assert not np.allclose(
            pair.matched_emg_residuals, pair.gaussian_assumed_residuals,
        )
    grouped = result.per_fit.groupby("pair_id")
    assert all(len(group) == 3 for _, group in grouped)
    assert all(group["true_lifetime_ns"].nunique() == 1 for _, group in grouped)


def test_classical_assumptions_receive_identical_emg_counts(monkeypatch, shape_dataset):
    import tcspc_toolkit.irf_evaluation as module

    original = module.fit_experimental_reconvolution
    fitted_counts = []

    def spy(measurement, **kwargs):
        fitted_counts.append(measurement.values.copy())
        return original(measurement, **kwargs)

    monkeypatch.setattr(module, "fit_experimental_reconvolution", spy)
    module.evaluate_irf_shape_classical(shape_dataset)
    assert len(fitted_counts) == 3 * len(shape_dataset.emg_external.y)
    for index in range(0, len(fitted_counts), 3):
        np.testing.assert_array_equal(fitted_counts[index], fitted_counts[index + 1])
        assert not np.array_equal(fitted_counts[index], fitted_counts[index + 2])


def test_high_count_shape_mismatch_has_material_lifetime_effect(classical_result):
    summary = summarize_irf_shape_classical(classical_result).set_index("assumed_irf")
    assert summary.loc["gaussian_assumed", "mae_ns"] > (
        summary.loc["matched_emg", "mae_ns"] + 0.01
    )
    assert summary.loc["gaussian_assumed", "mean_poisson_deviance"] > (
        summary.loc["matched_emg", "mean_poisson_deviance"]
    )
    # This aggregate fixed-seed effect does not assert per-realization deviance ordering.


def test_classical_uncertainty_is_conditional_not_irf_coverage(
    shape_dataset, classical_result,
):
    result = evaluate_irf_shape_conditional_uncertainty(
        shape_dataset, classical_result, pair_ids=(0,),
        n_bootstrap_resamples=3, random_seed=1_119,
    )
    assert len(result.per_fit) == 2
    assert set(result.per_fit["assumed_irf"]) == {
        "matched_emg", "gaussian_assumed",
    }
    assert set(result.per_fit["uncertainty_scope"]) == {
        "conditional_on_assumed_fixed_irf",
    }
    assert result.per_fit["covariance_valid"].all()
    assert result.per_fit["bootstrap_valid"].all()
    gaussian = result.per_fit.set_index("assumed_irf").loc["gaussian_assumed"]
    assert abs(gaussian["estimated_lifetime_ns"] - gaussian["true_lifetime_ns"]) > (
        2 * gaussian["local_lifetime_std_ns"]
    )


def test_gaussian_trained_ml_transfer_population_labels_and_scores(shape_dataset):
    result = evaluate_gaussian_trained_irf_transfer(
        shape_dataset,
        feature_config=FeatureConfig(
            tail_start_ns=5.0, early_stop_ns=3.0, late_start_ns=7.0,
        ),
    )
    assert result.n_gaussian_development_samples == 120
    assert len(result.feature_names) > 0
    assert set(result.summary["population"]) == {"gaussian_control", "emg_external"}
    assert set(result.summary["model"]) == {
        "ridge", "hist_gradient_boosting", "random_forest",
    }
    rf = result.per_prediction[result.per_prediction["model"] == "random_forest"]
    assert np.all(np.isfinite(rf["rf_tree_spread_ns"]))
    assert np.all(rf["rf_tree_spread_ns"] >= 0)
    for model, group in result.per_prediction.groupby("model"):
        assert len(group) == 16, model
        assert set(group["pair_id"]) == set(range(8))


def test_ml_fits_only_gaussian_development_population(monkeypatch, shape_dataset):
    import tcspc_toolkit.irf_evaluation as module

    fit_sizes = []
    original_ridge = module.make_ridge_pipeline
    original_hgb = module.make_hist_gradient_boosting_pipeline
    original_rf = module.RandomForestTreeSpreadEstimator

    class FitSpy:
        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, X, y):
            fit_sizes.append((len(X), len(y)))
            self.estimator.fit(X, y)
            return self

        def predict(self, X):
            return self.estimator.predict(X)

    monkeypatch.setattr(module, "make_ridge_pipeline", lambda: FitSpy(original_ridge()))
    monkeypatch.setattr(
        module, "make_hist_gradient_boosting_pipeline",
        lambda **kwargs: FitSpy(original_hgb(**kwargs)),
    )
    monkeypatch.setattr(
        module, "RandomForestTreeSpreadEstimator",
        lambda **kwargs: FitSpy(original_rf(**kwargs)),
    )
    evaluate_gaussian_trained_irf_transfer(
        shape_dataset,
        feature_config=FeatureConfig(
            tail_start_ns=5.0, early_stop_ns=3.0, late_start_ns=7.0,
        ),
    )
    assert fit_sizes == [(120, 120)] * 3


def test_leading_edge_regimes_are_separate_and_favorable_proxy_is_useful(leading_result):
    cases = {case.regime: case for case in leading_result.cases}
    assert len(cases) == 10
    favorable = cases["favorable"]
    assert favorable.proxy.profile is not None
    assert favorable.proxy_l1_shape_error < 0.2
    assert abs(favorable.proxy_peak_time_error_ns) < 0.1
    assert abs(favorable.proxy_fwhm_error_ns) < 0.1
    assert favorable.true_irf_fit.valid_fit
    assert favorable.proxy_irf_fit.valid_fit
    assert favorable.monoexponential_reference_lifetime_ns == 8.0


def test_short_lifetime_can_yield_valid_but_physically_poor_proxy(leading_result):
    cases = {case.regime: case for case in leading_result.cases}
    favorable_error = cases["favorable"].proxy_l1_shape_error
    for label in ("comparable_lifetime", "very_short_lifetime"):
        adverse = cases[label]
        assert adverse.proxy.profile is not None
        assert adverse.proxy_l1_shape_error > favorable_error + 0.2
        assert adverse.proxy_irf_fit is not None


def test_low_photon_and_high_background_diagnostics(leading_result):
    cases = {case.regime: case for case in leading_result.cases}
    assert cases["low_photon_count"].proxy_l1_shape_error > (
        cases["favorable"].proxy_l1_shape_error
    )
    high = cases["high_background"]
    assert "background_at_least_half_observed_rising_counts" in high.proxy.diagnostics.flags
    assert high.proxy.diagnostics.background_estimate_counts_per_bin > 0


def test_asymmetry_biexponential_and_physical_rise_keep_target_semantics(leading_result):
    cases = {case.regime: case for case in leading_result.cases}
    asymmetric = cases["strongly_asymmetric_irf"]
    assert asymmetric.proxy.profile is not None
    assert asymmetric.proxy_l1_shape_error > cases["favorable"].proxy_l1_shape_error
    time = asymmetric.true_prepared_irf.time_ns
    baseline_irf = cases["favorable"].true_prepared_irf.kernel
    asymmetric_irf = asymmetric.true_prepared_irf.kernel
    assert abs(_sampled_fwhm(time, asymmetric_irf) - _sampled_fwhm(time, baseline_irf)) < 1e-9
    assert abs(time[np.argmax(asymmetric_irf)] - time[np.argmax(baseline_irf)]) <= 0.02
    bi = cases["bi_exponential"]
    rise = cases["finite_physical_rise"]
    assert bi.proxy.profile is not None
    assert rise.proxy.profile is not None
    assert bi.monoexponential_reference_lifetime_ns is None
    assert rise.monoexponential_reference_lifetime_ns is None
    assert "short_lifetime_ns" in bi.generating_parameters
    assert "rise_time_ns" in rise.generating_parameters
    assert rise.proxy_l1_shape_error > cases["favorable"].proxy_l1_shape_error + 0.5
    assert np.isnan(
        leading_result.per_case.set_index("regime").loc[
            "bi_exponential", "proxy_irf_lifetime_error_ns"
        ]
    )


def test_misregistration_is_explicit_and_correct_window_is_control(leading_result):
    cases = {case.regime: case for case in leading_result.cases}
    correct = cases["translated_irf_correct_window"]
    wrong = cases["misregistered_analysis_window"]
    np.testing.assert_allclose(
        correct.true_prepared_irf.kernel, wrong.true_prepared_irf.kernel,
    )
    assert correct.proxy.profile is not None
    assert wrong.proxy.profile is None
    assert wrong.proxy.diagnostics.failure_reason == (
        "smoothed_peak_at_rising_window_boundary"
    )
    assert correct.proxy.diagnostics.requested_rising_edge_window_ns != (
        wrong.proxy.diagnostics.requested_rising_edge_window_ns
    )
    summary = summarize_leading_edge_regimes(leading_result)
    assert len(summary) == 10
    assert summary.set_index("regime").loc[
        "misregistered_analysis_window", "n_proxy_constructed"
    ] == 0
