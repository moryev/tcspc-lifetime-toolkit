"""Classical uncertainty results remain authoritative beneath interval projections."""

from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.classical_evaluation import ReconvolutionCurveResult
from tcspc_toolkit.classical_uncertainty import (
    CLASSICAL_UNCERTAINTY_METHODS,
    PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID,
    POISSON_LOCAL_COVARIANCE_METHOD_ID,
    REPEATED_POISSON_REFERENCE_METHOD_ID,
    ParametricPoissonBootstrapResult,
    PoissonLocalCovarianceResult,
    project_parametric_poisson_bootstrap_interval,
    project_poisson_local_covariance_interval,
)
from tcspc_toolkit.evaluation_core import build_point_evaluation
from tcspc_toolkit.evaluation_results import EvaluationBatch, MethodDescriptor
from tcspc_toolkit.evaluation_uncertainty import IntervalAttachment
from tcspc_toolkit.generalization_datasets import GeneralizationTestMeasurements
from tcspc_toolkit.uncertainty_evaluation import (
    PredictionIntervalResult, UncertaintyOutputKind, evaluate_prediction_intervals,
)
from tcspc_toolkit.uncertainty_robustness import (
    _evaluate_classical_uncertainty_test, build_classical_uncertainty_scorecard,
)


def _curve(*, lifetime=2.0, valid=True):
    return ReconvolutionCurveResult(
        initial_amplitude=100.0, initial_lifetime_ns=2.0,
        initial_background=1.0, initial_temporal_shift_ns=0.0,
        fitted_amplitude=100.0, fitted_lifetime_ns=lifetime,
        fitted_background=1.0, fitted_temporal_shift_ns=0.0,
        optimizer_success=True, valid_fit=valid, boundary_hit=False,
        poisson_nll=10.0, poisson_deviance=1.0, runtime_ms=1.0,
        failure_reason=None if valid else "fit_rejected", exception_message=None,
    )


def _covariance(*, std=0.2, valid=True):
    return PoissonLocalCovarianceResult(
        covariance_matrix=np.eye(4), amplitude_std=1.0,
        lifetime_std=std, background_std=0.1, temporal_shift_std=0.01,
        covariance_valid=valid, condition_number=3.0, information_rank=4,
        boundary_hit=False, failure_reason=None if valid else "ill_conditioned",
    )


def _bootstrap(*, lifetime=2.0, lower=1.8, upper=2.2, valid=True):
    return ParametricPoissonBootstrapResult(
        source_lifetime_ns=lifetime,
        lifetime_samples_ns=np.array([1.8, 2.0, 2.2, np.nan]),
        bootstrap_std_ns=0.2, bootstrap_median_ns=2.0,
        lower_ns=lower, upper_ns=upper, nominal_coverage=0.9,
        n_resamples=4, n_successful_fits=3, n_failed_fits=1,
        n_boundary_hits=0, fit_failure_rate=0.25, boundary_hit_rate=0.0,
        bootstrap_valid=valid, failure_reason=None if valid else "insufficient_refits",
    )


def _point_facts(curve=None, *, method_id="classical_reconvolution_mono_model"):
    curve = _curve() if curve is None else curve
    batch = EvaluationBatch("external-no-reference", ["curve-7"])
    method = MethodDescriptor(method_id, "classical", "raw_histogram")
    points = build_point_evaluation(
        batch=batch, method=method,
        lifetime_estimates_ns=[curve.fitted_lifetime_ns],
        is_valid=[curve.valid_fit], failure_reasons=[curve.failure_reason],
    )
    assert points.reference_comparisons.empty
    return batch, method, points, curve


def _project_covariance(batch, method, points, curve, covariance, level=0.9):
    return project_poisson_local_covariance_interval(
        batch=batch, points=points, method=method, sample_id="curve-7",
        curve=curve, covariance=covariance, nominal_level=level,
    )


def _project_bootstrap(batch, method, points, curve, bootstrap):
    return project_parametric_poisson_bootstrap_interval(
        batch=batch, points=points, method=method, sample_id="curve-7",
        curve=curve, bootstrap=bootstrap,
    )


def test_valid_covariance_and_bootstrap_share_point_but_not_uncertainty_identity():
    batch, method, points, curve = _point_facts()
    covariance = _project_covariance(batch, method, points, curve, _covariance())
    bootstrap_result = _bootstrap()
    bootstrap = _project_bootstrap(batch, method, points, curve, bootstrap_result)
    z = NormalDist().inv_cdf(0.95)
    covariance_row = covariance.intervals.iloc[0]
    bootstrap_row = bootstrap.intervals.iloc[0]
    assert covariance_row.lower_ns == pytest.approx(2.0 - z * 0.2)
    assert covariance_row.upper_ns == pytest.approx(2.0 + z * 0.2)
    assert (bootstrap_row.lower_ns, bootstrap_row.upper_ns) == (1.8, 2.2)
    assert covariance_row.nominal_level == bootstrap_row.nominal_level == 0.9
    assert covariance_row.is_valid_interval and bootstrap_row.is_valid_interval
    assert covariance_row.method_id == bootstrap_row.method_id == method.method_id
    assert covariance_row.representation_id == bootstrap_row.representation_id == "raw_histogram"
    assert {covariance_row.uncertainty_method_id, bootstrap_row.uncertainty_method_id} == {
        POISSON_LOCAL_COVARIANCE_METHOD_ID, PARAMETRIC_POISSON_BOOTSTRAP_METHOD_ID,
    }
    assert covariance_row.interval_kind != bootstrap_row.interval_kind
    combined = IntervalAttachment(points, pd.concat([covariance.intervals, bootstrap.intervals]))
    assert len(combined.intervals) == 2
    assert "covariance_matrix" not in combined.intervals
    assert "lifetime_samples_ns" not in combined.intervals
    assert bootstrap_result.n_successful_fits == 3
    assert "n_successful_fits" not in combined.intervals


def test_source_invalidity_retains_finite_or_nonfinite_bounds_without_promoting_validity():
    batch, method, points, curve = _point_facts()
    covariance = _project_covariance(batch, method, points, curve, _covariance(valid=False))
    row = covariance.intervals.iloc[0]
    assert np.isfinite([row.lower_ns, row.upper_ns]).all()
    assert not row.is_valid_interval
    nonfinite = _project_covariance(batch, method, points, curve, _covariance(std=np.nan, valid=False))
    assert np.isnan(nonfinite.intervals.lower_ns.iloc[0])
    bootstrap = _project_bootstrap(
        batch, method, points, curve, _bootstrap(valid=False),
    )
    assert bootstrap.intervals.lower_ns.iloc[0] == 1.8
    assert not bootstrap.intervals.is_valid_interval.iloc[0]
    failed = _project_bootstrap(
        batch, method, points, curve,
        replace(
            _bootstrap(lower=np.nan, upper=np.nan, valid=False),
            lifetime_samples_ns=np.array([2.0, np.nan, np.nan, np.nan]),
            bootstrap_std_ns=np.nan, bootstrap_median_ns=np.nan,
            n_successful_fits=1, n_failed_fits=3,
            fit_failure_rate=0.75,
            failure_reason="fewer_than_two_successful_refits",
        ),
    )
    assert np.isnan(failed.intervals.upper_ns.iloc[0])
    assert not failed.intervals.is_valid_interval.iloc[0]


def test_point_fit_and_interval_validity_remain_independent():
    batch, method, points, curve = _point_facts(_curve(valid=False))
    attached = _project_covariance(batch, method, points, curve, _covariance(valid=False))
    assert not points.points.is_valid.iloc[0]
    assert points.points.lifetime_estimate_ns.iloc[0] == 2.0
    assert attached.intervals.lower_ns.iloc[0] < attached.intervals.upper_ns.iloc[0]
    assert not attached.intervals.is_valid_interval.iloc[0]


def test_adapters_reject_wrong_point_or_source_lifetime_and_family():
    batch, method, points, curve = _point_facts()
    with pytest.raises(ValueError, match="source lifetime differs"):
        _project_covariance(batch, method, points, replace(curve, fitted_lifetime_ns=2.1), _covariance())
    with pytest.raises(ValueError, match="Bootstrap source lifetime differs"):
        _project_bootstrap(batch, method, points, curve, _bootstrap(lifetime=2.1))
    with pytest.raises(ValueError, match="declared classical identity"):
        _project_covariance(
            batch, MethodDescriptor("different_irf_variant", "classical", "raw_histogram"),
            points, curve, _covariance(),
        )
    with pytest.raises(ValueError, match="explicit classical"):
        _project_covariance(
            batch, MethodDescriptor(method.method_id, "ml", "raw_histogram"),
            points, curve, _covariance(),
        )
    with pytest.raises(ValueError, match="fit decision"):
        _project_covariance(batch, method, points, replace(curve, valid_fit=False), _covariance())


def test_nominal_level_and_source_validity_are_enforced_without_references():
    batch, method, points, curve = _point_facts()
    with pytest.raises(ValueError, match="nominal_level"):
        _project_covariance(batch, method, points, curve, _covariance(), level=1.0)
    with pytest.raises(ValueError, match="finite, ordered"):
        _project_covariance(batch, method, points, curve, _covariance(std=np.nan, valid=True))
    with pytest.raises(ValueError, match="finite, ordered"):
        _project_bootstrap(batch, method, points, curve, _bootstrap(lower=np.nan, valid=True))


def test_legacy_interval_metrics_remain_reference_dependent_and_numerically_unchanged():
    batch, method, points, curve = _point_facts(method_id="explicit_classical_variant")
    attached = _project_covariance(batch, method, points, curve, _covariance())
    bounds = attached.intervals.iloc[0]
    legacy = PredictionIntervalResult(
        prediction=np.array([curve.fitted_lifetime_ns]),
        lower=np.array([bounds.lower_ns]), upper=np.array([bounds.upper_ns]),
        nominal_coverage=0.9, method_id=POISSON_LOCAL_COVARIANCE_METHOD_ID,
    )
    metrics = evaluate_prediction_intervals(np.array([2.1]), legacy)
    assert metrics.n_samples == metrics.n_valid_intervals == 1
    assert metrics.empirical_coverage == 1.0
    assert metrics.mean_interval_width == pytest.approx(bounds.upper_ns - bounds.lower_ns)
    assert points.reference_comparisons.empty
    assert attached.intervals.method_id.iloc[0] == "explicit_classical_variant"


def test_repeated_poisson_is_an_empirical_reference_not_an_attachment_method():
    definition = CLASSICAL_UNCERTAINTY_METHODS[REPEATED_POISSON_REFERENCE_METHOD_ID]
    assert definition.output_kind is UncertaintyOutputKind.EMPIRICAL_REFERENCE


def test_frozen_classical_curve_projects_sources_without_changing_legacy_scorecard(monkeypatch):
    import tcspc_toolkit.uncertainty_robustness as robustness

    curve, covariance, bootstrap = _curve(), _covariance(), _bootstrap()
    test = GeneralizationTestMeasurements(
        test_id="A", time=np.arange(5, dtype=np.float64),
        X_histograms=np.ones((1, 5), dtype=np.int64), y=np.array([2.0]),
        metadata=pd.DataFrame({
            "sample_id": ["curve-7"], "test_id": ["A"],
            "irf_fwhm_ns": [0.25], "irf_shift_ns": [0.0],
        }),
    )
    monkeypatch.setattr(robustness, "generate_gaussian_irf", lambda **kwargs: np.ones(5))
    monkeypatch.setattr(robustness, "normalize_irf", lambda **kwargs: np.ones(5) / 5)
    monkeypatch.setattr(robustness, "fit_single_reconvolution_curve", lambda **kwargs: curve)
    monkeypatch.setattr(robustness, "_reconstruct_reconvolution_fit_result", lambda **kwargs: object())
    monkeypatch.setattr(
        robustness, "estimate_poisson_reconvolution_local_covariance", lambda **kwargs: covariance,
    )
    monkeypatch.setattr(
        robustness, "estimate_parametric_poisson_bootstrap", lambda **kwargs: bootstrap,
    )
    seen = {}
    original_covariance = robustness.project_poisson_local_covariance_interval
    original_bootstrap = robustness.project_parametric_poisson_bootstrap_interval

    def capture_covariance(**kwargs):
        seen["covariance"] = original_covariance(**kwargs)
        return seen["covariance"]

    def capture_bootstrap(**kwargs):
        seen["bootstrap"] = original_bootstrap(**kwargs)
        return seen["bootstrap"]

    monkeypatch.setattr(robustness, "project_poisson_local_covariance_interval", capture_covariance)
    monkeypatch.setattr(robustness, "project_parametric_poisson_bootstrap_interval", capture_bootstrap)
    table = _evaluate_classical_uncertainty_test(
        test=test, irf_centre_ns=1.0, temporal_shift_bounds=(-0.5, 0.5),
        rng=np.random.default_rng(17), n_bootstrap_resamples=4,
        nominal_coverage=0.9, background_fraction=0.1,
    )
    row = table.iloc[0]
    covariance_row = seen["covariance"].intervals.iloc[0]
    bootstrap_row = seen["bootstrap"].intervals.iloc[0]
    assert row.covariance_lower_ns == covariance_row.lower_ns
    assert row.covariance_upper_ns == covariance_row.upper_ns
    assert row.bootstrap_lower_ns == bootstrap_row.lower_ns
    assert row.bootstrap_upper_ns == bootstrap_row.upper_ns
    assert row.bootstrap_fit_failure_rate == 0.25
    assert row.classical_fit_valid and row.covariance_valid and row.bootstrap_valid
    assert covariance_row.evaluation_id == bootstrap_row.evaluation_id == "A"
    assert covariance_row.sample_id == bootstrap_row.sample_id == "curve-7"
    assert covariance_row.method_id == bootstrap_row.method_id == "classical_reconvolution_mono_model"

    per_curve = pd.concat(
        [table.assign(test_id=test_id) for test_id in "ABCDEF"], ignore_index=True,
    )
    scorecard = build_classical_uncertainty_scorecard(per_curve, nominal_coverage=0.9)
    assert len(scorecard) == 12
    assert scorecard.n_valid_predictions.tolist() == [1] * 12
    assert scorecard.empirical_coverage.tolist() == [1.0] * 12
    assert scorecard.fit_failure_rate.tolist() == [0.0] * 12
    assert scorecard.loc[scorecard.method == "parametric_bootstrap", "mean_width_ns"].tolist() == pytest.approx([0.4] * 6)
    assert scorecard.loc[scorecard.test_id == "F", "target_reference"].tolist() == [
        "dominant_component_tau_1", "dominant_component_tau_1",
    ]
