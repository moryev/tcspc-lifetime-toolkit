"""Experimental estimation, ML preparation, and reference boundaries."""

from pathlib import Path

import numpy as np
import pytest

from tcspc_toolkit.config import CountNormalization, FeatureConfig, PreprocessingConfig
from tcspc_toolkit.convolution import convolve_decay_with_irf
from tcspc_toolkit.exceptions import InvalidMeasurementError
from tcspc_toolkit.experimental import (
    evaluate_estimate_against_reference,
    fit_experimental_reconvolution,
    measurement_to_feature_table,
    measurement_to_histogram_batch,
)
from tcspc_toolkit.features import FEATURE_NAMES, extract_features
from tcspc_toolkit.irf import (
    IRFProfile,
    generate_emg_irf_profile,
    generate_gaussian_irf,
    generate_gaussian_irf_profile,
    normalize_irf,
)
from tcspc_toolkit.irf_preparation import PreparedIRF, prepare_irf
from tcspc_toolkit.measurement_io import load_sampled_irf_csv, load_tcspc_measurement_csv
from tcspc_toolkit.measurements import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
)
from tcspc_toolkit.ml_models import make_pca_histogram_ridge_pipeline
from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.preprocessing import (
    crop_time_window,
    detect_peak,
    estimate_background,
    normalize_counts,
    subtract_background,
)
from tcspc_toolkit.representations import normalize_histogram_batch
from tcspc_toolkit.simulation import (
    build_expected_counts_from_irf,
    simulate_irf_convolved_histogram,
)


def _deterministic_prepared_measurement(
    source_kind: str, *, attached_irf: SampledIRF | None = None
) -> tuple[TCSPCMeasurement, IRFProfile, PreparedIRF]:
    time = np.linspace(0.0, 16.0, 321)
    if source_kind == "gaussian":
        profile = generate_gaussian_irf_profile(
            time,
            gaussian_centre_ns=1.0,
            gaussian_fwhm_ns=0.4,
            provenance={"origin": "controlled_test"},
        )
    elif source_kind == "emg":
        profile = generate_emg_irf_profile(
            time,
            gaussian_centre_ns=1.0,
            gaussian_fwhm_ns=0.4,
            tail_time_ns=0.35,
            provenance={"origin": "controlled_test"},
        )
    else:
        raise ValueError("unsupported test IRF source")
    prepared = prepare_irf(profile, time)
    decay = monoexponential_decay(time, amplitude=1.0, lifetime=2.2, background=0.0)
    expected = build_expected_counts_from_irf(
        decay, prepared, signal_photon_count=400_000, background_per_bin=2.0
    )
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=np.rint(expected).astype(np.int64),
        data_kind=MeasurementDataKind.RAW_COUNTS,
        irf=attached_irf,
        provenance={"origin": "controlled_histogram"},
    )
    return measurement, profile, prepared


def _synthetic_experimental_style_measurement(tmp_path: Path) -> TCSPCMeasurement:
    """Generate a deterministic fixture without passing truth to the importer."""
    time = np.linspace(0.0, 20.0, 81)
    sampled_irf = generate_gaussian_irf(time, centre=1.0, fwhm=0.6, amplitude=10.0)
    normalized_irf = normalize_irf(time, sampled_irf)
    decay = monoexponential_decay(time, amplitude=1.0, lifetime=3.0, background=0.0)
    expectation = 600.0 * convolve_decay_with_irf(time, decay, normalized_irf) + 2.0
    counts = np.random.default_rng(42).poisson(expectation)
    path = tmp_path / "experimental_style.csv"
    rows = ["time_ps,counts,measured_irf"]
    rows.extend(
        f"{point * 1000:.17g},{int(count)},{irf_value:.17g}"
        for point, count, irf_value in zip(time, counts, sampled_irf, strict=True)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_tcspc_measurement_csv(
        path,
        time_unit="ps",
        data_kind="raw_counts",
        time_column="time_ps",
        irf_column="measured_irf",
    )


def test_imported_raw_curve_reconvolves_without_reference(tmp_path: Path) -> None:
    measurement = _synthetic_experimental_style_measurement(tmp_path)
    original_counts = measurement.values.copy()
    original_irf = measurement.irf.values.copy()
    result = fit_experimental_reconvolution(
        measurement, temporal_shift_bounds_ns=(-0.5, 0.5)
    )
    assert result.valid_fit
    assert abs(result.fitted_lifetime_ns - 3.0) < 0.5
    assert np.isfinite(result.poisson_nll)
    assert np.isfinite(result.poisson_deviance)
    assert result.runtime_ms >= 0.0
    assert result.failure_reason is None
    np.testing.assert_array_equal(measurement.values, original_counts)
    np.testing.assert_array_equal(measurement.irf.values, original_irf)


def test_attached_sampled_irf_still_passes_normalized_copy_to_poisson_fitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measurement = _synthetic_experimental_style_measurement(tmp_path)
    original_irf = measurement.irf.values.copy()
    sentinel = object()
    captured: dict[str, object] = {}

    def capture_fit(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "tcspc_toolkit.experimental.fit_single_reconvolution_curve", capture_fit
    )
    result = fit_experimental_reconvolution(
        measurement, temporal_shift_bounds_ns=(-0.5, 0.5)
    )
    assert result is sentinel
    np.testing.assert_array_equal(
        captured["irf"], normalize_irf(measurement.time_ns, original_irf)
    )
    assert captured["objective"] == "poisson"
    np.testing.assert_array_equal(measurement.irf.values, original_irf)


@pytest.mark.parametrize("source_kind", ["gaussian", "emg"])
def test_prepared_irf_reconvolution_recovers_controlled_lifetime(
    source_kind: str,
) -> None:
    measurement, profile, prepared = _deterministic_prepared_measurement(source_kind)
    result = fit_experimental_reconvolution(
        measurement,
        temporal_shift_bounds_ns=(-0.2, 0.2),
        prepared_irf=prepared,
    )
    assert result.valid_fit
    assert result.fitted_lifetime_ns == pytest.approx(2.2, abs=0.15)
    assert result.fitted_temporal_shift_ns == pytest.approx(0.0, abs=0.08)
    assert np.isfinite(result.poisson_deviance)
    assert prepared.source is profile


def test_explicit_prepared_irf_wins_over_different_attached_sampled_irf() -> None:
    time = np.linspace(0.0, 16.0, 321)
    attached = SampledIRF(
        time_ns=time,
        values=generate_gaussian_irf(time, centre=2.5, fwhm=0.4),
        metadata={"role": "deliberately_different"},
        provenance={"trace": "attached"},
    )
    measurement, profile, prepared = _deterministic_prepared_measurement(
        "emg", attached_irf=attached
    )
    original_counts = measurement.values.copy()
    original_time = measurement.time_ns.copy()
    original_attached_time = attached.time_ns.copy()
    original_attached = attached.values.copy()
    original_source_time = profile.time_ns.copy()
    original_source = profile.values.copy()
    original_prepared_time = prepared.time_ns.copy()
    original_kernel = prepared.kernel.copy()
    original_source_parameters = dict(profile.source_parameters)
    original_source_metadata = dict(profile.metadata)
    original_source_provenance = dict(profile.provenance)
    original_measurement_metadata = dict(measurement.metadata)
    original_measurement_provenance = dict(measurement.provenance)

    explicit = fit_experimental_reconvolution(
        measurement, temporal_shift_bounds_ns=(-0.2, 0.2), prepared_irf=prepared
    )
    attached_only = fit_experimental_reconvolution(
        measurement, temporal_shift_bounds_ns=(-0.2, 0.2)
    )
    no_attached = TCSPCMeasurement(
        time_ns=measurement.time_ns,
        values=measurement.values,
        data_kind=MeasurementDataKind.RAW_COUNTS,
    )
    prepared_only = fit_experimental_reconvolution(
        no_attached, temporal_shift_bounds_ns=(-0.2, 0.2), prepared_irf=prepared
    )
    assert explicit.valid_fit
    assert explicit.fitted_lifetime_ns == pytest.approx(2.2, abs=0.15)
    assert explicit.poisson_deviance == pytest.approx(prepared_only.poisson_deviance)
    assert explicit.fitted_lifetime_ns == pytest.approx(prepared_only.fitted_lifetime_ns)
    assert explicit.poisson_deviance < attached_only.poisson_deviance
    np.testing.assert_array_equal(measurement.values, original_counts)
    np.testing.assert_array_equal(measurement.time_ns, original_time)
    np.testing.assert_array_equal(attached.time_ns, original_attached_time)
    np.testing.assert_array_equal(attached.values, original_attached)
    np.testing.assert_array_equal(profile.time_ns, original_source_time)
    np.testing.assert_array_equal(profile.values, original_source)
    np.testing.assert_array_equal(prepared.time_ns, original_prepared_time)
    np.testing.assert_array_equal(prepared.kernel, original_kernel)
    assert dict(profile.source_parameters) == original_source_parameters
    assert dict(profile.metadata) == original_source_metadata
    assert dict(profile.provenance) == original_source_provenance
    assert dict(measurement.metadata) == original_measurement_metadata
    assert dict(measurement.provenance) == original_measurement_provenance
    assert dict(attached.metadata) == {"role": "deliberately_different"}
    assert dict(attached.provenance) == {"trace": "attached"}


def test_prepared_irf_grid_mismatch_requires_explicit_preparation() -> None:
    measurement, profile, _ = _deterministic_prepared_measurement("gaussian")
    mismatched = prepare_irf(
        profile, measurement.time_ns + 0.01, resampling="linear"
    )
    with pytest.raises(InvalidMeasurementError, match="prepare the IRF explicitly"):
        fit_experimental_reconvolution(
            measurement,
            temporal_shift_bounds_ns=(-0.2, 0.2),
            prepared_irf=mismatched,
        )


def test_experimental_reconvolution_rejects_non_prepared_irf() -> None:
    measurement, _, _ = _deterministic_prepared_measurement("gaussian")
    with pytest.raises(InvalidMeasurementError, match="must be a PreparedIRF"):
        fit_experimental_reconvolution(
            measurement,
            temporal_shift_bounds_ns=(-0.2, 0.2),
            prepared_irf=object(),
        )


def test_processed_intensity_rejected_even_with_prepared_irf() -> None:
    raw_measurement, _, prepared = _deterministic_prepared_measurement("gaussian")
    processed = TCSPCMeasurement(
        time_ns=raw_measurement.time_ns,
        values=raw_measurement.values.astype(float) / raw_measurement.values.sum(),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    with pytest.raises(InvalidMeasurementError, match="raw photon counts"):
        fit_experimental_reconvolution(
            processed,
            temporal_shift_bounds_ns=(-0.2, 0.2),
            prepared_irf=prepared,
        )


def test_shift_bounds_excluding_zero_fail_before_low_level_fitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measurement = _synthetic_experimental_style_measurement(tmp_path)

    def unexpected_fit(**kwargs: object) -> None:
        raise AssertionError("low-level fitter must not be called")

    monkeypatch.setattr(
        "tcspc_toolkit.experimental.fit_single_reconvolution_curve", unexpected_fit
    )
    with pytest.raises(InvalidMeasurementError, match="must include 0.0 ns"):
        fit_experimental_reconvolution(
            measurement, temporal_shift_bounds_ns=(0.1, 0.5)
        )


def test_nonzero_shift_fits_when_bounds_also_include_zero() -> None:
    time = np.linspace(0.0, 20.0, 81)
    measured_irf = generate_gaussian_irf(time, centre=1.0, fwhm=0.6)
    counts, _ = simulate_irf_convolved_histogram(
        time=time,
        lifetime_ns=3.0,
        signal_photon_count=60_000,
        background_per_bin=2.0,
        irf_centre_ns=1.0,
        irf_fwhm_ns=0.6,
        irf_shift_ns=0.3,
        rng=np.random.default_rng(44),
    )
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=counts,
        data_kind=MeasurementDataKind.RAW_COUNTS,
        irf=SampledIRF(time_ns=time, values=measured_irf),
    )
    result = fit_experimental_reconvolution(
        measurement, temporal_shift_bounds_ns=(-0.5, 0.5)
    )
    assert result.valid_fit
    assert result.fitted_temporal_shift_ns == pytest.approx(0.3, abs=0.12)


def test_committed_experimental_style_csv_workflow() -> None:
    examples = Path(__file__).resolve().parents[1] / "data" / "examples"
    irf = load_sampled_irf_csv(
        examples / "experimental_style_irf.csv",
        time_unit="ns",
        time_column="time_ns",
    )
    measurement = load_tcspc_measurement_csv(
        examples / "experimental_style_decay.csv",
        time_unit="ps",
        data_kind="raw_counts",
        time_column="time_ps",
        irf=irf,
        sample_id="synthetic-import-demo",
    )
    result = fit_experimental_reconvolution(
        measurement, temporal_shift_bounds_ns=(-0.5, 0.5)
    )
    assert result.valid_fit
    assert abs(result.fitted_lifetime_ns - 3.0) < 0.5


def test_imported_curve_reuses_preprocessing_without_changing_raw_source(
    tmp_path: Path,
) -> None:
    measurement = _synthetic_experimental_style_measurement(tmp_path)
    original = measurement.values.copy()
    config = PreprocessingConfig(
        background_start_bin=70,
        background_stop_bin=81,
        crop_start_ns=0.0,
        crop_stop_ns=10.0,
        normalization=CountNormalization.TOTAL,
    )
    background = estimate_background(
        measurement.require_raw_counts(),
        config.background_start_bin,
        config.background_stop_bin,
    )
    processed = subtract_background(measurement.values, background)
    peak_bin = detect_peak(measurement.values)
    cropped_time, cropped_counts = crop_time_window(
        measurement.time_ns,
        measurement.values,
        config.crop_start_ns,
        config.crop_stop_ns,
    )
    normalized = normalize_counts(cropped_counts, config.normalization)
    assert peak_bin >= 0
    assert processed.dtype == np.dtype(np.float64)
    assert cropped_time.size == cropped_counts.size
    np.testing.assert_allclose(normalized.sum(), 1.0)
    np.testing.assert_array_equal(measurement.values, original)


def test_processed_measurement_cannot_use_poisson_reconvolution() -> None:
    time = np.arange(5, dtype=float)
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=[1.0, 2.0, 5.0, 2.0, 1.0],
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
        irf=SampledIRF(time_ns=time, values=[0, 1, 3, 1, 0]),
    )
    with pytest.raises(InvalidMeasurementError, match="raw photon counts"):
        fit_experimental_reconvolution(
            measurement, temporal_shift_bounds_ns=(-0.5, 0.5)
        )


def test_reconvolution_requires_measured_irf() -> None:
    measurement = TCSPCMeasurement(
        time_ns=np.arange(5, dtype=float),
        values=[1, 2, 5, 2, 1],
        data_kind=MeasurementDataKind.RAW_COUNTS,
    )
    with pytest.raises(InvalidMeasurementError, match="measured IRF"):
        fit_experimental_reconvolution(
            measurement, temporal_shift_bounds_ns=(-0.5, 0.5)
        )


def test_ml_inputs_reuse_feature_and_representation_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measurement = _synthetic_experimental_style_measurement(tmp_path)
    config = FeatureConfig(
        tail_start_ns=6.0, early_stop_ns=2.0, late_start_ns=6.0
    )
    features = measurement_to_feature_table(
        measurement,
        training_time_ns=measurement.time_ns + 5e-13,
        feature_config=config,
        training_feature_names=FEATURE_NAMES,
    )
    expected = extract_features(measurement.time_ns, measurement.values, config)
    np.testing.assert_allclose(features.to_numpy(), expected.to_numpy())
    assert tuple(features.columns) == FEATURE_NAMES

    # Development curves are generated independently of the imported measurement.
    development_time_ns = np.linspace(0.0, 20.0, 81)
    development_lifetimes = np.linspace(1.5, 5.0, 12)
    rng = np.random.default_rng(78)
    development_counts = np.vstack(
        [
            simulate_irf_convolved_histogram(
                time=development_time_ns,
                lifetime_ns=float(lifetime),
                signal_photon_count=20_000,
                background_per_bin=2.0,
                irf_centre_ns=1.0,
                irf_fwhm_ns=0.6,
                irf_shift_ns=0.0,
                rng=rng,
            )[0]
            for lifetime in development_lifetimes
        ]
    )
    model = make_pca_histogram_ridge_pipeline(n_components=2)
    model.fit(development_counts, development_lifetimes)
    normalizer = model.named_steps["normalize"]
    pca = model.named_steps["pca"]
    scaler = model.named_steps["scaler"]
    regressor = model.named_steps["model"]
    original_components = pca.components_.copy()
    original_pca_mean = pca.mean_.copy()
    original_explained_variance = pca.explained_variance_.copy()
    original_mean = scaler.mean_.copy()
    original_scale = scaler.scale_.copy()
    original_variance = scaler.var_.copy()
    original_coefficients = regressor.coef_.copy()
    original_intercept = float(regressor.intercept_)

    batch = measurement_to_histogram_batch(
        measurement, training_time_ns=development_time_ns
    )
    normalized = normalize_histogram_batch(batch, mode=CountNormalization.TOTAL)
    np.testing.assert_allclose(normalized.sum(axis=1), [1.0])

    def forbid_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("experimental inference must only transform and predict")

    for fitted_step in (model, normalizer, pca, scaler, regressor):
        monkeypatch.setattr(fitted_step, "fit", forbid_fit)
    for fitted_step in (normalizer, pca, scaler):
        monkeypatch.setattr(fitted_step, "fit_transform", forbid_fit)

    prediction = model.predict(batch)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
    np.testing.assert_array_equal(pca.components_, original_components)
    np.testing.assert_array_equal(pca.mean_, original_pca_mean)
    np.testing.assert_array_equal(pca.explained_variance_, original_explained_variance)
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    np.testing.assert_array_equal(scaler.scale_, original_scale)
    np.testing.assert_array_equal(scaler.var_, original_variance)
    np.testing.assert_array_equal(regressor.coef_, original_coefficients)
    assert regressor.intercept_ == original_intercept


def test_ml_compatibility_rejects_grid_and_feature_mismatch(tmp_path: Path) -> None:
    measurement = _synthetic_experimental_style_measurement(tmp_path)
    config = FeatureConfig(tail_start_ns=6.0, early_stop_ns=2.0, late_start_ns=6.0)
    with pytest.raises(InvalidMeasurementError, match="training grid"):
        measurement_to_histogram_batch(
            measurement, training_time_ns=measurement.time_ns + 0.01
        )
    with pytest.raises(InvalidMeasurementError, match="feature names"):
        measurement_to_feature_table(
            measurement,
            training_time_ns=measurement.time_ns,
            feature_config=config,
            training_feature_names=tuple(reversed(FEATURE_NAMES)),
        )


def test_reference_evaluation_is_explicit_and_signed() -> None:
    result = evaluate_estimate_against_reference(
        2.8,
        reference_lifetime_ns=3.0,
        reference_metadata={"certificate": "reference-1"},
    )
    assert result.signed_error_ns == pytest.approx(-0.2)
    assert result.absolute_error_ns == pytest.approx(0.2)
    assert result.relative_error == pytest.approx(0.2 / 3.0)
    assert result.reference_metadata["certificate"] == "reference-1"
    with pytest.raises(InvalidMeasurementError, match="reference_lifetime_ns"):
        evaluate_estimate_against_reference(2.8, reference_lifetime_ns=None)
