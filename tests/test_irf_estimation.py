"""Numerical construction and physical limits of leading-edge IRF proxies."""

import numpy as np
import pytest
from scipy.signal import savgol_filter
from scipy.special import expit

from tcspc_toolkit.exceptions import InvalidHistogramError, InvalidMeasurementError
from tcspc_toolkit.irf import (
    IRFSourceKind,
    generate_gaussian_irf_profile,
    normalize_irf,
)
from tcspc_toolkit.irf_estimation import estimate_irf_from_leading_edge
from tcspc_toolkit.irf_preparation import prepare_irf
from tcspc_toolkit.measurements import MeasurementDataKind, TCSPCMeasurement
from tcspc_toolkit.models import monoexponential_decay
from tcspc_toolkit.preprocessing import estimate_background, subtract_background
from tcspc_toolkit.simulation import build_expected_counts_from_irf, sample_photon_counts


def _simple_measurement() -> TCSPCMeasurement:
    time = np.arange(0.0, 5.05, 0.05)
    counts = np.rint(
        10.0 + 300.0 * np.exp(-0.5 * ((time - 2.2) / 0.3) ** 2)
    ).astype(np.int64)
    return TCSPCMeasurement(
        time_ns=time,
        values=counts,
        data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id="test-histogram",
        provenance={"source": "test"},
    )


def _estimate(measurement: TCSPCMeasurement, **changes: object):
    options = {
        "background_window_ns": (0.0, 1.0),
        "rising_edge_window_ns": (1.0, 3.0),
        "smoothing_window_bins": 9,
        "polynomial_order": 3,
    }
    options.update(changes)
    return estimate_irf_from_leading_edge(measurement, **options)


def _synthetic_gaussian_measurement(
    lifetime_ns: float, seed: int = 741
) -> tuple[TCSPCMeasurement, np.ndarray]:
    time = np.arange(0.0, 20.01, 0.01)
    true_source = generate_gaussian_irf_profile(
        time, gaussian_centre_ns=2.0, gaussian_fwhm_ns=0.4
    )
    prepared = prepare_irf(true_source, time)
    true_kernel = prepared.kernel
    decay = monoexponential_decay(
        time, amplitude=1.0, lifetime=lifetime_ns, background=0.0
    )
    expected = build_expected_counts_from_irf(
        decay,
        prepared,
        signal_photon_count=5_000_000,
        background_per_bin=1.0,
    )
    counts = sample_photon_counts(expected, np.random.default_rng(seed))
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=counts,
        data_kind=MeasurementDataKind.RAW_COUNTS,
        sample_id="synthetic-fluorescence",
        provenance={"experiment": "controlled_leading_edge_test"},
    )
    return measurement, true_kernel


def _shape_error(time: np.ndarray, proxy: np.ndarray, truth: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(proxy - truth), x=time))


def test_leading_edge_requires_raw_counts() -> None:
    measurement = _simple_measurement()
    processed = TCSPCMeasurement(
        time_ns=measurement.time_ns,
        values=measurement.values.astype(float) / measurement.values.sum(),
        data_kind=MeasurementDataKind.PROCESSED_INTENSITY,
    )
    with pytest.raises(InvalidMeasurementError, match="raw photon counts"):
        _estimate(processed)


def test_nonuniform_measurement_grid_rejected_at_canonical_boundary() -> None:
    time = np.array([0.0, 0.05, 0.1, 0.17, 0.2])
    with pytest.raises((InvalidHistogramError, InvalidMeasurementError), match="uniform"):
        TCSPCMeasurement(
            time_ns=time,
            values=np.ones(time.size, dtype=np.int64),
            data_kind=MeasurementDataKind.RAW_COUNTS,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"background_window_ns": (1.0, 0.0)}, "background_window_ns"),
        ({"rising_edge_window_ns": (3.0, 1.0)}, "rising_edge_window_ns"),
        ({"background_window_ns": (0.0, 1.5)}, "precede"),
        ({"background_window_ns": (3.0, 4.0)}, "precede"),
        ({"background_window_ns": (-4.0, -3.0)}, "selects no"),
        ({"rising_edge_window_ns": (6.0, 7.0)}, "selects no"),
        ({"background_window_ns": (0.0, 0.05)}, "at least two"),
        ({"rising_edge_window_ns": (1.0, 1.2)}, "at least smoothing"),
        ({"smoothing_window_bins": 8}, "odd integer"),
        ({"smoothing_window_bins": 3, "polynomial_order": 3}, "polynomial_order"),
        ({"polynomial_order": 0}, "polynomial_order"),
    ],
)
def test_leading_edge_rejects_invalid_windows_and_smoothing(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _estimate(_simple_measurement(), **changes)


def test_favorable_leading_edge_returns_labelled_nonmutating_proxy() -> None:
    measurement, true_kernel = _synthetic_gaussian_measurement(lifetime_ns=8.0)
    original_counts = measurement.values.copy()
    original_provenance = dict(measurement.provenance)
    result = estimate_irf_from_leading_edge(
        measurement,
        background_window_ns=(0.0, 1.0),
        rising_edge_window_ns=(1.2, 3.0),
        smoothing_window_bins=31,
        polynomial_order=3,
    )
    assert result.profile is not None
    profile, diagnostics = result.profile, result.diagnostics
    assert diagnostics.status == "proxy_constructed"
    assert diagnostics.failure_reason is None
    assert profile.source_kind is IRFSourceKind.LEADING_EDGE_ESTIMATE
    assert np.all(profile.values >= 0.0)
    assert np.trapezoid(profile.values, x=profile.time_ns) == pytest.approx(1.0)
    assert not profile.time_ns.flags.writeable
    assert not profile.values.flags.writeable
    assert profile.provenance["estimator"] == "estimate_irf_from_leading_edge"
    assert profile.provenance["measurement_sample_id"] == measurement.sample_id
    assert profile.provenance["measurement_provenance"] == original_provenance
    assert profile.metadata["omitted_physical_term"] == "S(t)/tau"
    assert profile.source_parameters["smoothing_window_bins"] == 31
    assert diagnostics.requested_background_window_ns == (0.0, 1.0)
    assert diagnostics.background_bin_range == (0, 100)
    assert diagnostics.requested_rising_edge_window_ns == (1.2, 3.0)
    assert diagnostics.rising_edge_bin_range == (120, 300)
    assert diagnostics.sample_spacing_ns == pytest.approx(0.01)
    assert diagnostics.derivative_order == 1
    assert diagnostics.boundary_mode == "interp"
    assert diagnostics.final_normalized_area == pytest.approx(1.0)
    assert diagnostics.normalization_factor == pytest.approx(
        1.0 / diagnostics.proxy_area_before_normalization
    )
    assert diagnostics.proxy_fwhm_ns is not None
    assert abs(diagnostics.proxy_peak_time_ns - 2.0) < 0.08
    assert abs(diagnostics.proxy_fwhm_ns - 0.4) < 0.12
    assert _shape_error(measurement.time_ns, profile.values, true_kernel) < 0.35
    np.testing.assert_array_equal(measurement.values, original_counts)
    assert dict(measurement.provenance) == original_provenance
    prepared = prepare_irf(profile, measurement.time_ns)
    np.testing.assert_allclose(prepared.kernel, profile.values)


def test_no_rising_signal_returns_explained_failure() -> None:
    time = np.arange(0.0, 5.05, 0.05)
    measurement = TCSPCMeasurement(
        time_ns=time,
        values=np.zeros(time.size, dtype=np.int64),
        data_kind=MeasurementDataKind.RAW_COUNTS,
    )
    result = _estimate(measurement)
    assert result.profile is None
    assert result.diagnostics.status == "failed"
    assert result.diagnostics.failure_reason == "nonpositive_smoothed_signal_peak"
    assert "nonpositive_net_rising_window_signal" in result.diagnostics.flags
    assert result.diagnostics.background_estimate_counts_per_bin == 0.0


def test_peak_at_candidate_window_boundary_returns_failure() -> None:
    time = np.arange(0.0, 5.05, 0.05)
    counts = np.full(time.size, 10, dtype=np.int64)
    counts[time >= 1.0] += np.arange(np.count_nonzero(time >= 1.0), dtype=np.int64)
    measurement = TCSPCMeasurement(
        time_ns=time, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS
    )
    result = _estimate(measurement)
    assert result.profile is None
    assert result.diagnostics.failure_reason == "smoothed_peak_at_rising_window_boundary"


def test_negative_derivative_is_clipped_only_for_proxy_and_quantified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.arange(0.0, 5.05, 0.05)
    counts = np.rint(
        10.0
        + 100.0 * np.exp(-0.5 * ((time - 1.4) / 0.1) ** 2)
        + 500.0 * np.exp(-0.5 * ((time - 2.2) / 0.18) ** 2)
    ).astype(np.int64)
    counts[::2] -= 1  # Background mean is fractional; negative residuals survive.
    measurement = TCSPCMeasurement(
        time_ns=time, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS
    )
    captured_inputs: list[np.ndarray] = []

    def capture_savgol(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        captured_inputs.append(np.array(values, copy=True))
        return savgol_filter(values, *args, **kwargs)

    monkeypatch.setattr("tcspc_toolkit.irf_estimation.savgol_filter", capture_savgol)
    result = _estimate(measurement, smoothing_window_bins=7)
    assert result.profile is not None
    assert len(captured_inputs) == 2
    assert np.any(captured_inputs[0] < 0.0)
    background = estimate_background(measurement.values, 0, 20)
    working = subtract_background(measurement.values, background)
    np.testing.assert_array_equal(captured_inputs[0], working)
    derivative = savgol_filter(working, 7, 3, deriv=1, delta=0.05, mode="interp")
    start, stop = result.diagnostics.selected_rising_segment_bin_range
    positive = np.zeros_like(derivative)
    negative = np.zeros_like(derivative)
    positive[start:stop] = np.clip(derivative[start:stop], 0.0, None)
    negative[start:stop] = np.clip(-derivative[start:stop], 0.0, None)
    positive_area = float(np.trapezoid(positive, x=time))
    negative_area = float(np.trapezoid(negative, x=time))
    assert negative_area > 0.0
    assert result.diagnostics.negative_derivative_area_discarded_counts == pytest.approx(
        negative_area
    )
    assert result.diagnostics.negative_derivative_fraction == pytest.approx(
        negative_area / (positive_area + negative_area)
    )
    assert result.diagnostics.proxy_area_before_normalization == pytest.approx(positive_area)
    assert "negative_derivative_clipped_for_proxy" in result.diagnostics.flags
    assert np.all(result.profile.values[negative > 0.0] == 0.0)
    np.testing.assert_allclose(result.profile.values, normalize_irf(time, positive))


def test_separated_proxy_lobes_report_unavailable_fwhm_without_failure() -> None:
    time = np.arange(0.0, 5.05, 0.05)
    counts = np.rint(
        100.0
        + 100.0 * expit((time - 1.4) / 0.03)
        + 100.0 * expit((time - 2.2) / 0.03)
        - 250.0 * expit((time - 2.8) / 0.04)
    ).astype(np.int64)
    measurement = TCSPCMeasurement(
        time_ns=time, values=counts, data_kind=MeasurementDataKind.RAW_COUNTS
    )
    result = _estimate(measurement, smoothing_window_bins=7)
    assert result.profile is not None
    assert result.diagnostics.status == "proxy_constructed"
    assert result.diagnostics.proxy_fwhm_ns is None
    assert "proxy_fwhm_unavailable" in result.diagnostics.flags


def test_short_lifetime_has_worse_proxy_shape_without_diagnostic_causal_claim() -> None:
    long_measurement, true_kernel = _synthetic_gaussian_measurement(lifetime_ns=8.0)
    short_measurement, _ = _synthetic_gaussian_measurement(lifetime_ns=0.4)
    settings = {
        "background_window_ns": (0.0, 1.0),
        "rising_edge_window_ns": (1.2, 3.0),
        "smoothing_window_bins": 31,
    }
    long_result = estimate_irf_from_leading_edge(long_measurement, **settings)
    short_result = estimate_irf_from_leading_edge(short_measurement, **settings)
    assert long_result.profile is not None
    assert short_result.profile is not None
    long_error = _shape_error(long_measurement.time_ns, long_result.profile.values, true_kernel)
    short_error = _shape_error(short_measurement.time_ns, short_result.profile.values, true_kernel)
    assert short_error > long_error
