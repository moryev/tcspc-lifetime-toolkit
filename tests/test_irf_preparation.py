"""Explicit registration, resampling, and kernel preparation tests."""

import numpy as np
import pytest

from tcspc_toolkit.irf import IRFProfile, IRFSourceKind
from tcspc_toolkit.irf_preparation import prepare_irf


def _profile(time, values) -> IRFProfile:
    return IRFProfile(time, values, IRFSourceKind.IMPORTED_SAMPLED)


def test_compatible_grid_normalizes_derived_kernel_without_changing_source() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    source = _profile(time, values)
    prepared = prepare_irf(source, time)

    assert prepared.source is source
    np.testing.assert_array_equal(source.values, values)
    np.testing.assert_array_equal(prepared.kernel, values / 4.0)
    assert prepared.diagnostics.source_area == pytest.approx(4.0)
    assert prepared.diagnostics.retained_source_area == pytest.approx(4.0)
    assert prepared.diagnostics.support_retained_fraction == pytest.approx(1.0)
    assert prepared.diagnostics.support_loss_fraction == pytest.approx(0.0)
    assert prepared.diagnostics.target_area_before_normalization == pytest.approx(4.0)
    assert prepared.diagnostics.target_area_after_normalization == pytest.approx(1.0)
    assert prepared.diagnostics.normalization_factor == pytest.approx(0.25)
    assert prepared.diagnostics.source_fwhm_ns == pytest.approx(2.0)
    assert prepared.diagnostics.target_fwhm_ns == pytest.approx(2.0)
    assert prepared.diagnostics.target_bins_per_fwhm == pytest.approx(2.0)
    assert prepared.diagnostics.operations == ("compatible_grid_copy", "normalization")


def test_no_implicit_resampling_and_explicit_linear_interpolation() -> None:
    source = _profile([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])
    target = np.arange(0.0, 2.01, 0.5)
    with pytest.raises(ValueError, match="request resampling='linear'"):
        prepare_irf(source, target)

    prepared = prepare_irf(source, target, resampling="linear")
    expected = np.interp(target, source.time_ns, source.values)
    np.testing.assert_allclose(prepared.kernel, expected / np.trapezoid(expected, x=target))
    assert prepared.diagnostics.resampling_method == "linear"
    assert prepared.diagnostics.operations == ("linear_resampling", "normalization")
    assert prepared.diagnostics.support_loss_fraction == pytest.approx(0.0)


def test_compatible_grid_roundoff_does_not_force_resampling() -> None:
    target = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    source = _profile(target + 5e-13, [0.0, 1.0, 2.0, 1.0, 0.0])
    prepared = prepare_irf(source, target)
    assert prepared.diagnostics.resampling_method == "none"
    assert "compatible_grid_copy" in prepared.diagnostics.operations
    np.testing.assert_allclose(prepared.kernel, source.values / 0.4)


def test_large_time_origin_does_not_hide_misalignment() -> None:
    source_time = 1e6 + np.array([0.0, 0.1, 0.2, 0.3])
    source = _profile(source_time, [0.0, 1.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="grids differ"):
        prepare_irf(source, source_time + 0.01)


def test_nonuniform_source_resamples_to_uniform_target() -> None:
    source = _profile([0.0, 0.5, 1.5, 3.0], [0.0, 2.0, 1.0, 0.0])
    target = np.linspace(0.0, 3.0, 7)
    prepared = prepare_irf(source, target, resampling="linear")
    expected = np.interp(target, source.time_ns, source.values)

    np.testing.assert_allclose(prepared.kernel, expected / np.trapezoid(expected, x=target))
    assert prepared.diagnostics.source_spacing_min_ns == pytest.approx(0.5)
    assert prepared.diagnostics.source_spacing_max_ns == pytest.approx(1.5)
    assert prepared.diagnostics.target_spacing_ns == pytest.approx(0.5)
    assert np.trapezoid(prepared.kernel, x=prepared.time_ns) == pytest.approx(1.0)


def test_positive_registration_moves_peak_later_without_resampling() -> None:
    source = _profile([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 0.0])
    target = np.array([1.0, 2.0, 3.0, 4.0])
    prepared = prepare_irf(source, target, registration_offset_ns=1.0)

    assert target[np.argmax(prepared.kernel)] == 3.0
    assert prepared.diagnostics.registration_offset_ns == 1.0
    assert prepared.diagnostics.operations == (
        "registration", "compatible_grid_copy", "normalization"
    )
    assert prepared.diagnostics.support_loss_fraction == pytest.approx(0.0)


def test_fractional_registration_and_resampling_use_one_interpolation() -> None:
    source = _profile([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])
    target = np.arange(0.0, 3.01, 0.5)
    prepared = prepare_irf(
        source, target, resampling="linear", registration_offset_ns=0.25
    )
    expected = np.interp(
        target, source.time_ns + 0.25, source.values, left=0.0, right=0.0
    )

    np.testing.assert_allclose(prepared.kernel, expected / np.trapezoid(expected, x=target))
    assert prepared.diagnostics.operations == (
        "registration", "linear_resampling", "normalization"
    )
    assert prepared.diagnostics.support_loss_fraction == pytest.approx(0.0)


def test_zero_extrapolation_and_geometric_support_are_distinct_from_quadrature() -> None:
    source = _profile([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
    target = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    prepared = prepare_irf(source, target, resampling="linear")
    diagnostics = prepared.diagnostics

    assert prepared.kernel[0] == 0.0
    assert prepared.kernel[-1] == 0.0
    assert diagnostics.source_area == pytest.approx(3.0)
    assert diagnostics.retained_source_area == pytest.approx(3.0)
    assert diagnostics.support_loss_fraction == pytest.approx(0.0)
    assert diagnostics.target_area_before_normalization == pytest.approx(4.0)
    assert diagnostics.target_minus_retained_area == pytest.approx(1.0)


def test_support_loss_is_exact_piecewise_linear_overlap() -> None:
    source = _profile([0, 1, 2, 3, 4], [0, 1, 2, 1, 0])
    target = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    prepared = prepare_irf(source, target, resampling="linear")
    diagnostics = prepared.diagnostics

    assert diagnostics.source_area == pytest.approx(4.0)
    assert diagnostics.retained_source_area == pytest.approx(2.0)
    assert diagnostics.support_retained_fraction == pytest.approx(0.5)
    assert diagnostics.support_loss_fraction == pytest.approx(0.5)
    assert "support_truncated" in diagnostics.flags
    assert diagnostics.target_area_before_normalization == pytest.approx(2.0)
    assert diagnostics.target_fwhm_ns is None  # The peak is at the window boundary.


def test_support_integrates_partial_source_segments() -> None:
    source = _profile([0, 1, 2, 3, 4], [0, 1, 2, 1, 0])
    target = np.array([1.5, 2.5, 3.5, 4.5])
    diagnostics = prepare_irf(source, target, resampling="linear").diagnostics

    assert diagnostics.source_area == pytest.approx(4.0)
    assert diagnostics.retained_source_area == pytest.approx(2.875)
    assert diagnostics.support_retained_fraction == pytest.approx(2.875 / 4.0)
    assert diagnostics.target_area_before_normalization == pytest.approx(2.75)
    assert diagnostics.target_minus_retained_area == pytest.approx(-0.125)


def test_coarse_resampling_changes_area_without_geometric_support_loss() -> None:
    source = _profile([0, 1, 2, 3, 4], [0, 4, 2, 1, 0])
    target = np.array([0.0, 2.0, 4.0])
    diagnostics = prepare_irf(source, target, resampling="linear").diagnostics

    assert diagnostics.source_area == pytest.approx(7.0)
    assert diagnostics.retained_source_area == pytest.approx(7.0)
    assert diagnostics.support_loss_fraction == pytest.approx(0.0)
    assert diagnostics.target_area_before_normalization == pytest.approx(4.0)
    assert diagnostics.target_minus_retained_area == pytest.approx(-3.0)


def test_multilobed_profile_has_no_single_fwhm() -> None:
    source = _profile([0, 1, 2, 3, 4, 5, 6], [0, 2, 3, 0, 3, 2, 0])
    diagnostics = prepare_irf(source, source.time_ns).diagnostics
    assert diagnostics.source_fwhm_ns is None
    assert diagnostics.target_fwhm_ns is None
    assert diagnostics.target_bins_per_fwhm is None
    assert "source_fwhm_unavailable" in diagnostics.flags


def test_support_loss_limit_is_opt_in() -> None:
    source = _profile([0, 1, 2, 3, 4], [0, 1, 2, 1, 0])
    target = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    assert prepare_irf(source, target, resampling="linear").diagnostics.max_support_loss_fraction is None
    with pytest.raises(ValueError, match="support_loss_fraction 0.5 exceeds"):
        prepare_irf(
            source, target, resampling="linear", max_support_loss_fraction=0.4
        )
    accepted = prepare_irf(
        source, target, resampling="linear", max_support_loss_fraction=0.5
    )
    assert accepted.diagnostics.max_support_loss_fraction == 0.5


def test_prepared_arrays_are_defensive_and_read_only() -> None:
    source = _profile([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])
    target = np.array([0.0, 1.0, 2.0])
    prepared = prepare_irf(source, target)
    target[0] = -1.0

    np.testing.assert_array_equal(prepared.time_ns, [0.0, 1.0, 2.0])
    assert not prepared.time_ns.flags.writeable
    assert not prepared.kernel.flags.writeable
    with pytest.raises(ValueError):
        prepared.kernel[1] = 3.0
    np.testing.assert_array_equal(source.values, [0.0, 2.0, 0.0])
    np.testing.assert_array_equal(source.time_ns, [0.0, 1.0, 2.0])


@pytest.mark.parametrize(
    ("target", "options", "message"),
    [
        ([10.0, 11.0, 12.0], {"resampling": "linear"}, "positive finite target area"),
        ([0.0, 1.0, 2.0], {"registration_offset_ns": np.nan}, "must be finite"),
        ([0.0, 1.0, 2.0], {"resampling": "cubic"}, "resampling must"),
        ([0.0, 1.0, 2.0], {"max_support_loss_fraction": 1.1}, "between 0 and 1"),
        ([0.0, 1.1, 2.0], {"resampling": "linear"}, "approximately uniform"),
    ],
)
def test_preparation_rejects_invalid_inputs(target, options, message: str) -> None:
    source = _profile([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])
    with pytest.raises(ValueError, match=message):
        prepare_irf(source, target, **options)
