"""Sampled IRF source representation and origin tests."""

import numpy as np
import pytest

from tcspc_toolkit.irf import (
    IRFProfile,
    IRFSourceKind,
    generate_emg_irf,
    generate_emg_irf_profile,
    generate_gaussian_irf,
    generate_gaussian_irf_profile,
)
from tcspc_toolkit.irf_preparation import irf_profile_from_sampled_irf
from tcspc_toolkit.measurements import SampledIRF


def test_gaussian_profile_retains_source_scale_on_nonuniform_grid() -> None:
    time = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    values = generate_gaussian_irf(time, centre=0.3, fwhm=0.5, amplitude=3.0)
    profile = IRFProfile(time, values, IRFSourceKind.SYNTHETIC_GAUSSIAN)

    np.testing.assert_array_equal(profile.time_ns, time)
    np.testing.assert_array_equal(profile.values, values)
    assert np.trapezoid(profile.values, x=profile.time_ns) != pytest.approx(1.0)


@pytest.mark.parametrize(
    ("time", "values", "message"),
    [
        ([0.0, 0.2, 0.4], [0.0, 1.0], "must match"),
        ([0.0, 0.2, 0.4], [0.0, np.nan, 1.0], "finite"),
        ([0.0, np.inf, 0.4], [0.0, 1.0, 1.0], "finite"),
        ([0.0, 0.2, 0.4], [0.0, -1.0, 1.0], "non-negative"),
        ([0.0, 0.2, 0.4], [0.0, 0.0, 0.0], "positive integrated area"),
        ([0.0, 0.2, 0.2], [0.0, 1.0, 1.0], "strictly increasing"),
    ],
)
def test_profile_rejects_invalid_source(time, values, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IRFProfile(time, values, IRFSourceKind.SYNTHETIC_GAUSSIAN)


def test_profile_copies_and_protects_inputs() -> None:
    time = np.array([0.0, 0.2, 0.4])
    values = np.array([0.0, 2.0, 0.0])
    parameters = {"generator": {"fwhm_ns": 0.2}}
    metadata = {"instrument": {"channel": 2}}
    provenance = {"source": {"file": "irf.csv"}}
    profile = IRFProfile(
        time, values, IRFSourceKind.IMPORTED_SAMPLED,
        source_parameters=parameters, metadata=metadata, provenance=provenance,
    )
    time[0] = 9.0
    values[1] = 9.0
    parameters["generator"]["fwhm_ns"] = 9.0
    metadata["instrument"]["channel"] = 9
    provenance["source"]["file"] = "changed.csv"

    np.testing.assert_array_equal(profile.time_ns, [0.0, 0.2, 0.4])
    np.testing.assert_array_equal(profile.values, [0.0, 2.0, 0.0])
    assert profile.source_parameters["generator"]["fwhm_ns"] == 0.2
    assert profile.metadata["instrument"]["channel"] == 2
    assert profile.provenance["source"]["file"] == "irf.csv"
    assert not profile.time_ns.flags.writeable
    assert not profile.values.flags.writeable
    with pytest.raises((TypeError, ValueError)):
        profile.values[1] = 3.0
    with pytest.raises(TypeError):
        profile.metadata["new"] = 1


def test_profile_rejects_invalid_origin_or_mapping() -> None:
    with pytest.raises(ValueError, match="IRFSourceKind"):
        IRFProfile([0.0, 0.2, 0.4], [0.0, 2.0, 0.0], "measured")
    with pytest.raises(ValueError, match="source_parameters keys must be strings"):
        IRFProfile(
            [0.0, 0.2, 0.4], [0.0, 2.0, 0.0],
            IRFSourceKind.IMPORTED_SAMPLED, source_parameters={1: "invalid"},
        )


def test_sampled_irf_adapter_preserves_imported_trace() -> None:
    imported = SampledIRF(
        time_ns=[0.0, 0.2, 0.4, 0.6],
        values=[1.0, 4.0, 3.0, 0.0],
        metadata={"instrument": {"channel": 2}},
        provenance={"file": "measured-or-proxy.csv"},
    )
    original_time = imported.time_ns.copy()
    original_values = imported.values.copy()
    profile = irf_profile_from_sampled_irf(imported)

    assert profile.source_kind is IRFSourceKind.IMPORTED_SAMPLED
    np.testing.assert_array_equal(profile.time_ns, original_time)
    np.testing.assert_array_equal(profile.values, original_values)
    assert profile.metadata == imported.metadata
    assert profile.provenance == imported.provenance
    assert profile.source_parameters == {}
    assert not np.shares_memory(profile.time_ns, imported.time_ns)
    assert not np.shares_memory(profile.values, imported.values)
    assert not profile.values.flags.writeable
    np.testing.assert_array_equal(imported.time_ns, original_time)
    np.testing.assert_array_equal(imported.values, original_values)
    assert not imported.values.flags.writeable


def test_sampled_irf_adapter_requires_imported_carrier() -> None:
    with pytest.raises(TypeError, match="SampledIRF"):
        irf_profile_from_sampled_irf(object())


def test_emg_zero_tail_is_exact_existing_gaussian() -> None:
    time = np.linspace(-2.0, 5.0, 701)
    expected = generate_gaussian_irf(time, centre=1.0, fwhm=0.4, amplitude=2.5)
    actual = generate_emg_irf(
        time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4,
        tail_time_ns=0.0, amplitude=2.5,
    )
    np.testing.assert_array_equal(actual, expected)


def test_positive_emg_tail_has_positive_skew_and_preserves_source_scale() -> None:
    time = np.linspace(-5.0, 12.0, 3401)
    centre_ns = 1.0
    tail_ns = 0.6
    values = generate_emg_irf(time, centre_ns, 0.5, tail_ns, amplitude=2.0)
    area = float(np.trapezoid(values, x=time))
    mean = float(np.trapezoid(time * values, x=time) / area)
    third_moment = float(np.trapezoid((time - mean) ** 3 * values, x=time) / area)

    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert area == pytest.approx(
        2.0 * (0.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))) * np.sqrt(2.0 * np.pi),
        rel=1e-5,
    )
    assert mean == pytest.approx(centre_ns + tail_ns, abs=0.01)
    assert third_moment > 0.0
    assert time[np.argmax(values)] > centre_ns


@pytest.mark.parametrize(
    ("width", "tail"),
    [(0.0, 0.2), (-0.1, 0.2), (np.nan, 0.2), (0.5, -0.1), (0.5, np.inf)],
)
def test_emg_rejects_invalid_width_and_tail(width: float, tail: float) -> None:
    with pytest.raises(ValueError):
        generate_emg_irf(np.linspace(-1.0, 3.0, 81), 0.5, width, tail)


@pytest.mark.parametrize("tail", [0.01, 0.1, 1.0, 10.0])
def test_emg_is_finite_for_representative_tail_scales(tail: float) -> None:
    time = np.linspace(-3.0, 20.0, 2301)
    values = generate_emg_irf(time, 1.0, 0.4, tail)
    assert np.all(np.isfinite(values))
    assert np.max(values) > 0.0


def test_positive_tail_below_numerical_resolution_is_not_silently_gaussian() -> None:
    with pytest.raises(ValueError, match="below numerical resolution"):
        generate_emg_irf(np.array([0.0, 1.0]), 0.0, 1e308, 1e-300)


@pytest.mark.parametrize("tail", [5e-309, 1e-310, 1e-320])
def test_emg_rejects_nonzero_shape_whose_reciprocal_overflows(tail: float) -> None:
    width = 2.0 * np.sqrt(2.0 * np.log(2.0))  # sigma = 1 ns
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="below numerical resolution"):
            generate_emg_irf(np.array([-1.0, 0.0, 1.0]), 0.0, width, tail)


def test_emg_accepts_small_positive_tail_with_representable_reciprocal() -> None:
    width = 2.0 * np.sqrt(2.0 * np.log(2.0))
    time = np.array([-1.0, 0.0, 1.0])
    values = generate_emg_irf(time, 0.0, width, 6e-309)
    # The tail's effect is below roundoff here, but the EMG is still evaluated.
    np.testing.assert_allclose(values, np.exp(-0.5 * time**2), rtol=1e-12)
    assert np.all(values > 0.0)


def test_profile_factories_record_source_parameters_and_provenance() -> None:
    time = np.linspace(-1.0, 5.0, 121)
    metadata = {"instrument": {"run": 1}}
    provenance = {"study": "synthetic-check"}
    gaussian = generate_gaussian_irf_profile(
        time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4,
        amplitude=3.0, metadata=metadata, provenance=provenance,
    )
    emg = generate_emg_irf_profile(
        time, gaussian_centre_ns=1.0, gaussian_fwhm_ns=0.4,
        tail_time_ns=0.2, metadata=metadata, provenance=provenance,
    )
    assert gaussian.source_kind is IRFSourceKind.SYNTHETIC_GAUSSIAN
    assert gaussian.source_parameters["amplitude"] == 3.0
    assert gaussian.provenance["generator"] == "generate_gaussian_irf"
    np.testing.assert_array_equal(
        gaussian.values, generate_gaussian_irf(time, 1.0, 0.4, 3.0)
    )
    assert emg.source_kind is IRFSourceKind.SYNTHETIC_EMG
    assert emg.source_parameters["gaussian_fwhm_ns"] == 0.4
    assert emg.source_parameters["tail_time_ns"] == 0.2
    assert emg.provenance["generator"] == "generate_emg_irf"
    assert emg.provenance["study"] == "synthetic-check"
    assert emg.metadata["instrument"]["run"] == 1
    np.testing.assert_array_equal(
        emg.values, generate_emg_irf(time, 1.0, 0.4, 0.2)
    )


def test_profile_factory_rejects_conflicting_generator_provenance() -> None:
    with pytest.raises(ValueError, match="generator conflicts"):
        generate_emg_irf_profile(
            np.linspace(0.0, 5.0, 51),
            gaussian_centre_ns=1.0,
            gaussian_fwhm_ns=0.4,
            tail_time_ns=0.2,
            provenance={"generator": "another-method"},
        )
