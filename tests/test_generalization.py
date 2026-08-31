import pytest
from dataclasses import replace

from tcspc_toolkit.generalization import (
    DEVELOPMENT_ONLY_OPERATIONS,
    FINAL_ROBUSTNESS_TEST_IDS,
    DEFAULT_DEVELOPMENT_REFERENCE,
    DEFAULT_GENERALIZATION_SUITE_VERSION,
    FamiliarSimulationDomain,
    GeneralizationProtocol,
    GeneralizationRegime,
    GeneralizationShift,
    default_familiar_simulation_domain,
    default_generalization_numerics,
    default_generalization_protocol,
    default_generalization_suite,
)


def test_default_protocol_contains_tests_a_to_f() -> None:
    protocol = default_generalization_protocol()

    test_ids = tuple(
        regime.test_id
        for regime in protocol.regimes
    )

    assert test_ids == FINAL_ROBUSTNESS_TEST_IDS


def test_default_protocol_has_expected_shifts() -> None:
    protocol = default_generalization_protocol()

    expected = {
        "A": GeneralizationShift.IN_DISTRIBUTION,
        "B": GeneralizationShift.PHOTON_COUNT,
        "C": GeneralizationShift.IRF_WIDTH,
        "D": GeneralizationShift.BACKGROUND,
        "E": GeneralizationShift.TEMPORAL_ALIGNMENT,
        "F": GeneralizationShift.DECAY_MODEL,
    }

    for test_id, shift in expected.items():
        assert (
            protocol.get_regime(test_id).shift
            is shift
        )


def test_in_distribution_regime_changes_no_factors() -> None:
    protocol = default_generalization_protocol()

    regime = protocol.get_regime("A")

    assert regime.changed_factors == ()


def test_shifted_regimes_identify_changed_factors() -> None:
    protocol = default_generalization_protocol()

    for test_id in ("B", "C", "D", "E", "F"):
        regime = protocol.get_regime(test_id)

        assert regime.changed_factors


def test_protocol_marks_fitting_operations_as_development_only(
) -> None:
    protocol = default_generalization_protocol()

    assert (
        protocol.development_only_operations
        == DEVELOPMENT_ONLY_OPERATIONS
    )

    assert "scaler_fitting" in (
        protocol.development_only_operations
    )
    assert "pca_fitting" in (
        protocol.development_only_operations
    )
    assert "model_fitting" in (
        protocol.development_only_operations
    )
    assert "cross_validation" in (
        protocol.development_only_operations
    )


def test_regime_lookup_is_case_insensitive() -> None:
    protocol = default_generalization_protocol()

    assert (
        protocol.get_regime("c")
        == protocol.get_regime("C")
    )


def test_in_distribution_regime_rejects_changed_factors(
) -> None:
    with pytest.raises(
        ValueError,
        match="must not list changed factors",
    ):
        GeneralizationRegime(
            test_id="A",
            name="Invalid familiar regime",
            shift=(
                GeneralizationShift.IN_DISTRIBUTION
            ),
            description="Invalid test regime.",
            scientific_question="Why is this invalid?",
            changed_factors=(
                "signal_photon_count",
            ),
        )


def test_shifted_regime_requires_changed_factor() -> None:
    with pytest.raises(
        ValueError,
        match="must identify at least one changed factor",
    ):
        GeneralizationRegime(
            test_id="B",
            name="Invalid shifted regime",
            shift=GeneralizationShift.PHOTON_COUNT,
            description="Invalid test regime.",
            scientific_question="Why is this invalid?",
            changed_factors=(),
        )


def test_protocol_rejects_incomplete_a_to_f_suite() -> None:
    complete = default_generalization_protocol()

    with pytest.raises(
        ValueError,
        match="Tests A-F exactly once and in order",
    ):
        GeneralizationProtocol(
            regimes=complete.regimes[:-1],
            development_only_operations=(
                DEVELOPMENT_ONLY_OPERATIONS
            ),
            development_description=(
                "Development data."
            ),
            final_suite_description=(
                "Final data."
            ),
        )


def _make_familiar_domain(
) -> FamiliarSimulationDomain:
    return default_familiar_simulation_domain()


def test_default_generalization_numerics_are_frozen(
) -> None:
    familiar = _make_familiar_domain()

    numerics = default_generalization_numerics(
        familiar
    )

    assert numerics.test_b_low_photon_counts == (
        250,
        500,
    )
    assert numerics.test_b_high_photon_counts == (
        50_000,
    )

    assert numerics.test_c_irf_fwhm_values_ns == (
        0.60,
    )

    assert numerics.test_d_background_levels == (
        5.0,
    )

    assert numerics.test_e_irf_shift_values_ns == (
        -0.15,
        0.15,
    )

    assert numerics.test_f_secondary_fraction == 0.10
    assert (
        numerics.test_f_secondary_lifetime_factor
        == 2.0
    )


def test_test_b_values_are_outside_familiar_range(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    familiar_counts = (
        numerics.familiar.signal_photon_counts
    )

    assert all(
        value < min(familiar_counts)
        for value
        in numerics.test_b_low_photon_counts
    )

    assert all(
        value > max(familiar_counts)
        for value
        in numerics.test_b_high_photon_counts
    )


def test_test_c_is_broader_than_familiar_irf(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    familiar_max = max(
        numerics.familiar.irf_fwhm_values_ns
    )

    assert all(
        value > familiar_max
        for value
        in numerics.test_c_irf_fwhm_values_ns
    )


def test_test_d_is_above_familiar_background(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    familiar_max = max(
        numerics.familiar.background_levels
    )

    assert all(
        value > familiar_max
        for value
        in numerics.test_d_background_levels
    )


def test_test_e_uses_three_bin_offsets(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    assert numerics.test_e_shift_bins == (
        -3,
        3,
    )


def test_test_f_is_modest_biexponential_mismatch(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    assert (
        numerics.test_f_secondary_fraction
        == pytest.approx(0.10)
    )
    assert (
        numerics.test_f_secondary_lifetime_factor
        == pytest.approx(2.0)
    )


def test_final_suite_seeds_are_unique(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    assert len(set(numerics.test_seeds)) == 6

    assert numerics.seed_for("A") == 50_001
    assert numerics.seed_for("F") == 50_006


def test_numerics_reject_low_photon_count_inside_familiar_range(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    with pytest.raises(
        ValueError,
        match="below the familiar photon-count range",
    ):
        replace(
            numerics,
            test_b_low_photon_counts=(
                1_000,
            ),
        )


def test_numerics_reject_irf_width_inside_familiar_range(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    with pytest.raises(
        ValueError,
        match="broader than the familiar IRF range",
    ):
        replace(
            numerics,
            test_c_irf_fwhm_values_ns=(
                0.40,
            ),
        )


def test_numerics_reject_background_inside_familiar_range(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    with pytest.raises(
        ValueError,
        match="above the familiar background range",
    ):
        replace(
            numerics,
            test_d_background_levels=(
                2.0,
            ),
        )


def test_numerics_reject_temporal_shift_inside_familiar_range(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    with pytest.raises(
        ValueError,
        match="outside the familiar temporal-alignment range",
    ):
        replace(
            numerics,
            test_e_irf_shift_values_ns=(
                -0.05,
                0.15,
            ),
        )


def test_numerics_reject_strong_test_f_contamination(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    with pytest.raises(
        ValueError,
        match="interval",
    ):
        replace(
            numerics,
            test_f_secondary_fraction=0.50,
        )


def test_numerics_reject_duplicate_test_seeds(
) -> None:
    numerics = default_generalization_numerics(
        _make_familiar_domain()
    )

    with pytest.raises(
        ValueError,
        match="distinct random seeds",
    ):
        replace(
            numerics,
            test_seeds=(
                50_001,
                50_001,
                50_003,
                50_004,
                50_005,
                50_006,
            ),
        )


def test_default_familiar_domain_matches_week7_benchmark(
) -> None:
    familiar = (
        default_familiar_simulation_domain()
    )

    assert familiar.time_start_ns == pytest.approx(
        0.0
    )
    assert familiar.time_stop_ns == pytest.approx(
        20.0
    )
    assert familiar.time_step_ns == pytest.approx(
        0.05
    )

    assert familiar.n_time_bins == 400

    assert familiar.lifetime_values_ns == (
        1.0,
        2.0,
        3.0,
        4.0,
    )

    assert familiar.signal_photon_counts == (
        1_000,
        10_000,
    )

    assert familiar.background_levels == (
        0.5,
        2.0,
    )

    assert familiar.irf_centre_ns == pytest.approx(
        1.0
    )

    assert familiar.irf_fwhm_values_ns == (
        0.25,
        0.40,
    )

    assert familiar.irf_shift_values_ns == (
        -0.05,
        0.05,
    )


def test_default_suite_uses_canonical_familiar_domain(
) -> None:
    suite = default_generalization_suite()

    assert (
        suite.familiar
        == default_familiar_simulation_domain()
    )


def test_default_suite_has_expected_provenance(
) -> None:
    suite = default_generalization_suite()

    assert (
        suite.suite_version
        == DEFAULT_GENERALIZATION_SUITE_VERSION
    )

    assert (
        suite.development_reference
        == DEFAULT_DEVELOPMENT_REFERENCE
    )


def test_default_suite_contains_complete_a_to_f_protocol(
) -> None:
    suite = default_generalization_suite()

    test_ids = tuple(
        regime.test_id
        for regime in suite.protocol.regimes
    )

    assert test_ids == FINAL_ROBUSTNESS_TEST_IDS


def test_default_suite_exposes_regimes(
) -> None:
    suite = default_generalization_suite()

    assert (
        suite.regime("A").shift
        is GeneralizationShift.IN_DISTRIBUTION
    )

    assert (
        suite.regime("F").shift
        is GeneralizationShift.DECAY_MODEL
    )


def test_default_suite_exposes_reserved_seeds(
) -> None:
    suite = default_generalization_suite()

    assert suite.seed_for("A") == 50_001
    assert suite.seed_for("B") == 50_002
    assert suite.seed_for("C") == 50_003
    assert suite.seed_for("D") == 50_004
    assert suite.seed_for("E") == 50_005
    assert suite.seed_for("F") == 50_006


def test_suite_protocol_and_numerics_are_consistent(
) -> None:
    suite = default_generalization_suite()

    expected = {
        "A": (),
        "B": ("signal_photon_count",),
        "C": ("irf_fwhm_ns",),
        "D": ("background_per_bin",),
        "E": ("irf_shift_ns",),
        "F": ("decay_model",),
    }

    for test_id, changed_factors in expected.items():
        assert (
            suite.regime(test_id).changed_factors
            == changed_factors
        )


def test_suite_rejects_protocol_numeric_mismatch(
) -> None:
    suite = default_generalization_suite()

    invalid_regime = replace(
        suite.regime("C"),
        changed_factors=(
            "background_level",
        ),
    )

    invalid_regimes = tuple(
        (
            invalid_regime
            if regime.test_id == "C"
            else regime
        )
        for regime in suite.protocol.regimes
    )

    invalid_protocol = replace(
        suite.protocol,
        regimes=invalid_regimes,
    )

    with pytest.raises(
        ValueError,
        match="inconsistent with the numerical",
    ):
        replace(
            suite,
            protocol=invalid_protocol,
        )


def test_suite_regime_lookup_is_case_insensitive(
) -> None:
    suite = default_generalization_suite()

    assert suite.regime("e") == suite.regime("E")


def test_suite_seed_lookup_is_case_insensitive(
) -> None:
    suite = default_generalization_suite()

    assert suite.seed_for("f") == 50_006
