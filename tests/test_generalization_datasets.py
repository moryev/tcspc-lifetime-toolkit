import numpy as np
import pandas as pd
import pytest

from tcspc_toolkit.generalization import (
    default_generalization_suite,
)
from tcspc_toolkit.generalization_datasets import (
    PAIR_VARIANTS_PER_CONDITION,
    build_generalization_suite_manifest,
    build_generalization_time_axis,
    build_paired_parameter_design,
    build_test_f_parameter_table,
    build_test_parameter_table,
    generate_generalization_test,
    generate_generalization_test_suite,
)


def test_generalization_time_axis_matches_familiar_domain(
) -> None:
    definition = default_generalization_suite()

    time = build_generalization_time_axis(
        definition
    )

    assert time.shape == (400,)
    assert time[0] == pytest.approx(0.0)
    assert time[-1] == pytest.approx(19.95)


def test_paired_parameter_design_has_expected_size(
) -> None:
    definition = default_generalization_suite()

    design = build_paired_parameter_design(
        definition
    )

    expected_size = (
        len(definition.familiar.lifetime_values_ns)
        * len(definition.familiar.background_levels)
        * len(definition.familiar.irf_fwhm_values_ns)
        * len(definition.familiar.irf_shift_values_ns)
        * PAIR_VARIANTS_PER_CONDITION
    )

    assert len(design) == expected_size
    assert len(design) == 192


def test_paired_parameter_design_has_unique_pair_ids(
) -> None:
    design = build_paired_parameter_design(
        default_generalization_suite()
    )

    assert design["pair_id"].is_unique

    assert design["pair_id"].tolist() == list(
        range(len(design))
    )


def test_paired_parameter_design_balances_lifetimes(
) -> None:
    definition = default_generalization_suite()

    design = build_paired_parameter_design(
        definition
    )

    lifetime_counts = (
        design["lifetime_true_ns"]
        .value_counts()
        .sort_index()
    )

    assert lifetime_counts.nunique() == 1

    assert tuple(lifetime_counts.index) == (
        definition.familiar.lifetime_values_ns
    )


def test_paired_parameter_design_balances_photon_counts(
) -> None:
    definition = default_generalization_suite()

    design = build_paired_parameter_design(
        definition
    )

    photon_counts = (
        design["signal_photon_count_target"]
        .value_counts()
        .sort_index()
    )

    assert photon_counts.nunique() == 1

    assert tuple(photon_counts.index) == (
        definition.familiar.signal_photon_counts
    )


def test_test_a_uses_only_familiar_conditions(
) -> None:
    definition = default_generalization_suite()

    table = build_test_parameter_table(
        definition,
        "A",
    )

    familiar = definition.familiar

    assert set(
        table["lifetime_true_ns"]
    ) == set(
        familiar.lifetime_values_ns
    )

    assert set(
        table[
            "signal_photon_count_target"
        ]
    ) == set(
        familiar.signal_photon_counts
    )

    assert set(
        table["background_per_bin"]
    ) == set(
        familiar.background_levels
    )

    assert set(
        table["irf_fwhm_ns"]
    ) == set(
        familiar.irf_fwhm_values_ns
    )

    assert set(
        table["irf_shift_ns"]
    ) == set(
        familiar.irf_shift_values_ns
    )


def test_test_b_photon_counts_are_outside_familiar_support(
) -> None:
    definition = default_generalization_suite()

    table = build_test_parameter_table(
        definition,
        "B",
    )

    familiar_counts = (
        definition.familiar.signal_photon_counts
    )

    familiar_min = min(
        familiar_counts
    )
    familiar_max = max(
        familiar_counts
    )

    photon_counts = table[
        "signal_photon_count_target"
    ].to_numpy()

    assert np.all(
        (photon_counts < familiar_min)
        | (photon_counts > familiar_max)
    )

    expected_values = set(
        definition.numerics.test_b_low_photon_counts
        + definition.numerics.test_b_high_photon_counts
    )

    assert set(
        photon_counts
    ) == expected_values


def test_test_c_changes_only_irf_width(
) -> None:
    definition = default_generalization_suite()

    test_a = build_test_parameter_table(
        definition,
        "A",
    )

    test_c = build_test_parameter_table(
        definition,
        "C",
    )

    np.testing.assert_array_equal(
        test_c["pair_id"],
        test_a["pair_id"],
    )

    np.testing.assert_allclose(
        test_c["lifetime_true_ns"],
        test_a["lifetime_true_ns"],
    )

    np.testing.assert_array_equal(
        test_c[
            "signal_photon_count_target"
        ],
        test_a[
            "signal_photon_count_target"
        ],
    )

    np.testing.assert_allclose(
        test_c["background_per_bin"],
        test_a["background_per_bin"],
    )

    np.testing.assert_allclose(
        test_c["irf_shift_ns"],
        test_a["irf_shift_ns"],
    )

    assert set(
        test_c["irf_fwhm_ns"]
    ) == set(
        definition.numerics
        .test_c_irf_fwhm_values_ns
    )


def test_test_c_irf_is_broader_than_familiar_irf(
) -> None:
    definition = default_generalization_suite()

    test_c = build_test_parameter_table(
        definition,
        "C",
    )

    assert np.all(
        test_c["irf_fwhm_ns"].to_numpy()
        > max(
            definition.familiar
            .irf_fwhm_values_ns
        )
    )


def test_test_d_changes_only_background(
) -> None:
    definition = default_generalization_suite()

    test_a = build_test_parameter_table(
        definition,
        "A",
    )

    test_d = build_test_parameter_table(
        definition,
        "D",
    )

    np.testing.assert_array_equal(
        test_d["pair_id"],
        test_a["pair_id"],
    )

    np.testing.assert_allclose(
        test_d["lifetime_true_ns"],
        test_a["lifetime_true_ns"],
    )

    np.testing.assert_array_equal(
        test_d[
            "signal_photon_count_target"
        ],
        test_a[
            "signal_photon_count_target"
        ],
    )

    np.testing.assert_allclose(
        test_d["irf_fwhm_ns"],
        test_a["irf_fwhm_ns"],
    )

    np.testing.assert_allclose(
        test_d["irf_shift_ns"],
        test_a["irf_shift_ns"],
    )

    assert set(
        test_d["background_per_bin"]
    ) == set(
        definition.numerics
        .test_d_background_levels
    )


def test_test_d_background_is_above_familiar_support(
) -> None:
    definition = default_generalization_suite()

    test_d = build_test_parameter_table(
        definition,
        "D",
    )

    assert np.all(
        test_d[
            "background_per_bin"
        ].to_numpy()
        > max(
            definition.familiar
            .background_levels
        )
    )


def test_test_e_uses_requested_temporal_offsets(
) -> None:
    definition = default_generalization_suite()

    test_e = build_test_parameter_table(
        definition,
        "E",
    )

    assert set(
        test_e["irf_shift_ns"]
    ) == set(
        definition.numerics
        .test_e_irf_shift_values_ns
    )

    shift_bins = (
        test_e["irf_shift_ns"].to_numpy()
        / definition.familiar.time_step_ns
    )

    assert set(
        np.rint(
            shift_bins
        ).astype(
            np.int64
        )
    ) == {
        -3,
        3,
    }


def test_test_e_changes_only_temporal_alignment(
) -> None:
    definition = default_generalization_suite()

    test_a = build_test_parameter_table(
        definition,
        "A",
    )

    test_e = build_test_parameter_table(
        definition,
        "E",
    )

    np.testing.assert_array_equal(
        test_e["pair_id"],
        test_a["pair_id"],
    )

    np.testing.assert_allclose(
        test_e["lifetime_true_ns"],
        test_a["lifetime_true_ns"],
    )

    np.testing.assert_array_equal(
        test_e[
            "signal_photon_count_target"
        ],
        test_a[
            "signal_photon_count_target"
        ],
    )

    np.testing.assert_allclose(
        test_e["background_per_bin"],
        test_a["background_per_bin"],
    )

    np.testing.assert_allclose(
        test_e["irf_fwhm_ns"],
        test_a["irf_fwhm_ns"],
    )


def test_tests_a_to_e_have_pairwise_matched_lifetimes(
) -> None:
    definition = default_generalization_suite()

    reference = build_test_parameter_table(
        definition,
        "A",
    )

    for test_id in (
        "B",
        "C",
        "D",
        "E",
    ):
        shifted = build_test_parameter_table(
            definition,
            test_id,
        )

        np.testing.assert_array_equal(
            shifted["pair_id"],
            reference["pair_id"],
        )

        np.testing.assert_allclose(
            shifted["lifetime_true_ns"],
            reference["lifetime_true_ns"],
        )


def test_a_to_e_parameter_metadata_identifies_regimes(
) -> None:
    definition = default_generalization_suite()

    for test_id in (
        "A",
        "B",
        "C",
        "D",
        "E",
    ):
        table = build_test_parameter_table(
            definition,
            test_id,
        )

        regime = definition.regime(
            test_id
        )

        assert set(
            table["test_id"]
        ) == {
            test_id
        }

        assert set(
            table["regime_name"]
        ) == {
            regime.name
        }

        assert set(
            table["shift"]
        ) == {
            regime.shift.value
        }

        assert set(
            table["changed_factors"]
        ) == {
            ",".join(
                regime.changed_factors
            )
        }

        assert set(
            table["decay_model"]
        ) == {
            "monoexponential"
        }

        assert set(
            table["random_seed"]
        ) == {
            definition.seed_for(
                test_id
            )
        }


def test_generated_tests_a_to_e_have_expected_shapes(
) -> None:
    definition = default_generalization_suite()

    for test_id in (
        "A",
        "B",
        "C",
        "D",
        "E",
    ):
        test = (
            generate_generalization_test(
                definition,
                test_id,
            )
        )

        assert test.X_histograms.shape == (
            192,
            400,
        )

        assert test.y.shape == (
            192,
        )

        assert test.metadata.shape[0] == 192

        assert test.time.shape == (
            400,
        )


def test_generalization_test_generation_is_deterministic(
) -> None:
    definition = default_generalization_suite()

    first = (
        generate_generalization_test(
            definition,
            "C",
        )
    )

    second = (
        generate_generalization_test(
            definition,
            "C",
        )
    )

    np.testing.assert_array_equal(
        first.time,
        second.time,
    )

    np.testing.assert_array_equal(
        first.X_histograms,
        second.X_histograms,
    )

    np.testing.assert_array_equal(
        first.y,
        second.y,
    )

    pd.testing.assert_frame_equal(
        first.metadata,
        second.metadata,
    )


def test_generated_metadata_and_targets_are_aligned(
) -> None:
    definition = default_generalization_suite()

    test = (
        generate_generalization_test(
            definition,
            "B",
        )
    )

    np.testing.assert_allclose(
        test.y,
        test.metadata[
            "lifetime_true_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    )

    np.testing.assert_array_equal(
        test.metadata[
            "sample_id"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            len(test.y),
            dtype=np.int64,
        ),
    )


def test_test_f_preserves_test_a_physical_conditions(
) -> None:
    definition = default_generalization_suite()

    test_a = build_test_parameter_table(
        definition,
        "A",
    )

    test_f = build_test_f_parameter_table(
        definition
    )

    np.testing.assert_array_equal(
        test_f["pair_id"],
        test_a["pair_id"],
    )

    np.testing.assert_allclose(
        test_f["lifetime_true_ns"],
        test_a["lifetime_true_ns"],
    )

    np.testing.assert_array_equal(
        test_f[
            "signal_photon_count_target"
        ],
        test_a[
            "signal_photon_count_target"
        ],
    )

    np.testing.assert_allclose(
        test_f["background_per_bin"],
        test_a["background_per_bin"],
    )

    np.testing.assert_allclose(
        test_f["irf_fwhm_ns"],
        test_a["irf_fwhm_ns"],
    )

    np.testing.assert_allclose(
        test_f["irf_shift_ns"],
        test_a["irf_shift_ns"],
    )


def test_test_f_contains_secondary_exponential_component(
) -> None:
    definition = default_generalization_suite()

    test_f = build_test_f_parameter_table(
        definition
    )

    assert set(
        test_f[
            "secondary_fraction"
        ]
    ) == set(
        definition.numerics
        .test_f_secondary_fractions
    )

    np.testing.assert_allclose(
        test_f[
            "secondary_lifetime_ns"
        ].to_numpy(),
        (
            test_f[
                "primary_lifetime_ns"
            ].to_numpy()
            * definition.numerics
            .test_f_secondary_lifetime_factor
        ),
    )


def test_test_f_contains_weak_and_moderate_severity(
) -> None:
    definition = default_generalization_suite()

    test_f = build_test_f_parameter_table(
        definition
    )

    assert set(
        test_f[
            "model_mismatch_severity"
        ]
    ) == {
        "weak",
        "moderate",
    }

    severity_counts = (
        test_f[
            "model_mismatch_severity"
        ]
        .value_counts()
    )

    assert severity_counts[
        "weak"
    ] == 96

    assert severity_counts[
        "moderate"
    ] == 96


def test_test_f_severity_is_balanced_across_photon_counts(
) -> None:
    definition = default_generalization_suite()

    test_f = build_test_f_parameter_table(
        definition
    )

    balance = pd.crosstab(
        test_f[
            "model_mismatch_severity"
        ],
        test_f[
            "signal_photon_count_target"
        ],
    )

    assert set(
        balance.index
    ) == {
        "weak",
        "moderate",
    }

    assert set(
        balance.columns
    ) == {
        1_000,
        10_000,
    }

    np.testing.assert_array_equal(
        balance.to_numpy(),
        np.full(
            (
                2,
                2,
            ),
            48,
            dtype=np.int64,
        ),
    )


def test_test_f_signal_photon_weighted_lifetime_is_correct(
) -> None:
    definition = default_generalization_suite()

    test_f = build_test_f_parameter_table(
        definition
    )

    primary = test_f[
        "primary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    secondary = test_f[
        "secondary_lifetime_ns"
    ].to_numpy(
        dtype=np.float64
    )

    fraction = test_f[
        "secondary_fraction"
    ].to_numpy(
        dtype=np.float64
    )

    expected = (
        (1.0 - fraction)
        * primary
        + fraction
        * secondary
    )

    np.testing.assert_allclose(
        test_f[
            "signal_photon_weighted_lifetime_ns"
        ].to_numpy(
            dtype=np.float64
        ),
        expected,
    )


def test_generated_test_f_uses_primary_lifetime_as_target(
) -> None:
    definition = default_generalization_suite()

    test_f = generate_generalization_test(
        definition,
        "F",
    )

    np.testing.assert_allclose(
        test_f.y,
        test_f.metadata[
            "primary_lifetime_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    )

    np.testing.assert_allclose(
        test_f.y,
        test_f.metadata[
            "lifetime_true_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    )


def test_test_f_uses_expected_decay_model_metadata(
) -> None:
    definition = default_generalization_suite()

    test_f = build_test_f_parameter_table(
        definition
    )

    assert set(
        test_f["decay_model"]
    ) == {
        "biexponential"
    }

    assert set(
        test_f["changed_factors"]
    ) == {
        "decay_model"
    }

    assert set(
        test_f["random_seed"]
    ) == {
        50_006
    }


def test_test_f_target_is_primary_lifetime(
) -> None:
    definition = default_generalization_suite()

    test_f = (
        generate_generalization_test(
            definition,
            "F",
        )
    )

    np.testing.assert_allclose(
        test_f.y,
        test_f.metadata[
            "primary_lifetime_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    )

    np.testing.assert_allclose(
        test_f.y,
        test_f.metadata[
            "lifetime_true_ns"
        ].to_numpy(
            dtype=np.float64
        ),
    )


def test_generated_test_f_has_expected_shape(
) -> None:
    definition = default_generalization_suite()

    test_f = (
        generate_generalization_test(
            definition,
            "F",
        )
    )

    assert test_f.test_id == "F"

    assert test_f.X_histograms.shape == (
        192,
        400,
    )

    assert test_f.y.shape == (
        192,
    )

    assert test_f.metadata.shape[0] == 192

    assert test_f.time.shape == (
        400,
    )


def test_test_f_generation_is_deterministic(
) -> None:
    definition = default_generalization_suite()

    first = (
        generate_generalization_test(
            definition,
            "F",
        )
    )

    second = (
        generate_generalization_test(
            definition,
            "F"
        )
    )

    np.testing.assert_array_equal(
        first.time,
        second.time,
    )

    np.testing.assert_array_equal(
        first.X_histograms,
        second.X_histograms,
    )

    np.testing.assert_array_equal(
        first.y,
        second.y,
    )

    pd.testing.assert_frame_equal(
        first.metadata,
        second.metadata,
    )


def test_generalization_test_dispatcher_supports_a_to_f(
) -> None:
    definition = default_generalization_suite()

    for test_id in (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ):
        test = generate_generalization_test(
            definition,
            test_id,
        )

        assert test.test_id == test_id


def test_generalization_test_dispatcher_rejects_unknown_test(
) -> None:
    definition = default_generalization_suite()

    with pytest.raises(
        KeyError,
        match="Unknown generalization test",
    ):
        generate_generalization_test(
            definition,
            "G",
        )


def test_complete_generalization_suite_contains_a_to_f(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    assert tuple(
        test.test_id
        for test in suite.tests
    ) == (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    )


def test_tests_a_to_f_have_exactly_matched_lifetimes(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    reference = suite.get_test(
        "A"
    )

    for test_id in (
        "B",
        "C",
        "D",
        "E",
        "F",
    ):
        shifted = suite.get_test(
            test_id
        )

        np.testing.assert_array_equal(
            shifted.metadata[
                "pair_id"
            ].to_numpy(),
            reference.metadata[
                "pair_id"
            ].to_numpy(),
        )

        np.testing.assert_allclose(
            shifted.y,
            reference.y,
        )


def test_complete_generalization_suite_has_common_geometry(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    reference_time = suite.get_test(
        "A"
    ).time

    for test in suite.tests:
        assert test.X_histograms.shape == (
            192,
            400,
        )

        assert test.y.shape == (
            192,
        )

        assert test.metadata.shape[0] == 192

        np.testing.assert_array_equal(
            test.time,
            reference_time,
        )


def test_generated_suite_lookup_is_case_insensitive(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    assert (
        suite.get_test("f")
        is suite.get_test("F")
    )


def test_generated_suite_preserves_provenance(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    for test in suite.tests:
        assert set(
            test.metadata[
                "suite_version"
            ]
        ) == {
            suite.definition.suite_version
        }

        assert set(
            test.metadata[
                "development_reference"
            ]
        ) == {
            suite.definition
            .development_reference
        }

        assert set(
            test.metadata[
                "random_seed"
            ]
        ) == {
            suite.definition.seed_for(
                test.test_id
            )
        }


def test_generated_suite_uses_distinct_test_seeds(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    generated_seeds = tuple(
        int(
            test.metadata[
                "random_seed"
            ].iloc[0]
        )
        for test in suite.tests
    )

    assert len(
        set(generated_seeds)
    ) == 6

    assert generated_seeds == (
        50_001,
        50_002,
        50_003,
        50_004,
        50_005,
        50_006,
    )


def test_complete_generalization_suite_reports_size(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    assert suite.n_tests == 6

    assert (
        suite.n_samples_per_test
        == 192
    )

    assert suite.n_time_bins == 400

    assert (
        suite.n_total_histograms
        == 1_152
    )


def test_generalization_suite_manifest_has_one_row_per_test(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    manifest = (
        build_generalization_suite_manifest(
            suite
        )
    )

    assert manifest.shape[0] == 6

    assert tuple(
        manifest["test_id"]
    ) == (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    )

    assert tuple(
        manifest["random_seed"]
    ) == (
        50_001,
        50_002,
        50_003,
        50_004,
        50_005,
        50_006,
    )

    assert set(
        manifest["n_samples"]
    ) == {
        192
    }

    assert set(
        manifest["n_time_bins"]
    ) == {
        400
    }


def test_generalization_suite_manifest_preserves_provenance(
) -> None:
    suite = (
        generate_generalization_test_suite()
    )

    manifest = (
        build_generalization_suite_manifest(
            suite
        )
    )

    assert set(
        manifest["suite_version"]
    ) == {
        suite.definition.suite_version
    }

    assert set(
        manifest[
            "development_reference"
        ]
    ) == {
        suite.definition
        .development_reference
    }


