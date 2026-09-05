"""Generation of the final Week 8 robustness test suite."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tcspc_toolkit.generalization import (
    FINAL_ROBUSTNESS_TEST_IDS,
    GeneralizationSuiteDefinition,
    default_generalization_suite,
)
from tcspc_toolkit.simulation import (
    simulate_irf_convolved_biexponential_histogram,
    simulate_irf_convolved_histogram,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


PAIR_VARIANTS_PER_CONDITION = 6


@dataclass(frozen=True)
class GeneralizationTestMeasurements:
    """Measurements belonging to one final robustness test.

    Attributes
    ----------
    test_id:
        Stable A-F robustness-test identifier.
    time:
        Shared TCSPC time axis.
    X_histograms:
        Poisson-sampled histograms with shape
        ``(n_samples, n_time_bins)``.
    y:
        Primary true fluorescence lifetime for every sample.
    metadata:
        Physical conditions, pairing identifiers, regime
        information, and simulation diagnostics.
    """

    test_id: str
    time: FloatArray
    X_histograms: IntArray
    y: FloatArray
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if self.test_id not in FINAL_ROBUSTNESS_TEST_IDS:
            raise ValueError(
                "test_id must identify one of Tests A-F."
            )

        if self.time.ndim != 1:
            raise ValueError(
                "time must be one-dimensional."
            )

        if self.X_histograms.ndim != 2:
            raise ValueError(
                "X_histograms must be two-dimensional."
            )

        n_samples, n_time_bins = self.X_histograms.shape

        if n_time_bins != self.time.size:
            raise ValueError(
                "Histogram width must match the time axis."
            )

        if self.y.shape != (n_samples,):
            raise ValueError(
                "y must contain exactly one lifetime per histogram."
            )

        if self.metadata.shape[0] != n_samples:
            raise ValueError(
                "metadata must contain exactly one row per histogram."
            )


@dataclass(frozen=True)
class GeneralizationTestSuite:
    """Complete generated A-F robustness test suite."""

    definition: GeneralizationSuiteDefinition
    tests: tuple[
        GeneralizationTestMeasurements,
        GeneralizationTestMeasurements,
        GeneralizationTestMeasurements,
        GeneralizationTestMeasurements,
        GeneralizationTestMeasurements,
        GeneralizationTestMeasurements,
    ]

    def __post_init__(self) -> None:
        test_ids = tuple(
            test.test_id
            for test in self.tests
        )

        if test_ids != FINAL_ROBUSTNESS_TEST_IDS:
            raise ValueError(
                "Generated robustness suite must contain "
                "Tests A-F exactly once and in order."
            )

        reference = self.tests[0]

        reference_pair_ids = reference.metadata[
            "pair_id"
        ].to_numpy(
            dtype=np.int64
        )

        reference_lifetimes = reference.y

        for test in self.tests:
            if not np.array_equal(
                    test.time,
                    reference.time,
            ):
                raise ValueError(
                    "Tests A-F must share the same time axis."
                )

            if (
                    test.X_histograms.shape
                    != reference.X_histograms.shape
            ):
                raise ValueError(
                    "Tests A-F must share the same "
                    "histogram geometry."
                )

            pair_ids = test.metadata[
                "pair_id"
            ].to_numpy(
                dtype=np.int64
            )

            if not np.array_equal(
                    pair_ids,
                    reference_pair_ids,
            ):
                raise ValueError(
                    "Tests A-F must preserve the common "
                    "pair_id assignments."
                )

            if not np.array_equal(
                    test.y,
                    reference_lifetimes,
            ):
                raise ValueError(
                    "Tests A-F must preserve the paired "
                    "lifetime targets."
                )

            if set(
                    test.metadata["test_id"]
            ) != {
                test.test_id
            }:
                raise ValueError(
                    f"Metadata for Test {test.test_id} "
                    "contains inconsistent test identifiers."
                )

            if set(
                    test.metadata["suite_version"]
            ) != {
                self.definition.suite_version
            }:
                raise ValueError(
                    f"Metadata for Test {test.test_id} "
                    "contains an inconsistent suite version."
                )

            if set(
                    test.metadata["development_reference"]
            ) != {
                self.definition.development_reference
            }:
                raise ValueError(
                    f"Metadata for Test {test.test_id} "
                    "contains an inconsistent development "
                    "reference."
                )

            if set(
                    test.metadata["random_seed"]
            ) != {
                self.definition.seed_for(
                    test.test_id
                )
            }:
                raise ValueError(
                    f"Metadata for Test {test.test_id} "
                    "contains an inconsistent random seed."
                )

    @property
    def n_tests(self) -> int:
        """Return the number of robustness regimes."""

        return len(
            self.tests
        )

    @property
    def n_samples_per_test(self) -> int:
        """Return the number of paired samples in each test."""

        return int(
            self.tests[0].y.size
        )

    @property
    def n_time_bins(self) -> int:
        """Return the common histogram width."""

        return int(
            self.tests[0].time.size
        )

    @property
    def n_total_histograms(self) -> int:
        """Return the total number of A-F histograms."""

        return (
                self.n_tests
                * self.n_samples_per_test
        )

    def get_test(
        self,
        test_id: str,
    ) -> GeneralizationTestMeasurements:
        """Return one generated robustness test."""

        normalized_id = test_id.strip().upper()

        for test in self.tests:
            if test.test_id == normalized_id:
                return test

        raise KeyError(
            f"Unknown generalization test: {test_id!r}."
        )


def build_generalization_time_axis(
    definition: GeneralizationSuiteDefinition,
) -> FloatArray:
    """Construct the canonical TCSPC time axis."""

    familiar = definition.familiar

    return np.arange(
        familiar.time_start_ns,
        familiar.time_stop_ns,
        familiar.time_step_ns,
        dtype=np.float64,
    )


def build_paired_parameter_design(
    definition: GeneralizationSuiteDefinition,
) -> pd.DataFrame:
    """Build the common physical reference design for Tests A-F.

    Every row defines one paired comparison across the six
    robustness regimes.

    Tests A-F will preserve ``pair_id`` and ``lifetime_true_ns``.
    Tests B-F then modify only the factor specified by the
    generalization protocol.
    """

    familiar = definition.familiar

    rows: list[dict[str, float | int]] = []

    parameter_grid = product(
        familiar.lifetime_values_ns,
        familiar.background_levels,
        familiar.irf_fwhm_values_ns,
        familiar.irf_shift_values_ns,
        range(PAIR_VARIANTS_PER_CONDITION),
    )

    familiar_photon_counts = (
        familiar.signal_photon_counts
    )

    for pair_id, parameters in enumerate(
        parameter_grid
    ):
        (
            lifetime_ns,
            background_per_bin,
            irf_fwhm_ns,
            irf_shift_ns,
            pair_variant,
        ) = parameters

        signal_photon_count = (
            familiar_photon_counts[
                pair_variant
                % len(familiar_photon_counts)
            ]
        )

        rows.append(
            {
                "pair_id": pair_id,
                "pair_variant": pair_variant,
                "lifetime_true_ns": float(
                    lifetime_ns
                ),
                "signal_photon_count_target": int(
                    signal_photon_count
                ),
                "background_per_bin": float(
                    background_per_bin
                ),
                "irf_fwhm_ns": float(
                    irf_fwhm_ns
                ),
                "irf_shift_ns": float(
                    irf_shift_ns
                ),
            }
        )

    return pd.DataFrame(rows)


def build_test_parameter_table(
    definition: GeneralizationSuiteDefinition,
    test_id: str,
) -> pd.DataFrame:
    """Build the physical parameter table for one A-E test.

    Tests A-E share the same ``pair_id`` and primary lifetime
    assignments. Each shifted test modifies only the nuisance
    parameter defined by the Week 8 generalization protocol.

    Test F is excluded here because it requires a different
    decay model and is generated separately.
    """

    normalized_id = test_id.strip().upper()

    if normalized_id not in (
        "A",
        "B",
        "C",
        "D",
        "E",
    ):
        raise ValueError(
            "build_test_parameter_table supports only "
            "Tests A-E."
        )

    table = build_paired_parameter_design(
        definition
    ).copy(
        deep=True
    )

    familiar = definition.familiar
    numerics = definition.numerics

    pair_variants = table[
        "pair_variant"
    ].to_numpy(
        dtype=np.int64
    )

    if normalized_id == "A":
        pass

    elif normalized_id == "B":
        ood_photon_counts = (
            numerics.test_b_low_photon_counts
            + numerics.test_b_high_photon_counts
        )

        table[
            "signal_photon_count_target"
        ] = np.asarray(
            [
                ood_photon_counts[
                    pair_variant
                    % len(ood_photon_counts)
                ]
                for pair_variant in pair_variants
            ],
            dtype=np.int64,
        )

    elif normalized_id == "C":
        ood_irf_widths = (
            numerics.test_c_irf_fwhm_values_ns
        )

        table[
            "irf_fwhm_ns"
        ] = np.asarray(
            [
                ood_irf_widths[
                    pair_variant
                    % len(ood_irf_widths)
                ]
                for pair_variant in pair_variants
            ],
            dtype=np.float64,
        )

    elif normalized_id == "D":
        elevated_backgrounds = (
            numerics.test_d_background_levels
        )

        table[
            "background_per_bin"
        ] = np.asarray(
            [
                elevated_backgrounds[
                    pair_variant
                    % len(elevated_backgrounds)
                ]
                for pair_variant in pair_variants
            ],
            dtype=np.float64,
        )

    elif normalized_id == "E":
        temporal_shifts = (
            numerics.test_e_irf_shift_values_ns
        )

        table[
            "irf_shift_ns"
        ] = np.asarray(
            [
                temporal_shifts[
                    pair_variant
                    % len(temporal_shifts)
                ]
                for pair_variant in pair_variants
            ],
            dtype=np.float64,
        )

    regime = definition.regime(
        normalized_id
    )

    table.insert(
        0,
        "sample_id",
        np.arange(
            len(table),
            dtype=np.int64,
        ),
    )

    table["test_id"] = normalized_id
    table["regime_name"] = regime.name
    table["shift"] = regime.shift.value

    table["changed_factors"] = (
        ",".join(
            regime.changed_factors
        )
    )

    table["decay_model"] = (
        "monoexponential"
    )

    table["irf_centre_ns"] = float(
        familiar.irf_centre_ns
    )

    table["random_seed"] = (
        definition.seed_for(
            normalized_id
        )
    )

    table["suite_version"] = (
        definition.suite_version
    )

    table["development_reference"] = (
        definition.development_reference
    )

    return table


def build_test_f_parameter_table(
    definition: GeneralizationSuiteDefinition,
) -> pd.DataFrame:
    """Build the paired parameter table for Test F.

    Test F preserves the lifetime, photon-count, background,
    IRF-width, and temporal-alignment assignments of Test A.

    The only intentional change is the decay model: each curve
    contains a secondary exponential component with the fraction
    and lifetime factor frozen in the Week 8 protocol.
    """

    table = build_paired_parameter_design(
        definition
    ).copy(
        deep=True
    )

    familiar = definition.familiar
    numerics = definition.numerics
    regime = definition.regime("F")

    table.insert(
        0,
        "sample_id",
        np.arange(
            len(table),
            dtype=np.int64,
        ),
    )

    table["test_id"] = "F"
    table["regime_name"] = regime.name
    table["shift"] = regime.shift.value

    table["changed_factors"] = (
        ",".join(
            regime.changed_factors
        )
    )

    table["decay_model"] = (
        "biexponential"
    )

    table["primary_lifetime_ns"] = table[
        "lifetime_true_ns"
    ].to_numpy(
        dtype=np.float64
    )

    table["secondary_lifetime_ns"] = (
        table["primary_lifetime_ns"]
        * numerics.test_f_secondary_lifetime_factor
    )

    secondary_fractions = np.asarray(
        numerics.test_f_secondary_fractions,
        dtype=np.float64,
    )

    severity_labels = np.asarray(
        [
            "weak",
            "moderate",
        ],
        dtype=object,
    )

    severity_block_size = len(
        familiar.signal_photon_counts
    )

    severity_indices = (
                               table[
                                   "pair_id"
                               ].to_numpy(
                                   dtype=np.int64
                               )
                               // severity_block_size
                       ) % secondary_fractions.size

    table[
        "model_mismatch_severity"
    ] = severity_labels[
        severity_indices
    ]

    table[
        "secondary_fraction"
    ] = secondary_fractions[
        severity_indices
    ]

    table[
        "signal_photon_weighted_lifetime_ns"
    ] = (
            (
                    1.0
                    - table[
                        "secondary_fraction"
                    ]
            )
            * table[
                "primary_lifetime_ns"
            ]
            + table[
                "secondary_fraction"
            ]
            * table[
                "secondary_lifetime_ns"
            ]
    )

    table["secondary_lifetime_factor"] = float(
        numerics.test_f_secondary_lifetime_factor
    )

    table["irf_centre_ns"] = float(
        familiar.irf_centre_ns
    )

    table["random_seed"] = (
        definition.seed_for("F")
    )

    table["suite_version"] = (
        definition.suite_version
    )

    table["development_reference"] = (
        definition.development_reference
    )

    return table


def _generate_monoexponential_generalization_test(
    definition: GeneralizationSuiteDefinition,
    test_id: str,
) -> GeneralizationTestMeasurements:
    """Generate one final mono-exponential robustness test.

    This function generates Tests A-E using the test-specific
    frozen random seed stored in the generalization-suite
    definition.
    """

    normalized_id = test_id.strip().upper()

    if normalized_id not in (
        "A",
        "B",
        "C",
        "D",
        "E",
    ):
        raise ValueError(
            "This function supports only Tests A-E."
        )

    time = build_generalization_time_axis(
        definition
    )

    parameter_table = (
        build_test_parameter_table(
            definition,
            normalized_id,
        )
    )

    n_samples = len(
        parameter_table
    )

    X_histograms = np.empty(
        (
            n_samples,
            time.size,
        ),
        dtype=np.int64,
    )

    y = parameter_table[
        "lifetime_true_ns"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    rng = np.random.default_rng(
        definition.seed_for(
            normalized_id
        )
    )

    metadata_rows: list[
        dict[str, float | int | str]
    ] = []

    for row in parameter_table.itertuples(
        index=False
    ):
        (
            measured_counts,
            simulation_metadata,
        ) = simulate_irf_convolved_histogram(
            time=time,
            lifetime_ns=float(
                row.lifetime_true_ns
            ),
            signal_photon_count=int(
                row.signal_photon_count_target
            ),
            background_per_bin=float(
                row.background_per_bin
            ),
            irf_centre_ns=float(
                row.irf_centre_ns
            ),
            irf_fwhm_ns=float(
                row.irf_fwhm_ns
            ),
            irf_shift_ns=float(
                row.irf_shift_ns
            ),
            rng=rng,
        )

        X_histograms[
            row.sample_id
        ] = measured_counts

        metadata_row = dict(
            row._asdict()
        )

        metadata_row.update(
            {
                "expected_signal_counts": (
                    simulation_metadata[
                        "expected_signal_counts"
                    ]
                ),
                "expected_background_counts": (
                    simulation_metadata[
                        "expected_background_counts"
                    ]
                ),
                "expected_total_counts": (
                    simulation_metadata[
                        "expected_total_counts"
                    ]
                ),
                "measured_total_counts": (
                    simulation_metadata[
                        "measured_total_counts"
                    ]
                ),
            }
        )

        metadata_rows.append(
            metadata_row
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    return GeneralizationTestMeasurements(
        test_id=normalized_id,
        time=time,
        X_histograms=X_histograms,
        y=y,
        metadata=metadata,
    )


def _generate_biexponential_generalization_test(
    definition: GeneralizationSuiteDefinition,
) -> GeneralizationTestMeasurements:
    """Generate final Test F with controlled model mismatch."""

    time = build_generalization_time_axis(
        definition
    )

    parameter_table = (
        build_test_f_parameter_table(
            definition
        )
    )

    n_samples = len(
        parameter_table
    )

    X_histograms = np.empty(
        (
            n_samples,
            time.size,
        ),
        dtype=np.int64,
    )

    y = parameter_table[
        "lifetime_true_ns"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    rng = np.random.default_rng(
        definition.seed_for("F")
    )

    metadata_rows: list[
        dict[str, float | int | str]
    ] = []

    for row in parameter_table.itertuples(
        index=False
    ):
        (
            measured_counts,
            simulation_metadata,
        ) = (
            simulate_irf_convolved_biexponential_histogram(
                time=time,
                primary_lifetime_ns=float(
                    row.primary_lifetime_ns
                ),
                secondary_lifetime_ns=float(
                    row.secondary_lifetime_ns
                ),
                secondary_fraction=float(
                    row.secondary_fraction
                ),
                signal_photon_count=int(
                    row.signal_photon_count_target
                ),
                background_per_bin=float(
                    row.background_per_bin
                ),
                irf_centre_ns=float(
                    row.irf_centre_ns
                ),
                irf_fwhm_ns=float(
                    row.irf_fwhm_ns
                ),
                irf_shift_ns=float(
                    row.irf_shift_ns
                ),
                rng=rng,
            )
        )

        X_histograms[
            row.sample_id
        ] = measured_counts

        metadata_row = dict(
            row._asdict()
        )

        metadata_row.update(
            {
                "expected_signal_counts": (
                    simulation_metadata[
                        "expected_signal_counts"
                    ]
                ),
                "expected_background_counts": (
                    simulation_metadata[
                        "expected_background_counts"
                    ]
                ),
                "expected_total_counts": (
                    simulation_metadata[
                        "expected_total_counts"
                    ]
                ),
                "measured_total_counts": (
                    simulation_metadata[
                        "measured_total_counts"
                    ]
                ),
            }
        )

        metadata_rows.append(
            metadata_row
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    return GeneralizationTestMeasurements(
        test_id="F",
        time=time,
        X_histograms=X_histograms,
        y=y,
        metadata=metadata,
    )


def generate_generalization_test(
    definition: GeneralizationSuiteDefinition,
    test_id: str,
) -> GeneralizationTestMeasurements:
    """Generate one final A-F robustness test."""

    normalized_id = test_id.strip().upper()

    if normalized_id in (
        "A",
        "B",
        "C",
        "D",
        "E",
    ):
        return (
            _generate_monoexponential_generalization_test(
                definition,
                normalized_id,
            )
        )

    if normalized_id == "F":
        return (
            _generate_biexponential_generalization_test(
                definition
            )
        )

    raise KeyError(
        f"Unknown generalization test: {test_id!r}."
    )


def generate_generalization_test_suite(
    definition: (
        GeneralizationSuiteDefinition
        | None
    ) = None,
) -> GeneralizationTestSuite:
    """Generate the complete final A-F robustness test suite.

    If no explicit definition is supplied, the canonical frozen
    Week 8 generalization-suite definition is used.
    """

    if definition is None:
        definition = (
            default_generalization_suite()
        )

    tests = (
        generate_generalization_test(
            definition,
            "A",
        ),
        generate_generalization_test(
            definition,
            "B",
        ),
        generate_generalization_test(
            definition,
            "C",
        ),
        generate_generalization_test(
            definition,
            "D",
        ),
        generate_generalization_test(
            definition,
            "E",
        ),
        generate_generalization_test(
            definition,
            "F",
        ),
    )

    return GeneralizationTestSuite(
        definition=definition,
        tests=tests,
    )


def build_generalization_suite_manifest(
    suite: GeneralizationTestSuite,
) -> pd.DataFrame:
    """Summarize the generated A-F robustness suite.

    The manifest contains structural and provenance information
    only. It does not contain estimator predictions or performance
    metrics.
    """

    rows: list[
        dict[str, str | int]
    ] = []

    for test in suite.tests:
        regime = suite.definition.regime(
            test.test_id
        )

        rows.append(
            {
                "test_id": test.test_id,
                "regime_name": regime.name,
                "shift": regime.shift.value,
                "changed_factors": ",".join(
                    regime.changed_factors
                ),
                "random_seed": (
                    suite.definition.seed_for(
                        test.test_id
                    )
                ),
                "n_samples": int(
                    test.y.size
                ),
                "n_time_bins": int(
                    test.time.size
                ),
                "suite_version": (
                    suite.definition.suite_version
                ),
                "development_reference": (
                    suite.definition
                    .development_reference
                ),
            }
        )

    return pd.DataFrame(
        rows
    )
