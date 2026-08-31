"""Generalization protocol for TCSPC lifetime estimation."""

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite


FINAL_ROBUSTNESS_TEST_IDS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
)


DEVELOPMENT_ONLY_OPERATIONS = (
    "scaler_fitting",
    "pca_fitting",
    "model_fitting",
    "feature_selection",
    "hyperparameter_selection",
    "cross_validation",
)


class GeneralizationShift(str, Enum):
    """Primary distribution shift represented by a robustness regime."""

    IN_DISTRIBUTION = "in_distribution"
    PHOTON_COUNT = "photon_count"
    IRF_WIDTH = "irf_width"
    BACKGROUND = "background"
    TEMPORAL_ALIGNMENT = "temporal_alignment"
    DECAY_MODEL = "decay_model"


@dataclass(frozen=True)
class GeneralizationRegime:
    """Definition of one final generalization test regime."""

    test_id: str
    name: str
    shift: GeneralizationShift
    description: str
    scientific_question: str
    changed_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.test_id not in FINAL_ROBUSTNESS_TEST_IDS:
            raise ValueError(
                "test_id must be one of "
                f"{FINAL_ROBUSTNESS_TEST_IDS}, "
                f"got {self.test_id!r}."
            )

        if not self.name.strip():
            raise ValueError(
                "name must not be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "description must not be empty."
            )

        if not self.scientific_question.strip():
            raise ValueError(
                "scientific_question must not be empty."
            )

        if len(set(self.changed_factors)) != len(
            self.changed_factors
        ):
            raise ValueError(
                "changed_factors must not contain duplicates."
            )

        if (
            self.shift
            is GeneralizationShift.IN_DISTRIBUTION
            and self.changed_factors
        ):
            raise ValueError(
                "An in-distribution regime must not list "
                "changed factors."
            )

        if (
            self.shift
            is not GeneralizationShift.IN_DISTRIBUTION
            and not self.changed_factors
        ):
            raise ValueError(
                "A shifted regime must identify at least "
                "one changed factor."
            )


@dataclass(frozen=True)
class GeneralizationProtocol:
    """Scientific protocol for the final robustness test suite."""

    regimes: tuple[GeneralizationRegime, ...]
    development_only_operations: tuple[str, ...]
    development_description: str
    final_suite_description: str

    def __post_init__(self) -> None:
        test_ids = tuple(
            regime.test_id
            for regime in self.regimes
        )

        if test_ids != FINAL_ROBUSTNESS_TEST_IDS:
            raise ValueError(
                "The final robustness protocol must contain "
                "Tests A-F exactly once and in order."
            )

        names = tuple(
            regime.name
            for regime in self.regimes
        )

        if len(set(names)) != len(names):
            raise ValueError(
                "Generalization regime names must be unique."
            )

        if not self.development_only_operations:
            raise ValueError(
                "development_only_operations must not be empty."
            )

    def get_regime(
        self,
        test_id: str,
    ) -> GeneralizationRegime:
        """Return one regime by its A-F identifier."""

        normalized_id = test_id.strip().upper()

        for regime in self.regimes:
            if regime.test_id == normalized_id:
                return regime

        raise KeyError(
            f"Unknown generalization test: {test_id!r}."
        )


def default_generalization_protocol(
) -> GeneralizationProtocol:
    """Return the standard Week 8 generalization protocol."""

    regimes = (
        GeneralizationRegime(
            test_id="A",
            name="Familiar conditions",
            shift=GeneralizationShift.IN_DISTRIBUTION,
            description=(
                "New Poisson realizations drawn from the "
                "familiar lifetime, photon-count, background, "
                "IRF-width, and temporal-alignment "
                "distributions using a mono-exponential "
                "decay model."
            ),
            scientific_question=(
                "How well does the estimator generalize to "
                "new curves drawn from the same underlying "
                "distribution?"
            ),
            changed_factors=(),
        ),
        GeneralizationRegime(
            test_id="B",
            name="Unseen photon-count range",
            shift=GeneralizationShift.PHOTON_COUNT,
            description=(
                "Lifetime physics and instrument conditions "
                "remain familiar while photon counts are moved "
                "outside the training distribution."
            ),
            scientific_question=(
                "Does lifetime estimation remain robust when "
                "photon statistics differ from those seen "
                "during training?"
            ),
            changed_factors=(
                "signal_photon_count",
            ),
        ),
        GeneralizationRegime(
            test_id="C",
            name="IRF-width mismatch",
            shift=GeneralizationShift.IRF_WIDTH,
            description=(
                "Lifetime, photon-count, background, and "
                "temporal-alignment conditions remain familiar "
                "while the instrument response is broadened "
                "beyond the familiar IRF-width distribution."
            ),
            scientific_question=(
                "How dependent is the estimator on the exact "
                "temporal resolution encountered during "
                "training?"
            ),
            changed_factors=(
                "irf_fwhm_ns",
            ),
        ),
        GeneralizationRegime(
            test_id="D",
            name="Elevated background",
            shift=GeneralizationShift.BACKGROUND,
            description=(
                "Lifetime, photon-count, and IRF conditions "
                "remain familiar while background "
                "contamination is increased beyond the "
                "training distribution."
            ),
            scientific_question=(
                "How robust is lifetime estimation when the "
                "decay tail becomes increasingly dominated "
                "by background?"
            ),
            changed_factors=(
                "background_per_bin",
            ),
        ),
        GeneralizationRegime(
            test_id="E",
            name="Temporal misalignment",
            shift=(
                GeneralizationShift.TEMPORAL_ALIGNMENT
            ),
            description=(
                "Decay shape and IRF width remain familiar, "
                "but the temporal registration of the signal "
                "is displaced relative to the training "
                "distribution."
            ),
            scientific_question=(
                "How sensitive is the estimator to imperfect "
                "temporal alignment?"
            ),
            changed_factors=(
                "irf_shift_ns",
            ),
        ),
        GeneralizationRegime(
            test_id="F",
            name="Bi-exponential model mismatch",
            shift=GeneralizationShift.DECAY_MODEL,
            description=(
                "Models trained exclusively on "
                "mono-exponential decays are evaluated on "
                "curves containing a modest secondary "
                "exponential component."
            ),
            scientific_question=(
                "How robust is a mono-exponential lifetime "
                "estimator to modest violations of its decay "
                "model assumption?"
            ),
            changed_factors=(
                "decay_model",
            ),
        ),
    )

    return GeneralizationProtocol(
        regimes=regimes,
        development_only_operations=(
            DEVELOPMENT_ONLY_OPERATIONS
        ),
        development_description=(
            "Development and training data may be used for "
            "model fitting, preprocessing fitting, repeated "
            "cross-validation, feature selection, and "
            "hyperparameter selection."
        ),
        final_suite_description=(
            "Tests A-F form the final untouched robustness "
            "suite and may be used only after all fitting and "
            "model-selection decisions are complete."
        ),
    )


@dataclass(frozen=True)
class FamiliarSimulationDomain:
    """Physical simulation domain used during model development.

    This object defines what "familiar" means for the Week 8
    generalization protocol. Test A is drawn from this same
    physical domain using new Poisson realizations.

    Tests B-E modify exactly one nuisance dimension relative
    to this domain. Test F modifies the decay model.
    """

    time_start_ns: float
    time_stop_ns: float
    time_step_ns: float

    lifetime_values_ns: tuple[float, ...]
    signal_photon_counts: tuple[int, ...]
    background_levels: tuple[float, ...]

    irf_centre_ns: float
    irf_fwhm_values_ns: tuple[float, ...]
    irf_shift_values_ns: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.time_start_ns):
            raise ValueError(
                "time_start_ns must be finite."
            )

        if not isfinite(self.time_stop_ns):
            raise ValueError(
                "time_stop_ns must be finite."
            )

        if self.time_stop_ns <= self.time_start_ns:
            raise ValueError(
                "time_stop_ns must be greater than "
                "time_start_ns."
            )

        if (
            not isfinite(self.time_step_ns)
            or self.time_step_ns <= 0.0
        ):
            raise ValueError(
                "time_step_ns must be finite and positive."
            )

        n_steps = (
            self.time_stop_ns - self.time_start_ns
        ) / self.time_step_ns

        if not isclose(
            n_steps,
            round(n_steps),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "The time range must contain an integer "
                "number of time steps."
            )

        if not (
            self.time_start_ns
            <= self.irf_centre_ns
            < self.time_stop_ns
        ):
            raise ValueError(
                "irf_centre_ns must lie inside the "
                "TCSPC time window."
            )

        self._validate_float_values(
            "lifetime_values_ns",
            self.lifetime_values_ns,
            strictly_positive=True,
        )

        self._validate_count_values(
            self.signal_photon_counts,
        )

        self._validate_float_values(
            "background_levels",
            self.background_levels,
            non_negative=True,
        )

        self._validate_float_values(
            "irf_fwhm_values_ns",
            self.irf_fwhm_values_ns,
            strictly_positive=True,
        )

        self._validate_float_values(
            "irf_shift_values_ns",
            self.irf_shift_values_ns,
        )

    @staticmethod
    def _validate_float_values(
        name: str,
        values: tuple[float, ...],
        *,
        strictly_positive: bool = False,
        non_negative: bool = False,
    ) -> None:
        if not values:
            raise ValueError(
                f"{name} must not be empty."
            )

        if any(
            not isfinite(float(value))
            for value in values
        ):
            raise ValueError(
                f"{name} must contain only finite values."
            )

        if (
            strictly_positive
            and any(value <= 0.0 for value in values)
        ):
            raise ValueError(
                f"{name} must contain only positive values."
            )

        if (
            non_negative
            and any(value < 0.0 for value in values)
        ):
            raise ValueError(
                f"{name} must contain only non-negative "
                "values."
            )

        if len(set(values)) != len(values):
            raise ValueError(
                f"{name} must not contain duplicates."
            )

    @staticmethod
    def _validate_count_values(
        values: tuple[int, ...],
    ) -> None:
        if not values:
            raise ValueError(
                "signal_photon_counts must not be empty."
            )

        if any(value <= 0 for value in values):
            raise ValueError(
                "signal_photon_counts must contain only "
                "positive values."
            )

        if len(set(values)) != len(values):
            raise ValueError(
                "signal_photon_counts must not contain "
                "duplicates."
            )

    @property
    def n_time_bins(self) -> int:
        """Return the number of bins in the TCSPC time axis."""

        return round(
            (
                    self.time_stop_ns
                    - self.time_start_ns
            )
            / self.time_step_ns
        )

DEFAULT_GENERALIZATION_SUITE_VERSION = (
    "week8-day50-v1"
)

DEFAULT_DEVELOPMENT_REFERENCE = (
    "week7-notebook12-benchmark"
)


def default_familiar_simulation_domain(
) -> FamiliarSimulationDomain:
    """Return the canonical Week 7 development domain.

    This domain defines the physical parameter support considered
    familiar by the Week 8 robustness protocol.

    The time stop is exclusive, matching ``numpy.arange`` usage in
    the benchmark workflow.
    """

    return FamiliarSimulationDomain(
        time_start_ns=0.0,
        time_stop_ns=20.0,
        time_step_ns=0.05,

        lifetime_values_ns=(
            1.0,
            2.0,
            3.0,
            4.0,
        ),

        signal_photon_counts=(
            1_000,
            10_000,
        ),

        background_levels=(
            0.5,
            2.0,
        ),

        irf_centre_ns=1.0,

        irf_fwhm_values_ns=(
            0.25,
            0.40,
        ),

        irf_shift_values_ns=(
            -0.05,
            0.05,
        ),
    )


@dataclass(frozen=True)
class GeneralizationNumerics:
    """Numerical definitions of the final Week 8 A-F suite."""

    familiar: FamiliarSimulationDomain

    test_b_low_photon_counts: tuple[int, ...]
    test_b_high_photon_counts: tuple[int, ...]

    test_c_irf_fwhm_values_ns: tuple[float, ...]

    test_d_background_levels: tuple[float, ...]

    test_e_irf_shift_values_ns: tuple[float, ...]

    test_f_secondary_fraction: float
    test_f_secondary_lifetime_factor: float

    test_seeds: tuple[
        int,
        int,
        int,
        int,
        int,
        int,
    ]

    def __post_init__(self) -> None:
        familiar_min_count = min(
            self.familiar.signal_photon_counts
        )
        familiar_max_count = max(
            self.familiar.signal_photon_counts
        )

        if not self.test_b_low_photon_counts:
            raise ValueError(
                "Test B must contain at least one "
                "low-photon OOD value."
            )

        if any(
            count <= 0
            for count in self.test_b_low_photon_counts
        ):
            raise ValueError(
                "Test B photon counts must be positive."
            )

        if any(
            count >= familiar_min_count
            for count in self.test_b_low_photon_counts
        ):
            raise ValueError(
                "All low-photon Test B values must lie "
                "below the familiar photon-count range."
            )

        if not self.test_b_high_photon_counts:
            raise ValueError(
                "Test B must contain at least one "
                "high-photon OOD value."
            )

        if any(
            count <= familiar_max_count
            for count in self.test_b_high_photon_counts
        ):
            raise ValueError(
                "All high-photon Test B values must lie "
                "above the familiar photon-count range."
            )

        familiar_max_fwhm = max(
            self.familiar.irf_fwhm_values_ns
        )

        if not self.test_c_irf_fwhm_values_ns:
            raise ValueError(
                "Test C must contain at least one "
                "OOD IRF width."
            )

        if any(
            (
                not isfinite(value)
                or value <= familiar_max_fwhm
            )
            for value
            in self.test_c_irf_fwhm_values_ns
        ):
            raise ValueError(
                "Test C IRF widths must be finite and "
                "broader than the familiar IRF range."
            )

        familiar_max_background = max(
            self.familiar.background_levels
        )

        if not self.test_d_background_levels:
            raise ValueError(
                "Test D must contain at least one "
                "elevated background level."
            )

        if any(
            (
                not isfinite(value)
                or value <= familiar_max_background
            )
            for value
            in self.test_d_background_levels
        ):
            raise ValueError(
                "Test D background levels must be finite "
                "and above the familiar background range."
            )

        if not self.test_e_irf_shift_values_ns:
            raise ValueError(
                "Test E must contain temporal shifts."
            )

        familiar_min_shift = min(
            self.familiar.irf_shift_values_ns
        )
        familiar_max_shift = max(
            self.familiar.irf_shift_values_ns
        )

        if any(
            familiar_min_shift
            <= shift
            <= familiar_max_shift
            for shift in self.test_e_irf_shift_values_ns
        ):
            raise ValueError(
                "All Test E shifts must lie outside the "
                "familiar temporal-alignment range."
            )

        if not any(
            shift < familiar_min_shift
            for shift in self.test_e_irf_shift_values_ns
        ):
            raise ValueError(
                "Test E must include a negative-side "
                "temporal misalignment."
            )

        if not any(
            shift > familiar_max_shift
            for shift in self.test_e_irf_shift_values_ns
        ):
            raise ValueError(
                "Test E must include a positive-side "
                "temporal misalignment."
            )

        for shift in self.test_e_irf_shift_values_ns:
            shift_bins = (
                shift / self.familiar.time_step_ns
            )

            if not isclose(
                shift_bins,
                round(shift_bins),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Test E shifts must correspond to an "
                    "integer number of histogram bins."
                )

        if not (
            0.0
            < self.test_f_secondary_fraction
            <= 0.25
        ):
            raise ValueError(
                "Test F secondary fraction must be in "
                "the interval (0, 0.25]."
            )

        if (
            not isfinite(
                self.test_f_secondary_lifetime_factor
            )
            or self.test_f_secondary_lifetime_factor
            <= 1.0
        ):
            raise ValueError(
                "Test F secondary lifetime factor must "
                "be finite and greater than 1."
            )

        if len(self.test_seeds) != len(
            FINAL_ROBUSTNESS_TEST_IDS
        ):
            raise ValueError(
                "Exactly one random seed is required for "
                "each of Tests A-F."
            )

        if any(seed < 0 for seed in self.test_seeds):
            raise ValueError(
                "Test seeds must be non-negative."
            )

        if len(set(self.test_seeds)) != len(
            self.test_seeds
        ):
            raise ValueError(
                "Tests A-F must use distinct random seeds."
            )

    @property
    def test_e_shift_bins(
        self,
    ) -> tuple[int, ...]:
        """Return Test E temporal offsets in histogram bins."""

        return tuple(
            round(
                shift
                / self.familiar.time_step_ns
            )
            for shift
            in self.test_e_irf_shift_values_ns
        )

    def seed_for(
        self,
        test_id: str,
    ) -> int:
        """Return the reserved random seed for one A-F test."""

        normalized_id = test_id.strip().upper()

        try:
            index = FINAL_ROBUSTNESS_TEST_IDS.index(
                normalized_id
            )
        except ValueError as exc:
            raise KeyError(
                f"Unknown generalization test: "
                f"{test_id!r}."
            ) from exc

        return self.test_seeds[index]


@dataclass(frozen=True)
class GeneralizationSuiteDefinition:
    """Complete definition of the final Week 8 robustness suite.

    The suite binds together:

    - the scientific A-F protocol;
    - the familiar Week 7 development domain;
    - the frozen numerical OOD definitions;
    - reproducibility and provenance information.

    No simulated measurements are stored here.
    """

    suite_version: str
    development_reference: str
    protocol: GeneralizationProtocol
    numerics: GeneralizationNumerics

    def __post_init__(self) -> None:
        if not self.suite_version.strip():
            raise ValueError(
                "suite_version must not be empty."
            )

        if not self.development_reference.strip():
            raise ValueError(
                "development_reference must not be empty."
            )

        expected_changed_factors = {
            "A": (),
            "B": ("signal_photon_count",),
            "C": ("irf_fwhm_ns",),
            "D": ("background_per_bin",),
            "E": ("irf_shift_ns",),
            "F": ("decay_model",),
        }

        for test_id, expected in (
            expected_changed_factors.items()
        ):
            regime = self.protocol.get_regime(
                test_id
            )

            if regime.changed_factors != expected:
                raise ValueError(
                    f"Test {test_id} changed_factors "
                    "are inconsistent with the numerical "
                    "generalization suite."
                )

    @property
    def familiar(
        self,
    ) -> FamiliarSimulationDomain:
        """Return the familiar Week 7 simulation domain."""

        return self.numerics.familiar

    def regime(
        self,
        test_id: str,
    ) -> GeneralizationRegime:
        """Return one scientific A-F regime definition."""

        return self.protocol.get_regime(
            test_id
        )

    def seed_for(
        self,
        test_id: str,
    ) -> int:
        """Return the reserved random seed for one A-F test."""

        return self.numerics.seed_for(
            test_id
        )


def default_generalization_numerics(
    familiar: FamiliarSimulationDomain,
) -> GeneralizationNumerics:
    """Return the frozen numerical definitions for Week 8."""

    return GeneralizationNumerics(
        familiar=familiar,

        test_b_low_photon_counts=(
            250,
            500,
        ),
        test_b_high_photon_counts=(
            50_000,
        ),

        test_c_irf_fwhm_values_ns=(
            0.60,
        ),

        test_d_background_levels=(
            5.0,
        ),

        test_e_irf_shift_values_ns=(
            -0.15,
            0.15,
        ),

        test_f_secondary_fraction=0.10,
        test_f_secondary_lifetime_factor=2.0,

        test_seeds=(
            50_001,
            50_002,
            50_003,
            50_004,
            50_005,
            50_006,
        ),
    )


def default_generalization_suite(
) -> GeneralizationSuiteDefinition:
    """Return the canonical Week 8 A-F robustness suite."""

    familiar = (
        default_familiar_simulation_domain()
    )

    return GeneralizationSuiteDefinition(
        suite_version=(
            DEFAULT_GENERALIZATION_SUITE_VERSION
        ),
        development_reference=(
            DEFAULT_DEVELOPMENT_REFERENCE
        ),
        protocol=(
            default_generalization_protocol()
        ),
        numerics=(
            default_generalization_numerics(
                familiar
            )
        ),
    )
