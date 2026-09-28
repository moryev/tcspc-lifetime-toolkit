"""Public package interface for the TCSPC Lifetime Toolkit."""

from tcspc_toolkit.models import (
    monoexponential_decay,
)
from tcspc_toolkit.simulation import (
    build_expected_counts_from_irf,
    sample_photon_counts,
)

from tcspc_toolkit.datasets import (
    SyntheticDataset,
    generate_monoexponential_dataset,
)

from tcspc_toolkit.irf import (
    IRFProfile,
    IRFSourceKind,
    generate_emg_irf,
    generate_emg_irf_profile,
    generate_gaussian_irf,
    generate_gaussian_irf_profile,
    normalize_irf,
    shift_irf,
)

from tcspc_toolkit.convolution import (
    convolve_decay_with_irf,
)

from tcspc_toolkit.fitting import (
    LifetimeFitResult,
    ReconvolutionFitResult,
    fit_monoexponential_decay,
    fit_monoexponential_reconvolution,
)

from tcspc_toolkit.exceptions import (
    FeatureExtractionError,
    InvalidHistogramError,
    InvalidMeasurementError,
    TCSPCError,
)

from tcspc_toolkit.measurements import (
    MeasurementDataKind,
    SampledIRF,
    TCSPCMeasurement,
    TimeUnit,
    convert_time_to_ns,
)

from tcspc_toolkit.irf_preparation import (
    IRFPreparationDiagnostics,
    PreparedIRF,
    irf_profile_from_sampled_irf,
    prepare_irf,
)

from tcspc_toolkit.measurement_io import (
    load_sampled_irf_csv,
    load_tcspc_measurement_csv,
)

from tcspc_toolkit.experimental import (
    ReferenceLifetimeEvaluation,
    evaluate_estimate_against_reference,
    fit_experimental_reconvolution,
    measurement_to_feature_table,
    measurement_to_histogram_batch,
)

from tcspc_toolkit.preprocessing import (
    align_to_irf,
    crop_time_window,
    detect_peak,
    estimate_background,
    normalize_counts,
    rebin_histogram,
    subtract_background,
    validate_histogram,
)

from tcspc_toolkit.config import (
    CountNormalization,
    FeatureConfig,
    PreprocessingConfig,
    SimulationConfig,
    load_config,
    save_config,
)

from tcspc_toolkit.features import (
    FEATURE_NAMES,
    extract_feature_table,
    extract_features,
)

from tcspc_toolkit.representations import (
    cumulative_explained_variance,
    fit_pca_representation,
    normalize_histogram_batch,
    transform_pca_representation,
)

__all__ = [
    "monoexponential_decay",
    "sample_photon_counts",
    "SyntheticDataset",
    "generate_monoexponential_dataset",
    "generate_gaussian_irf",
    "generate_emg_irf",
    "generate_gaussian_irf_profile",
    "generate_emg_irf_profile",
    "IRFSourceKind",
    "IRFProfile",
    "IRFPreparationDiagnostics",
    "PreparedIRF",
    "irf_profile_from_sampled_irf",
    "prepare_irf",
    "normalize_irf",
    "shift_irf",
    "convolve_decay_with_irf",
    "LifetimeFitResult",
    "ReconvolutionFitResult",
    "fit_monoexponential_decay",
    "fit_monoexponential_reconvolution",
    "TCSPCError",
    "InvalidHistogramError",
    "InvalidMeasurementError",
    "FeatureExtractionError",
    "TimeUnit",
    "MeasurementDataKind",
    "SampledIRF",
    "TCSPCMeasurement",
    "convert_time_to_ns",
    "load_sampled_irf_csv",
    "load_tcspc_measurement_csv",
    "ReferenceLifetimeEvaluation",
    "fit_experimental_reconvolution",
    "measurement_to_feature_table",
    "measurement_to_histogram_batch",
    "evaluate_estimate_against_reference",
    "align_to_irf",
    "crop_time_window",
    "detect_peak",
    "estimate_background",
    "normalize_counts",
    "rebin_histogram",
    "subtract_background",
    "validate_histogram",
    "CountNormalization",
    "FeatureConfig",
    "PreprocessingConfig",
    "SimulationConfig",
    "load_config",
    "save_config",
    "FEATURE_NAMES",
    "extract_feature_table",
    "extract_features",
    "cumulative_explained_variance",
    "fit_pca_representation",
    "normalize_histogram_batch",
    "transform_pca_representation",
]
