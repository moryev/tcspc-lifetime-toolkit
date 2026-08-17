"""Public package interface for the TCSPC Lifetime Toolkit."""

from tcspc_toolkit.models import (
    monoexponential_decay,
)
from tcspc_toolkit.simulation import (
    sample_photon_counts,
)

from tcspc_toolkit.datasets import (
    SyntheticDataset,
    generate_monoexponential_dataset,
)

from tcspc_toolkit.irf import (
    generate_gaussian_irf,
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
    InvalidHistogramError,
    TCSPCError,
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
    PreprocessingConfig,
    SimulationConfig,
    load_config,
    save_config,
)

__all__ = [
    "monoexponential_decay",
    "sample_photon_counts",
    "SyntheticDataset",
    "generate_monoexponential_dataset",
    "generate_gaussian_irf",
    "normalize_irf",
    "shift_irf",
    "convolve_decay_with_irf",
    "LifetimeFitResult",
    "ReconvolutionFitResult",
    "fit_monoexponential_decay",
    "fit_monoexponential_reconvolution",
    "TCSPCError",
    "InvalidHistogramError",
    "align_to_irf",
    "crop_time_window",
    "detect_peak",
    "estimate_background",
    "normalize_counts",
    "rebin_histogram",
    "subtract_background",
    "validate_histogram",
    "CountNormalization",
    "PreprocessingConfig",
    "SimulationConfig",
    "load_config",
    "save_config",
]