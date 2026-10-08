"""Historical histogram-simulation API; dataset generation remains a submodule."""

from tcspc_toolkit.simulation.histograms import (
    sample_photon_counts,
    build_expected_counts_from_irf,
    simulate_irf_convolved_histogram,
    simulate_irf_convolved_biexponential_histogram,
    simulate_monoexponential_decay,
    simulate_biexponential_decay,
    simulate_multiexponential_decay,
)

__all__ = [
    "sample_photon_counts",
    "build_expected_counts_from_irf",
    "simulate_irf_convolved_histogram",
    "simulate_irf_convolved_biexponential_histogram",
    "simulate_monoexponential_decay",
    "simulate_biexponential_decay",
    "simulate_multiexponential_decay",
]
