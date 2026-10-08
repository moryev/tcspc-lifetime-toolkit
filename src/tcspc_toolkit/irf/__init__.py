"""Compatibility facade for the historical profile-level IRF API.

Direct exports remain supported through at least Issue #12. Preparation and
leading-edge estimation are explicit submodules, not eager exports here.
"""

from tcspc_toolkit.irf.profiles import (
    IRFSourceKind,
    IRFProfile,
    generate_gaussian_irf,
    generate_emg_irf,
    generate_gaussian_irf_profile,
    generate_emg_irf_profile,
    normalize_irf,
    shift_irf,
)

__all__ = [
    "IRFSourceKind",
    "IRFProfile",
    "generate_gaussian_irf",
    "generate_emg_irf",
    "generate_gaussian_irf_profile",
    "generate_emg_irf_profile",
    "normalize_irf",
    "shift_irf",
]
