"""Compatibility imports for :mod:`tcspc_toolkit.physics.convolution`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.physics.convolution import convolve_decay_with_irf

__all__ = ["convolve_decay_with_irf"]
