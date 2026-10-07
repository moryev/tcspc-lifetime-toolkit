"""Compatibility imports for :mod:`tcspc_toolkit.physics.models`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.physics.models import (
    monoexponential_decay,
    biexponential_decay,
    multiexponential_decay,
)

__all__ = ["monoexponential_decay", "biexponential_decay", "multiexponential_decay"]
