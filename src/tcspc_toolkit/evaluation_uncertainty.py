"""Compatibility imports for :mod:`tcspc_toolkit.evaluation.attachments`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.evaluation.attachments import (
    __all__,
    IntervalAttachment,
    ScoreAttachment,
)
