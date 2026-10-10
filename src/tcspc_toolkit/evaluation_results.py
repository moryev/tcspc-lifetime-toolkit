"""Compatibility imports for :mod:`tcspc_toolkit.evaluation.results`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.evaluation.results import (
    __all__,
    MethodDescriptor,
    LifetimeReference,
    EvaluationBatch,
    PointEvaluationResult,
)
