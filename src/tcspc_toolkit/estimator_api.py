"""Compatibility imports for :mod:`tcspc_toolkit.estimation.api`.

The flat path remains supported through at least Issue #12. Objects are direct
re-exports of the canonical implementation, with no wrappers or warnings.
"""

from tcspc_toolkit.estimation.api import (
    __all__,
    RegressorProtocol,
    EstimatorSpec,
    fit_regressors,
    predict_regressors,
)
