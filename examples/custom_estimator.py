"""Generic lifetime regression using prepared toy features, not a TCSPC benchmark.

From an installed checkout: python examples/custom_estimator.py
No frozen A-F data, representation fitting, or benchmark metrics are involved.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from tcspc_toolkit import EstimatorSpec, fit_regressors, predict_regressors


def run_example() -> NDArray[np.float64]:
    """Fit a user estimator and return predictions in test-row order."""
    # Representation construction is separate from estimator execution. In a
    # real workflow, learn any transforms on training data only, then reuse
    # them on test data. Here the feature matrices are already prepared.
    X_train = {"my-features": np.array([[0.0], [1.0], [2.0]])}
    X_test = {"my-features": np.array([[0.5], [1.5]])}
    y_train_ns = [1.0, 2.0, 3.0]
    # The caller owns matching feature meanings and positional sample order:
    # row i in every training representation corresponds to y_train_ns[i].
    # LinearRegression is a zero-argument factory, not a shared fitted object;
    # each selected estimator/representation pair receives fresh model state.
    specs = (EstimatorSpec("my-linear-model", LinearRegression, ("my-features",)),)
    fitted = fit_regressors(
        estimator_specs=specs,
        X_train_by_representation=X_train,
        y_train=y_train_ns,
    )
    predictions = predict_regressors(
        fitted_estimators=fitted,
        X_by_representation=X_test,
    )
    return predictions["my-linear-model"]["my-features"]


if __name__ == "__main__":
    prediction = run_example()
    print("Predicted lifetimes (ns):", prediction)
    print("Array shape:", prediction.shape, "dtype:", prediction.dtype)
