"""Method definitions for classical lifetime-fit uncertainty analysis."""

from tcspc_toolkit.uncertainty_evaluation import (
    UncertaintyMethodDefinition,
    UncertaintyOutputKind,
)


CLASSICAL_UNCERTAINTY_METHODS = {
    "covariance": UncertaintyMethodDefinition(
        method_id="covariance",
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=(
            "Local fitted-parameter uncertainty derived from the covariance approximation conditional on the assumed decay model."
        ),
        calibration_scope=(
            "Meaningful only to the extent that the fitted model, local approximation, and sampling assumptions are adequate."
        ),
    ),
    "parametric_bootstrap": UncertaintyMethodDefinition(
        method_id="parametric_bootstrap",
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=(
            "Sampling variability obtained by resimulating measurements from the fitted model and refitting them."
        ),
        calibration_scope=(
            "Conditional on the fitted physical model; it does not capture model misspecification such as Test F."
        ),
    ),
    "repeated_poisson_realizations": UncertaintyMethodDefinition(
        method_id="repeated_poisson_realizations",
        output_kind=UncertaintyOutputKind.EMPIRICAL_REFERENCE,
        interpretation=(
            "Empirical estimator variability over repeated Poisson measurements generated from known physics."
        ),
        calibration_scope=(
            "Reference benchmark for repeated-measurement variability, not a per-curve uncertainty estimate available in deployment."
        ),
    ),
}
