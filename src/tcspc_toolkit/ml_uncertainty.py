"""Method definitions for machine-learning uncertainty estimation."""

from tcspc_toolkit.uncertainty_evaluation import (
    UncertaintyMethodDefinition,
    UncertaintyOutputKind,
)


ML_UNCERTAINTY_METHODS = {
    "quantile_gradient_boosting": UncertaintyMethodDefinition(
        method_id="quantile_gradient_boosting",
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=(
            "Conditional predictive quantiles learned from the uncertainty-training distribution."
        ),
        calibration_scope=(
            "Nominal coverage is not guaranteed under finite-sample or distribution shift and must be measured on held-out data."
        ),
    ),
    "random_forest_tree_spread": UncertaintyMethodDefinition(
        method_id="random_forest_tree_spread",
        output_kind=UncertaintyOutputKind.UNCERTAINTY_SCORE,
        interpretation=(
            "Tree-to-tree ensemble disagreement used as an uncertainty ranking heuristic."
        ),
        calibration_scope=(
            "No nominal prediction-interval coverage claim is made unless a separate calibration procedure is applied."
        ),
    ),
    "ml_training_bootstrap": UncertaintyMethodDefinition(
        method_id="ml_training_bootstrap",
        output_kind=UncertaintyOutputKind.UNCERTAINTY_SCORE,
        interpretation=(
            "Sensitivity of predictions to uncertainty-training sample variation."
        ),
        calibration_scope=(
            "Bootstrap spread is not automatically a calibrated predictive interval for lifetime error."
        ),
    ),
    "split_conformal": UncertaintyMethodDefinition(
        method_id="split_conformal",
        output_kind=UncertaintyOutputKind.PREDICTION_INTERVAL,
        interpretation=(
            "Residual-based marginal prediction intervals calibrated on a dedicated development calibration subset."
        ),
        calibration_scope=(
            "Finite-sample marginal coverage is targeted under exchangeability; coverage can degrade under distribution shift."
        ),
    ),
}
