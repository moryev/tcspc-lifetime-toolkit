from tcspc_toolkit.classical_uncertainty import (
    CLASSICAL_UNCERTAINTY_METHODS,
)
from tcspc_toolkit.ml_uncertainty import (
    ML_UNCERTAINTY_METHODS,
)
from tcspc_toolkit.uncertainty_evaluation import (
    UncertaintyOutputKind,
)


def test_random_forest_tree_spread_is_not_labelled_as_prediction_interval() -> None:
    definition = ML_UNCERTAINTY_METHODS[
        "random_forest_tree_spread"
    ]

    assert (
        definition.output_kind
        is UncertaintyOutputKind.UNCERTAINTY_SCORE
    )


def test_ml_training_bootstrap_is_not_labelled_as_calibrated_interval() -> None:
    definition = ML_UNCERTAINTY_METHODS[
        "ml_training_bootstrap"
    ]

    assert (
        definition.output_kind
        is UncertaintyOutputKind.UNCERTAINTY_SCORE
    )


def test_repeated_poisson_realizations_are_empirical_reference() -> None:
    definition = CLASSICAL_UNCERTAINTY_METHODS[
        "repeated_poisson_realizations"
    ]

    assert (
        definition.output_kind
        is UncertaintyOutputKind.EMPIRICAL_REFERENCE
    )


def test_classical_parametric_bootstrap_states_model_conditionality() -> None:
    definition = CLASSICAL_UNCERTAINTY_METHODS[
        "parametric_bootstrap"
    ]

    assert (
        "Conditional on the fitted physical model"
        in definition.calibration_scope
    )
