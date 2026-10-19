"""Supported symbol aliases and intentional removal of old pickle globals."""

import ast
import importlib
import inspect
import json
import pickle
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from tcspc_toolkit.benchmarks.uncertainty import (
    DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED,
    DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED,
    evaluate_classical_uncertainty_robustness,
)


# Deliberate public contract, independent of implementation alias discovery.
RENAMED_APIS = (
    ("benchmarks.generalization", "Day53ABReport", "PhotonCountShiftReport"),
    ("benchmarks.generalization", "build_day53_ab_report", "build_photon_count_shift_report"),
    ("benchmarks.generalization", "Day55ModelMismatchReport", "DecayModelMismatchReport"),
    ("benchmarks.generalization", "build_day55_model_mismatch_report", "build_decay_model_mismatch_report"),
    ("benchmarks.generalization", "build_day55_severity_comparison", "build_model_mismatch_severity_comparison"),
    ("benchmarks.generalization", "Week8RobustnessReport", "GeneralizationRobustnessReport"),
    ("benchmarks.generalization", "build_week8_robustness_report", "build_generalization_robustness_report"),
    ("uncertainty.ml", "evaluate_week9_quantile_robustness_conditions", "evaluate_quantile_interval_robustness"),
    ("uncertainty.ml", "evaluate_week9_paired_quantile_response", "evaluate_paired_quantile_interval_response"),
    ("uncertainty.ml", "evaluate_week9_paired_ml_uncertainty_response", "evaluate_paired_uncertainty_score_response"),
    ("benchmarks.uncertainty", "DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED", "DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED"),
    ("benchmarks.uncertainty", "build_week9_interval_scorecard", "build_ml_interval_scorecard"),
    ("benchmarks.uncertainty", "build_week9_score_only_scorecard", "build_ml_uncertainty_scorecard"),
    ("benchmarks.uncertainty", "Week9MLUncertaintyRobustnessReport", "MLUncertaintyRobustnessReport"),
    ("benchmarks.uncertainty", "build_week9_ml_uncertainty_robustness_report", "build_ml_uncertainty_robustness_report"),
    ("benchmarks.uncertainty", "build_week9_classical_uncertainty_scorecard", "build_classical_uncertainty_scorecard"),
    ("benchmarks.uncertainty", "build_week9_classical_conditional_scorecard", "build_classical_conditional_uncertainty_scorecard"),
    ("benchmarks.uncertainty", "Week9ClassicalUncertaintyRobustnessReport", "ClassicalUncertaintyRobustnessReport"),
    ("benchmarks.uncertainty", "evaluate_week9_classical_uncertainty_robustness", "evaluate_classical_uncertainty_robustness"),
    ("benchmarks.uncertainty", "Week9ResidualMismatchReport", "ResidualMismatchReport"),
    ("benchmarks.uncertainty", "evaluate_week9_residual_mismatch_diagnostics", "evaluate_residual_mismatch_diagnostics"),
    ("persistence", "record_week9_interval_scorecard_row", "record_ml_interval_scorecard_row"),
    ("persistence", "record_week9_score_only_scorecard_row", "record_ml_uncertainty_scorecard_row"),
)


@pytest.mark.parametrize("module_name, legacy_name, canonical_name", RENAMED_APIS)
def test_legacy_name_is_the_canonical_implementation(module_name, legacy_name, canonical_name):
    module = importlib.import_module(f"tcspc_toolkit.{module_name}")
    canonical = getattr(module, canonical_name)
    assert getattr(module, legacy_name) is canonical
    if callable(canonical):
        assert canonical.__name__ == canonical_name


@pytest.mark.parametrize(
    "module_name, legacy_name, canonical_name",
    [row for row in RENAMED_APIS if row[1].endswith("Report")],
)
def test_report_fields_and_legacy_pickle_references(module_name, legacy_name, canonical_name):
    module = importlib.import_module(f"tcspc_toolkit.{module_name}")
    canonical = getattr(module, canonical_name)
    scalar_fields = {
        "conformal_correction_ns": 0.12,
        "nominal_coverage": 0.9,
        "n_bootstrap_resamples": 200,
    }
    values = {
        field.name: scalar_fields.get(field.name, pd.DataFrame({"value": [1.0, 2.0]}))
        for field in fields(canonical)
    }
    report = getattr(module, legacy_name)(**values)
    assert type(report) is canonical
    assert canonical.__dataclass_params__.frozen

    current_payload = pickle.dumps(report, protocol=0)
    # Protocol 0 encodes global names without length/frame prefixes. Substitute
    # the old module-global class reference to exercise historical unpickling.
    reference = f"c{canonical.__module__}\n{canonical_name}\n".encode()
    assert current_payload.count(reference) == 1
    removed_module = {
        "benchmarks.generalization": "generalization_evaluation",
        "benchmarks.uncertainty": "uncertainty_robustness",
    }[module_name]
    for name in (canonical_name, legacy_name):
        historical = current_payload.replace(
            reference, f"ctcspc_toolkit.{removed_module}\n{name}\n".encode(),
        )
        with pytest.raises(ModuleNotFoundError) as error:
            pickle.loads(historical)
        assert error.value.name == f"tcspc_toolkit.{removed_module}"
    # Both supported names resolve in the canonical module.
    canonical_alias_payload = current_payload.replace(
        reference, f"c{canonical.__module__}\n{legacy_name}\n".encode(),
    )
    for payload in (current_payload, canonical_alias_payload):
        restored = pickle.loads(payload)
        assert type(restored) is canonical
        assert vars(restored).keys() == values.keys()
        for name, expected in values.items():
            actual = getattr(restored, name)
            if isinstance(expected, pd.DataFrame):
                pd.testing.assert_frame_equal(actual, expected)
            else:
                assert actual == expected


def test_classical_bootstrap_seed_and_default_are_unchanged():
    assert DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED == DEFAULT_DAY62_CLASSICAL_BOOTSTRAP_SEED == 62_001
    parameter = inspect.signature(evaluate_classical_uncertainty_robustness).parameters[
        "bootstrap_random_seed"
    ]
    assert parameter.default == DEFAULT_CLASSICAL_ROBUSTNESS_BOOTSTRAP_SEED


@pytest.mark.parametrize("name", [
    "13_generalization_and_robustness.ipynb",
    "14_uncertainty_and_failure_awareness.ipynb",
])
def test_notebook_code_uses_resolvable_canonical_imports(name):
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    legacy_names = {old for _, old, _ in RENAMED_APIS}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in legacy_names
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tcspc_toolkit"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert alias.name not in legacy_names
                    assert hasattr(module, alias.name)
