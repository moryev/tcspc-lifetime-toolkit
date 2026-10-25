"""Keep policy selection separate from genuinely held-out observations."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment():
    return runpy.run_path(
        str(ROOT / "scripts/benchmark_biexponential_initialization.py")
    )


def synthetic_development_rows(experiment):
    protocol = experiment["load_protocol"]()
    rows = [
        dict(
            kind="metadata",
            mode="development",
            config_sha256=experiment["digest"](experiment["CONFIG"]),
            implementation_sha256="test-fixture",
        )
    ]
    for condition in protocol["conditions"]:
        for stream in protocol["development_streams"]:
            for index, policy in enumerate(protocol["policies"]):
                rows.append(
                    dict(
                        kind="fit",
                        mode="development",
                        condition_id=condition["id"],
                        stream=stream,
                        policy=policy,
                        settings=asdict(experiment["settings_for"](protocol, policy)),
                        centered_nll=10 + index * 0.001,
                        optimization=dict(
                            numerical_validation_passed=True,
                            exhausted_budgets=[],
                            total_seconds=1 + index,
                            starts=[{}] * (4 + index),
                            initialization_evaluations=92,
                            objective_evaluations=1200 + index * 1000,
                            validation_evaluations=49,
                            diagnostic_evaluations=30,
                        ),
                    )
                )
    return protocol, rows


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_selection_uses_development_quality_then_cost(experiment, tmp_path):
    protocol, rows = synthetic_development_rows(experiment)
    source, output = tmp_path / "development.jsonl", tmp_path / "selection.json"
    save_rows(source, rows)
    experiment["select_policy"](protocol, source, output)
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["selected_settings"]["policy"] == "economical"
    assert selection["selected_settings"]["max_objective_evaluations"] == 2000
    assert selection["selected_settings"]["max_starts"] == 6
    assert {r["stream"] for r in selection["development_references"]} == {0, 1}
    with pytest.raises(FileExistsError):
        experiment["select_policy"](protocol, source, output)


@pytest.mark.parametrize(
    "defect",
    ["held_out_stream", "held_out_mode", "missing", "duplicate", "different_config"],
)
def test_selection_rejects_contamination_and_incomplete_inputs(
    experiment, tmp_path, defect
):
    protocol, original = synthetic_development_rows(experiment)
    rows = deepcopy(original)
    if defect == "held_out_stream":
        rows[1]["stream"] = 2
    elif defect == "held_out_mode":
        rows[1]["mode"] = "held-out"
    elif defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[1])
    else:
        rows[0]["config_sha256"] = "wrong"
    source = tmp_path / "rows.jsonl"
    save_rows(source, rows)
    with pytest.raises(ValueError):
        experiment["select_policy"](protocol, source, tmp_path / "selection.json")


def test_selection_never_relaxes_quality_to_pick_fastest(experiment, tmp_path):
    protocol, rows = synthetic_development_rows(experiment)
    rows[1]["centered_nll"] = 11
    source = tmp_path / "rows.jsonl"
    save_rows(source, rows)
    output = tmp_path / "selection.json"
    experiment["select_policy"](protocol, source, output)
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["selected_settings"]["policy"] == "adaptive"
    assert selected["summaries"][0]["eligible"] is False
    assert len(selected["summaries"][0]["failures"]) == 1


@pytest.mark.parametrize("objective", [10.0, None])
def test_all_failed_policies_report_without_a_default(experiment, tmp_path, objective):
    protocol, rows = synthetic_development_rows(experiment)
    for row in rows[1:]:
        row["optimization"]["numerical_validation_passed"] = False
        row["centered_nll"] = objective
    source, output = tmp_path / "rows.jsonl", tmp_path / "selection.json"
    save_rows(source, rows)
    experiment["select_policy"](protocol, source, output)
    assert json.loads(output.read_text(encoding="utf-8"))["selected_settings"] is None


def test_representative_independent_optimization_audit(experiment):
    report = experiment["independent_audit"](experiment["load_protocol"](), 0)
    assert report["passed"], report
    assert report["absolute_difference"] <= 0.01
