"""Validate draft arithmetic and isolation without generating observations."""

import json
from pathlib import Path

from tcspc_toolkit.estimation.triexponential import TriexponentialFitSettings


def test_nonexecuting_manifest_population_pairing_and_review_gates():
    path = (
        Path(__file__).resolve().parents[1] / "configs/issue37_pr2_development_v1.json"
    )
    m = json.loads(path.read_text())
    assert m["execution_authorized"] is False
    rows, common = m["conditions"], m["defaults"]
    assert [r["id"] for r in rows] == [f"D{i:02d}" for i in range(18)]
    problems = sum(len(r.get("assumed_irfs", common["assumed_irfs"])) * 4 for r in rows)
    assert problems == m["population"]["fitting_problems"] == 80
    assert 18 * 4 == m["population"]["distinct_targets"] == 72
    assert problems * 6 + problems == m["population"]["total_tasks"] == 560
    assert len(m["evaluation_rules"]["control_ids"]) * 4 == 8
    assert (
        m["construction"]["observation_namespace"]
        != m["operational_pilot"]["namespace"]
    )
    assert not m["operational_pilot"]["execution_authorized"]
    assert not m["independent_audit_candidate"]["implemented"]
    for name, row in m["candidate_policies"].items():
        s = TriexponentialFitSettings.development(name)
        assert s.max_seed_starts == row["seed_starts"]
        assert s.total_start_limit == row["total_starts"]
        assert s.max_optimization_calls == row["optimization_model_calls"]
        assert s.wall_seconds == row["wall_seconds"]
    ceiling = (
        80 * 2 * sum(r["wall_seconds"] for r in m["candidate_policies"].values())
        + 80 * 900
    )
    assert ceiling == m["comprehensive_ceiling_worker_seconds"]


def test_declared_observation_identities_pair_assumptions_without_seed_changes():
    path = (
        Path(__file__).resolve().parents[1] / "configs/issue37_pr2_development_v1.json"
    )
    manifest = json.loads(path.read_text())
    defaults = manifest["defaults"]
    namespace = tuple(manifest["construction"]["observation_namespace"])
    identities, problems, streams = {}, {}, set()
    for number, condition in enumerate(manifest["conditions"]):
        resolved = defaults | condition
        assert resolved["deterministic_targets"] == 1
        assert resolved["poisson_replicates"] == 3
        for label in ("mean", "poisson-0", "poisson-1", "poisson-2"):
            observation = f"{condition['id']}:{label}"
            seed = None if label == "mean" else (*namespace, number, int(label[-1]))
            assert observation not in identities
            identities[observation] = seed
            if seed is not None:
                assert seed not in streams
                streams.add(seed)
            for assumed in resolved["assumed_irfs"]:
                problem = f"{observation}:{assumed}"
                assert problem not in problems
                problems[problem] = (observation, seed)
    assert len(identities) == 72 and len(problems) == 80 and len(streams) == 54
    controls = set(manifest["evaluation_rules"]["control_ids"])
    assert sum(p.split(":")[0] in controls for p in problems) == 8
    assert sum(p.split(":")[0] not in controls for p in problems) == 72
    for condition in ("D15", "D16"):
        for label in ("mean", "poisson-0", "poisson-1", "poisson-2"):
            assert (
                problems[f"{condition}:{label}:matched"]
                == problems[f"{condition}:{label}:gaussian"]
            )
    pilot_namespace = tuple(manifest["operational_pilot"]["namespace"])
    assert all(seed[: len(pilot_namespace)] != pilot_namespace for seed in streams)
