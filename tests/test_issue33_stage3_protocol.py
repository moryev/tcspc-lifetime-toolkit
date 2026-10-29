"""Protocol review checks: identities and metadata only, no scientific draws."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_issue33_stage3_protocol.py"
API = runpy.run_path(str(SCRIPT))


@pytest.fixture
def protocol():
    return json.loads(
        (ROOT / "configs/issue33_stage3_qualification_v1.json").read_text()
    )


def records(protocol):
    return [
        dict(condition_id=c, replicate=r, arm=a, classification="agreement")
        for c, r, a in API["population"](protocol)[2]
    ]


def test_population_totals_pairing_and_seed_separation(protocol):
    assert API["validate"](protocol) == protocol["expected_totals"]
    observations, production, audits = API["population"](protocol)
    assert (len(observations), len(production), len(audits)) == (200, 210, 133)
    assert sum(k[0] == 0 for k in audits) == 100
    assert sum(k[0] == 26 for k in observations) == 10
    assert sum(k[0] == 26 for k in production) == 20
    assert sum(k[0] == 26 for k in audits) == 6
    entropy = {API["observation_entropy"](protocol, k) for k in observations}
    assert len(entropy) == 200
    assert all(e[:4] == (33, 3, 20261029, 1) and e[4] == 0 for e in entropy)
    assert protocol["root_entropy"] != [28, 3, 20261025, 1]
    assert protocol["root_entropy"] != [28, 2, 20261023]
    for c, r in observations:
        assert len([p for p in production if p[:2] == (c, r)]) == (2 if c == 26 else 1)


@pytest.mark.parametrize(
    "path,value",
    [
        (("conditions", 0, "signal_photons"), 100000),
        (("conditions", 0, "replicates"), 99),
        (("conditions", 1, "replicates"), 11),
        (("conditions", 10, "arms"), ["matched"]),
        (("baseline", "shift_ns"), 0),
        (("grid", "bins"), 511),
        (("fit_settings", "max_starts"), 47),
        (("fit_settings", "max_objective_evaluations"), 60000),
        (("acceptance", "minimum_baseline_numerical_fraction"), 0.94),
        (
            (
                "acceptance",
                "maximum_component_median_absolute_relative_error_exclusive",
            ),
            0.06,
        ),
        (("acceptance", "maximum_component_absolute_relative_bias_exclusive"), 0.03),
        (("audit", "maximum_centered_nll_difference"), 0.02),
        (("audit", "max_forward_calls"), 100001),
        (("audit", "secondary_replicates"), [0, 1, 9]),
        (("audit", "primary_required_agreements"), 99),
        (("audit", "incomplete_counts_as_disagreement"), True),
        (("expected_totals", "scheduled_audits"), 132),
        (("execution", "retry_completed_failures"), True),
        (("execution", "resume_only_never_started_tasks"), False),
        (
            (
                "execution",
                "external_production_assessment_forward_calls_per_available_candidate",
            ),
            0,
        ),
        (
            ("outputs", "raw_directory"),
            "data/generated/issue28_pr3_qualification_20261026",
        ),
        (("root_entropy",), [28, 3, 20261025, 1]),
        (("streams", "qualification"), 2),
        (("observation_rng", "arms_share_observation"), False),
        (("observation_rng", "draws_authorized"), True),
        (("execution_status",), "authorized"),
        (("production_source_lf_sha256",), "0" * 64),
        (("historical_protocol", "lf_sha256"), "0" * 64),
    ],
)
def test_protocol_drift_is_rejected(protocol, path, value):
    changed = deepcopy(protocol)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        API["validate"](changed)


def test_audit_gate_keeps_incomplete_and_disagreement_denominators(protocol):
    all_records = records(protocol)
    complete = API["audit_tally"](protocol, all_records)
    assert complete["primary_audit_target_met"]
    assert complete["all_scheduled_audits_terminal"]
    assert complete["populations"]["primary"]["agreement"] == 100
    assert complete["populations"]["secondary"]["agreement"] == 33
    all_records[0]["classification"] = "audit_incomplete"
    all_records[1]["classification"] = "disagreement"
    all_records[2]["classification"] = "production_unavailable"
    classified = API["audit_tally"](protocol, all_records)
    primary = classified["populations"]["primary"]
    assert primary == dict(
        agreement=97,
        disagreement=1,
        audit_incomplete=1,
        production_unavailable=1,
        not_run=0,
    )
    assert sum(primary.values()) == 100
    assert not classified["primary_audit_target_met"]
    assert classified["all_scheduled_audits_terminal"]
    missing = API["audit_tally"](protocol, all_records[1:])
    assert missing["populations"]["primary"]["not_run"] == 1
    assert not missing["primary_audit_target_met"]
    assert not missing["all_scheduled_audits_terminal"]


def test_secondary_disagreement_is_diagnostic_only(protocol):
    supplied = records(protocol)
    supplied[-1]["classification"] = "disagreement"
    result = API["audit_tally"](protocol, supplied)
    assert result["primary_audit_target_met"]
    assert result["populations"]["secondary"]["disagreement"] == 1


@pytest.mark.parametrize("mode", ["duplicate", "foreign", "unknown_class"])
def test_audit_identity_integrity(protocol, mode):
    supplied = records(protocol)
    if mode == "duplicate":
        supplied.append(supplied[0])
    elif mode == "foreign":
        supplied[0] = {**supplied[0], "replicate": 100}
    else:
        supplied[0] = {**supplied[0], "classification": "success"}
    with pytest.raises(ValueError):
        API["audit_tally"](protocol, supplied)


def test_proposed_protocol_blocks_qualification_before_rng(protocol, monkeypatch):
    import numpy as np
    from tcspc_toolkit.benchmarks.multiexponential import observation_rng

    def forbidden(*args, **kwargs):
        pytest.fail("Protocol review must not construct a random stream")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    monkeypatch.setattr(np.random, "PCG64", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    API["validate"](protocol)
    API["population"](protocol)
    for stream in (0, 1):
        with pytest.raises(PermissionError):
            observation_rng(protocol, stream, 0, 0)


def test_cli_only_validates_and_cannot_execute(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout)["observations_generated"] == 0
    assert not list(tmp_path.iterdir())
    blocked = subprocess.run(
        [sys.executable, str(SCRIPT), "--execute"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert blocked.returncode != 0
    assert "unrecognized arguments" in blocked.stderr
    assert not list(tmp_path.iterdir())
