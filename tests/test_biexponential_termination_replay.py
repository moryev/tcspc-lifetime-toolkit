"""Non-qualification controls for passive instrumentation and strict budgets."""

from dataclasses import asdict
import io
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import tcspc_toolkit.estimation.multiexponential as fitting


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness():
    return runpy.run_path(str(ROOT / "scripts/diagnose_biexponential_termination.py"))


def problem(cls=fitting._Problem, **kwargs):
    t = np.linspace(0, 12, 96)
    k = np.exp(-0.5 * ((t - 0.5) / 0.1) ** 2)
    k /= np.trapezoid(k, x=t)
    shifted = np.interp(t - 0.037, t, k, left=0, right=0)
    mu = 1.5 + sum(
        a * np.convolve(np.exp(-t / tau), shifted)[: t.size] * (t[1] - t[0])
        for a, tau in [(200, 0.7), (75, 3)]
    )
    return cls(
        t,
        np.rint(mu),
        k,
        (0.1, 8),
        (-0.3, 0.3),
        fitting.BiexponentialFitSettings(max_iterations=8),
        **kwargs,
    )


def test_passive_instrumentation_preserves_entire_numerical_sequence(harness):
    recorder = harness["Recorder"](maximum=2000)
    plain = problem()
    initial = fitting.BiexponentialParameters((100, 100), (0.5, 2), 0.5, 0.05)
    callbacks = []
    original = plain.value_and_gradient

    def observe(x):
        value, gradient = original(x)
        callbacks.append((x.copy(), value, gradient.copy()))
        return value, gradient

    plain.value_and_gradient = observe
    model_original, minimize_original = (
        fitting.multiexponential_reconvolution_component_counts,
        fitting.minimize,
    )
    with harness["instrument"](recorder, solver_trace=False):
        baseline = fitting._refine(plain, plain.coordinates(initial), 0, "seed")
    trace = io.StringIO()
    observed_recorder = harness["Recorder"](trace, maximum=2000)
    observed = problem(harness["ObservedProblem"], recorder=observed_recorder)
    with harness["instrument"](observed_recorder):
        diagnostic = fitting._refine(observed, observed.coordinates(initial), 0, "seed")
    assert fitting.multiexponential_reconvolution_component_counts is model_original
    assert fitting.minimize is minimize_original
    left, right = asdict(baseline), asdict(diagnostic)
    left.pop("elapsed_seconds")
    right.pop("elapsed_seconds")
    assert left == right
    assert recorder.calls == observed_recorder.calls == baseline.objective_evaluations
    assert recorder.shared_inputs == observed_recorder.shared_inputs
    assert len(callbacks) == len(observed_recorder.callbacks)
    for (x, value, gradient), recorded in zip(
        callbacks, observed_recorder.callbacks, strict=True
    ):
        np.testing.assert_array_equal(x, recorded["x"])
        assert value == recorded["value"]
        np.testing.assert_array_equal(gradient, recorded["gradient"])
    assert observed_recorder.transitions
    assert any(
        json.loads(line)["event"] == "setulb_transition"
        for line in trace.getvalue().splitlines()
    )
    np.testing.assert_array_equal(plain.local_x, observed.local_x)


def test_common_cap_counts_actual_shared_calls_including_gradient_probes(harness):
    recorder = harness["Recorder"](maximum=5)
    p = problem(harness["ObservedProblem"], recorder=recorder)
    x = p.coordinates(fitting.BiexponentialParameters((200, 75), (0.7, 3), 1.5, 0.037))
    with harness["instrument"](recorder):
        with pytest.raises(harness["ReplayLimit"], match="forward_call_limit"):
            p.value_and_gradient(x)
    assert recorder.calls == 5
    assert len(recorder.shared_inputs) == 5
    assert np.isfinite(p.best_value)
    assert p.best_x is not None


def test_analytic_derivatives_are_independent_and_charged(harness):
    recorder = harness["Recorder"]()
    p = problem()
    x = p.coordinates(fitting.BiexponentialParameters((180, 80), (0.65, 2.8), 2, 0.037))
    with harness["instrument"](recorder):
        value, finite_difference = p.value_and_gradient(x)
        before = recorder.calls
        direct, _ = harness["independent_derivative"](p, x, recorder)
    assert recorder.calls - before == 6
    assert not direct["shift_interpolation_knot"]
    assert direct["objective"] == pytest.approx(value, abs=1e-9)
    np.testing.assert_allclose(
        direct["gradient"], finite_difference, rtol=1e-4, atol=1e-5
    )


def test_independent_derivative_cannot_bypass_common_cap(harness):
    recorder = harness["Recorder"](maximum=3)
    p = problem()
    x = p.coordinates(fitting.BiexponentialParameters((180, 80), (0.65, 2.8), 2, 0.037))
    with pytest.raises(harness["ReplayLimit"], match="forward_call_limit"):
        harness["independent_derivative"](p, x, recorder)
    assert recorder.calls == 3


def test_clock_limit_blocks_before_next_evaluation(harness):
    now = [0.0]
    recorder = harness["Recorder"](seconds=300, clock=lambda: now[0])
    now[0] = 300
    with pytest.raises(harness["ReplayLimit"], match="wall_time_limit"):
        recorder.charge("shared_model")
    assert recorder.calls == 0


def test_nonfinite_values_are_distinct_from_missing(harness):
    result = json.loads(harness["encode"]([np.inf, -np.inf, np.nan, None, 1.0]))
    assert result == ["Infinity", "-Infinity", "NaN", None, 1.0]


def test_reconstruction_does_not_call_initializer_or_model(harness):
    p = problem()
    initial = fitting.BiexponentialParameters((123.4, 0), (0.1, 8), 0, 0.3)
    x, record = harness["reconstruct"](p, asdict(initial))
    assert p.feasible(x)
    assert not any(p.counts.values())
    np.testing.assert_allclose(
        record["physical_roundtrip"], [123.4, 0, 0.1, 8, 0, 0.3], rtol=3e-15
    )
    assert record["historical_internal_coordinates_available"] is False


def test_no_overwrite_preserves_original_attempt(harness, tmp_path):
    path = tmp_path / "attempt.json"
    harness["write_new"](path, {"status": "incomplete"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        harness["write_new"](path, {"status": "retry"})
    assert path.read_bytes() == original


def test_existing_attempt_refuses_automatic_retry(harness, tmp_path, monkeypatch):
    marker = tmp_path / "r011-s10/attempt.json"
    harness["write_new"](marker, {"status": "interrupted"})
    namespace = harness["run"].__globals__
    monkeypatch.setitem(
        namespace,
        "verify_manifest",
        lambda _: {
            "protocol": {"cases": [[11, 10]]},
            "output_directory": str(tmp_path),
        },
    )
    with pytest.raises(FileExistsError, match="do not retry"):
        harness["run"]("unused")


def test_external_timeout_terminates_only_replay_process_tree(harness, monkeypatch):
    calls = []

    def wait(timeout=None):
        calls.append(("wait", timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired("child", timeout)

    child = SimpleNamespace(pid=54321, wait=wait, kill=lambda: calls.append("kill"))
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    assert harness["wait_bounded"](child, 300)
    if harness["os"].name == "nt":
        assert ["taskkill", "/PID", "54321", "/T", "/F"] in calls
    else:
        assert "kill" in calls
    assert calls[-1] == ("wait", None)


def test_manifest_rejects_changed_diagnostic_implementation(
    harness, tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"sources_lf_sha256": {"script": "old"}, "scipy_sources_sha256": {}})
    )
    monkeypatch.setitem(
        harness["verify_manifest"].__globals__,
        "source_snapshot",
        lambda: {"script": "changed"},
    )
    with pytest.raises(ValueError, match="implementation changed"):
        harness["verify_manifest"](path)
