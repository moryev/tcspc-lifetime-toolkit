"""Small public-workflow checks; the full Bayesian example is validated separately."""

from dataclasses import fields
from pathlib import Path
import re
import runpy
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    return runpy.run_path(str(ROOT / "examples" / f"{name}.py"))


def _marked(source, name):
    return textwrap.dedent(source.split(f"# README {name} begin\n", 1)[1]
                          .split(f"# README {name} end", 1)[0]).strip()


def test_readme_quickstart_is_the_executable_script_workflow():
    source = (ROOT / "examples/quickstart.py").read_text(encoding="utf-8")
    expected = _marked(source, "imports") + "\n\n" + _marked(source, "workflow")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("<!-- quickstart begin -->", 1)[1].split("<!-- quickstart end -->", 1)[0]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1).strip()
    assert code == expected
    namespace = {}
    exec(compile(code, "README.md quickstart", "exec"), namespace)
    fit = namespace["fit"]
    assert fit.success and fit.numerical_validation_passed
    assert abs(fit.lifetime - 2.5) < 0.1
    assert np.all(np.isfinite(fit.fitted_curve)) and np.all(fit.fitted_curve > 0)
    assert namespace["expected"].sum() == pytest.approx(100_000 + 512 * 2)
    assert np.issubdtype(namespace["counts"].dtype, np.integer)


@pytest.mark.parametrize("name, expected", [
    ("quickstart", "Synthetic lifetime: 2.5000 ns"),
    ("experimental_csv", "valid fit: True"),
    ("preprocessing_features", "separate normalized sum: 1.000000"),
    ("ml_lifetime_prediction", "72 training, 24 held out"),
    ("uncertainty_example", "Bootstrap valid: True"),
])
def test_standalone_commands(name, expected, tmp_path):
    # Absolute script paths also work outside the checkout; inputs are script-relative.
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / f"{name}.py")], cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert expected in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_csv_preparation_preserves_source_and_has_no_intrinsic_truth():
    measurement, sampled, prepared, fit = _example("experimental_csv")["run_example"]()
    assert measurement.time_ns[-1] == pytest.approx(20)
    assert measurement.require_raw_counts().sum() == 7552
    np.testing.assert_array_equal(sampled.values, prepared.source.values)
    assert np.trapezoid(prepared.kernel, x=measurement.time_ns) == pytest.approx(1)
    assert prepared.diagnostics.resampling_method == "none"
    assert "lifetime_true_ns" not in measurement.metadata
    assert fit.valid_fit and abs(fit.fitted_lifetime_ns - 3) < 0.15


def test_features_are_separate_from_raw_measurement():
    measurement, features, normalized = _example("preprocessing_features")["run_example"]()
    assert measurement.values.sum() == features.iloc[0]["total_counts"] == 7552
    assert np.isfinite(features.to_numpy()).all()
    assert normalized.sum() == pytest.approx(1)
    assert not np.shares_memory(normalized, measurement.values)


def test_ml_pipeline_fits_only_training_rows_and_beats_mean_baseline():
    model, train, test, prediction, mae, baseline_mae = _example("ml_lifetime_prediction")["run_example"]()
    assert len(train) == 72 and len(test) == 24
    assert not set(train) & set(test)
    assert model.named_steps["scaler"].n_samples_seen_ == 72
    assert np.isfinite(prediction).all() and np.all(prediction > 0)
    assert mae < baseline_mae


def test_figure_uses_supplied_fit_without_refitting(tmp_path, monkeypatch):
    example = _example("quickstart")
    result = example["run_example"]()

    def no_fit(*args, **kwargs):
        raise AssertionError("Figure generation must reuse the computed fit")

    monkeypatch.setitem(example["save_figure"].__globals__, "fit_monoexponential_reconvolution", no_fit)
    output = tmp_path / "quickstart.png"
    example["save_figure"](output, *result)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_bayesian_configuration_keeps_diagnostic_defaults_and_reproducible_inputs():
    from tcspc_toolkit.estimation.bayesian.sampling import BayesianSamplingConfig

    build = _example("bayesian_inference")["build_inputs"]
    measurement, prepared, priors, config = build()
    second, _, _, _ = build()
    np.testing.assert_array_equal(measurement.values, second.values)
    assert measurement.require_raw_counts().size == 161
    assert np.trapezoid(prepared.kernel, x=measurement.time_ns) == pytest.approx(1)
    assert priors.fixed_temporal_shift_ns == 0 and not priors.infer_temporal_shift
    assert config.max_production_steps == 10_000 and config.n_ensembles == 2
    for field in fields(BayesianSamplingConfig):
        if field.name.startswith(("min_", "max_")) and field.name != "max_production_steps":
            assert getattr(config, field.name) == field.default


def test_bayesian_example_dispatches_real_api_and_labels_rejected_diagnostics(monkeypatch, capsys):
    from tcspc_toolkit.estimation.bayesian.sampling import BayesianSamplingStatus

    example = _example("bayesian_inference")
    calls = []

    def sample(measurement, **kwargs):
        calls.append((measurement, kwargs))
        # Only a unit-test stub; the executable example constructs no posterior fixture.
        return SimpleNamespace(result=SimpleNamespace(
            status=BayesianSamplingStatus.INSUFFICIENT_SAMPLING,
            runtime=SimpleNamespace(total_seconds=0),
            diagnostics=SimpleNamespace(production_steps=1000, failure_reasons=("test limitation",)),
            parameter_summaries=(),
        ))

    monkeypatch.setitem(example["run_example"].__globals__, "fit_bayesian_monoexponential_reconvolution", sample)
    example["run_example"]()
    assert len(calls) == 1
    assert calls[0][1]["sampling_config"].random_seed == 1402
    output = capsys.readouterr().out
    assert "PROVISIONAL" in output and "test limitation" in output


def test_bayesian_missing_extra_has_installation_guidance(tmp_path):
    script = "\n".join([
        "import importlib.abc, runpy, sys",
        "class NoEmcee(importlib.abc.MetaPathFinder):",
        "    def find_spec(self, fullname, path=None, target=None):",
        "        if fullname == 'emcee': raise ModuleNotFoundError('blocked optional dependency', name='emcee')",
        "sys.meta_path.insert(0, NoEmcee())",
        f"runpy.run_path({str(ROOT / 'examples/bayesian_inference.py')!r}, run_name='__main__')",
    ])
    completed = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert 'python -m pip install -e ".[bayesian]"' in completed.stderr


def _prose(source):
    """Exclude fenced examples from Markdown link/heading inspection."""
    output, fence = [], None
    for line in source.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            if fence is None:
                fence = match[1][0]
            elif match[1][0] == fence:
                fence = None
        elif fence is None:
            output.append(line)
    return "\n".join(output)


def _anchors(path):
    source = _prose(path.read_text(encoding="utf-8"))
    anchors = set(re.findall(r'<a\s+id="([^"]+)"', source))
    occurrences = {}
    for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", source, re.M):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        previous = occurrences.get(slug, 0)
        occurrences[slug] = previous + 1
        anchors.add(slug + (f"-{previous}" if previous else ""))
    return anchors


def test_documentation_relative_links_and_anchors_resolve():
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")),
             ROOT / "examples/README.md", ROOT / "data/examples/README.md"]
    failures = []
    for path in paths:
        source = _prose(path.read_text(encoding="utf-8"))
        for label, target in re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", source):
            if urlsplit(target).scheme or "\\" in label:
                continue  # External URLs and mathematical expressions are not local links.
            relative, _, fragment = target.partition("#")
            destination = path.parent / unquote(relative) if relative else path
            if not destination.exists():
                failures.append((str(path.relative_to(ROOT)), target))
            elif fragment and destination.suffix == ".md" and unquote(fragment) not in _anchors(destination):
                failures.append((str(path.relative_to(ROOT)), target))
    assert not failures, failures
