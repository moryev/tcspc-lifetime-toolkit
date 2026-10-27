"""Canonical architecture and post-Issue-12 removed-module contracts.

The historical deny-list is fixed even though its source files no longer exist.
Retained package APIs and canonical dependency boundaries remain supported.
"""

import ast
import importlib
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
import subprocess
import sys

import pytest


PACKAGE = "tcspc_toolkit"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# Explicit historical manifest; no discovery from surviving files.
REMOVED_FLAT_FACADES = {
    "baselines": "estimation.baselines",
    "bayesian": "estimation.bayesian.posterior",
    "bayesian_evaluation": "benchmarks.bayesian_calibration",
    "bayesian_mismatch_evaluation": "benchmarks.bayesian_mismatch",
    "bayesian_predictive": "estimation.bayesian.predictive",
    "bayesian_sampling": "estimation.bayesian.sampling",
    "benchmark_plots": "evaluation.plots",
    "classical_evaluation": "estimation.classical",
    "classical_uncertainty": "uncertainty.classical",
    "conditional_evaluation": "evaluation.conditional",
    "convolution": "physics.convolution",
    "cross_validation": "ml.cross_validation",
    "datasets": "simulation.datasets",
    "estimator_api": "estimation.api",
    "evaluation_core": "evaluation.core",
    "evaluation_results": "evaluation.results",
    "evaluation_uncertainty": "evaluation.attachments",
    "experimental": "measurement.adapters",
    "features": "preprocessing.features",
    "fitting": "estimation.fitting",
    "forward_model": "physics.forward_model",
    "generalization": "benchmarks.generalization_protocol",
    "generalization_datasets": "benchmarks.generalization_datasets",
    "generalization_evaluation": "benchmarks.generalization",
    "irf_estimation": "irf.estimation",
    "irf_evaluation": "benchmarks.irf",
    "irf_preparation": "irf.preparation",
    "measurement_io": "measurement.io",
    "measurements": "measurement.records",
    "mismatch_evaluation": "benchmarks.mismatch",
    "ml_evaluation": "ml.evaluation",
    "ml_models": "ml.models",
    "ml_uncertainty": "uncertainty.ml",
    "models": "physics.models",
    "representations": "preprocessing.representations",
    "timing_evaluation": "benchmarks.timing",
    "uncertainty_evaluation": "uncertainty.evaluation",
    "uncertainty_robustness": "benchmarks.uncertainty",
}

RETAINED_PACKAGE_APIS = {
    "evaluation": "evaluation.diagnostics",
    "irf": "irf.profiles",
    "persistence": "persistence.sqlite",
    "preprocessing": "preprocessing.histograms",
    "simulation": "simulation.histograms",
}

SUPPORTED_MODULE_PATHS = (*RETAINED_PACKAGE_APIS, "config", "exceptions")

# Responsibility sets identify implementations, not their future spelling.
# Canonical dependency restrictions survive removal of historical paths.
EXTENSION = {f"{PACKAGE}.estimation.api"}
BASELINES = f"{PACKAGE}.estimation.baselines"
ESTIMATION = f"{PACKAGE}.estimation"
BAYESIAN = f"{ESTIMATION}.bayesian"
ML = f"{PACKAGE}.ml"
BENCHMARKS = f"{PACKAGE}.benchmarks"
GENERALIZATION_PROTOCOL = f"{BENCHMARKS}.generalization_protocol"
GENERALIZATION_DATASETS = f"{BENCHMARKS}.generalization_datasets"
GENERALIZATION = f"{BENCHMARKS}.generalization"
MISMATCH = f"{BENCHMARKS}.mismatch"
TIMING = f"{BENCHMARKS}.timing"
IRF_STUDY = f"{BENCHMARKS}.irf"
BAYESIAN_CALIBRATION = f"{BENCHMARKS}.bayesian_calibration"
BAYESIAN_MISMATCH = f"{BENCHMARKS}.bayesian_mismatch"
UNCERTAINTY_BENCHMARK = f"{BENCHMARKS}.uncertainty"
UNCERTAINTY = f"{PACKAGE}.uncertainty"
UNCERTAINTY_EVALUATION = f"{UNCERTAINTY}.evaluation"
UNCERTAINTY_ML = f"{UNCERTAINTY}.ml"
UNCERTAINTY_CLASSICAL = f"{UNCERTAINTY}.classical"
UNCERTAINTY_EVALUATION_IMPORTS = {
    "dataclasses",
    "enum",
    "numpy",
    "numpy.typing",
    "scipy.stats",
    GENERALIZATION_PROTOCOL,
}
BENCHMARK_IMPORTS = {
    UNCERTAINTY_BENCHMARK: {
        "__future__",
        "collections.abc",
        "dataclasses",
        "math",
        "typing",
        "numpy",
        "pandas",
        "numpy.typing",
        "tcspc_toolkit.evaluation.diagnostics",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.uncertainty.classical",
        "tcspc_toolkit.evaluation.core",
        "tcspc_toolkit.benchmarks.generalization_protocol",
        "tcspc_toolkit.benchmarks.generalization_datasets",
        "tcspc_toolkit.benchmarks.generalization",
        "tcspc_toolkit.evaluation.results",
        "tcspc_toolkit.evaluation.attachments",
        "tcspc_toolkit.irf.profiles",
        "tcspc_toolkit.ml.evaluation",
        "tcspc_toolkit.uncertainty.ml",
        "tcspc_toolkit.uncertainty.evaluation",
    },
    IRF_STUDY: {
        "__future__",
        "dataclasses",
        "itertools",
        "typing",
        "numpy",
        "pandas",
        "numpy.typing",
        "scipy.optimize",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.uncertainty.classical",
        "tcspc_toolkit.config",
        "tcspc_toolkit.evaluation.diagnostics",
        "tcspc_toolkit.measurement.adapters",
        "tcspc_toolkit.preprocessing.features",
        "tcspc_toolkit.irf.profiles",
        "tcspc_toolkit.irf.estimation",
        "tcspc_toolkit.irf.preparation",
        "tcspc_toolkit.measurement.records",
        "tcspc_toolkit.ml.evaluation",
        "tcspc_toolkit.ml.models",
        "tcspc_toolkit.uncertainty.ml",
        "tcspc_toolkit.physics.models",
        "tcspc_toolkit.preprocessing.histograms",
        "tcspc_toolkit.simulation.histograms",
    },
    BAYESIAN_CALIBRATION: {
        "__future__",
        "hashlib",
        "math",
        "time",
        "dataclasses",
        "statistics",
        "numpy",
        "numpy.typing",
        "tcspc_toolkit.estimation.bayesian.posterior",
        "tcspc_toolkit.estimation.bayesian.sampling",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.uncertainty.classical",
        "tcspc_toolkit.physics.forward_model",
        "tcspc_toolkit.irf.preparation",
        "tcspc_toolkit.measurement.records",
        "tcspc_toolkit.simulation.histograms",
    },
    BAYESIAN_MISMATCH: {
        "__future__",
        "hashlib",
        "math",
        "time",
        "dataclasses",
        "statistics",
        "typing",
        "numpy",
        "numpy.typing",
        "scipy.optimize",
        "tcspc_toolkit.estimation.bayesian.posterior",
        "tcspc_toolkit.benchmarks.bayesian_calibration",
        "tcspc_toolkit.estimation.bayesian.predictive",
        "tcspc_toolkit.estimation.bayesian.sampling",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.uncertainty.classical",
        "tcspc_toolkit.estimation.fitting",
        "tcspc_toolkit.physics.forward_model",
        "tcspc_toolkit.irf.profiles",
        "tcspc_toolkit.benchmarks.irf",
        "tcspc_toolkit.irf.preparation",
        "tcspc_toolkit.measurement.records",
        "tcspc_toolkit.simulation.histograms",
    },
    MISMATCH: {
        "__future__",
        "dataclasses",
        "numpy",
        "pandas",
        "numpy.typing",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.config",
        "tcspc_toolkit.preprocessing.features",
        "tcspc_toolkit.evaluation.results",
        "tcspc_toolkit.ml.evaluation",
        "tcspc_toolkit.ml.models",
        "tcspc_toolkit.simulation.histograms",
    },
    TIMING: {
        "__future__",
        "dataclasses",
        "time",
        "typing",
        "numpy",
        "pandas",
        "numpy.typing",
        "tcspc_toolkit.estimation.baselines",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.ml.evaluation",
        "tcspc_toolkit.ml.models",
    },
    BENCHMARKS: set(),
    GENERALIZATION_PROTOCOL: {"dataclasses", "enum", "math"},
    GENERALIZATION_DATASETS: {
        "__future__",
        "dataclasses",
        "itertools",
        "numpy",
        "pandas",
        "numpy.typing",
        GENERALIZATION_PROTOCOL,
        f"{PACKAGE}.simulation.histograms",
    },
    GENERALIZATION: {
        "__future__",
        "dataclasses",
        "numpy",
        "pandas",
        "numpy.typing",
        "collections.abc",
        "typing",
        "sklearn.decomposition",
        f"{PACKAGE}.estimation.baselines",
        f"{PACKAGE}.estimation.classical",
        f"{PACKAGE}.config",
        f"{PACKAGE}.estimation.api",
        f"{PACKAGE}.evaluation.core",
        f"{PACKAGE}.evaluation.results",
        f"{PACKAGE}.preprocessing.features",
        GENERALIZATION_PROTOCOL,
        GENERALIZATION_DATASETS,
        f"{PACKAGE}.irf.profiles",
        f"{PACKAGE}.ml.evaluation",
        f"{PACKAGE}.ml.models",
        f"{PACKAGE}.preprocessing.representations",
    },
}
MIGRATED_SHIMS = {f"{PACKAGE}.{name}" for name in REMOVED_FLAT_FACADES}
# A package facade must run as a parent initializer, but canonical code must
# not explicitly import its old re-export surface as an implementation.
PREPROCESSING = f"{PACKAGE}.preprocessing"
IRF = f"{PACKAGE}.irf"
MEASUREMENT = f"{PACKAGE}.measurement"
SIMULATION = f"{PACKAGE}.simulation"
EVALUATION = f"{PACKAGE}.evaluation"
DIAGNOSTICS = f"{EVALUATION}.diagnostics"
DIAGNOSTIC_EXPORTS = (
    "generate_fitted_signal",
    "calculate_residuals",
    "calculate_reduced_residuals",
    "calculate_absolute_lifetime_error",
    "calculate_relative_lifetime_error",
    "calculate_poisson_deviance_residuals",
    "calculate_lifetime_errors",
)
DIAGNOSTIC_IMPORTS = {
    "dataclasses",
    "numpy",
    "numpy.typing",
    "scipy.special",
    f"{PACKAGE}.physics.models",
    f"{ESTIMATION}.fitting",
}
COMPATIBILITY_SURFACES = MIGRATED_SHIMS | {
    PREPROCESSING,
    IRF,
    SIMULATION,
    EVALUATION,
    f"{PACKAGE}.persistence",
}
FACTUAL_IMPORTS = {
    f"{EVALUATION}.results": {
        "collections.abc",
        "dataclasses",
        "types",
        "typing",
        "numpy",
        "pandas",
        "numpy.typing",
    },
    f"{EVALUATION}.core": {
        "collections.abc",
        "numpy",
        "pandas",
        "numpy.typing",
        f"{EVALUATION}.results",
    },
    f"{EVALUATION}.attachments": {
        "dataclasses",
        "numpy",
        "pandas",
        f"{EVALUATION}.results",
    },
}
FACTUAL = set(FACTUAL_IMPORTS)
ANALYSIS_IMPORTS = {
    f"{EVALUATION}.conditional": {
        "__future__",
        "collections.abc",
        "numpy",
        "pandas",
        "numpy.typing",
        DIAGNOSTICS,
        f"{ESTIMATION}.classical",
        f"{EVALUATION}.core",
        f"{EVALUATION}.results",
        f"{ML}.evaluation",
    },
    f"{EVALUATION}.plots": {
        "__future__",
        "matplotlib.pyplot",
        "numpy",
        "pandas",
        "matplotlib.axes",
    },
}
PHYSICAL = {
    f"{PACKAGE}.{name}"
    for name in (
        "physics.models",
        "physics.convolution",
        "physics.forward_model",
        "irf.profiles",
    )
}
PHYSICS_IMPORTS = {
    f"{PACKAGE}.physics.models": {"numpy", "numpy.typing"},
    f"{PACKAGE}.physics.convolution": {"numpy", "numpy.typing", "scipy.signal"},
    f"{PACKAGE}.physics.forward_model": {
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.physics.models",
        f"{PACKAGE}.physics.convolution",
        f"{PACKAGE}.irf.profiles",
    },
}
CLASSICAL_IMPORTS = {
    f"{ESTIMATION}.multiexponential": {
        "__future__",
        "collections.abc",
        "copy",
        "dataclasses",
        "time",
        "types",
        "typing",
        "numpy",
        "numpy.typing",
        "scipy.optimize",
        DIAGNOSTICS,
        f"{PACKAGE}.irf.preparation",
        f"{PACKAGE}.physics.forward_model",
        f"{PREPROCESSING}.histograms",
    },
    f"{ESTIMATION}.fitting": {
        "collections.abc",
        "dataclasses",
        "numpy",
        "numpy.typing",
        "scipy.optimize",
        f"{PACKAGE}.physics.forward_model",
        f"{PACKAGE}.physics.models",
    },
    f"{ESTIMATION}.classical": {
        "__future__",
        "dataclasses",
        "time",
        "pandas",
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.physics.convolution",
        f"{PACKAGE}.physics.models",
        f"{PREPROCESSING}.histograms",
        f"{ESTIMATION}.fitting",
        DIAGNOSTICS,
    },
}
BAYESIAN_IMPORTS = {
    BAYESIAN: set(),
    f"{BAYESIAN}.posterior": {
        "__future__",
        "math",
        "dataclasses",
        "enum",
        "typing",
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.exceptions",
        f"{IRF}.selection",
        f"{IRF}.profiles",
        f"{IRF}.preparation",
        f"{MEASUREMENT}.records",
        f"{ESTIMATION}.fitting",
        f"{PACKAGE}.physics.forward_model",
        f"{BAYESIAN}.sampling",
    },
    f"{BAYESIAN}.sampling": {
        "__future__",
        "math",
        "time",
        "warnings",
        "dataclasses",
        "enum",
        "typing",
        "numpy",
        "numpy.typing",
        "emcee",
        "emcee.autocorr",
        f"{BAYESIAN}.posterior",
        f"{PACKAGE}.physics.forward_model",
        f"{IRF}.profiles",
        f"{IRF}.preparation",
        f"{MEASUREMENT}.records",
    },
    f"{BAYESIAN}.predictive": {
        "__future__",
        "math",
        "time",
        "dataclasses",
        "numpy",
        "numpy.typing",
        f"{BAYESIAN}.posterior",
        f"{BAYESIAN}.sampling",
        f"{PACKAGE}.physics.forward_model",
        f"{SIMULATION}.histograms",
        DIAGNOSTICS,
    },
}
ML_IMPORTS = {
    ML: set(),
    f"{ML}.models": {
        "__future__",
        "numpy",
        "sklearn.decomposition",
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.pipeline",
        "sklearn.preprocessing",
        f"{PACKAGE}.config",
        f"{ESTIMATION}.api",
        f"{PREPROCESSING}.representations",
    },
    f"{ML}.evaluation": {
        "dataclasses",
        "itertools",
        "numpy",
        "pandas",
        "numpy.typing",
        "typing",
        "sklearn.metrics",
        "sklearn.decomposition",
        "sklearn.model_selection",
        BASELINES,
        f"{PACKAGE}.config",
        f"{PREPROCESSING}.features",
        f"{PREPROCESSING}.representations",
        f"{SIMULATION}.histograms",
        f"{ML}.models",
        f"{EVALUATION}.core",
        f"{EVALUATION}.results",
    },
    f"{ML}.cross_validation": {
        "__future__",
        "collections.abc",
        "dataclasses",
        "typing",
        "numpy",
        "pandas",
        "numpy.typing",
        "sklearn.base",
        "sklearn.model_selection",
        f"{ML}.evaluation",
    },
}
SIMULATION_IMPORTS = {
    SIMULATION: {f"{SIMULATION}.histograms"},
    f"{SIMULATION}.histograms": {
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.physics.convolution",
        f"{PACKAGE}.physics.models",
        f"{IRF}.profiles",
        f"{IRF}.preparation",
    },
    f"{SIMULATION}.datasets": {
        "dataclasses",
        "numpy",
        "numpy.typing",
        "pandas",
        f"{PACKAGE}.physics.models",
    },
}
PERSISTENCE = f"{PACKAGE}.persistence"
PERSISTENCE_SQLITE = f"{PERSISTENCE}.sqlite"
IRF_IMPORTS = {
    IRF: {f"{IRF}.profiles"},
    f"{IRF}.profiles": {
        "__future__",
        "copy",
        "dataclasses",
        "enum",
        "types",
        "typing",
        "numpy",
        "numpy.typing",
        "scipy.stats",
    },
    f"{IRF}.preparation": {
        "__future__",
        "dataclasses",
        "typing",
        "numpy",
        "numpy.typing",
        f"{IRF}.profiles",
        f"{MEASUREMENT}.records",
        f"{PREPROCESSING}.histograms",
    },
    f"{IRF}.estimation": {
        "__future__",
        "dataclasses",
        "typing",
        "numpy",
        "numpy.typing",
        "scipy.signal",
        f"{IRF}.profiles",
        f"{IRF}.preparation",
        f"{MEASUREMENT}.records",
        f"{PREPROCESSING}.histograms",
    },
    f"{IRF}.selection": {
        "__future__",
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.exceptions",
        f"{IRF}.profiles",
        f"{IRF}.preparation",
        f"{MEASUREMENT}.records",
        f"{PREPROCESSING}.histograms",
    },
}
MEASUREMENT_IMPORTS = {
    MEASUREMENT: set(),
    f"{MEASUREMENT}.records": {
        "__future__",
        "copy",
        "dataclasses",
        "enum",
        "types",
        "typing",
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.exceptions",
        f"{IRF}.profiles",
        f"{PREPROCESSING}.histograms",
    },
    f"{MEASUREMENT}.io": {
        "__future__",
        "csv",
        "decimal",
        "pathlib",
        "typing",
        "numpy",
        f"{PACKAGE}.exceptions",
        f"{MEASUREMENT}.records",
    },
    f"{MEASUREMENT}.adapters": {
        "__future__",
        "copy",
        "dataclasses",
        "types",
        "typing",
        "numpy",
        "pandas",
        "numpy.typing",
        f"{ESTIMATION}.classical",
        f"{ESTIMATION}.multiexponential",
        f"{PACKAGE}.evaluation.reconvolution",
        f"{PACKAGE}.physics.forward_model",
        f"{PACKAGE}.config",
        DIAGNOSTICS,
        f"{PACKAGE}.exceptions",
        f"{PREPROCESSING}.features",
        f"{PREPROCESSING}.histograms",
        f"{IRF}.preparation",
        f"{IRF}.profiles",
        f"{IRF}.selection",
        f"{MEASUREMENT}.records",
    },
}
PREPROCESSING_IMPORTS = {
    PREPROCESSING: {f"{PREPROCESSING}.histograms"},
    f"{PREPROCESSING}.histograms": {
        "__future__",
        "logging",
        "numpy",
        "numpy.typing",
        f"{PACKAGE}.config",
        f"{PACKAGE}.exceptions",
    },
    f"{PREPROCESSING}.features": {
        "__future__",
        "numpy",
        "pandas",
        "numpy.typing",
        f"{PACKAGE}.config",
        f"{PACKAGE}.exceptions",
        f"{PREPROCESSING}.histograms",
    },
    f"{PREPROCESSING}.representations": {
        "__future__",
        "numpy",
        "numpy.typing",
        "sklearn.decomposition",
        f"{PACKAGE}.config",
        f"{PREPROCESSING}.histograms",
    },
}


def _imports(source, module, *, is_package=False):
    """Static imports with eager, function-local, and typing-only classification.

    Conservatively visit runtime branches (including try/except and class
    bodies). TYPE_CHECKING branches and function bodies are not eager. This
    does not execute code or pretend to resolve arbitrary dynamic imports.
    """
    tree = ast.parse(source)
    typing_names = {"typing"}
    checking_names = {"TYPE_CHECKING"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            typing_names.update(
                a.asname or a.name for a in node.names if a.name == "typing"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "typing":
            checking_names.update(
                a.asname or a.name for a in node.names if a.name == "TYPE_CHECKING"
            )
    records = []

    class Visitor(ast.NodeVisitor):
        kind = "eager"

        def visit_Import(self, node):
            records.extend((self.kind, a.name, ()) for a in node.names)

        def visit_ImportFrom(self, node):
            base = node.module or ""
            if node.level:
                package = module if is_package else module.rpartition(".")[0]
                base = importlib.util.resolve_name("." * node.level + base, package)
            records.append((self.kind, base, tuple(a.name for a in node.names)))

        def visit_FunctionDef(self, node):
            previous = self.kind
            if self.kind != "typing":
                self.kind = "deferred"
            for statement in node.body:
                self.visit(statement)
            self.kind = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_If(self, node):
            test = node.test
            checking = (isinstance(test, ast.Name) and test.id in checking_names) or (
                isinstance(test, ast.Attribute)
                and test.attr == "TYPE_CHECKING"
                and isinstance(test.value, ast.Name)
                and test.value.id in typing_names
            )
            previous = self.kind
            if checking:
                self.kind = "typing"
            for statement in node.body:
                self.visit(statement)
            self.kind = previous
            for statement in node.orelse:
                self.visit(statement)

    Visitor().visit(tree)
    return records


def _targets(base, names, modules):
    # `from package import child` resolves child modules, not just the package.
    children = {f"{base}.{name}" for name in names} & modules
    needs_base = not names or len(children) != len(names)
    return children | ({base} if needs_base and base in modules else set())


def _eager_graph(records, packages):
    modules = set(records)
    graph = {name: set() for name in modules}
    for module, imports in records.items():
        for kind, base, names in imports:
            if kind != "eager":
                continue
            targets = _targets(base, names, modules)
            for target in targets.copy():
                # Account for package initializers reached across package
                # boundaries, but not the already-running parent/root package.
                targets.update(
                    p
                    for p in packages
                    if target.startswith(p + ".")
                    and not module.startswith(p + ".")
                    and module != p
                )
            graph[module].update(targets - {module})
    return graph


@pytest.fixture(scope="module")
def package_imports():
    records, packages = {}, set()
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT.parent).with_suffix("")
        is_package = path.name == "__init__.py"
        parts = relative.parts[:-1] if is_package else relative.parts
        module = ".".join(parts)
        if is_package:
            packages.add(module)
        records[module] = _imports(
            path.read_text(encoding="utf-8"), module, is_package=is_package
        )
    return records, packages


def test_retained_package_and_support_modules_remain_importable():
    for name in SUPPORTED_MODULE_PATHS:
        assert importlib.import_module(f"{PACKAGE}.{name}") is not None


def test_removed_flat_modules_are_absent_without_an_artificial_blocker(tmp_path):
    assert len(REMOVED_FLAT_FACADES) == 38
    for name in REMOVED_FLAT_FACADES:
        assert not (SOURCE_ROOT / f"{name}.py").exists()
        assert not (SOURCE_ROOT / name).exists()
    script = f"""
import importlib
import importlib.util
import sys
import tcspc_toolkit as toolkit
removed = {sorted(REMOVED_FLAT_FACADES)!r}
# Loading every canonical replacement must not install aliases as a side effect.
for target in {sorted(set(REMOVED_FLAT_FACADES.values()))!r}:
    importlib.import_module('tcspc_toolkit.' + target)
assert '__getattr__' not in vars(toolkit)
for short in removed:
    name = 'tcspc_toolkit.' + short
    assert name not in sys.modules, name
    assert not hasattr(toolkit, short), short
    assert importlib.util.find_spec(name) is None, name
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        assert error.name == name, (name, error.name)
    else:
        raise AssertionError('Removed module imported: ' + name)
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_responsibility_dependency_direction(package_imports):
    records, _ = package_imports
    assert EXTENSION | FACTUAL | PHYSICAL <= records.keys()
    allowed_external = {
        "__future__",
        "collections",
        "dataclasses",
        "types",
        "typing",
        "numpy",
        "pandas",
    }
    for module in EXTENSION | FACTUAL:
        for _, base, names in records[module]:
            if base == PACKAGE or base.startswith(PACKAGE + "."):
                targets = _targets(base, names, set(records)) or {base}
                assert targets <= (FACTUAL if module in FACTUAL else set()), (
                    module,
                    targets,
                )
            else:
                assert base.split(".")[0] in allowed_external, (module, base)
                if module in EXTENSION:
                    assert base.split(".")[0] != "pandas", (module, base)
    for module in PHYSICAL:
        for _, base, names in records[module]:
            if base == PACKAGE or base.startswith(PACKAGE + "."):
                assert (_targets(base, names, set(records)) or {base}) <= PHYSICAL, (
                    module,
                    base,
                )
    for module, imports in records.items():
        if module == PERSISTENCE or module.startswith(PERSISTENCE + "."):
            continue
        for _, base, names in imports:
            targets = _targets(base, names, set(records)) | {base}
            assert not any(
                t == PERSISTENCE or t.startswith(PERSISTENCE + ".") for t in targets
            ), (module, targets)


def test_migrated_initializers_and_canonical_consumers_stay_independent(
    package_imports,
):
    records, _ = package_imports
    assert records[ESTIMATION] == []  # No eager, deferred, or typing imports.
    assert records[BAYESIAN] == []
    assert records[ML] == []
    assert records[f"{PACKAGE}.physics"] == []
    assert records[MEASUREMENT] == []
    # Preserve the existing DummyRegressor dependency, not a new algorithm.
    assert {base for _, base, _ in records[BASELINES]} == {
        "__future__",
        "numpy",
        "numpy.typing",
        "sklearn.dummy",
    }
    for module, allowed in PHYSICS_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in CLASSICAL_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in BAYESIAN_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in ML_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in PREPROCESSING_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in IRF_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in MEASUREMENT_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    for module, allowed in SIMULATION_IMPORTS.items():
        assert {base for _, base, _ in records[module]} <= allowed, module
    assert records[SIMULATION] and all(
        kind == "eager" and base == f"{SIMULATION}.histograms"
        for kind, base, _ in records[SIMULATION]
    )
    assert records[IRF] and all(
        kind == "eager" and base == f"{IRF}.profiles" for kind, base, _ in records[IRF]
    )
    # Records may depend on profiles, never on preparation via an aggregate
    # initializer: preparation itself consumes those canonical records.
    assert {
        base
        for _, base, _ in records[f"{MEASUREMENT}.records"]
        if base == IRF or base.startswith(IRF + ".")
    } == {f"{IRF}.profiles"}
    assert records[PREPROCESSING] and all(
        kind == "eager" and base == f"{PREPROCESSING}.histograms"
        for kind, base, _ in records[PREPROCESSING]
    )
    assert records[EVALUATION] == [("eager", DIAGNOSTICS, DIAGNOSTIC_EXPORTS)]
    assert {base for _, base, _ in records[DIAGNOSTICS]} == DIAGNOSTIC_IMPORTS
    for module, allowed in FACTUAL_IMPORTS.items():
        assert {base for _, base, _ in records[module]} == allowed, module
    for module, allowed in ANALYSIS_IMPORTS.items():
        assert {base for _, base, _ in records[module]} == allowed, module
    # Diagnostics legitimately uses a fitting result. This is not a dependency
    # of the factual modules: their own source imports are guarded separately
    # above, regardless of root/parent initializer costs observed at runtime.
    for module, imports in records.items():
        for _, base, names in imports:
            targets = _targets(base, names, set(records)) | {base}
            assert not targets.intersection(COMPATIBILITY_SURFACES), (module, targets)
            if module.startswith((f"{PACKAGE}.bayesian", BAYESIAN)):
                assert not targets.intersection(
                    {
                        f"{PACKAGE}.experimental",
                        f"{MEASUREMENT}.adapters",
                    }
                ), (module, targets)


def test_benchmark_protocol_data_dependencies_preserve_existing_edge_kinds(
    package_imports,
):
    records, _ = package_imports
    tree = ast.parse(
        (SOURCE_ROOT / "benchmarks" / "__init__.py").read_text(encoding="utf-8")
    )
    assert len(tree.body) == 1 and ast.get_docstring(tree)
    assert records[BENCHMARKS] == []
    for module, allowed in BENCHMARK_IMPORTS.items():
        assert {base for _, base, _ in records[module]} == allowed, module
        assert all(kind == "eager" for kind, _, _ in records[module]), module

    def benchmark_edges(module):
        return {
            (kind, base)
            for kind, base, _ in records[module]
            if base == BENCHMARKS or base.startswith(BENCHMARKS + ".")
        }

    assert benchmark_edges(UNCERTAINTY_ML) == {
        ("typing", GENERALIZATION_PROTOCOL),
        ("typing", GENERALIZATION_DATASETS),
        ("typing", GENERALIZATION),
    }
    assert ("typing", GENERALIZATION, ("GeneralizationPreparedData",)) in records[
        UNCERTAINTY_ML
    ]
    assert benchmark_edges(PERSISTENCE_SQLITE) == {
        ("deferred", GENERALIZATION_DATASETS),
        ("deferred", GENERALIZATION),
        ("deferred", BAYESIAN_CALIBRATION),
        ("deferred", BAYESIAN_MISMATCH),
        ("deferred", UNCERTAINTY_BENCHMARK),
    }
    # Existing scientific provenance guard, not a protocol-independent metric layer.
    assert ("eager", GENERALIZATION_PROTOCOL, ("FINAL_ROBUSTNESS_TEST_IDS",)) in (
        records[UNCERTAINTY_EVALUATION]
    )
    assert benchmark_edges(GENERALIZATION) == {
        ("eager", GENERALIZATION_PROTOCOL),
        ("eager", GENERALIZATION_DATASETS),
    }
    assert benchmark_edges(UNCERTAINTY_BENCHMARK) == {
        ("eager", GENERALIZATION_PROTOCOL),
        ("eager", GENERALIZATION_DATASETS),
        ("eager", GENERALIZATION),
    }
    consumers = {
        (module, kind, names)
        for module, imports in records.items()
        for kind, base, names in imports
        if base == GENERALIZATION
    }
    assert consumers == {
        (UNCERTAINTY_ML, "typing", ("GeneralizationPreparedData",)),
        (UNCERTAINTY_BENCHMARK, "eager", ("GeneralizationPreparedData",)),
        (PERSISTENCE_SQLITE, "deferred", ("RobustnessMetrics",)),
    }
    foundations = (
        f"{PACKAGE}.physics",
        IRF,
        MEASUREMENT,
        SIMULATION,
        PREPROCESSING,
        ESTIMATION,
    )
    for module in records:
        if (
            module in FACTUAL
            or module == f"{ML}.models"
            or any(module == p or module.startswith(p + ".") for p in foundations)
        ):
            assert not benchmark_edges(module), module


@pytest.mark.parametrize(
    "module", [BENCHMARKS, GENERALIZATION_PROTOCOL, GENERALIZATION_DATASETS]
)
def test_fresh_benchmark_imports_do_not_aggregate_studies(tmp_path, module):
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence', 'tcspc_toolkit.ml',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.generalization_evaluation',
    {GENERALIZATION!r},
    'tcspc_toolkit.uncertainty_evaluation', 'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml',
    'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty.classical', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
    'tcspc_toolkit.irf_evaluation', 'tcspc_toolkit.benchmarks.irf', 'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.benchmarks.mismatch',
    'tcspc_toolkit.timing_evaluation', 'tcspc_toolkit.benchmarks.timing', 'tcspc_toolkit.bayesian_evaluation', 'tcspc_toolkit.benchmarks.bayesian_calibration',
    'tcspc_toolkit.bayesian_mismatch_evaluation', 'tcspc_toolkit.benchmarks.bayesian_mismatch',
]
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
assert not any(n == {BENCHMARKS!r} or n.startswith({BENCHMARKS + "."!r}) for n in sys.modules)
namespace = importlib.import_module({BENCHMARKS!r})
assert not any(n.startswith({BENCHMARKS + "."!r}) for n in sys.modules)
importlib.import_module({module!r})
expected = {{
    {BENCHMARKS!r}: set(),
    {GENERALIZATION_PROTOCOL!r}: {{{GENERALIZATION_PROTOCOL!r}}},
    {GENERALIZATION_DATASETS!r}: {{{GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r}}},
    'tcspc_toolkit.generalization': {{{GENERALIZATION_PROTOCOL!r}}},
    'tcspc_toolkit.generalization_datasets': {{{GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r}}},
}}[{module!r}]
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == expected
assert not hasattr(namespace, 'GeneralizationSuiteDefinition')
assert not hasattr(namespace, 'GeneralizationTestSuite')
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


# Final Issue-#3 inventory. Namespace-only initializers count as support, not
# scientific implementations. Historical facades remain supported through #12.
CLOSURE_CANONICAL_FILES = {
    "benchmarks/bayesian_calibration.py",
    "benchmarks/bayesian_mismatch.py",
    "benchmarks/generalization.py",
    "benchmarks/generalization_datasets.py",
    "benchmarks/generalization_protocol.py",
    "benchmarks/irf.py",
    "benchmarks/mismatch.py",
    "benchmarks/multiexponential.py",
    "benchmarks/timing.py",
    "benchmarks/uncertainty.py",
    "estimation/api.py",
    "estimation/baselines.py",
    "estimation/bayesian/posterior.py",
    "estimation/bayesian/predictive.py",
    "estimation/bayesian/sampling.py",
    "estimation/classical.py",
    "estimation/fitting.py",
    "estimation/multiexponential.py",
    "evaluation/attachments.py",
    "evaluation/conditional.py",
    "evaluation/core.py",
    "evaluation/diagnostics.py",
    "evaluation/plots.py",
    "evaluation/reconvolution.py",
    "evaluation/results.py",
    "irf/estimation.py",
    "irf/preparation.py",
    "irf/profiles.py",
    "irf/selection.py",
    "measurement/adapters.py",
    "measurement/io.py",
    "measurement/records.py",
    "ml/cross_validation.py",
    "ml/evaluation.py",
    "ml/models.py",
    "persistence/sqlite.py",
    "physics/convolution.py",
    "physics/forward_model.py",
    "physics/models.py",
    "preprocessing/features.py",
    "preprocessing/histograms.py",
    "preprocessing/representations.py",
    "simulation/datasets.py",
    "simulation/histograms.py",
    "uncertainty/classical.py",
    "uncertainty/evaluation.py",
    "uncertainty/ml.py",
}
CLOSURE_SUPPORT_FILES = {
    "__init__.py",
    "__main__.py",
    "benchmarks/__init__.py",
    "cli.py",
    "config.py",
    "estimation/__init__.py",
    "estimation/bayesian/__init__.py",
    "exceptions.py",
    "measurement/__init__.py",
    "ml/__init__.py",
    "physics/__init__.py",
    "uncertainty/__init__.py",
}


# Historical removed paths remain forbidden in Python, notebooks and docs.
REMOVED_FLAT_NAMES = set(REMOVED_FLAT_FACADES)


def _python_flat_facade_dependencies(source, package=None):
    """Find imports, embedded Python and literal patch/import targets, not labels.

    Notebook callers supply code-cell source only, never saved outputs.
    Historical SQL provenance labels and negative guards are not imports.
    Removed-name registries retain their separate negative import/pickle tests.
    """
    import re
    import textwrap

    found = []

    def facade(name):
        prefix, separator, rest = name.partition(".")
        first = rest.split(".")[0]
        return (
            first
            if separator and prefix == PACKAGE and first in REMOVED_FLAT_NAMES
            else None
        )

    class Visitor(ast.NodeVisitor):
        scope = "<module>"
        offset = 0
        embedded = False

        def record(self, node, kind, target, names=()):
            old = facade(target)
            if old:
                found.append((self.scope, kind, old, names, self.offset + node.lineno))

        def visit_FunctionDef(self, node):
            previous = self.scope
            self.scope = node.name
            self.generic_visit(node)
            self.scope = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                self.record(
                    node, "embedded-import" if self.embedded else "import", alias.name
                )

        def visit_ImportFrom(self, node):
            base = node.module or ""
            if node.level:
                if package is None:
                    return
                base = importlib.util.resolve_name("." * node.level + base, package)
            names = tuple((alias.name, alias.asname) for alias in node.names)
            kind = "embedded-import" if self.embedded else "import"
            self.record(node, kind, base, names)
            if base == PACKAGE:
                for alias in node.names:
                    self.record(node, kind, f"{base}.{alias.name}", names)

        def visit_Call(self, node):
            operation = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if operation in {
                "setattr",
                "delattr",
                "patch",
                "object",
                "import_module",
                "__import__",
            }:
                targets = node.args[:1] + [
                    k.value for k in node.keywords if k.arg == "target"
                ]
                for target in targets:
                    if isinstance(target, ast.Constant) and isinstance(
                        target.value, str
                    ):
                        self.record(node, "string-target", target.value)
            self.generic_visit(node)

        def visit_Constant(self, node):
            if not isinstance(node.value, str):
                return
            # Includes concatenated subprocess strings, without executing them.
            if not re.search(r"(?m)^\s*(?:from|import)\s+", node.value):
                return
            try:
                snippet = ast.parse(textwrap.dedent(node.value))
            except SyntaxError:
                # F-string fragments can be incomplete Python. Inspect complete
                # literal import statements, not strings in their deny lists.
                statements = []
                pattern = r"(?m)^[ \t]*(?:from\s+[\w.]+\s+import\s+(?:\([^)]*\)|[^\n]+)|import\s+[^\n]+)"
                for match in re.finditer(pattern, node.value):
                    try:
                        statement = ast.parse(textwrap.dedent(match.group()))
                    except SyntaxError:
                        continue  # Dynamically assembled module name, not a literal import.
                    ast.increment_lineno(
                        statement, node.value[: match.start()].count("\n")
                    )
                    statements.extend(statement.body)
                snippet = ast.Module(body=statements, type_ignores=[])
            previous = self.offset, self.embedded
            self.offset += node.lineno - 1
            self.embedded = True
            self.visit(snippet)
            self.offset, self.embedded = previous

    Visitor().visit(ast.parse(source))
    return found


def test_python_consumers_have_no_removed_flat_dependencies():
    """No compatibility-import exceptions remain in first-party Python code."""
    assert len(REMOVED_FLAT_NAMES) == 38
    repository = SOURCE_ROOT.parents[1]
    violations = []
    for directory in (
        SOURCE_ROOT,
        repository / "scripts",
        repository / "examples",
        repository / "tests",
    ):
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(repository).as_posix()
            package = None
            if path.is_relative_to(SOURCE_ROOT):
                package = ".".join(
                    (PACKAGE, *path.relative_to(SOURCE_ROOT).parent.parts)
                )
            for scope, kind, old, names, line in _python_flat_facade_dependencies(
                path.read_text(encoding="utf-8"),
                package,
            ):
                violations.append((relative, line, scope, kind, old))
    assert not violations, violations


@pytest.mark.parametrize(
    "template",
    [
        "import {package}.fitting",
        "import {package}.fitting as fits",
        "from {package}.fitting import ReconvolutionFitResult",
        "from {package} import fitting as fits",
        "from {package} import (\n    fitting,\n)",
        "from . import fitting",
        "from .fitting import ReconvolutionFitResult",
        "code = 'import {package}.fitting'",
        "code = ('from {package}.fitting import '\n        'ReconvolutionFitResult\\n')",
        "code = '''import {package}.fitting\nnot yet formatted ?'''",
        'monkeypatch.setattr("{package}.fitting.curve_fit", replacement)',
        'mock.patch("{package}.fitting.curve_fit")',
        'mock.patch(target="{package}.fitting.curve_fit")',
        'importlib.import_module("{package}.fitting")',
    ],
)
def test_python_retirement_guard_detects_executable_flat_dependencies(template):
    source = template.format(package=PACKAGE)
    assert _python_flat_facade_dependencies(source, PACKAGE)


@pytest.mark.parametrize(
    "template",
    [
        "from {package} import fit_monoexponential_reconvolution",
        "from {package} import persistence as store",
        "from {package}.irf import generate_gaussian_irf",
        "from {package}.preprocessing import validate_histogram",
        "from {package}.simulation import sample_photon_counts",
        "from {package}.evaluation import calculate_residuals",
        "from {package}.persistence import connect_database",
        "from {package}.estimation.fitting import ReconvolutionFitResult",
        'source_type = "{package}.fitting.ReconvolutionFitResult"',
        'blocked = ["{package}.fitting", "{package}.ml_models"]',
        'monkeypatch.setattr("{package}.estimation.fitting.curve_fit", replacement)',
    ],
)
def test_python_retirement_guard_preserves_root_package_and_provenance_contracts(
    template,
):
    assert not _python_flat_facade_dependencies(
        template.format(package=PACKAGE), PACKAGE
    )


def _notebook_code_sources(document):
    """Ignore outputs/metadata and scientific Markdown, without executing cells."""
    for index, cell in enumerate(document["cells"], start=1):
        if cell["cell_type"] == "code":
            yield index, "".join(cell["source"])


@pytest.mark.parametrize(
    "path",
    sorted(SOURCE_ROOT.parents[1].joinpath("notebooks").glob("*.ipynb")),
    ids=lambda path: path.stem,
)
def test_notebook_code_compiles_and_is_ready_for_flat_facade_retirement(path):
    import json

    cells = list(_notebook_code_sources(json.loads(path.read_text(encoding="utf-8"))))
    assert cells
    for index, source in cells:
        compile(source, f"{path.name}:cell {index}", "exec")  # Compilation only.
        assert not _python_flat_facade_dependencies(source), (path.name, index)


def test_notebook_imports_resolve_in_fresh_processes_without_running_workflows(
    tmp_path,
):
    import json

    for path in sorted(SOURCE_ROOT.parents[1].joinpath("notebooks").glob("*.ipynb")):
        imports = []
        for _, source in _notebook_code_sources(
            json.loads(path.read_text(encoding="utf-8"))
        ):
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    PACKAGE
                ):
                    imports.append(ast.unparse(node))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == PACKAGE or alias.name.startswith(
                            PACKAGE + "."
                        ):
                            imports.append(ast.unparse(ast.Import(names=[alias])))
        assert imports, path.name
        # Execute ONLY selected import statements, not setup/configuration code,
        # fitting, plotting, file I/O, Week-9 cache writes or Bayesian sampling.
        script = "\n".join(
            [
                "import sys, importlib.abc",
                "class NoEmcee(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname == 'emcee' or fullname.startswith('emcee.'):",
                "            raise AssertionError('Notebook imports must not load emcee')",
                "sys.meta_path.insert(0, NoEmcee())",
                *imports,
                "assert 'emcee' not in sys.modules",
            ]
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, (
            path.name,
            completed.stdout,
            completed.stderr,
        )
        assert list(tmp_path.iterdir()) == [], path.name


def test_notebook_readiness_scans_code_not_saved_outputs_or_markdown():
    historical = "from {package}.fitting import ReconvolutionFitResult".format(
        package=PACKAGE
    )
    document = {
        "cells": [
            {"cell_type": "markdown", "source": [historical]},
            {
                "cell_type": "code",
                "source": ["from tcspc_toolkit import fit_monoexponential_decay"],
                "outputs": [{"text": historical}],
                "metadata": {"historical_note": historical},
            },
            {"cell_type": "code", "source": [historical]},
        ]
    }
    sources = list(_notebook_code_sources(document))
    assert [index for index, _ in sources] == [2, 3]
    assert not _python_flat_facade_dependencies(sources[0][1])
    assert _python_flat_facade_dependencies(sources[1][1])


def _markdown_flat_facade_dependencies(source):
    """Inspect Python fences/inline import examples, not mapping/history prose.

    Historical names in tables, release notes and frozen provenance labels are
    not executable examples. Any historical *import example* needs a precise
    reviewed exception in the repository test, not a blanket docs exclusion.
    """
    import re
    import textwrap

    snippets = []
    fence = None
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines, start=1):
        marker = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence:
            delimiter, language, start, body = fence
            if (
                marker
                and marker[1][0] == delimiter[0]
                and len(marker[1]) >= len(delimiter)
            ):
                if language in {"", "python", "py", "pycon", "python3"}:
                    snippets.append((start, "".join(body)))
                fence = None
            else:
                body.append(line)
        elif marker:
            fence = (marker[1], marker[2].strip().lower(), index + 1, [])
        else:
            for inline in re.finditer(r"`([^`\n]+)`", line):
                if re.match(r"(?:from|import)\s+", inline[1]):
                    snippets.append((index, inline[1]))
    assert fence is None, "Unclosed documentation fence"
    found = []
    for start, snippet in snippets:
        snippet = textwrap.dedent(snippet)
        snippet = re.sub(r"(?m)^(?:>>> |\.\.\. )", "", snippet)
        try:
            ast.parse(snippet)
        except SyntaxError:
            # Incomplete examples/REPL output: reuse the embedded-import parser.
            snippet = f"example = {snippet!r}"
        for scope, kind, old, names, line in _python_flat_facade_dependencies(snippet):
            found.append((start + line - 1, scope, kind, old, names))
    return found


def test_current_documentation_examples_are_ready_for_flat_facade_retirement():
    repository = SOURCE_ROOT.parents[1]
    paths = sorted(
        set(repository.glob("*.md"))
        | set(repository.joinpath("docs").rglob("*.md"))
        | set(repository.joinpath("examples").rglob("*.md"))
        | set(repository.joinpath("data/examples").rglob("*.md"))
    )
    assert paths
    violations = []
    for path in paths:
        violations.extend(
            (path.relative_to(repository).as_posix(), *item)
            for item in _markdown_flat_facade_dependencies(
                path.read_text(encoding="utf-8")
            )
        )
    # No current executable documentation example needs a historical exception.
    assert not violations, violations


@pytest.mark.parametrize(
    "template",
    [
        "```python\nfrom {package}.fitting import ReconvolutionFitResult\n```",
        "~~~py\nfrom {package} import fitting\n~~~",
        "```pycon\n>>> import {package}.fitting\n>>> result\n<result>\n```",
        "```\nfrom {package}.fitting import (\n    ReconvolutionFitResult,\n)\n```",
        "Use `from {package}.fitting import ReconvolutionFitResult` here.",
        '```python\nimportlib.import_module("{package}.fitting")\n```',
        '```python\nmonkeypatch.setattr("{package}.fitting.curve_fit", replacement)\n```',
    ],
)
def test_documentation_readiness_detects_active_examples(template):
    assert _markdown_flat_facade_dependencies(template.format(package=PACKAGE))


@pytest.mark.parametrize(
    "template",
    [
        "Historical mapping: `{package}.fitting` -> `estimation.fitting`.",
        "| Historical | Canonical |\n| `{package}.fitting` | `estimation.fitting` |",
        "```text\n{package}.fitting.ReconvolutionFitResult\n```",
        '```python\nsource_type = "{package}.fitting.ReconvolutionFitResult"\n```',
        "```python\nfrom {package} import fit_monoexponential_reconvolution\n```",
        "```python\nfrom {package}.persistence import connect_database\n```",
    ],
)
def test_documentation_readiness_preserves_historical_labels_and_public_apis(template):
    assert not _markdown_flat_facade_dependencies(template.format(package=PACKAGE))


def test_final_inventory_has_no_unplanned_flat_scientific_implementation():
    facade_files = {name + "/__init__.py" for name in RETAINED_PACKAGE_APIS}
    expected = CLOSURE_CANONICAL_FILES | CLOSURE_SUPPORT_FILES | facade_files
    actual = {
        path.relative_to(SOURCE_ROOT).as_posix() for path in SOURCE_ROOT.rglob("*.py")
    }
    assert actual == expected
    assert len(actual) == 64
    assert len(CLOSURE_CANONICAL_FILES) == 47
    assert len(CLOSURE_SUPPORT_FILES) == 12
    assert len(facade_files) == 5
    assert len(expected) == 47 + 12 + 5  # Each file belongs to exactly one class.
    for relative in CLOSURE_SUPPORT_FILES:
        if "/" in relative:
            tree = ast.parse((SOURCE_ROOT / relative).read_text(encoding="utf-8"))
            assert len(tree.body) == 1 and ast.get_docstring(tree), relative


def test_retained_package_apis_and_root_keep_canonical_identity():
    import pickle

    for old_name, new_name in RETAINED_PACKAGE_APIS.items():
        old = importlib.import_module(f"{PACKAGE}.{old_name}")
        new = importlib.import_module(f"{PACKAGE}.{new_name}")
        tree = ast.parse(Path(old.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                assert isinstance(node.value.value, str)
            elif isinstance(node, ast.ImportFrom):
                base = node.module
                if node.level:
                    base = importlib.util.resolve_name(
                        "." * node.level + base, old.__package__
                    )
                assert base == new.__name__, old_name
                for alias in node.names:
                    assert alias.name != "*"
                    name = alias.asname or alias.name
                    assert getattr(old, name) is getattr(new, alias.name), (
                        old_name,
                        name,
                    )
                    imported.add(name)
            elif isinstance(node, ast.Assign):
                assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                assert node.targets[0].id == "__all__"
                assert tuple(ast.literal_eval(node.value)) == tuple(old.__all__)
            else:
                pytest.fail(
                    f"Implementation logic in facade {old_name}: {ast.dump(node)}"
                )
        assert set(old.__all__) <= imported
        for name in imported - {"__all__"}:
            assert getattr(old, name) is getattr(new, name)
            assert pickle.loads(f"c{old.__name__}\n{name}\n.".encode()) is getattr(
                new, name
            )
    # The root's independently pinned ordered 66-name contract has canonical identity.
    root = importlib.import_module(PACKAGE)
    tree = ast.parse((SOURCE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    identities = {
        alias.asname or alias.name: getattr(
            importlib.import_module(node.module), alias.name
        )
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert len(root.__all__) == 66
    for name in root.__all__:
        assert getattr(root, name) is identities[name]


def test_no_production_module_fabricates_module_metadata():
    for path in SOURCE_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                assert not any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "__module__"
                    or isinstance(target, ast.Name)
                    and target.id == "__module__"
                    for target in targets
                ), (path, node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "setattr" and len(node.args) >= 2:
                    assert not (
                        isinstance(node.args[1], ast.Constant)
                        and node.args[1].value == "__module__"
                    ), (path, node.lineno)


def test_developer_scripts_use_canonical_imports_and_load_without_workflows(
    package_imports,
):
    import runpy

    records, _ = package_imports
    for path in sorted((SOURCE_ROOT.parents[1] / "scripts").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for _, base, names in _imports(source, path.stem):
            assert not (_targets(base, names, set(records)) & COMPATIBILITY_SURFACES), (
                path
            )
            assert base not in COMPATIBILITY_SURFACES, path
        # Loading definitions is not running main(), profiles, MCMC or corrections.
        namespace = runpy.run_path(str(path), run_name="issue3_import_audit")
        assert callable(namespace["main"])


def test_correction_provenance_source_is_canonical_implementation():
    script = (
        SOURCE_ROOT.parents[1] / "scripts" / "correct_issue4_mismatch_references.py"
    )
    tree = ast.parse(script.read_text(encoding="utf-8"))
    calls = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "correction_source_sha256"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "_sha256" and len(call.args) == 1 and not call.keywords
    # Resolve only the literal path expression; never execute the correction workflow.
    expression = call.args[0]
    parts = []
    while isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
        part = ast.literal_eval(expression.right)
        assert isinstance(part, str)
        parts.append(part)
        expression = expression.left
    assert isinstance(expression, ast.Name) and expression.id == "ROOT"
    source = SOURCE_ROOT.parents[1].joinpath(*reversed(parts)).resolve()
    canonical = importlib.import_module(BAYESIAN_MISMATCH)
    assert source.is_file()
    assert source == Path(canonical.__file__).resolve()
    assert source != (SOURCE_ROOT / "bayesian_mismatch_evaluation.py").resolve()


def test_no_test_patches_a_historical_facade_as_an_implementation(package_imports):
    records, _ = package_imports
    modules = set(records) | MIGRATED_SHIMS
    observed_exceptions = set()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update(
                    {
                        a.asname or a.name: a.name
                        for a in node.names
                        if a.name in modules
                    }
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                aliases.update(
                    {
                        a.asname or a.name: f"{node.module}.{a.name}"
                        for a in node.names
                        if f"{node.module}.{a.name}" in modules
                    }
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            operation = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if operation not in {"setattr", "patch", "object"}:
                continue
            target = node.args[0]
            owner = aliases.get(target.id) if isinstance(target, ast.Name) else None
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                candidates = [
                    m
                    for m in modules
                    if target.value == m or target.value.startswith(m + ".")
                ]
                owner = max(candidates, key=len) if candidates else None
            if owner not in COMPATIBILITY_SURFACES:
                continue
            attribute = (
                node.args[1].value
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                else None
            )
            seam = (path.name, owner, attribute)
            # The public example itself calls store.connect_database: consumer seam,
            # not an expectation that a facade proxies implementation globals.
            assert seam == (
                "test_persistence_example.py",
                PERSISTENCE,
                "connect_database",
            )
            observed_exceptions.add(seam)
    assert observed_exceptions == {
        ("test_persistence_example.py", PERSISTENCE, "connect_database")
    }


# Stage-8C persistence inventory, pinned independently of the package facade.
PERSISTENCE_EXPORTS = (
    "SCHEMA_VERSION",
    "PersistenceSchemaError",
    "initialize_database",
    "connect_database",
    "transaction",
    "DuplicatePolicy",
    "PersistenceConflictError",
    "record_run",
    "record_artifact",
    "record_irf_source",
    "record_legacy_irf_source",
    "record_prepared_irf",
    "record_supplied_kernel",
    "record_simulation_condition",
    "record_issue4_condition",
    "record_measurement",
    "record_synthetic_observation",
    "record_benchmark_observation",
    "link_run_measurement",
    "bayesian_model_configuration",
    "record_model_version",
    "record_model_assumption",
    "record_trusted_lifetime_reference",
    "record_pseudo_true_reference",
    "record_issue4_pseudo_true_reference",
    "record_reconvolution_fit",
    "record_bayesian_result",
    "record_issue4_bayesian_result",
    "record_regression_predictions",
    "record_baseline_predictions",
    "record_scalar_prediction",
    "record_prediction_intervals",
    "record_quantile_intervals",
    "record_conformal_intervals",
    "record_uncertainty_scores",
    "record_classical_covariance_uncertainty",
    "record_parametric_bootstrap_uncertainty",
    "BenchmarkMetricScope",
    "record_interval_metrics",
    "record_quantile_interval_metrics",
    "record_uncertainty_score_metrics",
    "record_selective_prediction_metrics",
    "record_repeated_poisson_metrics",
    "record_regression_metrics",
    "record_robustness_metrics",
    "record_reconvolution_benchmark_metrics",
    "record_ml_interval_scorecard_row",
    "record_ml_uncertainty_scorecard_row",
    "QueryResult",
    "query_to_dataframe",
    "query_runs",
    "query_artifacts",
    "query_measurements",
    "query_irf_sources",
    "query_prepared_irfs",
    "query_results",
    "query_uncertainty",
    "query_bayesian",
    "query_references",
    "query_metrics",
    "record_week9_interval_scorecard_row",
    "record_week9_score_only_scorecard_row",
)
PERSISTENCE_EAGER_IMPORTS = {
    "__future__",
    "hashlib",
    "itertools",
    "json",
    "math",
    "sqlite3",
    "collections.abc",
    "contextlib",
    "dataclasses",
    "datetime",
    "enum",
    "pathlib",
    "typing",
    "numpy",
    "numpy.typing",
}
PERSISTENCE_DEFERRED_IMPORT_COUNTS = {
    "tcspc_toolkit.preprocessing.histograms": 2,
    "tcspc_toolkit.irf.profiles": 2,
    "tcspc_toolkit.irf.estimation": 2,
    "tcspc_toolkit.measurement.records": 3,
    "tcspc_toolkit.irf.preparation": 2,
    "tcspc_toolkit.benchmarks.bayesian_calibration": 3,
    "tcspc_toolkit.benchmarks.bayesian_mismatch": 4,
    "tcspc_toolkit.benchmarks.generalization_datasets": 1,
    "tcspc_toolkit.ml.evaluation": 3,
    "tcspc_toolkit.estimation.bayesian.posterior": 3,
    "tcspc_toolkit.estimation.bayesian.sampling": 5,
    "tcspc_toolkit.estimation.classical": 2,
    "tcspc_toolkit.estimation.fitting": 1,
    "tcspc_toolkit.estimation.bayesian.predictive": 3,
    "tcspc_toolkit.uncertainty.evaluation": 8,
    "tcspc_toolkit.benchmarks.uncertainty": 2,
    "tcspc_toolkit.uncertainty.ml": 4,
    "tcspc_toolkit.uncertainty.classical": 4,
    "tcspc_toolkit.benchmarks.generalization": 1,
    "pandas": 2,
}
PERSISTENCE_STORED_IDENTITIES = {
    "tcspc_toolkit.fitting.ReconvolutionFitResult",
    "tcspc_toolkit.bayesian_evaluation.Issue4Bayesian",
    "tcspc_toolkit.bayesian_sampling.BayesianReconvolutionResult",
    "tcspc_toolkit.baselines.predict_constant_mean_baseline",
    "tcspc_toolkit.baselines.estimate_lifetime_from_mean_arrival",
    "tcspc_toolkit.classical_evaluation.ReconvolutionCurveResult",
    "tcspc_toolkit.ml_evaluation.RegressionBenchmarkResult",
}


def test_persistence_package_layout_and_direct_public_facade():
    assert not (SOURCE_ROOT / "persistence.py").exists()
    directory = SOURCE_ROOT / "persistence"
    assert {path.name for path in directory.glob("*.py")} == {
        "__init__.py",
        "sqlite.py",
    }
    tree = ast.parse((directory / "__init__.py").read_text(encoding="utf-8"))
    assert ast.get_docstring(tree)
    assert len(tree.body) == 3
    exports = tree.body[1]
    assert isinstance(exports, ast.ImportFrom)
    assert exports.level == 1 and exports.module == "sqlite"
    assert tuple(alias.name for alias in exports.names) == PERSISTENCE_EXPORTS
    assert all(alias.asname is None for alias in exports.names)
    declaration = tree.body[2]
    assert isinstance(declaration, ast.Assign)
    assert len(declaration.targets) == 1 and declaration.targets[0].id == "__all__"
    assert tuple(ast.literal_eval(declaration.value)) == PERSISTENCE_EXPORTS
    # The only real schema definitions live in sqlite.py, never in the facade.
    for name in ("SCHEMA_VERSION", "_TABLES", "_INDEXES"):
        definitions = [
            path.name
            for path in directory.glob("*.py")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == name
        ]
        assert definitions == ["sqlite.py"], (name, definitions)


def test_persistence_public_identity_aliases_and_historical_pickle_lookup():
    import inspect
    import pickle

    old = importlib.import_module(PERSISTENCE)
    new = importlib.import_module(PERSISTENCE_SQLITE)
    assert Path(old.__file__).name == "__init__.py"
    assert tuple(old.__all__) == PERSISTENCE_EXPORTS
    assert {name for name in vars(old) if not name.startswith("_")} == {
        *PERSISTENCE_EXPORTS,
        "sqlite",
    }
    for name in PERSISTENCE_EXPORTS:
        value = getattr(new, name)
        assert getattr(old, name) is value
        # Protocol-0 GLOBAL resolves historical module-qualified lookups.
        historical = f"c{PERSISTENCE}\n{name}\n.".encode()
        assert pickle.loads(historical) is value
        if inspect.isfunction(value) or inspect.isclass(value):
            assert value.__module__ == PERSISTENCE_SQLITE
            assert pickle.loads(pickle.dumps(value)) is value
    assert (
        old.record_week9_interval_scorecard_row is new.record_ml_interval_scorecard_row
    )
    assert (
        old.record_week9_score_only_scorecard_row
        is new.record_ml_uncertainty_scorecard_row
    )
    for value in (
        new.QueryResult(("id",), ((1,),)),
        new.BenchmarkMetricScope(1, "selection", "development", "generating_mono"),
        new.PersistenceSchemaError("schema"),
        new.PersistenceConflictError("conflict"),
    ):
        payload = pickle.dumps(value, protocol=0)
        qualified = PERSISTENCE_SQLITE.encode() + b"\n"
        assert qualified in payload
        restored = pickle.loads(
            payload.replace(qualified, PERSISTENCE.encode() + b"\n")
        )
        assert type(restored) is type(value)
        if isinstance(value, Exception):
            assert restored.args == value.args
        else:
            assert restored == value


def test_persistence_schema_v1_fingerprint_and_stored_identities():
    implementation = importlib.import_module(PERSISTENCE_SQLITE)
    assert implementation.SCHEMA_VERSION == 1
    assert implementation._definition_sha256() == (
        "1edaa036c1e9ffe00927e87d7d14aad89e995c11a0a68172f6cc68842e833cf9"
    )
    tree = ast.parse(Path(implementation.__file__).read_text(encoding="utf-8"))
    stored = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(PACKAGE + ".")
    ]
    assert len(stored) == 7
    assert set(stored) == PERSISTENCE_STORED_IDENTITIES
    # Executable adapter imports are canonical; stored identities stay historical.
    imports = {
        base
        for _, base, _ in _imports(
            Path(implementation.__file__).read_text(encoding="utf-8"),
            PERSISTENCE_SQLITE,
        )
    }
    assert {
        f"{ESTIMATION}.fitting",
        f"{ESTIMATION}.classical",
        f"{ML}.evaluation",
        BAYESIAN_CALIBRATION,
        f"{BAYESIAN}.sampling",
    } <= imports
    assert not imports & COMPATIBILITY_SURFACES


def test_persistence_import_classification_and_incoming_edges(package_imports):
    from collections import Counter

    records, packages = package_imports
    assert records[PERSISTENCE] == [("eager", PERSISTENCE_SQLITE, PERSISTENCE_EXPORTS)]
    implementation = records[PERSISTENCE_SQLITE]
    eager = [base for kind, base, _ in implementation if kind == "eager"]
    assert len(eager) == 15 and set(eager) == PERSISTENCE_EAGER_IMPORTS
    assert Counter(base for kind, base, _ in implementation if kind == "deferred") == (
        PERSISTENCE_DEFERRED_IMPORT_COUNTS
    )
    assert not any(kind == "typing" for kind, _, _ in implementation)
    assert (
        sum(
            kind == "deferred" and base.startswith(PACKAGE + ".")
            for kind, base, _ in implementation
        )
        == 55
    )
    incoming = [
        (module, kind, base)
        for module, imports in records.items()
        for kind, base, names in imports
        if _targets(base, names, set(records)) & {PERSISTENCE, PERSISTENCE_SQLITE}
    ]
    assert incoming == [(PERSISTENCE, "eager", PERSISTENCE_SQLITE)]
    graph = _eager_graph(records, packages)
    # The root initializer is already running, not a new dependency edge.
    assert graph[PERSISTENCE] == {PERSISTENCE_SQLITE}
    assert graph[PERSISTENCE_SQLITE] == set()
    tuple(TopologicalSorter(graph).static_order())


@pytest.mark.parametrize("module", [PERSISTENCE, PERSISTENCE_SQLITE])
def test_fresh_persistence_import_has_no_database_or_scientific_side_effects(
    tmp_path, module
):
    script = f"""
import builtins
import importlib
import importlib.abc
import os
import sys
class NoSampler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'emcee' or fullname.startswith('emcee.'):
            raise AssertionError('persistence import requires optional emcee')
sys.meta_path.insert(0, NoSampler())
# Root initialization has inherited ML/pandas cost; this stage does not optimize it.
import tcspc_toolkit
before = set(sys.modules)
assert {PERSISTENCE!r} not in before
seen = {{}}
original_import = builtins.__import__
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller in ({PERSISTENCE!r}, {PERSISTENCE_SQLITE!r}):
        seen.setdefault(caller, set()).add(name)
        if caller == {PERSISTENCE_SQLITE!r}:
            assert name in {PERSISTENCE_EAGER_IMPORTS!r}, (caller, name)
    return original_import(name, globals, locals, fromlist, level)
def no_database_or_writes(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('import opened a database')
    if event == 'open':
        mode, flags = args[1:3]
        assert not isinstance(mode, str) or not any(c in mode for c in 'wax+'), args
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC), args
sys.addaudithook(no_database_or_writes)
builtins.__import__ = track
importlib.import_module({module!r})
assert seen[{PERSISTENCE!r}] == {{'sqlite'}}
assert seen[{PERSISTENCE_SQLITE!r}] == {PERSISTENCE_EAGER_IMPORTS!r}
added = set(sys.modules) - before
blocked = (
    'pandas', 'sklearn', 'emcee', 'matplotlib', 'tcspc_toolkit.benchmarks',
    'tcspc_toolkit.uncertainty', 'tcspc_toolkit.estimation', 'tcspc_toolkit.ml',
)
assert not any(n == b or n.startswith(b + '.') for n in added for b in blocked), added
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", [GENERALIZATION])
def test_fresh_generalization_execution_uses_only_canonical_dependencies(
    tmp_path, module
):
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = {sorted(MIGRATED_SHIMS - {f"{PACKAGE}.generalization_evaluation"})!r} + [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.uncertainty_evaluation',
    'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml', 'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty.classical',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty', 'tcspc_toolkit.irf_evaluation', 'tcspc_toolkit.benchmarks.irf',
    'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.benchmarks.mismatch', 'tcspc_toolkit.timing_evaluation', 'tcspc_toolkit.benchmarks.timing',
    'tcspc_toolkit.bayesian_evaluation', 'tcspc_toolkit.benchmarks.bayesian_calibration', 'tcspc_toolkit.bayesian_mismatch_evaluation', 'tcspc_toolkit.benchmarks.bayesian_mismatch',
]
if {module!r} == {GENERALIZATION!r}:
    blocked.append('tcspc_toolkit.generalization_evaluation')
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
exports = tuple(tcspc_toolkit.__all__)
assert len(exports) == 66
namespace = importlib.import_module({BENCHMARKS!r})
assert not any(n.startswith({BENCHMARKS + "."!r}) for n in sys.modules)
importlib.import_module({module!r})
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == {{
    {GENERALIZATION!r}, {GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r},
}}
assert not hasattr(namespace, 'GeneralizationPreparedData')
assert tuple(tcspc_toolkit.__all__) == exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_fresh_typing_and_deferred_consumers_do_not_load_generalization_execution(
    tmp_path,
):
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = [{GENERALIZATION!r}, 'tcspc_toolkit.generalization_evaluation', 'emcee']
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith('emcee.'):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('import opened a database')
sys.meta_path.insert(0, BoundaryGuard())
sys.addaudithook(no_database)
for module in ({UNCERTAINTY_ML!r}, 'tcspc_toolkit.persistence'):
    importlib.import_module(module)
assert not attempts, attempts
assert not set(blocked).intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_week9_benchmark_has_only_existing_eager_dependencies_and_local_consumers(
    package_imports,
):
    records, _ = package_imports
    assert {base for _, base, _ in records[UNCERTAINTY_BENCHMARK]} == BENCHMARK_IMPORTS[
        UNCERTAINTY_BENCHMARK
    ]
    assert all(kind == "eager" for kind, _, _ in records[UNCERTAINTY_BENCHMARK])
    consumers = [
        (module, kind, names)
        for module, imports in records.items()
        for kind, base, names in imports
        if UNCERTAINTY_BENCHMARK in _targets(base, names, set(records))
    ]
    assert {(module, kind) for module, kind, _ in consumers} == {
        (PERSISTENCE_SQLITE, "deferred"),
    }
    assert [
        (kind, names)
        for module, kind, names in consumers
        if module == PERSISTENCE_SQLITE
    ] == [
        ("deferred", ("CONFORMALIZED_QUANTILE_METHOD_ID",)),
        ("deferred", ("CONFORMALIZED_QUANTILE_METHOD_ID",)),
    ]
    # This exact consumer set also forbids reverse edges from reusable layers.
    assert len(consumers) == 2
    assert not (SOURCE_ROOT / "uncertainty" / "conformal.py").exists()
    tree = ast.parse(
        (SOURCE_ROOT / "benchmarks" / "uncertainty.py").read_text(encoding="utf-8")
    )
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {
        "ConformalizedQuantileCalibrationResult",
        "_finite_sample_conformal_quantile",
        "conformalize_quantile_gradient_boosting",
        "ConformalizedQuantileExternalResult",
        "evaluate_frozen_conformalized_quantile",
    } <= definitions


@pytest.mark.parametrize("module", [UNCERTAINTY_BENCHMARK])
def test_fresh_week9_benchmark_import_preserves_namespace_and_optional_boundaries(
    tmp_path, module
):
    blocked = sorted(MIGRATED_SHIMS - {module}) + [
        PERSISTENCE,
        BAYESIAN,
        IRF_STUDY,
        BAYESIAN_CALIBRATION,
        BAYESIAN_MISMATCH,
        MISMATCH,
        TIMING,
        "emcee",
        "sqlite3",
        "matplotlib",
    ]
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = {blocked!r}
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
exports = tuple(tcspc_toolkit.__all__)
assert len(exports) == 66
inherited = set(sys.modules)
# The root already loads sklearn. Each namespace adds only itself, no scientific
# children or new sklearn/sampling/database/plotting dependencies.
namespace = importlib.import_module({BENCHMARKS!r})
assert set(sys.modules) - inherited == {{{BENCHMARKS!r}}}
uncertainty = importlib.import_module({UNCERTAINTY!r})
assert set(sys.modules) - inherited == {{{BENCHMARKS!r}, {UNCERTAINTY!r}}}
importlib.import_module({module!r})
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == {{
    {UNCERTAINTY_BENCHMARK!r}, {GENERALIZATION!r},
    {GENERALIZATION_PROTOCOL!r}, {GENERALIZATION_DATASETS!r},
}}
assert {{n for n in sys.modules if n.startswith({UNCERTAINTY + "."!r})}} == {{
    {UNCERTAINTY_EVALUATION!r}, {UNCERTAINTY_ML!r}, {UNCERTAINTY_CLASSICAL!r},
}}
assert not hasattr(namespace, 'ConformalizedQuantileCalibrationResult')
assert not hasattr(uncertainty, 'ConformalizedQuantileCalibrationResult')
assert tuple(tcspc_toolkit.__all__) == exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_uncertainty_evaluation_retains_protocol_policy_and_consumer_edge_kinds(
    package_imports,
):
    records, _ = package_imports
    tree = ast.parse(
        (SOURCE_ROOT / "uncertainty" / "__init__.py").read_text(encoding="utf-8")
    )
    assert len(tree.body) == 1 and ast.get_docstring(tree)
    assert records[UNCERTAINTY] == []
    assert {
        base for _, base, _ in records[UNCERTAINTY_EVALUATION]
    } == UNCERTAINTY_EVALUATION_IMPORTS
    assert all(kind == "eager" for kind, _, _ in records[UNCERTAINTY_EVALUATION])
    internal = [
        (kind, base, names)
        for kind, base, names in records[UNCERTAINTY_EVALUATION]
        if base.startswith(PACKAGE + ".")
    ]
    assert internal == [
        ("eager", GENERALIZATION_PROTOCOL, ("FINAL_ROBUSTNESS_TEST_IDS",))
    ]
    consumers = [
        (module, kind, names)
        for module, imports in records.items()
        for kind, base, names in imports
        if base == UNCERTAINTY_EVALUATION
    ]
    assert {(module, kind) for module, kind, _ in consumers} == {
        (UNCERTAINTY_ML, "eager"),
        (UNCERTAINTY_CLASSICAL, "eager"),
        (UNCERTAINTY_BENCHMARK, "eager"),
        (PERSISTENCE_SQLITE, "deferred"),
    }
    assert len(consumers) == 11  # Three eager consumers, eight local imports.
    assert sum(module == PERSISTENCE_SQLITE for module, _, _ in consumers) == 8
    foundations = (
        f"{PACKAGE}.physics",
        IRF,
        MEASUREMENT,
        SIMULATION,
        PREPROCESSING,
        ESTIMATION,
    )
    for module, imports in records.items():
        if (
            module in FACTUAL
            or module == f"{ML}.models"
            or any(module == p or module.startswith(p + ".") for p in foundations)
        ):
            for _, base, names in imports:
                targets = _targets(base, names, set(records)) | {base}
                assert not any(
                    t == UNCERTAINTY
                    or t.startswith(UNCERTAINTY + ".")
                    or t == f"{PACKAGE}.uncertainty_evaluation"
                    for t in targets
                ), module


@pytest.mark.parametrize("module", [UNCERTAINTY, UNCERTAINTY_EVALUATION])
def test_fresh_uncertainty_imports_preserve_initializer_and_protocol_only_boundary(
    tmp_path, module
):
    # Keep the real root, which already loads sklearn. Attribute new imports to
    # their source and compare added modules; do not replace root initialization.
    script = f"""
import builtins
import importlib
import importlib.abc
import sys
attempts = []
blocked = {sorted(MIGRATED_SHIMS - {f"{PACKAGE}.uncertainty_evaluation"})!r} + [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence', 'tcspc_toolkit.ml',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml',
    'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty.classical', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
    'tcspc_toolkit.evaluation.attachments', {GENERALIZATION!r}, {GENERALIZATION_DATASETS!r},
    'tcspc_toolkit.irf_evaluation', 'tcspc_toolkit.benchmarks.irf', 'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.benchmarks.mismatch',
    'tcspc_toolkit.timing_evaluation', 'tcspc_toolkit.benchmarks.timing', 'tcspc_toolkit.bayesian_evaluation', 'tcspc_toolkit.benchmarks.bayesian_calibration',
    'tcspc_toolkit.bayesian_mismatch_evaluation', 'tcspc_toolkit.benchmarks.bayesian_mismatch',
]
if {module!r} != 'tcspc_toolkit.uncertainty_evaluation':
    blocked.append('tcspc_toolkit.uncertainty_evaluation')
if {module!r} == {UNCERTAINTY!r}:
    blocked.extend([{BENCHMARKS!r}, {UNCERTAINTY_EVALUATION!r}])
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
root_exports = tuple(tcspc_toolkit.__all__)
assert len(root_exports) == 66
assert not any(n == {UNCERTAINTY!r} or n.startswith({UNCERTAINTY + "."!r}) for n in sys.modules)
assert not any(n == {BENCHMARKS!r} or n.startswith({BENCHMARKS + "."!r}) for n in sys.modules)
inherited = set(sys.modules)
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {UNCERTAINTY!r} or caller.startswith({UNCERTAINTY + "."!r}):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = track
try:
    namespace = importlib.import_module({UNCERTAINTY!r})
    assert set(sys.modules) - inherited == {{{UNCERTAINTY!r}}}
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert not seen.get({UNCERTAINTY!r})
if {module!r} == {UNCERTAINTY!r}:
    assert set(sys.modules) - inherited == {{{UNCERTAINTY!r}}}
    assert not seen
else:
    assert seen[{UNCERTAINTY_EVALUATION!r}] == {UNCERTAINTY_EVALUATION_IMPORTS!r}
    assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == {{
        {GENERALIZATION_PROTOCOL!r},
    }}
assert not any(n == 'sklearn' or n.startswith('sklearn.') for n in set(sys.modules) - inherited)
assert not hasattr(namespace, 'PredictionIntervalResult')
assert not hasattr(namespace, 'IntervalAttachment')
assert tuple(tcspc_toolkit.__all__) == root_exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


UNCERTAINTY_METHOD_IMPORTS = {
    UNCERTAINTY_ML: {
        "__future__",
        "dataclasses",
        "numpy",
        "pandas",
        "numpy.typing",
        "typing",
        "sklearn.base",
        "scipy.stats",
        "tcspc_toolkit.evaluation.core",
        "tcspc_toolkit.evaluation.results",
        "tcspc_toolkit.evaluation.attachments",
        "tcspc_toolkit.ml.evaluation",
        "tcspc_toolkit.ml.models",
        "tcspc_toolkit.uncertainty.evaluation",
        "tcspc_toolkit.benchmarks.generalization_protocol",
        "tcspc_toolkit.benchmarks.generalization",
        "tcspc_toolkit.benchmarks.generalization_datasets",
    },
    UNCERTAINTY_CLASSICAL: {
        "__future__",
        "dataclasses",
        "numpy",
        "pandas",
        "numpy.typing",
        "statistics",
        "tcspc_toolkit.estimation.classical",
        "tcspc_toolkit.irf.profiles",
        "tcspc_toolkit.simulation.histograms",
        "tcspc_toolkit.estimation.fitting",
        "tcspc_toolkit.evaluation.results",
        "tcspc_toolkit.evaluation.attachments",
        "tcspc_toolkit.physics.forward_model",
        "tcspc_toolkit.uncertainty.evaluation",
    },
}


def test_uncertainty_method_dependencies_and_consumers_preserve_edge_kinds(
    package_imports,
):
    records, _ = package_imports
    assert records[UNCERTAINTY] == []
    for module, allowed in UNCERTAINTY_METHOD_IMPORTS.items():
        assert {base for _, base, _ in records[module]} == allowed
        assert not any(kind == "deferred" for kind, _, _ in records[module])
        assert all(
            kind == "eager"
            for kind, base, _ in records[module]
            if not base.startswith(BENCHMARKS + ".")
        )
    assert [
        (kind, base, names)
        for kind, base, names in records[UNCERTAINTY_ML]
        if kind == "typing"
    ] == [
        ("typing", GENERALIZATION_PROTOCOL, ("GeneralizationSuiteDefinition",)),
        ("typing", GENERALIZATION, ("GeneralizationPreparedData",)),
        ("typing", GENERALIZATION_DATASETS, ("GeneralizationTestMeasurements",)),
    ]
    assert all(kind == "eager" for kind, _, _ in records[UNCERTAINTY_CLASSICAL])
    expected = {
        UNCERTAINTY_ML: {
            (IRF_STUDY, "eager"),
            (UNCERTAINTY_BENCHMARK, "eager"),
            (PERSISTENCE_SQLITE, "deferred"),
        },
        UNCERTAINTY_CLASSICAL: {
            (BAYESIAN_CALIBRATION, "eager"),
            (BAYESIAN_MISMATCH, "eager"),
            (IRF_STUDY, "eager"),
            (UNCERTAINTY_BENCHMARK, "eager"),
            (PERSISTENCE_SQLITE, "deferred"),
        },
    }
    for target, expected_edges in expected.items():
        consumers = [
            (module, kind)
            for module, imports in records.items()
            for kind, base, _ in imports
            if base == target
        ]
        assert set(consumers) == expected_edges
        assert (
            len(consumers) == len(expected_edges) + 3
        )  # Four persistence-local imports.
        assert consumers.count((PERSISTENCE_SQLITE, "deferred")) == 4


@pytest.mark.parametrize(
    "module, canonical",
    [(UNCERTAINTY_ML, UNCERTAINTY_ML), (UNCERTAINTY_CLASSICAL, UNCERTAINTY_CLASSICAL)],
)
def test_fresh_uncertainty_methods_do_not_load_studies_or_sibling(
    tmp_path, module, canonical
):
    sibling = UNCERTAINTY_CLASSICAL if canonical == UNCERTAINTY_ML else UNCERTAINTY_ML
    blocked = sorted(MIGRATED_SHIMS - {module}) + [
        sibling,
        GENERALIZATION,
        GENERALIZATION_DATASETS,
        PERSISTENCE,
        f"{PACKAGE}.uncertainty_robustness",
        UNCERTAINTY_BENCHMARK,
        f"{PACKAGE}.irf_evaluation",
        f"{BENCHMARKS}.irf",
        f"{PACKAGE}.bayesian_evaluation",
        f"{BENCHMARKS}.bayesian_calibration",
        f"{PACKAGE}.bayesian_mismatch_evaluation",
        f"{BENCHMARKS}.bayesian_mismatch",
        f"{PACKAGE}.mismatch_evaluation",
        MISMATCH,
        f"{PACKAGE}.timing_evaluation",
        TIMING,
        BAYESIAN,
        "emcee",
        "sqlite3",
    ]
    if canonical == UNCERTAINTY_CLASSICAL:
        blocked.append(ML)
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = {blocked!r}
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
root_exports = tuple(tcspc_toolkit.__all__)
assert len(root_exports) == 66
inherited = set(sys.modules)
namespace = importlib.import_module({UNCERTAINTY!r})
assert set(sys.modules) - inherited == {{{UNCERTAINTY!r}}}
importlib.import_module({module!r})
assert {{n for n in sys.modules if n.startswith({UNCERTAINTY + "."!r})}} == {{
    {UNCERTAINTY_EVALUATION!r}, {canonical!r},
}}
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == {{
    {GENERALIZATION_PROTOCOL!r},
}}
assert not hasattr(namespace, 'QuantileGradientBoostingIntervalEstimator')
assert not hasattr(namespace, 'PoissonLocalCovarianceResult')
assert tuple(tcspc_toolkit.__all__) == root_exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_week7_studies_remain_independent_of_other_benchmarks(package_imports):
    records, _ = package_imports
    assert records[BENCHMARKS] == []
    for canonical in (MISMATCH, TIMING):
        assert {base for _, base, _ in records[canonical]} == BENCHMARK_IMPORTS[
            canonical
        ]
        assert all(kind == "eager" for kind, _, _ in records[canonical])
        assert not any(
            base == BENCHMARKS or base.startswith(BENCHMARKS + ".")
            for _, base, _ in records[canonical]
        )
        consumers = [
            (module, kind)
            for module, imports in records.items()
            for kind, base, names in imports
            if canonical in _targets(base, names, set(records))
        ]
        assert consumers == []
    # No production/private or persistence-local consumer existed at relocation.
    assert not any(
        base in {MISMATCH, TIMING} for _, base, _ in records[PERSISTENCE_SQLITE]
    )


@pytest.mark.parametrize("module, canonical", [(MISMATCH, MISMATCH), (TIMING, TIMING)])
def test_fresh_week7_studies_load_only_the_explicit_benchmark_child(
    tmp_path, module, canonical
):
    sibling = TIMING if canonical == MISMATCH else MISMATCH
    blocked = sorted(MIGRATED_SHIMS - {module}) + [
        sibling,
        GENERALIZATION_PROTOCOL,
        GENERALIZATION_DATASETS,
        GENERALIZATION,
        UNCERTAINTY,
        f"{PACKAGE}.uncertainty_robustness",
        UNCERTAINTY_BENCHMARK,
        f"{PACKAGE}.irf_evaluation",
        f"{BENCHMARKS}.irf",
        f"{PACKAGE}.bayesian_evaluation",
        f"{BENCHMARKS}.bayesian_calibration",
        f"{PACKAGE}.bayesian_mismatch_evaluation",
        f"{BENCHMARKS}.bayesian_mismatch",
        BAYESIAN,
        PERSISTENCE,
        "emcee",
        "sqlite3",
    ]
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = {blocked!r}
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
root_exports = tuple(tcspc_toolkit.__all__)
assert len(root_exports) == 66
inherited = set(sys.modules)
namespace = importlib.import_module({BENCHMARKS!r})
assert set(sys.modules) - inherited == {{{BENCHMARKS!r}}}
importlib.import_module({module!r})
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == {{{canonical!r}}}
assert not hasattr(namespace, 'EstimatorMismatchResult')
assert not hasattr(namespace, 'InferenceTimingResult')
assert tuple(tcspc_toolkit.__all__) == root_exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_irf_bayesian_study_edges_and_private_helpers_remain_canonical(package_imports):
    records, _ = package_imports
    from collections import Counter

    expected_consumers = {
        IRF_STUDY: Counter(
            {
                (BAYESIAN_MISMATCH, "eager"): 1,
            }
        ),
        BAYESIAN_CALIBRATION: Counter(
            {
                (BAYESIAN_MISMATCH, "eager"): 1,
                (PERSISTENCE_SQLITE, "deferred"): 3,
            }
        ),
        BAYESIAN_MISMATCH: Counter(
            {
                (PERSISTENCE_SQLITE, "deferred"): 4,
            }
        ),
    }
    for canonical, expected in expected_consumers.items():
        assert {base for _, base, _ in records[canonical]} == BENCHMARK_IMPORTS[
            canonical
        ]
        assert all(kind == "eager" for kind, _, _ in records[canonical])
        consumers = Counter(
            (module, kind)
            for module, imports in records.items()
            for kind, base, names in imports
            if canonical in _targets(base, names, set(records))
        )
        assert consumers == expected
        benchmark_edges = {
            base
            for _, base, _ in records[canonical]
            if base.startswith(BENCHMARKS + ".")
        }
        assert benchmark_edges == (
            {IRF_STUDY, BAYESIAN_CALIBRATION}
            if canonical == BAYESIAN_MISMATCH
            else set()
        )
    assert ("eager", IRF_STUDY, ("_match_gaussian_to_emg_profile",)) in records[
        BAYESIAN_MISMATCH
    ]
    assert (
        "eager",
        BAYESIAN_CALIBRATION,
        (
            "Issue4Bayesian",
            "Issue4Bootstrap",
            "Issue4ClassicalFit",
            "Issue4Covariance",
            "_unavailable_bootstrap",
            "_unavailable_covariance",
        ),
    ) in records[BAYESIAN_MISMATCH]
    for filename, canonical in (
        ("run_issue4_bayesian_benchmarks.py", BAYESIAN_CALIBRATION),
        ("audit_issue4_stage5_classical_fits.py", BAYESIAN_CALIBRATION),
        ("run_issue4_bayesian_mismatch_benchmarks.py", BAYESIAN_MISMATCH),
        ("correct_issue4_mismatch_references.py", BAYESIAN_MISMATCH),
    ):
        source = (SOURCE_ROOT.parents[1] / "scripts" / filename).read_text(
            encoding="utf-8"
        )
        imports = _imports(source, f"scripts.{Path(filename).stem}")
        edges = [
            (kind, base)
            for kind, base, _ in imports
            if base in expected_consumers
            or base
            in {
                f"{PACKAGE}.irf_evaluation",
                f"{PACKAGE}.bayesian_evaluation",
                f"{PACKAGE}.bayesian_mismatch_evaluation",
            }
        ]
        assert edges == [("eager", canonical)]


@pytest.mark.parametrize(
    "module, canonical",
    [
        (IRF_STUDY, IRF_STUDY),
        (BAYESIAN_CALIBRATION, BAYESIAN_CALIBRATION),
        (BAYESIAN_MISMATCH, BAYESIAN_MISMATCH),
    ],
)
def test_fresh_irf_bayesian_studies_preserve_optional_sampler_and_isolation(
    tmp_path,
    module,
    canonical,
):
    expected_children = {
        IRF_STUDY: {IRF_STUDY, GENERALIZATION_PROTOCOL},
        BAYESIAN_CALIBRATION: {BAYESIAN_CALIBRATION, GENERALIZATION_PROTOCOL},
        BAYESIAN_MISMATCH: {
            BAYESIAN_MISMATCH,
            BAYESIAN_CALIBRATION,
            IRF_STUDY,
            GENERALIZATION_PROTOCOL,
        },
    }[canonical]
    blocked = sorted(MIGRATED_SHIMS - {module}) + [
        GENERALIZATION,
        GENERALIZATION_DATASETS,
        MISMATCH,
        TIMING,
        f"{PACKAGE}.uncertainty_robustness",
        UNCERTAINTY_BENCHMARK,
        PERSISTENCE,
        "emcee",
        "sqlite3",
    ]
    if canonical == IRF_STUDY:
        blocked += [BAYESIAN, BAYESIAN_CALIBRATION, BAYESIAN_MISMATCH]
    elif canonical == BAYESIAN_CALIBRATION:
        blocked += [IRF_STUDY, BAYESIAN_MISMATCH]
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = {blocked!r}
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
import tcspc_toolkit
root_exports = tuple(tcspc_toolkit.__all__)
assert len(root_exports) == 66
inherited = set(sys.modules)
namespace = importlib.import_module({BENCHMARKS!r})
assert set(sys.modules) - inherited == {{{BENCHMARKS!r}}}
importlib.import_module({module!r})
# Protocol configuration is an existing transitive uncertainty dependency,
# not a direct IRF-study dependency or an execution/data-generation edge.
assert {{n for n in sys.modules if n.startswith({BENCHMARKS + "."!r})}} == {expected_children!r}
assert not hasattr(namespace, 'IRFShapeMismatchConfig')
assert not hasattr(namespace, 'Issue4Bayesian')
assert not hasattr(namespace, 'MismatchReport')
assert tuple(tcspc_toolkit.__all__) == root_exports
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_fresh_foundations_do_not_load_benchmarks(tmp_path):
    # The real root already loads sklearn/fitting. None of these imports may
    # load even benchmark configuration, much less execution or reporting.
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
blocked = [{BENCHMARKS!r}, {UNCERTAINTY!r}, 'tcspc_toolkit.generalization',
           'tcspc_toolkit.generalization_datasets', 'tcspc_toolkit.generalization_evaluation',
           'emcee', 'sqlite3', 'tcspc_toolkit.persistence']
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
for module in ('tcspc_toolkit', 'tcspc_toolkit.estimation.api',
               'tcspc_toolkit.ml.models', 'tcspc_toolkit.evaluation.results'):
    importlib.import_module(module)
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_preprocessing_collision_has_one_package_implementation():
    assert not (SOURCE_ROOT / "preprocessing.py").exists()
    package = importlib.import_module(PREPROCESSING)
    assert (
        Path(package.__file__).resolve()
        == SOURCE_ROOT / "preprocessing" / "__init__.py"
    )
    assert hasattr(package, "__path__")


def test_irf_collision_preserves_only_the_historical_profile_surface():
    assert not (SOURCE_ROOT / "irf.py").exists()
    package = importlib.import_module(IRF)
    assert Path(package.__file__).resolve() == SOURCE_ROOT / "irf" / "__init__.py"
    assert hasattr(package, "__path__")
    # Importing canonical children must not add their APIs to the facade.
    importlib.import_module(f"{IRF}.preparation")
    importlib.import_module(f"{IRF}.estimation")
    importlib.import_module(f"{IRF}.selection")
    for name in (
        "PreparedIRF",
        "prepare_irf",
        "LeadingEdgeIRFResult",
        "estimate_irf_from_leading_edge",
        "_sampled_fwhm",
        "_resolve_reconvolution_irf",
    ):
        assert not hasattr(package, name), name


def test_simulation_collision_preserves_only_the_historical_histogram_surface():
    assert not (SOURCE_ROOT / "simulation.py").exists()
    package = importlib.import_module(SIMULATION)
    assert (
        Path(package.__file__).resolve() == SOURCE_ROOT / "simulation" / "__init__.py"
    )
    assert hasattr(package, "__path__")
    importlib.import_module(f"{SIMULATION}.datasets")
    for name in (
        "SyntheticDataset",
        "generate_monoexponential_dataset",
        "generate_grouped_monoexponential_dataset",
        "monoexponential_decay",
    ):
        assert not hasattr(package, name), name


def test_evaluation_collision_preserves_only_the_historical_diagnostics_surface():
    assert not (SOURCE_ROOT / "evaluation.py").exists()
    package = importlib.import_module(EVALUATION)
    canonical = importlib.import_module(DIAGNOSTICS)
    root = importlib.import_module(PACKAGE)
    assert (
        Path(package.__file__).resolve() == SOURCE_ROOT / "evaluation" / "__init__.py"
    )
    assert hasattr(package, "__path__")
    assert tuple(package.__all__) == DIAGNOSTIC_EXPORTS
    assert not set(DIAGNOSTIC_EXPORTS).intersection(root.__all__)
    for name in DIAGNOSTIC_EXPORTS:
        assert getattr(package, name) is getattr(canonical, name)
        assert getattr(canonical, name).__module__ == DIAGNOSTICS
    for name in (
        "LifetimeFitResult",
        "monoexponential_decay",
        "np",
        "NDArray",
        "PointEvaluationResult",
        "build_point_evaluation",
        "IntervalAttachment",
        "build_prediction_diagnostics",
        "plot_true_vs_predicted",
    ):
        assert not hasattr(package, name), name


@pytest.mark.parametrize("module", [EVALUATION, DIAGNOSTICS])
def test_fresh_evaluation_facade_loads_only_historical_diagnostics(tmp_path, module):
    # Keep real root initialization, which already imports fitting and diagnostics.
    # Attribute import edges to their owner, not everything present in sys.modules.
    script = f"""
import builtins
import importlib
import importlib.abc
import sys
attempts = []
blocked = [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence',
    'tcspc_toolkit.evaluation_results', 'tcspc_toolkit.evaluation_core',
    'tcspc_toolkit.evaluation_uncertainty', 'tcspc_toolkit.conditional_evaluation',
    'tcspc_toolkit.evaluation.results', 'tcspc_toolkit.evaluation.core',
    'tcspc_toolkit.evaluation.attachments',
    'tcspc_toolkit.evaluation.conditional', 'tcspc_toolkit.evaluation.plots',
    'tcspc_toolkit.benchmark_plots', 'matplotlib', 'tcspc_toolkit.ml',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.benchmarks', 'tcspc_toolkit.generalization',
    'tcspc_toolkit.generalization_datasets', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.uncertainty_evaluation', 'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml',
    'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty.classical', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller in ({EVALUATION!r}, {DIAGNOSTICS!r}):
        base = importlib.util.resolve_name('.' * level + name, globals['__package__']) if level else name
        seen.setdefault(caller, set()).add(base)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert seen[{EVALUATION!r}] == {{{DIAGNOSTICS!r}}}
assert seen[{DIAGNOSTICS!r}] == {DIAGNOSTIC_IMPORTS!r}
facade = importlib.import_module({EVALUATION!r})
canonical = importlib.import_module({DIAGNOSTICS!r})
assert tuple(facade.__all__) == {DIAGNOSTIC_EXPORTS!r}
for name in {DIAGNOSTIC_EXPORTS!r}:
    assert getattr(facade, name) is getattr(canonical, name)
assert 'tcspc_toolkit.estimation.fitting' in sys.modules
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(FACTUAL_IMPORTS))
def test_fresh_factual_imports_separate_source_dependencies_from_parent_cost(
    tmp_path, module
):
    # Retain real root/parent initialization. Fitting belongs to historical
    # diagnostics, not the factual source modules whose imports we trace here.
    script = f"""
import builtins
import importlib
import importlib.abc
import sys
attempts = []
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence', 'tcspc_toolkit.ml',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.conditional_evaluation',
    'tcspc_toolkit.evaluation.conditional', 'tcspc_toolkit.evaluation.plots',
    'tcspc_toolkit.benchmark_plots', 'matplotlib', 'tcspc_toolkit.benchmarks', 'tcspc_toolkit.generalization',
    'tcspc_toolkit.generalization_datasets', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.benchmarks.mismatch', 'tcspc_toolkit.uncertainty_evaluation',
    'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml', 'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty.classical',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {EVALUATION!r} or caller.startswith({EVALUATION + "."!r}):
        base = importlib.util.resolve_name('.' * level + name, globals['__package__']) if level else name
        seen.setdefault(caller, set()).add(base)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert seen[{EVALUATION!r}] == {{{DIAGNOSTICS!r}}}
assert seen[{DIAGNOSTICS!r}] == {DIAGNOSTIC_IMPORTS!r}
assert 'tcspc_toolkit.estimation.fitting' in sys.modules
expected = {{{module!r}, {EVALUATION + ".results"!r}}}
comparison = {EVALUATION + ".reconvolution"!r}
# The new measurement comparison is imported by the historical root adapter.
# Its statistics-only dependencies must not leak into the factual layer itself.
assert set(seen) == expected | {{{EVALUATION!r}, {DIAGNOSTICS!r}, comparison}}
assert seen[comparison] == {{'dataclasses', 'numpy', 'numpy.typing', {DIAGNOSTICS!r}}}
for caller in expected:
    assert seen[caller] == {FACTUAL_IMPORTS!r}[caller], (caller, seen[caller])
assert tuple(sys.modules[{EVALUATION!r}].__all__) == {DIAGNOSTIC_EXPORTS!r}
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module, canonical",
    [
        (f"{EVALUATION}.conditional", f"{EVALUATION}.conditional"),
        (f"{EVALUATION}.plots", f"{EVALUATION}.plots"),
    ],
)
def test_fresh_analysis_imports_require_explicit_submodules(
    tmp_path, module, canonical
):
    script = f"""
import importlib
import importlib.abc
import os
import sys
os.environ['MPLBACKEND'] = 'Agg'
attempts = []
blocked = [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence',
    'tcspc_toolkit.estimation.bayesian', 'tcspc_toolkit.benchmarks', 'tcspc_toolkit.generalization',
    'tcspc_toolkit.generalization_datasets', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.mismatch_evaluation', 'tcspc_toolkit.benchmarks.mismatch', 'tcspc_toolkit.uncertainty_evaluation',
    'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml', 'tcspc_toolkit.classical_uncertainty', 'tcspc_toolkit.uncertainty.classical',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BoundaryGuard())
facade = importlib.import_module({EVALUATION!r})
assert tuple(facade.__all__) == {DIAGNOSTIC_EXPORTS!r}
assert not any(n in sys.modules for n in {sorted(FACTUAL | set(ANALYSIS_IMPORTS))!r})
assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules)
loaded = importlib.import_module({module!r})
implementation = importlib.import_module({canonical!r})
if {canonical!r} == {EVALUATION + ".conditional"!r}:
    assert {EVALUATION + ".plots"!r} not in sys.modules
    assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules)
else:
    assert 'matplotlib.pyplot' in sys.modules
    assert {EVALUATION + ".conditional"!r} not in sys.modules
    assert not any(n in sys.modules for n in {sorted(FACTUAL)!r})
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
assert loaded is implementation
public_names = (
    ('build_prediction_diagnostics', 'build_ml_prediction_diagnostics',
     'build_classical_prediction_diagnostics', 'summarize_conditional_performance',
     'assign_numeric_regimes', 'add_benchmark_regimes', 'summarize_standard_regimes')
    if {canonical!r} == {EVALUATION + ".conditional"!r} else
    ('plot_true_vs_predicted', 'plot_error_histogram', 'plot_error_vs_condition',
     'plot_paired_absolute_error_comparison')
)
for name in public_names:
    assert callable(getattr(implementation, name))
    assert not hasattr(facade, name)
assert tuple(facade.__all__) == {DIAGNOSTIC_EXPORTS!r}
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_bayesian_and_experimental_consumers_share_the_canonical_irf_resolver():
    selection = importlib.import_module(f"{IRF}.selection")
    resolver = selection._resolve_reconvolution_irf
    assert resolver.__module__ == f"{IRF}.selection"
    for module in (f"{BAYESIAN}.posterior", f"{MEASUREMENT}.adapters"):
        assert importlib.import_module(module)._resolve_reconvolution_irf is resolver


def test_bayesian_sampling_edges_remain_deferred(package_imports):
    records, _ = package_imports
    internal = {
        module: {
            (kind, base)
            for kind, base, _ in records[module]
            if base.startswith(BAYESIAN + ".")
        }
        for module in BAYESIAN_IMPORTS
    }
    assert internal == {
        BAYESIAN: set(),
        f"{BAYESIAN}.posterior": {("deferred", f"{BAYESIAN}.sampling")},
        f"{BAYESIAN}.sampling": {("eager", f"{BAYESIAN}.posterior")},
        f"{BAYESIAN}.predictive": {
            ("eager", f"{BAYESIAN}.posterior"),
            ("eager", f"{BAYESIAN}.sampling"),
        },
    }
    sampler_edges = [
        (module, kind, base)
        for module in BAYESIAN_IMPORTS
        for kind, base, _ in records[module]
        if base == "emcee" or base.startswith("emcee.")
    ]
    assert sampler_edges
    assert all(
        module == f"{BAYESIAN}.sampling" and kind == "deferred"
        for module, kind, _ in sampler_edges
    )


@pytest.mark.parametrize("module", sorted(BAYESIAN_IMPORTS))
def test_fresh_bayesian_import_preserves_optional_sampler_and_minimal_initializers(
    tmp_path, module
):
    script = f"""
import importlib
import importlib.abc
import sys
attempts = []
class NoSampler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'emcee' or fullname.startswith('emcee.'):
            attempts.append(fullname)
            raise ModuleNotFoundError('optional sampler unavailable', name=fullname)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('Bayesian import opened a database')
sys.meta_path.insert(0, NoSampler())
sys.addaudithook(no_database)
import tcspc_toolkit
assert not any(n == {BAYESIAN!r} or n.startswith({BAYESIAN + "."!r}) for n in sys.modules)
importlib.import_module({module!r})
expected = {{
    {BAYESIAN!r}: set(),
    {BAYESIAN + ".posterior"!r}: {{'posterior'}},
    {BAYESIAN + ".sampling"!r}: {{'posterior', 'sampling'}},
    {BAYESIAN + ".predictive"!r}: {{'posterior', 'sampling', 'predictive'}},
}}[{module!r}]
loaded = {{n.rsplit('.', 1)[-1] for n in sys.modules if n.startswith({BAYESIAN + "."!r})}}
assert loaded == expected, loaded
assert not attempts, attempts
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'emcee', 'sqlite3', 'tcspc_toolkit.persistence', 'tcspc_toolkit.bayesian_evaluation', 'tcspc_toolkit.benchmarks.bayesian_calibration',
    'tcspc_toolkit.bayesian_mismatch_evaluation', 'tcspc_toolkit.benchmarks.bayesian_mismatch', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(ML_IMPORTS))
def test_fresh_ml_import_preserves_minimal_initializer_and_dependency_direction(
    tmp_path, module
):
    # Keep real root initialization (which already imports sklearn/PCA) and
    # attribute dependencies to their actual caller, not to the ML initializer.
    script = f"""
import builtins
import importlib
import importlib.abc
import sys
attempts = []
class NoSampler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'emcee' or fullname.startswith('emcee.'):
            attempts.append(fullname)
            raise ModuleNotFoundError('optional sampler unavailable', name=fullname)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('ML import opened a database')
sys.meta_path.insert(0, NoSampler())
sys.addaudithook(no_database)
import tcspc_toolkit
import tcspc_toolkit.estimation.api
assert not any(n == {ML!r} or n.startswith({ML + "."!r}) for n in sys.modules)
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {ML!r} or caller.startswith({ML + "."!r}):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert not seen.get({ML!r})
for caller, imports in seen.items():
    assert imports <= {ML_IMPORTS!r}[caller], (caller, imports)
expected = {{
    {ML!r}: set(),
    {ML + ".models"!r}: {{'models'}},
    {ML + ".evaluation"!r}: {{'models', 'evaluation'}},
    {ML + ".cross_validation"!r}: {{'models', 'evaluation', 'cross_validation'}},
}}[{module!r}]
loaded = {{n.rsplit('.', 1)[-1] for n in sys.modules if n.startswith({ML + "."!r})}}
assert loaded == expected, loaded
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    {BAYESIAN!r}, 'emcee', 'sqlite3', 'tcspc_toolkit.persistence',
    'tcspc_toolkit.generalization_evaluation', 'tcspc_toolkit.ml_uncertainty', 'tcspc_toolkit.uncertainty.ml',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_no_eager_internal_cycles(package_imports):
    records, packages = package_imports
    # The posterior -> sampling import is deliberately deferred; the reverse
    # dependency is eager. Do not turn the combined graph into an eager graph.
    tuple(TopologicalSorter(_eager_graph(records, packages)).static_order())


def test_import_graph_distinguishes_typing_local_and_initializer_edges():
    sources = {
        "demo": ("from . import leaf", True),
        "demo.leaf": (
            """
import typing as t
from typing import TYPE_CHECKING as TC
if TC:
    from demo import typed
if t.TYPE_CHECKING:
    import demo.typed
else:
    from . import eager
class Container:
    from . import eager
    def method(self):
        from . import local
async def later():
    from . import local
""",
            False,
        ),
        "demo.eager": ("", False),
        "demo.local": ("from . import leaf", False),
        "demo.typed": ("from . import leaf", False),
        "outside": ("import demo.leaf", False),
    }
    records = {
        name: _imports(source, name, is_package=package)
        for name, (source, package) in sources.items()
    }
    assert set(records["demo.leaf"]) >= {
        ("typing", "demo", ("typed",)),
        ("typing", "demo.typed", ()),
        ("eager", "demo", ("eager",)),
        ("deferred", "demo", ("local",)),
    }
    graph = _eager_graph(records, {"demo"})
    # Imports from a child do not re-execute its own parent initializer.
    assert graph["demo.leaf"] == {"demo.eager"}
    assert graph["outside"] == {"demo", "demo.leaf"}
    tuple(TopologicalSorter(graph).static_order())
    # An eager return edge really does form a cycle; no modules are created.
    graph["demo.eager"].add("demo.leaf")
    with pytest.raises(CycleError):
        tuple(TopologicalSorter(graph).static_order())


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
@pytest.mark.parametrize("generic_only", [True, False], ids=["generic", "supported"])
def test_fresh_import_order_without_sampler_or_database(
    tmp_path, reverse, generic_only
):
    names = (
        [
            "estimation.api",
            "evaluation.results",
            "evaluation.core",
            "evaluation.attachments",
        ]
        if generic_only
        else [
            *SUPPORTED_MODULE_PATHS,
            "persistence.sqlite",
            "estimation.api",
            "estimation.baselines",
            "estimation.fitting",
            "estimation.classical",
            "estimation.bayesian.posterior",
            "estimation.bayesian.sampling",
            "estimation.bayesian.predictive",
            "ml.models",
            "ml.evaluation",
            "ml.cross_validation",
            "physics.models",
            "physics.convolution",
            "physics.forward_model",
            "preprocessing.histograms",
            "preprocessing.features",
            "preprocessing.representations",
            "irf.profiles",
            "irf.preparation",
            "irf.estimation",
            "irf.selection",
            "measurement.records",
            "measurement.io",
            "measurement.adapters",
            "simulation.histograms",
            "simulation.datasets",
            "evaluation.diagnostics",
            "evaluation.results",
            "evaluation.core",
            "evaluation.attachments",
            "evaluation.conditional",
            "evaluation.plots",
            "benchmarks.generalization_protocol",
            "benchmarks.generalization_datasets",
            "benchmarks.generalization",
            "benchmarks.mismatch",
            "benchmarks.timing",
            "benchmarks.irf",
            "benchmarks.bayesian_calibration",
            "benchmarks.bayesian_mismatch",
            "benchmarks.uncertainty",
            "uncertainty",
            "uncertainty.evaluation",
            "uncertainty.ml",
            "uncertainty.classical",
        ]
    )
    if reverse:
        names.reverse()
    blocked = ["emcee"]
    if generic_only:
        blocked += [
            PERSISTENCE,
            f"{PACKAGE}.bayesian_sampling",
            BAYESIAN,
            ML,
            BENCHMARKS,
            f"{PACKAGE}.generalization_evaluation",
            f"{PACKAGE}.ml_models",
        ]
    script = f"""
import importlib
import importlib.abc
import sys
blocked = {blocked!r}
attempts = []
class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError(fullname, name=fullname)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('import opened a database')
sys.meta_path.insert(0, BoundaryGuard())
sys.addaudithook(no_database)
for name in {names!r}:
    importlib.import_module('tcspc_toolkit.' + name)
assert not attempts, attempts
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(PREPROCESSING_IMPORTS))
def test_fresh_preprocessing_facade_loads_only_histograms(tmp_path, module):
    # Keep real root initialization. The root already exposes features/PCA;
    # distinguish its imports from those introduced by the package facade.
    script = f"""
import builtins
import importlib
import sys
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {PREPROCESSING!r} or caller.startswith({PREPROCESSING + "."!r}):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('preprocessing import opened a database')
sys.addaudithook(no_database)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert seen[{PREPROCESSING!r}] == {{{PREPROCESSING + ".histograms"!r}}}
allowed = {PREPROCESSING_IMPORTS!r}
for caller, imports in seen.items():
    assert imports <= allowed[caller], (caller, imports)
from tcspc_toolkit.preprocessing import validate_histogram, normalize_counts, CountNormalization
from tcspc_toolkit.preprocessing.histograms import validate_histogram as canonical_validate
from tcspc_toolkit.preprocessing.histograms import normalize_counts as canonical_normalize
from tcspc_toolkit.config import CountNormalization as canonical_mode
assert validate_histogram is canonical_validate
assert normalize_counts is canonical_normalize
assert CountNormalization is canonical_mode
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'tcspc_toolkit.persistence', 'emcee', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.ml_models', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(IRF_IMPORTS))
def test_fresh_irf_facade_loads_only_profiles(tmp_path, module):
    # Root initialization already exposes preparation/estimation. Trace the
    # actual caller rather than attributing that inherited cost to the facade.
    script = f"""
import builtins
import importlib
import sys
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {IRF!r} or caller.startswith({IRF + "."!r}):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('IRF import opened a database')
sys.addaudithook(no_database)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert seen[{IRF!r}] == {{{IRF + ".profiles"!r}}}
allowed = {IRF_IMPORTS!r}
for caller, imports in seen.items():
    assert imports <= allowed[caller], (caller, imports)
from tcspc_toolkit.irf import IRFProfile, normalize_irf
from tcspc_toolkit.irf.profiles import IRFProfile as canonical_profile
from tcspc_toolkit.irf.profiles import normalize_irf as canonical_normalize
assert IRFProfile is canonical_profile
assert normalize_irf is canonical_normalize
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'tcspc_toolkit.persistence', 'emcee', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.ml_models', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(MEASUREMENT_IMPORTS))
def test_fresh_measurement_initializer_does_not_aggregate_its_subtree(tmp_path, module):
    # Root imports the public adapters already. Trace the importing module so
    # those inherited dependencies are not attributed to the empty initializer.
    script = f"""
import builtins
import importlib
import sys
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {MEASUREMENT!r} or caller.startswith({MEASUREMENT + "."!r}):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('measurement import opened a database')
sys.addaudithook(no_database)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert not seen.get({MEASUREMENT!r})
allowed = {MEASUREMENT_IMPORTS!r}
for caller, imports in seen.items():
    assert imports <= allowed[caller], (caller, imports)
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'tcspc_toolkit.persistence', 'emcee', 'tcspc_toolkit.bayesian',
    'tcspc_toolkit.generalization_evaluation', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(SIMULATION_IMPORTS))
def test_fresh_simulation_facade_loads_only_histograms(tmp_path, module):
    # Root already exports SyntheticDataset. Trace the actual importing module
    # rather than treating that inherited cost as a facade-to-datasets edge.
    script = f"""
import builtins
import importlib
import sys
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == {SIMULATION!r} or caller.startswith({SIMULATION + "."!r}):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('simulation import opened a database')
sys.addaudithook(no_database)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert seen[{SIMULATION!r}] == {{{SIMULATION + ".histograms"!r}}}
allowed = {SIMULATION_IMPORTS!r}
for caller, imports in seen.items():
    assert imports <= allowed[caller], (caller, imports)
from tcspc_toolkit.simulation import sample_photon_counts
from tcspc_toolkit.simulation.histograms import sample_photon_counts as canonical_sample
assert sample_photon_counts is canonical_sample
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'tcspc_toolkit.persistence', 'emcee', 'tcspc_toolkit.bayesian',
    'tcspc_toolkit.generalization_evaluation', 'tcspc_toolkit.ml_models',
    'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", sorted(PHYSICS_IMPORTS))
def test_fresh_physics_import_adds_only_low_level_dependencies(tmp_path, module):
    # Trace real imports, retaining normal root initialization. The root's
    # existing fitting/sklearn imports are not dependencies of physics itself.
    script = f"""
import builtins
import importlib
import sys
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == 'tcspc_toolkit.physics' or caller.startswith('tcspc_toolkit.physics.'):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('physical import opened a database')
sys.addaudithook(no_database)
builtins.__import__ = track
try:
    importlib.import_module({module!r})
finally:
    builtins.__import__ = original_import
assert seen[{module!r}]
assert not seen.get('tcspc_toolkit.physics')
allowed = {PHYSICS_IMPORTS!r}
for caller, imports in seen.items():
    assert imports <= allowed[caller], (caller, imports)
blocked = {sorted(MIGRATED_SHIMS)!r} + [
    'tcspc_toolkit.persistence', 'emcee', 'tcspc_toolkit.generalization_evaluation',
    'tcspc_toolkit.ml_models', 'tcspc_toolkit.uncertainty_robustness', 'tcspc_toolkit.benchmarks.uncertainty',
]
assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in blocked)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "module", ["api", "baselines", "fitting", "classical", "multiexponential"]
)
def test_fresh_canonical_estimation_import_adds_only_its_own_dependencies(
    tmp_path, module
):
    # A normal dotted import necessarily initializes tcspc_toolkit first. That
    # existing root imports fitting/sklearn. Trace actual import callers to
    # distinguish those inherited costs from dependencies of the new package;
    # do not substitute a fake root or change root initialization for this move.
    script = f"""
import builtins
import importlib
import sys
original_import = builtins.__import__
seen = {{}}
def track(name, globals=None, locals=None, fromlist=(), level=0):
    caller = (globals or {{}}).get('__name__', '')
    if caller == 'tcspc_toolkit.estimation' or caller.startswith('tcspc_toolkit.estimation.'):
        seen.setdefault(caller, set()).add(name)
    return original_import(name, globals, locals, fromlist, level)
def no_database(event, args):
    if event == 'sqlite3.connect':
        raise AssertionError('estimation import opened a database')
sys.addaudithook(no_database)
builtins.__import__ = track
try:
    importlib.import_module('tcspc_toolkit.estimation.{module}')
finally:
    builtins.__import__ = original_import
assert not seen.get('tcspc_toolkit.estimation')
assert seen['tcspc_toolkit.estimation.api'] <= {{
    '__future__', 'collections.abc', 'dataclasses', 'typing', 'numpy', 'numpy.typing',
}}
if {module!r} == 'baselines':
    assert seen['tcspc_toolkit.estimation.baselines'] == {{
        '__future__', 'numpy', 'numpy.typing', 'sklearn.dummy',
    }}
else:
    assert 'tcspc_toolkit.estimation.baselines' not in sys.modules
for caller, allowed in {CLASSICAL_IMPORTS!r}.items():
    assert seen.get(caller, set()) <= allowed, (caller, seen.get(caller))
assert 'tcspc_toolkit.estimator_api' not in sys.modules
assert 'tcspc_toolkit.baselines' not in sys.modules
assert 'tcspc_toolkit.fitting' not in sys.modules
assert 'tcspc_toolkit.classical_evaluation' not in sys.modules
assert 'tcspc_toolkit.persistence' not in sys.modules
assert 'emcee' not in sys.modules
assert 'tcspc_toolkit.bayesian_sampling' not in sys.modules
assert not any(n == {BAYESIAN!r} or n.startswith({BAYESIAN + "."!r}) for n in sys.modules)
assert 'tcspc_toolkit.generalization_evaluation' not in sys.modules
assert not any(n == {ML!r} or n.startswith({ML + "."!r}) for n in sys.modules)
assert 'tcspc_toolkit.uncertainty_robustness' not in sys.modules
assert 'tcspc_toolkit.benchmarks.uncertainty' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []
